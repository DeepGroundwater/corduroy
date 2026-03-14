use ndarray::Array2;

#[derive(Debug, thiserror::Error)]
pub enum RegridError {
    #[error("grid must have at least 2 points, got {0}")]
    GridTooSmall(usize),
    #[error("grid coordinates must be strictly monotonic")]
    NonMonotonic,
    #[error("latitude {0} out of range [-90, 90]")]
    InvalidLatitude(f64),
    #[error(
        "data shape ({data_rows}, {data_cols}) doesn't match source grid ({grid_rows}, {grid_cols})"
    )]
    ShapeMismatch {
        data_rows: usize,
        data_cols: usize,
        grid_rows: usize,
        grid_cols: usize,
    },
}

// ---------------------------------------------------------------------------
// Internal weight representation
// ---------------------------------------------------------------------------

/// One row of sparse 1D remapping weights: (source_index, weight) pairs.
#[derive(Debug, Clone)]
struct WeightRow {
    entries: Vec<(usize, f64)>,
}

/// 1D conservative remapping weights in CSR-like layout.
/// rows[i] gives the weights mapping source cells → target cell i.
#[derive(Debug, Clone)]
struct Weights1D {
    rows: Vec<WeightRow>,
}

// ---------------------------------------------------------------------------
// Utility functions
// ---------------------------------------------------------------------------

/// Infer cell edges from cell centers using midpoint interpolation.
///
/// For n centers, produces n+1 edges. Boundary edges extend half a step
/// beyond the first/last center.
pub fn centers_to_edges(centers: &[f64]) -> Vec<f64> {
    let n = centers.len();
    assert!(n >= 2);
    let mut edges = Vec::with_capacity(n + 1);
    // first boundary: half step before first center
    edges.push(centers[0] - (centers[1] - centers[0]) / 2.0);
    // interior edges: midpoints
    for i in 0..n - 1 {
        edges.push((centers[i] + centers[i + 1]) / 2.0);
    }
    // last boundary: half step after last center
    edges.push(centers[n - 1] + (centers[n - 1] - centers[n - 2]) / 2.0);
    edges
}

fn validate_monotonic(values: &[f64]) -> Result<(), RegridError> {
    if values.len() < 2 {
        return Ok(());
    }
    let ascending = values.windows(2).all(|w| w[0] < w[1]);
    let descending = values.windows(2).all(|w| w[0] > w[1]);
    if !ascending && !descending {
        return Err(RegridError::NonMonotonic);
    }
    Ok(())
}

/// Compute 1D overlap weights between source and target cells.
///
/// `src_edges` and `tgt_edges` are in the same coordinate space
/// (sin(lat) for latitude, degrees for longitude). Handles both
/// ascending and descending edge orderings.
fn compute_1d_weights(src_edges: &[f64], tgt_edges: &[f64]) -> Weights1D {
    let n_src = src_edges.len() - 1;
    let n_tgt = tgt_edges.len() - 1;

    let mut rows = Vec::with_capacity(n_tgt);

    for i in 0..n_tgt {
        let tgt_lo = tgt_edges[i].min(tgt_edges[i + 1]);
        let tgt_hi = tgt_edges[i].max(tgt_edges[i + 1]);
        let tgt_span = tgt_hi - tgt_lo;

        let mut entries = Vec::new();

        if tgt_span > 1e-15 {
            for j in 0..n_src {
                let src_lo = src_edges[j].min(src_edges[j + 1]);
                let src_hi = src_edges[j].max(src_edges[j + 1]);

                let overlap = (tgt_hi.min(src_hi) - tgt_lo.max(src_lo)).max(0.0);
                if overlap > 1e-15 {
                    entries.push((j, overlap / tgt_span));
                }
            }
        }

        rows.push(WeightRow { entries });
    }

    Weights1D { rows }
}

// ---------------------------------------------------------------------------
// Separable 1D weight application
// ---------------------------------------------------------------------------

/// Apply 1D weights along the latitude (row) axis.
/// Input: (n_src_lat, n_cols) → Output: (n_tgt_lat, n_cols)
fn apply_lat(weights: &Weights1D, data: &Array2<f64>) -> Array2<f64> {
    let (_, n_cols) = data.dim();
    let n_tgt = weights.rows.len();
    let mut out = Array2::zeros((n_tgt, n_cols));

    for (i, row) in weights.rows.iter().enumerate() {
        for &(j, w) in &row.entries {
            let src_row = data.row(j);
            let mut tgt_row = out.row_mut(i);
            tgt_row.scaled_add(w, &src_row);
        }
    }
    out
}

/// Apply 1D weights along the longitude (column) axis.
/// Input: (n_rows, n_src_lon) → Output: (n_rows, n_tgt_lon)
fn apply_lon(weights: &Weights1D, data: &Array2<f64>) -> Array2<f64> {
    let (n_rows, _) = data.dim();
    let n_tgt = weights.rows.len();
    let mut out = Array2::zeros((n_rows, n_tgt));

    for r in 0..n_rows {
        for (j, row) in weights.rows.iter().enumerate() {
            let mut sum = 0.0;
            for &(k, w) in &row.entries {
                sum += w * data[[r, k]];
            }
            out[[r, j]] = sum;
        }
    }
    out
}

// ---------------------------------------------------------------------------
// ConservativeRegridder
// ---------------------------------------------------------------------------

/// Conservative regridder for rectilinear grids on a sphere.
///
/// Preserves the area-weighted integral of the field during remapping.
/// Latitude uses sin(lat) transformation for correct spherical area
/// weighting; longitude uses linear overlap.
///
/// Weights are precomputed at construction time and reused for each
/// call to `regrid()`. The remapping is separable (lat then lon),
/// with f64 accumulation to avoid precision loss.
///
/// NaN handling: source NaN cells are excluded and the remaining
/// weights are renormalized (mask-based approach). If all contributing
/// source cells are NaN, the output cell is NaN.
#[derive(Debug, Clone)]
pub struct ConservativeRegridder {
    lat_weights: Weights1D,
    lon_weights: Weights1D,
    n_src_lat: usize,
    n_src_lon: usize,
    n_tgt_lat: usize,
    n_tgt_lon: usize,
}

impl ConservativeRegridder {
    /// Build a conservative regridder from source and target grid centers.
    ///
    /// - `src_lats`, `tgt_lats`: latitude centers in degrees, must be strictly
    ///   monotonic (ascending or descending) and within [-90, 90].
    /// - `src_lons`, `tgt_lons`: longitude centers in degrees, must be strictly
    ///   monotonic.
    pub fn new(
        src_lats: &[f64],
        src_lons: &[f64],
        tgt_lats: &[f64],
        tgt_lons: &[f64],
    ) -> Result<Self, RegridError> {
        // Validate sizes
        for (grid, name) in [
            (src_lats, "src_lats"),
            (src_lons, "src_lons"),
            (tgt_lats, "tgt_lats"),
            (tgt_lons, "tgt_lons"),
        ] {
            if grid.len() < 2 {
                let _ = name; // suppress unused warning
                return Err(RegridError::GridTooSmall(grid.len()));
            }
        }

        // Validate latitudes in [-90, 90]
        for &lat in src_lats.iter().chain(tgt_lats.iter()) {
            if !(-90.0..=90.0).contains(&lat) {
                return Err(RegridError::InvalidLatitude(lat));
            }
        }

        // Validate monotonicity
        validate_monotonic(src_lats)?;
        validate_monotonic(src_lons)?;
        validate_monotonic(tgt_lats)?;
        validate_monotonic(tgt_lons)?;

        // Compute cell edges
        let src_lat_edges = centers_to_edges(src_lats);
        let src_lon_edges = centers_to_edges(src_lons);
        let tgt_lat_edges = centers_to_edges(tgt_lats);
        let tgt_lon_edges = centers_to_edges(tgt_lons);

        // Clamp latitude edges to [-90, 90] before sin transform
        let src_lat_edges: Vec<f64> =
            src_lat_edges.iter().map(|&e| e.clamp(-90.0, 90.0)).collect();
        let tgt_lat_edges: Vec<f64> =
            tgt_lat_edges.iter().map(|&e| e.clamp(-90.0, 90.0)).collect();

        // Transform latitude edges to sin-space for spherical area weighting
        let src_sin: Vec<f64> = src_lat_edges.iter().map(|&e| e.to_radians().sin()).collect();
        let tgt_sin: Vec<f64> = tgt_lat_edges.iter().map(|&e| e.to_radians().sin()).collect();

        let lat_weights = compute_1d_weights(&src_sin, &tgt_sin);
        let lon_weights = compute_1d_weights(&src_lon_edges, &tgt_lon_edges);

        Ok(Self {
            lat_weights,
            lon_weights,
            n_src_lat: src_lats.len(),
            n_src_lon: src_lons.len(),
            n_tgt_lat: tgt_lats.len(),
            n_tgt_lon: tgt_lons.len(),
        })
    }

    /// Target grid dimensions (n_lat, n_lon).
    pub fn target_shape(&self) -> (usize, usize) {
        (self.n_tgt_lat, self.n_tgt_lon)
    }

    /// Regrid a 2D field (lat × lon) using precomputed weights.
    ///
    /// NaN values in the source are handled by mask-based renormalization:
    /// a validity mask is regridded alongside the data, then used to
    /// normalize the output. Fully-NaN target cells remain NaN.
    pub fn regrid(&self, data: &Array2<f32>) -> Result<Array2<f32>, RegridError> {
        let (n_lat, n_lon) = data.dim();
        if n_lat != self.n_src_lat || n_lon != self.n_src_lon {
            return Err(RegridError::ShapeMismatch {
                data_rows: n_lat,
                data_cols: n_lon,
                grid_rows: self.n_src_lat,
                grid_cols: self.n_src_lon,
            });
        }

        // Build f64 clean data (NaN→0) and validity mask
        let mut clean = Array2::<f64>::zeros((n_lat, n_lon));
        let mut mask = Array2::<f64>::zeros((n_lat, n_lon));
        for ((r, c), &v) in data.indexed_iter() {
            if !v.is_nan() {
                clean[[r, c]] = v as f64;
                mask[[r, c]] = 1.0;
            }
        }

        // Separable regrid: lat then lon
        let clean_lat = apply_lat(&self.lat_weights, &clean);
        let mask_lat = apply_lat(&self.lat_weights, &mask);

        let clean_out = apply_lon(&self.lon_weights, &clean_lat);
        let mask_out = apply_lon(&self.lon_weights, &mask_lat);

        // Normalize by mask and convert to f32
        let mut result = Array2::<f32>::from_elem((self.n_tgt_lat, self.n_tgt_lon), f32::NAN);
        for ((r, c), v) in result.indexed_iter_mut() {
            let m = mask_out[[r, c]];
            if m > 1e-15 {
                *v = (clean_out[[r, c]] / m) as f32;
            }
        }

        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    /// Helper: uniformly spaced latitudes (descending, like ERA5).
    fn uniform_lats(n: usize, start: f64, step: f64) -> Vec<f64> {
        (0..n).map(|i| start - i as f64 * step).collect()
    }

    /// Helper: uniformly spaced longitudes (ascending).
    fn uniform_lons(n: usize, start: f64, step: f64) -> Vec<f64> {
        (0..n).map(|i| start + i as f64 * step).collect()
    }

    #[test]
    fn test_centers_to_edges_uniform() {
        let centers = vec![10.0, 20.0, 30.0];
        let edges = centers_to_edges(&centers);
        assert_eq!(edges.len(), 4);
        assert!((edges[0] - 5.0).abs() < 1e-12);
        assert!((edges[1] - 15.0).abs() < 1e-12);
        assert!((edges[2] - 25.0).abs() < 1e-12);
        assert!((edges[3] - 35.0).abs() < 1e-12);
    }

    #[test]
    fn test_centers_to_edges_nonuniform() {
        let centers = vec![0.0, 1.0, 4.0];
        let edges = centers_to_edges(&centers);
        // first edge: 0 - (1-0)/2 = -0.5
        assert!((edges[0] - (-0.5)).abs() < 1e-12);
        // mid: (0+1)/2 = 0.5, (1+4)/2 = 2.5
        assert!((edges[1] - 0.5).abs() < 1e-12);
        assert!((edges[2] - 2.5).abs() < 1e-12);
        // last: 4 + (4-1)/2 = 5.5
        assert!((edges[3] - 5.5).abs() < 1e-12);
    }

    #[test]
    fn test_identity_regrid() {
        // Same source and target grid → output ≈ input
        let lats = uniform_lats(10, 45.0, 1.0); // 45, 44, ..., 36
        let lons = uniform_lons(10, 0.0, 1.0); // 0, 1, ..., 9

        let regridder = ConservativeRegridder::new(&lats, &lons, &lats, &lons).unwrap();
        assert_eq!(regridder.target_shape(), (10, 10));

        let data = Array2::from_shape_fn((10, 10), |(i, j)| (i * 10 + j) as f32);
        let result = regridder.regrid(&data).unwrap();

        for ((r, c), &v) in result.indexed_iter() {
            let expected = data[[r, c]];
            assert!(
                (v - expected).abs() < 1e-4,
                "at ({r},{c}): got {v}, expected {expected}"
            );
        }
    }

    #[test]
    fn test_coarsening_conserves_mean() {
        // 2x coarsening: 20×20 → 10×10
        // Uniform field should be preserved exactly
        let src_lats = uniform_lats(20, 45.0, 0.5);
        let src_lons = uniform_lons(20, 0.0, 0.5);
        let tgt_lats = uniform_lats(10, 44.75, 1.0);
        let tgt_lons = uniform_lons(10, 0.25, 1.0);

        let regridder = ConservativeRegridder::new(&src_lats, &src_lons, &tgt_lats, &tgt_lons)
            .unwrap();

        // Uniform value → should stay the same
        let data = Array2::from_elem((20, 20), 42.0_f32);
        let result = regridder.regrid(&data).unwrap();

        for &v in result.iter() {
            assert!(
                (v - 42.0).abs() < 1e-3,
                "uniform field should be preserved, got {v}"
            );
        }
    }

    #[test]
    fn test_area_conservation() {
        // Check that the area-weighted sum is preserved (not just mean).
        // For a small region where sin(lat) doesn't vary much, the
        // area-weighted integral should be approximately preserved.
        let src_lats = uniform_lats(8, 2.0, 0.5); // near equator: 2, 1.5, 1, 0.5, 0, -0.5, -1, -1.5
        let src_lons = uniform_lons(8, 10.0, 0.5);
        let tgt_lats = uniform_lats(4, 1.75, 1.0);
        let tgt_lons = uniform_lons(4, 10.25, 1.0);

        let regridder = ConservativeRegridder::new(&src_lats, &src_lons, &tgt_lats, &tgt_lons)
            .unwrap();

        let data = Array2::from_shape_fn((8, 8), |(i, j)| (i + j) as f32);
        let result = regridder.regrid(&data).unwrap();

        // Compute area-weighted sums using sin(lat) edges
        let src_lat_edges = centers_to_edges(&src_lats);
        let tgt_lat_edges = centers_to_edges(&tgt_lats);
        let src_lon_edges = centers_to_edges(&src_lons);
        let tgt_lon_edges = centers_to_edges(&tgt_lons);

        fn area_sum(
            data: &Array2<f32>,
            lat_edges: &[f64],
            lon_edges: &[f64],
        ) -> f64 {
            let (nr, nc) = data.dim();
            let mut total = 0.0;
            for r in 0..nr {
                let dlat = (lat_edges[r].to_radians().sin()
                    - lat_edges[r + 1].to_radians().sin())
                .abs();
                for c in 0..nc {
                    let dlon = (lon_edges[c + 1] - lon_edges[c]).abs();
                    if !data[[r, c]].is_nan() {
                        total += data[[r, c]] as f64 * dlat * dlon;
                    }
                }
            }
            total
        }

        let src_integral = area_sum(&data, &src_lat_edges, &src_lon_edges);
        let tgt_integral = area_sum(&result, &tgt_lat_edges, &tgt_lon_edges);

        let rel_err = (src_integral - tgt_integral).abs() / src_integral.abs();
        assert!(
            rel_err < 0.01,
            "area integral not conserved: src={src_integral}, tgt={tgt_integral}, rel_err={rel_err}"
        );
    }

    #[test]
    fn test_nan_handling() {
        let lats = uniform_lats(4, 45.0, 1.0);
        let lons = uniform_lons(4, 0.0, 1.0);
        let tgt_lats = uniform_lats(2, 44.5, 2.0);
        let tgt_lons = uniform_lons(2, 0.5, 2.0);

        let regridder =
            ConservativeRegridder::new(&lats, &lons, &tgt_lats, &tgt_lons).unwrap();

        let mut data = Array2::from_elem((4, 4), 1.0_f32);
        data[[0, 0]] = f32::NAN;

        let result = regridder.regrid(&data).unwrap();
        // No output cells should be NaN because each target cell has
        // at least some valid source contributions
        for &v in result.iter() {
            assert!(!v.is_nan(), "unexpected NaN in output");
        }
        // All values should be close to 1.0 (renormalized around the NaN)
        for &v in result.iter() {
            assert!(
                (v - 1.0).abs() < 0.1,
                "expected ~1.0 after NaN renormalization, got {v}"
            );
        }
    }

    #[test]
    fn test_all_nan_produces_nan() {
        let lats = uniform_lats(4, 45.0, 1.0);
        let lons = uniform_lons(4, 0.0, 1.0);

        let regridder =
            ConservativeRegridder::new(&lats, &lons, &lats, &lons).unwrap();

        let data = Array2::from_elem((4, 4), f32::NAN);
        let result = regridder.regrid(&data).unwrap();
        for &v in result.iter() {
            assert!(v.is_nan(), "expected NaN output for all-NaN input");
        }
    }

    #[test]
    fn test_shape_mismatch_error() {
        let lats = uniform_lats(4, 45.0, 1.0);
        let lons = uniform_lons(4, 0.0, 1.0);

        let regridder =
            ConservativeRegridder::new(&lats, &lons, &lats, &lons).unwrap();

        let wrong_data = Array2::zeros((3, 4));
        assert!(regridder.regrid(&wrong_data).is_err());
    }

    #[test]
    fn test_validation_non_monotonic() {
        let bad_lats = vec![45.0, 44.0, 46.0, 43.0]; // not monotonic
        let lons = uniform_lons(4, 0.0, 1.0);
        assert!(ConservativeRegridder::new(&bad_lats, &lons, &bad_lats, &lons).is_err());
    }

    #[test]
    fn test_validation_lat_out_of_range() {
        let bad_lats = vec![89.0, 90.0, 91.0, 92.0];
        let lons = uniform_lons(4, 0.0, 1.0);
        assert!(ConservativeRegridder::new(&bad_lats, &lons, &bad_lats, &lons).is_err());
    }

    #[test]
    fn test_validation_grid_too_small() {
        let tiny = vec![45.0]; // only 1 point
        let lons = uniform_lons(4, 0.0, 1.0);
        assert!(ConservativeRegridder::new(&tiny, &lons, &tiny, &lons).is_err());
    }

    #[test]
    fn test_ascending_lats() {
        // Ascending source latitudes (S→N) should work too
        let src_lats: Vec<f64> = (0..10).map(|i| -45.0 + i as f64).collect();
        let src_lons = uniform_lons(10, 0.0, 1.0);
        let tgt_lats: Vec<f64> = (0..5).map(|i| -44.5 + i as f64 * 2.0).collect();
        let tgt_lons = uniform_lons(5, 0.5, 2.0);

        let regridder =
            ConservativeRegridder::new(&src_lats, &src_lons, &tgt_lats, &tgt_lons).unwrap();

        let data = Array2::from_elem((10, 10), 7.0_f32);
        let result = regridder.regrid(&data).unwrap();

        for &v in result.iter() {
            assert!(
                (v - 7.0).abs() < 0.1,
                "uniform field should be preserved, got {v}"
            );
        }
    }
}
