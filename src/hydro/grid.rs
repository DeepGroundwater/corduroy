use ndarray::Array2;

use super::HydroError;

/// Uniform raster domain: nrows × ncols cells, each W metres on a side.
#[derive(Debug, Clone)]
pub struct RasterDomain {
    pub nrows: usize,
    pub ncols: usize,
    /// Cell width W (m) — same in x and y.
    pub cell_size: f64,
}

impl RasterDomain {
    pub fn new(nrows: usize, ncols: usize, cell_size: f64) -> Result<Self, HydroError> {
        if nrows == 0 || ncols == 0 {
            return Err(HydroError::InvalidParameter {
                msg: format!("grid must have at least 1 row and 1 col, got ({nrows}, {ncols})"),
            });
        }
        if cell_size <= 0.0 {
            return Err(HydroError::InvalidParameter {
                msg: format!("cell_size must be positive, got {cell_size}"),
            });
        }
        Ok(Self { nrows, ncols, cell_size })
    }
}

// ---------------------------------------------------------------------------
// Digital Elevation Model
// ---------------------------------------------------------------------------

/// Ground surface elevation E(j,k) in metres above datum.
/// Row j increases southward; column k increases eastward.
#[derive(Debug, Clone)]
pub struct DemGrid {
    pub domain: RasterDomain,
    /// shape: (nrows, ncols)
    pub elev: Array2<f64>,
}

impl DemGrid {
    pub fn new(domain: RasterDomain, elev: Array2<f64>) -> Result<Self, HydroError> {
        let shape = elev.shape();
        if shape[0] != domain.nrows || shape[1] != domain.ncols {
            return Err(HydroError::GridShapeMismatch {
                expected_rows: domain.nrows,
                expected_cols: domain.ncols,
                got_rows: shape[0],
                got_cols: shape[1],
            });
        }
        Ok(Self { domain, elev })
    }

    /// Bed slope Sox in the x-direction at the face between (row, col-1) and (row, col).
    /// Sox = [E(row, col-1) - E(row, col)] / W   (Eq 7)
    /// Positive means flow from col-1 → col is downhill.
    /// Panics if col == 0 (no left face exists).
    pub fn bed_slope_x(&self, row: usize, col: usize) -> f64 {
        debug_assert!(col > 0, "no x-face at col=0");
        (self.elev[[row, col - 1]] - self.elev[[row, col]]) / self.domain.cell_size
    }

    /// Bed slope Soy in the y-direction at the face between (row-1, col) and (row, col).
    /// Soy = [E(row-1, col) - E(row, col)] / W
    /// Positive means flow from row-1 → row is downhill.
    /// Panics if row == 0 (no top face exists).
    pub fn bed_slope_y(&self, row: usize, col: usize) -> f64 {
        debug_assert!(row > 0, "no y-face at row=0");
        (self.elev[[row - 1, col]] - self.elev[[row, col]]) / self.domain.cell_size
    }
}

// ---------------------------------------------------------------------------
// Soil / roughness grid
// ---------------------------------------------------------------------------

/// Per-cell soil and roughness properties for infiltration and overland flow.
#[derive(Debug, Clone)]
pub struct SoilGrid {
    pub domain: RasterDomain,
    /// Saturated hydraulic conductivity K (m/s)
    pub k_sat: Array2<f64>,
    /// Capillary suction head at wetting front Hf (m)
    pub h_cap: Array2<f64>,
    /// Initial moisture deficit Md = θe - θi (dimensionless, 0–1)
    pub m_def: Array2<f64>,
    /// Manning roughness coefficient n for overland flow
    pub mann_n: Array2<f64>,
    /// Retention storage depth (m); surface water does not run off until exceeded
    pub retention: Array2<f64>,
}

impl SoilGrid {
    pub fn new(
        domain: RasterDomain,
        k_sat: Array2<f64>,
        h_cap: Array2<f64>,
        m_def: Array2<f64>,
        mann_n: Array2<f64>,
        retention: Array2<f64>,
    ) -> Result<Self, HydroError> {
        for (name, arr) in [
            ("k_sat", &k_sat),
            ("h_cap", &h_cap),
            ("m_def", &m_def),
            ("mann_n", &mann_n),
            ("retention", &retention),
        ] {
            let s = arr.shape();
            if s[0] != domain.nrows || s[1] != domain.ncols {
                return Err(HydroError::GridShapeMismatch {
                    expected_rows: domain.nrows,
                    expected_cols: domain.ncols,
                    got_rows: s[0],
                    got_cols: s[1],
                });
            }
            let _ = name; // suppress unused warning in release
        }
        Ok(Self { domain, k_sat, h_cap, m_def, mann_n, retention })
    }

    /// Convenience: spatially uniform soil properties over the entire domain.
    pub fn uniform(
        domain: RasterDomain,
        k_sat: f64,
        h_cap: f64,
        m_def: f64,
        mann_n: f64,
        retention: f64,
    ) -> Self {
        let shape = (domain.nrows, domain.ncols);
        Self {
            k_sat: Array2::from_elem(shape, k_sat),
            h_cap: Array2::from_elem(shape, h_cap),
            m_def: Array2::from_elem(shape, m_def),
            mann_n: Array2::from_elem(shape, mann_n),
            retention: Array2::from_elem(shape, retention),
            domain,
        }
    }
}

// ---------------------------------------------------------------------------
// Channel network
// ---------------------------------------------------------------------------

/// Trapezoidal cross-section geometry for a channel reach.
#[derive(Debug, Clone)]
pub struct CrossSection {
    /// Bottom width (m)
    pub bottom_width: f64,
    /// Side slope z (horizontal / vertical, i.e. 1:z bank angle)
    pub side_slope: f64,
    /// Bankfull depth DCH (m); water above this spills to floodplain
    pub bank_depth: f64,
}

impl CrossSection {
    /// Flow cross-sectional area (m²) for depth d.
    pub fn area(&self, d: f64) -> f64 {
        let d = d.max(0.0);
        (self.bottom_width + self.side_slope * d) * d
    }

    /// Wetted perimeter (m) for depth d.
    pub fn wetted_perimeter(&self, d: f64) -> f64 {
        let d = d.max(0.0);
        self.bottom_width + 2.0 * d * (1.0_f64 + self.side_slope * self.side_slope).sqrt()
    }

    /// Hydraulic radius R = A / P (m).
    pub fn hydraulic_radius(&self, d: f64) -> f64 {
        let p = self.wetted_perimeter(d);
        if p < 1e-12 {
            0.0
        } else {
            self.area(d) / p
        }
    }

    /// Bankfull cross-sectional area (m²).
    pub fn bankfull_area(&self) -> f64 {
        self.area(self.bank_depth)
    }
}

/// A single channel reach: an ordered list of (row, col) cells flowing downstream.
#[derive(Debug, Clone)]
pub struct ChannelReach {
    pub reach_id: usize,
    /// Cell indices in downstream order: (row, col)
    pub cells: Vec<(usize, usize)>,
    /// Manning n for in-channel flow
    pub mann_n: f64,
    /// Cross-section (assumed uniform along the reach)
    pub cross_section: CrossSection,
    /// Segment length between successive nodes (m). Length = cells.len() - 1.
    pub segment_lengths: Vec<f64>,
}

impl ChannelReach {
    pub fn new(
        reach_id: usize,
        cells: Vec<(usize, usize)>,
        mann_n: f64,
        cross_section: CrossSection,
        cell_size: f64,
    ) -> Result<Self, HydroError> {
        if cells.len() < 2 {
            return Err(HydroError::ChannelError {
                msg: format!("reach {reach_id} needs at least 2 cells, got {}", cells.len()),
            });
        }
        // Default: straight segment = cell_size (can be overridden for diagonal cells)
        let segment_lengths = vec![cell_size; cells.len() - 1];
        Ok(Self { reach_id, cells, mann_n, cross_section, segment_lengths })
    }

    pub fn n_nodes(&self) -> usize {
        self.cells.len()
    }
}

/// The complete channel network for the watershed.
#[derive(Debug, Clone)]
pub struct ChannelNetwork {
    pub reaches: Vec<ChannelReach>,
    /// Boolean mask: true where a raster cell is a channel cell.
    /// shape: (nrows, ncols)
    pub channel_mask: Array2<bool>,
}

impl ChannelNetwork {
    pub fn new(reaches: Vec<ChannelReach>, nrows: usize, ncols: usize) -> Self {
        let mut mask = Array2::from_elem((nrows, ncols), false);
        for reach in &reaches {
            for &(r, c) in &reach.cells {
                mask[[r, c]] = true;
            }
        }
        Self { reaches, channel_mask: mask }
    }

    /// Returns an empty network (no channels — pure overland flow).
    pub fn empty(nrows: usize, ncols: usize) -> Self {
        Self {
            reaches: Vec::new(),
            channel_mask: Array2::from_elem((nrows, ncols), false),
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dem_bed_slopes() {
        // Plane tilted uniformly in x: E(j,k) = 10.0 - k * 0.01 * W
        let w = 100.0_f64;
        let domain = RasterDomain::new(3, 5, w).unwrap();
        let mut elev = Array2::zeros((3, 5));
        for k in 0..5_usize {
            for j in 0..3_usize {
                elev[[j, k]] = 10.0 - k as f64 * 0.01 * w;
            }
        }
        let dem = DemGrid::new(domain, elev).unwrap();
        // Sox = [E(j,k-1) - E(j,k)] / W = 0.01 * W / W = 0.01
        for j in 0..3 {
            for k in 1..5 {
                let s = dem.bed_slope_x(j, k);
                assert!((s - 0.01).abs() < 1e-12, "expected 0.01 got {s}");
            }
        }
        // No slope in y
        for j in 1..3 {
            for k in 0..5 {
                let s = dem.bed_slope_y(j, k);
                assert!(s.abs() < 1e-12, "expected 0.0 got {s}");
            }
        }
    }

    #[test]
    fn cross_section_geometry() {
        // Rectangular channel: side_slope = 0, bottom_width = 2.0 m, depth = 1.0 m
        let xs = CrossSection { bottom_width: 2.0, side_slope: 0.0, bank_depth: 1.5 };
        let d = 1.0;
        assert!((xs.area(d) - 2.0).abs() < 1e-12);
        assert!((xs.wetted_perimeter(d) - 4.0).abs() < 1e-12);
        assert!((xs.hydraulic_radius(d) - 0.5).abs() < 1e-12);
    }

    #[test]
    fn channel_mask_built_correctly() {
        let reach = ChannelReach::new(
            0,
            vec![(0, 0), (1, 0), (2, 0)],
            0.03,
            CrossSection { bottom_width: 1.0, side_slope: 0.0, bank_depth: 1.0 },
            100.0,
        )
        .unwrap();
        let net = ChannelNetwork::new(vec![reach], 3, 3);
        assert!(net.channel_mask[[0, 0]]);
        assert!(net.channel_mask[[1, 0]]);
        assert!(net.channel_mask[[2, 0]]);
        assert!(!net.channel_mask[[0, 1]]);
    }
}
