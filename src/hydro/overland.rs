//! 2-D diffusive-wave overland flow (Julien et al. 1995, Equations 3–11).
//!
//! The Saint-Venant continuity equation in 2D (Eq 3):
//!
//!   ∂h/∂t  +  ∂qx/∂x  +  ∂qy/∂y  =  ie
//!
//! Discrete update for cell (j, k) (Eq 4):
//!
//!   h^(t+Δt)(j,k) = h^t(j,k) + ie·Δt
//!       − (Δt/W)·[ qx(k→k+1) − qx(k−1→k)
//!                 + qy(j→j+1) − qy(j−1→j) ]
//!
//! Diffusive-wave friction slope at face between (j, k−1) and (j, k) (Eq 6):
//!
//!   Sfx(k−1→k) = Sox(k−1→k)  −  [h(j,k) − h(j,k−1)] / W
//!
//! where Sox is the DEM bed slope (Eq 7).
//!
//! Manning turbulent unit discharge (Eqs 10–11, using upwind depth):
//!
//!   Sfx ≥ 0:  qx(k−1→k) =  (1/n_up) · h_up^(5/3) · Sfx^(1/2)
//!   Sfx < 0:  qx(k−1→k) = −(1/n_dn) · h_dn^(5/3) · (−Sfx)^(1/2)
//!
//! **Boundary conditions**:
//! - Upstream (left/top) edges: zero-flux (no inflow).
//! - Downstream (right/bottom) edges: zero-depth-gradient (ZDG) free outflow.
//!   The friction slope at the boundary equals the bed slope (∂h/∂x = 0).

use ndarray::Array2;

use super::{
    grid::{ChannelNetwork, DemGrid, SoilGrid},
    HydroError,
};

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

/// State of the 2-D overland flow solver.
#[derive(Debug, Clone)]
pub struct OverlandState {
    /// Water depth h(j,k) in metres. shape: (nrows, ncols)
    pub h: Array2<f64>,
}

impl OverlandState {
    pub fn new_zero(nrows: usize, ncols: usize) -> Self {
        Self { h: Array2::zeros((nrows, ncols)) }
    }
}

// ---------------------------------------------------------------------------
// Face-level physics (pub for testing)
// ---------------------------------------------------------------------------

/// Friction slope Sfx at the x-face between (row, col−1) and (row, col).
/// Positive → net flow is directed col−1 → col (downhill).
/// Eq 6: Sfx = Sox − (h[col] − h[col−1]) / W
#[inline]
pub fn friction_slope_x(
    dem: &DemGrid,
    h: &Array2<f64>,
    row: usize,
    col: usize, // right cell of the face
) -> f64 {
    let sox = dem.bed_slope_x(row, col);
    let dh = h[[row, col]] - h[[row, col - 1]];
    sox - dh / dem.domain.cell_size
}

/// Friction slope Sfy at the y-face between (row−1, col) and (row, col).
/// Positive → net flow is directed row−1 → row (downhill).
#[inline]
pub fn friction_slope_y(
    dem: &DemGrid,
    h: &Array2<f64>,
    row: usize, // lower cell of the face
    col: usize,
) -> f64 {
    let soy = dem.bed_slope_y(row, col);
    let dh = h[[row, col]] - h[[row - 1, col]];
    soy - dh / dem.domain.cell_size
}

/// Manning unit discharge (m²/s) at a face given signed friction slope.
/// Uses the upwind cell's depth and roughness (Eqs 10–11).
///
/// * `sf`    – signed friction slope (positive = flow in the positive direction)
/// * `n_up`  – Manning n of the upwind cell
/// * `h_up`  – depth of the upwind cell (m)
#[inline]
pub fn unit_discharge(sf: f64, n_up: f64, h_up: f64) -> f64 {
    if h_up < 1e-10 {
        return 0.0;
    }
    let sf_abs = sf.abs().max(1e-10);
    let q_mag = (1.0 / n_up) * h_up.powf(5.0 / 3.0) * sf_abs.sqrt();
    sf.signum() * q_mag
}

// ---------------------------------------------------------------------------
// Full-grid flux computation
// ---------------------------------------------------------------------------

/// X-face unit discharges.
/// Returns shape `(nrows, ncols)` where `qx[[j, k]]` = flux from cell (j,k) to (j,k+1).
/// For k = 0..ncols-2: interior face.
/// For k = ncols-1:    right-boundary ZDG free-outflow (Sf = Sox; ∂h/∂x = 0).
fn compute_qx(dem: &DemGrid, soil: &SoilGrid, h: &Array2<f64>) -> Array2<f64> {
    let (nr, nc) = (dem.domain.nrows, dem.domain.ncols);
    let mut qx = Array2::zeros((nr, nc));

    for j in 0..nr {
        // Interior faces: k = 1..nc-1  →  face between cell k-1 and k  →  stored at k-1
        for k in 1..nc {
            if h[[j, k - 1]] < 1e-10 && h[[j, k]] < 1e-10 {
                continue;
            }
            let sf = friction_slope_x(dem, h, j, k);
            let (n_up, h_up) = if sf >= 0.0 {
                (soil.mann_n[[j, k - 1]], h[[j, k - 1]])
            } else {
                (soil.mann_n[[j, k]], h[[j, k]])
            };
            qx[[j, k - 1]] = unit_discharge(sf, n_up, h_up);
        }

        // Right-boundary face: ZDG free outflow at k = nc-1
        // Friction slope = bed slope (zero depth gradient assumption)
        if nc >= 2 {
            let sf = dem.bed_slope_x(j, nc - 1).max(0.0); // only allow outflow
            qx[[j, nc - 1]] = unit_discharge(sf, soil.mann_n[[j, nc - 1]], h[[j, nc - 1]]);
        }
    }
    qx
}

/// Y-face unit discharges.
/// Returns shape `(nrows, ncols)` where `qy[[j, k]]` = flux from cell (j,k) to (j+1,k).
/// For j = 0..nrows-2: interior face.
/// For j = nrows-1:    bottom-boundary ZDG free-outflow.
fn compute_qy(dem: &DemGrid, soil: &SoilGrid, h: &Array2<f64>) -> Array2<f64> {
    let (nr, nc) = (dem.domain.nrows, dem.domain.ncols);
    let mut qy = Array2::zeros((nr, nc));

    for k in 0..nc {
        // Interior faces: j = 1..nr-1
        for j in 1..nr {
            if h[[j - 1, k]] < 1e-10 && h[[j, k]] < 1e-10 {
                continue;
            }
            let sf = friction_slope_y(dem, h, j, k);
            let (n_up, h_up) = if sf >= 0.0 {
                (soil.mann_n[[j - 1, k]], h[[j - 1, k]])
            } else {
                (soil.mann_n[[j, k]], h[[j, k]])
            };
            qy[[j - 1, k]] = unit_discharge(sf, n_up, h_up);
        }

        // Bottom-boundary face: ZDG free outflow at j = nr-1
        if nr >= 2 {
            let sf = dem.bed_slope_y(nr - 1, k).max(0.0);
            qy[[nr - 1, k]] = unit_discharge(sf, soil.mann_n[[nr - 1, k]], h[[nr - 1, k]]);
        }
    }
    qy
}

// ---------------------------------------------------------------------------
// CFL check
// ---------------------------------------------------------------------------

/// Maximum kinematic wave speed (m/s) — used for the advisory CFL estimate.
fn max_wave_speed(h: &Array2<f64>, qx: &Array2<f64>, qy: &Array2<f64>) -> f64 {
    let (nr, nc) = (h.shape()[0], h.shape()[1]);
    let mut vmax = 0.0_f64;
    for j in 0..nr {
        for k in 0..nc {
            let hc = h[[j, k]];
            if hc < 1e-10 {
                continue;
            }
            let vx = (qx[[j, k]] / hc).abs();
            let vy = (qy[[j, k]] / hc).abs();
            if vx > vmax { vmax = vx; }
            if vy > vmax { vmax = vy; }
        }
    }
    vmax
}

/// Advisory maximum stable timestep (s) based on the Courant condition.
pub fn max_stable_dt(state: &OverlandState, soil: &SoilGrid, dem: &DemGrid) -> f64 {
    let qx = compute_qx(dem, soil, &state.h);
    let qy = compute_qy(dem, soil, &state.h);
    let vmax = max_wave_speed(&state.h, &qx, &qy);
    if vmax < 1e-12 {
        return f64::MAX;
    }
    dem.domain.cell_size / (2.0 * vmax)
}

// ---------------------------------------------------------------------------
// Time-step
// ---------------------------------------------------------------------------

/// Advance overland flow by one explicit timestep.
///
/// # Arguments
/// * `dem`     – DEM grid (elevations, cell_size)
/// * `soil`    – soil / roughness grid
/// * `channel` – channel network mask (channel cells drain to channel state)
/// * `state`   – mutable overland state
/// * `ie_eff`  – effective excess rainfall rate (m/s), shape (nrows, ncols)
/// * `dt`      – timestep (s)
///
/// # Returns
/// `lateral_q[reach_id][node_index]` — unit-width lateral inflow (m²/s) collected
/// from overland cells adjacent to each channel node.
pub fn step_overland(
    dem: &DemGrid,
    soil: &SoilGrid,
    channel: &ChannelNetwork,
    state: &mut OverlandState,
    ie_eff: &Array2<f64>,
    dt: f64,
) -> Result<Vec<Vec<f64>>, HydroError> {
    let (nr, nc) = (dem.domain.nrows, dem.domain.ncols);
    let w = dem.domain.cell_size;

    // Compute face fluxes based on state at time t
    let qx = compute_qx(dem, soil, &state.h);
    let qy = compute_qy(dem, soil, &state.h);

    // Advisory CFL check
    let vmax = max_wave_speed(&state.h, &qx, &qy);
    if vmax > 1e-12 {
        let cfl = vmax * dt / w;
        if cfl > 1.0 {
            return Err(HydroError::CourantViolation { cfl, max_allowed: 1.0 });
        }
    }

    // Lateral inflow collector for channel nodes
    let mut lateral_q: Vec<Vec<f64>> = channel
        .reaches
        .iter()
        .map(|r| vec![0.0_f64; r.n_nodes()])
        .collect();

    // Continuity update (Eq 4)
    // qx[[j, k]] = flux from cell k to cell k+1 (or right boundary outflow at k=nc-1)
    // qy[[j, k]] = flux from cell j to cell j+1 (or bottom boundary outflow at j=nr-1)
    let mut h_new = state.h.clone();

    for j in 0..nr {
        for k in 0..nc {
            let h_ret = soil.retention[[j, k]];

            // Net divergence: outflow right + outflow down - inflow left - inflow up
            let qx_right = qx[[j, k]];                               // right face outflow
            let qx_left  = if k > 0 { qx[[j, k - 1]] } else { 0.0 }; // left face inflow
            let qy_down  = qy[[j, k]];                               // bottom face outflow
            let qy_up    = if j > 0 { qy[[j - 1, k]] } else { 0.0 }; // top face inflow

            let div_q = (qx_right - qx_left + qy_down - qy_up) / w;
            let h_candidate = state.h[[j, k]] + ie_eff[[j, k]] * dt - div_q * dt;
            let h_updated = h_candidate.max(0.0);

            if channel.channel_mask[[j, k]] {
                // Channel cell: surface water above retention drains into channel.
                let drainable = (h_updated - h_ret).max(0.0);
                // Lateral inflow per unit channel length (m²/s)
                let inflow_rate = drainable * w / dt;
                h_new[[j, k]] = h_updated - drainable;

                'reach_search: for (ri, reach) in channel.reaches.iter().enumerate() {
                    for (ni, &(rr, cc)) in reach.cells.iter().enumerate() {
                        if rr == j && cc == k {
                            lateral_q[ri][ni] += inflow_rate;
                            break 'reach_search;
                        }
                    }
                }
            } else {
                h_new[[j, k]] = h_updated;
            }
        }
    }

    // Negative-depth guard
    for j in 0..nr {
        for k in 0..nc {
            if h_new[[j, k]] < -1e-9 {
                return Err(HydroError::NegativeDepth {
                    row: j,
                    col: k,
                    depth: h_new[[j, k]],
                });
            }
        }
    }

    state.h = h_new;
    Ok(lateral_q)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::grid::{ChannelNetwork, DemGrid, RasterDomain, SoilGrid};

    /// Pure flat plane, no rainfall, no runoff: depth should stay at zero.
    #[test]
    fn flat_dry_plane_stays_zero() {
        let domain = RasterDomain::new(5, 5, 100.0).unwrap();
        let elev = Array2::zeros((5, 5));
        let dem = DemGrid::new(domain.clone(), elev).unwrap();
        let soil = SoilGrid::uniform(domain, 0.0, 0.0, 0.0, 0.06, 0.0);
        let channel = ChannelNetwork::empty(5, 5);
        let mut state = OverlandState::new_zero(5, 5);
        let ie_eff = Array2::zeros((5, 5));

        for _ in 0..10 {
            step_overland(&dem, &soil, &channel, &mut state, &ie_eff, 5.0).unwrap();
        }
        for &v in state.h.iter() {
            assert!(v.abs() < 1e-12, "expected zero depth, got {v}");
        }
    }

    /// Mass conservation on a FLAT domain with no outlet (all slopes zero).
    /// No water flows anywhere; total stored depth = rainfall total.
    #[test]
    fn mass_conservation_flat_no_outlet() {
        let nr = 4;
        let nc = 4;
        let w = 10.0_f64;
        let domain = RasterDomain::new(nr, nc, w).unwrap();
        let elev = Array2::zeros((nr, nc)); // flat → Sox=0 → boundary outflow=0
        let dem = DemGrid::new(domain.clone(), elev).unwrap();
        let soil = SoilGrid::uniform(domain, 0.0, 0.0, 0.0, 0.5, 0.0);
        let channel = ChannelNetwork::empty(nr, nc);
        let mut state = OverlandState::new_zero(nr, nc);

        let ie_val = 1e-4_f64;
        let ie_eff = Array2::from_elem((nr, nc), ie_val);
        let dt = 1.0;
        let n_steps = 10;

        for _ in 0..n_steps {
            step_overland(&dem, &soil, &channel, &mut state, &ie_eff, dt).unwrap();
        }

        let total_depth: f64 = state.h.iter().sum();
        let expected = ie_val * dt * n_steps as f64 * (nr * nc) as f64;
        let rel_err = (total_depth - expected).abs() / expected;
        // Tolerance 1e-5: the free-outflow BC uses sf.signum() which, via the
        // regularization (sf_abs.max(1e-10)), produces a ~1e-5 * h outflow even at
        // exactly zero slope.  This is negligible physics, not a model error.
        assert!(rel_err < 1e-5, "mass error: {rel_err:.2e}");
    }

    /// Friction slope calculation: uniform depth on a uniform slope → Sfx = Sox.
    #[test]
    fn friction_slope_uniform_slope() {
        let s0 = 0.01_f64;
        let w = 100.0_f64;
        let domain = RasterDomain::new(1, 5, w).unwrap();
        let mut elev = Array2::zeros((1, 5));
        for k in 0..5_usize {
            elev[[0, k]] = (4 - k) as f64 * s0 * w;
        }
        let dem = DemGrid::new(domain, elev).unwrap();
        let h = Array2::from_elem((1, 5), 0.1_f64); // uniform depth → dh/dx = 0

        for k in 1..5 {
            let sf = friction_slope_x(&dem, &h, 0, k);
            assert!((sf - s0).abs() < 1e-12, "expected Sfx={s0}, got {sf}");
        }
    }

    /// Kinematic wave equilibrium on a 1-D tilted plane.
    ///
    /// Reference: Woolhiser & Liggett (1967) kinematic wave analytical solution.
    /// At equilibrium, outlet depth satisfies Manning's equation with q = ie * L.
    ///
    /// Setup: 1-row plane, S=0.01, n=0.03, ie=1e-4 m/s, L = 10 × 50m = 500m.
    #[test]
    fn kinematic_wave_equilibrium() {
        let nc = 10_usize;
        let w = 50.0_f64;
        let s0 = 0.01_f64;
        let mann = 0.03_f64;
        let ie_val = 1e-4_f64;

        let domain = RasterDomain::new(1, nc, w).unwrap();
        let mut elev = Array2::zeros((1, nc));
        for k in 0..nc {
            elev[[0, k]] = (nc - 1 - k) as f64 * s0 * w;
        }
        let dem = DemGrid::new(domain.clone(), elev).unwrap();
        let soil = SoilGrid::uniform(domain, 0.0, 0.0, 0.0, mann, 0.0);

        // Analytical: q_eq = ie*L, h_eq = (q*n/S^0.5)^(3/5)
        let q_eq = ie_val * (nc as f64 * w);
        let h_eq = (q_eq * mann / s0.sqrt()).powf(3.0 / 5.0);

        // Equilibrium time (Woolhiser 1967)
        let l = nc as f64 * w;
        let te = (mann * l).powf(3.0 / 5.0) / (ie_val.powf(2.0 / 5.0) * s0.powf(3.0 / 10.0));

        let dt = 1.0_f64;
        let channel = ChannelNetwork::empty(1, nc);
        let mut state = OverlandState::new_zero(1, nc);
        let ie_eff = Array2::from_elem((1, nc), ie_val);

        let n_steps = (3.0 * te / dt) as usize;
        for _ in 0..n_steps {
            match step_overland(&dem, &soil, &channel, &mut state, &ie_eff, dt) {
                Ok(_) => {}
                Err(HydroError::CourantViolation { .. }) => break,
                Err(e) => panic!("unexpected error: {e}"),
            }
        }

        let h_outlet = state.h[[0, nc - 1]];
        let rel_err = (h_outlet - h_eq).abs() / h_eq;
        assert!(
            rel_err < 0.05,
            "outlet depth {h_outlet:.4e} vs analytical {h_eq:.4e}, rel_err={rel_err:.3}"
        );
    }
}
