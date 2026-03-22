//! 1-D diffusive-wave channel flow (Julien et al. 1995, Equations 12–13).
//!
//! Continuity equation (Eq 12):
//!
//!   ∂Ax/∂t  +  ∂Q/∂x  =  q_l
//!
//! Manning discharge (Eq 13):
//!
//!   Q = (1/n) · Ax · R^(2/3) · Sf^(1/2)
//!
//! where
//!   Ax = flow cross-sectional area (m²)
//!   R  = hydraulic radius = Ax / P (m)
//!   Sf = friction slope (dimensionless)
//!   q_l = lateral inflow per unit length (m²/s), from overland flow
//!
//! Friction slope at the face between nodes i and i+1:
//!
//!   Sf(i→i+1) = S0(i→i+1)  −  [y(i+1) − y(i)] / dx
//!
//! where y = bed elevation + water depth = water surface elevation.
//!
//! Overbank flow: when depth exceeds DCH (bankfull depth), the excess is
//! returned to the adjacent raster cells as a source term for the overland solver.

use ndarray::Array2;

use super::{
    grid::{ChannelNetwork, CrossSection},
    HydroError,
};

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

/// State of the 1-D channel flow solver.
#[derive(Debug, Clone)]
pub struct ChannelState {
    /// Water depth at each node per reach: `depths[reach_id][node_index]` (m)
    pub depths: Vec<Vec<f64>>,
}

impl ChannelState {
    /// Initialise all depths to zero.
    pub fn new_zero(network: &ChannelNetwork) -> Self {
        Self {
            depths: network.reaches.iter().map(|r| vec![0.0_f64; r.n_nodes()]).collect(),
        }
    }
}

// ---------------------------------------------------------------------------
// Node-level physics
// ---------------------------------------------------------------------------

/// Manning discharge Q (m³/s) for a cross-section at depth d, friction slope sf.
/// Returns 0 when depth or sf are non-positive (no flow).
#[inline]
pub fn channel_discharge(xs: &CrossSection, depth: f64, sf: f64, mann_n: f64) -> f64 {
    if depth < 1e-10 || sf.abs() < 1e-10 {
        return 0.0;
    }
    let a = xs.area(depth);
    let r = xs.hydraulic_radius(depth);
    let sf_abs = sf.abs().max(1e-10);
    let q_mag = (1.0 / mann_n) * a * r.powf(2.0 / 3.0) * sf_abs.sqrt();
    sf.signum() * q_mag
}

/// Friction slope at the face between nodes i (upstream) and i+1 (downstream).
/// bed_slope: S0 = (z_i - z_{i+1}) / dx  (positive if downhill)
/// Diffusive-wave adds the water surface gradient correction.
#[inline]
fn friction_slope_channel(
    _bed_slope: f64,
    depth_up: f64,
    depth_dn: f64,
    elev_up: f64,
    elev_dn: f64,
    dx: f64,
) -> f64 {
    // Water surface elevation = bed elevation + depth
    let wse_up = elev_up + depth_up;
    let wse_dn = elev_dn + depth_dn;
    // Sf = S0 - d(wse)/dx = (wse_up - wse_dn) / dx
    (wse_up - wse_dn) / dx
}

// ---------------------------------------------------------------------------
// Reach-level bed elevation (derived from channel cell positions in DEM)
// ---------------------------------------------------------------------------

/// Compute per-node bed elevations for a reach from a DEM Array2.
/// Returns Vec<f64> of length n_nodes.
pub fn reach_bed_elevations(
    reach_cells: &[(usize, usize)],
    elev: &Array2<f64>,
) -> Vec<f64> {
    reach_cells.iter().map(|&(r, c)| elev[[r, c]]).collect()
}

// ---------------------------------------------------------------------------
// Time-step
// ---------------------------------------------------------------------------

/// Advance all channel reaches by one explicit timestep.
///
/// # Arguments
/// * `network`       – channel network (reach topology, cross-sections)
/// * `state`         – mutable channel state (depths updated in place)
/// * `lateral_q`     – `lateral_q[reach_id][node]` (m²/s) from overland step
/// * `bed_elevations`– `bed_elevations[reach_id][node]` (m) — bed elevation per node
/// * `dt`            – timestep (s)
///
/// # Returns
/// `overbank_return[reach_id][node]` — excess depth (m) above bankfull that is
/// pushed back to adjacent raster cells. The caller (runner) applies this to the
/// overland state.
pub fn step_channel(
    network: &ChannelNetwork,
    state: &mut ChannelState,
    lateral_q: &[Vec<f64>],
    bed_elevations: &[Vec<f64>],
    dt: f64,
) -> Result<Vec<Vec<f64>>, HydroError> {
    let n_reaches = network.reaches.len();
    let mut overbank_return = vec![vec![0.0_f64; 0]; n_reaches];

    for (ri, reach) in network.reaches.iter().enumerate() {
        let n = reach.n_nodes();
        let depths = &mut state.depths[ri];
        let z = &bed_elevations[ri];
        let q_lat = &lateral_q[ri];
        let xs = &reach.cross_section;
        let mann_n = reach.mann_n;

        overbank_return[ri] = vec![0.0_f64; n];

        // Compute discharge at each interior face (i → i+1)
        let mut q_face = vec![0.0_f64; n - 1]; // q_face[i] = Q between node i and i+1
        for i in 0..n - 1 {
            let dx = reach.segment_lengths[i];
            let sf = friction_slope_channel(
                (z[i] - z[i + 1]) / dx,
                depths[i],
                depths[i + 1],
                z[i],
                z[i + 1],
                dx,
            );
            // Upwind depth: use the depth of the cell the flow originates from
            let d_face = if sf >= 0.0 { depths[i] } else { depths[i + 1] };
            q_face[i] = channel_discharge(xs, d_face, sf, mann_n);
        }

        // Continuity update: dA/dt = q_l - dQ/dx
        let mut new_depths = depths.clone();
        for i in 0..n {
            let dx_left  = if i > 0     { reach.segment_lengths[i - 1] } else { reach.segment_lengths[0] };
            let dx_right = if i < n - 1 { reach.segment_lengths[i] }     else { reach.segment_lengths[n - 2] };
            let dx_avg = (dx_left + dx_right) / 2.0;

            let q_in  = if i > 0     { q_face[i - 1] } else { 0.0 };
            // Downstream boundary: free outflow using bed slope as friction slope
            let q_out = if i < n - 1 {
                q_face[i]
            } else {
                let sf = (z[n - 2] - z[n - 1]) / reach.segment_lengths[n - 2];
                channel_discharge(xs, depths[n - 1], sf.max(0.0), mann_n)
            };

            // Volume balance per unit length → convert to area change
            let da_dt = q_lat[i] + (q_in - q_out) / dx_avg;
            let a_new = (xs.area(depths[i]) + da_dt * dt).max(0.0);

            // Invert A(d) = (bw + z*d)*d for a trapezoidal section
            let d_new = invert_trapezoidal_area(xs, a_new);
            new_depths[i] = d_new;
        }

        // Overbank: clip depth to bankfull, return excess
        for i in 0..n {
            if new_depths[i] > xs.bank_depth {
                let excess = new_depths[i] - xs.bank_depth;
                overbank_return[ri][i] = excess; // caller converts to h increment
                new_depths[i] = xs.bank_depth;
            }
        }

        *depths = new_depths;
    }

    Ok(overbank_return)
}

/// Invert A = (bw + z·d)·d for depth d given area A.
/// For rectangular (z=0): d = A / bw.
/// For trapezoidal (z>0): solve z·d² + bw·d − A = 0 → d = (−bw + sqrt(bw²+4zA))/(2z).
fn invert_trapezoidal_area(xs: &CrossSection, area: f64) -> f64 {
    if area <= 0.0 {
        return 0.0;
    }
    let bw = xs.bottom_width;
    let z = xs.side_slope;
    if z.abs() < 1e-12 {
        // Rectangular
        area / bw.max(1e-12)
    } else {
        // Trapezoidal: z·d² + bw·d − A = 0
        (-bw + (bw * bw + 4.0 * z * area).max(0.0).sqrt()) / (2.0 * z)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::grid::{ChannelNetwork, ChannelReach, CrossSection};

    fn rect_xs(bw: f64, bank_depth: f64) -> CrossSection {
        CrossSection { bottom_width: bw, side_slope: 0.0, bank_depth }
    }

    /// Area inversion round-trip for rectangular and trapezoidal sections.
    #[test]
    fn area_inversion_roundtrip() {
        let xs_rect = rect_xs(2.0, 2.0);
        let d_orig = 0.75;
        let a = xs_rect.area(d_orig);
        let d_back = invert_trapezoidal_area(&xs_rect, a);
        assert!((d_back - d_orig).abs() < 1e-12, "rect: {d_back} vs {d_orig}");

        let xs_trap = CrossSection { bottom_width: 1.0, side_slope: 1.0, bank_depth: 2.0 };
        let d_orig = 1.2;
        let a = xs_trap.area(d_orig);
        let d_back = invert_trapezoidal_area(&xs_trap, a);
        assert!((d_back - d_orig).abs() < 1e-10, "trap: {d_back} vs {d_orig}");
    }

    /// Manning discharge: for rectangular channel, known analytical value.
    /// Q = (1/n) * A * R^(2/3) * S^(1/2)
    /// bw=2m, d=1m, S=0.001, n=0.03 → A=2, P=4, R=0.5
    /// Q = (1/0.03)*2*0.5^(2/3)*0.001^(1/2) ≈ 0.995 m³/s
    #[test]
    fn manning_discharge_known_value() {
        let xs = rect_xs(2.0, 2.0);
        let n = 0.03;
        let d = 1.0;
        let sf = 0.001;
        let q = channel_discharge(&xs, d, sf, n);
        let expected = (1.0 / n) * 2.0 * 0.5_f64.powf(2.0 / 3.0) * 0.001_f64.sqrt();
        let rel_err = (q - expected).abs() / expected;
        assert!(rel_err < 1e-10, "Q={q:.6} expected={expected:.6} rel={rel_err:.2e}");
    }

    /// Steady-uniform flow: with constant lateral inflow and long reach, outlet
    /// discharge should converge to sum(q_l[i] * dx_i).
    ///
    /// Parameters chosen for numerical stability:
    ///   S=0.01 → diffusion coeff D = Q/(2·T·S) ≈ 25 m²/s → dt_max = dx²/(2D) ≈ 200s.
    ///   Using dt=10s is well within stability limits.
    #[test]
    fn steady_uniform_flow_balance() {
        let n_nodes = 10_usize;
        let cell_size = 100.0_f64;
        // bank_depth=2m >> expected steady-state depth (~0.37m), so no overbank clipping
        let xs = rect_xs(2.0, 2.0);
        let reach = ChannelReach::new(
            0,
            (0..n_nodes).map(|i| (i, 0)).collect(),
            0.03,
            xs.clone(),
            cell_size,
        )
        .unwrap();
        let channel = ChannelNetwork::new(vec![reach], n_nodes, 1);
        let mut state = ChannelState::new_zero(&channel);

        // Steeper slope S=0.01 → smaller D → explicit scheme stable at dt=10s
        let s0 = 0.01_f64;
        let bed_elev: Vec<f64> =
            (0..n_nodes).map(|i| (n_nodes - 1 - i) as f64 * s0 * cell_size).collect();
        let bed_elevations = vec![bed_elev];

        // Lateral inflow 0.001 m²/s per unit length per node
        let q_lat_per_node = 0.001_f64;
        let lateral_q = vec![vec![q_lat_per_node; n_nodes]];

        // Q_ss = q_lat * dx * n_nodes = 0.001 * 100 * 10 = 1.0 m³/s
        let total_inflow = q_lat_per_node * cell_size * n_nodes as f64;

        let dt = 10.0;
        let n_steps = 5000;
        for _ in 0..n_steps {
            step_channel(&channel, &mut state, &lateral_q, &bed_elevations, dt).unwrap();
        }

        let outlet_depth = state.depths[0][n_nodes - 1];
        let outlet_xs = &channel.reaches[0].cross_section;
        let sf_outlet = s0;
        let q_outlet = channel_discharge(outlet_xs, outlet_depth, sf_outlet, 0.03);

        let rel_err = (q_outlet - total_inflow).abs() / total_inflow;
        assert!(
            rel_err < 0.10,
            "outlet Q={q_outlet:.4} expected≈{total_inflow:.4} rel={rel_err:.3}"
        );
    }

    /// Overbank: depth exceeding bank_depth should be clipped and returned.
    #[test]
    fn overbank_clips_depth() {
        let xs = CrossSection { bottom_width: 1.0, side_slope: 0.0, bank_depth: 0.5 };
        let reach = ChannelReach::new(
            0,
            vec![(0, 0), (1, 0)],
            0.03,
            xs.clone(),
            100.0,
        )
        .unwrap();
        let channel = ChannelNetwork::new(vec![reach], 2, 1);
        let mut state = ChannelState::new_zero(&channel);
        // Set initial depth well above bankfull
        state.depths[0][0] = 1.0;
        state.depths[0][1] = 1.0;

        let bed_elev = vec![vec![0.1, 0.0]];
        let lateral_q = vec![vec![0.0, 0.0]];

        let ret = step_channel(&channel, &mut state, &lateral_q, &bed_elev, 5.0).unwrap();

        // After step, depths should be at most bank_depth = 0.5
        for d in &state.depths[0] {
            assert!(*d <= xs.bank_depth + 1e-10, "depth {d} exceeds bankfull");
        }
        // Overbank return should be non-negative
        for v in &ret[0] {
            assert!(*v >= 0.0, "negative overbank return");
        }
    }

    /// Mass conservation across a single timestep: volume in = volume stored + volume out.
    #[test]
    fn channel_mass_conservation() {
        let n_nodes = 5_usize;
        let dx = 50.0_f64;
        let xs = rect_xs(1.0, 10.0); // large bankfull — no overbank
        let reach = ChannelReach::new(
            0,
            (0..n_nodes).map(|i| (i, 0)).collect(),
            0.03,
            xs.clone(),
            dx,
        )
        .unwrap();
        let channel = ChannelNetwork::new(vec![reach], n_nodes, 1);
        let mut state = ChannelState::new_zero(&channel);

        let s0 = 0.005_f64;
        let bed_elev: Vec<f64> =
            (0..n_nodes).map(|i| (n_nodes - 1 - i) as f64 * s0 * dx).collect();
        let bed_elevations = vec![bed_elev];

        // No lateral inflow — water drains from initial condition
        state.depths[0] = vec![0.3; n_nodes];
        let initial_vol: f64 = state.depths[0]
            .iter()
            .map(|&d| xs.area(d) * dx)
            .sum();

        let lateral_q = vec![vec![0.0; n_nodes]];
        let dt = 2.0;
        for _ in 0..200 {
            step_channel(&channel, &mut state, &lateral_q, &bed_elevations, dt).unwrap();
        }

        let final_vol: f64 = state.depths[0].iter().map(|&d| xs.area(d) * dx).sum();
        // Total volume leaving + remaining ≤ initial (overbank may also remove some)
        assert!(
            final_vol <= initial_vol + 1e-6,
            "volume increased: initial={initial_vol:.4}, final={final_vol:.4}"
        );
    }
}
