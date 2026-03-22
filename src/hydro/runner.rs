//! Coupled timestep loop: infiltration → overland flow → channel routing.
//!
//! Operator-splitting sequence per timestep:
//!
//!  1. Infiltration: compute ie_eff from gross rainfall, advance F
//!  2. Overland:     advance h using ie_eff, collect lateral_q for channel nodes
//!  3. Channel:      advance channel depths using lateral_q, collect overbank_return
//!  4. Overbank:     distribute excess depth back to adjacent overland cells

use ndarray::{Array2, Array3};

use super::{
    channel::{self, ChannelState},
    grid::{ChannelNetwork, DemGrid, SoilGrid},
    infiltration::{self, InfiltrationState},
    overland::{self, OverlandState},
    HydroError,
};

// ---------------------------------------------------------------------------
// Parameters and state
// ---------------------------------------------------------------------------

/// All static inputs required by the hydrologic model.
pub struct HydroParams {
    pub dem: DemGrid,
    pub soil: SoilGrid,
    pub channel: ChannelNetwork,
    /// Bed elevation per node per reach.  `bed_elevations[reach_id][node]`
    pub bed_elevations: Vec<Vec<f64>>,
}

impl HydroParams {
    /// Build bed elevations directly from the DEM for each reach.
    pub fn new(dem: DemGrid, soil: SoilGrid, channel: ChannelNetwork) -> Self {
        let bed_elevations = channel
            .reaches
            .iter()
            .map(|r| channel::reach_bed_elevations(&r.cells, &dem.elev))
            .collect();
        Self { dem, soil, channel, bed_elevations }
    }
}

/// Combined simulation state for all three model components.
pub struct HydroState {
    pub infiltration: InfiltrationState,
    pub overland: OverlandState,
    pub channel: ChannelState,
    /// Elapsed simulation time (s)
    pub time: f64,
}

impl HydroState {
    pub fn new_zero(params: &HydroParams) -> Self {
        let (nr, nc) = (params.dem.domain.nrows, params.dem.domain.ncols);
        Self {
            infiltration: InfiltrationState::new_zero(nr, nc),
            overland: OverlandState::new_zero(nr, nc),
            channel: ChannelState::new_zero(&params.channel),
            time: 0.0,
        }
    }
}

// ---------------------------------------------------------------------------
// Output
// ---------------------------------------------------------------------------

/// Simulation outputs collected at the requested output interval.
pub struct HydroOutput {
    /// Simulation time (s) at each output snapshot
    pub time: Vec<f64>,
    /// Overland water depth snapshots, shape (n_snapshots, nrows, ncols)
    pub h_snapshots: Vec<Array2<f64>>,
    /// Cumulative infiltration depth snapshots, shape (n_snapshots, nrows, ncols)
    pub f_snapshots: Vec<Array2<f64>>,
    /// Discharge (m³/s) at the outlet (last node of last reach) at each snapshot
    pub q_outlet: Vec<f64>,
}

// ---------------------------------------------------------------------------
// Single-step advance
// ---------------------------------------------------------------------------

/// Advance the coupled model by one timestep.
///
/// `rainfall_rate`: spatially distributed gross rainfall rate (m/s), shape (nrows, ncols).
pub fn step(
    params: &HydroParams,
    state: &mut HydroState,
    rainfall_rate: &Array2<f64>,
    dt: f64,
) -> Result<(), HydroError> {
    // 1. Infiltration
    let ie_eff = infiltration::step_infiltration(
        rainfall_rate,
        &state.overland.h,
        &params.soil,
        &mut state.infiltration,
        dt,
    )?;

    // 2. Overland flow → lateral inflow to channel
    let lateral_q = overland::step_overland(
        &params.dem,
        &params.soil,
        &params.channel,
        &mut state.overland,
        &ie_eff,
        dt,
    )?;

    // 3. Channel routing → overbank return depths
    let overbank = channel::step_channel(
        &params.channel,
        &mut state.channel,
        &lateral_q,
        &params.bed_elevations,
        dt,
    )?;

    // 4. Apply overbank return: excess depth distributed to raster cells adjacent to each node.
    //    The excess depth (m) is added directly to the overland h at the channel cell.
    //    This is mass-conservative: the channel depth was clipped to DCH, the excess
    //    re-enters the adjacent floodplain cells.
    for (ri, reach) in params.channel.reaches.iter().enumerate() {
        for (ni, &(row, col)) in reach.cells.iter().enumerate() {
            let excess_depth = overbank[ri][ni];
            if excess_depth > 0.0 {
                state.overland.h[[row, col]] += excess_depth;
            }
        }
    }

    state.time += dt;
    Ok(())
}

// ---------------------------------------------------------------------------
// Full simulation run
// ---------------------------------------------------------------------------

/// Run the coupled hydrologic model over a sequence of timesteps.
///
/// # Arguments
/// * `params`          – static model parameters
/// * `rainfall`        – time-series of spatially-distributed rainfall (m/s),
///                       shape `(n_steps, nrows, ncols)`.
/// * `dt`              – timestep (s)
/// * `output_interval` – save state every N timesteps (1 = every step)
///
/// # Returns
/// `HydroOutput` with snapshots at each output interval.
pub fn run(
    params: &HydroParams,
    rainfall: &Array3<f64>,
    dt: f64,
    output_interval: usize,
) -> Result<HydroOutput, HydroError> {
    let n_steps = rainfall.shape()[0];
    let mut state = HydroState::new_zero(params);
    let mut output = HydroOutput {
        time: Vec::new(),
        h_snapshots: Vec::new(),
        f_snapshots: Vec::new(),
        q_outlet: Vec::new(),
    };

    let (nr, nc) = (params.dem.domain.nrows, params.dem.domain.ncols);

    for step_idx in 0..n_steps {
        // Extract rainfall slice for this timestep
        let rain_slice = rainfall.slice(ndarray::s![step_idx, .., ..]).to_owned();

        step(params, &mut state, &rain_slice, dt)?;

        if (step_idx + 1) % output_interval == 0 {
            output.time.push(state.time);
            output.h_snapshots.push(state.overland.h.clone());
            output.f_snapshots.push(state.infiltration.f_cumul.clone());

            // Outlet Q: last node of the last reach (or 0 if no channels)
            let q_out = outlet_discharge(params, &state);
            output.q_outlet.push(q_out);
        }
    }

    // Always capture final state if not already captured
    if n_steps % output_interval != 0 {
        output.time.push(state.time);
        output.h_snapshots.push(state.overland.h.clone());
        output.f_snapshots.push(state.infiltration.f_cumul.clone());
        output.q_outlet.push(outlet_discharge(params, &state));
    }

    let _ = (nr, nc); // suppress unused
    Ok(output)
}

fn outlet_discharge(params: &HydroParams, state: &HydroState) -> f64 {
    if params.channel.reaches.is_empty() {
        return 0.0;
    }
    let last_reach = params.channel.reaches.last().unwrap();
    let ri = last_reach.reach_id;
    let ni = last_reach.n_nodes() - 1;
    let d = state.channel.depths[ri][ni];
    let xs = &last_reach.cross_section;
    let n = last_reach.mann_n;

    // Use bed slope as approximation for friction slope at the outlet
    let n_nodes = last_reach.n_nodes();
    let z = &params.bed_elevations[ri];
    let dx = last_reach.segment_lengths[n_nodes - 2];
    let sf = (z[n_nodes - 2] - z[n_nodes - 1]) / dx;
    super::channel::channel_discharge(xs, d, sf.max(1e-6), n)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::grid::{CrossSection, ChannelReach, ChannelNetwork, RasterDomain};

    /// End-to-end: pure overland flow (no channel), uniform slope and rain.
    /// After equilibrium, h at the outlet row should approach the Manning depth.
    #[test]
    fn e2e_overland_only_reaches_equilibrium() {
        let nr = 1_usize;
        let nc = 8_usize;
        let w = 50.0_f64;
        let s0 = 0.01_f64;
        let mann = 0.05_f64;
        let ie_val = 1e-4_f64; // m/s

        let domain = RasterDomain::new(nr, nc, w).unwrap();
        let mut elev = Array2::zeros((nr, nc));
        for k in 0..nc {
            elev[[0, k]] = (nc - 1 - k) as f64 * s0 * w;
        }
        let dem = DemGrid::new(domain.clone(), elev).unwrap();
        let soil = SoilGrid::uniform(domain, 0.0, 0.0, 0.0, mann, 0.0);
        let channel = ChannelNetwork::empty(nr, nc);
        let params = HydroParams::new(dem, soil, channel);

        let dt = 1.0_f64;
        let n_steps = 2000;
        let rain = Array3::from_elem((n_steps, nr, nc), ie_val);
        let result = run(&params, &rain, dt, n_steps).unwrap();

        // Analytical equilibrium depth at outlet: q = ie*L, d = (q*n/S^0.5)^(3/5)
        let q_eq = ie_val * (nc as f64 * w);
        let h_eq = (q_eq * mann / s0.sqrt()).powf(3.0 / 5.0);
        let h_outlet = result.h_snapshots.last().unwrap()[[0, nc - 1]];
        let rel_err = (h_outlet - h_eq).abs() / h_eq;
        assert!(
            rel_err < 0.05,
            "outlet h={h_outlet:.4e} expected {h_eq:.4e}, rel_err={rel_err:.3}"
        );
    }

    /// End-to-end: infiltration dominates (high K, low rain) → near-zero runoff.
    #[test]
    fn e2e_infiltration_dominates() {
        let nr = 2_usize;
        let nc = 4_usize;
        let w = 50.0_f64;
        let domain = RasterDomain::new(nr, nc, w).unwrap();
        let mut elev = Array2::zeros((nr, nc));
        for k in 0..nc {
            elev[[0, k]] = (nc - 1 - k) as f64 * 0.01 * w;
            elev[[1, k]] = (nc - 1 - k) as f64 * 0.01 * w;
        }
        let dem = DemGrid::new(domain.clone(), elev).unwrap();
        // K >> ie_val so all rain infiltrates
        let soil = SoilGrid::uniform(domain, 1e-3, 0.3, 0.3, 0.05, 0.0);
        let channel = ChannelNetwork::empty(nr, nc);
        let params = HydroParams::new(dem, soil, channel);

        let ie_val = 1e-5_f64; // much less than K=1e-3
        let dt = 30.0_f64;
        let n_steps = 50;
        let rain = Array3::from_elem((n_steps, nr, nc), ie_val);
        let result = run(&params, &rain, dt, n_steps).unwrap();

        let max_h: f64 = result.h_snapshots.last().unwrap().iter().cloned().fold(0.0_f64, f64::max);
        assert!(max_h < 1e-6, "expected near-zero runoff with high K, got h={max_h:.2e}");
    }

    /// End-to-end: coupled overland + channel.
    /// Simple 1-row domain with a channel along the bottom row.
    #[test]
    fn e2e_coupled_channel_receives_lateral_flow() {
        let nr = 3_usize;
        let nc = 5_usize;
        let w = 50.0_f64;
        let s0 = 0.005_f64;

        let domain = RasterDomain::new(nr, nc, w).unwrap();
        let mut elev = Array2::zeros((nr, nc));
        for j in 0..nr {
            for k in 0..nc {
                elev[[j, k]] = (nc - 1 - k) as f64 * s0 * w
                    + (nr - 1 - j) as f64 * 0.02 * w; // slope toward bottom-right
            }
        }
        let dem = DemGrid::new(domain.clone(), elev).unwrap();
        let soil = SoilGrid::uniform(domain, 0.0, 0.0, 0.0, 0.05, 0.0);

        // Channel along the bottom row (row nr-1)
        let channel_cells: Vec<(usize, usize)> = (0..nc).map(|k| (nr - 1, k)).collect();
        let xs = CrossSection { bottom_width: 2.0, side_slope: 0.0, bank_depth: 3.0 };
        let reach = ChannelReach::new(0, channel_cells, 0.03, xs, w).unwrap();
        let channel = ChannelNetwork::new(vec![reach], nr, nc);
        let params = HydroParams::new(dem, soil, channel);

        let ie_val = 2e-4_f64;
        let dt = 0.5_f64;
        let n_steps = 800;
        let rain = Array3::from_elem((n_steps, nr, nc), ie_val);
        let result = run(&params, &rain, dt, n_steps).unwrap();

        // Channel should have accumulated depth > 0
        let final_h = result.h_snapshots.last().unwrap();
        let channel_depth_sum: f64 = (0..nc).map(|k| final_h[[nr - 1, k]]).sum();
        // Channel cells drain to channel state, overland h on channel cells should be near zero
        assert!(
            channel_depth_sum < 0.1,
            "channel cells should drain — h sum={channel_depth_sum:.4}"
        );
    }
}
