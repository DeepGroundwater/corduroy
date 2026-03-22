//! Green-Ampt infiltration model (Julien et al. 1995, Equations 1–2).
//!
//! The Green-Ampt equation models infiltration into a homogeneous, deep,
//! well-drained soil column:
//!
//!   f = K * (1 + Hf * Md / F)                                   (Eq 1)
//!
//! where
//!   f  = infiltration rate (m/s)
//!   K  = saturated hydraulic conductivity (m/s)
//!   Hf = capillary pressure head at the wetting front (m)
//!   Md = initial moisture deficit = θe - θi (dimensionless)
//!   F  = cumulative infiltrated depth (m)
//!
//! The mid-timestep solution (Eq 2) avoids iteration by solving the
//! implicit F^(t+Δt) = F^t + f * Δt analytically:
//!
//!   f^(t+Δt/2) = (1/2Δt) * { (KΔt - 2F^t)
//!                              + sqrt[(KΔt - 2F^t)² + 8(KF^t + KHf·Md)Δt] }

use ndarray::Array2;

use super::{
    grid::SoilGrid,
    HydroError,
};

/// Per-cell infiltration state: cumulative depth F (m) infiltrated since simulation start.
/// Initialised to a small epsilon (not zero) to avoid division-by-zero in Eq 1 at t=0.
#[derive(Debug, Clone)]
pub struct InfiltrationState {
    /// Cumulative infiltrated depth F(j,k) in metres. shape: (nrows, ncols)
    pub f_cumul: Array2<f64>,
}

impl InfiltrationState {
    /// Initialise with F = ε for all cells (prevents 1/F singularity at t=0).
    pub fn new_zero(nrows: usize, ncols: usize) -> Self {
        Self {
            f_cumul: Array2::from_elem((nrows, ncols), 1e-12),
        }
    }
}

// ---------------------------------------------------------------------------
// Cell-level functions (public for unit testing and Python exposure)
// ---------------------------------------------------------------------------

/// Infiltration capacity f (m/s) at a single cell via Green-Ampt Eq 1.
///
/// Returns K when Hf·Md = 0 (already saturated soil).
#[inline]
pub fn green_ampt_capacity(k_sat: f64, h_cap: f64, m_def: f64, f_cumul: f64) -> f64 {
    k_sat * (1.0 + h_cap * m_def / f_cumul.max(1e-12))
}

/// Mid-timestep infiltration rate f^(t+Δt/2) (m/s) via Eq 2.
///
/// Derivation: set F^(t+Δt) = F^t + f·Δt, substitute f = K(1 + Hf·Md/F)
/// at the midpoint F = F^t + f·Δt/2, rearrange to a quadratic in f·Δt.
///
/// The discriminant is always ≥ 0 for non-negative parameters.
#[inline]
pub fn green_ampt_midstep(k_sat: f64, h_cap: f64, m_def: f64, f_t: f64, dt: f64) -> f64 {
    let f_t = f_t.max(1e-12);
    let a = k_sat * dt - 2.0 * f_t;
    let discriminant = a * a + 8.0 * (k_sat * f_t + k_sat * h_cap * m_def) * dt;
    // discriminant is guaranteed ≥ 0; clamp for floating-point safety
    (a + discriminant.max(0.0).sqrt()) / (2.0 * dt)
}

// ---------------------------------------------------------------------------
// Grid-level step
// ---------------------------------------------------------------------------

/// Advance infiltration by one timestep for all cells.
///
/// # Arguments
/// * `ie_gross` – gross rainfall rate (m/s) per cell, shape (nrows, ncols)
/// * `h`        – current overland water depth (m) per cell
/// * `soil`     – soil parameter grids
/// * `state`    – mutable infiltration state (F updated in place)
/// * `dt`       – timestep (s)
///
/// # Returns
/// Effective excess rainfall rate `ie_eff` (m/s) = ie_gross − actual_infiltration_rate.
/// Always ≥ 0.
pub fn step_infiltration(
    ie_gross: &Array2<f64>,
    h: &Array2<f64>,
    soil: &SoilGrid,
    state: &mut InfiltrationState,
    dt: f64,
) -> Result<Array2<f64>, HydroError> {
    let (nrows, ncols) = (soil.domain.nrows, soil.domain.ncols);
    let mut ie_eff = Array2::zeros((nrows, ncols));

    for j in 0..nrows {
        for k in 0..ncols {
            let k_sat = soil.k_sat[[j, k]];
            let h_cap = soil.h_cap[[j, k]];
            let m_def = soil.m_def[[j, k]];
            let f_t = state.f_cumul[[j, k]];
            let ie_g = ie_gross[[j, k]];
            let h_cur = h[[j, k]];

            // Compute infiltration capacity via mid-timestep Eq 2
            let f_cap = green_ampt_midstep(k_sat, h_cap, m_def, f_t, dt);

            // Available water rate: existing surface water + incoming rain
            // Cannot infiltrate more than what's physically present
            let available_rate = (h_cur + ie_g * dt) / dt;
            let actual_f = f_cap.min(available_rate).max(0.0);

            // Advance cumulative infiltration
            state.f_cumul[[j, k]] = f_t + actual_f * dt;

            ie_eff[[j, k]] = (ie_g - actual_f).max(0.0);
        }
    }

    Ok(ie_eff)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Eq 1 is a direct algebraic identity — verify floating-point accuracy.
    #[test]
    fn capacity_algebraic_identity() {
        let (k, hf, md, f) = (1e-5, 0.316, 0.274, 0.05);
        let expected = k * (1.0 + hf * md / f);
        let got = green_ampt_capacity(k, hf, md, f);
        assert!((got - expected).abs() < 1e-20, "expected {expected}, got {got}");
    }

    /// When Md = 0 (saturated soil), capacity should equal K regardless of F.
    #[test]
    fn capacity_saturated_soil() {
        let k = 3e-6;
        let f = green_ampt_capacity(k, 0.2, 0.0, 0.001);
        assert!((f - k).abs() < 1e-15, "saturated: expected K={k}, got {f}");
    }

    /// Mid-timestep solution must satisfy the implicit equation to high accuracy.
    /// The implicit equation is: f = K*(1 + Hf*Md / (F_t + f*dt/2))
    #[test]
    fn midstep_satisfies_implicit_equation() {
        let (k, hf, md, f_t, dt) = (1e-5, 0.316, 0.274, 0.01, 30.0);
        let f_mid = green_ampt_midstep(k, hf, md, f_t, dt);
        let f_mid_check = k * (1.0 + hf * md / (f_t + f_mid * dt / 2.0).max(1e-12));
        let rel_err = (f_mid - f_mid_check).abs() / f_mid_check;
        assert!(rel_err < 1e-10, "implicit residual too large: {rel_err:.2e}");
    }

    /// Energy balance: F^(t+dt) = F^t + actual_f * dt exactly.
    #[test]
    fn cumulative_depth_mass_balance() {
        use super::super::grid::{RasterDomain, SoilGrid};
        let domain = RasterDomain::new(1, 1, 100.0).unwrap();
        let soil = SoilGrid::uniform(domain, 1e-5, 0.316, 0.274, 0.06, 0.0);
        let mut state = InfiltrationState::new_zero(1, 1);
        let f_before = state.f_cumul[[0, 0]];

        let ie_gross = Array2::from_elem((1, 1), 5e-5_f64); // rain > K → some runoff
        let h = Array2::zeros((1, 1));
        let dt = 30.0;

        step_infiltration(&ie_gross, &h, &soil, &mut state, dt).unwrap();
        let f_after = state.f_cumul[[0, 0]];
        let actual_f_rate = (f_after - f_before) / dt;

        // ie_eff = ie_gross - actual_f_rate ≥ 0
        assert!(actual_f_rate >= 0.0);
        assert!(actual_f_rate <= ie_gross[[0, 0]] + 1e-15);
    }

    /// With zero rain and zero surface water, no infiltration should occur.
    #[test]
    fn no_infiltration_when_dry() {
        use super::super::grid::{RasterDomain, SoilGrid};
        let domain = RasterDomain::new(2, 2, 50.0).unwrap();
        let soil = SoilGrid::uniform(domain, 1e-5, 0.3, 0.3, 0.05, 0.0);
        let mut state = InfiltrationState::new_zero(2, 2);
        let f_before = state.f_cumul.clone();

        let ie_gross = Array2::zeros((2, 2));
        let h = Array2::zeros((2, 2));
        let ie_eff = step_infiltration(&ie_gross, &h, &soil, &mut state, 60.0).unwrap();

        // No available water → no infiltration, no excess
        for v in ie_eff.iter() {
            assert!(*v < 1e-15, "unexpected excess: {v}");
        }
        for (a, b) in state.f_cumul.iter().zip(f_before.iter()) {
            assert!((*a - *b).abs() < 1e-20, "F changed with no water source");
        }
    }

    /// Multi-step simulation: F should monotonically increase and f should
    /// decrease over time (as the wetting front deepens).
    #[test]
    fn f_increases_f_rate_decreases() {
        use super::super::grid::{RasterDomain, SoilGrid};
        let domain = RasterDomain::new(1, 1, 100.0).unwrap();
        // High rain rate so infiltration capacity limits, not water availability
        let soil = SoilGrid::uniform(domain, 1e-5, 0.316, 0.274, 0.06, 0.0);
        let mut state = InfiltrationState::new_zero(1, 1);
        let ie_gross = Array2::from_elem((1, 1), 1e-3_f64); // 1 mm/s — well above K
        let h = Array2::zeros((1, 1));
        let dt = 60.0;

        let mut prev_f_rate = f64::MAX;
        let mut prev_f_cumul = 0.0;
        for _ in 0..20 {
            let f_before = state.f_cumul[[0, 0]];
            step_infiltration(&ie_gross, &h, &soil, &mut state, dt).unwrap();
            let f_after = state.f_cumul[[0, 0]];
            let rate = (f_after - f_before) / dt;

            assert!(f_after > prev_f_cumul, "F should increase monotonically");
            assert!(rate <= prev_f_rate + 1e-15, "infiltration rate should not increase");

            prev_f_rate = rate;
            prev_f_cumul = f_after;
        }
    }
}
