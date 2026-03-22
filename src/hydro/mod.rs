pub mod channel;
pub mod grid;
pub mod infiltration;
pub mod overland;
pub mod runner;

pub use runner::{run, HydroOutput, HydroParams, HydroState};

#[derive(Debug, thiserror::Error)]
pub enum HydroError {
    #[error("grid shape mismatch: expected ({expected_rows}, {expected_cols}), got ({got_rows}, {got_cols})")]
    GridShapeMismatch {
        expected_rows: usize,
        expected_cols: usize,
        got_rows: usize,
        got_cols: usize,
    },
    #[error("negative water depth at cell ({row}, {col}): {depth:.6e} m")]
    NegativeDepth { row: usize, col: usize, depth: f64 },
    #[error("CFL violation: computed Courant number {cfl:.3} exceeds limit {max_allowed:.3}")]
    CourantViolation { cfl: f64, max_allowed: f64 },
    #[error("channel error: {msg}")]
    ChannelError { msg: String },
    #[error("invalid parameter: {msg}")]
    InvalidParameter { msg: String },
}
