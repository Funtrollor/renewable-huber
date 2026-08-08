//! Numerical kernels shared by the engine.
//!
//! Split by what they operate on rather than by size: `vector` is elementwise
//! and reduction work over a batch, `gram` accumulates the p-by-p matrices, and
//! `objective` is what the solver actually minimizes. The two bandwidth and
//! lambda policies below are scalar-only and belong with neither.

pub(crate) mod gram;
pub(crate) mod objective;
pub(crate) mod vector;

use rh_core::{Penalty, UpdateConfig};

pub(crate) fn bandwidth(n_total: f64, n_features: usize, scale: f64, tau: f64) -> f64 {
    let log_features = (n_features.max(2) as f64).ln();
    (scale / (n_total.sqrt() * log_features)).min(tau)
}

pub(crate) fn lambda_value(n_total: f64, n_features: usize, config: UpdateConfig) -> f64 {
    if config.penalty == Penalty::None {
        return 0.0;
    }
    config.lambda_scale * config.tau * (((n_features.max(2) as f64).ln() / n_total).sqrt())
}
