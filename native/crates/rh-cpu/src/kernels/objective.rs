//! The smoothed objective and the diagnostic value reported to callers.

use rh_core::{scalar_from_f64, BatchView, CoreError, Penalty, State, UpdateConfig};

use crate::kernels::vector::{dot, huber_loss, residual, sign};
use crate::scalar::CpuScalar;
use crate::workspace::Workspace;

pub(crate) fn smooth_objective<T: CpuScalar>(
    batch: BatchView<'_, T>,
    beta: &[T],
    state: &State<T>,
    config: UpdateConfig,
    n_total: f64,
    workspace: &mut Workspace<T>,
) -> Result<f64, CoreError> {
    let p = batch.n_parameters;
    residual(
        batch.x_design,
        batch.n_rows,
        p,
        beta,
        batch.y,
        &mut workspace.residual,
    );
    let tau = scalar_from_f64::<T>(config.tau)?;
    let mut current = T::zero();
    for row in 0..batch.n_rows {
        let weight = batch.sample_weight.map_or(T::one(), |weights| weights[row]);
        current += weight * huber_loss(workspace.residual[row], tau);
    }
    for (delta, (beta_value, state_value)) in workspace
        .delta
        .iter_mut()
        .zip(beta.iter().zip(state.coefficients.iter()))
    {
        *delta = *beta_value - *state_value;
    }
    let mut historical = T::zero();
    for row in 0..p {
        historical += workspace.delta[row]
            * dot(&state.information[row * p..(row + 1) * p], &workspace.delta);
    }
    let half = T::one() / (T::one() + T::one());
    let divisor = scalar_from_f64::<T>(n_total)?;
    let mut objective = (current + half * historical) / divisor;
    if config.penalty == Penalty::L1 && state.n_samples_seen > 0 {
        let historical_scale =
            scalar_from_f64::<T>(state.weight_sum / n_total * state.previous_lambda)?;
        let mut product = T::zero();
        for parameter in 0..p {
            if !(state.fit_intercept && parameter + 1 == p) {
                product += workspace.delta[parameter] * sign(state.coefficients[parameter]);
            }
        }
        objective -= historical_scale * product;
    }
    objective.to_f64().ok_or(CoreError::ScalarConversion)
}

pub(crate) fn diagnostic_objective<T: CpuScalar>(
    smooth_objective: f64,
    coefficients: &[T],
    state: &State<T>,
    penalty: Penalty,
    lambda_value: f64,
) -> f64 {
    if penalty == Penalty::None {
        return smooth_objective;
    }
    let last_penalized = coefficients.len() - usize::from(state.fit_intercept);
    let l1 = coefficients[..last_penalized]
        .iter()
        .filter_map(|value| num_traits::Float::abs(*value).to_f64())
        .sum::<f64>();
    smooth_objective + lambda_value * l1
}
