#ifndef RENEWABLE_HUBER_RH_CUDA_OBJECTIVE_CUH
#define RENEWABLE_HUBER_RH_CUDA_OBJECTIVE_CUH

/*
 * The objective, its gradient and Hessian, and the line-search rounds that
 * evaluate the objective at candidate points, replayed through a CUDA Graph
 * when the engine has graphs enabled.
 *
 * CandidateRoundEvaluator stays here rather than in a file of its own: it
 * owns the round's graph and mutates the engine's graph counters directly,
 * and the gradient and Hessian below consume the residuals it leaves.
 */

#include "engine_state.cuh"
#include "huber_kernels.cuh"

#include <cstdint>

namespace rh_cuda::engine {

/// Smoothed objective value and the convergence terms computed alongside it.
struct ObjectiveResult {
    double objective = 0.0;
    /// ||candidate - beta||, beta being the point the round searched from.
    double difference_norm = 0.0;
    /// ||candidate||.
    double beta_norm = 0.0;
    /// gradient . (candidate - beta); only when ObjectiveTerms::gradient.
    double gradient_dot = 0.0;
};

/// Inputs every round of one update shares.  The pointers stay fixed for the
/// lifetime of that update, which is what lets a captured CUDA Graph replay
/// them.
template <typename T>
struct ObjectiveTerms {
    /// LAMM: the gradient forming proximal candidates, also reduced against
    /// (candidate - beta) for the majorization bound.  Null for Newton.
    const T* gradient = nullptr;
    /// L1 historical subgradient sign(history) on penalized coordinates.  When
    /// set, the objective subtracts penalty_scale * sign . (c - history).
    const T* penalty_sign = nullptr;
    /// weight_sum / n_total * previous_lambda; host-side, never captured.
    double penalty_scale = 0.0;
    /// Coordinates the proximal threshold applies to.
    int64_t penalized_count = 0;
};

/// The candidates of one round, in the order the line search tries them.
struct CandidateRound {
    int width = 1;
    rh_cuda::CandidateForm form = rh_cuda::kCandidateCopy;
    /// Newton step, or 1 / phi for a proximal candidate; cast to T exactly as
    /// the scalar it replaces was.
    double scale[rh_cuda::kCandidateRoundWidth] = {};
    /// lambda / phi for a proximal candidate.
    double threshold[rh_cuda::kCandidateRoundWidth] = {};
};

/// Accumulate the gradient and Hessian for the current Newton step from
/// `residual`, the residual of `beta`.
template <typename T>
void compute_gradient_hessian(
    RhCudaEngine* engine,
    int rows,
    const T* beta,
    const T* residual,
    const T* weights,
    T tau,
    T bandwidth,
    double n_total,
    T ridge
);

/// Accumulate only the gradient of the L1 smooth surrogate at `beta` from
/// `residual`, the residual of `beta`.
template <typename T>
void compute_l1_gradient(
    RhCudaEngine* engine,
    int rows,
    const T* beta,
    const T* residual,
    const T* weights,
    T tau,
    double n_total,
    const T* penalty_sign,
    double penalty_scale
);

/// Form the renewable information matrix committed by this batch.  A null
/// `residual` recomputes the residual of d_trial_beta first.
template <typename T>
void final_information(
    RhCudaEngine* engine,
    int rows,
    const T* residual,
    const T* weights,
    T tau,
    T bandwidth
);

/// Evaluates line-search rounds for one update: up to kCandidateRoundWidth
/// candidates formed from d_trial_beta, with one transfer and one stream
/// synchronization per round.  When the engine has graphs enabled the round
/// is captured once and replayed; capture is best effort, and a failure
/// disables graphs for this engine and falls back to stream launches,
/// counting the fallback.
class CandidateRoundEvaluator final {
public:
    explicit CandidateRoundEvaluator(RhCudaEngine* engine);
    ~CandidateRoundEvaluator() noexcept;

    CandidateRoundEvaluator(const CandidateRoundEvaluator&) = delete;
    CandidateRoundEvaluator& operator=(const CandidateRoundEvaluator&) = delete;

    /// Evaluate `round` and fill results[0 .. round.width).  The candidates
    /// and residuals stay readable until the next round.
    template <typename T>
    void evaluate(
        int rows,
        const T* weights,
        T tau,
        double n_total,
        const ObjectiveTerms<T>& terms,
        const CandidateRound& round,
        ObjectiveResult* results
    );

    /// Candidate `k` of the last round.
    template <typename T>
    const T* candidate(int k) const;

    /// Residual of candidate `k` of the last round.
    template <typename T>
    const T* residual(int rows, int k) const;

private:
    template <typename T>
    void enqueue(int rows, const T* weights, T tau, const ObjectiveTerms<T>& terms);

    template <typename T>
    bool capture(int rows, const T* weights, T tau, const ObjectiveTerms<T>& terms);

    RhCudaEngine* engine_;
    bool graphs_ = false;
    bool launched_once_ = false;
    cudaGraph_t graph_ = nullptr;
    cudaGraphExec_t execution_ = nullptr;
};

}  // namespace rh_cuda::engine

#endif  // RENEWABLE_HUBER_RH_CUDA_OBJECTIVE_CUH
