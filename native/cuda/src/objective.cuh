#ifndef RENEWABLE_HUBER_RH_CUDA_OBJECTIVE_CUH
#define RENEWABLE_HUBER_RH_CUDA_OBJECTIVE_CUH

/*
 * The objective, its gradient and Hessian, and the CUDA Graph that replays the
 * line-search candidate evaluation.
 *
 * CandidateObjectiveGraph stays here rather than in a file of its own: it is a
 * thin wrapper over enqueue_/finish_smooth_objective and mutates the engine's
 * graph counters directly, so separating it would force those two internals to
 * become exported templates for no gain.
 */

#include "engine_state.cuh"

#include <cstdint>

namespace rh_cuda::engine {

/// Smoothed objective value and the convergence terms computed alongside it.
struct ObjectiveResult {
    double objective = 0.0;
    double difference_norm = 0.0;
    double beta_norm = 0.0;
};

/// Evaluate the smoothed Huber objective at `beta`.
template <typename T>
ObjectiveResult smooth_objective(
    RhCudaEngine* engine,
    int rows,
    const T* beta,
    const T* weights,
    T tau,
    double n_total,
    const T* previous_beta = nullptr
);

/// Accumulate the gradient and Hessian for the current Newton step.
template <typename T>
void compute_gradient_hessian(
    RhCudaEngine* engine,
    int rows,
    const T* beta,
    const T* weights,
    T tau,
    T bandwidth,
    double n_total,
    T ridge
);

/// Form the renewable information matrix committed by this batch.
template <typename T>
void final_information(
    RhCudaEngine* engine,
    int rows,
    const T* weights,
    T tau,
    T bandwidth,
    bool residual_is_current
);

/// Line-search candidate evaluation, replayed through a CUDA Graph when the
/// engine has graphs enabled and capture succeeded.  Capture is best effort:
/// a failure disables graphs for this engine and falls back to ordinary
/// stream launches, counting the fallback.
class CandidateObjectiveGraph final {
public:
    explicit CandidateObjectiveGraph(RhCudaEngine* engine);
    ~CandidateObjectiveGraph() noexcept;

    CandidateObjectiveGraph(const CandidateObjectiveGraph&) = delete;
    CandidateObjectiveGraph& operator=(const CandidateObjectiveGraph&) = delete;

    template <typename T>
    ObjectiveResult evaluate(
        int rows,
        const T* beta,
        const T* weights,
        T tau,
        double n_total,
        const T* previous_beta
    );

private:
    template <typename T>
    bool capture(int rows, const T* beta, const T* weights, T tau, const T* previous_beta);

    RhCudaEngine* engine_;
    bool enabled_ = false;
    bool launched_once_ = false;
    cudaGraph_t graph_ = nullptr;
    cudaGraphExec_t execution_ = nullptr;
};

}  // namespace rh_cuda::engine

#endif  // RENEWABLE_HUBER_RH_CUDA_OBJECTIVE_CUH
