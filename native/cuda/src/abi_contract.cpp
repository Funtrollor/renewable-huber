/*
 * Compile-time mirror of native/contracts/rh_cuda_contract.json.
 *
 * This translation unit defines nothing.  It exists so the real compiler --
 * not a source-text parser -- proves that rh_cuda.h still lays its public
 * structs out the way every other language binding assumes.  A field reorder,
 * a type change, or a 32-bit target breaks the build here rather than
 * silently misreading memory across the ABI.
 *
 * rh_cuda.h includes only <stdint.h>, so this file compiles without CUDA:
 *
 *     g++ -std=c++17 -fsyntax-only -I native/cuda/include \
 *         native/cuda/src/abi_contract.cpp
 *
 * CI runs exactly that on a runner with no GPU and no CUDA Toolkit.
 *
 * Generated from the manifest; keep the two in step.
 */

#include "rh_cuda.h"

#include <cstddef>

static_assert(sizeof(void*) == 8, "the renewable-huber CUDA ABI is 64-bit only");
static_assert(RH_CUDA_ABI_VERSION == 1u, "ABI version drifted from the contract manifest");


static_assert(RH_CUDA_STATUS_SUCCESS == 0, "status code RH_CUDA_STATUS_SUCCESS drifted from the contract manifest");
static_assert(RH_CUDA_STATUS_INVALID_ARGUMENT == 1, "status code RH_CUDA_STATUS_INVALID_ARGUMENT drifted from the contract manifest");
static_assert(RH_CUDA_STATUS_UNAVAILABLE == 2, "status code RH_CUDA_STATUS_UNAVAILABLE drifted from the contract manifest");
static_assert(RH_CUDA_STATUS_ALLOCATION_FAILED == 3, "status code RH_CUDA_STATUS_ALLOCATION_FAILED drifted from the contract manifest");
static_assert(RH_CUDA_STATUS_CUDA_ERROR == 4, "status code RH_CUDA_STATUS_CUDA_ERROR drifted from the contract manifest");
static_assert(RH_CUDA_STATUS_CUBLAS_ERROR == 5, "status code RH_CUDA_STATUS_CUBLAS_ERROR drifted from the contract manifest");
static_assert(RH_CUDA_STATUS_CUSOLVER_ERROR == 6, "status code RH_CUDA_STATUS_CUSOLVER_ERROR drifted from the contract manifest");
static_assert(RH_CUDA_STATUS_NUMERICAL_ERROR == 7, "status code RH_CUDA_STATUS_NUMERICAL_ERROR drifted from the contract manifest");
static_assert(RH_CUDA_STATUS_INTERNAL_ERROR == 8, "status code RH_CUDA_STATUS_INTERNAL_ERROR drifted from the contract manifest");

static_assert(RH_CUDA_DTYPE_FLOAT32 == 1, "dtype code RH_CUDA_DTYPE_FLOAT32 drifted from the contract manifest");
static_assert(RH_CUDA_DTYPE_FLOAT64 == 2, "dtype code RH_CUDA_DTYPE_FLOAT64 drifted from the contract manifest");

static_assert(RH_CUDA_ENGINE_FLAG_CUDA_GRAPHS == 1u, "engine flag RH_CUDA_ENGINE_FLAG_CUDA_GRAPHS drifted from the contract manifest");
static_assert(RH_CUDA_ENGINE_FLAG_FAST_MATH == 2u, "engine flag RH_CUDA_ENGINE_FLAG_FAST_MATH drifted from the contract manifest");
static_assert(RH_CUDA_ENGINE_FLAG_KNOWN_MASK == 3u, "engine flag RH_CUDA_ENGINE_FLAG_KNOWN_MASK drifted from the contract manifest");

static_assert(sizeof(RhCudaEngineOptions) == 32, "RhCudaEngineOptions size drifted from the contract manifest");
static_assert(offsetof(RhCudaEngineOptions, abi_version) == 0, "RhCudaEngineOptions.abi_version offset drifted from the contract manifest");
static_assert(sizeof(((RhCudaEngineOptions*)0)->abi_version) == 4, "RhCudaEngineOptions.abi_version width drifted from the contract manifest");
static_assert(offsetof(RhCudaEngineOptions, struct_size) == 4, "RhCudaEngineOptions.struct_size offset drifted from the contract manifest");
static_assert(sizeof(((RhCudaEngineOptions*)0)->struct_size) == 4, "RhCudaEngineOptions.struct_size width drifted from the contract manifest");
static_assert(offsetof(RhCudaEngineOptions, dtype) == 8, "RhCudaEngineOptions.dtype offset drifted from the contract manifest");
static_assert(sizeof(((RhCudaEngineOptions*)0)->dtype) == 4, "RhCudaEngineOptions.dtype width drifted from the contract manifest");
static_assert(offsetof(RhCudaEngineOptions, device_id) == 12, "RhCudaEngineOptions.device_id offset drifted from the contract manifest");
static_assert(sizeof(((RhCudaEngineOptions*)0)->device_id) == 4, "RhCudaEngineOptions.device_id width drifted from the contract manifest");
static_assert(offsetof(RhCudaEngineOptions, n_parameters) == 16, "RhCudaEngineOptions.n_parameters offset drifted from the contract manifest");
static_assert(sizeof(((RhCudaEngineOptions*)0)->n_parameters) == 8, "RhCudaEngineOptions.n_parameters width drifted from the contract manifest");
static_assert(offsetof(RhCudaEngineOptions, reserved0) == 24, "RhCudaEngineOptions.reserved0 offset drifted from the contract manifest");
static_assert(sizeof(((RhCudaEngineOptions*)0)->reserved0) == 8, "RhCudaEngineOptions.reserved0 width drifted from the contract manifest");

static_assert(sizeof(RhCudaHostStateView) == 56, "RhCudaHostStateView size drifted from the contract manifest");
static_assert(offsetof(RhCudaHostStateView, abi_version) == 0, "RhCudaHostStateView.abi_version offset drifted from the contract manifest");
static_assert(sizeof(((RhCudaHostStateView*)0)->abi_version) == 4, "RhCudaHostStateView.abi_version width drifted from the contract manifest");
static_assert(offsetof(RhCudaHostStateView, struct_size) == 4, "RhCudaHostStateView.struct_size offset drifted from the contract manifest");
static_assert(sizeof(((RhCudaHostStateView*)0)->struct_size) == 4, "RhCudaHostStateView.struct_size width drifted from the contract manifest");
static_assert(offsetof(RhCudaHostStateView, coefficients) == 8, "RhCudaHostStateView.coefficients offset drifted from the contract manifest");
static_assert(sizeof(((RhCudaHostStateView*)0)->coefficients) == 8, "RhCudaHostStateView.coefficients width drifted from the contract manifest");
static_assert(offsetof(RhCudaHostStateView, information) == 16, "RhCudaHostStateView.information offset drifted from the contract manifest");
static_assert(sizeof(((RhCudaHostStateView*)0)->information) == 8, "RhCudaHostStateView.information width drifted from the contract manifest");
static_assert(offsetof(RhCudaHostStateView, n_samples_seen) == 24, "RhCudaHostStateView.n_samples_seen offset drifted from the contract manifest");
static_assert(sizeof(((RhCudaHostStateView*)0)->n_samples_seen) == 8, "RhCudaHostStateView.n_samples_seen width drifted from the contract manifest");
static_assert(offsetof(RhCudaHostStateView, batch_count) == 32, "RhCudaHostStateView.batch_count offset drifted from the contract manifest");
static_assert(sizeof(((RhCudaHostStateView*)0)->batch_count) == 8, "RhCudaHostStateView.batch_count width drifted from the contract manifest");
static_assert(offsetof(RhCudaHostStateView, previous_lambda) == 40, "RhCudaHostStateView.previous_lambda offset drifted from the contract manifest");
static_assert(sizeof(((RhCudaHostStateView*)0)->previous_lambda) == 8, "RhCudaHostStateView.previous_lambda width drifted from the contract manifest");
static_assert(offsetof(RhCudaHostStateView, weight_sum) == 48, "RhCudaHostStateView.weight_sum offset drifted from the contract manifest");
static_assert(sizeof(((RhCudaHostStateView*)0)->weight_sum) == 8, "RhCudaHostStateView.weight_sum width drifted from the contract manifest");

static_assert(sizeof(RhCudaHostState) == 56, "RhCudaHostState size drifted from the contract manifest");
static_assert(offsetof(RhCudaHostState, abi_version) == 0, "RhCudaHostState.abi_version offset drifted from the contract manifest");
static_assert(sizeof(((RhCudaHostState*)0)->abi_version) == 4, "RhCudaHostState.abi_version width drifted from the contract manifest");
static_assert(offsetof(RhCudaHostState, struct_size) == 4, "RhCudaHostState.struct_size offset drifted from the contract manifest");
static_assert(sizeof(((RhCudaHostState*)0)->struct_size) == 4, "RhCudaHostState.struct_size width drifted from the contract manifest");
static_assert(offsetof(RhCudaHostState, coefficients) == 8, "RhCudaHostState.coefficients offset drifted from the contract manifest");
static_assert(sizeof(((RhCudaHostState*)0)->coefficients) == 8, "RhCudaHostState.coefficients width drifted from the contract manifest");
static_assert(offsetof(RhCudaHostState, information) == 16, "RhCudaHostState.information offset drifted from the contract manifest");
static_assert(sizeof(((RhCudaHostState*)0)->information) == 8, "RhCudaHostState.information width drifted from the contract manifest");
static_assert(offsetof(RhCudaHostState, n_samples_seen) == 24, "RhCudaHostState.n_samples_seen offset drifted from the contract manifest");
static_assert(sizeof(((RhCudaHostState*)0)->n_samples_seen) == 8, "RhCudaHostState.n_samples_seen width drifted from the contract manifest");
static_assert(offsetof(RhCudaHostState, batch_count) == 32, "RhCudaHostState.batch_count offset drifted from the contract manifest");
static_assert(sizeof(((RhCudaHostState*)0)->batch_count) == 8, "RhCudaHostState.batch_count width drifted from the contract manifest");
static_assert(offsetof(RhCudaHostState, previous_lambda) == 40, "RhCudaHostState.previous_lambda offset drifted from the contract manifest");
static_assert(sizeof(((RhCudaHostState*)0)->previous_lambda) == 8, "RhCudaHostState.previous_lambda width drifted from the contract manifest");
static_assert(offsetof(RhCudaHostState, weight_sum) == 48, "RhCudaHostState.weight_sum offset drifted from the contract manifest");
static_assert(sizeof(((RhCudaHostState*)0)->weight_sum) == 8, "RhCudaHostState.weight_sum width drifted from the contract manifest");

static_assert(sizeof(RhCudaUnpenalizedConfig) == 56, "RhCudaUnpenalizedConfig size drifted from the contract manifest");
static_assert(offsetof(RhCudaUnpenalizedConfig, abi_version) == 0, "RhCudaUnpenalizedConfig.abi_version offset drifted from the contract manifest");
static_assert(sizeof(((RhCudaUnpenalizedConfig*)0)->abi_version) == 4, "RhCudaUnpenalizedConfig.abi_version width drifted from the contract manifest");
static_assert(offsetof(RhCudaUnpenalizedConfig, struct_size) == 4, "RhCudaUnpenalizedConfig.struct_size offset drifted from the contract manifest");
static_assert(sizeof(((RhCudaUnpenalizedConfig*)0)->struct_size) == 4, "RhCudaUnpenalizedConfig.struct_size width drifted from the contract manifest");
static_assert(offsetof(RhCudaUnpenalizedConfig, n_features_in) == 8, "RhCudaUnpenalizedConfig.n_features_in offset drifted from the contract manifest");
static_assert(sizeof(((RhCudaUnpenalizedConfig*)0)->n_features_in) == 8, "RhCudaUnpenalizedConfig.n_features_in width drifted from the contract manifest");
static_assert(offsetof(RhCudaUnpenalizedConfig, max_iter) == 16, "RhCudaUnpenalizedConfig.max_iter offset drifted from the contract manifest");
static_assert(sizeof(((RhCudaUnpenalizedConfig*)0)->max_iter) == 8, "RhCudaUnpenalizedConfig.max_iter width drifted from the contract manifest");
static_assert(offsetof(RhCudaUnpenalizedConfig, tau) == 24, "RhCudaUnpenalizedConfig.tau offset drifted from the contract manifest");
static_assert(sizeof(((RhCudaUnpenalizedConfig*)0)->tau) == 8, "RhCudaUnpenalizedConfig.tau width drifted from the contract manifest");
static_assert(offsetof(RhCudaUnpenalizedConfig, bandwidth_scale) == 32, "RhCudaUnpenalizedConfig.bandwidth_scale offset drifted from the contract manifest");
static_assert(sizeof(((RhCudaUnpenalizedConfig*)0)->bandwidth_scale) == 8, "RhCudaUnpenalizedConfig.bandwidth_scale width drifted from the contract manifest");
static_assert(offsetof(RhCudaUnpenalizedConfig, tolerance) == 40, "RhCudaUnpenalizedConfig.tolerance offset drifted from the contract manifest");
static_assert(sizeof(((RhCudaUnpenalizedConfig*)0)->tolerance) == 8, "RhCudaUnpenalizedConfig.tolerance width drifted from the contract manifest");
static_assert(offsetof(RhCudaUnpenalizedConfig, ridge) == 48, "RhCudaUnpenalizedConfig.ridge offset drifted from the contract manifest");
static_assert(sizeof(((RhCudaUnpenalizedConfig*)0)->ridge) == 8, "RhCudaUnpenalizedConfig.ridge width drifted from the contract manifest");

static_assert(sizeof(RhCudaHostBatch) == 56, "RhCudaHostBatch size drifted from the contract manifest");
static_assert(offsetof(RhCudaHostBatch, abi_version) == 0, "RhCudaHostBatch.abi_version offset drifted from the contract manifest");
static_assert(sizeof(((RhCudaHostBatch*)0)->abi_version) == 4, "RhCudaHostBatch.abi_version width drifted from the contract manifest");
static_assert(offsetof(RhCudaHostBatch, struct_size) == 4, "RhCudaHostBatch.struct_size offset drifted from the contract manifest");
static_assert(sizeof(((RhCudaHostBatch*)0)->struct_size) == 4, "RhCudaHostBatch.struct_size width drifted from the contract manifest");
static_assert(offsetof(RhCudaHostBatch, x_design) == 8, "RhCudaHostBatch.x_design offset drifted from the contract manifest");
static_assert(sizeof(((RhCudaHostBatch*)0)->x_design) == 8, "RhCudaHostBatch.x_design width drifted from the contract manifest");
static_assert(offsetof(RhCudaHostBatch, y) == 16, "RhCudaHostBatch.y offset drifted from the contract manifest");
static_assert(sizeof(((RhCudaHostBatch*)0)->y) == 8, "RhCudaHostBatch.y width drifted from the contract manifest");
static_assert(offsetof(RhCudaHostBatch, sample_weight) == 24, "RhCudaHostBatch.sample_weight offset drifted from the contract manifest");
static_assert(sizeof(((RhCudaHostBatch*)0)->sample_weight) == 8, "RhCudaHostBatch.sample_weight width drifted from the contract manifest");
static_assert(offsetof(RhCudaHostBatch, n_rows) == 32, "RhCudaHostBatch.n_rows offset drifted from the contract manifest");
static_assert(sizeof(((RhCudaHostBatch*)0)->n_rows) == 8, "RhCudaHostBatch.n_rows width drifted from the contract manifest");
static_assert(offsetof(RhCudaHostBatch, n_columns) == 40, "RhCudaHostBatch.n_columns offset drifted from the contract manifest");
static_assert(sizeof(((RhCudaHostBatch*)0)->n_columns) == 8, "RhCudaHostBatch.n_columns width drifted from the contract manifest");
static_assert(offsetof(RhCudaHostBatch, batch_weight) == 48, "RhCudaHostBatch.batch_weight offset drifted from the contract manifest");
static_assert(sizeof(((RhCudaHostBatch*)0)->batch_weight) == 8, "RhCudaHostBatch.batch_weight width drifted from the contract manifest");

static_assert(sizeof(RhCudaDeviceBatch) == 56, "RhCudaDeviceBatch size drifted from the contract manifest");
static_assert(offsetof(RhCudaDeviceBatch, abi_version) == 0, "RhCudaDeviceBatch.abi_version offset drifted from the contract manifest");
static_assert(sizeof(((RhCudaDeviceBatch*)0)->abi_version) == 4, "RhCudaDeviceBatch.abi_version width drifted from the contract manifest");
static_assert(offsetof(RhCudaDeviceBatch, struct_size) == 4, "RhCudaDeviceBatch.struct_size offset drifted from the contract manifest");
static_assert(sizeof(((RhCudaDeviceBatch*)0)->struct_size) == 4, "RhCudaDeviceBatch.struct_size width drifted from the contract manifest");
static_assert(offsetof(RhCudaDeviceBatch, x_design) == 8, "RhCudaDeviceBatch.x_design offset drifted from the contract manifest");
static_assert(sizeof(((RhCudaDeviceBatch*)0)->x_design) == 8, "RhCudaDeviceBatch.x_design width drifted from the contract manifest");
static_assert(offsetof(RhCudaDeviceBatch, y) == 16, "RhCudaDeviceBatch.y offset drifted from the contract manifest");
static_assert(sizeof(((RhCudaDeviceBatch*)0)->y) == 8, "RhCudaDeviceBatch.y width drifted from the contract manifest");
static_assert(offsetof(RhCudaDeviceBatch, sample_weight) == 24, "RhCudaDeviceBatch.sample_weight offset drifted from the contract manifest");
static_assert(sizeof(((RhCudaDeviceBatch*)0)->sample_weight) == 8, "RhCudaDeviceBatch.sample_weight width drifted from the contract manifest");
static_assert(offsetof(RhCudaDeviceBatch, n_rows) == 32, "RhCudaDeviceBatch.n_rows offset drifted from the contract manifest");
static_assert(sizeof(((RhCudaDeviceBatch*)0)->n_rows) == 8, "RhCudaDeviceBatch.n_rows width drifted from the contract manifest");
static_assert(offsetof(RhCudaDeviceBatch, n_columns) == 40, "RhCudaDeviceBatch.n_columns offset drifted from the contract manifest");
static_assert(sizeof(((RhCudaDeviceBatch*)0)->n_columns) == 8, "RhCudaDeviceBatch.n_columns width drifted from the contract manifest");
static_assert(offsetof(RhCudaDeviceBatch, batch_weight) == 48, "RhCudaDeviceBatch.batch_weight offset drifted from the contract manifest");
static_assert(sizeof(((RhCudaDeviceBatch*)0)->batch_weight) == 8, "RhCudaDeviceBatch.batch_weight width drifted from the contract manifest");

static_assert(sizeof(RhCudaHostPrediction) == 40, "RhCudaHostPrediction size drifted from the contract manifest");
static_assert(offsetof(RhCudaHostPrediction, abi_version) == 0, "RhCudaHostPrediction.abi_version offset drifted from the contract manifest");
static_assert(sizeof(((RhCudaHostPrediction*)0)->abi_version) == 4, "RhCudaHostPrediction.abi_version width drifted from the contract manifest");
static_assert(offsetof(RhCudaHostPrediction, struct_size) == 4, "RhCudaHostPrediction.struct_size offset drifted from the contract manifest");
static_assert(sizeof(((RhCudaHostPrediction*)0)->struct_size) == 4, "RhCudaHostPrediction.struct_size width drifted from the contract manifest");
static_assert(offsetof(RhCudaHostPrediction, x_design) == 8, "RhCudaHostPrediction.x_design offset drifted from the contract manifest");
static_assert(sizeof(((RhCudaHostPrediction*)0)->x_design) == 8, "RhCudaHostPrediction.x_design width drifted from the contract manifest");
static_assert(offsetof(RhCudaHostPrediction, prediction) == 16, "RhCudaHostPrediction.prediction offset drifted from the contract manifest");
static_assert(sizeof(((RhCudaHostPrediction*)0)->prediction) == 8, "RhCudaHostPrediction.prediction width drifted from the contract manifest");
static_assert(offsetof(RhCudaHostPrediction, n_rows) == 24, "RhCudaHostPrediction.n_rows offset drifted from the contract manifest");
static_assert(sizeof(((RhCudaHostPrediction*)0)->n_rows) == 8, "RhCudaHostPrediction.n_rows width drifted from the contract manifest");
static_assert(offsetof(RhCudaHostPrediction, n_columns) == 32, "RhCudaHostPrediction.n_columns offset drifted from the contract manifest");
static_assert(sizeof(((RhCudaHostPrediction*)0)->n_columns) == 8, "RhCudaHostPrediction.n_columns width drifted from the contract manifest");

static_assert(sizeof(RhCudaDiagnostics) == 48, "RhCudaDiagnostics size drifted from the contract manifest");
static_assert(offsetof(RhCudaDiagnostics, abi_version) == 0, "RhCudaDiagnostics.abi_version offset drifted from the contract manifest");
static_assert(sizeof(((RhCudaDiagnostics*)0)->abi_version) == 4, "RhCudaDiagnostics.abi_version width drifted from the contract manifest");
static_assert(offsetof(RhCudaDiagnostics, struct_size) == 4, "RhCudaDiagnostics.struct_size offset drifted from the contract manifest");
static_assert(sizeof(((RhCudaDiagnostics*)0)->struct_size) == 4, "RhCudaDiagnostics.struct_size width drifted from the contract manifest");
static_assert(offsetof(RhCudaDiagnostics, iterations) == 8, "RhCudaDiagnostics.iterations offset drifted from the contract manifest");
static_assert(sizeof(((RhCudaDiagnostics*)0)->iterations) == 8, "RhCudaDiagnostics.iterations width drifted from the contract manifest");
static_assert(offsetof(RhCudaDiagnostics, converged) == 16, "RhCudaDiagnostics.converged offset drifted from the contract manifest");
static_assert(sizeof(((RhCudaDiagnostics*)0)->converged) == 4, "RhCudaDiagnostics.converged width drifted from the contract manifest");
static_assert(offsetof(RhCudaDiagnostics, used_regularized_fallback) == 20, "RhCudaDiagnostics.used_regularized_fallback offset drifted from the contract manifest");
static_assert(sizeof(((RhCudaDiagnostics*)0)->used_regularized_fallback) == 4, "RhCudaDiagnostics.used_regularized_fallback width drifted from the contract manifest");
static_assert(offsetof(RhCudaDiagnostics, objective) == 24, "RhCudaDiagnostics.objective offset drifted from the contract manifest");
static_assert(sizeof(((RhCudaDiagnostics*)0)->objective) == 8, "RhCudaDiagnostics.objective width drifted from the contract manifest");
static_assert(offsetof(RhCudaDiagnostics, lambda_value) == 32, "RhCudaDiagnostics.lambda_value offset drifted from the contract manifest");
static_assert(sizeof(((RhCudaDiagnostics*)0)->lambda_value) == 8, "RhCudaDiagnostics.lambda_value width drifted from the contract manifest");
static_assert(offsetof(RhCudaDiagnostics, bandwidth) == 40, "RhCudaDiagnostics.bandwidth offset drifted from the contract manifest");
static_assert(sizeof(((RhCudaDiagnostics*)0)->bandwidth) == 8, "RhCudaDiagnostics.bandwidth width drifted from the contract manifest");

static_assert(sizeof(RhCudaRuntimeInfo) == 24, "RhCudaRuntimeInfo size drifted from the contract manifest");
static_assert(offsetof(RhCudaRuntimeInfo, abi_version) == 0, "RhCudaRuntimeInfo.abi_version offset drifted from the contract manifest");
static_assert(sizeof(((RhCudaRuntimeInfo*)0)->abi_version) == 4, "RhCudaRuntimeInfo.abi_version width drifted from the contract manifest");
static_assert(offsetof(RhCudaRuntimeInfo, struct_size) == 4, "RhCudaRuntimeInfo.struct_size offset drifted from the contract manifest");
static_assert(sizeof(((RhCudaRuntimeInfo*)0)->struct_size) == 4, "RhCudaRuntimeInfo.struct_size width drifted from the contract manifest");
static_assert(offsetof(RhCudaRuntimeInfo, runtime_version) == 8, "RhCudaRuntimeInfo.runtime_version offset drifted from the contract manifest");
static_assert(sizeof(((RhCudaRuntimeInfo*)0)->runtime_version) == 4, "RhCudaRuntimeInfo.runtime_version width drifted from the contract manifest");
static_assert(offsetof(RhCudaRuntimeInfo, driver_version) == 12, "RhCudaRuntimeInfo.driver_version offset drifted from the contract manifest");
static_assert(sizeof(((RhCudaRuntimeInfo*)0)->driver_version) == 4, "RhCudaRuntimeInfo.driver_version width drifted from the contract manifest");
static_assert(offsetof(RhCudaRuntimeInfo, device_count) == 16, "RhCudaRuntimeInfo.device_count offset drifted from the contract manifest");
static_assert(sizeof(((RhCudaRuntimeInfo*)0)->device_count) == 4, "RhCudaRuntimeInfo.device_count width drifted from the contract manifest");
static_assert(offsetof(RhCudaRuntimeInfo, reserved0) == 20, "RhCudaRuntimeInfo.reserved0 offset drifted from the contract manifest");
static_assert(sizeof(((RhCudaRuntimeInfo*)0)->reserved0) == 4, "RhCudaRuntimeInfo.reserved0 width drifted from the contract manifest");

static_assert(sizeof(RhCudaEngineFeatures) == 48, "RhCudaEngineFeatures size drifted from the contract manifest");
static_assert(offsetof(RhCudaEngineFeatures, abi_version) == 0, "RhCudaEngineFeatures.abi_version offset drifted from the contract manifest");
static_assert(sizeof(((RhCudaEngineFeatures*)0)->abi_version) == 4, "RhCudaEngineFeatures.abi_version width drifted from the contract manifest");
static_assert(offsetof(RhCudaEngineFeatures, struct_size) == 4, "RhCudaEngineFeatures.struct_size offset drifted from the contract manifest");
static_assert(sizeof(((RhCudaEngineFeatures*)0)->struct_size) == 4, "RhCudaEngineFeatures.struct_size width drifted from the contract manifest");
static_assert(offsetof(RhCudaEngineFeatures, requested_flags) == 8, "RhCudaEngineFeatures.requested_flags offset drifted from the contract manifest");
static_assert(sizeof(((RhCudaEngineFeatures*)0)->requested_flags) == 8, "RhCudaEngineFeatures.requested_flags width drifted from the contract manifest");
static_assert(offsetof(RhCudaEngineFeatures, enabled_flags) == 16, "RhCudaEngineFeatures.enabled_flags offset drifted from the contract manifest");
static_assert(sizeof(((RhCudaEngineFeatures*)0)->enabled_flags) == 8, "RhCudaEngineFeatures.enabled_flags width drifted from the contract manifest");
static_assert(offsetof(RhCudaEngineFeatures, graph_captures) == 24, "RhCudaEngineFeatures.graph_captures offset drifted from the contract manifest");
static_assert(sizeof(((RhCudaEngineFeatures*)0)->graph_captures) == 8, "RhCudaEngineFeatures.graph_captures width drifted from the contract manifest");
static_assert(offsetof(RhCudaEngineFeatures, graph_replays) == 32, "RhCudaEngineFeatures.graph_replays offset drifted from the contract manifest");
static_assert(sizeof(((RhCudaEngineFeatures*)0)->graph_replays) == 8, "RhCudaEngineFeatures.graph_replays width drifted from the contract manifest");
static_assert(offsetof(RhCudaEngineFeatures, graph_fallbacks) == 40, "RhCudaEngineFeatures.graph_fallbacks offset drifted from the contract manifest");
static_assert(sizeof(((RhCudaEngineFeatures*)0)->graph_fallbacks) == 8, "RhCudaEngineFeatures.graph_fallbacks width drifted from the contract manifest");
