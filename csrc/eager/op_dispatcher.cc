/*
 * Copyright 2025 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "csrc/eager/op_dispatcher.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "ATen/ScalarOps.h"
#include "ATen/core/ATen_fwd.h"
#include "absl/base/no_destructor.h"
#include "absl/container/flat_hash_map.h"
#include "absl/log/absl_check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "c10/util/Optional.h"
#include "csrc/common/aten_utils.h"
#include "csrc/common/cache_key.h"
#include "csrc/common/context_states.h"
#include "csrc/common/device_type.h"
#include "csrc/common/dimension_types.h"
#include "csrc/common/dtype.h"
#include "csrc/common/error_utils.h"
#include "csrc/common/shape.h"
#include "csrc/common/to_string.h"
#include "csrc/eager/device_buffer.h"
#include "csrc/eager/eager_mode.h"
#include "csrc/eager/tensor_to_buffer.h"
#include "csrc/ops/copy_from/cpu_to_tpu.h"
#include "csrc/ops/macros/kernel.h"
#include "csrc/ops/op_builder_utils.h"
#include "csrc/ops/op_names.h"
#include "csrc/ops/resize/resize_aten_kernels.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "torch/headeronly/core/DeviceType.h"
#include "torch/headeronly/core/ScalarType.h"
#include "tsl/profiler/lib/traceme.h"

namespace torch_tpu {

namespace {

// Internal helper for MakeTensor that returns the
// DeviceBufferRef that will get wrapped into the at::Tensor.
absl::StatusOr<DeviceBufferRef> MakeBuffer(
    const at::Scalar& scalar,
    c10::optional<at::ScalarType> scalar_type_opt = std::nullopt) {
  at::ScalarType scalar_type = scalar_type_opt.value_or(scalar.type());
  if (scalar_type == at::ScalarType::ComplexDouble) {
    // TPU does not support complex128. Use complex64 instead.
    scalar_type = at::ScalarType::ComplexFloat;
  }

  // In DeferAll or compiled mode, scalars are assumed to be constants for
  // optimal performance at the cost of increased specialization.
  // In other eager modes (DeferNever or DeferAndFuse) scalars are treated as
  // arguments to increase code reuse.
  const bool constant_mode =
      GetEagerMode() == EagerMode::kInternalCompileFxGraph ||
      GetEagerMode() == EagerMode::kInternalDeferAll;

  if (!constant_mode) {
    // Variable execution mode: materialize the scalar to a DeviceBufferRef.
    // This treats the scalar as an argument rather than a constant, which
    // decreases compiler specialization and improves code reuse.
    auto hashable_scalar = HashableScalar{
        .scalar = scalar,
        .scalar_type = scalar_type,
    };

    // See if we've already created a DeviceBufferRef for this scalar; if we
    // have, don't recreate it.
    //
    // We make this thread_local to avoid synchronization overhead. This means
    // that we may miss some opportunities to reuse scalars across threads,
    // but it might be a net win overall given that scalar buffers are cheap
    // to create.
    static thread_local  // CPP_THREAD_LOCAL_OK=unrelated to Python threads.
        absl::flat_hash_map<HashableScalar, DeviceBufferRef>
            scalar_map;
    if (auto it = scalar_map.find(hashable_scalar); it != scalar_map.end()) {
      return it->second;
    }

    // New scalar, create a DeviceBufferRef for it.
    // Create a TraceMe so that the host-to-device copy is visible in xprof.
    tsl::profiler::TraceMe t(
        [scalar] { return absl::StrCat("ScalarBuffer_", ToString(scalar)); });

    // Upgrade the scalar to a tensor on CPU, using CPU-optimized code paths.
    at::Tensor scalar_tensor =
        c10::scalar_to_tensor(scalar, /*device=*/at::kCPU);
    // Then cast the tensor to the appropriate type on CPU, if necessary.
    if (scalar_tensor.scalar_type() != scalar_type) {
      scalar_tensor = scalar_tensor.to(scalar_type);
    }

    // Then copy it to device as a materialized DeviceBufferRef.
    TT_ASSIGN_OR_RETURN(
        DeviceBufferRef buf_ref,
        CopyCpuToTpuBuffer(scalar_tensor, /*non_blocking=*/true));
    scalar_map.insert({hashable_scalar, buf_ref});
    return buf_ref;
  }

  // Constant execution mode: create an mlir::Constant op to represent the
  // value. This ensures that the compiler is able to specialize for this value
  // and produce an optimized executable, provided this value is consistent
  // on future executions.
  TT_ASSIGN_OR_RETURN(auto scalar_element_type,
                      ConvertTo<mlir::ElementType>(scalar_type));
  Shape shape(Dimensions{}, scalar_element_type);

  TT_ASSIGN_OR_RETURN(auto op_param_cache_keys,
                      TT_MAKE_OP_PARAM_CACHE_KEYS(scalar));
  auto op_builder =
      [scalar, scalar_element_type](
          mlir::MlirBuilder& builder) -> absl::StatusOr<mlir::MlirOp> {
    return MakeConstant(builder, scalar, scalar_element_type);
  };

  return DispatchOp<0>(std::move(op_builder),
                       /*inputs=*/{},
                       // Override the op name as this is a general utility
                       // rather than a specific op.
                       {.op_name = OpName::kScalarTensor,
                        .out_dtype = scalar_element_type,
                        .out_dims = Dimensions{},
                        .op_param_cache_keys = std::move(op_param_cache_keys)});
}

bool IsEligibleDonor(const at::Tensor& in, const at::Tensor& out,
                     mlir::ElementType out_dtype,
                     at::IntArrayRef destination_dims) {
  if (!out.is_alias_of(in) || !in.is_contiguous() || in.storage_offset() != 0 ||
      in.is_conj() || in.numel() == 0 || in.sizes() != destination_dims) {
    return false;
  }
  const auto in_dtype = ConvertTo<mlir::ElementType>(in.scalar_type());
  if (!in_dtype.ok() || *in_dtype != out_dtype) {
    return false;
  }
  // When running on TPU tensors, verify that the underlying DeviceBufferRef has
  // exclusive ownership (`live_data_ptrs == 1`). In TorchTPU, `CopyTpuToTpu`
  // performs shallow buffer sharing for `.clone()` when shapes and dtypes
  // match, creating separate `c10::Storage` instances that share the same
  // device buffer. Because `is_alias_of` only inspects storage pointers, it
  // returns false between a tensor and its clone, but each tensor's storage
  // aliases itself. Checking `live_data_ptrs() > 1` ensures in-place operations
  // on a clone do not donate and invalidate the buffer shared by the original.
  //
  // Note: We explicitly guard this check with `GetPrivateUse1DeviceType()`
  // because standalone C++ unit tests (e.g. in op_dispatcher_test) test ATen
  // layout, contiguity, and offset properties using CPU tensors. In standalone
  // C++ test binaries without Python runtime extensions, ATen view kernels like
  // `aten::as_strided` are not registered for `PrivateUse1`. Guarding with
  // `GetPrivateUse1DeviceType()` allows those C++ unit tests to run on CPU
  // tensors without `GetBaseBuffer` failing, while strictly enforcing exclusive
  // ownership for real TPU tensors in production.
  if (in.device().type() == GetPrivateUse1DeviceType()) {
    const auto in_buf = GetBaseBuffer(in);
    return in_buf.ok() && in_buf->device_buffer_list()->live_data_ptrs() <= 1;
  }
  return true;
}

void MarkDonatedAndAliases(size_t donated_idx,
                           absl::Span<const at::Tensor> inputs,
                           std::vector<bool>& input_donated) {
  const at::Tensor& donated_in = inputs[donated_idx];
  for (size_t k = 0; k < inputs.size(); ++k) {
    if (inputs[k].is_alias_of(donated_in)) {
      input_donated[k] = true;
    }
  }
}

}  // namespace

namespace internal {

[[nodiscard]] OpDispatchFailure& GetOpDispatchFailure() {
  static absl::NoDestructor<OpDispatchFailure> failure;
  return *failure;
}

void SetOpDispatchFailure(std::string op_base_name,
                          std::string failure_message) {
  GetOpDispatchFailure() = {std::move(op_base_name),
                            std::move(failure_message)};
}

std::string EncodeParamCacheKey(const std::optional<PromotedScalar>& value) {
  return value.has_value() ? "s" : "";
}

// Automatically detects and configures buffer donations from `inputs` into
// multiple `outputs` in eager mode.
//
// Safety and layout invariants verified for each donor-destination pair:
// 1. In-place aliasing: destination must alias donor (is_alias_of).
// 2. Feature flag & execution mode: donation must be enabled via
//    IsInplaceBufferDonationEnabled() and restricted to synchronous eager modes
//    (kDeferNever, kDeferNeverAndLaunchBlocking). In deferred execution
//    (kDeferAndFuse), automated donation is currently not done.
// 3. Dense layout: donor must be contiguous, zero-offset, unconjugated,
//    non-empty, and identical in shape to the destination dimensions.
// 4. Data type: donor element type must match the destination element type.
// 5. Single donation constraint: In XLA / StableHLO, each device buffer can be
//    donated to at most ONE output buffer. Once an input is donated, it and all
//    its storage aliases (is_alias_of) are marked as donated to prevent any
//    duplicate buffer donation across multiple outputs.
// 6. Unique ownership constraint: donor must not share its DeviceBufferRef with
//    other non-alias tensors (live_data_ptrs == 1), preventing donation from
//    destroying buffers shared via shallow-copy clone semantics.
void AutoDonateInPlaceBuffers(
    absl::Span<const at::Tensor> outputs,
    absl::Span<const mlir::ElementType> out_dtypes,
    absl::Span<const absl::Span<const int64_t>> out_dims_list,
    absl::Span<const at::Tensor> inputs, Indices& donated_indices) {
  if (!donated_indices.empty() || !IsInplaceBufferDonationEnabled() ||
      !IsDeferNeverMode(GetEagerMode())) {
    return;
  }

  std::vector<bool> input_donated(inputs.size(), false);
  for (size_t out_idx = 0; out_idx < outputs.size(); ++out_idx) {
    const at::Tensor& out = outputs[out_idx];
    if (!out.defined()) {
      continue;
    }
    const auto out_dtype = out_dtypes[out_idx];
    const at::IntArrayRef destination_dims(out_dims_list[out_idx].data(),
                                           out_dims_list[out_idx].size());
    for (size_t in_idx = 0; in_idx < inputs.size(); ++in_idx) {
      if (input_donated[in_idx]) {
        continue;
      }
      if (IsEligibleDonor(inputs[in_idx], out, out_dtype, destination_dims)) {
        donated_indices.push_back(static_cast<int64_t>(in_idx));
        MarkDonatedAndAliases(in_idx, inputs, input_donated);
        break;
      }
    }
  }
}

absl::Status AssignBufferToOutput(const at::Tensor& output,
                                  DeviceBufferRef result_buffer) {
  TT_RET_CHECK(  // ERROR_COV_INFEASIBLE=Output tensors are always defined.
      output.defined(), error::kInvalidArgument)
      << "expected defined output tensor";
  TT_RETURN_IF_ERROR(
      ResizeTensorIfShapeDiffers(output, result_buffer.dimensions()));
  return AssignBufferToAtTensor(std::move(result_buffer), output);
}

absl::Status AssignBuffersToOutputs(
    absl::Span<const at::Tensor> outputs,
    absl::Span<DeviceBufferRef> result_buffers) {
  ABSL_CHECK_EQ(outputs.size(), result_buffers.size())  // CRASH_OK
      << "Mismatch between number of output tensors and result buffers.";
  for (size_t i = 0; i < outputs.size(); ++i) {
    TT_RETURN_IF_ERROR(
        AssignBufferToOutput(outputs[i], std::move(result_buffers[i])));
  }
  return absl::OkStatus();
}

}  // namespace internal

absl::StatusOr<at::Tensor> MakeTensor(
    const at::Scalar& scalar, c10::optional<at::ScalarType> scalar_type_opt) {
  TT_ASSIGN_OR_RETURN(  // TODO: Test by forcing "scalar_tensor" to fail.
      DeviceBufferRef buffer, MakeBuffer(scalar, scalar_type_opt));
  return MakeTensor(std::move(buffer));
}

PromotedScalar PromoteScalar(at::Scalar scalar) {
  return PromotedScalar(
      [](const at::Scalar& scalar,
         std::optional<at::ScalarType> scalar_type_opt)
          -> absl::StatusOr<at::Tensor> {
        TT_ASSIGN_OR_RETURN(DeviceBufferRef buffer,
                            MakeBuffer(scalar, scalar_type_opt));
        return MakeTensor(std::move(buffer));
      },
      std::move(scalar));
}

std::optional<PromotedScalar> PromoteScalar(std::optional<at::Scalar> scalar) {
  if (!scalar.has_value()) {
    return std::nullopt;
  }
  return PromoteScalar(scalar.value());
}

std::vector<PromotedScalar> PromoteScalar(at::ArrayRef<at::Scalar> scalars) {
  std::vector<PromotedScalar> promoted_scalars;
  promoted_scalars.reserve(scalars.size());
  for (const auto& scalar : scalars) {
    promoted_scalars.push_back(PromoteScalar(scalar));
  }
  return promoted_scalars;
}

}  // namespace torch_tpu
