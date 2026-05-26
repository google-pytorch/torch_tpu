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

#include "torch_tpu/eager/op_dispatcher.h"

#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "ATen/ScalarOps.h"
#include "ATen/core/ATen_fwd.h"
#include "absl/base/no_destructor.h"
#include "absl/container/flat_hash_map.h"
#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "c10/util/Optional.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "torch/headeronly/core/DeviceType.h"
#include "torch/headeronly/core/ScalarType.h"
#include "torch_tpu/common/aten_utils.h"
#include "torch_tpu/common/cache_key.h"
#include "torch_tpu/common/context_states.h"
#include "torch_tpu/common/dimension_types.h"
#include "torch_tpu/common/dtype.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/common/shape.h"
#include "torch_tpu/common/to_string.h"
#include "torch_tpu/eager/device_buffer.h"
#include "torch_tpu/eager/eager_mode.h"
#include "torch_tpu/eager/tensor_to_buffer.h"
#include "torch_tpu/ops/copy_from/cpu_to_tpu.h"
#include "torch_tpu/ops/macros/kernel.h"
#include "torch_tpu/ops/op_builder_utils.h"
#include "torch_tpu/ops/op_names.h"
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
  if (GetEagerMode() != EagerMode::kInternalDeferAll) {
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

std::string FormatParamCacheKey(const std::optional<PromotedScalar>& value) {
  return value.has_value() ? "s" : "";
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
