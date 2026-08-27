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

#include "torch_tpu/csrc/ops/exponential/exponential_aten_kernels.h"

#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

#include "ATen/core/ATen_fwd.h"
#include "ATen/core/Generator.h"
#include "absl/status/statusor.h"
#include "llvm/ADT/ArrayRef.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"
#include "torch_tpu/csrc/common/cache_key.h"
#include "torch_tpu/csrc/common/dimension_types.h"
#include "torch_tpu/csrc/common/dtype.h"
#include "torch_tpu/csrc/common/error_utils.h"
#include "torch_tpu/csrc/common/fixed_size_span.h"
#include "torch_tpu/csrc/common/to_string.h"
#include "torch_tpu/csrc/common/utils.h"
#include "torch_tpu/csrc/eager/device_buffer.h"
#include "torch_tpu/csrc/eager/op_dispatcher.h"
#include "torch_tpu/csrc/ops/macros/kernel.h"
#include "torch_tpu/csrc/ops/op_builder_utils.h"
#include "torch_tpu/csrc/ops/op_names.h"
#include "torch_tpu/csrc/ops/rng_utils.h"
#include "torch_tpu/csrc/ops/uniform/uniform.h"

namespace torch_tpu {
namespace {

absl::StatusOr<MlirOpResults<1>> BuildExponentialShlo(
    mlir::MlirOp rng_input_state, mlir::MlirOp lambd_op,
    const llvm::ArrayRef<int64_t> sizes, const mlir::ElementType mlir_type) {
  // Generate uniform distribution in [0, 1).
  TT_ASSIGN_OR_RETURN(
      auto uniform_results,
      BuildUniformShlo(rng_input_state, 0.0, 1.0, sizes, mlir_type));

  mlir::MlirOp u = uniform_results;
  auto& builder = u.getBuilder();

  // exponential = -ln(1 - U) / lambda
  // U is in [0, 1). 1 - U is in (0, 1].

  mlir::MlirOp one_op = MakeScalarConstant(builder, 1.0, mlir_type);

  // Broadcast scalars to match tensor shape
  TT_ASSIGN_OR_RETURN((auto [u_bcast, lambd_bcast]),
                      ApplyBroadcastIfNeeded(u, lambd_op));

  auto tensor_type = GetTensorTypeOrDie(u_bcast);
  one_op = mlir::stablehlo::BroadcastInDim(tensor_type, one_op, {});

  auto one_minus_u = mlir::stablehlo::Subtract(one_op, u_bcast);
  auto log_one_minus_u = mlir::stablehlo::Log(one_minus_u);
  auto neg_log = mlir::stablehlo::Neg(log_one_minus_u);
  auto result = mlir::stablehlo::Div(neg_log, lambd_bcast);

  return result;
}

NAryMlirOpBuilder<2, 1> GetExponentialFunctional(
    Dimensions dims, mlir::ElementType output_dtype) {
  return [dims, output_dtype](FixedSizeSpan<mlir::MlirOp, 2> inputs) {
    auto [rng_input_state_op, lambd_op] = inputs;
    return BuildExponentialShlo(rng_input_state_op, lambd_op, dims,
                                output_dtype);
  };
}

}  // namespace

at::Tensor& AtenExponential_(at::Tensor& self, double lambd,
                             std::optional<at::Generator> generator) {
  PromotedScalar promoted_lambd = PromoteScalar(at::Scalar(lambd));
  TT_KERNEL(
      OpName::kExponential_, param_keys, (self, promoted_lambd, generator), {
        if (self.numel() == 0) {
          return self;
        }
        TT_CHECK_THROW(self.is_floating_point(), error::kInvalidArgument)
            << "expected input tensor dtype to be a floating-point real type, "
               "got "
            << torch_tpu::ToString(self.scalar_type());

        TT_ASSIGN_OR_THROW(mlir::ElementType output_dtype,
                           ConvertTo<mlir::ElementType>(self.scalar_type()));
        auto dims = CopyIntVector(self.sizes());

        TT_ASSIGN_OR_THROW(at::Tensor lambd_tensor,
                           promoted_lambd.GetTensor(self.scalar_type()));

        TT_THROW_IF_ERROR(DispatchRngOp(
            self, generator,
            [&](at::Tensor rng_input_state)
                -> absl::StatusOr<std::vector<DeviceBufferRef>> {
              TT_ASSIGN_OR_RETURN(
                  auto buf, (DispatchOp<2, 1>(
                                GetExponentialFunctional(dims, output_dtype),
                                {rng_input_state, lambd_tensor},
                                {.out_dtype = output_dtype,
                                 .out_dims = self.sizes(),
                                 .op_param_cache_keys = std::move(param_keys),
                                 .split_mode = OpSplitMode::kSplitAfter})));
              return std::vector<DeviceBufferRef>{std::move(buf)};
            }));
        return self;
      });
}

}  // namespace torch_tpu
