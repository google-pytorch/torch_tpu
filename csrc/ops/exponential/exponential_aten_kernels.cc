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

#include "csrc/ops/exponential/exponential_aten_kernels.h"

#include <cstdint>
#include <limits>
#include <optional>
#include <utility>

#include "ATen/core/ATen_fwd.h"
#include "ATen/core/Generator.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "csrc/common/aten_utils.h"
#include "csrc/common/cache_key.h"
#include "csrc/common/dimension_types.h"
#include "csrc/common/dtype.h"
#include "csrc/common/error_utils.h"
#include "csrc/common/fixed_size_span.h"
#include "csrc/common/to_string.h"
#include "csrc/common/utils.h"
#include "csrc/eager/op_dispatcher.h"
#include "csrc/ops/macros/kernel.h"
#include "csrc/ops/op_builder_utils.h"
#include "csrc/ops/op_names.h"
#include "csrc/ops/rng_utils.h"
#include "csrc/ops/uniform/uniform.h"
#include "llvm/ADT/ArrayRef.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"
#include "torch/headeronly/core/ScalarType.h"
#include "torch/headeronly/util/BFloat16.h"
#include "torch/headeronly/util/Half.h"

namespace torch_tpu {
namespace {

absl::StatusOr<MlirOpResults<1>> BuildExponentialShlo(
    mlir::MlirOp rng_input_state, mlir::MlirOp lambd_op,
    const llvm::ArrayRef<int64_t> sizes, const mlir::ElementType output_dtype) {
  TT_ASSIGN_OR_RETURN(const mlir::ElementType compute_dtype,
                      InferComputationDtype(output_dtype));

  // Generate uniform distribution in [0, 1) using compute_dtype (F32 for
  // F16/BF16/F32, F64 for F64).
  TT_ASSIGN_OR_RETURN(
      auto uniform_results,
      BuildUniformShlo(rng_input_state, 0.0, 1.0, sizes, compute_dtype));

  mlir::MlirOp u = uniform_results;
  TT_ASSIGN_OR_RETURN(lambd_op, CastIfNeeded(lambd_op, compute_dtype));

  // Broadcast scalar lambda to match tensor shape.
  TT_ASSIGN_OR_RETURN((auto [u_bcast, lambd_bcast]),
                      ApplyBroadcastIfNeeded(u, lambd_op));

  // exponential = -ln(1 - U) / lambda
  // U is in [0, 1), so 1 - U is in (0, 1].
  // When U == 0, 1 - U == 1 and ln(1 - U) == 0, which would yield -0.0 and
  // violate exponential_'s strictly positive (0, inf) support.
  // Cap ln(1 - U) at -epsilon / 2 before negation and division by lambda.
  const double eps =
      (compute_dtype == mlir::ElementType::F64)
          ? std::numeric_limits<double>::epsilon()
          : static_cast<double>(std::numeric_limits<float>::epsilon());
  mlir::MlirOp one_op = MakeConstantLike(u_bcast, 1.0);
  mlir::MlirOp max_log_op = MakeConstantLike(u_bcast, -eps / 2.0);

  auto one_minus_u = mlir::stablehlo::Subtract(one_op, u_bcast);
  auto log_one_minus_u = mlir::stablehlo::Log(one_minus_u);
  log_one_minus_u = mlir::stablehlo::Min(log_one_minus_u, max_log_op);
  auto neg_log = mlir::stablehlo::Neg(log_one_minus_u);
  auto result = mlir::stablehlo::Div(neg_log, lambd_bcast);

  // For reduced-precision outputs (F16/BF16), prevent strictly positive
  // compute_dtype samples (e.g. 2^-24 / lambd when lambd >= 2) from rounding
  // down to 0.0 upon conversion to output_dtype, while preserving 0.0 when
  // lambd == +inf. Half uses denorm_min (2^-24, which is normal in f32),
  // whereas BFloat16 uses min (2^-126, since TPU flushes f32/bf16 subnormals).
  if (output_dtype == mlir::ElementType::F16 ||
      output_dtype == mlir::ElementType::BF16) {
    const double min_pos =
        (output_dtype == mlir::ElementType::F16)
            ? static_cast<double>(std::numeric_limits<c10::Half>::denorm_min())
            : static_cast<double>(std::numeric_limits<c10::BFloat16>::min());
    mlir::MlirOp zero_op = MakeConstantLike(result, 0.0);
    mlir::MlirOp min_pos_op = MakeConstantLike(result, min_pos);
    mlir::MlirOp is_pos = mlir::stablehlo::Compare(
        result, zero_op, mlir::stablehlo::ComparisonDirection::GT);
    mlir::MlirOp clamped = mlir::stablehlo::Max(result, min_pos_op);
    result = mlir::stablehlo::Select(is_pos, clamped, result);
  }

  return CastIfNeeded(result, output_dtype);
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
            << ToString(self.scalar_type());

        TT_ASSIGN_OR_THROW(mlir::ElementType output_dtype,
                           ConvertTo<mlir::ElementType>(self.scalar_type()));
        TT_ASSIGN_OR_THROW(const mlir::ElementType compute_dtype,
                           InferComputationDtype(output_dtype));
        const at::ScalarType compute_scalar_type =
            ConvertTo<at::ScalarType>(compute_dtype);
        auto dims = CopyIntVector(self.sizes());

        TT_ASSIGN_OR_THROW(at::Tensor lambd_tensor,
                           promoted_lambd.GetTensor(compute_scalar_type));

        TT_THROW_IF_ERROR(DispatchRngOpOut(
            self, generator, output_dtype,
            [&]  // REF_CAPTURE_OK=called synchronously
            (at::Tensor rng_input_state) -> absl::Status {
              return DispatchOpOut<2, 1>(
                  GetExponentialFunctional(dims, output_dtype),
                  {rng_input_state, lambd_tensor}, self,
                  {.out_dtype = output_dtype,
                   .out_dims = self.sizes(),
                   .op_param_cache_keys = std::move(param_keys)});
            },
            RngUsage{.num_elements = self.numel(),
                     .bit_width = TorchEquivalentBitwidth(compute_dtype)}));
        return self;
      });
}

}  // namespace torch_tpu
