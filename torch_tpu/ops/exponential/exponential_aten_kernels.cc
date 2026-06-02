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

#include "torch_tpu/ops/exponential/exponential_aten_kernels.h"

#include <cstdint>
#include <optional>
#include <utility>

#include "ATen/core/ATen_fwd.h"
#include "ATen/core/Generator.h"
#include "absl/status/statusor.h"
#include "llvm/ADT/ArrayRef.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"
#include "torch_tpu/common/dimension_types.h"
#include "torch_tpu/common/dtype.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/common/to_string.h"
#include "torch_tpu/common/utils.h"
#include "torch_tpu/eager/op_dispatcher.h"
#include "torch_tpu/ops/macros/kernel.h"
#include "torch_tpu/ops/op_builder_utils.h"
#include "torch_tpu/ops/op_names.h"
#include "torch_tpu/ops/rng_utils.h"
#include "torch_tpu/ops/uniform/uniform.h"

namespace torch_tpu {
namespace {

absl::StatusOr<MlirOpResults<2>> BuildExponentialShlo(
    mlir::MlirOp rng_input_state, const double lambd,
    const llvm::ArrayRef<int64_t> sizes, const mlir::ElementType mlir_type) {
  // Generate uniform distribution in [0, 1).
  TT_ASSIGN_OR_RETURN(
      auto uniform_results,
      BuildUniformShlo(rng_input_state, 0.0, 1.0, sizes, mlir_type));

  mlir::MlirOp rng_output_state = uniform_results[0];
  mlir::MlirOp u = uniform_results[1];
  auto& builder = u.getBuilder();

  // exponential = -ln(1 - U) / lambda
  // U is in [0, 1). 1 - U is in (0, 1].

  mlir::MlirOp lambd_op = MakeScalarConstant(builder, lambd, mlir_type);
  mlir::MlirOp one_op = MakeScalarConstant(builder, 1.0, mlir_type);

  // Broadcast scalars to match tensor shape
  auto tensor_type = GetTensorTypeOrDie(u);
  lambd_op = mlir::stablehlo::BroadcastInDim(tensor_type, lambd_op, {});
  one_op = mlir::stablehlo::BroadcastInDim(tensor_type, one_op, {});

  auto one_minus_u = mlir::stablehlo::Subtract(one_op, u);
  auto log_one_minus_u = mlir::stablehlo::Log(one_minus_u);
  auto neg_log = mlir::stablehlo::Neg(log_one_minus_u);
  auto result = mlir::stablehlo::Div(neg_log, lambd_op);

  return {{rng_output_state, result}};
}

NAryMlirOpBuilder<1, 2> GetExponentialFunctional(Dimensions dims,
                                                 mlir::ElementType output_dtype,
                                                 double lambd) {
  return [dims, output_dtype, lambd](mlir::MlirOp rng_input_state) {
    return BuildExponentialShlo(rng_input_state, lambd, dims, output_dtype);
  };
}

}  // namespace

at::Tensor& AtenExponential_(at::Tensor& self, double lambd,
                             std::optional<at::Generator> generator) {
  TT_KERNEL(OpName::kExponential_, param_keys, (self, lambd, generator), {
    if (self.numel() == 0) {
      return self;
    }
    TT_CHECK_THROW(self.is_floating_point(), error::kInvalidArgument)
        << "expected input tensor dtype to be a floating-point real type, got "
        << torch_tpu::ToString(self.scalar_type());

    TT_ASSIGN_OR_THROW(mlir::ElementType output_dtype,
                       ConvertTo<mlir::ElementType>(self.scalar_type()));
    auto dims = CopyIntVector(self.sizes());
    TT_THROW_IF_ERROR(
        DispatchRngOp(self, generator, [&](at::Tensor rng_input_state) {
          return DispatchOp<1, 2>(
              GetExponentialFunctional(dims, output_dtype, lambd),
              {rng_input_state},
              {.out_dtypes = {mlir::ElementType::UI64, output_dtype},
               .out_dims_list = {{2}, self.sizes()},
               .op_param_cache_keys = std::move(param_keys)});
        }));
    return self;
  });
}

}  // namespace torch_tpu
