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

#include "csrc/ops/index_copy/index_copy_aten_kernels.h"

#include <cstdint>
#include <utility>

#include "ATen/core/ATen_fwd.h"
#include "ATen/core/TensorBody.h"
#include "absl/log/absl_log.h"
#include "absl/log/log.h"
#include "absl/types/span.h"
#include "c10/core/ScalarType.h"
#include "csrc/common/aten_utils.h"
#include "csrc/common/cache_key.h"
#include "csrc/common/dimension_types.h"
#include "csrc/common/dtype.h"
#include "csrc/common/error_utils.h"
#include "csrc/common/fixed_size_span.h"
#include "csrc/common/utils.h"
#include "csrc/eager/op_dispatcher.h"
#include "csrc/ops/index_utils.h"
#include "csrc/ops/macros/kernel.h"
#include "csrc/ops/op_builder_utils.h"
#include "csrc/ops/op_names.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Types.h"
#include "mlir/Support/DebugStringHelper.h"
#include "mlir/Support/LLVM.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"
#include "torch/headeronly/core/ScalarType.h"

namespace torch_tpu {

namespace {

namespace stablehlo = mlir::stablehlo;

mlir::MlirOp BuildIndexCopyShlo(mlir::MlirOp self, int64_t dim,
                                mlir::MlirOp index, mlir::MlirOp source,
                                mlir::ElementType computation_dtype) {
  const mlir::RankedTensorType self_type = GetTensorTypeOrDie(self);
  const mlir::RankedTensorType source_type = GetTensorTypeOrDie(source);
  const mlir::RankedTensorType index_type = GetTensorTypeOrDie(index);
  ABSL_VLOG(2) << "BuildIndexCopyShlo:"
               << ", dim: " << dim
               << ", self_type: " << mlir::debugString(self_type)
               << ", source_type: " << mlir::debugString(source_type)
               << ", index_type: " << mlir::debugString(index_type);
  if (self_type.getRank() == 0) {
    return source;
  }

  // Convert arguments to the computation type if necessary.
  mlir::Type computation_type =
      mlir::getElementType(self.getContext(), computation_dtype);
  ABSL_VLOG(2) << "computation_type: " << mlir::debugString(computation_type);
  if (self_type.getElementType() != computation_type) {
    self = mlir::stablehlo::ConvertElementType(self, computation_type);
  }
  if (source_type.getElementType() != computation_type) {
    source = mlir::stablehlo::ConvertElementType(source, computation_type);
  }

  const Dimensions all_other_dims =
      GetBatchDimensions(self_type.getRank(), dim);

  index = mlir::stablehlo::Reshape(index, {index_type.getDimSize(0), 1});
  stablehlo::ScatterDimensionNumbersAttr scatter_dimension_numbers =
      stablehlo::ScatterDimensionNumbersAttr::get(
          &self.getContext(),
          /*update_window_dims=*/all_other_dims,
          /*inserted_window_dims=*/{dim},
          /*input_batching_dims=*/{},
          /*scatter_indices_batching_dims=*/{},
          /*scatter_dims_to_operand_dims=*/{dim},
          /*index_vector_dim=*/1);
  ABSL_VLOG(2) << "BuildScatterShlo: ScatterDimensionNumbers = "
               << mlir::debugString(scatter_dimension_numbers);

  // Create a region builder callback.
  auto block_type = self_type.clone({}, computation_type);
  auto region_builder = [block_type](mlir::RegionBuilder& builder) {
    mlir::MlirOp _ = mlir::Argument(builder, block_type);
    mlir::MlirOp arg1 = mlir::Argument(builder, block_type);
    // Region returns the second argument (from source).
    stablehlo::Return(builder, {arg1});
  };

  return stablehlo::Scatter(self, index, source, region_builder,
                            scatter_dimension_numbers)[0];
}

absl::Status IndexCopy(const at::Tensor& self, int64_t dim,
                       const at::Tensor& index, const at::Tensor& source,
                       at::Tensor& out, OpParamCacheKeys param_keys) {
  TT_RETURN_IF_ERROR(ValidateOutDtype(out, self.scalar_type(),
                                      /*allow_out_dtype_cast=*/false));
  TT_ASSIGN_OR_RETURN(dim,
                      ValidateIndexInputsAndGetDim(self, dim, index, source));

  at::ScalarType promoted_scalar_type =
      c10::promoteTypes(self.scalar_type(), source.scalar_type());
  TT_ASSIGN_OR_RETURN(const auto computation_dtype,
                      ConvertTo<mlir::ElementType>(promoted_scalar_type));

  auto index_copy_op_builder =
      [dim, computation_dtype](FixedSizeSpan<mlir::MlirOp, 3> inputs) {
        auto& [self, index, source] = inputs;
        return BuildIndexCopyShlo(self, dim, index, source, computation_dtype);
      };

  TT_ASSIGN_OR_RETURN(const auto output_dtype,
                      ConvertTo<mlir::ElementType>(self.scalar_type()));
  Dimensions output_dims = CopyIntVector(self.sizes());

  return DispatchOpOut<3>(std::move(index_copy_op_builder),
                          {self, index, source}, out,
                          {.out_dtype = output_dtype,
                           .out_dims = output_dims,
                           .op_param_cache_keys = std::move(param_keys)});
}

}  // namespace

at::Tensor& AtenIndexCopyOut(const at::Tensor& self, int64_t dim,
                             const at::Tensor& index, const at::Tensor& source,
                             at::Tensor& out) {
  TT_KERNEL(
      OpName::kIndexCopyOut, param_keys, (self, dim, index, source, out), {
        TT_THROW_IF_ERROR(
            IndexCopy(self, dim, index, source, out, std::move(param_keys)));
        return out;
      });
}

}  // namespace torch_tpu
