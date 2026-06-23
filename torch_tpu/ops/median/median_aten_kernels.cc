/*
 * Copyright 2026 Google LLC
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

#include "torch_tpu/ops/median/median_aten_kernels.h"

#include <cstdint>
#include <utility>

#include "ATen/core/TensorBody.h"
#include "absl/status/statusor.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"
#include "torch_tpu/common/aten_utils.h"
#include "torch_tpu/common/dimension_types.h"
#include "torch_tpu/common/dtype.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/common/to_string.h"
#include "torch_tpu/eager/device_buffer.h"
#include "torch_tpu/eager/op_dispatcher.h"
#include "torch_tpu/eager/tensor_to_buffer.h"
#include "torch_tpu/ops/macros/kernel.h"
#include "torch_tpu/ops/op_builder_utils.h"
#include "torch_tpu/ops/op_names.h"
#include "torch_tpu/ops/sort/sort.h"

namespace torch_tpu {
namespace {

namespace stablehlo = mlir::stablehlo;

llvm::SmallVector<mlir::MlirOp, 4> CreateStartIndices(
    mlir::MlirBuilder& builder, const mlir::MlirOp index, const int64_t rank,
    const int64_t dim) {
  const mlir::IntegerType i64 = builder.getOpBuilder().getI64Type();
  llvm::SmallVector<mlir::MlirOp, 4> indices(
      rank, MakeScalarConstant(builder, 0, i64));
  indices[dim] = index;
  return indices;
}

struct MedianShloOutputs {
  mlir::MlirOp values;
  mlir::MlirOp indices;
};

MedianShloOutputs BuildMedianShlo(mlir::MlirOp input, int64_t dim) {
  const mlir::RankedTensorType input_type = GetTensorTypeOrDie(input);
  mlir::MlirBuilder& builder = input.getBuilder();
  const mlir::IntegerType i64 = builder.getOpBuilder().getI64Type();

  // 1. Sort the input along dim.
  SortShloOutputs sorted =
      BuildSortShlo(input, /*stable=*/true, dim, /*descending=*/false);

  // 2. Compute the median index: (n - 1) / 2
  mlir::MlirOp median_idx_op;
  if (!input_type.isDynamicDim(dim)) {
    int64_t n = input_type.getShape()[dim];
    int64_t idx = (n - 1) / 2;
    median_idx_op = MakeScalarConstant(builder, idx, i64);
  } else {
    mlir::MlirOp n_op = stablehlo::GetDimensionSize(input, dim);
    if (!n_op.getType().isInteger(64)) {
      n_op = stablehlo::ConvertElementType(n_op, i64);
    }
    mlir::MlirOp one = MakeScalarConstant(builder, 1, i64);
    mlir::MlirOp two = MakeScalarConstant(builder, 2, i64);
    mlir::MlirOp n_minus_one = stablehlo::Subtract(n_op, one);
    median_idx_op = stablehlo::Div(n_minus_one, two);
  }

  // 3. Slice the sorted values and indices at median_idx.
  const int64_t rank = input_type.getRank();
  const llvm::SmallVector<mlir::MlirOp, 4> start_indices =
      CreateStartIndices(builder, median_idx_op, rank, dim);

  llvm::SmallVector<int64_t> slice_shape =
      llvm::to_vector<4>(input_type.getShape());
  slice_shape[dim] = 1;

  mlir::MlirOp sliced_values =
      stablehlo::DynamicSlice(sorted.values, start_indices, slice_shape);
  mlir::MlirOp sliced_indices =
      stablehlo::DynamicSlice(sorted.indices, start_indices, slice_shape);

  for (int64_t i = 0; i < rank; ++i) {
    if (i != dim && input_type.isDynamicDim(i)) {
      mlir::MlirOp dim_size = stablehlo::GetDimensionSize(input, i);
      sliced_values = stablehlo::SetDimensionSize(sliced_values, dim_size, i);
      sliced_indices = stablehlo::SetDimensionSize(sliced_indices, dim_size, i);
    }
  }

  return MedianShloOutputs{.values = sliced_values, .indices = sliced_indices};
}

}  // namespace

at::Tensor AtenMedian(const at::Tensor& self) {
  if (self.numel() == 0) {
    return at::full({}, std::numeric_limits<double>::quiet_NaN(),
                    self.options());
  }
  TT_KERNEL(OpName::kMedian, param_keys, (self), {
    TT_CHECK_THROW(!IsComplex(self), error::kInvalidArgument)
        << "expected the input dtype not to be complex, got "
        << ToString(self.scalar_type());

    if (self.dim() == 0) {
      return self.clone();
    }

    at::Tensor input = self;
    Dimensions output_dims = {};  // Scalar output.

    TT_ASSIGN_OR_THROW(mlir::ElementType output_dtype,
                       ConvertTo<mlir::ElementType>(self.scalar_type()));

    auto op_builder =
        [](mlir::MlirOp input) -> absl::StatusOr<MlirOpResults<1>> {
      mlir::MlirOp flattened = Flatten(input);
      auto median_outputs = BuildMedianShlo(flattened, 0);
      TT_ASSIGN_OR_RETURN(mlir::MlirOp squeezed,
                          Squeeze(median_outputs.values, {0}));
      return {squeezed};
    };

    TT_ASSIGN_OR_THROW(
        auto result_buf,
        DispatchOp<1>(std::move(op_builder), input,
                      {.out_dtype = output_dtype,
                       .out_dims = std::move(output_dims),
                       .op_param_cache_keys = std::move(param_keys)}));

    TT_ASSIGN_OR_THROW(at::Tensor out,
                       MakeEmptyTensor({}, self.scalar_type(), self.device()));
    TT_THROW_IF_ERROR(AssignBufferToAtTensor(std::move(result_buf), out));
    return out;
  });
}

}  // namespace torch_tpu
