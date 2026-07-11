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

#include "torch_tpu/ops/grouped_mm/grouped_mm_aten_kernels.h"

#include <algorithm>
#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

#include "ATen/core/TensorBody.h"
#include "absl/log/absl_check.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "c10/util/SmallVector.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Types.h"
#include "stablehlo/dialect/ChloOps.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/ChloBuilder.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"
#include "torch/headeronly/core/ScalarType.h"
#include "torch_tpu/common/cache_key.h"
#include "torch_tpu/common/dimension_types.h"
#include "torch_tpu/common/dtype.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/eager/device_buffer.h"
#include "torch_tpu/eager/op_dispatcher.h"
#include "torch_tpu/eager/tensor_to_buffer.h"
#include "torch_tpu/ops/macros/kernel.h"
#include "torch_tpu/ops/op_builder_utils.h"
#include "torch_tpu/ops/op_names.h"
#include "torch_tpu/ops/precision_context.h"

namespace torch_tpu {
namespace {

// Lowers grouped matmul to MLIR StableHLO/CHLO.
//
// Dimension Notation (mirrors CHLO_RaggedDotOp):
//   b: batch dimension (expert batch size in MoE)
//   m: lhs non-contracting dimension (tokens per expert)
//   k: contracting dimension (input channels / hidden size)
//   n: rhs non-contracting dimension (output channels / projection size)
//   g: group dimension (expert count / number of partitions along ragged dim)
//
// Lowering Cases:
// - Case 1 (2D x 3D with offs):
//   Input:  self [m_total, k], mat2 [g, k, n], offs [g]
//   Output: [m_total, n]
//   Maps to RaggedDot Mode 1 (lhs non-contracting dim 'm' is ragged).
//   The leading dimension 'm_total' is partitioned into g groups [m_i, ...].
//
// - Case 2 (3D x 2D with offs):
//   Input:  self [g, b, k], mat2 [k, Sum(n_i)], offs [g]
//   Output: [b, Sum(n_i)]
//   Each expert i has output channel size n_i.
//   We transpose to map to RaggedDot Mode 1, where mat2 is the lhs and self is
//   the rhs in RaggedDot:
//     mat2 -> [Sum(n_i), k] (partitioned along dim 0 into g groups of [n_i, k])
//     self -> [g, k, b]
//   RaggedDot produces [Sum(n_i), b], which is transposed back to [b, Sum(n_i)]
//
// - Case 3 (2D x 2D with offs):
//   Input:  self [m, Sum(k_i)], mat2 [Sum(k_i), n], offs [g]
//   Output: [g, m, n]
//   Maps to RaggedDot Mode 2 (Contracting dim 'k' is ragged across g groups).
//   Self's trailing dim (dim 1) and mat2's leading dim (dim 0) share the
//   same ragged partitioning along Sum(k_i).
//
// - Case 4 (3D x 3D without offs):
//   Input:  self [b, m, k], mat2 [b, k, n]
//   Output: [b, m, n]
//   Degrades to standard batched matrix multiplication (DotGeneral).
absl::StatusOr<mlir::MlirOp> BuildGroupedMmShlo(
    mlir::MlirOp self_op, mlir::MlirOp mat2_op,
    std::optional<mlir::MlirOp> offs_op, mlir::ElementType out_dtype,
    mlir::stablehlo::Precision precision) {
  mlir::MlirBuilder& builder = self_op.getBuilder();
  const mlir::RankedTensorType self_type = GetTensorTypeOrDie(self_op);
  const mlir::RankedTensorType mat2_type = GetTensorTypeOrDie(mat2_op);

  const bool a_is_2d = self_type.getRank() == 2;
  const bool b_is_2d = mat2_type.getRank() == 2;

  mlir::chlo::Precision chlo_precision =
      static_cast<mlir::chlo::Precision>(precision);
  auto precision_attr =
      mlir::chlo::PrecisionAttr::get(&self_op.getContext(), chlo_precision);
  auto precision_config = mlir::ArrayAttr::get(
      &self_op.getContext(), {precision_attr, precision_attr});

  if (a_is_2d && !b_is_2d) {
    // Case 1 (2D x 3D with offs): lhs first dimension is ragged (Mode 1).
    ABSL_CHECK(offs_op.has_value());  // CRASH_OK
    const mlir::RankedTensorType offs_type = GetTensorTypeOrDie(*offs_op);
    const int64_t num_groups = offs_type.getShape()[0];

    mlir::MlirOp zero_op =
        MakeConstant(builder, 0, offs_type.getElementType(), {1});
    const Strides strides(1, 1);
    const Indices start_indices(1, 0);
    const Indices limit_indices(1, std::max<int64_t>(0, num_groups - 1));
    mlir::MlirOp offs_slice =
        mlir::stablehlo::Slice(*offs_op, start_indices, limit_indices, strides);
    mlir::MlirOp concat_op =
        mlir::stablehlo::Concatenate(builder, {zero_op, offs_slice}, 0);
    mlir::MlirOp group_sizes = mlir::stablehlo::Subtract(*offs_op, concat_op);

    const mlir::chlo::RaggedDotDimensionNumbersAttr dimension_numbers =
        mlir::chlo::RaggedDotDimensionNumbersAttr::get(
            &self_op.getContext(), /*lhsBatchingDimensions=*/{},
            /*rhsBatchingDimensions=*/{},
            /*lhsContractingDimensions=*/{1},
            /*rhsContractingDimensions=*/{1},
            /*lhsRaggedDimensions=*/{0},
            /*rhsGroupDimensions=*/{0});

    const mlir::Type out_type = mlir::makeTensorType(
        self_op.getContext(),
        {self_type.getShape()[0], mat2_type.getShape()[2]}, out_dtype);
    return mlir::chlo::RaggedDot(out_type, self_op, mat2_op, group_sizes,
                                 dimension_numbers, precision_config);
  } else if (!a_is_2d && b_is_2d) {
    // Case 2 (3D x 2D with offs): rhs tailing dimension is ragged (Mode 1).
    ABSL_CHECK(offs_op.has_value());  // CRASH_OK
    const mlir::RankedTensorType offs_type = GetTensorTypeOrDie(*offs_op);
    const int64_t num_groups = offs_type.getShape()[0];

    mlir::MlirOp zero_op =
        MakeConstant(builder, 0, offs_type.getElementType(), {1});
    const Strides strides(1, 1);
    const Indices start_indices(1, 0);
    const Indices limit_indices(1, std::max<int64_t>(0, num_groups - 1));
    mlir::MlirOp offs_slice =
        mlir::stablehlo::Slice(*offs_op, start_indices, limit_indices, strides);
    mlir::MlirOp concat_op =
        mlir::stablehlo::Concatenate(builder, {zero_op, offs_slice}, 0);
    mlir::MlirOp group_sizes = mlir::stablehlo::Subtract(*offs_op, concat_op);

    mlir::MlirOp mat2_transposed = mlir::stablehlo::Transpose(mat2_op, {1, 0});
    mlir::MlirOp self_transposed =
        mlir::stablehlo::Transpose(self_op, {0, 2, 1});

    const mlir::chlo::RaggedDotDimensionNumbersAttr dimension_numbers =
        mlir::chlo::RaggedDotDimensionNumbersAttr::get(
            &self_op.getContext(), /*lhsBatchingDimensions=*/{},
            /*rhsBatchingDimensions=*/{},
            /*lhsContractingDimensions=*/{1},
            /*rhsContractingDimensions=*/{1},
            /*lhsRaggedDimensions=*/{0},
            /*rhsGroupDimensions=*/{0});

    const mlir::Type ragged_out_type = mlir::makeTensorType(
        self_op.getContext(),
        {mat2_type.getShape()[1], self_type.getShape()[1]}, out_dtype);
    mlir::MlirOp ragged_dot =
        mlir::chlo::RaggedDot(ragged_out_type, mat2_transposed, self_transposed,
                              group_sizes, dimension_numbers, precision_config);
    return mlir::stablehlo::Transpose(ragged_dot, {1, 0});
  } else if (a_is_2d && b_is_2d) {
    // Case 3 (2D x 2D with offs): contracting dimension is ragged (Mode 2).
    // The trailing dimension of lhs and the leading dimension of rhs are
    // partitioned according to the same offs tensor.
    ABSL_CHECK(offs_op.has_value());  // CRASH_OK
    const mlir::RankedTensorType offs_type = GetTensorTypeOrDie(*offs_op);
    const int64_t num_groups = offs_type.getShape()[0];

    mlir::MlirOp zero_op =
        MakeConstant(builder, 0, offs_type.getElementType(), {1});
    const Strides strides(1, 1);
    const Indices start_indices(1, 0);
    const Indices limit_indices(1, std::max<int64_t>(0, num_groups - 1));
    mlir::MlirOp offs_slice =
        mlir::stablehlo::Slice(*offs_op, start_indices, limit_indices, strides);
    mlir::MlirOp concat_op =
        mlir::stablehlo::Concatenate(builder, {zero_op, offs_slice}, 0);
    mlir::MlirOp group_sizes = mlir::stablehlo::Subtract(*offs_op, concat_op);

    const mlir::chlo::RaggedDotDimensionNumbersAttr dimension_numbers =
        mlir::chlo::RaggedDotDimensionNumbersAttr::get(
            &self_op.getContext(), /*lhsBatchingDimensions=*/{},
            /*rhsBatchingDimensions=*/{},
            /*lhsContractingDimensions=*/{1},
            /*rhsContractingDimensions=*/{0},
            /*lhsRaggedDimensions=*/{1},
            /*rhsGroupDimensions=*/{});

    const auto out_type = mlir::makeTensorType(
        self_op.getContext(),
        {num_groups, self_type.getShape()[0], mat2_type.getShape()[1]},
        out_dtype);
    return mlir::chlo::RaggedDot(out_type, self_op, mat2_op, group_sizes,
                                 dimension_numbers, precision_config);
  } else {
    // Case 4 (3D x 3D without offs): fallback to standard batched matrix
    // multiplication (DotGeneral).
    auto precision_config = mlir::stablehlo::PrecisionConfigAttr::get(
        &self_op.getContext(), {precision, precision});
    const mlir::stablehlo::DotDimensionNumbersAttr dot_dimension_numbers =
        mlir::stablehlo::DotDimensionNumbersAttr::get(
            &self_op.getContext(),
            /*lhs_batch_dimensions=*/{0},
            /*rhs_batch_dimensions=*/{0},
            /*lhs_contracting_dimensions=*/{2},
            /*rhs_contracting_dimensions=*/{1});
    return mlir::stablehlo::DotGeneral(self_op, mat2_op, dot_dimension_numbers,
                                       precision_config);
  }
}
absl::StatusOr<DeviceBufferRef> GroupedMm(
    const at::Tensor& self, const at::Tensor& mat2,
    const std::optional<at::Tensor>& offs,
    const std::optional<at::Tensor>& bias,
    std::optional<at::ScalarType> out_dtype, OpParamCacheKeys& param_keys) {
  // In ragged grouped GEMM, adding bias inside the kernel requires
  // expensive dynamic indexing because each expert group has an irregular
  // token count. To optimize performance, PyTorch CUDA/CPU kernels
  // explicitly reject bias.
  // Eager models also perform out.add_(bias) externally.
  TT_RET_CHECK(!bias.has_value() || !bias->defined(),
               error::kPythonNotImplementedError)
      << "expected bias to be undefined, got defined tensor";

  const bool a_is_2d = self.dim() == 2;
  const bool b_is_2d = mat2.dim() == 2;
  TT_RET_CHECK(self.dim() == 2 || self.dim() == 3, error::kInvalidArgument)
      << "expected self to be 2D or 3D, got " << self.dim() << "D";
  TT_RET_CHECK(mat2.dim() == 2 || mat2.dim() == 3, error::kInvalidArgument)
      << "expected mat2 to be 2D or 3D, got " << mat2.dim() << "D";
  TT_RET_CHECK(offs.has_value() == (a_is_2d || b_is_2d),
               error::kInvalidArgument)
      << "expected offs to be provided if and only if either self or mat2 is "
         "2D";

  c10::SmallVector<int64_t, 3> out_size;
  if (a_is_2d) {
    if (b_is_2d) {
      out_size = {offs->size(0), self.size(0), mat2.size(1)};
    } else {
      TT_RET_CHECK(offs->size(0) == mat2.size(0), error::kInvalidArgument)
          << "expected offs batch size to match mat2 batch size, got "
          << offs->size(0) << " and " << mat2.size(0);
      out_size = {self.size(0), mat2.size(-1)};
    }
  } else {
    if (b_is_2d) {
      TT_RET_CHECK(offs->size(0) == self.size(0), error::kInvalidArgument)
          << "expected offs batch size to match self batch size, got "
          << offs->size(0) << " and " << self.size(0);
      out_size = {self.size(1), mat2.size(1)};
    } else {
      TT_RET_CHECK(self.size(0) == mat2.size(0), error::kInvalidArgument)
          << "expected self batch size to match mat2 batch size, got "
          << self.size(0) << " and " << mat2.size(0);
      out_size = {self.size(0), self.size(1), mat2.size(-1)};
    }
  }

  at::ScalarType target_dtype = out_dtype.value_or(self.scalar_type());
  TT_ASSIGN_OR_RETURN(mlir::ElementType output_elem_type,
                      ConvertTo<mlir::ElementType>(target_dtype));

  Dimensions out_dims_vec(out_size.begin(), out_size.end());

  const auto current_precision = GetAndAddPrecisionTo(param_keys);

  std::vector<at::Tensor> inputs = {self, mat2};
  if (offs.has_value()) {
    inputs.push_back(*offs);
  }

  auto op_builder =
      [output_elem_type, current_precision](
          absl::Span<mlir::MlirOp> inputs,
          mlir::MlirBuilder& builder) -> absl::StatusOr<mlir::MlirOp> {
    auto self_op = inputs[0];
    auto mat2_op = inputs[1];
    std::optional<mlir::MlirOp> offs_op =
        inputs.size() > 2 ? std::make_optional(inputs[2]) : std::nullopt;
    TT_ASSIGN_OR_RETURN(
        auto result, BuildGroupedMmShlo(self_op, mat2_op, offs_op,
                                        output_elem_type, current_precision));
    return result;
  };

  TT_ASSIGN_OR_RETURN(
      auto result_buffer,
      DispatchOp<kDynamicSize>(std::move(op_builder), inputs,
                               {.out_dtype = output_elem_type,
                                .out_dims = out_dims_vec,
                                .op_param_cache_keys = std::move(param_keys)}));
  return result_buffer;
}
}  // namespace

at::Tensor AtenGroupedMm(const at::Tensor& self, const at::Tensor& mat2,
                         const std::optional<at::Tensor>& offs,
                         const std::optional<at::Tensor>& bias,
                         std::optional<at::ScalarType> out_dtype) {
  TT_KERNEL(OpName::kGroupedMm, param_keys, (self, mat2, offs, bias, out_dtype),
            {
              TT_ASSIGN_OR_THROW(
                  auto result_buffer,
                  GroupedMm(self, mat2, offs, bias, out_dtype, param_keys));
              return MakeTensor(result_buffer);
            });
}

}  // namespace torch_tpu
