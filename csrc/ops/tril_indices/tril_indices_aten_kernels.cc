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

#include "csrc/ops/tril_indices/tril_indices_aten_kernels.h"

#include <cstdint>
#include <optional>
#include <utility>

#include "ATen/core/ATen_fwd.h"
#include "ATen/native/TensorFactories.h"
#include "absl/status/statusor.h"
#include "c10/core/Device.h"
#include "csrc/common/dtype.h"
#include "csrc/common/error_utils.h"
#include "csrc/common/to_string.h"
#include "csrc/ops/macros/kernel.h"
#include "csrc/ops/nonzero/nonzero.h"
#include "csrc/ops/nullary_aten_kernels.h"
#include "csrc/ops/op_builder_utils.h"
#include "csrc/ops/op_names.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Types.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"
#include "torch/headeronly/core/Layout.h"
#include "torch/headeronly/core/ScalarType.h"

namespace torch_tpu {

namespace {

absl::StatusOr<mlir::MlirOp> BuildTrilIndicesShlo(mlir::MlirBuilder& builder,
                                                  int64_t row, int64_t col,
                                                  int64_t offset,
                                                  int64_t tril_size,
                                                  mlir::Type element_type) {
  // Based on jnp.tril_indices StableHLO dump, should be good for TPU
  // Create a grid of row indices (i) of shape (row, col).
  auto iota_row = mlir::stablehlo::Iota(
      builder,
      mlir::RankedTensorType::get({row, col},
                                  builder.getOpBuilder().getI64Type()),
      /*iota_dimension=*/0);

  // Create a grid of column indices (j) of shape (row, col).
  auto iota_col = mlir::stablehlo::Iota(
      builder,
      mlir::RankedTensorType::get({row, col},
                                  builder.getOpBuilder().getI64Type()),
      /*iota_dimension=*/1);

  // Create a constant for the offset and broadcast it to (row, col).
  mlir::MlirOp offset_op = MakeConstantLike(iota_row, offset);

  // Calculate `i + offset` and compare `j <= i + offset` to get the boolean
  // mask.
  auto i_plus_offset = mlir::stablehlo::Add(iota_row, offset_op);
  auto mask = mlir::stablehlo::Compare(
      iota_col, i_plus_offset, mlir::stablehlo::ComparisonDirection::LE);

  // Reuse the BuildNonzeroShlo function to get the coordinates of non-zero
  // values.
  TT_ASSIGN_OR_RETURN(mlir::MlirOp nonzero_op,
                      BuildNonzeroShlo(tril_size, mask));

  // The result from BuildNonzeroShlo is shape (tril_size, 2), expected
  // tril_indices is (2, tril_size). So we need to transpose it.
  auto transposed_indices =
      mlir::stablehlo::Transpose(nonzero_op, /*permutation=*/{1, 0});

  return mlir::stablehlo::ConvertElementType(transposed_indices, element_type);
}

}  // namespace

static absl::StatusOr<MlirNullaryOpBuilder> GetTrilIndicesFunctional(
    int64_t row, int64_t col, int64_t offset, int64_t tril_size,
    at::ScalarType dtype) {
  TT_ASSIGN_OR_RETURN(auto mlir_elem_type, ConvertTo<mlir::ElementType>(dtype));
  return [row, col, offset, tril_size, mlir_elem_type](
             mlir::MlirBuilder& builder) -> absl::StatusOr<mlir::MlirOp> {
    TT_ASSIGN_OR_RETURN(const mlir::Type mlir_type,
                        GetMlirType(builder.getContext(), mlir_elem_type));
    return BuildTrilIndicesShlo(builder, row, col, offset, tril_size,
                                mlir_type);
  };
}

at::Tensor AtenTrilIndices(int64_t row, int64_t col, int64_t offset,
                           std::optional<at::ScalarType> dtype_opt,
                           std::optional<at::Layout> layout_opt,
                           std::optional<at::Device> device_opt,
                           std::optional<bool> pin_memory_opt) {
  TT_KERNEL(
      OpName::kTrilIndices, param_keys,
      (row, col, offset, dtype_opt, layout_opt, device_opt, pin_memory_opt), {
        CheckDeviceIsTpu(device_opt, "tril_indices");
        const at::ScalarType dtype = dtype_opt.value_or(at::ScalarType::Long);

        at::native::check_args(row, col, layout_opt);

        const int64_t tril_size = at::native::get_tril_size(row, col, offset);
        if (tril_size > 0) {
          // PyTorch CUDA only dispatches int32 and int64 when tril_size > 0;
          // zero-size outputs bypass kernel dispatch on CUDA.
          TT_CHECK_THROW(
              dtype == at::ScalarType::Int || dtype == at::ScalarType::Long,
              error::kInvalidArgument)
              << "expected the dtype to be either int32 or int64, got "
              << ToString(dtype);
        }
        TT_ASSIGN_OR_THROW(
            auto builder,
            GetTrilIndicesFunctional(row, col, offset, tril_size, dtype));

        return ApplyNullaryOp(std::move(builder), dtype, {2, tril_size},
                              std::move(param_keys));
      });
}

}  // namespace torch_tpu
