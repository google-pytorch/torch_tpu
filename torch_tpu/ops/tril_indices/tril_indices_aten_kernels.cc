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

#include "torch_tpu/ops/tril_indices/tril_indices_aten_kernels.h"

#include <cstdint>
#include <optional>
#include <utility>

#include "ATen/core/ATen_fwd.h"
#include "ATen/native/TensorFactories.h"
#include "absl/status/statusor.h"
#include "c10/core/Device.h"
#include "mlir/IR/Types.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "torch/headeronly/core/Layout.h"
#include "torch/headeronly/core/ScalarType.h"
#include "torch_tpu/common/dtype.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/common/to_string.h"
#include "torch_tpu/ops/macros/kernel.h"
#include "torch_tpu/ops/nullary_aten_kernels.h"
#include "torch_tpu/ops/op_builder_utils.h"
#include "torch_tpu/ops/op_names.h"
#include "torch_tpu/ops/tril_indices/tril_indices.h"
#include "xla/xla_data.pb.h"

namespace torch_tpu {

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
        at::ScalarType dtype = dtype_opt.value_or(at::ScalarType::Long);

        // PyTorch native devices (CPU and CUDA) only support int32 and int64.
        TT_CHECK_THROW(
            dtype == at::ScalarType::Long || dtype == at::ScalarType::Int,
            error::kInvalidArgument)
            << "expected the dtype to be either int32 or int64, got "
            << ToString(dtype);

        at::native::check_args(row, col, layout_opt);

        int64_t tril_size = at::native::get_tril_size(row, col, offset);
        TT_ASSIGN_OR_THROW(
            auto builder,
            GetTrilIndicesFunctional(row, col, offset, tril_size, dtype));

        return ApplyNullaryOp(std::move(builder), dtype, {2, tril_size},
                              std::move(param_keys));
      });
}

}  // namespace torch_tpu
