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

#include "torch_tpu/ops/sort/sort_aten_kernels.h"

#include <cstdint>
#include <optional>
#include <tuple>
#include <utility>

#include "ATen/core/TensorBase.h"
#include "ATen/native/Resize.h"
#include "absl/status/statusor.h"
#include "c10/core/ScalarType.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "torch/csrc/autograd/generated/variable_factories.h"
#include "torch_tpu/common/aten_utils.h"
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
#include "torch_tpu/ops/sort/sort.h"
#include "xla/xla_data.pb.h"

namespace torch_tpu {

absl::StatusOr<DeviceBufferRefArray<2>> SortHelper(OpParamCacheKeys param_keys,
                                                   const at::Tensor& self,
                                                   const bool stable,
                                                   const int64_t dim,
                                                   const bool descending) {
  int64_t normalized_dim = 0;
  if (self.dim() > 0) {
    TT_ASSIGN_OR_RETURN(normalized_dim, SafeWrapDim(dim, self.dim()));
  }
  Dimensions output_dims = CopyIntVector(self.sizes());
  TT_ASSIGN_OR_RETURN(const auto elem_type,
                      ConvertTo<mlir::ElementType>(self.scalar_type()));
  auto op_builder =
      [stable, normalized_dim,
       descending](mlir::MlirOp input) -> absl::StatusOr<MlirOpResults<2>> {
    if (GetTensorTypeOrDie(input).getRank() == 0) {
      mlir::MlirBuilder& builder = input.getBuilder();
      return {{input, MakeScalarConstant(builder, int64_t{0},
                                         builder.getOpBuilder().getI64Type())}};
    }
    auto sort_shlo_outputs =
        BuildSortShlo(input, stable, normalized_dim, descending);
    return {{sort_shlo_outputs.values, sort_shlo_outputs.indices}};
  };
  TT_ASSIGN_OR_RETURN(
      auto result_buffers,
      (DispatchOp<1, 2>(std::move(op_builder), self,
                        {.out_dtypes = {elem_type, mlir::ElementType::I64},
                         .out_dims_list = {output_dims, output_dims},
                         .op_param_cache_keys = std::move(param_keys)})));
  return result_buffers;
}

std::tuple<at::Tensor&, at::Tensor&> AtenSortValuesStable(
    const at::Tensor& self, std::optional<bool> stable_opt, int64_t dim,
    bool descending, at::Tensor& values, at::Tensor& indices) {
  TT_KERNEL(OpName::kSortValuesStable, param_keys,
            (self, stable_opt, dim, descending, values, indices), {
              bool stable = stable_opt.value_or(false);
              TT_ASSIGN_OR_THROW(
                  (auto [values_buf, indices_buf]),
                  SortHelper(std::move(param_keys), self,
                             /*stable=*/stable, dim, descending));
              TT_THROW_IF_ERROR(
                  AssignBufferToAtTensor(std::move(values_buf), values));
              TT_THROW_IF_ERROR(
                  AssignBufferToAtTensor(std::move(indices_buf), indices));
              return {values, indices};
            });
}

}  // namespace torch_tpu
