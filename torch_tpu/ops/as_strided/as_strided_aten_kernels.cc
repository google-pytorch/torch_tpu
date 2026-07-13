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

#include "torch_tpu/ops/as_strided/as_strided_aten_kernels.h"

#include <cstdint>
#include <optional>
#include <utility>

#include "ATen/core/TensorBody.h"
#include "absl/status/statusor.h"
#include "c10/core/Storage.h"
#include "c10/core/SymInt.h"
#include "c10/core/SymIntArrayRef.h"
#include "c10/core/TensorImpl.h"
#include "c10/util/ArrayRef.h"
#include "c10/util/Optional.h"
#include "c10/util/intrusive_ptr.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "torch_tpu/common/cache_key.h"
#include "torch_tpu/common/dimension_types.h"
#include "torch_tpu/common/dtype.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/common/utils.h"
#include "torch_tpu/eager/device_buffer.h"
#include "torch_tpu/eager/tensor_to_buffer.h"
#include "torch_tpu/ops/macros/kernel.h"
#include "torch_tpu/ops/op_names.h"
#include "torch_tpu/ops/stride/stride_helper.h"

namespace torch_tpu {

absl::StatusOr<at::Tensor> AsStrided(const at::Tensor& self,
                                     c10::IntArrayRef size,
                                     c10::IntArrayRef stride,
                                     int64_t storage_offset) {
  TT_RET_CHECK(  // ERROR_COV_INFEASIBLE==PyTorch catches this error first.
      self.defined(), error::kInvalidArgument)
      << "the input tensor cannot be undefined";
  TT_ASSIGN_OR_RETURN(DeviceBufferRef base_buffer_ref, GetBaseBuffer(self));

  const int64_t storage_numel = base_buffer_ref.num_elements();
  const mlir::ElementType storage_element_type = base_buffer_ref.element_type();
  TT_ASSIGN_OR_RETURN(mlir::ElementType tensor_element_type,
                      ConvertTo<mlir::ElementType>(self.scalar_type()));

  Dimensions size_dims = CopyIntVector(size);
  Strides stride_dims = CopyIntVector(stride);

  TT_RETURN_IF_ERROR(CheckProvidedLayoutDataFitsInStorage(
      storage_numel, storage_element_type, size_dims, stride_dims,
      storage_offset, tensor_element_type));

  c10::Storage storage_copy = self.storage();
  at::Tensor tensor(c10::make_intrusive<c10::TensorImpl>(
      c10::TensorImpl::VIEW, std::move(storage_copy), self.key_set(),
      self.dtype()));
  tensor.unsafeGetTensorImpl()->set_sizes_and_strides(size_dims, stride_dims,
                                                      storage_offset);
  return tensor;
}

at::Tensor AtenAsStrided(const at::Tensor& self, c10::SymIntArrayRef size_sym,
                         c10::SymIntArrayRef stride_sym,
                         c10::optional<c10::SymInt> storage_offset_sym_opt) {
  // SymInts need to be resolved before TT_KERNEL so that they have concrete
  // values for logging.
  Dimensions new_sizes;
  new_sizes.reserve(size_sym.size());
  for (const auto& s : size_sym)
    new_sizes.push_back(s.guard_int(__FILE__, __LINE__));

  Strides new_strides;
  new_strides.reserve(stride_sym.size());
  for (const auto& s : stride_sym)
    new_strides.push_back(s.guard_int(__FILE__, __LINE__));

  int64_t new_storage_offset =
      storage_offset_sym_opt.has_value()
          ? storage_offset_sym_opt.value().guard_int(__FILE__, __LINE__)
          : self.storage_offset();

  TT_KERNEL(OpName::kAsStrided, _,
            (self,
             // No need to include these in the cache key as this "kernel"
             // doesn't lower to SHLO.
             IgnoreInCacheKey(new_sizes, "Doesn't affect SHLO"),
             IgnoreInCacheKey(new_strides, "Doesn't affect SHLO"),
             IgnoreInCacheKey(new_storage_offset, "Doesn't affect SHLO")),
            {
              TT_ASSIGN_OR_THROW(
                  at::Tensor result,
                  AsStrided(self, new_sizes, new_strides, new_storage_offset));
              return result;
            });
}

}  // namespace torch_tpu
