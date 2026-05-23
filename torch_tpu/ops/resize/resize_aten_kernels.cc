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

#include "torch_tpu/ops/resize/resize_aten_kernels.h"

#include <cstdint>
#include <utility>

#include "ATen/core/ATen_fwd.h"
#include "ATen/core/TensorBody.h"
#include "absl/log/absl_log.h"
#include "c10/core/SymIntArrayRef.h"
#include "c10/util/ArrayRef.h"
#include "c10/util/Optional.h"
#include "c10/util/accumulate.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "torch/headeronly/core/MemoryFormat.h"
#include "torch_tpu/common/aten_utils.h"
#include "torch_tpu/common/cache_key.h"
#include "torch_tpu/common/dimension_types.h"
#include "torch_tpu/common/dtype.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/eager/device_buffer.h"
#include "torch_tpu/eager/tensor_to_buffer.h"
#include "torch_tpu/ops/as_strided/as_strided_aten_kernels.h"
#include "torch_tpu/ops/macros/kernel.h"
#include "torch_tpu/ops/nullary_aten_kernels.h"
#include "torch_tpu/ops/op_names.h"
#include "torch_tpu/ops/stride/stride_helper.h"
#include "xla/xla_data.pb.h"

namespace torch_tpu {

const at::Tensor& AtenResize_(
    const at::Tensor& self, c10::IntArrayRef size,
    c10::optional<at::MemoryFormat> memory_format_opt) {
  TT_KERNEL(
      OpName::kResize_, _,
      (self, IgnoreInCacheKey(size, "Legacy usage"),
       IgnoreInCacheKey(memory_format_opt, "Legacy usage")),
      {
        TT_CHECK_THROW(!memory_format_opt.has_value() ||
                           *memory_format_opt == at::MemoryFormat::Contiguous,
                       error::kUnimplemented)
            << "non-contiguous memory formats are not yet supported";

        // Validate new size, and determine if this is shrinking or growing.
        TT_ASSIGN_OR_THROW(DeviceBufferRef base_buffer_ref,
                           GetBaseBuffer(self));
        const int64_t old_storage_capacity = base_buffer_ref.num_elements();
        TT_ASSIGN_OR_THROW(const auto mlir_dtype,
                           ConvertTo<mlir::ElementType>(self.scalar_type()));
        Dimensions new_size_vec = CopyIntVector(size);
        TT_THROW_IF_ERROR(ValidateTensorByteSize(new_size_vec, mlir_dtype));
        const int64_t new_storage_capacity =
            c10::multiply_integers(new_size_vec);

        if (new_storage_capacity <= old_storage_capacity) {
          ABSL_VLOG(1)
              << "[C++ KERNEL AtenResize_] Resize is shrinking or staying "
                 "the same size. Updating layout on existing tensor "
                 "without allocating new storage.";
        } else {
          ABSL_VLOG(1)
              << "[C++ KERNEL AtenResize_] Resize is growing. Swapping "
                 "tensor storage with a new empty storage.";
          TT_ASSIGN_OR_THROW(
              at::Tensor larger_empty,
              MakeEmptyTensor(size, self.scalar_type(), self.device()));

          // Take a view on the new tensor corresponding to the existing data.
          Strides base_strides =
              CalculateStridesContiguous(base_buffer_ref.dimensions());
          c10::SymIntArrayRef base_sizes_sym =
              c10::fromIntArrayRefKnownNonNegative(
                  base_buffer_ref.dimensions());
          c10::SymIntArrayRef base_strides_sym =
              c10::fromIntArrayRefKnownNonNegative(base_strides);
          at::Tensor view_on_empty =
              AtenAsStrided(larger_empty, base_sizes_sym, base_strides_sym,
                            /*storage_offset=*/0);

          // Copy the existing data into the view window. This will leave some
          // elements uninitialized in the resized tensor.
          TT_THROW_IF_ERROR(
              AssignBufferToAtTensor(base_buffer_ref, view_on_empty));

          // Then, set the c10::DataPtr of the existing tensor to be this newly-
          // resized data buffer, and update the size on the c10::Storage.
          // This ensures that all views will use this new larger storage; this
          // will not invalidate any existing views as they will all have at
          // least as much data as they require.
          self.storage().set_data_ptr(
              std::move(larger_empty.storage().mutable_data_ptr()));
          self.storage().set_nbytes(larger_empty.storage().nbytes());
        }
        // The tensor is always interpreted as contiguous after a resize.
        self.unsafeGetTensorImpl()->set_sizes_contiguous(size);
        self.unsafeGetTensorImpl()->set_storage_offset(0);
        return self;
      });
}

}  // namespace torch_tpu
