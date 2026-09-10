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

#include "csrc/ops/resize/resize_aten_kernels.h"

#include <cstdint>
#include <utility>

#include "ATen/ceil_div.h"
#include "ATen/core/ATen_fwd.h"
#include "ATen/core/TensorBody.h"
#include "absl/log/absl_log.h"
#include "absl/status/status.h"
#include "c10/util/ArrayRef.h"
#include "c10/util/Optional.h"
#include "c10/util/accumulate.h"
#include "csrc/common/cache_key.h"
#include "csrc/common/dimension_types.h"
#include "csrc/common/dtype.h"
#include "csrc/common/error_utils.h"
#include "csrc/common/utils.h"
#include "csrc/eager/device_buffer.h"
#include "csrc/eager/tensor_to_buffer.h"
#include "csrc/ops/as_strided/as_strided_aten_kernels.h"
#include "csrc/ops/macros/kernel.h"
#include "csrc/ops/op_names.h"
#include "csrc/ops/stride/stride_helper.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "torch/headeronly/core/MemoryFormat.h"
#include "torch/headeronly/core/ScalarType.h"
#include "xla/shape_util.h"
#include "xla/xla_data.pb.h"

namespace torch_tpu {

absl::Status ResizeTensor(const at::Tensor& self, c10::IntArrayRef size) {
  // Validate the requested new size against the tensor's logical dtype.
  // Then, compare the requested capacity in bytes (implied by the new size and
  // the self tensor's logical dtype) against the current physical storage
  // capacity in bytes (implied by the base buffer) to determine if we need
  // to grow the storage.
  TT_ASSIGN_OR_RETURN(DeviceBufferRef base_buffer_ref, GetBaseBuffer(self));
  TT_ASSIGN_OR_RETURN(const int64_t old_storage_capacity_bytes,
                      ValidateTensorByteSize(base_buffer_ref.dimensions(),
                                             base_buffer_ref.element_type()));
  TT_ASSIGN_OR_RETURN(const auto self_mlir_dtype,
                      ConvertTo<mlir::ElementType>(self.scalar_type()));
  Dimensions new_size_vec = CopyIntVector(size);
  TT_ASSIGN_OR_RETURN(const int64_t new_storage_capacity_bytes,
                      ValidateTensorByteSize(new_size_vec, self_mlir_dtype));

  if (new_storage_capacity_bytes <= old_storage_capacity_bytes) {
    ABSL_VLOG(1) << "[C++ KERNEL AtenResize_] Resize is shrinking or staying "
                    "the same size. Updating layout on existing tensor "
                    "without allocating new storage.";
  } else {
    ABSL_VLOG(1) << "[C++ KERNEL AtenResize_] Resize is growing. Swapping "
                    "tensor storage with a new empty storage.";

    // Since the tensor being resized (self) might be a view with a different
    // dtype than its physical storage (e.g. a bool view of a uint32 tensor), we
    // must allocate the new storage using the base buffer's element type
    // (storage type). This preserves the element type of the shared storage and
    // avoids type mismatches when copying the existing data from the old
    // storage to the new one.
    const int64_t base_element_size_bytes =
        xla::ShapeUtil::ByteSizeOfPrimitiveType(
            ConvertTo<xla::PrimitiveType>(base_buffer_ref.element_type()));
    const int64_t base_elements_needed =
        at::ceil_div(new_storage_capacity_bytes, base_element_size_bytes);
    at::ScalarType base_scalar_type =
        ConvertTo<at::ScalarType>(base_buffer_ref.element_type());
    TT_ASSIGN_OR_RETURN(at::Tensor larger_empty,
                        MakeEmptyTensor({base_elements_needed},
                                        base_scalar_type, self.device()));

    // Take a view on the new tensor corresponding to the existing data.
    Strides base_strides =
        CalculateStridesContiguous(base_buffer_ref.dimensions());
    TT_ASSIGN_OR_RETURN(
        at::Tensor view_on_empty,
        AsStrided(larger_empty, base_buffer_ref.dimensions(), base_strides,
                  /*storage_offset=*/0));

    // Copy the existing data into the view window. This will leave some
    // elements uninitialized in the resized tensor.
    TT_RETURN_IF_ERROR(AssignBufferToAtTensor(base_buffer_ref, view_on_empty));

    // Then, set the c10::DataPtr of the existing tensor to be this newly-
    // resized data buffer, and update the size on the c10::Storage.
    // This ensures that all views will use this new larger storage; this
    // will not invalidate any existing views as they will all have at least as
    // much data as they require.
    self.storage().set_data_ptr(
        std::move(larger_empty.storage().mutable_data_ptr()));
    self.storage().set_nbytes(larger_empty.storage().nbytes());
  }
  // The tensor is always interpreted as contiguous after a resize.
  self.unsafeGetTensorImpl()->set_sizes_contiguous(size);
  self.unsafeGetTensorImpl()->set_storage_offset(0);
  return absl::OkStatus();
}

const at::Tensor& AtenResize_(
    const at::Tensor& self, c10::IntArrayRef size,
    c10::optional<at::MemoryFormat> memory_format_opt) {
  TT_KERNEL(OpName::kResize_, _,
            (self, IgnoreInCacheKey(size, "Doesn't affect SHLO"),
             IgnoreInCacheKey(memory_format_opt, "Doesn't affect SHLO")),
            {
              TT_CHECK_THROW(
                  !memory_format_opt.has_value() ||
                      *memory_format_opt == at::MemoryFormat::Contiguous,
                  error::kPythonNotImplementedError)
                  << "non-contiguous memory formats are not yet supported";

              TT_THROW_IF_ERROR(ResizeTensor(self, size));
              return self;
            });
}

}  // namespace torch_tpu
