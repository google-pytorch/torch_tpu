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

#include "torch_tpu/ops/set/set_aten_kernels.h"

#include <cstdint>
#include <utility>

#include "ATen/core/TensorBody.h"
#include "c10/core/Storage.h"
#include "c10/core/SymInt.h"
#include "c10/core/SymIntArrayRef.h"
#include "c10/core/TensorImpl.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "torch_tpu/common/cache_key.h"
#include "torch_tpu/common/dimension_types.h"
#include "torch_tpu/common/dtype.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/eager/device_buffer.h"
#include "torch_tpu/eager/tensor_to_buffer.h"
#include "torch_tpu/ops/macros/kernel.h"
#include "torch_tpu/ops/op_names.h"
#include "torch_tpu/ops/stride/stride_helper.h"

namespace torch_tpu {

at::Tensor& AtenSet_(at::Tensor& self) {
  TT_KERNEL(OpName::kSet_, _, (self), {
    TT_ASSIGN_OR_THROW(auto element_type,
                       ConvertTo<mlir::ElementType>(self.scalar_type()));
    TT_ASSIGN_OR_THROW(DeviceBufferRef zero_size_buffer_ref,
                       DeviceBufferList::CreateEmpty({0}, element_type));
    c10::Storage storage = MakeStorage(zero_size_buffer_ref);
    c10::TensorImpl* impl = self.unsafeGetTensorImpl();
    impl->set_storage_keep_dtype(std::move(storage));
    impl->set_storage_offset(0);
    impl->set_sizes_contiguous({0});
    return self;
  });
}

at::Tensor& AtenSet_SourceStorage(at::Tensor& self, c10::Storage src) {
  TT_KERNEL(OpName::kSet_SourceStorage, _,
            (self, IgnoreInCacheKey(src, "Legacy usage")), {
              c10::TensorImpl* impl = self.unsafeGetTensorImpl();
              const int64_t element_size = self.element_size();
              const int64_t numel =
                  element_size > 0 ? src.nbytes() / element_size : 0;
              impl->set_storage_keep_dtype(std::move(src));
              impl->set_storage_offset(0);
              impl->set_sizes_contiguous({numel});
              return self;
            });
}

at::Tensor& AtenSet_SourceStorageOffset(at::Tensor& self, c10::Storage src,
                                        c10::SymInt storage_offset,
                                        c10::SymIntArrayRef size,
                                        c10::SymIntArrayRef stride) {
  int64_t concrete_storage_offset =
      storage_offset.guard_int(__FILE__, __LINE__);

  Dimensions size_vec;
  for (const auto& s : size) {
    size_vec.push_back(s.guard_int(__FILE__, __LINE__));
  }
  Strides stride_vec;
  for (const auto& s : stride) {
    stride_vec.push_back(s.guard_int(__FILE__, __LINE__));
  }
  TT_KERNEL(
      OpName::kSet_SourceStorageOffset, _,
      (self, IgnoreInCacheKey(src, "Legacy usage"),
       IgnoreInCacheKey(concrete_storage_offset, "Legacy usage"),
       IgnoreInCacheKey(size_vec, "Legacy usage"),
       IgnoreInCacheKey(stride_vec, "Legacy usage")),
      {
        c10::TensorImpl* impl = self.unsafeGetTensorImpl();
        TT_ASSIGN_OR_THROW(DeviceBufferRef buffer_ref, GetBaseBuffer(src));
        const int64_t storage_numel = buffer_ref.num_elements();
        const mlir::ElementType storage_dtype = buffer_ref.element_type();
        TT_ASSIGN_OR_THROW(const mlir::ElementType tensor_dtype,
                           ConvertTo<mlir::ElementType>(self.scalar_type()));
        TT_THROW_IF_ERROR(CheckProvidedLayoutDataFitsInStorage(
            storage_numel, storage_dtype, size_vec, stride_vec,
            concrete_storage_offset, tensor_dtype));
        impl->set_storage_keep_dtype(std::move(src));
        impl->set_sizes_and_strides(/*sizes=*/size_vec,
                                    /*strides=*/stride_vec,
                                    /*storage_offset=*/concrete_storage_offset);
        return self;
      });
}

at::Tensor& AtenSet_SourceTensor(at::Tensor& self, const at::Tensor& src) {
  TT_KERNEL(OpName::kSet_SourceTensor, _, (self, src), {
    auto tensor_impl = src.unsafeGetTensorImpl();
    c10::Storage src_storage = tensor_impl->storage();
    c10::TensorImpl* impl = self.unsafeGetTensorImpl();
    impl->set_storage_keep_dtype(std::move(src_storage));
    impl->set_sizes_and_strides(
        /*sizes=*/tensor_impl->sizes(), /*strides=*/tensor_impl->strides(),
        /*storage_offset=*/tensor_impl->storage_offset());
    return self;
  });
}

}  // namespace torch_tpu
