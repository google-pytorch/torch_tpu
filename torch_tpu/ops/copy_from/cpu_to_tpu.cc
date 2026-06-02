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

#include "torch_tpu/ops/copy_from/cpu_to_tpu.h"

#include <optional>
#include <utility>

#include "ATen/core/TensorBody.h"
#include "absl/log/absl_log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "torch_tpu/common/dimension_types.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/common/layout_utils.h"
#include "torch_tpu/common/to_string.h"
#include "torch_tpu/eager/device_buffer.h"
#include "torch_tpu/eager/device_buffer_utils.h"
#include "torch_tpu/eager/tensor_to_buffer.h"
#include "torch_tpu/pjrt/pjrt_utils.h"
#include "xla/xla_data.pb.h"

namespace torch_tpu {

absl::StatusOr<DeviceBufferRef> CopyCpuToTpuBuffer(const at::Tensor& src,
                                                   bool non_blocking) {
  at::Tensor contiguous_src_for_tpu = src.contiguous();
  ABSL_VLOG(1) << "[AtenCopyFrom] CPU -> TPU: Ensured CPU tensor is "
                  "contiguous with dtype "
               << ToString(src.scalar_type()) << ". Tensor: " << ToString(src);

  TT_ASSIGN_OR_RETURN(TpuLayout tpu_layout,
                      ResolveTpuLayout(contiguous_src_for_tpu));

  if (contiguous_src_for_tpu.numel() == 0) {
    // If there's no data to actually copy, then we can just create a zero-sized
    // buffer directly and avoid blocking on host/device synchronization.
    return CreateZeroSizeDeviceBufferRef(std::move(tpu_layout.sizes),
                                         tpu_layout.element_type);
  }

  return TpuMallocAndMemcpyHtoD(
      contiguous_src_for_tpu.data_ptr(), tpu_layout.element_type,
      tpu_layout.sizes,
      non_blocking ? std::make_optional(contiguous_src_for_tpu) : std::nullopt);
}

absl::Status CopyCpuToTpu(const at::Tensor& src, const at::Tensor& dest,
                          bool non_blocking) {
  ABSL_VLOG(1) << "[AtenCopyFrom] CPU -> TPU copy path for "
               << ToString(src, "src");
  at::Tensor src_with_dest_dtype = src;
  if (src.scalar_type() != dest.scalar_type()) {
    ABSL_VLOG(1) << "[AtenCopyFrom] CPU -> TPU: Converting CPU tensor "
                    "dtype from "
                 << ToString(src.scalar_type()) << " to "
                 << ToString(dest.scalar_type());
    // Do the type conversion on the host before transfer.
    src_with_dest_dtype = src.to(dest.scalar_type());
  }

  TT_ASSIGN_OR_RETURN(DeviceBufferRef tpu_buf,
                      CopyCpuToTpuBuffer(src_with_dest_dtype, non_blocking),
                      _.SetPrepend()
                          << "transfer to 'tpu' device failed with: ");
  TT_RETURN_IF_ERROR(AssignBufferToAtTensor(tpu_buf, dest));
  return absl::OkStatus();
}

}  // namespace torch_tpu
