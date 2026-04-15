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

#include "torch_tpu/ops/copy_from/tpu_to_cpu.h"

#include <cstdint>
#include <utility>

#include "absl/log/absl_check.h"
#include "absl/log/absl_log.h"
#include "absl/status/status.h"
#include "absl/strings/str_join.h"
#include "absl/types/span.h"
#include "ATen/core/ATen_fwd.h"
#include "ATen/core/TensorBody.h"
#include "ATen/ops/empty.h"
#include "torch/headeronly/core/ScalarType.h"
#include "torch_tpu/common/dtype.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/common/to_string.h"
#include "torch_tpu/eager/device_buffer.h"
#include "torch_tpu/eager/materialize.h"
#include "torch_tpu/pjrt/pjrt_utils.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "tsl/profiler/lib/traceme.h"

namespace torch_tpu {

namespace {

// Translates a low-level XLA OOM error to an error that is easier to understand
// by a pytorch user. If the status is not an XLA OOM error, returns it as is.
//
// `elem_type` is the element type of the tensor that caused the XLA OOM error.
// `dims` is the dimensions of the tensor that caused the XLA OOM error.
static absl::Status TranslateXlaTensorOomError(const absl::Status& status,
                                               at::ScalarType elem_type,
                                               absl::Span<const int64_t> dims) {
  TT_ASSIGN_OR_RETURN(  // ERROR_COV_INFEASIBLE=All PyTorch dtypes are supported
                        // by mlir::ElementType.
      const auto dtype, ConvertTo<mlir::ElementType>(elem_type));
  TT_RET_CHECK(!IsXlaOomError(status), error::kResourceExhausted)
      << "the TPU ran out of memory while awaiting the materialization of "
         "value "
      << ToString(dtype) << "[" << absl::StrJoin(dims, ", ") << "]:\n"
      << status.message();
  return status;
}

}  // namespace

absl::Status CopyTpuToCpu(const at::Tensor& src, const at::Tensor& dest,
                          bool non_blocking) {
  tsl::profiler::TraceMe trace("CopyTpuToCpu");
  ABSL_VLOG(1) << "[AtenCopyFrom] TPU -> CPU copy path for "
               << ToString(src, "src");

  if (src.numel() == 0) {
    // Shortcut: no data to move, just set dest's storage to an empty
    // c10::StorageImpl.
    at::Tensor empty_dest = at::empty(dest.sizes(), dest.options());
    dest.unsafeGetTensorImpl()->set_storage_keep_dtype(empty_dest.storage());
    return absl::OkStatus();
  }

  // Materialize if deferred, no-op if already materialized.
  // If src is a view, this will materialize both the base buffer and the
  // ephemeral view buffer.
  TT_ASSIGN_OR_RETURN(const DeviceBufferRef materialized_src_buf, [&]() {
    tsl::profiler::TraceMe trace_mat("CopyTpuToCpu::GetMaterialized");
    return GetMaterialized(src);
  }());

  ABSL_CHECK_EQ(materialized_src_buf.state(),  // CRASH_OK
                DeviceBufferRefState::kMaterialized)
      << "expected materialized buffer after GetMaterialized";

  // Fast path: if the destination tensor is fully allocated, contiguous, and
  // has the exact same size and dtype as the source, we can copy directly
  // into its memory. This enables true non-blocking transfers because we
  // don't need to invoke a secondary PyTorch CPU copy_() afterwards.
  bool can_copy_directly = dest.is_contiguous() &&
                           dest.scalar_type() == src.scalar_type() &&
                           dest.nbytes() == materialized_src_buf.size_bytes();

  if (can_copy_directly) {
    // Store the status first, so that we can translate the OpenXLA OOM error
    // message if needed.
    absl::Status status = TpuMemcpyDtoHDirect(materialized_src_buf,
                                              dest.data_ptr(), non_blocking);
    TT_RETURN_IF_ERROR(TranslateXlaTensorOomError(
        std::move(status), dest.scalar_type(), src.sizes()));
    return absl::OkStatus();
  }

  // Fallback: copy into a temporary buffer and rely on PyTorch's CPU copy_()
  // to handle type conversions and non-contiguous layouts. This must be
  // synchronous, otherwise the CPU copy_() will read garbage.
  TT_ASSIGN_OR_RETURN(
      at::Tensor cpu_tensor_receiver,
      TpuMemcpyDtoH(materialized_src_buf, /*non_blocking=*/false),
      TranslateXlaTensorOomError(_, dest.scalar_type(), src.sizes()));

  // Redispatch using copy_() on the CPU to do any type or layout conversions
  // needed.
  {
    tsl::profiler::TraceMe trace_copy("AtenCopy_ (Redispatch)");
    dest.copy_(cpu_tensor_receiver, non_blocking);
  }
  return absl::OkStatus();
}

}  // namespace torch_tpu
