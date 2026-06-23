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

#include "torch_tpu/ops/copy_from/tpu_to_tpu.h"

#include <utility>

#include "ATen/core/TensorBody.h"
#include "absl/log/absl_log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "c10/core/Device.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"
#include "torch_tpu/common/cache_key.h"
#include "torch_tpu/common/dtype.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/common/to_string.h"
#include "torch_tpu/eager/device_buffer.h"
#include "torch_tpu/eager/op_dispatcher.h"
#include "torch_tpu/eager/tensor_to_buffer.h"
#include "torch_tpu/ops/op_builder_utils.h"

namespace torch_tpu {

absl::Status CopyTpuToTpu(const at::Tensor& src, const at::Tensor& dest) {
  ABSL_VLOG(1) << "[AtenCopyFrom] TPU -> TPU copy path for "
               << ToString(src, "src");
  TT_RET_CHECK(  // ERROR_COV_INFEASIBLE=Can't use 2 TPUs in one host.
      src.device().index() == dest.device().index(), error::kUnimplemented)
      << "expected source and destination device indices to match, got source "
         "device '"
      << src.device() << "' vs destination device '" << dest.device() << "'";
  TT_RET_CHECK(  // ERROR_COV_INFEASIBLE=AtenCopyFrom (sole caller) catches this
                 // error when trying to broadcast the inputs.
      src.sizes() == dest.sizes(), error::kInvalidArgument)
      << "_copy_from does not support resizing. Please use "
         "_copy_from_and_resize instead.";

  if (src.dtype() == dest.dtype()) {
    // Shape and type match, can simply reuse the existing DeviceBufferRef
    // from src for dest.
    TT_ASSIGN_OR_RETURN(const DeviceBufferRef src_buf, GetBuffer(src));
    return AssignBufferToAtTensor(src_buf, dest);
  }

  // If the dtype is different, then we need to dispatch a StableHLO
  // ConvertElementType op to do the type conversion.
  TT_ASSIGN_OR_RETURN(const auto out_dtype,
                      ConvertTo<mlir::ElementType>(dest.scalar_type()));
  auto unary_op_builder =
      [out_dtype](mlir::MlirOp input) -> absl::StatusOr<mlir::MlirOp> {
    return mlir::stablehlo::ConvertElementType(input, out_dtype);
  };
  TT_ASSIGN_OR_RETURN(
      auto new_buf,
      DispatchOp<1>(std::move(unary_op_builder), src,
                    {.out_dtype = out_dtype,
                     .out_dims = dest.sizes(),
                     .op_param_cache_keys = OpParamCacheKeys::Empty()}),
      _.SetPrepend()
          << "copy from 'tpu' to 'tpu' device (dtype change) failed with: ");
  return AssignBufferToAtTensor(new_buf, dest);
}

}  // namespace torch_tpu
