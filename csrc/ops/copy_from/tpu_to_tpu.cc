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

#include "csrc/ops/copy_from/tpu_to_tpu.h"

#include <utility>

#include "ATen/core/TensorBody.h"
#include "absl/log/absl_check.h"
#include "absl/log/absl_log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "c10/core/Device.h"
#include "c10/core/ScalarType.h"
#include "csrc/common/cache_key.h"
#include "csrc/common/dimension_types.h"
#include "csrc/common/dtype.h"
#include "csrc/common/error_utils.h"
#include "csrc/common/to_string.h"
#include "csrc/common/utils.h"
#include "csrc/eager/device_buffer.h"
#include "csrc/eager/op_dispatcher.h"
#include "csrc/eager/tensor_to_buffer.h"
#include "csrc/ops/copy_from/copy_common.h"
#include "csrc/ops/op_builder_utils.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"

namespace torch_tpu {
namespace {

// Slices the first half of elements to discard the padding.
// A packed FP4 tensor has 2N physical elements on TPU (1 byte per unpacked
// element). When dequantizing, we retrieve the first N elements and discard the
// padding.
absl::StatusOr<mlir::MlirOp> SliceFP4(mlir::MlirOp input) {
  auto input_type = GetTensorTypeOrDie(input);
  auto rank = input_type.getRank();

  const Indices start_indices(rank, 0);
  Indices limit_indices = CopyIntVector(input_type.getShape());
  if (rank > 0) {
    ABSL_DCHECK_EQ(limit_indices.back() % 2, 0)
        << "Expected even last dimension for FP4, got " << limit_indices.back();
    limit_indices.back() /= 2;
  }
  const Strides strides(rank, 1);

  return mlir::stablehlo::Slice(input, start_indices, limit_indices, strides);
}

// Zero-pads the last dimension to double the size (logical N -> physical 2N).
absl::StatusOr<mlir::MlirOp> PadFP4(mlir::MlirOp input) {
  auto input_type = GetTensorTypeOrDie(input);
  auto rank = input_type.getRank();

  const Dimensions low_padding(rank, 0);
  Dimensions high_padding(rank, 0);
  if (rank > 0) {
    high_padding.back() = input_type.getShape().back();
  }
  const Dimensions interior_padding(rank, 0);

  auto zero_pad_value =
      MakeScalarConstant(input.getBuilder(), 0.0, input_type.getElementType());

  return mlir::stablehlo::Pad(input, zero_pad_value, low_padding, high_padding,
                              interior_padding);
}

}  // namespace

absl::Status CopyTpuToTpu(const at::Tensor& src, const at::Tensor& dest) {
  ABSL_VLOG(1) << "[AtenCopyFrom] TPU -> TPU copy path for "
               << ToString(src, "src");
  TT_RET_CHECK(  // ERROR_COV_INFEASIBLE=Can't use 2 TPUs in one host.
      src.device().index() == dest.device().index(),
      error::kPythonNotImplementedError)
      << "expected source and destination device indices to match, got source "
         "device '"
      << src.device() << "' vs destination device '" << dest.device() << "'";

  // Broadcast src to match dest's shape, and transpose both to be row-major.
  // Without this, an operation like `x.t().copy_(y.t())` would miss the
  // IsFullStorageReinterpretView optimization below, and would execute as
  // transpose -> transpose needlessly when it could just do full overwrite.
  at::Tensor viewed_src = src;
  at::Tensor viewed_dest = dest;
  TT_RETURN_IF_ERROR(BroadcastAndTranspose(viewed_src, viewed_dest));
  ABSL_CHECK(viewed_src.sizes() == viewed_dest.sizes());  // CRASH_OK

  if (src.dtype() == dest.dtype()) {
    // Shape and type match, can simply reuse the existing DeviceBufferRef
    // from src for dest.
    //
    // If both src and dest are pure reinterpretations of all the bytes of
    // src's base buffer, dest can share src's base buffer directly, regardless
    // of its element type.
    TT_ASSIGN_OR_RETURN(DeviceBufferRef src_base_buf,
                        GetBaseBuffer(viewed_src));
    if (IsFullStorageReinterpretView(viewed_src, src_base_buf) &&
        IsFullStorageReinterpretView(viewed_dest, src_base_buf)) {
      return AssignBufferAsStorageBase(std::move(src_base_buf), viewed_dest);
    }
    TT_ASSIGN_OR_RETURN(const DeviceBufferRef src_buf, GetBuffer(viewed_src));
    return AssignBufferToAtTensor(src_buf, viewed_dest);
  }

  // If the dtype is different or shape is different, then we need to dispatch
  // a StableHLO op.
  const bool is_dest_fp4 = dest.scalar_type() == at::kFloat4_e2m1fn_x2;
  const bool is_src_fp4 = src.scalar_type() == at::kFloat4_e2m1fn_x2;

  if (is_dest_fp4 || is_src_fp4) {
    TT_RET_CHECK(viewed_dest.dim() > 0, error::kInvalidArgument)
        << "expected float4_e2m1fn_x2 tensors to be at least 1-dimensional, "
           "got 0-dimensional";
  }

  TT_ASSIGN_OR_RETURN(const auto out_dtype,
                      ConvertTo<mlir::ElementType>(dest.scalar_type()));

  auto copy_op_builder =
      [out_dtype, is_src_fp4,
       is_dest_fp4](mlir::MlirOp input) -> absl::StatusOr<mlir::MlirOp> {
    auto formatted = input;

    // 1. If source is FP4, it is physically 2N. Slice it to logical shape N
    //    before any broadcasting.
    if (is_src_fp4) {
      TT_ASSIGN_OR_RETURN(formatted, SliceFP4(formatted));
    }

    // 2. Cast to the target element type.
    if (GetElementTypeOrDie(formatted) != out_dtype) {
      formatted = mlir::stablehlo::ConvertElementType(formatted, out_dtype);
    }

    // 3. If destination is FP4, it must be physically 2M. Pad it from logical
    //    shape M to physical shape 2M.
    if (is_dest_fp4) {
      TT_ASSIGN_OR_RETURN(formatted, PadFP4(formatted));
    }

    return formatted;
  };

  Dimensions out_dims = CopyIntVector(viewed_dest.sizes());
  if (is_dest_fp4) {
    out_dims.back() *= 2;
  }

  TT_ASSIGN_OR_RETURN(
      auto new_buf,
      DispatchOp<1>(std::move(copy_op_builder), viewed_src,
                    {.out_dtype = out_dtype,
                     .out_dims = out_dims,
                     .op_param_cache_keys = OpParamCacheKeys::Empty()}),
      _.SetPrepend() << "failed to copy from 'tpu' to 'tpu' device: ");
  return AssignBufferToAtTensor(new_buf, viewed_dest);
}

}  // namespace torch_tpu
