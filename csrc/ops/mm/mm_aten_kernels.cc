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

#include "torch_tpu/csrc/ops/mm/mm_aten_kernels.h"

#include <cstdint>
#include <optional>
#include <string_view>
#include <utility>

#include "ATen/core/TensorBody.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "c10/core/ScalarType.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"
#include "torch/headeronly/core/ScalarType.h"
#include "torch_tpu/csrc/common/aten_utils.h"
#include "torch_tpu/csrc/common/cache_key.h"
#include "torch_tpu/csrc/common/dtype.h"
#include "torch_tpu/csrc/common/error_utils.h"
#include "torch_tpu/csrc/common/fixed_size_span.h"
#include "torch_tpu/csrc/common/to_string.h"
#include "torch_tpu/csrc/eager/device_buffer.h"
#include "torch_tpu/csrc/eager/op_dispatcher.h"
#include "torch_tpu/csrc/eager/tensor_to_buffer.h"
#include "torch_tpu/csrc/ops/macros/kernel.h"
#include "torch_tpu/csrc/ops/mm/mm.h"
#include "torch_tpu/csrc/ops/op_builder_utils.h"
#include "torch_tpu/csrc/ops/op_names.h"
#include "torch_tpu/csrc/ops/precision_context.h"
#include "torch_tpu/csrc/ops/resize/resize_aten_kernels.h"

namespace torch_tpu {
namespace {

absl::Status CheckMmOutInputs(const at::Tensor& lhs, const at::Tensor& rhs,
                              at::Tensor& out,
                              std::optional<at::ScalarType> out_dtype) {
  // DType checks.
  TT_RET_CHECK(lhs.scalar_type() == rhs.scalar_type(), error::kInvalidArgument)
      << "expected the two arguments to have the same dtype, got "
      << ToString(lhs.scalar_type()) << " vs " << ToString(rhs.scalar_type());

  if (out_dtype.has_value()) {
    auto out_scalar_type = out_dtype.value();
    TT_RET_CHECK(out_scalar_type == out.scalar_type(), error::kInvalidArgument)
        << "expected out_dtype to be the same as the dtype of the out tensor, "
           "got out_dtype "
        << ToString(out_scalar_type) << " vs out tensor dtype "
        << ToString(out.scalar_type());
  } else {
    // For mm, the inputs and output should have the same dtype
    TT_RET_CHECK(out.scalar_type() == lhs.scalar_type(),
                 error::kInvalidArgument)
        << "expected the output to have the same dtype as inputs,"
        << " got out dtype " << ToString(out.scalar_type())
        << " vs inputs dtype " << ToString(lhs.scalar_type());
  }

  // Dimension checks.
  TT_RETURN_IF_ERROR(CheckIsMatrix(lhs, /* arg_name= */ "first"));
  TT_RETURN_IF_ERROR(CheckIsMatrix(rhs, /* arg_name= */ "second"));
  TT_RET_CHECK(lhs.size(1) == rhs.size(0), error::kInvalidArgument)
      << "expected the column size of the first matrix to match the row size "
         "of the second matrix, got shape "
      << ToString(lhs.sizes()) << " vs " << ToString(rhs.sizes()) << " where "
      << lhs.size(1) << " != " << rhs.size(0);

  // Must come after the dtype and shape checks. CUDA validates those criteria
  // before reaching the dispatch that reports the dtype as unimplemented.
  TT_RET_CHECK(
      lhs.numel() == 0 || rhs.numel() == 0 ||
          (lhs.scalar_type() != at::kBool && lhs.scalar_type() != at::kInt),
      error::kPythonNotImplementedError)
      << "not implemented for " << ToString(lhs.scalar_type());

  return absl::OkStatus();
}

absl::StatusOr<DeviceBufferRef> Mm(
    const at::Tensor& lhs, const at::Tensor& rhs, at::Tensor& out,
    OpParamCacheKeys param_keys,
    std::optional<at::ScalarType> out_dtype = std::nullopt) {
  TT_RETURN_IF_ERROR(CheckMmOutInputs(lhs, rhs, out, out_dtype));
  int64_t output_dims[2] = {lhs.size(0), rhs.size(1)};

  // if out_dtype is not specified, use the scalar type of the inputs
  at::ScalarType target_scalar_type =
      out_dtype.has_value() ? out_dtype.value() : lhs.scalar_type();
  TT_ASSIGN_OR_RETURN(mlir::ElementType target_elem_dtype,
                      ConvertTo<mlir::ElementType>(target_scalar_type));

  const auto current_precision = GetAndAddPrecisionTo(param_keys);
  bool needs_conversion = (lhs.scalar_type() != target_scalar_type);

  auto op_builder = [current_precision, target_elem_dtype,
                     needs_conversion](FixedSizeSpan<mlir::MlirOp, 2> inputs)
      -> absl::StatusOr<mlir::MlirOp> {
    mlir::MlirOp lhs_op = inputs[0];
    mlir::MlirOp rhs_op = inputs[1];

    if (needs_conversion) {
      lhs_op = mlir::stablehlo::ConvertElementType(lhs_op, target_elem_dtype);
      rhs_op = mlir::stablehlo::ConvertElementType(rhs_op, target_elem_dtype);
    }

    return BuildMmShlo(lhs_op, rhs_op, current_precision);
  };

  TT_ASSIGN_OR_RETURN(
      auto result_buf,
      DispatchOp<2>(std::move(op_builder), {lhs, rhs},
                    {.out_dtype = target_elem_dtype,
                     .out_dims = output_dims,
                     .op_param_cache_keys = std::move(param_keys)}));
  return result_buf;
}

}  // namespace

at::Tensor AtenMmDtype(const at::Tensor& lhs, const at::Tensor& rhs,
                       at::ScalarType out_dtype) {
  TT_KERNEL(OpName::kMmDtype, param_keys, (lhs, rhs, out_dtype), {
    int64_t output_dims[2] = {lhs.size(0), rhs.size(1)};
    TT_ASSIGN_OR_THROW(at::Tensor out,
                       MakeEmptyTensor(output_dims, out_dtype, lhs.device()));

    TT_ASSIGN_OR_THROW(auto result_buf,
                       Mm(lhs, rhs, out, std::move(param_keys), out_dtype));
    TT_THROW_IF_ERROR(AssignBufferToAtTensor(std::move(result_buf), out));
    return out;
  });
}

at::Tensor& AtenMmDtypeOut(const at::Tensor& lhs, const at::Tensor& rhs,
                           at::ScalarType out_dtype, at::Tensor& out) {
  TT_KERNEL(OpName::kMmDtypeOut, param_keys, (lhs, rhs, out_dtype, out), {
    TT_ASSIGN_OR_THROW(auto result_buf,
                       Mm(lhs, rhs, out, std::move(param_keys), out_dtype));
    int64_t output_dims[2] = {lhs.size(0), rhs.size(1)};
    TT_THROW_IF_ERROR(ResizeTensorIfShapeDiffers(out, output_dims));
    TT_THROW_IF_ERROR(AssignBufferToAtTensor(std::move(result_buf), out));
    return out;
  });
}

at::Tensor& AtenMmOut(const at::Tensor& lhs, const at::Tensor& rhs,
                      at::Tensor& out) {
  TT_KERNEL(OpName::kMmOut, param_keys, (lhs, rhs, out), {
    TT_ASSIGN_OR_THROW(auto result_buf,
                       Mm(lhs, rhs, out, std::move(param_keys)));
    int64_t output_dims[2] = {lhs.size(0), rhs.size(1)};
    TT_THROW_IF_ERROR(ResizeTensorIfShapeDiffers(out, output_dims));
    TT_THROW_IF_ERROR(AssignBufferToAtTensor(std::move(result_buf), out));
    return out;
  });
}

}  // namespace torch_tpu
