/*
 * Copyright 2026 Google LLC
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

// ANY_OP_IMPL_FILENAME_OK=shared validation helpers used across multiple
// upsample *_aten_kernels files

#include "csrc/ops/upsample/upsample_utils.h"

#include <cstdint>
#include <string_view>

#include "ATen/core/ATen_fwd.h"
#include "ATen/core/TensorBody.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "c10/core/ScalarType.h"
#include "csrc/common/dimension_types.h"
#include "csrc/common/error_utils.h"
#include "csrc/common/to_string.h"
#include "csrc/common/utils.h"
#include "csrc/ops/op_builder_utils.h"
#include "mlir/IR/BuiltinTypes.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "torch/headeronly/core/ScalarType.h"

namespace torch_tpu {
namespace {

absl::Status ValidateUpsampleDtype(const at::Tensor& tensor,
                                   const UpsampleDtypePolicy dtype_policy) {
  const at::ScalarType dtype = tensor.scalar_type();
  const bool supported =
      (dtype_policy == UpsampleDtypePolicy::kFloatingOnly)
          ? at::isFloatingType(dtype)
          : (dtype != at::kBool && dtype != at::kInt && dtype != at::kLong);
  TT_RET_CHECK(supported, error::kPythonNotImplementedError)
      << "not implemented for " << ToString(dtype);
  return absl::OkStatus();
}

bool AreSpatialSizesPositive(const at::IntArrayRef input_sizes,
                             const at::IntArrayRef output_size,
                             const int64_t spatial_dims) {
  for (int64_t i = 0; i < spatial_dims; ++i) {
    if (input_sizes[i + 2] <= 0 || output_size[i] <= 0) {
      return false;
    }
  }
  return true;
}

absl::Status ValidateUpsampleOutputDevice(const at::Tensor& input,
                                          const at::Tensor& output,
                                          const std::string_view output_name) {
  TT_RET_CHECK(output.device() == input.device(), error::kInvalidArgument)
      << "expected " << output_name << " device " << ToString(input.device())
      << ", got " << ToString(output.device());
  return absl::OkStatus();
}

}  // namespace

absl::StatusOr<Dimensions> ValidateUpsampleForwardInputs(
    const at::Tensor& self, const at::IntArrayRef output_size,
    const at::Tensor& out, const int64_t spatial_dims,
    const UpsampleDtypePolicy dtype_policy) {
  TT_RET_CHECK(static_cast<int64_t>(output_size.size()) == spatial_dims,
               error::kInvalidArgument)
      << "expected output_size to have "
      << FormatCount(spatial_dims, "element", "elements") << ", got "
      << ToString(output_size.size());

  const int64_t expected_rank = spatial_dims + 2;
  TT_RET_CHECK(self.dim() == expected_rank, error::kInvalidArgument)
      << "expected " << ToString(expected_rank) << "D input tensor, got "
      << ToString(self.dim()) << "D tensor";

  TT_RET_CHECK(AreSpatialSizesPositive(self.sizes(), output_size, spatial_dims),
               error::kInvalidArgument)
      << "expected input and output spatial sizes to be greater than 0, got "
         "input: "
      << ToString(self.sizes()) << ", output_size: " << ToString(output_size);

  TT_RET_CHECK(self.size(1) != 0, error::kInvalidArgument)
      << "expected non-empty channel dimension, got input: "
      << ToString(self.sizes());

  TT_RETURN_IF_ERROR(ValidateUpsampleDtype(self, dtype_policy));

  TT_RET_CHECK(self.scalar_type() == out.scalar_type(), error::kInvalidArgument)
      << "expected out dtype " << ToString(self.scalar_type()) << ", got "
      << ToString(out.scalar_type());

  TT_RETURN_IF_ERROR(ValidateUpsampleOutputDevice(self, out, "out"));

  Dimensions expected_output_shape;
  expected_output_shape.reserve(expected_rank);
  expected_output_shape.push_back(self.size(0));
  expected_output_shape.push_back(self.size(1));
  for (int64_t i = 0; i < spatial_dims; ++i) {
    expected_output_shape.push_back(output_size[i]);
  }
  return expected_output_shape;
}

absl::StatusOr<Dimensions> ValidateUpsampleBackwardInputs(
    const at::Tensor& grad_output, const at::IntArrayRef output_size,
    const at::IntArrayRef input_size, const at::Tensor& grad_input,
    const int64_t spatial_dims, const UpsampleDtypePolicy dtype_policy) {
  TT_RET_CHECK(static_cast<int64_t>(output_size.size()) == spatial_dims,
               error::kInvalidArgument)
      << "expected output_size to have "
      << FormatCount(spatial_dims, "element", "elements") << ", got "
      << ToString(output_size.size());

  const int64_t expected_rank = spatial_dims + 2;
  TT_RET_CHECK(static_cast<int64_t>(input_size.size()) == expected_rank,
               error::kInvalidArgument)
      << "expected input_size to have " << ToString(expected_rank)
      << " elements, got " << ToString(input_size.size());

  TT_RET_CHECK(AreSpatialSizesPositive(input_size, output_size, spatial_dims),
               error::kInvalidArgument)
      << "expected input and output spatial sizes to be greater than 0, got "
         "input_size: "
      << ToString(input_size) << ", output_size: " << ToString(output_size);

  TT_RET_CHECK(grad_output.dim() == expected_rank, error::kInvalidArgument)
      << "expected " << ToString(expected_rank) << "D grad_output tensor, got "
      << ToString(grad_output.dim()) << "D tensor";

  Dimensions expected_grad_output_shape;
  expected_grad_output_shape.reserve(expected_rank);
  expected_grad_output_shape.push_back(input_size[0]);
  expected_grad_output_shape.push_back(input_size[1]);
  for (int64_t i = 0; i < spatial_dims; ++i) {
    expected_grad_output_shape.push_back(output_size[i]);
  }
  TT_RET_CHECK(
      grad_output.sizes() == AsArrayRef<int64_t>(expected_grad_output_shape),
      error::kInvalidArgument)
      << "expected grad_output to have size "
      << ToString(expected_grad_output_shape) << ", got "
      << ToString(grad_output.sizes());

  TT_RETURN_IF_ERROR(ValidateUpsampleDtype(grad_output, dtype_policy));

  TT_RET_CHECK(grad_output.scalar_type() == grad_input.scalar_type(),
               error::kInvalidArgument)
      << "expected grad_input dtype " << ToString(grad_output.scalar_type())
      << ", got " << ToString(grad_input.scalar_type());

  TT_RETURN_IF_ERROR(
      ValidateUpsampleOutputDevice(grad_output, grad_input, "grad_input"));

  return CopyIntVector(input_size);
}

absl::Status ValidateUpsampleSpatialDimsStatic(mlir::MlirOp operand) {
  const mlir::RankedTensorType type = GetTensorTypeOrDie(operand);
  for (int64_t dim = 2; dim < type.getRank(); ++dim) {
    TT_RET_CHECK(!type.isDynamicDim(dim), error::kPythonNotImplementedError)
        << "expected spatial dimension " << ToString(dim)
        << " to have a static size, got dynamic size";
  }
  return absl::OkStatus();
}

}  // namespace torch_tpu
