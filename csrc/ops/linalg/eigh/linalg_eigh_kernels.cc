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

#include "csrc/ops/linalg/eigh/linalg_eigh_kernels.h"

#include <cstdint>
#include <string_view>
#include <tuple>
#include <utility>

#include "ATen/MemoryOverlap.h"
#include "ATen/core/TensorBody.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "c10/core/ScalarType.h"
#include "csrc/common/aten_utils.h"
#include "csrc/common/cache_key.h"
#include "csrc/common/dimension_types.h"
#include "csrc/common/dtype.h"
#include "csrc/common/error_utils.h"
#include "csrc/common/to_string.h"
#include "csrc/common/utils.h"
#include "csrc/eager/op_dispatcher.h"
#include "csrc/ops/linalg/eigh/eigh_builder_kernels.h"
#include "csrc/ops/macros/kernel.h"
#include "csrc/ops/op_builder_utils.h"
#include "csrc/ops/op_names.h"
#include "csrc/ops/resize/resize_aten_kernels.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "torch/headeronly/core/ScalarType.h"

namespace torch_tpu {
namespace {

absl::Status LinalgEighOut(const at::Tensor& self, std::string_view uplo,
                           bool compute_v, at::Tensor& eigenvalues,
                           at::Tensor& eigenvectors,
                           OpParamCacheKeys param_keys) {
  TT_RET_CHECK(self.dim() >= 2, error::kInvalidArgument)
      << "expected input tensor to have at least 2 dimensions, got "
      << self.dim();
  const int64_t m = self.size(-2);
  const int64_t n = self.size(-1);
  TT_RET_CHECK(m == n, error::kInvalidArgument)
      << "expected input tensor's last 2 dimensions to be equal, got [" << m
      << ", " << n << "]";
  const bool lower = uplo == "L" || uplo == "l";
  TT_RET_CHECK(lower || uplo == "U" || uplo == "u", error::kInvalidArgument)
      << "expected UPLO to be 'L' or 'U', got '" << uplo << "'";
  TT_RET_CHECK(eigenvalues.device() == self.device() &&
                   eigenvectors.device() == self.device(),
               error::kInvalidArgument)
      << "expected input and outputs on the same TPU device, got "
      << ToString(self.device()) << ", " << ToString(eigenvalues.device())
      << ", and " << ToString(eigenvectors.device());

  const at::ScalarType vector_dtype = self.scalar_type();
  const at::ScalarType value_dtype = c10::toRealValueType(vector_dtype);
  TT_RETURN_IF_ERROR(ValidateOutDtype(eigenvalues, value_dtype,
                                      /*allow_out_dtype_cast=*/false));
  TT_RETURN_IF_ERROR(ValidateOutDtype(eigenvectors, vector_dtype,
                                      /*allow_out_dtype_cast=*/false));
  const Dimensions value_dims =
      CopyIntVector(self.sizes().slice(0, self.dim() - 1));
  const Dimensions vector_dims =
      compute_v ? CopyIntVector(self.sizes()) : Dimensions{0};
  if (self.numel() == 0) {
    TT_RETURN_IF_ERROR(ResizeTensorIfShapeDiffers(eigenvalues, value_dims));
    return ResizeTensorIfShapeDiffers(eigenvectors, vector_dims);
  }

  // The XLA Jacobi path can discard FP64-scale eigenvalue gaps. Restrict
  // nonempty Eigh to the single-precision types validated on TPU.
  TT_RET_CHECK(vector_dtype == at::kFloat || vector_dtype == at::kComplexFloat,
               error::kInvalidArgument)
      << "expected the input dtype to be float32 or complex64, got "
      << ToString(vector_dtype);
  if (compute_v && eigenvectors.sizes() == self.sizes()) {
    at::assert_no_partial_overlap(eigenvectors, self);
  }

  const EighTriangle triangle =
      lower ? EighTriangle::kLower : EighTriangle::kUpper;
  const auto op_builder =
      [triangle,
       compute_v](mlir::MlirOp input) -> absl::StatusOr<MlirOpResults<2>> {
    TT_ASSIGN_OR_RETURN(MlirOpResults<2> result,
                        BuildEighShlo(input, triangle, EighOrder::kAscending));
    if (!compute_v) {
      TT_ASSIGN_OR_RETURN(
          result[0],
          MakeZeroSizedTensor(input.getBuilder(),
                              GetTensorTypeOrDie(input).getElementType()));
    }
    return MlirOpResults<2>{result[1], result[0]};
  };
  TT_ASSIGN_OR_RETURN(const mlir::ElementType vector_element_type,
                      ConvertTo<mlir::ElementType>(vector_dtype));
  return DispatchOpOut<1, 2>(
      op_builder, self, {eigenvalues, eigenvectors},
      {.out_dtypes = {RealComponentOf(vector_element_type),
                      vector_element_type},
       .out_dims_list = {value_dims, vector_dims},
       .op_param_cache_keys = std::move(param_keys)});
}

}  // namespace

std::tuple<at::Tensor&, at::Tensor&> AtenLinalgEighEigenvalues(
    const at::Tensor& self, std::string_view uplo, bool compute_v,
    at::Tensor& eigenvalues, at::Tensor& eigenvectors) {
  TT_KERNEL(
      OpName::kLinalgEighEigenvalues, param_keys,
      (self, uplo, compute_v, eigenvalues, eigenvectors), {
        TT_THROW_IF_ERROR(LinalgEighOut(self, uplo, compute_v, eigenvalues,
                                        eigenvectors, std::move(param_keys)));
        return {eigenvalues, eigenvectors};
      });
}

}  // namespace torch_tpu
