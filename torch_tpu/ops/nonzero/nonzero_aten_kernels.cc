// Copyright 2025 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "torch_tpu/ops/nonzero/nonzero_aten_kernels.h"

#include <cstdint>
#include <utility>

#include "ATen/core/ATen_fwd.h"
#include "ATen/core/ScalarType.h"
#include "ATen/core/TensorBody.h"
#include "absl/functional/bind_front.h"
#include "absl/log/absl_log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "torch/headeronly/core/ScalarType.h"
#include "torch_tpu/common/cache_key.h"
#include "torch_tpu/common/dimension_types.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/ops/macros/kernel.h"
#include "torch_tpu/ops/nonzero/nonzero.h"
#include "torch_tpu/ops/op_builder_utils.h"
#include "torch_tpu/ops/op_names.h"
#include "torch_tpu/ops/unary_aten_kernels.h"

namespace torch_tpu {

namespace {

absl::StatusOr<at::Tensor> GetNumNonZeroElements(const at::Tensor& self) {
  TT_ASSIGN_OR_RETURN(auto num_non_zero_elements,
                      UnaryOp(self,
                              absl::bind_front(BuildNonzeroGetOutputSizeShlo,
                                               mlir::ElementType::I64),
                              // Override the OpName passed to TT_KERNEL()
                              // as this is not for implementing a specific op,
                              // but rather a shared utility.
                              {.op_name = OpName::kNonzeroSize,
                               .op_param_cache_keys = OpParamCacheKeys::Empty(),
                               .out_dtype = mlir::ElementType::I64,
                               .out_dims = Dimensions()}));
  return num_non_zero_elements;
}

absl::StatusOr<at::Tensor> Nonzero(const at::Tensor& self,
                                   const int64_t num_non_zero_elements) {
  TT_ASSIGN_OR_RETURN(  // ERROR_COV_INFEASIBLE=the op creates cache keys
                        // successfully.
      auto param_keys, TT_MAKE_OP_PARAM_CACHE_KEYS(num_non_zero_elements));

  Dimensions out_dims = {num_non_zero_elements, self.dim()};
  TT_ASSIGN_OR_RETURN(
      auto result,
      UnaryOp(self, absl::bind_front(BuildNonzeroShlo, num_non_zero_elements),
              {.op_param_cache_keys = std::move(param_keys),
               .out_dtype = mlir::ElementType::I64,
               .out_dims = std::move(out_dims)}));
  return result;
}

absl::Status NonzeroOut(const at::Tensor& self, at::Tensor& out,
                        const int64_t num_non_zero_elements) {
  TT_ASSIGN_OR_RETURN(  // ERROR_COV_INFEASIBLE=the op creates cache keys
                        // successfully.
      auto param_keys, TT_MAKE_OP_PARAM_CACHE_KEYS(num_non_zero_elements));

  Dimensions out_dims = {num_non_zero_elements, self.dim()};
  return UnaryOpOut(self, out,
                    absl::bind_front(BuildNonzeroShlo, num_non_zero_elements),
                    {.op_param_cache_keys = std::move(param_keys),
                     .out_dtype = mlir::ElementType::I64,
                     .out_dims = std::move(out_dims)});
}

absl::StatusOr<at::Tensor> AtenNonzeroHelper(const at::Tensor& self) {
  // Get number of non-zero elements in the tensor as a tensor of size 1.
  TT_ASSIGN_OR_RETURN(auto num_non_zero_elements, GetNumNonZeroElements(self));
  // Move `num_non_zero_elements` to the CPU. This transition is necessary to
  // dispatch the nonzero op.
  const int64_t num_non_zero_elements_value =
      num_non_zero_elements.cpu().item<int64_t>();
  ABSL_VLOG(3) << "[AtenNonzeroHelper] num_non_zero_elements: "
               << num_non_zero_elements_value;

  // Get the indices of the non-zero elements in the tensor as a tensor of
  // size (num_non_zero_elements, rank_of_input_tensor).
  TT_ASSIGN_OR_RETURN(auto result, Nonzero(self, num_non_zero_elements_value));
  return result;
}

absl::Status AtenNonzeroOutHelper(const at::Tensor& self, at::Tensor& out) {
  // Get number of non-zero elements in the tensor as a tensor of size 1.
  TT_ASSIGN_OR_RETURN(auto num_non_zero_elements, GetNumNonZeroElements(self));
  // Move `num_non_zero_elements` to the CPU. This transition is necessary to
  // dispatch the nonzero op.
  const int64_t num_non_zero_elements_value =
      num_non_zero_elements.cpu().item<int64_t>();
  ABSL_VLOG(3) << "[AtenNonzeroOutHelper] num_non_zero_elements: "
               << num_non_zero_elements_value;

  // Get the indices of the non-zero elements in the tensor as a tensor of
  // size (num_non_zero_elements, rank_of_input_tensor).
  return NonzeroOut(self, out, num_non_zero_elements_value);
}

}  // namespace

at::Tensor AtenNonzero(const at::Tensor& self) {
  TT_KERNEL(OpName::kNonzero, _, (self), {
    TT_ASSIGN_OR_THROW(at::Tensor result, AtenNonzeroHelper(self));
    return result;
  });
}

at::Tensor& AtenNonzeroOut(const at::Tensor& self, at::Tensor& out) {
  TT_KERNEL(OpName::kNonzeroOut, _, (self, out), {
    TT_THROW_IF_ERROR(AtenNonzeroOutHelper(self, out));
    return out;
  });
}

}  // namespace torch_tpu
