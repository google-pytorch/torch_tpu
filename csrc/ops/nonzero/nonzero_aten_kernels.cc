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

#include "csrc/ops/nonzero/nonzero_aten_kernels.h"

#include <cstdint>
#include <optional>
#include <utility>

#include "ATen/MemoryOverlap.h"
#include "ATen/core/ATen_fwd.h"
#include "ATen/core/ScalarType.h"
#include "ATen/core/TensorBody.h"
#include "absl/functional/bind_front.h"
#include "absl/log/absl_log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "c10/core/ScalarType.h"
#include "c10/core/SymInt.h"
#include "csrc/common/aten_utils.h"
#include "csrc/common/cache_key.h"
#include "csrc/common/device_type.h"
#include "csrc/common/dimension_types.h"
#include "csrc/common/error_utils.h"
#include "csrc/ops/macros/kernel.h"
#include "csrc/ops/nonzero/nonzero.h"
#include "csrc/ops/nullary_aten_kernels.h"
#include "csrc/ops/op_builder_utils.h"
#include "csrc/ops/op_names.h"
#include "csrc/ops/resize/resize_aten_kernels.h"
#include "csrc/ops/unary_aten_kernels.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"

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
  Dimensions out_dims = {num_non_zero_elements, self.dim()};
  TT_ASSIGN_OR_RETURN(
      auto result,
      UnaryOp(self, absl::bind_front(BuildNonzeroShlo, num_non_zero_elements),
              {.op_param_cache_keys = OpParamCacheKeys::Empty(),
               .out_dtype = mlir::ElementType::I64,
               .out_dims = std::move(out_dims)}));
  return result;
}

absl::Status NonzeroOut(const at::Tensor& self, at::Tensor& out,
                        const int64_t num_non_zero_elements) {
  Dimensions out_dims = {num_non_zero_elements, self.dim()};
  return UnaryOpOut(self, out,
                    absl::bind_front(BuildNonzeroShlo, num_non_zero_elements),
                    {.op_param_cache_keys = OpParamCacheKeys::Empty(),
                     .out_dtype = mlir::ElementType::I64,
                     .out_dims = std::move(out_dims),
                     .allow_out_dtype_cast = false});
}

absl::StatusOr<at::Tensor> AtenNonzeroHelper(const at::Tensor& self) {
  // Get number of non-zero elements in the tensor as a tensor of size 1.
  TT_ASSIGN_OR_RETURN(auto num_non_zero_elements, GetNumNonZeroElements(self));
  // Move `num_non_zero_elements` to the CPU. This transition is necessary to
  // dispatch the nonzero op. Unlike cpu(), item() invalidates a graph capture,
  // as nonzero does on CUDA.
  const int64_t num_non_zero_elements_value =
      num_non_zero_elements.item<int64_t>();
  ABSL_VLOG(3) << "[AtenNonzeroHelper] num_non_zero_elements: "
               << num_non_zero_elements_value;

  // No nonzeros: return an empty (0, rank) result rather than compiling an
  // index graph per input shape.
  if (num_non_zero_elements_value == 0) {
    return AtenEfficientZeroTensor({0, self.dim()}, at::kLong, std::nullopt,
                                   self.device(), std::nullopt);
  }

  // Get the indices of the non-zero elements in the tensor as a tensor of
  // size (num_non_zero_elements, rank_of_input_tensor).
  TT_ASSIGN_OR_RETURN(auto result, Nonzero(self, num_non_zero_elements_value));
  return result;
}

absl::Status ValidateNonzeroOutTensor(const at::Tensor& self,
                                      const at::Tensor& out) {
  at::assert_no_internal_overlap(out);
  at::assert_no_overlap(out, self);
  TT_RET_CHECK(IsPrivateUse1Device(out), error::kInvalidArgument)
      << "expected the output tensor to be on "
      << GetPrivateUse1DeviceDebugName() << ", got " << out.device();
  return ValidateOutDtype(out, at::kLong, /*allow_out_dtype_cast=*/false);
}

absl::Status ValidateNonzeroStaticSize(const int64_t size_int) {
  TT_RET_CHECK(size_int >= 0, error::kInvalidArgument)
      << "expected 'size' to be >= 0, got " << size_int;
  return absl::OkStatus();
}

absl::Status AtenNonzeroOutHelper(const at::Tensor& self, at::Tensor& out) {
  TT_RETURN_IF_ERROR(ValidateNonzeroOutTensor(self, out));

  // Get number of non-zero elements in the tensor as a tensor of size 1.
  TT_ASSIGN_OR_RETURN(auto num_non_zero_elements, GetNumNonZeroElements(self));
  // Move `num_non_zero_elements` to the CPU. This transition is necessary to
  // dispatch the nonzero op. Unlike cpu(), item() invalidates a graph capture,
  // as nonzero does on CUDA.
  const int64_t num_non_zero_elements_value =
      num_non_zero_elements.item<int64_t>();
  ABSL_VLOG(3) << "[AtenNonzeroOutHelper] num_non_zero_elements: "
               << num_non_zero_elements_value;

  // No nonzeros: resize `out` to (0, rank) rather than compiling an index graph
  // per input shape.
  if (num_non_zero_elements_value == 0) {
    return ResizeTensorIfShapeDiffers(out, {0, self.dim()});
  }

  // Get the indices of the non-zero elements in the tensor as a tensor of
  // size (num_non_zero_elements, rank_of_input_tensor).
  return NonzeroOut(self, out, num_non_zero_elements_value);
}

absl::Status AtenNonzeroStaticOutHelper(const at::Tensor& self,
                                        const int64_t size_int,
                                        const int64_t fill_value,
                                        OpParamCacheKeys param_keys,
                                        at::Tensor& out) {
  TT_RETURN_IF_ERROR(ValidateNonzeroStaticSize(size_int));
  TT_RETURN_IF_ERROR(ValidateNonzeroOutTensor(self, out));
  const Dimensions out_dims = {size_int, self.dim()};
  return UnaryOpOut(
      self, out, absl::bind_front(BuildNonzeroStaticShlo, size_int, fill_value),
      {.op_param_cache_keys = std::move(param_keys),
       .out_dtype = mlir::ElementType::I64,
       .out_dims = out_dims,
       .allow_out_dtype_cast = false});
}

absl::StatusOr<at::Tensor> AtenNonzeroStaticHelper(
    const at::Tensor& self, const int64_t size_int, const int64_t fill_value,
    OpParamCacheKeys param_keys) {
  TT_RETURN_IF_ERROR(ValidateNonzeroStaticSize(size_int));
  const Dimensions out_dims = {size_int, self.dim()};
  return UnaryOp(self,
                 absl::bind_front(BuildNonzeroStaticShlo, size_int, fill_value),
                 {.op_param_cache_keys = std::move(param_keys),
                  .out_dtype = mlir::ElementType::I64,
                  .out_dims = out_dims});
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

at::Tensor AtenNonzeroStatic(const at::Tensor& self, c10::SymInt size,
                             int64_t fill_value) {
  TT_KERNEL(OpName::kNonzeroStatic, param_keys, (self, size, fill_value), {
    const int64_t size_int = size.expect_int();
    TT_ASSIGN_OR_THROW(at::Tensor result,
                       AtenNonzeroStaticHelper(self, size_int, fill_value,
                                               std::move(param_keys)));
    return result;
  });
}

at::Tensor& AtenNonzeroStaticOut(const at::Tensor& self, c10::SymInt size,
                                 int64_t fill_value, at::Tensor& out) {
  TT_KERNEL(OpName::kNonzeroStaticOut, param_keys,
            (self, size, fill_value, out), {
              const int64_t size_int = size.expect_int();
              TT_THROW_IF_ERROR(AtenNonzeroStaticOutHelper(
                  self, size_int, fill_value, std::move(param_keys), out));
              return out;
            });
}

}  // namespace torch_tpu
