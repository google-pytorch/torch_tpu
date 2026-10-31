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

#include "csrc/ops/logical/logical_aten_kernels.h"

#include <utility>

#include "ATen/core/ATen_fwd.h"
#include "absl/status/status.h"
#include "c10/core/ScalarType.h"
#include "csrc/common/cache_key.h"
#include "csrc/common/dtype.h"
#include "csrc/common/error_utils.h"
#include "csrc/ops/binary_aten_kernels.h"
#include "csrc/ops/logical/logical.h"
#include "csrc/ops/macros/kernel.h"
#include "csrc/ops/op_builder_utils.h"
#include "csrc/ops/op_names.h"
#include "csrc/ops/unary_aten_kernels.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"

namespace torch_tpu {

namespace {

// This helper function for binary logical operations validates output dtype
// against kBool and dispatches via BinaryOpOut.
absl::Status LogicalBinaryOutImpl(const at::Tensor& self,
                                  const at::Tensor& other, at::Tensor& out,
                                  MlirBinaryOpBuilder core_op_builder) {
  TT_ASSIGN_OR_RETURN(const auto bool_dtype,
                      ConvertTo<mlir::ElementType>(at::kBool));
  return BinaryOpOut(self, other, out, std::move(core_op_builder),
                     {.op_param_cache_keys = OpParamCacheKeys::Empty(),
                      .result_dtype = bool_dtype});
}

// This helper function for unary logical operations validates output dtype
// against kBool and dispatches via UnaryOpOut.
absl::Status LogicalUnaryOutImpl(const at::Tensor& self, at::Tensor& out,
                                 MlirUnaryOpBuilder core_op_builder) {
  TT_ASSIGN_OR_RETURN(const auto bool_dtype,
                      ConvertTo<mlir::ElementType>(at::kBool));
  return UnaryOpOut(self, out, std::move(core_op_builder),
                    {.op_param_cache_keys = OpParamCacheKeys::Empty(),
                     .out_dtype = bool_dtype});
}

}  // namespace

at::Tensor& AtenLogicalAndOut(const at::Tensor& self, const at::Tensor& other,
                              at::Tensor& out) {
  TT_KERNEL(OpName::kLogicalAndOut, _, (self, other, out), {
    TT_THROW_IF_ERROR(
        LogicalBinaryOutImpl(self, other, out, BuildLogicalAndShlo));
    return out;
  });
}

at::Tensor& AtenLogicalOrOut(const at::Tensor& self, const at::Tensor& other,
                             at::Tensor& out) {
  TT_KERNEL(OpName::kLogicalOrOut, _, (self, other, out), {
    TT_THROW_IF_ERROR(
        LogicalBinaryOutImpl(self, other, out, BuildLogicalOrShlo));
    return out;
  });
}

at::Tensor& AtenLogicalXorOut(const at::Tensor& self, const at::Tensor& other,
                              at::Tensor& out) {
  TT_KERNEL(OpName::kLogicalXorOut, _, (self, other, out), {
    TT_THROW_IF_ERROR(
        LogicalBinaryOutImpl(self, other, out, BuildLogicalXorShlo));
    return out;
  });
}

at::Tensor& AtenLogicalNotOut(const at::Tensor& self, at::Tensor& out) {
  TT_KERNEL(OpName::kLogicalNotOut, _, (self, out), {
    TT_THROW_IF_ERROR(LogicalUnaryOutImpl(self, out, BuildLogicalNotShlo));
    return out;
  });
}

}  // namespace torch_tpu
