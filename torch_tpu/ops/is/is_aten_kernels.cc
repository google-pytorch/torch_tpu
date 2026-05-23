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

#include "torch_tpu/ops/is/is_aten_kernels.h"

#include "ATen/core/TensorBody.h"
#include "c10/core/ScalarType.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "torch_tpu/common/cache_key.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/ops/is/is.h"
#include "torch_tpu/ops/macros/kernel.h"
#include "torch_tpu/ops/op_names.h"
#include "torch_tpu/ops/unary_aten_kernels.h"

namespace torch_tpu {

at::Tensor AtenIsNan(const at::Tensor& self) {
  TT_KERNEL(OpName::kIsNan, _, (self), {
    TT_ASSIGN_OR_THROW(
        auto result,
        ::torch_tpu::UnaryOp(self, BuildIsNanShlo,
                             {.op_param_cache_keys = OpParamCacheKeys::Empty(),
                              .out_dtype = mlir::ElementType::PRED}));
    return result;
  });
}

at::Tensor& AtenIsNegInfOut(const at::Tensor& self, at::Tensor& out) {
  TT_KERNEL(OpName::kIsNegInfOut, _, (self, out), {
    TT_THROW_IF_ERROR(::torch_tpu::UnaryOpOut(
        self, out, BuildIsNegInfShlo,
        {.op_param_cache_keys = OpParamCacheKeys::Empty(),
         .out_dtype = mlir::ElementType::PRED}));
    return out;
  });
}

at::Tensor& AtenIsPosInfOut(const at::Tensor& self, at::Tensor& out) {
  TT_KERNEL(OpName::kIsPosInfOut, _, (self, out), {
    TT_THROW_IF_ERROR(::torch_tpu::UnaryOpOut(
        self, out, BuildIsPosInfShlo,
        {.op_param_cache_keys = OpParamCacheKeys::Empty(),
         .out_dtype = mlir::ElementType::PRED}));
    return out;
  });
}

}  // namespace torch_tpu
