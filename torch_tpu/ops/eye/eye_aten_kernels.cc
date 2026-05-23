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

#include "torch_tpu/ops/eye/eye_aten_kernels.h"

#include <cstdint>
#include <utility>

#include "ATen/core/TensorBody.h"
#include "absl/status/statusor.h"
#include "c10/core/SymInt.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "torch/headeronly/core/ScalarType.h"
#include "torch_tpu/common/cache_key.h"
#include "torch_tpu/common/dtype.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/ops/eye/eye_lib.h"
#include "torch_tpu/ops/macros/kernel.h"
#include "torch_tpu/ops/nullary_aten_kernels.h"
#include "torch_tpu/ops/op_builder_utils.h"
#include "torch_tpu/ops/op_names.h"

namespace torch_tpu {
namespace {

absl::StatusOr<MlirNullaryOpBuilder> GetEyeOpBuilder(
    const int64_t n, const int64_t m, const c10::ScalarType output_dtype) {
  TT_ASSIGN_OR_RETURN(const auto element_type,
                      ConvertTo<mlir::ElementType>(output_dtype));

  return [n, m, element_type](
             mlir::MlirBuilder& builder) -> absl::StatusOr<mlir::MlirOp> {
    return BuildEyeShlo(builder, element_type, n, m);
  };
}
}  // namespace

at::Tensor& AtenEyeMOut(c10::SymInt n, c10::SymInt m, at::Tensor& out) {
  TT_KERNEL(OpName::kEyeMOut, param_keys, (n, m, out), {
    const int64_t n_int = n.guard_int(__FILE__, __LINE__);
    const int64_t m_int = m.guard_int(__FILE__, __LINE__);
    const c10::ScalarType output_dtype = out.scalar_type();

    TT_ASSIGN_OR_THROW(auto op_builder,
                       GetEyeOpBuilder(n_int, m_int, output_dtype));

    TT_THROW_IF_ERROR(ApplyNullaryOpOut(out, std::move(op_builder),
                                        output_dtype, {n_int, m_int},
                                        std::move(param_keys)));
    return out;
  });
}

at::Tensor& AtenEyeOut(c10::SymInt n, at::Tensor& out) {
  TT_KERNEL(OpName::kEyeOut, _, (IgnoreInCacheKey(n, "Legacy usage"), out),
            { return AtenEyeMOut(n, n, out); });
}

}  // namespace torch_tpu
