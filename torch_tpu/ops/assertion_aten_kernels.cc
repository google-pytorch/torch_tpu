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

#include "torch_tpu/ops/assertion_aten_kernels.h"

#include <string_view>
#include <utility>

#include "ATen/core/TensorBody.h"
#include "absl/log/log.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "torch_tpu/common/cache_key.h"
#include "torch_tpu/common/dtype.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/eager/op_dispatcher.h"
#include "torch_tpu/ops/macros/kernel.h"
#include "torch_tpu/ops/op_builder_utils.h"
#include "torch_tpu/ops/op_names.h"

namespace torch_tpu {

// TODO(b/510886286): This is a temporary no-op stub.
// Implement a proper async check using a parallel checker program or
// device-side assertions to avoid blocking the host.
void AtenAssertAsync(const at::Tensor& self) {
  TT_KERNEL(OpName::kAssertAsync, _, (self), {
    auto op_builder =
        [](mlir::MlirOp input_op) -> absl::StatusOr<mlir::MlirOp> {
      return input_op;
    };

    TT_ASSIGN_OR_THROW(auto out_dtype,
                       ConvertTo<mlir::ElementType>(self.scalar_type()));

    auto status_or_buf =
        DispatchOp<1>(op_builder, self,
                      /*options=*/
                      {.out_dtype = out_dtype,
                       .out_dims = self.sizes(),
                       .op_param_cache_keys = OpParamCacheKeys::Empty()});

    TT_THROW_IF_ERROR(status_or_buf.status());
  });
}

// TODO(b/510886286): This is a temporary no-op stub.
// Implement a proper async check using a parallel checker program or
// device-side assertions to avoid blocking the host.
void AtenAssertAsyncMsg(const at::Tensor& self, std::string_view assert_msg) {
  TT_KERNEL(
      OpName::kAssertAsyncMsg, param_keys,
      (self, IgnoreInCacheKey(assert_msg, "message does not affect TPU graph")),
      {
        auto op_builder =
            [](mlir::MlirOp input_op) -> absl::StatusOr<mlir::MlirOp> {
          return input_op;
        };

        TT_ASSIGN_OR_THROW(auto out_dtype,
                           ConvertTo<mlir::ElementType>(self.scalar_type()));

        auto status_or_buf =
            DispatchOp<1>(op_builder, self,
                          /*options=*/
                          {.out_dtype = out_dtype,
                           .out_dims = self.sizes(),
                           .op_param_cache_keys = std::move(param_keys)});

        TT_THROW_IF_ERROR(status_or_buf.status());
      });
}

}  // namespace torch_tpu
