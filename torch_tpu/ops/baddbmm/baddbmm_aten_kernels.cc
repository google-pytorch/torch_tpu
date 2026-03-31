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

#include "torch_tpu/ops/baddbmm/baddbmm_aten_kernels.h"

#include <cstdint>
#include <utility>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "mlir/IR/BuiltinTypes.h"
#include "ATen/core/ATen_fwd.h"
#include "ATen/core/TensorBody.h"
#include "ATen/native/Resize.h"
#include "torch/headeronly/core/ScalarType.h"
#include "torch_tpu/common/cache_key.h"
#include "torch_tpu/common/dtype.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/common/fixed_size_span.h"
#include "torch_tpu/common/shape.h"
#include "torch_tpu/common/to_string.h"
#include "torch_tpu/eager/device_buffer.h"
#include "torch_tpu/eager/op_dispatcher.h"
#include "torch_tpu/eager/tensor_to_buffer.h"
#include "torch_tpu/ops/baddbmm/baddbmm.h"
#include "torch_tpu/ops/macros/kernel.h"
#include "torch_tpu/ops/op_builder_utils.h"
#include "torch_tpu/ops/op_names.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"

namespace torch_tpu {

namespace {

absl::StatusOr<DeviceBufferRef> Baddbmm(
    const OpName op_name, const at::Tensor& self, const at::Tensor& batch1,
    const at::Tensor& batch2, const at::Tensor& beta, const at::Tensor& alpha,
    at::ScalarType out_dtype, OpParamCacheKeys param_keys) {
  TT_ASSIGN_OR_RETURN(const auto out_mlir_dtype,
                      ConvertTo<mlir::ElementType>(out_dtype));

  auto op_builder = [out_mlir_dtype](FixedSizeSpan<mlir::MlirOp, 5> inputs) {
    auto& [self, batch1, batch2, beta, alpha] = inputs;
    return BuildBaddbmmShlo(self, batch1, batch2, beta, alpha, out_mlir_dtype);
  };

  const Dimensions out_dims = {batch1.size(0), batch1.size(1), batch2.size(2)};

  return DispatchOp<5>(op_name, std::move(op_builder),
                       {self, batch1, batch2, beta, alpha},
                       {.out_dtype = out_mlir_dtype,
                        .out_dims = out_dims,
                        .op_param_cache_keys = std::move(param_keys)});
}

}  // namespace

at::Tensor AtenBaddbmmDtype(const at::Tensor& self, const at::Tensor& batch1,
                            const at::Tensor& batch2, at::ScalarType out_dtype,
                            const at::Scalar& beta, const at::Scalar& alpha) {
  TT_KERNEL(OpName::kBaddbmmDtype, param_keys,
            (self, batch1, batch2, IgnoreInCacheKey(out_dtype),
             IgnoreInCacheKey(beta), IgnoreInCacheKey(alpha)),
            {
              TT_ASSIGN_OR_THROW(at::Tensor beta_tensor,
                                 MakeTensor(beta, self.scalar_type()));
              TT_ASSIGN_OR_THROW(at::Tensor alpha_tensor,
                                 MakeTensor(alpha, self.scalar_type()));
              TT_ASSIGN_OR_THROW(auto result_buffer,
                                 Baddbmm(OpName::kBaddbmmDtype, self, batch1,
                                         batch2, beta_tensor, alpha_tensor,
                                         out_dtype, std::move(param_keys)));
              return MakeTensor(std::move(result_buffer));
            });
}

at::Tensor& AtenBaddbmmDtypeOut(const at::Tensor& self,
                                const at::Tensor& batch1,
                                const at::Tensor& batch2,
                                at::ScalarType out_dtype,
                                const at::Scalar& beta, const at::Scalar& alpha,
                                at::Tensor& out) {
  TT_KERNEL(OpName::kBaddbmmDtypeOut, param_keys,
            (self, batch1, batch2, IgnoreInCacheKey(out_dtype),
             IgnoreInCacheKey(beta), IgnoreInCacheKey(alpha), out),
            {
              TT_ASSIGN_OR_THROW(at::Tensor beta_tensor,
                                 MakeTensor(beta, self.scalar_type()));
              TT_ASSIGN_OR_THROW(at::Tensor alpha_tensor,
                                 MakeTensor(alpha, self.scalar_type()));
              TT_ASSIGN_OR_THROW(auto result_buffer,
                                 Baddbmm(OpName::kBaddbmmDtypeOut, self, batch1,
                                         batch2, beta_tensor, alpha_tensor,
                                         out_dtype, std::move(param_keys)));
              TT_THROW_IF_ERROR(
                  AssignBufferToAtTensor(std::move(result_buffer), out));
              return out;
            });
}

at::Tensor& AtenBaddbmmOut(const at::Tensor& self, const at::Tensor& batch1,
                           const at::Tensor& batch2, const at::Scalar& beta,
                           const at::Scalar& alpha, at::Tensor& out) {
  TT_KERNEL(
      OpName::kBaddbmmOut, param_keys,
      (self, batch1, batch2, IgnoreInCacheKey(beta), IgnoreInCacheKey(alpha),
       out),
      {
        TT_ASSIGN_OR_THROW(at::Tensor beta_tensor,
                           MakeTensor(beta, self.scalar_type()));
        TT_ASSIGN_OR_THROW(at::Tensor alpha_tensor,
                           MakeTensor(alpha, self.scalar_type()));
        TT_ASSIGN_OR_THROW(
            auto result_buffer,
            Baddbmm(OpName::kBaddbmmOut, self, batch1, batch2, beta_tensor,
                    alpha_tensor, out.scalar_type(), std::move(param_keys)));
        TT_THROW_IF_ERROR(
            AssignBufferToAtTensor(std::move(result_buffer), out));
        return out;
      });
}

}  // namespace torch_tpu
