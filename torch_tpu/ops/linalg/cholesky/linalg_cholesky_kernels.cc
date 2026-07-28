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
#include "torch_tpu/ops/linalg/cholesky/linalg_cholesky_kernels.h"

#include "ATen/core/TensorBody.h"
#include "absl/status/status.h"
#include "c10/core/ScalarType.h"
#include "mlir/IR/BuiltinTypes.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "torch_tpu/common/cache_key.h"
#include "torch_tpu/common/dimension_types.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/common/utils.h"
#include "torch_tpu/eager/op_dispatcher.h"
#include "torch_tpu/eager/tensor_to_buffer.h"
#include "torch_tpu/ops/macros/kernel.h"
#include "torch_tpu/ops/op_builder_utils.h"
#include "torch_tpu/ops/op_names.h"
#include "torch_tpu/ops/resize/resize_aten_kernels.h"

namespace torch_tpu {

namespace {

auto LinalgCholeskyBuilder(bool lower) {
  return [lower](mlir::MlirOp input) -> absl::StatusOr<mlir::MlirOp> {
    auto& builder = input.getBuilder();
    auto& op_builder = builder.getOpBuilder();

    auto tensor_type = GetTensorTypeOrDie(input);
    const int rank = tensor_type.getShape().size();
    TT_RET_CHECK(rank >= 2, error::kInvalidArgument)
        << "input must have at least 2 dimensions, got " << rank;

    auto lower_attr =
        op_builder.getNamedAttr("lower", op_builder.getBoolAttr(lower));
    auto cholesky = mlir::stablehlo::CholeskyOp::create(
        op_builder, input.getValue().getLoc(), tensor_type, {input.getValue()},
        {lower_attr});

    return mlir::MlirOp(builder, cholesky.getResult());
  };
}

}  // namespace

at::Tensor& AtenLinalgCholeskyOut(const at::Tensor& self, bool upper,
                                  at::Tensor& out) {
  TT_KERNEL(
      OpName::kLinalgCholeskyOut, param_keys,
      (self, IgnoreInCacheKey(upper, "Doesn't affect SHLO type"), out), {
        TT_CHECK_THROW(self.dim() >= 2, error::kInvalidArgument)
            << "input must have at least 2 dimensions, got " << self.dim();

        TT_ASSIGN_OR_THROW(mlir::ElementType out_mlir_type,
                           ConvertTo<mlir::ElementType>(self.scalar_type()));

        auto step_keys = param_keys.Clone();
        TT_THROW_IF_ERROR(step_keys.SetParam("upper", upper));

        TT_ASSIGN_OR_THROW(
            auto buffer,
            DispatchOp<1>(LinalgCholeskyBuilder(/*lower=*/!upper), {self},
                          {.out_dtype = out_mlir_type,
                           .out_dims = CopyIntVector(self.sizes()),
                           .op_param_cache_keys = std::move(step_keys)}));

        TT_THROW_IF_ERROR(ResizeTensorIfShapeDiffers(out, self.sizes()));
        TT_THROW_IF_ERROR(AssignBufferToAtTensor(std::move(buffer), out));
        return out;
      });
}

at::Tensor AtenLinalgCholesky(const at::Tensor& self, bool upper) {
  TT_KERNEL(
      OpName::kLinalgCholesky, param_keys,
      (self, IgnoreInCacheKey(upper, "Doesn't affect SHLO type")), {
        TT_CHECK_THROW(self.dim() >= 2, error::kInvalidArgument)
            << "input must have at least 2 dimensions, got " << self.dim();

        TT_ASSIGN_OR_THROW(mlir::ElementType out_mlir_type,
                           ConvertTo<mlir::ElementType>(self.scalar_type()));

        auto step_keys = param_keys.Clone();
        TT_THROW_IF_ERROR(step_keys.SetParam("upper", upper));

        TT_ASSIGN_OR_THROW(
            auto buffer,
            DispatchOp<1>(LinalgCholeskyBuilder(/*lower=*/!upper), {self},
                          {.out_dtype = out_mlir_type,
                           .out_dims = CopyIntVector(self.sizes()),
                           .op_param_cache_keys = std::move(step_keys)}));

        return MakeTensor(std::move(buffer));
      });
}

// Stub implementation for linalg_cholesky_ex (it expects an info tensor which
// stablehlo doesn't natively return)
std::tuple<at::Tensor&, at::Tensor&> AtenLinalgCholeskyExL(
    const at::Tensor& self, bool upper, bool check_errors, at::Tensor& L,
    at::Tensor& info) {
  TT_KERNEL(OpName::kLinalgCholeskyExL, _,
            (self, IgnoreInCacheKey(upper, "Doesn't affect SHLO type"),
             IgnoreInCacheKey(check_errors, "Doesn't affect SHLO"), L, info),
            {
              AtenLinalgCholeskyOut(self, upper, L);

              // info is 0 for success.
              Dimensions batch_dims(self.sizes().begin(),
                                    self.sizes().begin() + self.dim() - 2);
              TT_THROW_IF_ERROR(ResizeTensorIfShapeDiffers(info, batch_dims));
              info.zero_();

              return {L, info};
            });
}

std::tuple<at::Tensor, at::Tensor> AtenLinalgCholeskyEx(const at::Tensor& self,
                                                        bool upper,
                                                        bool check_errors) {
  TT_KERNEL(OpName::kLinalgCholeskyEx, _,
            (self, IgnoreInCacheKey(upper, "Doesn't affect SHLO type"),
             IgnoreInCacheKey(check_errors, "Doesn't affect SHLO")),
            {
              at::Tensor L = AtenLinalgCholesky(self, upper);

              Dimensions batch_dims(self.sizes().begin(),
                                    self.sizes().begin() + self.dim() - 2);
              at::Tensor info =
                  at::zeros(batch_dims, self.options().dtype(at::kInt));

              return {L, info};
            });
}

at::Tensor AtenCholesky(const at::Tensor& self, bool upper) {
  TT_KERNEL(OpName::kCholesky, _,
            (self, IgnoreInCacheKey(upper, "Doesn't affect SHLO type")),
            { return AtenLinalgCholesky(self, upper); });
}

at::Tensor& AtenCholeskyOut(const at::Tensor& self, bool upper,
                            at::Tensor& out) {
  TT_KERNEL(OpName::kCholeskyOut, _,
            (self, IgnoreInCacheKey(upper, "Doesn't affect SHLO type"), out),
            { return AtenLinalgCholeskyOut(self, upper, out); });
}

}  // namespace torch_tpu
