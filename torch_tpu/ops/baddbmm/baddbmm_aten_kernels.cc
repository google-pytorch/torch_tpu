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

#include <complex>
#include <utility>
#include <vector>

#include "ATen/core/ATen_fwd.h"
#include "absl/log/absl_check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Types.h"
#include "mlir/Support/LLVM.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"
#include "torch/headeronly/core/ScalarType.h"
#include "torch_tpu/common/aten_utils.h"
#include "torch_tpu/common/cache_key.h"
#include "torch_tpu/common/dimension_types.h"
#include "torch_tpu/common/dtype.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/common/to_string.h"
#include "torch_tpu/eager/device_buffer.h"
#include "torch_tpu/eager/op_dispatcher.h"
#include "torch_tpu/eager/tensor_to_buffer.h"
#include "torch_tpu/ops/bmm/bmm.h"
#include "torch_tpu/ops/macros/kernel.h"
#include "torch_tpu/ops/op_builder_utils.h"
#include "torch_tpu/ops/op_names.h"
#include "torch_tpu/ops/precision_context.h"

namespace torch_tpu {
namespace {

absl::Status CheckBaddbmmOut(const at::Tensor& out) {
  TT_RET_CHECK(!IsBool(out), error::kInvalidArgument)
      << "expected out tensor to have dtype float32, got bool";
  return absl::OkStatus();
}

absl::Status CheckBaddbmmInputs(const at::Tensor& self,
                                const at::Tensor& batch1,
                                const at::Tensor& batch2) {
  TT_RET_CHECK(batch1.dim() == 3, error::kInvalidArgument)
      << "expected batch1 to be a 3D tensor (batch of matrices), got "
      << batch1.dim() << "D";
  TT_RET_CHECK(batch2.dim() == 3, error::kInvalidArgument)
      << "expected batch2 to be a 3D tensor (batch of matrices), got "
      << batch2.dim() << "D";

  TT_RET_CHECK(batch1.size(0) == batch2.size(0), error::kInvalidArgument)
      << "expected the batch dimension of the first argument (of shape "
      << ToString(batch1.sizes())
      << ") to match the batch dimension of the second argument (of shape "
      << ToString(batch2.sizes()) << "), got " << batch1.size(0) << " vs "
      << batch2.size(0);
  TT_RET_CHECK(batch1.size(2) == batch2.size(1), error::kInvalidArgument)
      << "expected the last dimension of the first argument (of shape "
      << ToString(batch1.sizes())
      << ") to match the second dimension of the second argument (of shape "
      << ToString(batch2.sizes()) << "), got " << batch1.size(2) << " vs "
      << batch2.size(1);

  return absl::OkStatus();
}

mlir::MlirOp BuildSanitizedScalarOp(mlir::MlirBuilder& builder,
                                    const at::Scalar& s,
                                    mlir::Type element_type) {
  if (mlir::isa<mlir::ComplexType>(element_type)) {
    auto c = s.toComplexDouble();
    return MakeScalarConstant(builder, std::complex<double>(c.real(), c.imag()),
                              element_type);
  } else if (mlir::isa<mlir::FloatType>(element_type)) {
    return MakeScalarConstant(builder, s.toDouble(), element_type);
  } else {
    return MakeScalarConstant(builder, s.toLong(), element_type);
  }
}

absl::StatusOr<mlir::MlirOp> BuildBaddbmmShlo(
    mlir::MlirOp self_op, mlir::MlirOp batch1_op, mlir::MlirOp batch2_op,
    const at::Scalar& beta, const at::Scalar& alpha, at::ScalarType out_dtype) {
  auto& builder = self_op.getBuilder();
  TT_ASSIGN_OR_RETURN(mlir::ElementType out_dtype_mlir,
                      ConvertTo<mlir::ElementType>(out_dtype));

  const auto current_precision = GetPrecision();
  TT_ASSIGN_OR_RETURN(
      mlir::MlirOp bmm_res,
      BuildBmmShlo(batch1_op, batch2_op, out_dtype_mlir, current_precision));
  auto calc_type = GetTensorTypeOrDie(bmm_res).getElementType();

  // If alpha is not 1, multiply bmm_res by alpha.
  if (!alpha.equal(1.0) && !alpha.equal(1)) {
    mlir::MlirOp alpha_scalar =
        BuildSanitizedScalarOp(builder, alpha, calc_type);
    TT_ASSIGN_OR_RETURN(auto alpha_tensor,
                        BroadcastIfNeeded(alpha_scalar, bmm_res));
    bmm_res = mlir::stablehlo::Mul(bmm_res, alpha_tensor);
  }

  // If beta is 0, content of input is ignored (no NaN / infinity propagation)
  if (beta.equal(0.0) || beta.equal(0)) {
    return bmm_res;
  }

  // Broadcast self to match bmm_res shape
  TT_ASSIGN_OR_RETURN(auto self_bcst, BroadcastIfNeeded(self_op, bmm_res));

  // If beta is not 1, multiply self_bcst by beta.
  if (!beta.equal(1.0) && !beta.equal(1)) {
    auto beta_scalar = BuildSanitizedScalarOp(builder, beta, calc_type);
    TT_ASSIGN_OR_RETURN(auto beta_tensor,
                        BroadcastIfNeeded(beta_scalar, self_bcst));
    self_bcst = mlir::stablehlo::Mul(self_bcst, beta_tensor);
  }

  return mlir::stablehlo::Add(self_bcst, bmm_res);
}

absl::StatusOr<DeviceBufferRef> Baddbmm(
    const at::Tensor& self, const at::Tensor& batch1, const at::Tensor& batch2,
    const at::Scalar& beta, const at::Scalar& alpha, at::ScalarType out_dtype,
    OpParamCacheKeys& param_keys) {
  TT_RETURN_IF_ERROR(CheckBaddbmmInputs(self, batch1, batch2));

  TT_ASSIGN_OR_RETURN(mlir::ElementType out_dtype_mlir,
                      ConvertTo<mlir::ElementType>(out_dtype));
  Dimensions out_dims = {batch1.size(0), batch1.size(1), batch2.size(2)};

  const std::vector<at::Tensor> inputs = {self, batch1, batch2};

  auto op_builder =
      [beta, alpha, out_dtype](
          absl::Span<mlir::MlirOp> inputs_op,
          mlir::MlirBuilder& builder) -> absl::StatusOr<mlir::MlirOp> {
    ABSL_CHECK_EQ(  // CRASH_OK=Dispatcher parameter binding safety boundary.
        inputs_op.size(), 3);
    const mlir::MlirOp self_op = inputs_op[0];
    const mlir::MlirOp batch1_op = inputs_op[1];
    const mlir::MlirOp batch2_op = inputs_op[2];
    return BuildBaddbmmShlo(self_op, batch1_op, batch2_op, beta, alpha,
                            out_dtype);
  };

  TT_ASSIGN_OR_RETURN(
      auto result_buffer,
      DispatchOp<kDynamicSize>(
          std::move(op_builder), inputs,
          // It's OK for both calls to Baddbmm() to use the same OpName for
          // dispatching, as the logic is the same for both.
          {.op_name = OpName::kBaddbmmOut,
           .out_dtype = out_dtype_mlir,
           .out_dims = out_dims,
           .op_param_cache_keys = std::move(param_keys)}));
  return result_buffer;
}

}  // namespace

at::Tensor AtenBaddbmmDtype(const at::Tensor& self, const at::Tensor& batch1,
                            const at::Tensor& batch2, at::ScalarType out_dtype,
                            const at::Scalar& beta, const at::Scalar& alpha) {
  TT_KERNEL(OpName::kBaddbmmDtype, param_keys,
            (self, batch1, batch2, out_dtype, beta, alpha), {
              TT_ASSIGN_OR_THROW(auto result_buffer,
                                 Baddbmm(self, batch1, batch2, beta, alpha,
                                         out_dtype, param_keys));
              return MakeTensor(result_buffer);
            });
}

at::Tensor& AtenBaddbmmDtypeOut(const at::Tensor& self,
                                const at::Tensor& batch1,
                                const at::Tensor& batch2,
                                at::ScalarType out_dtype,
                                const at::Scalar& beta, const at::Scalar& alpha,
                                at::Tensor& out) {
  TT_KERNEL(OpName::kBaddbmmDtypeOut, param_keys,
            (self, batch1, batch2, out_dtype, beta, alpha, out), {
              TT_THROW_IF_ERROR(CheckBaddbmmOut(out));
              TT_ASSIGN_OR_THROW(auto result_buffer,
                                 Baddbmm(self, batch1, batch2, beta, alpha,
                                         out_dtype, param_keys));
              TT_THROW_IF_ERROR(
                  AssignBufferToAtTensor(std::move(result_buffer), out));
              return out;
            });
}

at::Tensor& AtenBaddbmmOut(const at::Tensor& self, const at::Tensor& batch1,
                           const at::Tensor& batch2, const at::Scalar& beta,
                           const at::Scalar& alpha, at::Tensor& out) {
  TT_KERNEL(OpName::kBaddbmmOut, _,
            (self, batch1, batch2, IgnoreInCacheKey(beta, "Legacy usage"),
             IgnoreInCacheKey(alpha, "Legacy usage"), out),
            {
              return AtenBaddbmmDtypeOut(self, batch1, batch2,
                                         out.scalar_type(), beta, alpha, out);
            });
}

}  // namespace torch_tpu
