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

#include "csrc/ops/addcdiv/addcdiv_aten_kernels.h"

#include <optional>
#include <utility>

#include "ATen/core/ATen_fwd.h"
#include "ATen/native/TypeProperties.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "csrc/common/aten_utils.h"
#include "csrc/common/cache_key.h"
#include "csrc/common/device_type.h"
#include "csrc/common/dimension_types.h"
#include "csrc/common/dtype.h"
#include "csrc/common/error_utils.h"
#include "csrc/common/fixed_size_span.h"
#include "csrc/common/to_string.h"
#include "csrc/eager/op_dispatcher.h"
#include "csrc/ops/binary.h"
#include "csrc/ops/macros/kernel.h"
#include "csrc/ops/op_builder_utils.h"
#include "csrc/ops/op_names.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"

namespace torch_tpu {
namespace {

// Returns self + value * (tensor1 / tensor2). If value is nullopt, assumes
// value is 1 and omits the multiplication.
absl::StatusOr<mlir::MlirOp> BuildAddcdivShlo(mlir::MlirOp self,
                                              mlir::MlirOp tensor1,
                                              mlir::MlirOp tensor2,
                                              std::optional<mlir::MlirOp> value,
                                              mlir::ElementType promoted_dtype,
                                              mlir::ElementType out_dtype) {
  TT_ASSIGN_OR_RETURN(mlir::ElementType compute_dtype,
                      InferComputationDtype(promoted_dtype));

  TT_ASSIGN_OR_RETURN(self, ComplexSafeConvertElementType(self, compute_dtype));
  TT_ASSIGN_OR_RETURN(tensor1,
                      ComplexSafeConvertElementType(tensor1, compute_dtype));
  TT_ASSIGN_OR_RETURN(tensor2,
                      ComplexSafeConvertElementType(tensor2, compute_dtype));

  TT_ASSIGN_OR_RETURN(mlir::MlirOp div, BuildDivShlo(tensor1, tensor2));
  mlir::MlirOp term = div;
  if (value.has_value()) {
    TT_ASSIGN_OR_RETURN(mlir::MlirOp value_op,
                        ComplexSafeConvertElementType(*value, compute_dtype));
    TT_ASSIGN_OR_RETURN(term, BuildMulShlo(value_op, div));
  }
  TT_ASSIGN_OR_RETURN(mlir::MlirOp result, BuildAddShlo(self, term));
  return ComplexSafeConvertElementType(result, out_dtype);
}

}  // namespace

at::Tensor& AtenAddcdivOut(const at::Tensor& self, const at::Tensor& tensor1,
                           const at::Tensor& tensor2, const at::Scalar& value,
                           at::Tensor& out) {
  auto promoted_value = PromoteScalar(value).AvoidPromoting(ScalarValue::kOne);
  TT_KERNEL(
      OpName::kAddcdivOut, param_keys,
      (self, tensor1, tensor2, promoted_value, out), {
        TT_CHECK_THROW(IsPrivateUse1Device(out), error::kInvalidArgument)
            << "expected out tensor to be on "
            << GetPrivateUse1DeviceDebugName() << ", got " << out.device();

        TT_CHECK_THROW(!IsIntegral(tensor1) || !IsIntegral(tensor2),
                       error::kInvalidArgument)
            << "integer division with addcdiv is no longer supported; use "
               "(input + value * torch.trunc(tensor1 / "
               "tensor2)).to(input.dtype) "
               "for integer inputs and (input + value * tensor1 / tensor2) for "
               "float inputs";

        TT_ASSIGN_OR_THROW(
            const Dimensions expected_size,
            InferSize(self.sizes(), tensor1.sizes(), tensor2.sizes()));

        if (out.is_alias_of(self) || out.is_alias_of(tensor1) ||
            out.is_alias_of(tensor2)) {
          TT_CHECK_THROW(absl::MakeConstSpan(out.sizes()) == expected_size,
                         error::kInvalidArgument)
              << "output with shape " << ToString(out.sizes())
              << " doesn't match the broadcast shape of the tensor being "
                 "operated on in-place, which has shape "
              << ToString(expected_size);
        }

        at::native::ResultTypeState state = {};
        state = at::native::update_result_type_state(self, state);
        state = at::native::update_result_type_state(tensor1, state);
        state = at::native::update_result_type_state(tensor2, state);
        const at::ScalarType promoted_type = at::native::result_type(state);
        TT_THROW_IF_ERROR(ValidateNotLowPrecisionFloat(promoted_type));
        TT_THROW_IF_ERROR(ValidateNotLowPrecisionFloat(out));
        TT_THROW_IF_ERROR(ValidateOutDtype(out, promoted_type,
                                           /*allow_out_dtype_cast=*/true));

        const at::ScalarType acc_type = ToAccumulateType(promoted_type);
        TT_ASSIGN_OR_THROW(const bool can_cast_value,
                           CanCastScalarWithoutOverflow(value, acc_type));
        TT_CHECK_THROW(can_cast_value, error::kInvalidArgument)
            << "value cannot be converted to type " << ToString(acc_type)
            << " without overflow";

        TT_ASSIGN_OR_THROW(mlir::ElementType promoted_dtype,
                           ConvertTo<mlir::ElementType>(promoted_type));
        TT_ASSIGN_OR_THROW(mlir::ElementType out_dtype,
                           ConvertTo<mlir::ElementType>(out.scalar_type()));

        if (promoted_value.IsOne()) {
          auto op_builder = [promoted_dtype,
                             out_dtype](FixedSizeSpan<mlir::MlirOp, 3> inputs)
              -> absl::StatusOr<mlir::MlirOp> {
            auto& [self_op, tensor1_op, tensor2_op] = inputs;
            return BuildAddcdivShlo(self_op, tensor1_op, tensor2_op,
                                    std::nullopt, promoted_dtype, out_dtype);
          };

          TT_THROW_IF_ERROR(DispatchOpOut<3>(
              std::move(op_builder), {self, tensor1, tensor2}, out,
              {.out_dtype = out_dtype,
               .out_dims = expected_size,
               .op_param_cache_keys = std::move(param_keys)}));
          return out;
        }

        TT_ASSIGN_OR_THROW(at::Tensor value_tensor,
                           promoted_value.GetTensor(acc_type));

        // Build the op.
        auto op_builder = [promoted_dtype,
                           out_dtype](FixedSizeSpan<mlir::MlirOp, 4> inputs)
            -> absl::StatusOr<mlir::MlirOp> {
          auto& [self_op, tensor1_op, tensor2_op, value_op] = inputs;
          return BuildAddcdivShlo(self_op, tensor1_op, tensor2_op, value_op,
                                  promoted_dtype, out_dtype);
        };

        TT_THROW_IF_ERROR(DispatchOpOut<4>(
            std::move(op_builder), {self, tensor1, tensor2, value_tensor}, out,
            {.out_dtype = out_dtype,
             .out_dims = expected_size,
             .op_param_cache_keys = std::move(param_keys)}));
        return out;
      });
}

}  // namespace torch_tpu
