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

#include "csrc/ops/reductions/var_aten_kernels.h"

#include <array>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <tuple>
#include <utility>

#include "ATen/core/ATen_fwd.h"
#include "ATen/core/TensorBody.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "c10/core/ScalarType.h"
#include "c10/util/Exception.h"
#include "c10/util/OptionalArrayRef.h"
#include "csrc/common/cache_key.h"
#include "csrc/common/dimension_types.h"
#include "csrc/common/dtype.h"
#include "csrc/common/error_utils.h"
#include "csrc/common/fixed_size_span.h"
#include "csrc/eager/op_dispatcher.h"
#include "csrc/eager/tensor_to_buffer.h"
#include "csrc/ops/binary.h"
#include "csrc/ops/macros/kernel.h"
#include "csrc/ops/op_builder_utils.h"
#include "csrc/ops/op_names.h"
#include "csrc/ops/python_context.h"
#include "csrc/ops/reductions/mean_aten_kernels.h"
#include "csrc/ops/reductions/reduction_utils.h"
#include "csrc/ops/reductions/reductions.h"
#include "csrc/ops/reductions/sum.h"
#include "csrc/ops/unary.h"
#include "llvm/Support/Casting.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Types.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"
#include "stablehlo/transforms/StablehloBroadcastLowering.h"
#include "torch/headeronly/core/ScalarType.h"

namespace torch_tpu {

namespace {

NAryMlirOpBuilder<2, 1> CreateVarBuilder(Dimensions dim_vec,
                                         ReductionMode reduction_mode,
                                         int64_t reduction_size,
                                         mlir::ElementType scalar_dtype_mlir) {
  return [dim_vec = std::move(dim_vec), reduction_mode, reduction_size,
          scalar_dtype_mlir](FixedSizeSpan<mlir::MlirOp, 2> inputs)
             -> absl::StatusOr<mlir::MlirOp> {
    auto& [self_op, correction_op] = inputs;
    mlir::MlirBuilder& builder = self_op.getBuilder();
    mlir::Type type = GetTensorTypeOrDie(self_op).getElementType();

    // sum_tensor = ReduceSum(self, dim, keep_dims=true)
    TT_ASSIGN_OR_RETURN(
        auto sum_op, BuildSumShlo(self_op, dim_vec, ReductionMode::kKeepDims));

    // mean = sum_tensor / reduction_size
    mlir::MlirOp reduction_size_const;
    if (llvm::isa<mlir::ComplexType>(type)) {
      reduction_size_const = MakeScalarConstant(
          builder,
          std::complex<double>(static_cast<double>(reduction_size), 0.0), type);
    } else {
      reduction_size_const = MakeScalarConstant(
          builder, static_cast<double>(reduction_size), type);
    }
    TT_ASSIGN_OR_RETURN(auto mean, BuildDivShlo(sum_op, reduction_size_const));

    // diff = self - mean
    TT_ASSIGN_OR_RETURN(auto diff, BuildSubShlo(self_op, mean));

    // conj_diff = conj(diff)
    TT_ASSIGN_OR_RETURN(auto conj_diff, BuildConjPhysicalShlo(diff));

    // squared_diff = diff * conj_diff
    TT_ASSIGN_OR_RETURN(auto squared_diff, BuildMulShlo(diff, conj_diff));

    // sum_sq_tensor = ReduceSum(squared_diff, dim, keep_dims=keep_dim)
    TT_ASSIGN_OR_RETURN(
        auto sum_sq_op,
        BuildSumShlo(squared_diff, dim_vec, reduction_mode, scalar_dtype_mlir));

    // denom = max(reduction_size - correction, 0)
    mlir::Type corr_type = GetTensorTypeOrDie(correction_op).getElementType();
    auto reduction_size_const_corr = MakeScalarConstant(
        builder, static_cast<double>(reduction_size), corr_type);
    TT_ASSIGN_OR_RETURN(auto denom_diff,
                        BuildSubShlo(reduction_size_const_corr, correction_op));
    auto zero = MakeScalarConstant(builder, 0.0, corr_type);
    TT_ASSIGN_OR_RETURN(auto denom, BuildMaximumShlo(denom_diff, zero));

    // res = sum_sq_tensor / denom
    return BuildDivShlo(sum_sq_op, denom);
  };
}

void WarnIfInvalidCorrection(OpName op_name, double correction_val,
                             int64_t reduction_size) {
  const double denom = reduction_size - correction_val;
  if (denom <= 0) {
    TORCH_WARN(GetRootOpName(op_name),
               "(): degrees of freedom (i.e., reduction "
               "size - correction) should "
               "be positive, got reduction size = ",
               reduction_size, ", correction = ", correction_val,
               ", and degrees of freedom = ", denom);
  }
}

}  // namespace

at::Tensor AtenVar(const at::Tensor& self, c10::OptionalArrayRef<int64_t> dim,
                   const std::optional<at::Scalar>& correction, bool keep_dim) {
  PromotedScalar promoted_correction =
      PromoteScalar(correction.value_or(at::Scalar(1.0)));

  TT_KERNEL(
      OpName::kVar, param_keys, (self, dim, promoted_correction, keep_dim), {
        c10::ScalarType scalar_dtype = self.scalar_type();
        TT_THROW_IF_ERROR(ValidateFloatOrComplex(scalar_dtype));
        if (c10::isComplexType(scalar_dtype)) {
          scalar_dtype = c10::toRealValueType(scalar_dtype);
        }

        const ReductionMode reduction_mode =
            keep_dim ? ReductionMode::kKeepDims : ReductionMode::kDropDims;
        TT_ASSIGN_OR_THROW(Dimensions canonicalized_dims,
                           CanonicalizeDims(self, dim));
        Dimensions output_dims = GetSizesAfterReduction(
            self.sizes(), reduction_mode, canonicalized_dims);

        TT_ASSIGN_OR_THROW(at::Tensor correction_tensor,
                           promoted_correction.GetTensor(scalar_dtype));
        TT_ASSIGN_OR_THROW(int64_t reduction_size,
                           GetReductionFactor(self, dim));
        double correction_val = promoted_correction.scalar().toDouble();
        WarnIfInvalidCorrection(OpName::kVar, correction_val, reduction_size);

        TT_ASSIGN_OR_THROW(mlir::ElementType scalar_dtype_mlir,
                           internal::ToElementType(scalar_dtype));

        auto var_builder =
            CreateVarBuilder(std::move(canonicalized_dims), reduction_mode,
                             reduction_size, scalar_dtype_mlir);

        TT_ASSIGN_OR_THROW(
            auto result_bufs,
            (DispatchOp<2, 1>(std::move(var_builder), {self, correction_tensor},
                              {.out_dtype = scalar_dtype_mlir,
                               .out_dims = output_dims,
                               .op_param_cache_keys = std::move(param_keys)})));
        return MakeTensor(std::move(result_bufs));
      });
}

at::Tensor& AtenVarOut(const at::Tensor& self,
                       c10::OptionalArrayRef<int64_t> dim,
                       const std::optional<at::Scalar>& correction,
                       bool keep_dim, at::Tensor& out) {
  PromotedScalar promoted_correction =
      PromoteScalar(correction.value_or(at::Scalar(1.0)));

  TT_KERNEL(
      OpName::kVarOut, param_keys,
      (self, dim, promoted_correction, keep_dim, out), {
        c10::ScalarType scalar_dtype = out.scalar_type();
        TT_THROW_IF_ERROR(ValidateFloatOrComplex(scalar_dtype));

        const ReductionMode reduction_mode =
            keep_dim ? ReductionMode::kKeepDims : ReductionMode::kDropDims;
        TT_ASSIGN_OR_THROW(Dimensions canonicalized_dims,
                           CanonicalizeDims(self, dim));
        Dimensions output_dims = GetSizesAfterReduction(
            self.sizes(), reduction_mode, canonicalized_dims);

        TT_ASSIGN_OR_THROW(at::Tensor correction_tensor,
                           promoted_correction.GetTensor(scalar_dtype));
        TT_ASSIGN_OR_THROW(const int64_t reduction_size,
                           GetReductionFactor(self, dim));
        double correction_val = promoted_correction.scalar().toDouble();
        WarnIfInvalidCorrection(OpName::kVarOut, correction_val,
                                reduction_size);

        TT_ASSIGN_OR_THROW(mlir::ElementType scalar_dtype_mlir,
                           internal::ToElementType(scalar_dtype));

        auto var_builder =
            CreateVarBuilder(std::move(canonicalized_dims), reduction_mode,
                             reduction_size, scalar_dtype_mlir);

        TT_THROW_IF_ERROR((DispatchOpOut<2, 1>(
            std::move(var_builder), {self, correction_tensor}, out,
            {.out_dtype = scalar_dtype_mlir,
             .out_dims = output_dims,
             .op_param_cache_keys = std::move(param_keys)})));
        return out;
      });
}

std::tuple<at::Tensor, at::Tensor> AtenVarMeanCorrection(
    const at::Tensor& self, c10::OptionalArrayRef<int64_t> dim,
    const std::optional<at::Scalar>& correction, bool keep_dim) {
  PromotedScalar promoted_correction =
      PromoteScalar(correction.value_or(at::Scalar(1.0)));

  TT_KERNEL(
      OpName::kVarMeanCorrection, param_keys,
      (self, dim, promoted_correction, keep_dim), {
        auto scalar_dtype = self.scalar_type();
        TT_THROW_IF_ERROR(ValidateFloatOrComplex(scalar_dtype));
        auto self_scalar_type = scalar_dtype;
        if (c10::isComplexType(scalar_dtype)) {
          scalar_dtype = c10::toRealValueType(scalar_dtype);
        }

        const ReductionMode reduction_mode =
            keep_dim ? ReductionMode::kKeepDims : ReductionMode::kDropDims;
        TT_ASSIGN_OR_THROW(Dimensions canonical_dims,
                           CanonicalizeDims(self, dim));
        Dimensions output_dims = GetSizesAfterReduction(
            self.sizes(), reduction_mode, canonical_dims);

        TT_ASSIGN_OR_THROW(const mlir::ElementType element_type,
                           ConvertTo<mlir::ElementType>(scalar_dtype));
        TT_ASSIGN_OR_THROW(const mlir::ElementType self_element_type_mlir,
                           ConvertTo<mlir::ElementType>(self_scalar_type));

        // Upcast BF16/F16 inputs to F32 to perform the entire calculation
        // in high precision, avoiding precision loss
        std::optional<mlir::ElementType> computation_dtype = std::nullopt;
        const bool upcast_var = (element_type == mlir::ElementType::BF16 ||
                                 element_type == mlir::ElementType::F16);
        if (upcast_var) {
          computation_dtype = mlir::ElementType::F32;
        }

        TT_ASSIGN_OR_THROW(const at::Tensor correction_tensor,
                           promoted_correction.GetTensor(scalar_dtype));
        TT_ASSIGN_OR_THROW(const int64_t reduction_size,
                           GetReductionFactor(self, dim));
        double correction_val = promoted_correction.scalar().toDouble();
        WarnIfInvalidCorrection(OpName::kVarMeanCorrection, correction_val,
                                reduction_size);

        auto var_mean_builder =
            [canonical_dims = std::move(canonical_dims), reduction_mode,
             reduction_size, keep_dim](FixedSizeSpan<mlir::MlirOp, 2> inputs)
            -> absl::StatusOr<std::array<mlir::MlirOp, 2>> {
          auto& [self_op, correction_op] = inputs;
          mlir::MlirBuilder& builder = self_op.getBuilder();

          TT_ASSIGN_OR_RETURN(
              mlir::ElementType mean_type,
              ConvertTo<mlir::ElementType>(
                  GetTensorTypeOrDie(self_op).getElementType()));
          TT_ASSIGN_OR_RETURN(
              mlir::ElementType var_type,
              ConvertTo<mlir::ElementType>(
                  GetTensorTypeOrDie(correction_op).getElementType()));

          mlir::MlirOp mean_keepdims;
          TT_ASSIGN_OR_RETURN(
              mean_keepdims,
              BuildMeanShlo(self_op, canonical_dims, ReductionMode::kKeepDims,
                            mean_type));

          mlir::MlirOp mean_out;
          if (keep_dim) {
            mean_out = mean_keepdims;
          } else {
            // Squeeze the reduced dimensions dynamically
            mlir::stablehlo::Dimensions input_dims = GetDimensions(self_op);
            mlir::stablehlo::Dimensions squeezed_dims;
            squeezed_dims.reserve(input_dims.size() - canonical_dims.size());
            for (size_t i = 0; i < input_dims.size(); ++i) {
              bool is_reduced = false;
              for (int64_t dim : canonical_dims) {
                if (dim == static_cast<int64_t>(i)) {
                  is_reduced = true;
                  break;
                }
              }
              if (!is_reduced) {
                squeezed_dims.push_back(input_dims[i]);
              }
            }
            mlir::Type output_element_type =
                GetTensorTypeOrDie(mean_keepdims).getElementType();
            mlir::RankedTensorType target_type =
                mlir::stablehlo::getRankedTensorType(squeezed_dims,
                                                     output_element_type);
            mean_out = mlir::stablehlo::Reshape(target_type, mean_keepdims);
          }

          // diff = self - mean_keepdims
          TT_ASSIGN_OR_RETURN(auto diff, BuildSubShlo(self_op, mean_keepdims));

          // conj_diff = conj(diff)
          TT_ASSIGN_OR_RETURN(auto conj_diff, BuildConjPhysicalShlo(diff));

          // squared_diff = diff * conj_diff
          TT_ASSIGN_OR_RETURN(auto squared_diff, BuildMulShlo(diff, conj_diff));

          // sum_sq = ReduceSum(squared_diff, dim, keep_dims=keep_dim)
          TT_ASSIGN_OR_RETURN(auto sum_sq,
                              BuildSumShlo(squared_diff, canonical_dims,
                                           reduction_mode, var_type));

          // denom = max(reduction_size - correction, 0)
          auto reduction_size_const = MakeScalarConstant(
              builder, static_cast<double>(reduction_size), var_type);
          TT_ASSIGN_OR_RETURN(
              auto denom_diff,
              BuildSubShlo(reduction_size_const, correction_op));
          auto zero = MakeScalarConstant(builder, 0.0, var_type);
          TT_ASSIGN_OR_RETURN(auto denom, BuildMaximumShlo(denom_diff, zero));

          // var = sum_sq / denom
          TT_ASSIGN_OR_RETURN(auto var, BuildDivShlo(sum_sq, denom));

          return std::array<mlir::MlirOp, 2>{var, mean_out};
        };

        TT_ASSIGN_OR_THROW(
            (auto [var_buf, mean_buf]),
            (DispatchOp<2, 2>(
                std::move(var_mean_builder), {self, correction_tensor},
                {.out_dtypes = {element_type, self_element_type_mlir},
                 .out_dims_list = {output_dims, output_dims},
                 .computation_dtype = computation_dtype,
                 .op_param_cache_keys = std::move(param_keys)})));

        return std::make_tuple(MakeTensor(std::move(var_buf)),
                               MakeTensor(std::move(mean_buf)));
      });
}

}  // namespace torch_tpu
