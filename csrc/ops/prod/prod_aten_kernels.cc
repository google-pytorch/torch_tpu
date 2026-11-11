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

#include "csrc/ops/prod/prod_aten_kernels.h"

#include <cstdint>
#include <optional>
#include <utility>

#include "ATen/WrapDimUtils.h"
#include "ATen/core/ATen_fwd.h"
#include "ATen/core/ScalarType.h"
#include "absl/functional/bind_front.h"
#include "absl/log/absl_log.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "csrc/common/aten_utils.h"
#include "csrc/common/cache_key.h"
#include "csrc/common/device_type.h"
#include "csrc/common/dimension_types.h"
#include "csrc/common/dtype.h"
#include "csrc/common/error_utils.h"
#include "csrc/eager/device_buffer.h"
#include "csrc/eager/tensor_to_buffer.h"
#include "csrc/ops/copy_from/tpu_to_tpu.h"
#include "csrc/ops/macros/kernel.h"
#include "csrc/ops/op_builder_utils.h"
#include "csrc/ops/op_names.h"
#include "csrc/ops/reductions/reduction_utils.h"
#include "csrc/ops/reductions/reductions.h"
#include "csrc/ops/unary_aten_kernels.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Types.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"

namespace torch_tpu {
namespace {

namespace stablehlo = mlir::stablehlo;

// Helper to create the initial value for the product reduction (1 or True)
mlir::MlirOp GetProdInitValue(mlir::MlirBuilder& builder,
                              mlir::Type element_type) {
  if (element_type.isInteger(/*width=*/1)) {
    // For boolean, the multiplicative identity is True.
    return MakeScalarConstant(builder, true, element_type);
  }
  // For float, integer, complex, the multiplicative identity is 1.
  return MakeScalarConstant(builder, /*value=*/1, element_type);
}

mlir::MlirOp BuildProdShlo(std::optional<int64_t> dim, ReductionMode mode,
                           std::optional<mlir::ElementType> dtype,
                           mlir::MlirOp input) {
  ABSL_VLOG(1) << "BuildProdShlo input: " << input.ToString()
               << " dim: " << (dim ? absl::StrCat(*dim) : "None")
               << " keep_dim: " << (mode == ReductionMode::kKeepDims)
               << " dtype: " << (dtype ? absl::StrCat(*dtype) : "None");

  mlir::MlirBuilder& builder = input.getBuilder();

  const mlir::RankedTensorType input_tensor_type = GetTensorTypeOrDie(input);

  mlir::Type element_type = input_tensor_type.getElementType();
  // Respect the dtype if provided, otherwise always convert to int64 for
  // boolean and integer types for accumulation.
  if (dtype.has_value()) {
    element_type = getElementType(builder.getContext(), dtype.value());
  } else if (input_tensor_type.getElementType().isInteger()) {
    element_type = builder.getOpBuilder().getI64Type();
  }

  mlir::MlirOp converted_input = input;
  if (input_tensor_type.getElementType() != element_type) {
    converted_input = stablehlo::ConvertElementType(input, element_type);
  }

  mlir::MlirOp init_value = GetProdInitValue(builder, element_type);

  auto reduce_fn = [element_type](mlir::RegionBuilder& rb) {
    stablehlo::buildReduceBody<stablehlo::MulOp>(element_type, rb.getRegion(),
                                                 rb.getOpBuilder());
  };

  int64_t rank = input_tensor_type.getRank();

  Dimensions dims;
  if (dim.has_value()) {
    // If dim is provided, then reduce only the specified dimension.
    int64_t normalized_dim =
        at::maybe_wrap_dim(*dim, rank);  // MAYBE_WRAP_DIM_OK=Verified in kernel
    dims.push_back(normalized_dim);
  } else {
    // If dim is not provided, then reduce all dimensions.
    // This is the default behavior of torch.prod.
    dims = GetAllDimensions(converted_input);
  }

  mlir::MlirOp result_op = stablehlo::Reduce(builder, {converted_input},
                                             {init_value}, reduce_fn, dims)[0];

  if (dim.has_value() && mode == ReductionMode::kKeepDims) {
    int64_t normalized_dim =
        at::maybe_wrap_dim(*dim, rank);  // MAYBE_WRAP_DIM_OK=Verified in kernel
    result_op = BuildKeepDimsShlo(converted_input, result_op, {normalized_dim});
  }
  return result_op;
}

at::ScalarType InferProdDtype(at::ScalarType input_dtype,
                              std::optional<at::ScalarType> dtype) {
  if (dtype.has_value()) {
    // If the user specifies a dtype, use it.
    return dtype.value();
  }
  if (IsIntegral(input_dtype)) {
    return c10::ScalarType::Long;
  }
  // Otherwise, the output type is the same as the input type.
  return input_dtype;
}

// Helper function to handle 0-dimensional tensor (scalar) product reduction.
// Since a 0-dimensional tensor contains only a single element, its product is
// equivalent to a copy (if the dtype matches) or a type cast (if the dtype
// differs).
absl::StatusOr<at::Tensor> ProdZeroDim(const at::Tensor& self,
                                       at::ScalarType inferred_dtype,
                                       OpParamCacheKeys param_keys) {
  TT_RETURN_IF_ERROR(ValidateNotLowPrecisionFloat(inferred_dtype));
  if (self.scalar_type() == inferred_dtype) {
    // If input and output dtypes match, wrap the existing DeviceBufferRef in
    // a new ATen tensor. This avoids dispatching an identity copy op.
    TT_ASSIGN_OR_RETURN(const DeviceBufferRef src_buf, GetBuffer(self));
    return MakeTensor(src_buf, self.device().index());
  }

  // If the dtype differs, compile and dispatch a StableHLO ConvertElementType
  // operation via UnaryOp to perform type conversion natively on TPU.
  TT_ASSIGN_OR_RETURN(auto inferred_mlir_dtype,
                      ConvertTo<mlir::ElementType>(inferred_dtype));
  auto convert_builder =
      [inferred_mlir_dtype](
          mlir::MlirOp input) -> absl::StatusOr<mlir::MlirOp> {
    return mlir::stablehlo::ConvertElementType(input, inferred_mlir_dtype);
  };
  TT_ASSIGN_OR_RETURN(auto result,
                      UnaryOp(self, convert_builder,
                              {.op_param_cache_keys = std::move(param_keys),
                               .out_dtype = inferred_mlir_dtype,
                               .allow_low_precision_float =
                                   !IsLowPrecisionFloat(inferred_dtype)}));
  return result;
}

absl::StatusOr<at::Tensor> AtenProdHelper(const at::Tensor& self,
                                          std::optional<int64_t> dim,
                                          bool keep_dim,
                                          std::optional<at::ScalarType> dtype,
                                          OpParamCacheKeys param_keys) {
  at::ScalarType inferred_dtype = InferProdDtype(self.scalar_type(), dtype);

  if (self.dim() == 0) {
    return ProdZeroDim(self, inferred_dtype, std::move(param_keys));
  }

  std::optional<int64_t> normalized_dim = std::nullopt;
  if (dim.has_value()) {
    TT_ASSIGN_OR_RETURN(normalized_dim, SafeWrapDim(*dim, self.dim()));
  }

  std::optional<mlir::ElementType> dtype_mlir_type = std::nullopt;
  if (dtype.has_value()) {
    TT_ASSIGN_OR_RETURN(  // ERROR_COV_INFEASIBLE=there is no way to trigger
                          // this error as PyTorch attempts to create empty
                          // tensor with this dtype and errors out early.
        dtype_mlir_type, ConvertTo<mlir::ElementType>(dtype.value()));
  }

  TT_ASSIGN_OR_RETURN(auto inferred_mlir_dtype,
                      ConvertTo<mlir::ElementType>(inferred_dtype));

  ReductionMode mode =
      keep_dim ? ReductionMode::kKeepDims : ReductionMode::kDropDims;
  Dimensions output_dims =
      normalized_dim.has_value()
          ? GetSizesAfterReduction(self.sizes(), mode, {*normalized_dim})
          : Dimensions();

  TT_ASSIGN_OR_RETURN(
      auto result, UnaryOp(self,
                           absl::bind_front(BuildProdShlo, normalized_dim, mode,
                                            dtype_mlir_type),
                           {.op_param_cache_keys = std::move(param_keys),
                            .out_dtype = inferred_mlir_dtype,
                            .out_dims = std::move(output_dims),
                            .allow_low_precision_float =
                                !IsLowPrecisionFloat(inferred_dtype)}));
  return result;
}

}  // namespace

at::Tensor AtenProd(const at::Tensor& self,
                    std::optional<at::ScalarType> dtype) {
  TT_KERNEL(OpName::kProd, param_keys, (self, dtype), {
    TT_ASSIGN_OR_THROW(
        at::Tensor result,
        AtenProdHelper(self, /*dim=*/std::nullopt,
                       /*keep_dim=*/false, dtype, std::move(param_keys)));
    return result;
  });
}

at::Tensor& AtenProdDimOut(const at::Tensor& self, int64_t dim, bool keep_dim,
                           std::optional<at::ScalarType> dtype,
                           at::Tensor& out) {
  TT_KERNEL(
      OpName::kProdDimOut, param_keys, (self, dim, keep_dim, dtype, out), {
        TT_CHECK_THROW(IsPrivateUse1Device(out), error::kInvalidArgument)
            << "expected the output tensor to be on '"
            << GetPrivateUse1DeviceDebugName() << "', got '" << out.device()
            << "'";
        TT_ASSIGN_OR_THROW(
            at::Tensor result,
            AtenProdHelper(self, dim, keep_dim, dtype, std::move(param_keys)));
        TT_THROW_IF_ERROR(CopyTpuToTpu(result, out));
        return out;
      });
}
}  // namespace torch_tpu
