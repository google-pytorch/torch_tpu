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

#include "torch_tpu/ops/foreach_aten_kernels.h"

#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string_view>
#include <utility>
#include <vector>

#include "ATen/core/ATen_fwd.h"
#include "absl/functional/any_invocable.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "c10/core/DefaultDtype.h"
#include "c10/core/ScalarType.h"
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
#include "torch_tpu/common/utils.h"
#include "torch_tpu/eager/device_buffer.h"
#include "torch_tpu/eager/op_dispatcher.h"
#include "torch_tpu/eager/tensor_to_buffer.h"
#include "torch_tpu/ops/binary.h"
#include "torch_tpu/ops/binary_aten_kernels.h"
#include "torch_tpu/ops/clamp/clamp_aten_kernels.h"
#include "torch_tpu/ops/copy_from/copy_from_aten_kernels.h"
#include "torch_tpu/ops/linalg/vector_norm/aten_vector_norm_kernels.h"
#include "torch_tpu/ops/macros/kernel.h"
#include "torch_tpu/ops/min_max/min_max_aten_kernels.h"
#include "torch_tpu/ops/nullary_aten_kernels.h"
#include "torch_tpu/ops/op_builder_utils.h"
#include "torch_tpu/ops/op_names.h"
#include "torch_tpu/ops/round/round.h"
#include "torch_tpu/ops/sigmoid/sigmoid_aten_kernels.h"
#include "torch_tpu/ops/to_copy/to_copy_aten_kernels.h"
#include "torch_tpu/ops/unary.h"

namespace torch_tpu {

namespace {

c10::ScalarType GetOutputDtypeFromTensorAndScalar(const at::Tensor& tensor,
                                                  const at::Scalar& scalar) {
  if ((IsFloatingPoint(tensor) && scalar.isFloatingPoint()) ||
      (IsInteger(tensor) && scalar.isIntegral(/*includeBool=*/true))) {
    // If both tensor and scalar are floating point or integral, then use the
    // tensor's dtype, unless the tensor is boolean (see below).
    return tensor.scalar_type();
  } else if (IsIntegral(tensor) && scalar.isFloatingPoint()) {
    // If tensor is integral and scalar is floating point, then use the
    // default floating point dtype.
    return c10::get_default_dtype_as_scalartype();
  } else {
    // If the tensor is boolean and the scalar is integer, just normally
    // promote the types to the scalar's integer type.
    return c10::promoteTypes(tensor.scalar_type(), scalar.type());
  }
}

// If the input scalar is Boolean, converts it to int. Otherwise, returns it
// unchanged.
at::Scalar BoolScalarToInt(const at::Scalar& scalar) {
  if (scalar.type() == at::ScalarType::Bool) {
    return at::Scalar(scalar.to<bool>() ? 1 : 0);
  }
  return scalar;
}

// If the input tensor is Boolean, converts it to int. Otherwise, returns it
// unchanged.
at::Tensor BoolTensorToInt(const at::Tensor& tensor) {
  if (tensor.scalar_type() == at::kBool) {
    return tensor.to(at::kInt);
  }
  return tensor;
}

absl::StatusOr<mlir::SmallVector<mlir::MlirOp>> BuildForeachShlo(
    absl::Span<const mlir::MlirOp> self, absl::Span<const mlir::MlirOp> other,
    absl::Span<const mlir::ElementType> out_dtypes,
    absl::AnyInvocable<mlir::MlirOp(mlir::MlirOp&, mlir::MlirOp&)>
        tensor_transform,
    mlir::MlirBuilder& builder) {
  mlir::SmallVector<mlir::MlirOp> results;
  results.reserve(self.size());
  for (int i = 0; i < self.size(); ++i) {
    mlir::MlirOp current_self = self[i];
    mlir::MlirOp current_other = other[i];

    TT_ASSIGN_OR_RETURN(current_self,
                        CastIfNeeded(current_self, out_dtypes[i]));
    TT_ASSIGN_OR_RETURN(current_other,
                        CastIfNeeded(current_other, out_dtypes[i]));

    std::array<mlir::MlirOp, 2> broadcasted_ops;
    TT_ASSIGN_OR_RETURN(broadcasted_ops,
                        ApplyBroadcastIfNeeded(current_self, current_other));
    current_self = broadcasted_ops[0];
    current_other = broadcasted_ops[1];

    mlir::MlirOp result = tensor_transform(current_self, current_other);
    results.push_back(result);
  }
  return results;
}

absl::StatusOr<mlir::SmallVector<mlir::MlirOp>> BuildForeachAddcdivShlo(
    absl::Span<const mlir::MlirOp> self, absl::Span<const mlir::MlirOp> tensor1,
    absl::Span<const mlir::MlirOp> tensor2,
    absl::Span<const mlir::MlirOp> value,
    absl::Span<const mlir::ElementType> out_dtypes,
    mlir::MlirBuilder& builder) {
  mlir::SmallVector<mlir::MlirOp> results;
  results.reserve(self.size());
  for (auto i = 0; i < self.size(); ++i) {
    mlir::MlirOp current_self = self[i];
    mlir::MlirOp current_tensor1 = tensor1[i];
    mlir::MlirOp current_tensor2 = tensor2[i];
    mlir::MlirOp current_value = value[i];

    TT_ASSIGN_OR_RETURN(current_self,
                        CastIfNeeded(current_self, out_dtypes[i]));
    TT_ASSIGN_OR_RETURN(current_tensor1,
                        CastIfNeeded(current_tensor1, out_dtypes[i]));
    TT_ASSIGN_OR_RETURN(current_tensor2,
                        CastIfNeeded(current_tensor2, out_dtypes[i]));
    TT_ASSIGN_OR_RETURN(current_value,
                        CastIfNeeded(current_value, out_dtypes[i]));

    TT_ASSIGN_OR_RETURN(mlir::MlirOp div,
                        BuildDivShlo(current_tensor1, current_tensor2));
    TT_ASSIGN_OR_RETURN(mlir::MlirOp value_div,
                        BuildMulShlo(current_value, div));
    TT_ASSIGN_OR_RETURN(mlir::MlirOp result,
                        BuildAddShlo(current_self, value_div));
    results.push_back(result);
  }
  return results;
}

absl::StatusOr<mlir::SmallVector<mlir::MlirOp>> BuildForeachAddcmulShlo(
    absl::Span<const mlir::MlirOp> self, absl::Span<const mlir::MlirOp> tensor1,
    absl::Span<const mlir::MlirOp> tensor2,
    absl::Span<const mlir::MlirOp> value,
    absl::Span<const mlir::ElementType> out_dtypes,
    mlir::MlirBuilder& builder) {
  mlir::SmallVector<mlir::MlirOp> results;
  results.reserve(self.size());
  for (auto i = 0; i < self.size(); ++i) {
    mlir::MlirOp current_self = self[i];
    mlir::MlirOp current_tensor1 = tensor1[i];
    mlir::MlirOp current_tensor2 = tensor2[i];
    mlir::MlirOp current_value = value[i];

    TT_ASSIGN_OR_RETURN(current_self,
                        CastIfNeeded(current_self, out_dtypes[i]));
    TT_ASSIGN_OR_RETURN(current_tensor1,
                        CastIfNeeded(current_tensor1, out_dtypes[i]));
    TT_ASSIGN_OR_RETURN(current_tensor2,
                        CastIfNeeded(current_tensor2, out_dtypes[i]));
    TT_ASSIGN_OR_RETURN(current_value,
                        CastIfNeeded(current_value, out_dtypes[i]));

    TT_ASSIGN_OR_RETURN(mlir::MlirOp product,
                        BuildMulShlo(current_tensor1, current_tensor2));
    TT_ASSIGN_OR_RETURN(mlir::MlirOp value_product,
                        BuildMulShlo(current_value, product));
    TT_ASSIGN_OR_RETURN(mlir::MlirOp result,
                        BuildAddShlo(current_self, value_product));
    results.push_back(result);
  }
  return results;
}

absl::StatusOr<mlir::SmallVector<mlir::MlirOp>> BuildForeachLerpShlo(
    absl::Span<const mlir::MlirOp> self, absl::Span<const mlir::MlirOp> other,
    absl::Span<const mlir::MlirOp> weight,
    absl::Span<const mlir::ElementType> out_dtypes,
    mlir::MlirBuilder& builder) {
  mlir::SmallVector<mlir::MlirOp> results;
  results.reserve(self.size());
  for (auto i = 0; i < self.size(); ++i) {
    mlir::MlirOp current_self = self[i];
    mlir::MlirOp current_other = other[i];
    mlir::MlirOp current_weight = weight[i];

    TT_ASSIGN_OR_RETURN(current_self,
                        CastIfNeeded(current_self, out_dtypes[i]));
    TT_ASSIGN_OR_RETURN(current_other,
                        CastIfNeeded(current_other, out_dtypes[i]));
    TT_ASSIGN_OR_RETURN(current_weight,
                        CastIfNeeded(current_weight, out_dtypes[i]));

    // lerp(start, end, weight) = start + weight * (end - start)
    TT_ASSIGN_OR_RETURN(mlir::MlirOp diff,
                        BuildSubShlo(current_other, current_self));
    TT_ASSIGN_OR_RETURN(mlir::MlirOp weighted_diff,
                        BuildMulShlo(current_weight, diff));
    TT_ASSIGN_OR_RETURN(mlir::MlirOp result,
                        BuildAddShlo(current_self, weighted_diff));
    results.push_back(result);
  }
  return results;
}

using DtypeVec = std::vector<mlir::ElementType>;
using DtypeSpan = absl::Span<const mlir::ElementType>;

std::vector<at::Tensor> ForeachConvertToTensor(
    std::vector<DeviceBufferRef> result_buffers, DtypeSpan out_dtypes) {
  std::vector<at::Tensor> result;
  result.reserve(result_buffers.size());
  for (auto i = 0; i < result_buffers.size(); ++i) {
    at::Tensor tensor = MakeTensor(std::move(result_buffers[i]));
    at::ScalarType target_dtype = ConvertTo<at::ScalarType>(out_dtypes[i]);
    if (tensor.scalar_type() != target_dtype) {
      tensor = AtenToCopy(tensor, target_dtype, /*layout=*/std::nullopt,
                          /*device=*/std::nullopt, /*pin_memory=*/std::nullopt,
                          /*non_blocking=*/false,
                          /*memory_format=*/std::nullopt);
    }
    result.push_back(std::move(tensor));
  }
  return result;
}

absl::Status ForeachAssignToTensor(std::vector<DeviceBufferRef> result_buffers,
                                   at::TensorList self, DtypeSpan out_dtypes) {
  for (auto i = 0; i < result_buffers.size(); ++i) {
    at::Tensor tensor = MakeTensor(std::move(result_buffers[i]));
    at::ScalarType target_dtype = ConvertTo<at::ScalarType>(out_dtypes[i]);
    if (tensor.scalar_type() != target_dtype) {
      tensor = AtenToCopy(tensor, target_dtype, /*layout=*/std::nullopt,
                          /*device=*/std::nullopt, /*pin_memory=*/std::nullopt,
                          /*non_blocking=*/false,
                          /*memory_format=*/std::nullopt);
    }
    at::Tensor& self_tensor = const_cast<at::Tensor&>(self[i]);
    AtenCopy_(self_tensor, tensor, /*non_blocking=*/false);
  }
  return absl::OkStatus();
}

absl::Status ForeachAssignToTensor(std::vector<DeviceBufferRef> result_buffers,
                                   at::TensorList self) {
  for (int i = 0; i < result_buffers.size(); ++i) {
    TT_RETURN_IF_ERROR(
        AssignBufferToAtTensor(std::move(result_buffers[i]), self[i]));
  }
  return absl::OkStatus();
}

std::vector<absl::Span<const int64_t>> GetDimsList(at::TensorList tensor_list) {
  std::vector<absl::Span<const int64_t>> dims_list;
  dims_list.reserve(tensor_list.size());
  for (int i = 0; i < tensor_list.size(); ++i) {
    dims_list.push_back(tensor_list[i].sizes());
  }
  return dims_list;
}

absl::StatusOr<DtypeVec> GetOutputDtypes(at::TensorList self,
                                         bool cast_integral_to_float = false,
                                         bool cast_complex_to_float = false) {
  const at::ScalarType default_dtype = c10::get_default_dtype_as_scalartype();
  DtypeVec out_dtypes_vec;
  out_dtypes_vec.reserve(self.size());
  for (size_t i = 0; i < self.size(); ++i) {
    c10::ScalarType tensor_type;
    if (cast_integral_to_float && IsIntegral(self[i])) {
      tensor_type = default_dtype;
    } else if (cast_complex_to_float && IsComplex(self[i])) {
      tensor_type = at::toRealValueType(self[i].scalar_type());
    } else {
      tensor_type = self[i].scalar_type();
    }
    TT_ASSIGN_OR_RETURN(auto output_element_type,
                        ConvertTo<mlir::ElementType>(tensor_type));
    out_dtypes_vec.push_back(output_element_type);
  }
  return out_dtypes_vec;
}

absl::StatusOr<DtypeVec> GetOutputDtypes(at::TensorList self,
                                         at::TensorList other,
                                         bool is_div = false) {
  const at::ScalarType default_aten_type =
      c10::get_default_dtype_as_scalartype();
  DtypeVec out_dtypes_vec;
  out_dtypes_vec.reserve(self.size());
  for (size_t i = 0; i < self.size(); ++i) {
    at::ScalarType output_scalar_type;
    if (is_div &&
        c10::isIntegralType(self[i].scalar_type(), /*includeBool=*/true) &&
        c10::isIntegralType(other[i].scalar_type(), /*includeBool=*/true)) {
      output_scalar_type = default_aten_type;
    } else {
      output_scalar_type =
          c10::promoteTypes(self[i].scalar_type(), other[i].scalar_type());
    }
    TT_ASSIGN_OR_RETURN(const auto output_element_type,
                        ConvertTo<mlir::ElementType>(output_scalar_type));
    out_dtypes_vec.push_back(output_element_type);
  }
  return out_dtypes_vec;
}

absl::StatusOr<DtypeVec> GetOutputDtypes(at::TensorList self,
                                         const at::Scalar& scalar,
                                         bool is_div = false) {
  const at::ScalarType default_aten_type =
      c10::get_default_dtype_as_scalartype();
  DtypeVec out_dtypes_vec;
  out_dtypes_vec.reserve(self.size());
  for (size_t i = 0; i < self.size(); ++i) {
    at::ScalarType output_scalar_type;
    if (is_div &&
        c10::isIntegralType(self[i].scalar_type(), /*includeBool=*/true) &&
        c10::isIntegralType(scalar.type(), /*includeBool=*/true)) {
      output_scalar_type = default_aten_type;
    } else {
      output_scalar_type = GetOutputDtypeFromTensorAndScalar(self[i], scalar);
    }
    TT_ASSIGN_OR_RETURN(const auto output_element_type,
                        ConvertTo<mlir::ElementType>(output_scalar_type));
    out_dtypes_vec.push_back(output_element_type);
  }
  return out_dtypes_vec;
}

absl::StatusOr<DtypeVec> GetOutputDtypes(at::TensorList self,
                                         at::ArrayRef<at::Scalar> scalars,
                                         bool is_div = false) {
  const at::ScalarType default_aten_type =
      c10::get_default_dtype_as_scalartype();
  DtypeVec out_dtypes_vec;
  out_dtypes_vec.reserve(self.size());
  for (size_t i = 0; i < self.size(); ++i) {
    at::ScalarType output_scalar_type;
    if (is_div &&
        c10::isIntegralType(self[i].scalar_type(), /*includeBool=*/true) &&
        c10::isIntegralType(scalars[i].type(), /*includeBool=*/true)) {
      output_scalar_type = default_aten_type;
    } else {
      output_scalar_type =
          GetOutputDtypeFromTensorAndScalar(self[i], scalars[i]);
    }
    TT_ASSIGN_OR_RETURN(const auto output_element_type,
                        ConvertTo<mlir::ElementType>(output_scalar_type));
    out_dtypes_vec.push_back(output_element_type);
  }
  return out_dtypes_vec;
}

// Helper functions for input validation.

absl::Status CheckScalarType(mlir::ElementType out_dtype,
                             mlir::ElementType compute_dtype,
                             at::ScalarType tensor_type,
                             at::ScalarType scalar_type) {
  TT_RET_CHECK(out_dtype == compute_dtype, error::kInvalidArgument)
      << "expected the scalar dtype to be castable to the tensor dtype "
         "(e.g. bool to int or int to float), got "
      << ToString(scalar_type) << " and " << ToString(tensor_type);
  return absl::OkStatus();
}

absl::Status CheckInplaceScalarType(at::TensorList self,
                                    const at::Scalar& scalar) {
  TT_ASSIGN_OR_RETURN(auto out_dtypes, GetOutputDtypes(self));
  TT_ASSIGN_OR_RETURN(const auto result_out_dtypes,
                      GetOutputDtypes(self, scalar));
  for (size_t i = 0; i < self.size(); ++i) {
    TT_RETURN_IF_ERROR(CheckScalarType(out_dtypes[i], result_out_dtypes[i],
                                       self[i].scalar_type(), scalar.type()));
  }
  return absl::OkStatus();
}

absl::Status CheckInplaceScalarType(at::TensorList self,
                                    at::ArrayRef<at::Scalar> scalars) {
  TT_ASSIGN_OR_RETURN(auto out_dtypes, GetOutputDtypes(self));
  TT_ASSIGN_OR_RETURN(const auto result_out_dtypes,
                      GetOutputDtypes(self, scalars));
  for (size_t i = 0; i < self.size(); ++i) {
    TT_RETURN_IF_ERROR(CheckScalarType(out_dtypes[i], result_out_dtypes[i],
                                       self[i].scalar_type(),
                                       scalars[i].type()));
  }
  return absl::OkStatus();
}

inline absl::Status CheckNotBool(const at::Scalar& scalar,
                                 const std::string_view arg_name) {
  TT_RET_CHECK(!IsBool(scalar), error::kInvalidArgument)
      << "expected the " << arg_name << " argument not to be bool, got "
      << ToString(scalar);
  return absl::OkStatus();
}

absl::Status CheckNotBool(at::ArrayRef<at::Scalar> scalars,
                          const std::string_view arg_name) {
  const Indices& bool_indices =
      FilterIndices(scalars.size(),
                    [scalars](const int64_t i) { return IsBool(scalars[i]); });

  TT_RET_CHECK(bool_indices.empty(), error::kInvalidArgument)
      << "expected all " << scalars.size() << " scalars in the " << arg_name
      << " list not to be bool, got "
      << FormatCount(bool_indices.size(), /* singular= */ "bool scalar",
                     /* plural= */ "bool scalars")
      << ": "
      << GetValueAtIndexErrorStr(bool_indices, [scalars](const int64_t i) {
           return ToString(scalars[i]);
         });

  return absl::OkStatus();
}

template <typename IsType>
absl::Status CheckTensorsNotTypeImpl(at::TensorList tensors,
                                     const std::string_view arg_name,
                                     const IsType& is_type,
                                     const std::string_view type_name) {
  const Indices& bad_indices = FilterIndices(
      tensors.size(),
      [tensors, &is_type](const int64_t i) { return is_type(tensors[i]); });

  TT_RET_CHECK(bad_indices.empty(), error::kInvalidArgument)
      << "expected all " << tensors.size() << " tensors in the " << arg_name
      << " list not to be " << type_name << ", got "
      << FormatCount(bad_indices.size(),
                     /* singular= */ absl::StrCat(type_name, " tensor"),
                     /* plural= */ absl::StrCat(type_name, " tensors"))
      << ": "
      << GetValueAtIndexErrorStr(bad_indices, [tensors](const int64_t i) {
           return ToString(tensors[i].scalar_type());
         });

  return absl::OkStatus();
}

absl::Status CheckNotBool(at::TensorList tensors,
                          const std::string_view arg_name) {
  TT_RETURN_IF_ERROR(CheckTensorsNotTypeImpl(tensors, arg_name,
                                             /* is_type= */ IsBool<at::Tensor>,
                                             /* type_name= */ "bool"));
  return absl::OkStatus();
}

absl::Status CheckNotIntegral(at::TensorList tensors,
                              const std::string_view arg_name) {
  TT_RETURN_IF_ERROR(
      CheckTensorsNotTypeImpl(tensors, arg_name,
                              /* is_type= */ IsIntegral<at::Tensor>,
                              /* type_name= */ "integral"));
  return absl::OkStatus();
}

absl::Status CheckNotComplex(at::TensorList tensors,
                             const std::string_view arg_name) {
  TT_RETURN_IF_ERROR(
      CheckTensorsNotTypeImpl(tensors, arg_name,
                              /* is_type= */ IsComplex<at::Tensor>,
                              /* type_name= */ "complex"));
  return absl::OkStatus();
}

absl::Status CheckPairwiseAddcdivAtLeastOneNotIntegral(
    at::TensorList tensors1, at::TensorList tensors2) {
  const Indices& bad_indices =
      FilterIndices(tensors1.size(), [tensors1, tensors2](const int64_t i) {
        return IsIntegral(tensors1[i]) && IsIntegral(tensors2[i]);
      });

  TT_RET_CHECK(bad_indices.empty(), error::kInvalidArgument)
      << "expected at least one non-integral tensor in each of the "
      << tensors1.size()
      << " dividend (second tensor list) and divisor (third tensor list) "
         "pairs, got "
      << FormatCount(bad_indices.size(),
                     /* singular= */ "integral dividend-divisor tensor pair",
                     /* plural= */ "integral dividend-divisor tensor pairs")
      << ": "
      << GetValueAtIndexErrorStr(bad_indices, [tensors1,
                                               tensors2](const int64_t i) {
           return absl::StrCat("(", ToString(tensors1[i].scalar_type()), ", ",
                               ToString(tensors2[i].scalar_type()), ")");
         });

  return absl::OkStatus();
}

absl::StatusOr<std::vector<DeviceBufferRef>> ForeachUnaryOp(
    at::TensorList self, DtypeSpan out_dtypes,
    absl::AnyInvocable<absl::StatusOr<mlir::MlirOp>(mlir::MlirOp,
                                                    mlir::ElementType) const>
        tensor_transform,
    // OpName to use for dispatching. If omitted, use the op name from the
    // active TT_KERNEL() context.
    std::optional<OpName> op_name = std::nullopt,
    // If true, cast all inputs to the corresponding output dtype before
    // applying the tensor_transform.
    bool cast_inputs = true) {
  auto op_builder =
      [out_dtypes = DtypeVec(out_dtypes.begin(), out_dtypes.end()), cast_inputs,
       tensor_transform = std::move(tensor_transform)](
          absl::Span<mlir::MlirOp> inputs, mlir::MlirBuilder& builder)
      -> absl::StatusOr<mlir::SmallVector<mlir::MlirOp>> {
    mlir::SmallVector<mlir::MlirOp> results;
    results.reserve(inputs.size());
    for (int i = 0; i < inputs.size(); ++i) {
      mlir::MlirOp input = inputs[i];
      if (cast_inputs) {
        TT_ASSIGN_OR_RETURN(input, CastIfNeeded(input, out_dtypes[i]));
      }
      TT_ASSIGN_OR_RETURN(mlir::MlirOp result,
                          tensor_transform(input, out_dtypes[i]));
      results.push_back(result);
    }
    return results;
  };

  std::vector<at::Tensor> inputs(self.begin(), self.end());
  const auto out_dims_list = GetDimsList(self);
  DispatchOpOptions<torch_tpu::kDynamicSize> options = {
      .op_name = op_name,
      .out_dtypes = out_dtypes,
      .out_dims_list = out_dims_list,
      .op_param_cache_keys = OpParamCacheKeys::Empty()};

  TT_ASSIGN_OR_RETURN(std::vector<DeviceBufferRef> result_buffers,
                      (DispatchOp<kDynamicSize, kDynamicSize>(
                          std::move(op_builder), inputs, std::move(options))));
  return result_buffers;
}

absl::StatusOr<std::vector<DeviceBufferRef>> ForeachUnaryOp(
    at::TensorList self, DtypeSpan out_dtypes,
    absl::AnyInvocable<absl::StatusOr<mlir::MlirOp>(mlir::MlirOp) const>
        tensor_transform,
    // OpName to use for dispatching. If omitted, use the op name from the
    // active TT_KERNEL() context.
    std::optional<OpName> op_name = std::nullopt,
    // If true, cast all inputs to the corresponding output dtype before
    // applying the tensor_transform.
    bool cast_inputs = true) {
  auto tmp_tensor_transform = [tensor_transform = std::move(tensor_transform)](
                                  mlir::MlirOp mlir_op, mlir::ElementType)
      -> absl::StatusOr<mlir::MlirOp> { return tensor_transform(mlir_op); };
  return ForeachUnaryOp(self, out_dtypes, std::move(tmp_tensor_transform),
                        op_name, cast_inputs);
}

std::vector<DeviceBufferRef> ForeachAddList(
    at::TensorList self, at::TensorList other,
    MaybePromotedScalar promoted_alpha, DtypeSpan out_dtypes,
    OpParamCacheKeys param_keys = OpParamCacheKeys::Empty()) {
  // self and other are guaranteed to have the same size.
  // The error is handled by the upstream torch.
  size_t num_tensors = self.size();

  // The op builder.
  std::vector<at::Tensor> inputs(self.begin(), self.end());
  inputs.insert(inputs.end(), other.begin(), other.end());
  bool alpha_is_one = promoted_alpha.IsOne();
  bool alpha_is_minus_one = promoted_alpha.IsMinusOne();
  // We create different Shlo based on whether alpha is one, minus one or not.
  // Alpha is not multiplied if equal to 1.0 or -1.0.
  TT_THROW_IF_ERROR(param_keys.SetParam("alpha_is_one", alpha_is_one));
  TT_THROW_IF_ERROR(
      param_keys.SetParam("alpha_is_minus_one", alpha_is_minus_one));
  if (!alpha_is_one && !alpha_is_minus_one) {
    for (auto i = 0; i < num_tensors; ++i) {
      at::ScalarType scalar_type = ConvertTo<at::ScalarType>(out_dtypes[i]);
      TT_ASSIGN_OR_THROW(at::Tensor alpha_tensor,
                         promoted_alpha.GetTensor(scalar_type));
      inputs.push_back(alpha_tensor);
    }
  }

  auto op_builder =
      [alpha_is_one, alpha_is_minus_one, num_tensors,
       out_dtypes = DtypeVec(out_dtypes.begin(), out_dtypes.end())](
          absl::Span<mlir::MlirOp> inputs, mlir::MlirBuilder& builder)
      -> absl::StatusOr<mlir::SmallVector<mlir::MlirOp>> {
    absl::Span<mlir::MlirOp> self_ops = inputs.subspan(0, num_tensors);
    absl::Span<mlir::MlirOp> other_ops =
        inputs.subspan(num_tensors, num_tensors);

    // If alpha is 1.0, do a simple addition without multiplying by alpha.
    if (alpha_is_one) {
      return BuildForeachShlo(self_ops, other_ops, out_dtypes,
                              mlir::stablehlo::Add, builder);
    }

    // If alpha is -1.0, do a simple subtraction without multiplying by alpha.
    if (alpha_is_minus_one) {
      return BuildForeachShlo(self_ops, other_ops, out_dtypes,
                              mlir::stablehlo::Subtract, builder);
    }

    absl::Span<mlir::MlirOp> alpha_ops =
        inputs.subspan(2 * num_tensors, 3 * num_tensors);
    TT_ASSIGN_OR_RETURN(auto new_other_ops,
                        BuildForeachShlo(other_ops, alpha_ops, out_dtypes,
                                         mlir::stablehlo::Mul, builder));
    return BuildForeachShlo(self_ops, absl::MakeSpan(new_other_ops), out_dtypes,
                            mlir::stablehlo::Add, builder);
  };

  // Dispatch the op and prepare results.
  const auto out_dims_list = GetDimsList(self);
  DispatchOpOptions<kDynamicSize> options = {
      // Share the same OpName for all ForeachAddList() calls, as the underlying
      // Shlo is the same.
      .op_name = OpName::kForeachAddCommon,
      .out_dtypes = out_dtypes,
      .out_dims_list = absl::MakeConstSpan(out_dims_list),
      .op_param_cache_keys = std::move(param_keys),
  };
  TT_ASSIGN_OR_THROW(auto result_buffers,
                     (DispatchOp<kDynamicSize, kDynamicSize>(
                         std::move(op_builder), inputs, std::move(options))));
  return result_buffers;
}

std::vector<DeviceBufferRef> ForeachAddcdiv(at::TensorList self,
                                            at::TensorList tensor1,
                                            at::TensorList tensor2,
                                            at::TensorList value,
                                            DtypeSpan out_dtypes) {
  size_t num_tensors = self.size();

  std::vector<at::Tensor> inputs;
  inputs.reserve(4 * num_tensors);
  inputs.insert(inputs.end(), self.begin(), self.end());
  inputs.insert(inputs.end(), tensor1.begin(), tensor1.end());
  inputs.insert(inputs.end(), tensor2.begin(), tensor2.end());
  inputs.insert(inputs.end(), value.begin(), value.end());

  auto op_builder =
      [num_tensors,
       out_dtypes = DtypeVec(out_dtypes.begin(), out_dtypes.end())](
          absl::Span<mlir::MlirOp> inputs, mlir::MlirBuilder& builder)
      -> absl::StatusOr<mlir::SmallVector<mlir::MlirOp>> {
    absl::Span<mlir::MlirOp> self_ops = inputs.subspan(0, num_tensors);
    absl::Span<mlir::MlirOp> tensor1_ops =
        inputs.subspan(num_tensors, num_tensors);
    absl::Span<mlir::MlirOp> tensor2_ops =
        inputs.subspan(2 * num_tensors, num_tensors);
    absl::Span<mlir::MlirOp> value_ops =
        inputs.subspan(3 * num_tensors, num_tensors);

    return BuildForeachAddcdivShlo(self_ops, tensor1_ops, tensor2_ops,
                                   value_ops, out_dtypes, builder);
  };

  const auto out_dims_list = GetDimsList(self);
  DispatchOpOptions<kDynamicSize> options = {
      // Share the same OpName for all ForeachAddcdiv() calls, as the underlying
      // Shlo is the same.
      .op_name = OpName::kForeachAddcdivTensor,
      .out_dtypes = out_dtypes,
      .out_dims_list = out_dims_list,
      .op_param_cache_keys = OpParamCacheKeys::Empty(),
  };

  TT_ASSIGN_OR_THROW(auto result_buffers,
                     (DispatchOp<kDynamicSize, kDynamicSize>(
                         std::move(op_builder), inputs, std::move(options))));
  return result_buffers;
}

std::vector<DeviceBufferRef> ForeachAddcmul(at::TensorList self,
                                            at::TensorList tensor1,
                                            at::TensorList tensor2,
                                            at::TensorList value,
                                            DtypeSpan out_dtypes) {
  size_t num_tensors = self.size();

  std::vector<at::Tensor> inputs;
  inputs.reserve(4 * num_tensors);
  inputs.insert(inputs.end(), self.begin(), self.end());
  inputs.insert(inputs.end(), tensor1.begin(), tensor1.end());
  inputs.insert(inputs.end(), tensor2.begin(), tensor2.end());
  inputs.insert(inputs.end(), value.begin(), value.end());

  auto op_builder =
      [num_tensors,
       out_dtypes = DtypeVec(out_dtypes.begin(), out_dtypes.end())](
          absl::Span<mlir::MlirOp> inputs, mlir::MlirBuilder& builder)
      -> absl::StatusOr<mlir::SmallVector<mlir::MlirOp>> {
    absl::Span<mlir::MlirOp> self_ops = inputs.subspan(0, num_tensors);
    absl::Span<mlir::MlirOp> tensor1_ops =
        inputs.subspan(num_tensors, num_tensors);
    absl::Span<mlir::MlirOp> tensor2_ops =
        inputs.subspan(2 * num_tensors, num_tensors);
    absl::Span<mlir::MlirOp> value_ops =
        inputs.subspan(3 * num_tensors, num_tensors);

    return BuildForeachAddcmulShlo(self_ops, tensor1_ops, tensor2_ops,
                                   value_ops, out_dtypes, builder);
  };

  const auto out_dims_list = GetDimsList(self);
  DispatchOpOptions<kDynamicSize> options = {
      // Share the same OpName for all calls ForeachAddcmul(), as the
      // underlying Shlo is the same.
      .op_name = OpName::kForeachAddcmulTensor,
      .out_dtypes = out_dtypes,
      .out_dims_list = out_dims_list,
      .op_param_cache_keys = OpParamCacheKeys::Empty(),
  };

  TT_ASSIGN_OR_THROW(auto result_buffers,
                     (DispatchOp<kDynamicSize, kDynamicSize>(
                         std::move(op_builder), inputs, std::move(options))));
  return result_buffers;
}

std::vector<DeviceBufferRef> ForeachDiv(at::TensorList self,
                                        at::TensorList other,
                                        DtypeSpan out_dtypes) {
  size_t num_tensors = self.size();

  std::vector<at::Tensor> inputs(self.begin(), self.end());
  inputs.insert(inputs.end(), other.begin(), other.end());

  at::ScalarType default_aten_type = c10::get_default_dtype_as_scalartype();
  TT_ASSIGN_OR_THROW(auto default_dtype,
                     ConvertTo<mlir::ElementType>(default_aten_type));

  auto op_builder =
      [num_tensors, out_dtypes = DtypeVec(out_dtypes.begin(), out_dtypes.end()),
       default_dtype](absl::Span<mlir::MlirOp> inputs,
                      mlir::MlirBuilder& builder)
      -> absl::StatusOr<mlir::SmallVector<mlir::MlirOp>> {
    absl::Span<mlir::MlirOp> self_ops = inputs.subspan(0, num_tensors);
    absl::Span<mlir::MlirOp> other_ops =
        inputs.subspan(num_tensors, num_tensors);

    mlir::SmallVector<mlir::MlirOp> results;
    results.reserve(num_tensors);
    for (size_t i = 0; i < num_tensors; ++i) {
      TT_ASSIGN_OR_RETURN(
          (auto [promoted_self_op, promoted_other_op]),
          ConvertIfIntegers(self_ops[i], other_ops[i], default_dtype));
      TT_ASSIGN_OR_RETURN(promoted_self_op,
                          CastIfNeeded(promoted_self_op, out_dtypes[i]));
      TT_ASSIGN_OR_RETURN(promoted_other_op,
                          CastIfNeeded(promoted_other_op, out_dtypes[i]));
      TT_ASSIGN_OR_RETURN(mlir::MlirOp div_op,
                          BuildDivShlo(promoted_self_op, promoted_other_op));
      results.push_back(div_op);
    }
    return results;
  };

  const auto out_dims_list = GetDimsList(self);
  DispatchOpOptions<kDynamicSize> options = {
      // Share the same OpName for all ForeachDiv() calls, as the underlying
      // Shlo is the same.
      .op_name = OpName::kForeachDivCommon,
      .out_dtypes = out_dtypes,
      .out_dims_list = out_dims_list,
      .op_param_cache_keys = OpParamCacheKeys::Empty(),
  };

  TT_ASSIGN_OR_THROW(auto result_buffers,
                     (DispatchOp<kDynamicSize, kDynamicSize>(
                         std::move(op_builder), inputs, std::move(options))));
  return result_buffers;
}

std::vector<DeviceBufferRef> ForeachLerp(at::TensorList self,
                                         at::TensorList other,
                                         at::TensorList weight,
                                         DtypeSpan out_dtypes) {
  size_t num_tensors = self.size();

  std::vector<at::Tensor> inputs;
  inputs.reserve(3 * num_tensors);
  inputs.insert(inputs.end(), self.begin(), self.end());
  inputs.insert(inputs.end(), other.begin(), other.end());
  inputs.insert(inputs.end(), weight.begin(), weight.end());

  auto op_builder =
      [num_tensors,
       out_dtypes = DtypeVec(out_dtypes.begin(), out_dtypes.end())](
          absl::Span<mlir::MlirOp> inputs, mlir::MlirBuilder& builder)
      -> absl::StatusOr<mlir::SmallVector<mlir::MlirOp>> {
    absl::Span<mlir::MlirOp> self_ops = inputs.subspan(0, num_tensors);
    absl::Span<mlir::MlirOp> other_ops =
        inputs.subspan(num_tensors, num_tensors);
    absl::Span<mlir::MlirOp> weight_ops =
        inputs.subspan(2 * num_tensors, num_tensors);

    return BuildForeachLerpShlo(self_ops, other_ops, weight_ops, out_dtypes,
                                builder);
  };

  const auto out_dims_list = GetDimsList(self);
  DispatchOpOptions<kDynamicSize> options = {
      // Share the same OpName for all ForeachLerp() calls, as the underlying
      // Shlo is the same.
      .op_name = OpName::kForeachLerpCommon,
      .out_dtypes = out_dtypes,
      .out_dims_list = out_dims_list,
      .op_param_cache_keys = OpParamCacheKeys::Empty(),
  };

  TT_ASSIGN_OR_THROW(auto result_buffers,
                     (DispatchOp<kDynamicSize, kDynamicSize>(
                         std::move(op_builder), inputs, std::move(options))));
  return result_buffers;
}

std::vector<DeviceBufferRef> ForeachMulList(at::TensorList self,
                                            at::TensorList other,
                                            DtypeSpan out_dtypes) {
  // self and other are guaranteed to have the same size.
  // The error is handled by the upstream torch.
  const size_t num_tensors = self.size();
  auto op_builder =
      [num_tensors,
       out_dtypes = DtypeVec(out_dtypes.begin(), out_dtypes.end())](
          absl::Span<mlir::MlirOp> inputs, mlir::MlirBuilder& builder)
      -> absl::StatusOr<mlir::SmallVector<mlir::MlirOp>> {
    absl::Span<mlir::MlirOp> self_ops = inputs.subspan(0, num_tensors);
    absl::Span<mlir::MlirOp> other_ops =
        inputs.subspan(num_tensors, num_tensors);
    return BuildForeachShlo(self_ops, other_ops, out_dtypes,
                            mlir::stablehlo::Mul, builder);
  };

  std::vector<at::Tensor> inputs(self.begin(), self.end());
  inputs.insert(inputs.end(), other.begin(), other.end());
  const auto out_dims_list = GetDimsList(self);
  DispatchOpOptions<kDynamicSize> options = {
      // Share the same OpName for all ForeachMulList() calls, as the underlying
      // Shlo is the same.
      .op_name = OpName::kForeachMulCommon,
      .out_dtypes = out_dtypes,
      .out_dims_list = out_dims_list,
      .op_param_cache_keys = OpParamCacheKeys::Empty(),
  };

  // Dispatch the op and prepare results.
  TT_ASSIGN_OR_THROW(auto result_buffers,
                     (DispatchOp<kDynamicSize, kDynamicSize>(
                         std::move(op_builder), inputs, std::move(options))));
  return result_buffers;
}

}  // namespace

std::vector<at::Tensor> AtenForeachAbs(at::TensorList self) {
  TT_KERNEL(OpName::kForeachAbs, _, (self), {
    TT_ASSIGN_OR_THROW(auto out_dtypes,
                       GetOutputDtypes(self, /*cast_integral_to_float=*/false,
                                       /*cast_complex_to_float=*/true));
    TT_ASSIGN_OR_THROW(auto result_buffers,
                       ForeachUnaryOp(self, out_dtypes, BuildAbsShlo,
                                      // Share OpName with AtenForeachAbs_() as
                                      // the underlying Shlo is the same.
                                      OpName::kForeachAbsCommon,
                                      /*cast_inputs=*/false));
    return ForeachConvertToTensor(result_buffers, out_dtypes);
  });
}

void AtenForeachAbs_(at::TensorList self) {
  TT_KERNEL(OpName::kForeachAbs_, _, (self), {
    // _foreach_abs_ does not support complex dtype.
    TT_THROW_IF_ERROR(CheckNotComplex(self, /* arg_name= */ "self"));
    TT_ASSIGN_OR_THROW(auto out_dtypes, GetOutputDtypes(self));
    TT_ASSIGN_OR_THROW(auto result_buffers,
                       ForeachUnaryOp(self, std::move(out_dtypes), BuildAbsShlo,
                                      // Share OpName with AtenForeachAbs() as
                                      // the underlying Shlo is the same.
                                      OpName::kForeachAbsCommon,
                                      /*cast_inputs=*/false));
    TT_THROW_IF_ERROR(ForeachAssignToTensor(result_buffers, self));
  });
}

std::vector<at::Tensor> AtenForeachAcos(at::TensorList self) {
  TT_KERNEL(OpName::kForeachAcos, _, (self), {
    TT_ASSIGN_OR_THROW(auto out_dtypes,
                       GetOutputDtypes(self, /*cast_integral_to_float=*/true));
    TT_ASSIGN_OR_THROW(auto result_buffers,
                       ForeachUnaryOp(self, out_dtypes, BuildAcosShlo));
    return ForeachConvertToTensor(result_buffers, out_dtypes);
  });
}

void AtenForeachAcos_(at::TensorList self) {
  TT_KERNEL(OpName::kForeachAcos_, _, (self), {
    TT_THROW_IF_ERROR(CheckNotIntegral(self, /* arg_name= */ "self"));
    TT_ASSIGN_OR_THROW(auto out_dtypes, GetOutputDtypes(self));
    TT_ASSIGN_OR_THROW(
        auto result_buffers,
        ForeachUnaryOp(self, std::move(out_dtypes), BuildAcosShlo,
                       // Share OpName with AtenForeachAcos() as
                       // the underlying Shlo is the same.
                       OpName::kForeachAcos));
    TT_THROW_IF_ERROR(ForeachAssignToTensor(result_buffers, self));
  });
}

std::vector<at::Tensor> AtenForeachAsin(at::TensorList self) {
  TT_KERNEL(OpName::kForeachAsin, _, (self), {
    TT_ASSIGN_OR_THROW(auto out_dtypes,
                       GetOutputDtypes(self, /*cast_integral_to_float=*/true));
    TT_ASSIGN_OR_THROW(auto result_buffers,
                       ForeachUnaryOp(self, out_dtypes, BuildAsinShlo));
    return ForeachConvertToTensor(result_buffers, out_dtypes);
  });
}

void AtenForeachAsin_(at::TensorList self) {
  TT_KERNEL(OpName::kForeachAsin_, _, (self), {
    TT_THROW_IF_ERROR(CheckNotIntegral(self, /* arg_name= */ "self"));
    TT_ASSIGN_OR_THROW(auto out_dtypes, GetOutputDtypes(self));
    TT_ASSIGN_OR_THROW(
        auto result_buffers,
        ForeachUnaryOp(self, std::move(out_dtypes), BuildAsinShlo,
                       // Share OpName with AtenForeachAsin() as
                       // the underlying Shlo is the same.
                       OpName::kForeachAsin));
    TT_THROW_IF_ERROR(ForeachAssignToTensor(result_buffers, self));
  });
}

std::vector<at::Tensor> AtenForeachAtan(at::TensorList self) {
  TT_KERNEL(OpName::kForeachAtan, _, (self), {
    TT_ASSIGN_OR_THROW(auto out_dtypes,
                       GetOutputDtypes(self, /*cast_integral_to_float=*/true));
    TT_ASSIGN_OR_THROW(auto result_buffers,
                       ForeachUnaryOp(self, out_dtypes, BuildAtanShlo));
    return ForeachConvertToTensor(result_buffers, out_dtypes);
  });
}

void AtenForeachAtan_(at::TensorList self) {
  TT_KERNEL(OpName::kForeachAtan_, _, (self), {
    TT_THROW_IF_ERROR(CheckNotIntegral(self, /* arg_name= */ "self"));
    TT_ASSIGN_OR_THROW(auto out_dtypes, GetOutputDtypes(self));
    TT_ASSIGN_OR_THROW(
        auto result_buffers,
        ForeachUnaryOp(self, std::move(out_dtypes), BuildAtanShlo,
                       // Share OpName with AtenForeachAtan() as
                       // the underlying Shlo is the same.
                       OpName::kForeachAtan));
    TT_THROW_IF_ERROR(ForeachAssignToTensor(result_buffers, self));
  });
}

std::vector<at::Tensor> AtenForeachCeil(at::TensorList self) {
  TT_KERNEL(OpName::kForeachCeil, _, (self), {
    TT_THROW_IF_ERROR(CheckNotComplex(self, /*arg_name=*/"self"));
    TT_ASSIGN_OR_THROW(auto out_dtypes, GetOutputDtypes(self));
    TT_ASSIGN_OR_THROW(auto result_buffers,
                       ForeachUnaryOp(self, out_dtypes, BuildCeilShlo));
    return ForeachConvertToTensor(result_buffers, out_dtypes);
  });
}

void AtenForeachCeil_(at::TensorList self) {
  TT_KERNEL(OpName::kForeachCeil_, _, (self), {
    TT_THROW_IF_ERROR(CheckNotComplex(self, /*arg_name=*/"self"));
    TT_ASSIGN_OR_THROW(auto out_dtypes, GetOutputDtypes(self));
    TT_ASSIGN_OR_THROW(
        auto result_buffers,
        ForeachUnaryOp(self, std::move(out_dtypes), BuildCeilShlo,
                       // Share OpName with AtenForeachCeil() as
                       // the underlying Shlo is the same.
                       OpName::kForeachCeil));
    TT_THROW_IF_ERROR(ForeachAssignToTensor(result_buffers, self));
  });
}

std::vector<at::Tensor> AtenForeachCos(at::TensorList self) {
  TT_KERNEL(OpName::kForeachCos, _, (self), {
    TT_ASSIGN_OR_THROW(auto out_dtypes,
                       GetOutputDtypes(self, /*cast_integral_to_float=*/true));
    TT_ASSIGN_OR_THROW(auto result_buffers,
                       ForeachUnaryOp(self, out_dtypes, BuildCosShlo));
    return ForeachConvertToTensor(result_buffers, out_dtypes);
  });
}

void AtenForeachCos_(at::TensorList self) {
  TT_KERNEL(OpName::kForeachCos_, _, (self), {
    TT_THROW_IF_ERROR(CheckNotIntegral(self, /* arg_name= */ "self"));
    TT_ASSIGN_OR_THROW(auto out_dtypes, GetOutputDtypes(self));
    TT_ASSIGN_OR_THROW(auto result_buffers,
                       ForeachUnaryOp(self, std::move(out_dtypes), BuildCosShlo,
                                      // Share OpName with AtenForeachCos() as
                                      // the underlying Shlo is the same.
                                      OpName::kForeachCos));
    TT_THROW_IF_ERROR(ForeachAssignToTensor(result_buffers, self));
  });
}

std::vector<at::Tensor> AtenForeachCosh(at::TensorList self) {
  TT_KERNEL(OpName::kForeachCosh, _, (self), {
    TT_ASSIGN_OR_THROW(auto out_dtypes,
                       GetOutputDtypes(self, /*cast_integral_to_float=*/true));
    TT_ASSIGN_OR_THROW(auto result_buffers,
                       ForeachUnaryOp(self, out_dtypes, BuildCoshShlo));
    return ForeachConvertToTensor(result_buffers, out_dtypes);
  });
}

void AtenForeachCosh_(at::TensorList self) {
  TT_KERNEL(OpName::kForeachCosh_, _, (self), {
    TT_THROW_IF_ERROR(CheckNotIntegral(self, /* arg_name= */ "self"));
    TT_ASSIGN_OR_THROW(auto out_dtypes, GetOutputDtypes(self));
    TT_ASSIGN_OR_THROW(
        auto result_buffers,
        ForeachUnaryOp(self, std::move(out_dtypes), BuildCoshShlo,
                       // Share OpName with AtenForeachCosh() as
                       // the underlying Shlo is the same.
                       OpName::kForeachCosh));
    TT_THROW_IF_ERROR(ForeachAssignToTensor(result_buffers, self));
  });
}

std::vector<at::Tensor> AtenForeachErf(at::TensorList self) {
  TT_KERNEL(OpName::kForeachErf, _, (self), {
    TT_THROW_IF_ERROR(CheckNotComplex(self, /*arg_name=*/"self"));
    TT_ASSIGN_OR_THROW(auto out_dtypes,
                       GetOutputDtypes(self, /*cast_integral_to_float=*/true));
    TT_ASSIGN_OR_THROW(auto result_buffers,
                       ForeachUnaryOp(self, out_dtypes, BuildErfShlo));
    return ForeachConvertToTensor(result_buffers, out_dtypes);
  });
}

void AtenForeachErf_(at::TensorList self) {
  TT_KERNEL(OpName::kForeachErf_, _, (self), {
    TT_THROW_IF_ERROR(CheckNotComplex(self, /*arg_name=*/"self"));
    TT_THROW_IF_ERROR(CheckNotIntegral(self, /* arg_name= */ "self"));
    TT_ASSIGN_OR_THROW(auto out_dtypes, GetOutputDtypes(self));
    TT_ASSIGN_OR_THROW(auto result_buffers,
                       ForeachUnaryOp(self, std::move(out_dtypes), BuildErfShlo,
                                      // Share OpName with AtenForeachErf() as
                                      // the underlying Shlo is the same.
                                      OpName::kForeachErf));
    TT_THROW_IF_ERROR(ForeachAssignToTensor(result_buffers, self));
  });
}

std::vector<at::Tensor> AtenForeachErfc(at::TensorList self) {
  TT_KERNEL(OpName::kForeachErfc, _, (self), {
    TT_THROW_IF_ERROR(CheckNotComplex(self, /*arg_name=*/"self"));
    TT_ASSIGN_OR_THROW(auto out_dtypes,
                       GetOutputDtypes(self, /*cast_integral_to_float=*/true));
    TT_ASSIGN_OR_THROW(auto result_buffers,
                       ForeachUnaryOp(self, out_dtypes, BuildErfcShlo));
    return ForeachConvertToTensor(result_buffers, out_dtypes);
  });
}

void AtenForeachErfc_(at::TensorList self) {
  TT_KERNEL(OpName::kForeachErfc_, _, (self), {
    TT_THROW_IF_ERROR(CheckNotComplex(self, /*arg_name=*/"self"));
    TT_THROW_IF_ERROR(CheckNotIntegral(self, /* arg_name= */ "self"));
    TT_ASSIGN_OR_THROW(auto out_dtypes, GetOutputDtypes(self));
    TT_ASSIGN_OR_THROW(
        auto result_buffers,
        ForeachUnaryOp(self, std::move(out_dtypes), BuildErfcShlo,
                       // Share OpName with AtenForeachErfc() as
                       // the underlying Shlo is the same.
                       OpName::kForeachErfc));
    TT_THROW_IF_ERROR(ForeachAssignToTensor(result_buffers, self));
  });
}

std::vector<at::Tensor> AtenForeachExp(at::TensorList self) {
  TT_KERNEL(OpName::kForeachExp, _, (self), {
    TT_ASSIGN_OR_THROW(auto out_dtypes,
                       GetOutputDtypes(self, /*cast_integral_to_float=*/true));
    TT_ASSIGN_OR_THROW(auto result_buffers,
                       ForeachUnaryOp(self, out_dtypes, BuildExpShlo));
    return ForeachConvertToTensor(result_buffers, out_dtypes);
  });
}

void AtenForeachExp_(at::TensorList self) {
  TT_KERNEL(OpName::kForeachExp_, _, (self), {
    TT_THROW_IF_ERROR(CheckNotIntegral(self, /* arg_name= */ "self"));
    TT_ASSIGN_OR_THROW(auto out_dtypes, GetOutputDtypes(self));
    TT_ASSIGN_OR_THROW(auto result_buffers,
                       ForeachUnaryOp(self, std::move(out_dtypes), BuildExpShlo,
                                      // Share OpName with AtenForeachExp() as
                                      // the underlying Shlo is the same.
                                      OpName::kForeachExp));
    TT_THROW_IF_ERROR(ForeachAssignToTensor(result_buffers, self));
  });
}

std::vector<at::Tensor> AtenForeachExpm1(at::TensorList self) {
  TT_KERNEL(OpName::kForeachExpm1, _, (self), {
    TT_ASSIGN_OR_THROW(auto out_dtypes,
                       GetOutputDtypes(self, /*cast_integral_to_float=*/true));
    TT_ASSIGN_OR_THROW(auto result_buffers,
                       ForeachUnaryOp(self, out_dtypes, BuildExpm1Shlo));
    return ForeachConvertToTensor(result_buffers, out_dtypes);
  });
}

void AtenForeachExpm1_(at::TensorList self) {
  TT_KERNEL(OpName::kForeachExpm1_, _, (self), {
    TT_THROW_IF_ERROR(CheckNotIntegral(self, /* arg_name= */ "self"));
    TT_ASSIGN_OR_THROW(auto out_dtypes, GetOutputDtypes(self));
    TT_ASSIGN_OR_THROW(
        auto result_buffers,
        ForeachUnaryOp(self, std::move(out_dtypes), BuildExpm1Shlo,
                       // Share OpName with AtenForeachExpm1() as
                       // the underlying Shlo is the same.
                       OpName::kForeachExpm1));
    TT_THROW_IF_ERROR(ForeachAssignToTensor(result_buffers, self));
  });
}

std::vector<at::Tensor> AtenForeachFloor(at::TensorList self) {
  TT_KERNEL(OpName::kForeachFloor, _, (self), {
    TT_THROW_IF_ERROR(CheckNotComplex(self, /*arg_name=*/"self"));
    TT_ASSIGN_OR_THROW(auto out_dtypes, GetOutputDtypes(self));
    TT_ASSIGN_OR_THROW(auto result_buffers,
                       ForeachUnaryOp(self, out_dtypes, BuildFloorShlo));
    return ForeachConvertToTensor(result_buffers, out_dtypes);
  });
}

void AtenForeachFloor_(at::TensorList self) {
  TT_KERNEL(OpName::kForeachFloor_, _, (self), {
    TT_THROW_IF_ERROR(CheckNotComplex(self, /*arg_name=*/"self"));
    TT_ASSIGN_OR_THROW(auto out_dtypes, GetOutputDtypes(self));
    TT_ASSIGN_OR_THROW(
        auto result_buffers,
        ForeachUnaryOp(self, std::move(out_dtypes), BuildFloorShlo,
                       // Share OpName with AtenForeachFloor() as
                       // the underlying Shlo is the same.
                       OpName::kForeachFloor));
    TT_THROW_IF_ERROR(ForeachAssignToTensor(result_buffers, self));
  });
}

std::vector<at::Tensor> AtenForeachFrac(at::TensorList self) {
  TT_KERNEL(OpName::kForeachFrac, _, (self), {
    TT_THROW_IF_ERROR(CheckNotComplex(self, /*arg_name=*/"self"));
    TT_THROW_IF_ERROR(CheckNotIntegral(self, /* arg_name= */ "self"));
    TT_ASSIGN_OR_THROW(auto out_dtypes,
                       GetOutputDtypes(self, /*cast_integral_to_float=*/true));
    TT_ASSIGN_OR_THROW(auto result_buffers,
                       ForeachUnaryOp(self, out_dtypes, BuildFracShlo));
    return ForeachConvertToTensor(result_buffers, out_dtypes);
  });
}

void AtenForeachFrac_(at::TensorList self) {
  TT_KERNEL(OpName::kForeachFrac_, _, (self), {
    TT_THROW_IF_ERROR(CheckNotComplex(self, /*arg_name=*/"self"));
    TT_THROW_IF_ERROR(CheckNotIntegral(self, /* arg_name= */ "self"));
    TT_ASSIGN_OR_THROW(auto out_dtypes, GetOutputDtypes(self));
    TT_ASSIGN_OR_THROW(
        auto result_buffers,
        ForeachUnaryOp(self, std::move(out_dtypes), BuildFracShlo,
                       // Share OpName with AtenForeachFrac() as
                       // the underlying Shlo is the same.
                       OpName::kForeachFrac));
    TT_THROW_IF_ERROR(ForeachAssignToTensor(result_buffers, self));
  });
}

std::vector<at::Tensor> AtenForeachLgamma(at::TensorList self) {
  TT_KERNEL(OpName::kForeachLgamma, _, (self), {
    TT_THROW_IF_ERROR(CheckNotComplex(self, /*arg_name=*/"self"));
    TT_ASSIGN_OR_THROW(auto out_dtypes,
                       GetOutputDtypes(self, /*cast_integral_to_float=*/true));
    TT_ASSIGN_OR_THROW(auto result_buffers,
                       ForeachUnaryOp(self, out_dtypes, BuildLgammaShlo));
    return ForeachConvertToTensor(result_buffers, out_dtypes);
  });
}

void AtenForeachLgamma_(at::TensorList self) {
  TT_KERNEL(OpName::kForeachLgamma_, _, (self), {
    TT_THROW_IF_ERROR(CheckNotComplex(self, /*arg_name=*/"self"));
    TT_ASSIGN_OR_THROW(auto out_dtypes, GetOutputDtypes(self));
    TT_ASSIGN_OR_THROW(
        auto result_buffers,
        ForeachUnaryOp(self, std::move(out_dtypes), BuildLgammaShlo,
                       // Share OpName with AtenForeachLgamma()
                       // as the underlying Shlo is the same.
                       OpName::kForeachLgamma));
    TT_THROW_IF_ERROR(ForeachAssignToTensor(result_buffers, self));
  });
}

std::vector<at::Tensor> AtenForeachLog(at::TensorList self) {
  TT_KERNEL(OpName::kForeachLog, _, (self), {
    TT_ASSIGN_OR_THROW(auto out_dtypes,
                       GetOutputDtypes(self, /*cast_integral_to_float=*/true));
    TT_ASSIGN_OR_THROW(auto result_buffers,
                       ForeachUnaryOp(self, out_dtypes, BuildLogShlo));
    return ForeachConvertToTensor(result_buffers, out_dtypes);
  });
}

void AtenForeachLog_(at::TensorList self) {
  TT_KERNEL(OpName::kForeachLog_, _, (self), {
    TT_THROW_IF_ERROR(CheckNotIntegral(self, /* arg_name= */ "self"));
    TT_ASSIGN_OR_THROW(auto out_dtypes, GetOutputDtypes(self));
    TT_ASSIGN_OR_THROW(auto result_buffers,
                       ForeachUnaryOp(self, std::move(out_dtypes), BuildLogShlo,
                                      // Share OpName with AtenForeachLog() as
                                      // the underlying Shlo is the same.
                                      OpName::kForeachLog));
    TT_THROW_IF_ERROR(ForeachAssignToTensor(result_buffers, self));
  });
}

std::vector<at::Tensor> AtenForeachLog10(at::TensorList self) {
  TT_KERNEL(OpName::kForeachLog10, _, (self), {
    TT_ASSIGN_OR_THROW(auto out_dtypes,
                       GetOutputDtypes(self, /*cast_integral_to_float=*/true));
    TT_ASSIGN_OR_THROW(auto result_buffers,
                       ForeachUnaryOp(self, out_dtypes, BuildLog10Shlo));
    return ForeachConvertToTensor(result_buffers, out_dtypes);
  });
}

void AtenForeachLog10_(at::TensorList self) {
  TT_KERNEL(OpName::kForeachLog10_, _, (self), {
    TT_THROW_IF_ERROR(CheckNotIntegral(self, /* arg_name= */ "self"));
    TT_ASSIGN_OR_THROW(auto out_dtypes, GetOutputDtypes(self));
    TT_ASSIGN_OR_THROW(
        auto result_buffers,
        ForeachUnaryOp(self, std::move(out_dtypes), BuildLog10Shlo,
                       // Share OpName with AtenForeachLog10() as
                       // the underlying Shlo is the same.
                       OpName::kForeachLog10));
    TT_THROW_IF_ERROR(ForeachAssignToTensor(result_buffers, self));
  });
}

std::vector<at::Tensor> AtenForeachLog1p(at::TensorList self) {
  TT_KERNEL(OpName::kForeachLog1p, _, (self), {
    TT_ASSIGN_OR_THROW(auto out_dtypes,
                       GetOutputDtypes(self, /*cast_integral_to_float=*/true));
    TT_ASSIGN_OR_THROW(auto result_buffers,
                       ForeachUnaryOp(self, out_dtypes, BuildLog1pShlo));
    return ForeachConvertToTensor(result_buffers, out_dtypes);
  });
}

void AtenForeachLog1p_(at::TensorList self) {
  TT_KERNEL(OpName::kForeachLog1p_, _, (self), {
    TT_THROW_IF_ERROR(CheckNotIntegral(self, /* arg_name= */ "self"));
    TT_ASSIGN_OR_THROW(auto out_dtypes, GetOutputDtypes(self));
    TT_ASSIGN_OR_THROW(
        auto result_buffers,
        ForeachUnaryOp(self, std::move(out_dtypes), BuildLog1pShlo,
                       // Share OpName with AtenForeachLog1p() as
                       // the underlying Shlo is the same.
                       OpName::kForeachLog1p));
    TT_THROW_IF_ERROR(ForeachAssignToTensor(result_buffers, self));
  });
}

std::vector<at::Tensor> AtenForeachLog2(at::TensorList self) {
  TT_KERNEL(OpName::kForeachLog2, _, (self), {
    TT_ASSIGN_OR_THROW(auto out_dtypes,
                       GetOutputDtypes(self, /*cast_integral_to_float=*/true));
    TT_ASSIGN_OR_THROW(auto result_buffers,
                       ForeachUnaryOp(self, out_dtypes, BuildLog2Shlo));
    return ForeachConvertToTensor(result_buffers, out_dtypes);
  });
}

void AtenForeachLog2_(at::TensorList self) {
  TT_KERNEL(OpName::kForeachLog2_, _, (self), {
    TT_THROW_IF_ERROR(CheckNotIntegral(self, /* arg_name= */ "self"));
    TT_ASSIGN_OR_THROW(auto out_dtypes, GetOutputDtypes(self));
    TT_ASSIGN_OR_THROW(
        auto result_buffers,
        ForeachUnaryOp(self, std::move(out_dtypes), BuildLog2Shlo,
                       // Share OpName with AtenForeachLog2() as
                       // the underlying Shlo is the same.
                       OpName::kForeachLog2));
    TT_THROW_IF_ERROR(ForeachAssignToTensor(result_buffers, self));
  });
}

std::vector<at::Tensor> AtenForeachNeg(at::TensorList self) {
  TT_KERNEL(OpName::kForeachNeg, _, (self), {
    TT_ASSIGN_OR_THROW(auto out_dtypes, GetOutputDtypes(self));
    TT_ASSIGN_OR_THROW(auto result_buffers,
                       ForeachUnaryOp(self, out_dtypes, BuildNegShlo));
    return ForeachConvertToTensor(result_buffers, out_dtypes);
  });
}

void AtenForeachNeg_(at::TensorList self) {
  TT_KERNEL(OpName::kForeachNeg_, _, (self), {
    TT_ASSIGN_OR_THROW(auto out_dtypes, GetOutputDtypes(self));
    TT_ASSIGN_OR_THROW(auto result_buffers,
                       ForeachUnaryOp(self, std::move(out_dtypes), BuildNegShlo,
                                      // Share OpName with AtenForeachNeg() as
                                      // the underlying Shlo is the same.
                                      OpName::kForeachNeg));
    TT_THROW_IF_ERROR(ForeachAssignToTensor(result_buffers, self));
  });
}

std::vector<at::Tensor> AtenForeachReciprocal(at::TensorList self) {
  TT_KERNEL(OpName::kForeachReciprocal, _, (self), {
    TT_ASSIGN_OR_THROW(auto out_dtypes,
                       GetOutputDtypes(self, /*cast_integral_to_float=*/true));
    TT_ASSIGN_OR_THROW(auto result_buffers,
                       ForeachUnaryOp(self, out_dtypes, BuildReciprocalShlo));
    return ForeachConvertToTensor(result_buffers, out_dtypes);
  });
}

void AtenForeachReciprocal_(at::TensorList self) {
  TT_KERNEL(OpName::kForeachReciprocal_, _, (self), {
    TT_THROW_IF_ERROR(CheckNotIntegral(self, /* arg_name= */ "self"));
    TT_ASSIGN_OR_THROW(auto out_dtypes, GetOutputDtypes(self));
    TT_ASSIGN_OR_THROW(
        auto result_buffers,
        ForeachUnaryOp(self, std::move(out_dtypes), BuildReciprocalShlo,
                       // Share OpName with AtenForeachReciprocal() as
                       // the underlying Shlo is the same.
                       OpName::kForeachReciprocal));
    TT_THROW_IF_ERROR(ForeachAssignToTensor(result_buffers, self));
  });
}

absl::StatusOr<std::vector<DeviceBufferRef>> ForeachRound(
    at::TensorList self, DtypeSpan out_dtypes) {
  TT_RETURN_IF_ERROR(CheckNotBool(self, /* arg_name= */ "self"));
  // BuildRoundShlo has a different signature from the other unary transforms.
  auto tensor_transform = [](mlir::MlirOp input, mlir::ElementType) {
    return BuildRoundShlo(input, 0);
  };
  return ForeachUnaryOp(self, out_dtypes, std::move(tensor_transform),
                        OpName::kForeachRoundCommon);
}

std::vector<at::Tensor> AtenForeachRound(at::TensorList self) {
  TT_KERNEL(OpName::kForeachRound, _, (self), {
    TT_THROW_IF_ERROR(CheckNotComplex(self, /*arg_name=*/"self"));
    TT_ASSIGN_OR_THROW(auto out_dtypes, GetOutputDtypes(self));
    TT_ASSIGN_OR_THROW(auto result_buffers, ForeachRound(self, out_dtypes));
    return ForeachConvertToTensor(result_buffers, out_dtypes);
  });
}

void AtenForeachRound_(at::TensorList self) {
  TT_KERNEL(OpName::kForeachRound_, _, (self), {
    TT_THROW_IF_ERROR(CheckNotComplex(self, /*arg_name=*/"self"));
    TT_ASSIGN_OR_THROW(auto out_dtypes, GetOutputDtypes(self));
    TT_ASSIGN_OR_THROW(auto result_buffers,
                       ForeachRound(self, std::move(out_dtypes)));
    TT_THROW_IF_ERROR(ForeachAssignToTensor(result_buffers, self));
  });
}

std::vector<at::Tensor> AtenForeachRsqrt(at::TensorList self) {
  TT_KERNEL(OpName::kForeachRsqrt, _, (self), {
    TT_ASSIGN_OR_THROW(auto out_dtypes,
                       GetOutputDtypes(self, /*cast_integral_to_float=*/true));
    TT_ASSIGN_OR_THROW(auto result_buffers,
                       ForeachUnaryOp(self, out_dtypes, BuildRsqrtShlo));
    return ForeachConvertToTensor(result_buffers, out_dtypes);
  });
}

void AtenForeachRsqrt_(at::TensorList self) {
  TT_KERNEL(OpName::kForeachRsqrt_, _, (self), {
    TT_THROW_IF_ERROR(CheckNotIntegral(self, /* arg_name= */ "self"));
    TT_ASSIGN_OR_THROW(auto out_dtypes, GetOutputDtypes(self));
    TT_ASSIGN_OR_THROW(
        auto result_buffers,
        ForeachUnaryOp(self, std::move(out_dtypes), BuildRsqrtShlo,
                       // Share OpName with AtenForeachRsqrt() as
                       // the underlying Shlo is the same.
                       OpName::kForeachRsqrt));
    TT_THROW_IF_ERROR(ForeachAssignToTensor(result_buffers, self));
  });
}

std::vector<at::Tensor> AtenForeachSigmoid(at::TensorList self) {
  TT_KERNEL(OpName::kForeachSigmoid, _, (self), {
    TT_ASSIGN_OR_THROW(auto out_dtypes,
                       GetOutputDtypes(self, /*cast_integral_to_float=*/true));
    TT_ASSIGN_OR_THROW(auto result_buffers,
                       ForeachUnaryOp(self, out_dtypes, BuildSigmoidShlo));
    return ForeachConvertToTensor(result_buffers, out_dtypes);
  });
}

void AtenForeachSigmoid_(at::TensorList self) {
  TT_KERNEL(OpName::kForeachSigmoid_, _, (self), {
    TT_THROW_IF_ERROR(CheckNotIntegral(self, /* arg_name= */ "self"));
    TT_ASSIGN_OR_THROW(auto out_dtypes, GetOutputDtypes(self));
    TT_ASSIGN_OR_THROW(
        auto result_buffers,
        ForeachUnaryOp(self, std::move(out_dtypes), BuildSigmoidShlo,
                       // Share OpName with AtenForeachSigmoid()
                       // as the underlying Shlo is the same.
                       OpName::kForeachSigmoid));
    TT_THROW_IF_ERROR(ForeachAssignToTensor(result_buffers, self));
  });
}

std::vector<at::Tensor> AtenForeachSign(at::TensorList self) {
  TT_KERNEL(OpName::kForeachSign, _, (self), {
    TT_THROW_IF_ERROR(CheckNotComplex(self, /* arg_name= */ "self"));
    TT_ASSIGN_OR_THROW(auto out_dtypes, GetOutputDtypes(self));
    TT_ASSIGN_OR_THROW(auto result_buffers,
                       ForeachUnaryOp(self, out_dtypes, BuildSignShlo));
    return ForeachConvertToTensor(result_buffers, out_dtypes);
  });
}

void AtenForeachSign_(at::TensorList self) {
  TT_KERNEL(OpName::kForeachSign_, _, (self), {
    TT_THROW_IF_ERROR(CheckNotComplex(self, /* arg_name= */ "self"));
    TT_ASSIGN_OR_THROW(auto out_dtypes, GetOutputDtypes(self));
    TT_ASSIGN_OR_THROW(
        auto result_buffers,
        ForeachUnaryOp(self, std::move(out_dtypes), BuildSignShlo,
                       // Share OpName with AtenForeachSign() as
                       // the underlying Shlo is the same.
                       OpName::kForeachSign));
    TT_THROW_IF_ERROR(ForeachAssignToTensor(result_buffers, self));
  });
}

std::vector<at::Tensor> AtenForeachSin(at::TensorList self) {
  TT_KERNEL(OpName::kForeachSin, _, (self), {
    TT_ASSIGN_OR_THROW(auto out_dtypes,
                       GetOutputDtypes(self, /*cast_integral_to_float=*/true));
    TT_ASSIGN_OR_THROW(auto result_buffers,
                       ForeachUnaryOp(self, out_dtypes, BuildSinShlo));
    return ForeachConvertToTensor(result_buffers, out_dtypes);
  });
}

void AtenForeachSin_(at::TensorList self) {
  TT_KERNEL(OpName::kForeachSin_, _, (self), {
    TT_THROW_IF_ERROR(CheckNotIntegral(self, /* arg_name= */ "self"));
    TT_ASSIGN_OR_THROW(auto out_dtypes, GetOutputDtypes(self));
    TT_ASSIGN_OR_THROW(auto result_buffers,
                       ForeachUnaryOp(self, std::move(out_dtypes), BuildSinShlo,
                                      // Share OpName with AtenForeachSin() as
                                      // the underlying Shlo is the same.
                                      OpName::kForeachSin));
    TT_THROW_IF_ERROR(ForeachAssignToTensor(result_buffers, self));
  });
}

std::vector<at::Tensor> AtenForeachSinh(at::TensorList self) {
  TT_KERNEL(OpName::kForeachSinh, _, (self), {
    TT_ASSIGN_OR_THROW(auto out_dtypes,
                       GetOutputDtypes(self, /*cast_integral_to_float=*/true));
    TT_ASSIGN_OR_THROW(auto result_buffers,
                       ForeachUnaryOp(self, out_dtypes, BuildSinhShlo));
    return ForeachConvertToTensor(result_buffers, out_dtypes);
  });
}

void AtenForeachSinh_(at::TensorList self) {
  TT_KERNEL(OpName::kForeachSinh_, _, (self), {
    TT_THROW_IF_ERROR(CheckNotIntegral(self, /* arg_name= */ "self"));
    TT_ASSIGN_OR_THROW(auto out_dtypes, GetOutputDtypes(self));
    TT_ASSIGN_OR_THROW(
        auto result_buffers,
        ForeachUnaryOp(self, std::move(out_dtypes), BuildSinhShlo,
                       // Share OpName with AtenForeachSinh() as
                       // the underlying Shlo is the same.
                       OpName::kForeachSinh));
    TT_THROW_IF_ERROR(ForeachAssignToTensor(result_buffers, self));
  });
}

std::vector<at::Tensor> AtenForeachSqrt(at::TensorList self) {
  TT_KERNEL(OpName::kForeachSqrt, _, (self), {
    TT_ASSIGN_OR_THROW(auto out_dtypes,
                       GetOutputDtypes(self, /*cast_integral_to_float=*/true));
    TT_ASSIGN_OR_THROW(auto result_buffers,
                       ForeachUnaryOp(self, out_dtypes, BuildSqrtShlo));
    return ForeachConvertToTensor(result_buffers, out_dtypes);
  });
}

void AtenForeachSqrt_(at::TensorList self) {
  TT_KERNEL(OpName::kForeachSqrt_, _, (self), {
    TT_THROW_IF_ERROR(CheckNotIntegral(self, /* arg_name= */ "self"));
    TT_ASSIGN_OR_THROW(auto out_dtypes, GetOutputDtypes(self));
    TT_ASSIGN_OR_THROW(
        auto result_buffers,
        ForeachUnaryOp(self, std::move(out_dtypes), BuildSqrtShlo,
                       // Share OpName with AtenForeachSqrt() as
                       // the underlying Shlo is the same.
                       OpName::kForeachSqrt));
    TT_THROW_IF_ERROR(ForeachAssignToTensor(result_buffers, self));
  });
}

std::vector<at::Tensor> AtenForeachTan(at::TensorList self) {
  TT_KERNEL(OpName::kForeachTan, _, (self), {
    TT_ASSIGN_OR_THROW(auto out_dtypes,
                       GetOutputDtypes(self, /*cast_integral_to_float=*/true));
    TT_ASSIGN_OR_THROW(auto result_buffers,
                       ForeachUnaryOp(self, out_dtypes, BuildTanShlo));
    return ForeachConvertToTensor(result_buffers, out_dtypes);
  });
}

void AtenForeachTan_(at::TensorList self) {
  TT_KERNEL(OpName::kForeachTan_, _, (self), {
    TT_THROW_IF_ERROR(CheckNotIntegral(self, /* arg_name= */ "self"));
    TT_ASSIGN_OR_THROW(auto out_dtypes, GetOutputDtypes(self));
    TT_ASSIGN_OR_THROW(auto result_buffers,
                       ForeachUnaryOp(self, std::move(out_dtypes), BuildTanShlo,
                                      // Share OpName with AtenForeachTan() as
                                      // the underlying Shlo is the same.
                                      OpName::kForeachTan));
    TT_THROW_IF_ERROR(ForeachAssignToTensor(result_buffers, self));
  });
}

std::vector<at::Tensor> AtenForeachTanh(at::TensorList self) {
  TT_KERNEL(OpName::kForeachTanh, _, (self), {
    TT_ASSIGN_OR_THROW(auto out_dtypes,
                       GetOutputDtypes(self, /*cast_integral_to_float=*/true));
    TT_ASSIGN_OR_THROW(auto result_buffers,
                       ForeachUnaryOp(self, out_dtypes, BuildTanhShlo));
    return ForeachConvertToTensor(result_buffers, out_dtypes);
  });
}

void AtenForeachTanh_(at::TensorList self) {
  TT_KERNEL(OpName::kForeachTanh_, _, (self), {
    TT_THROW_IF_ERROR(CheckNotIntegral(self, /* arg_name= */ "self"));
    TT_ASSIGN_OR_THROW(auto out_dtypes, GetOutputDtypes(self));
    TT_ASSIGN_OR_THROW(
        auto result_buffers,
        ForeachUnaryOp(self, std::move(out_dtypes), BuildTanhShlo,
                       // Share OpName with AtenForeachTanh() as
                       // the underlying Shlo is the same.
                       OpName::kForeachTanh));
    TT_THROW_IF_ERROR(ForeachAssignToTensor(result_buffers, self));
  });
}

std::vector<at::Tensor> AtenForeachTrunc(at::TensorList self) {
  TT_KERNEL(OpName::kForeachTrunc, _, (self), {
    TT_THROW_IF_ERROR(CheckNotComplex(self, /*arg_name=*/"self"));
    TT_THROW_IF_ERROR(CheckNotBool(self, /* arg_name= */ "self"));
    TT_ASSIGN_OR_THROW(auto out_dtypes, GetOutputDtypes(self));
    TT_ASSIGN_OR_THROW(auto result_buffers,
                       ForeachUnaryOp(self, out_dtypes, BuildTruncShlo));
    return ForeachConvertToTensor(result_buffers, out_dtypes);
  });
}

void AtenForeachTrunc_(at::TensorList self) {
  TT_KERNEL(OpName::kForeachTrunc_, _, (self), {
    TT_THROW_IF_ERROR(CheckNotComplex(self, /*arg_name=*/"self"));
    TT_THROW_IF_ERROR(CheckNotBool(self, /* arg_name= */ "self"));
    TT_ASSIGN_OR_THROW(auto out_dtypes, GetOutputDtypes(self));
    TT_ASSIGN_OR_THROW(
        auto result_buffers,
        ForeachUnaryOp(self, std::move(out_dtypes), BuildTruncShlo,
                       // Share OpName with AtenForeachTrunc() as
                       // the underlying Shlo is the same.
                       OpName::kForeachTrunc));
    TT_THROW_IF_ERROR(ForeachAssignToTensor(result_buffers, self));
  });
}

void AtenForeachZero_(at::TensorList self) {
  TT_KERNEL(OpName::kForeachZero_, _, (self), {
    TT_ASSIGN_OR_THROW(auto out_dtypes, GetOutputDtypes(self));
    // BuildZeroShlo is not implemented.
    auto tensor_transform = [](mlir::MlirOp input, mlir::ElementType) {
      return MakeConstantLike(input, 0.0);
    };
    TT_ASSIGN_OR_THROW(auto result_buffers,
                       ForeachUnaryOp(self, std::move(out_dtypes),
                                      std::move(tensor_transform)));
    TT_THROW_IF_ERROR(ForeachAssignToTensor(result_buffers, self));
  });
}

std::vector<at::Tensor> AtenForeachAddList(at::TensorList self,
                                           at::TensorList other,
                                           const at::Scalar& alpha) {
  auto promoted_alpha = PromoteScalar(alpha).AvoidPromoting(
      ScalarValue::kOne, ScalarValue::kMinusOne);
  TT_KERNEL(
      OpName::kForeachAddList, param_keys, (self, other, promoted_alpha), {
        TT_ASSIGN_OR_THROW(auto out_dtypes, GetOutputDtypes(self, other));

        // Check for invalid input types.
        size_t num_tensors = self.size();
        for (size_t i = 0; i < num_tensors; ++i) {
          TT_CHECK_THROW(!(c10::isIntegralType(self[i].scalar_type(), true) &&
                           c10::isIntegralType(other[i].scalar_type(), true) &&
                           !c10::isIntegralType(alpha.type(), true)),
                         error::kInvalidArgument)
              << "expected alpha to be integral for integral input tensors, "
                 "got "
              << ToString(alpha.type());
          TT_CHECK_THROW(!alpha.isBoolean() ||
                             (self[i].scalar_type() == at::ScalarType::Bool &&
                              other[i].scalar_type() == at::ScalarType::Bool),
                         error::kInvalidArgument)
              << "expected input tensor dtypes to be bool when alpha dtype is "
                 "bool, got "
              << ToString(self[i].scalar_type()) << " and "
              << ToString(other[i].scalar_type());
        }

        return ForeachConvertToTensor(
            ForeachAddList(self, other, std::move(promoted_alpha), out_dtypes,
                           std::move(param_keys)),
            out_dtypes);
      });
}

std::vector<at::Tensor> AtenForeachAddScalar(at::TensorList self,
                                             const at::Scalar& scalar) {
  auto promoted_scalar = PromoteScalar(scalar);
  TT_KERNEL(OpName::kForeachAddScalar, _, (self, promoted_scalar), {
    TT_ASSIGN_OR_THROW(auto out_dtypes, GetOutputDtypes(self, scalar));
    std::vector<at::Tensor> other;
    other.reserve(self.size());
    for (size_t i = 0; i < self.size(); ++i) {
      at::ScalarType scalar_type = ConvertTo<at::ScalarType>(out_dtypes[i]);
      TT_ASSIGN_OR_THROW(at::Tensor scalar_tensor,
                         promoted_scalar.GetTensor(scalar_type));
      other.push_back(scalar_tensor);
    }
    return ForeachConvertToTensor(
        ForeachAddList(
            self, other,
            PromoteScalar(at::Scalar(1.0)).AvoidPromoting(ScalarValue::kOne),
            out_dtypes),
        out_dtypes);
  });
}

std::vector<at::Tensor> AtenForeachAddScalarList(
    at::TensorList self, at::ArrayRef<at::Scalar> scalars) {
  auto promoted_scalars = PromoteScalar(scalars);
  TT_KERNEL(OpName::kForeachAddScalarList, _, (self, promoted_scalars), {
    TT_ASSIGN_OR_THROW(auto out_dtypes, GetOutputDtypes(self, scalars));
    std::vector<at::Tensor> other;
    other.reserve(self.size());
    for (size_t i = 0; i < self.size(); ++i) {
      at::ScalarType scalar_type = ConvertTo<at::ScalarType>(out_dtypes[i]);
      TT_ASSIGN_OR_THROW(at::Tensor scalar_tensor,
                         promoted_scalars[i].GetTensor(scalar_type));
      other.push_back(scalar_tensor);
    }
    return ForeachConvertToTensor(
        ForeachAddList(
            self, other,
            PromoteScalar(at::Scalar(1.0)).AvoidPromoting(ScalarValue::kOne),
            out_dtypes),
        out_dtypes);
  });
}

std::vector<at::Tensor> AtenForeachAddTensor(at::TensorList self,
                                             const at::Tensor& other,
                                             const at::Scalar& alpha) {
  auto promoted_alpha = PromoteScalar(alpha).AvoidPromoting(
      ScalarValue::kOne, ScalarValue::kMinusOne);
  TT_KERNEL(
      OpName::kForeachAddTensor, param_keys, (self, other, promoted_alpha), {
        std::vector<at::Tensor> other_list(self.size(), other);
        TT_ASSIGN_OR_THROW(auto out_dtypes, GetOutputDtypes(self, other_list));
        return ForeachConvertToTensor(
            ForeachAddList(self, other_list, std::move(promoted_alpha),
                           out_dtypes, std::move(param_keys)),
            out_dtypes);
      });
}

void AtenForeachAdd_List(at::TensorList self, at::TensorList other,
                         const at::Scalar& alpha) {
  auto promoted_alpha = PromoteScalar(alpha).AvoidPromoting(
      ScalarValue::kOne, ScalarValue::kMinusOne);
  TT_KERNEL(OpName::kForeachAdd_List, param_keys, (self, other, promoted_alpha),
            {
              TT_ASSIGN_OR_THROW(auto out_dtypes, GetOutputDtypes(self));
              TT_THROW_IF_ERROR(ForeachAssignToTensor(
                  ForeachAddList(self, other, std::move(promoted_alpha),
                                 out_dtypes, std::move(param_keys)),
                  self, out_dtypes));
            });
}

void AtenForeachAdd_Scalar(at::TensorList self, const at::Scalar& scalar) {
  auto promoted_scalar = PromoteScalar(scalar);
  TT_KERNEL(OpName::kForeachAdd_Scalar, _, (self, promoted_scalar), {
    TT_THROW_IF_ERROR(CheckInplaceScalarType(self, scalar));
    TT_ASSIGN_OR_THROW(auto out_dtypes, GetOutputDtypes(self));
    std::vector<at::Tensor> other;
    other.reserve(self.size());
    for (size_t i = 0; i < self.size(); ++i) {
      TT_ASSIGN_OR_THROW(at::Tensor scalar_tensor,
                         promoted_scalar.GetTensor(self[i].scalar_type()));
      other.push_back(scalar_tensor);
    }
    TT_THROW_IF_ERROR(ForeachAssignToTensor(
        ForeachAddList(
            self, other,
            PromoteScalar(at::Scalar(1.0)).AvoidPromoting(ScalarValue::kOne),
            out_dtypes),
        self, out_dtypes));
  });
}

void AtenForeachAdd_ScalarList(at::TensorList self,
                               at::ArrayRef<at::Scalar> scalars) {
  auto promoted_scalars = PromoteScalar(scalars);
  TT_KERNEL(OpName::kForeachAdd_ScalarList, _, (self, promoted_scalars), {
    TT_THROW_IF_ERROR(CheckInplaceScalarType(self, scalars));
    TT_ASSIGN_OR_THROW(auto out_dtypes, GetOutputDtypes(self));
    std::vector<at::Tensor> other;
    other.reserve(self.size());
    for (size_t i = 0; i < self.size(); ++i) {
      TT_ASSIGN_OR_THROW(at::Tensor scalar_tensor,
                         promoted_scalars[i].GetTensor(self[i].scalar_type()));
      other.push_back(scalar_tensor);
    }
    TT_THROW_IF_ERROR(ForeachAssignToTensor(
        ForeachAddList(
            self, other,
            PromoteScalar(at::Scalar(1.0)).AvoidPromoting(ScalarValue::kOne),
            out_dtypes),
        self, out_dtypes));
  });
}

void AtenForeachAdd_Tensor(at::TensorList self, const at::Tensor& other,
                           const at::Scalar& alpha) {
  auto promoted_alpha = PromoteScalar(alpha).AvoidPromoting(
      ScalarValue::kOne, ScalarValue::kMinusOne);
  TT_KERNEL(OpName::kForeachAdd_Tensor, param_keys,
            (self, other, promoted_alpha), {
              std::vector<at::Tensor> other_list(self.size(), other);
              TT_ASSIGN_OR_THROW(auto out_dtypes, GetOutputDtypes(self));
              TT_THROW_IF_ERROR(ForeachAssignToTensor(
                  ForeachAddList(self, other_list, std::move(promoted_alpha),
                                 out_dtypes, std::move(param_keys)),
                  self, out_dtypes));
            });
}

std::vector<at::Tensor> AtenForeachAddcdivScalar(at::TensorList self,
                                                 at::TensorList tensor1,
                                                 at::TensorList tensor2,
                                                 const at::Scalar& value) {
  auto promoted_value = PromoteScalar(value);
  TT_KERNEL(
      OpName::kForeachAddcdivScalar, _,
      (self, tensor1, tensor2, promoted_value), {
        // _foreach_div supports two integral tensors, but _foreach_addcdiv
        // doesn't.
        TT_THROW_IF_ERROR(
            CheckPairwiseAddcdivAtLeastOneNotIntegral(tensor1, tensor2));
        TT_ASSIGN_OR_THROW(auto out_dtypes, GetOutputDtypes(self));
        std::vector<at::Tensor> value_list;
        value_list.reserve(self.size());
        for (size_t i = 0; i < self.size(); ++i) {
          at::ScalarType scalar_type = ConvertTo<at::ScalarType>(out_dtypes[i]);
          TT_ASSIGN_OR_THROW(at::Tensor value_tensor,
                             promoted_value.GetTensor(scalar_type));
          value_list.push_back(value_tensor);
        }
        return ForeachConvertToTensor(
            ForeachAddcdiv(self, tensor1, tensor2, value_list, out_dtypes),
            out_dtypes);
      });
}

std::vector<at::Tensor> AtenForeachAddcdivScalarList(
    at::TensorList self, at::TensorList tensor1, at::TensorList tensor2,
    at::ArrayRef<at::Scalar> scalars) {
  auto promoted_scalars = PromoteScalar(scalars);
  TT_KERNEL(
      OpName::kForeachAddcdivScalarList, _,
      (self, tensor1, tensor2, promoted_scalars), {
        // _foreach_div supports two integral tensors, but
        // _foreach_addcdiv doesn't.
        TT_THROW_IF_ERROR(
            CheckPairwiseAddcdivAtLeastOneNotIntegral(tensor1, tensor2));
        TT_ASSIGN_OR_THROW(auto out_dtypes, GetOutputDtypes(self));
        std::vector<at::Tensor> value_list;
        value_list.reserve(self.size());
        for (size_t i = 0; i < self.size(); ++i) {
          at::ScalarType scalar_type = ConvertTo<at::ScalarType>(out_dtypes[i]);
          TT_ASSIGN_OR_THROW(at::Tensor value_tensor,
                             promoted_scalars[i].GetTensor(scalar_type));
          value_list.push_back(value_tensor);
        }
        return ForeachConvertToTensor(
            ForeachAddcdiv(self, tensor1, tensor2, value_list, out_dtypes),
            out_dtypes);
      });
}

std::vector<at::Tensor> AtenForeachAddcdivTensor(at::TensorList self,
                                                 at::TensorList tensor1,
                                                 at::TensorList tensor2,
                                                 const at::Tensor& scalars) {
  TT_KERNEL(
      OpName::kForeachAddcdivTensor, _, (self, tensor1, tensor2, scalars), {
        // _foreach_div supports two integral tensors, but _foreach_addcdiv
        // doesn't.
        TT_THROW_IF_ERROR(
            CheckPairwiseAddcdivAtLeastOneNotIntegral(tensor1, tensor2));
        TT_ASSIGN_OR_THROW(auto out_dtypes, GetOutputDtypes(self));
        std::vector<at::Tensor> value_list(self.size(), scalars);
        return ForeachConvertToTensor(
            ForeachAddcdiv(self, tensor1, tensor2, value_list, out_dtypes),
            out_dtypes);
      });
}

void AtenForeachAddcdiv_Scalar(at::TensorList self, at::TensorList tensor1,
                               at::TensorList tensor2,
                               const at::Scalar& value) {
  auto promoted_value = PromoteScalar(value);
  TT_KERNEL(
      OpName::kForeachAddcdiv_Scalar, _,
      (self, tensor1, tensor2, promoted_value), {
        // _foreach_div supports two integral tensors, but _foreach_addcdiv
        // doesn't.
        TT_THROW_IF_ERROR(
            CheckPairwiseAddcdivAtLeastOneNotIntegral(tensor1, tensor2));
        TT_ASSIGN_OR_THROW(auto out_dtypes, GetOutputDtypes(self));
        std::vector<at::Tensor> value_list;
        value_list.reserve(self.size());
        for (size_t i = 0; i < self.size(); ++i) {
          TT_ASSIGN_OR_THROW(at::Tensor value_tensor,
                             promoted_value.GetTensor(self[i].scalar_type()));
          value_list.push_back(value_tensor);
        }
        TT_THROW_IF_ERROR(ForeachAssignToTensor(
            ForeachAddcdiv(self, tensor1, tensor2, value_list, out_dtypes),
            self, out_dtypes));
      });
}

void AtenForeachAddcdiv_ScalarList(at::TensorList self, at::TensorList tensor1,
                                   at::TensorList tensor2,
                                   at::ArrayRef<at::Scalar> scalars) {
  auto promoted_scalars = PromoteScalar(scalars);
  TT_KERNEL(
      OpName::kForeachAddcdiv_ScalarList, _,
      (self, tensor1, tensor2, promoted_scalars), {
        // _foreach_div supports two integral tensors, but
        // _foreach_addcdiv doesn't.
        TT_THROW_IF_ERROR(
            CheckPairwiseAddcdivAtLeastOneNotIntegral(tensor1, tensor2));
        TT_ASSIGN_OR_THROW(auto out_dtypes, GetOutputDtypes(self));
        std::vector<at::Tensor> value_list;
        value_list.reserve(self.size());
        for (size_t i = 0; i < self.size(); ++i) {
          TT_ASSIGN_OR_THROW(
              at::Tensor value_tensor,
              promoted_scalars[i].GetTensor(self[i].scalar_type()));
          value_list.push_back(value_tensor);
        }
        TT_THROW_IF_ERROR(ForeachAssignToTensor(
            ForeachAddcdiv(self, tensor1, tensor2, value_list, out_dtypes),
            self, out_dtypes));
      });
}

void AtenForeachAddcdiv_Tensor(at::TensorList self, at::TensorList tensor1,
                               at::TensorList tensor2,
                               const at::Tensor& scalars) {
  TT_KERNEL(
      OpName::kForeachAddcdiv_Tensor, _, (self, tensor1, tensor2, scalars), {
        // _foreach_div supports two integral tensors, but _foreach_addcdiv
        // doesn't.
        TT_THROW_IF_ERROR(
            CheckPairwiseAddcdivAtLeastOneNotIntegral(tensor1, tensor2));
        TT_ASSIGN_OR_THROW(auto out_dtypes, GetOutputDtypes(self));
        std::vector<at::Tensor> value_list(self.size(), scalars);
        TT_THROW_IF_ERROR(ForeachAssignToTensor(
            ForeachAddcdiv(self, tensor1, tensor2, value_list, out_dtypes),
            self, out_dtypes));
      });
}

std::vector<at::Tensor> AtenForeachAddcmulScalar(at::TensorList self,
                                                 at::TensorList tensor1,
                                                 at::TensorList tensor2,
                                                 const at::Scalar& value) {
  auto promoted_value = PromoteScalar(value);
  TT_KERNEL(
      OpName::kForeachAddcmulScalar, _,
      (self, tensor1, tensor2, promoted_value), {
        // _foreach_mul and _foreach_add supports bool tensors, but
        // _foreach_addcmul doesn't.
        TT_THROW_IF_ERROR(CheckNotBool(self, /* arg_name= */ "self"));
        TT_ASSIGN_OR_THROW(auto out_dtypes, GetOutputDtypes(self));
        std::vector<at::Tensor> value_list;
        value_list.reserve(self.size());
        for (size_t i = 0; i < self.size(); ++i) {
          at::ScalarType scalar_type = ConvertTo<at::ScalarType>(out_dtypes[i]);
          TT_ASSIGN_OR_THROW(at::Tensor value_tensor,
                             promoted_value.GetTensor(scalar_type));
          value_list.push_back(value_tensor);
        }
        return ForeachConvertToTensor(
            ForeachAddcmul(self, tensor1, tensor2, value_list, out_dtypes),
            out_dtypes);
      });
}

std::vector<at::Tensor> AtenForeachAddcmulScalarList(
    at::TensorList self, at::TensorList tensor1, at::TensorList tensor2,
    at::ArrayRef<at::Scalar> scalars) {
  auto promoted_scalars = PromoteScalar(scalars);
  TT_KERNEL(
      OpName::kForeachAddcmulScalarList, _,
      (self, tensor1, tensor2, promoted_scalars), {
        // _foreach_mul and _foreach_add supports bool tensors, but
        // _foreach_addcmul doesn't.
        TT_THROW_IF_ERROR(CheckNotBool(self, /* arg_name= */ "self"));
        TT_ASSIGN_OR_THROW(auto out_dtypes, GetOutputDtypes(self));
        std::vector<at::Tensor> value_list;
        value_list.reserve(self.size());
        for (size_t i = 0; i < self.size(); ++i) {
          at::ScalarType scalar_type = ConvertTo<at::ScalarType>(out_dtypes[i]);
          TT_ASSIGN_OR_THROW(at::Tensor value_tensor,
                             promoted_scalars[i].GetTensor(scalar_type));
          value_list.push_back(value_tensor);
        }
        return ForeachConvertToTensor(
            ForeachAddcmul(self, tensor1, tensor2, value_list, out_dtypes),
            out_dtypes);
      });
}

std::vector<at::Tensor> AtenForeachAddcmulTensor(at::TensorList self,
                                                 at::TensorList tensor1,
                                                 at::TensorList tensor2,
                                                 const at::Tensor& scalars) {
  TT_KERNEL(
      OpName::kForeachAddcmulTensor, _, (self, tensor1, tensor2, scalars), {
        // _foreach_mul and _foreach_add supports bool tensors, but
        // _foreach_addcmul doesn't.
        TT_THROW_IF_ERROR(CheckNotBool(self, /* arg_name= */ "self"));
        TT_ASSIGN_OR_THROW(auto out_dtypes, GetOutputDtypes(self));
        std::vector<at::Tensor> value_list(self.size(), scalars);
        return ForeachConvertToTensor(
            ForeachAddcmul(self, tensor1, tensor2, value_list, out_dtypes),
            out_dtypes);
      });
}

void AtenForeachAddcmul_Scalar(at::TensorList self, at::TensorList tensor1,
                               at::TensorList tensor2,
                               const at::Scalar& value) {
  auto promoted_value = PromoteScalar(value);
  TT_KERNEL(
      OpName::kForeachAddcmul_Scalar, _,
      (self, tensor1, tensor2, promoted_value), {
        // _foreach_mul and _foreach_add supports bool tensors, but
        // _foreach_addcmul doesn't.
        TT_THROW_IF_ERROR(CheckNotBool(self, /* arg_name= */ "self"));
        TT_ASSIGN_OR_THROW(auto out_dtypes, GetOutputDtypes(self));
        std::vector<at::Tensor> value_list;
        value_list.reserve(self.size());
        for (size_t i = 0; i < self.size(); ++i) {
          TT_ASSIGN_OR_THROW(at::Tensor value_tensor,
                             promoted_value.GetTensor(self[i].scalar_type()));
          value_list.push_back(value_tensor);
        }
        TT_THROW_IF_ERROR(ForeachAssignToTensor(
            ForeachAddcmul(self, tensor1, tensor2, value_list, out_dtypes),
            self, out_dtypes));
      });
}

void AtenForeachAddcmul_ScalarList(at::TensorList self, at::TensorList tensor1,
                                   at::TensorList tensor2,
                                   at::ArrayRef<at::Scalar> scalars) {
  auto promoted_scalars = PromoteScalar(scalars);
  TT_KERNEL(
      OpName::kForeachAddcmul_ScalarList, _,
      (self, tensor1, tensor2, promoted_scalars), {
        // _foreach_mul and _foreach_add supports bool tensors, but
        // _foreach_addcmul doesn't.
        TT_THROW_IF_ERROR(CheckNotBool(self, /* arg_name= */ "self"));
        TT_ASSIGN_OR_THROW(auto out_dtypes, GetOutputDtypes(self));
        std::vector<at::Tensor> value_list;
        value_list.reserve(self.size());
        for (size_t i = 0; i < self.size(); ++i) {
          TT_ASSIGN_OR_THROW(
              at::Tensor value_tensor,
              promoted_scalars[i].GetTensor(self[i].scalar_type()));
          value_list.push_back(value_tensor);
        }
        TT_THROW_IF_ERROR(ForeachAssignToTensor(
            ForeachAddcmul(self, tensor1, tensor2, value_list, out_dtypes),
            self, out_dtypes));
      });
}

void AtenForeachAddcmul_Tensor(at::TensorList self, at::TensorList tensor1,
                               at::TensorList tensor2,
                               const at::Tensor& scalars) {
  TT_KERNEL(
      OpName::kForeachAddcmul_Tensor, _, (self, tensor1, tensor2, scalars), {
        // _foreach_mul and _foreach_add supports bool tensors, but
        // _foreach_addcmul doesn't.
        TT_THROW_IF_ERROR(CheckNotBool(self, /* arg_name= */ "self"));
        TT_ASSIGN_OR_THROW(auto out_dtypes, GetOutputDtypes(self));
        std::vector<at::Tensor> value_list(self.size(), scalars);
        TT_THROW_IF_ERROR(ForeachAssignToTensor(
            ForeachAddcmul(self, tensor1, tensor2, value_list, out_dtypes),
            self, out_dtypes));
      });
}

std::vector<at::Tensor> AtenForeachDivList(at::TensorList self,
                                           at::TensorList other) {
  TT_KERNEL(OpName::kForeachDivList, _, (self, other), {
    TT_ASSIGN_OR_THROW(auto out_dtypes,
                       GetOutputDtypes(self, other, /*is_div=*/true));
    return ForeachConvertToTensor(ForeachDiv(self, other, out_dtypes),
                                  out_dtypes);
  });
}

std::vector<at::Tensor> AtenForeachDivScalar(at::TensorList self,
                                             const at::Scalar& scalar) {
  auto promoted_scalar = PromoteScalar(scalar);
  TT_KERNEL(OpName::kForeachDivScalar, _, (self, promoted_scalar), {
    TT_ASSIGN_OR_THROW(auto out_dtypes,
                       GetOutputDtypes(self, scalar, /*is_div=*/true));
    std::vector<at::Tensor> other;
    other.reserve(self.size());
    for (size_t i = 0; i < self.size(); ++i) {
      at::ScalarType scalar_type = ConvertTo<at::ScalarType>(out_dtypes[i]);
      TT_ASSIGN_OR_THROW(at::Tensor scalar_tensor,
                         promoted_scalar.GetTensor(scalar_type));
      other.push_back(scalar_tensor);
    }
    return ForeachConvertToTensor(ForeachDiv(self, other, out_dtypes),
                                  out_dtypes);
  });
}

std::vector<at::Tensor> AtenForeachDivScalarList(
    at::TensorList self, at::ArrayRef<at::Scalar> scalars) {
  auto promoted_scalars = PromoteScalar(scalars);
  TT_KERNEL(OpName::kForeachDivScalarList, _, (self, promoted_scalars), {
    TT_ASSIGN_OR_THROW(auto out_dtypes,
                       GetOutputDtypes(self, scalars, /*is_div=*/true));
    std::vector<at::Tensor> other;
    other.reserve(self.size());
    for (size_t i = 0; i < self.size(); ++i) {
      at::ScalarType scalar_type = ConvertTo<at::ScalarType>(out_dtypes[i]);
      TT_ASSIGN_OR_THROW(at::Tensor scalar_tensor,
                         promoted_scalars[i].GetTensor(scalar_type));
      other.push_back(scalar_tensor);
    }
    return ForeachConvertToTensor(ForeachDiv(self, other, out_dtypes),
                                  out_dtypes);
  });
}

std::vector<at::Tensor> AtenForeachDivTensor(at::TensorList self,
                                             const at::Tensor& other) {
  TT_KERNEL(OpName::kForeachDivTensor, _, (self, other), {
    std::vector<at::Tensor> other_list(self.size(), other);
    TT_ASSIGN_OR_THROW(auto out_dtypes,
                       GetOutputDtypes(self, other_list, /*is_div=*/true));
    return ForeachConvertToTensor(ForeachDiv(self, other_list, out_dtypes),
                                  out_dtypes);
  });
}

void AtenForeachDiv_List(at::TensorList self, at::TensorList other) {
  TT_KERNEL(OpName::kForeachDiv_List, _, (self, other), {
    TT_ASSIGN_OR_THROW(auto out_dtypes,
                       GetOutputDtypes(self, other, /*is_div=*/true));
    TT_THROW_IF_ERROR(ForeachAssignToTensor(ForeachDiv(self, other, out_dtypes),
                                            self, out_dtypes));
  });
}

void AtenForeachDiv_Scalar(at::TensorList self, const at::Scalar& scalar) {
  auto promoted_scalar = PromoteScalar(scalar);
  TT_KERNEL(OpName::kForeachDiv_Scalar, _, (self, promoted_scalar), {
    TT_ASSIGN_OR_THROW(auto out_dtypes,
                       GetOutputDtypes(self, scalar, /*is_div=*/true));
    std::vector<at::Tensor> other;
    other.reserve(self.size());
    for (size_t i = 0; i < self.size(); ++i) {
      at::ScalarType scalar_type = ConvertTo<at::ScalarType>(out_dtypes[i]);
      TT_ASSIGN_OR_THROW(at::Tensor scalar_tensor,
                         promoted_scalar.GetTensor(scalar_type));
      other.push_back(scalar_tensor);
    }
    TT_THROW_IF_ERROR(ForeachAssignToTensor(ForeachDiv(self, other, out_dtypes),
                                            self, out_dtypes));
  });
}

void AtenForeachDiv_ScalarList(at::TensorList self,
                               at::ArrayRef<at::Scalar> scalars) {
  auto promoted_scalars = PromoteScalar(scalars);
  TT_KERNEL(OpName::kForeachDiv_ScalarList, _, (self, promoted_scalars), {
    TT_ASSIGN_OR_THROW(auto out_dtypes,
                       GetOutputDtypes(self, scalars, /*is_div=*/true));
    std::vector<at::Tensor> other;
    other.reserve(self.size());
    for (size_t i = 0; i < self.size(); ++i) {
      at::ScalarType scalar_type = ConvertTo<at::ScalarType>(out_dtypes[i]);
      TT_ASSIGN_OR_THROW(at::Tensor scalar_tensor,
                         promoted_scalars[i].GetTensor(scalar_type));
      other.push_back(scalar_tensor);
    }
    TT_THROW_IF_ERROR(ForeachAssignToTensor(ForeachDiv(self, other, out_dtypes),
                                            self, out_dtypes));
  });
}

void AtenForeachDiv_Tensor(at::TensorList self, const at::Tensor& other) {
  TT_KERNEL(OpName::kForeachDiv_Tensor, _, (self, other), {
    std::vector<at::Tensor> other_list(self.size(), other);
    TT_ASSIGN_OR_THROW(auto out_dtypes,
                       GetOutputDtypes(self, other_list, /*is_div=*/true));
    TT_THROW_IF_ERROR(ForeachAssignToTensor(
        ForeachDiv(self, other_list, out_dtypes), self, out_dtypes));
  });
}

std::vector<at::Tensor> AtenForeachLerpList(at::TensorList self,
                                            at::TensorList other,
                                            at::TensorList weight) {
  TT_KERNEL(OpName::kForeachLerpList, _, (self, other, weight), {
    TT_THROW_IF_ERROR(CheckNotIntegral(self, /* arg_name= */ "self"));
    TT_ASSIGN_OR_THROW(auto out_dtypes, GetOutputDtypes(self));
    return ForeachConvertToTensor(ForeachLerp(self, other, weight, out_dtypes),
                                  out_dtypes);
  });
}

std::vector<at::Tensor> AtenForeachLerpScalar(at::TensorList self,
                                              at::TensorList other,
                                              const at::Scalar& weight) {
  auto promoted_weight = PromoteScalar(weight);
  TT_KERNEL(OpName::kForeachLerpScalar, _, (self, other, promoted_weight), {
    TT_THROW_IF_ERROR(CheckNotIntegral(self, /* arg_name= */ "self"));
    TT_ASSIGN_OR_THROW(auto out_dtypes, GetOutputDtypes(self));
    std::vector<at::Tensor> weight_list;
    weight_list.reserve(self.size());
    for (size_t i = 0; i < self.size(); ++i) {
      at::ScalarType scalar_type = ConvertTo<at::ScalarType>(out_dtypes[i]);
      TT_ASSIGN_OR_THROW(at::Tensor weight_tensor,
                         promoted_weight.GetTensor(scalar_type));
      weight_list.push_back(weight_tensor);
    }
    return ForeachConvertToTensor(
        ForeachLerp(self, other, weight_list, out_dtypes), out_dtypes);
  });
}

std::vector<at::Tensor> AtenForeachLerpScalarList(
    at::TensorList self, at::TensorList other,
    at::ArrayRef<at::Scalar> scalars) {
  auto promoted_scalars = PromoteScalar(scalars);
  TT_KERNEL(
      OpName::kForeachLerpScalarList, _, (self, other, promoted_scalars), {
        TT_THROW_IF_ERROR(CheckNotIntegral(self, /* arg_name= */ "self"));
        TT_ASSIGN_OR_THROW(auto out_dtypes, GetOutputDtypes(self));
        std::vector<at::Tensor> weight_list;
        weight_list.reserve(self.size());
        for (size_t i = 0; i < self.size(); ++i) {
          at::ScalarType scalar_type = ConvertTo<at::ScalarType>(out_dtypes[i]);
          TT_ASSIGN_OR_THROW(at::Tensor weight_tensor,
                             promoted_scalars[i].GetTensor(scalar_type));
          weight_list.push_back(weight_tensor);
        }
        return ForeachConvertToTensor(
            ForeachLerp(self, other, weight_list, out_dtypes), out_dtypes);
      });
}

void AtenForeachLerp_List(at::TensorList self, at::TensorList other,
                          at::TensorList weight) {
  TT_KERNEL(OpName::kForeachLerp_List, _, (self, other, weight), {
    TT_THROW_IF_ERROR(CheckNotIntegral(self, /* arg_name= */ "self"));
    TT_ASSIGN_OR_THROW(auto out_dtypes, GetOutputDtypes(self));
    TT_THROW_IF_ERROR(ForeachAssignToTensor(
        ForeachLerp(self, other, weight, out_dtypes), self, out_dtypes));
  });
}

void AtenForeachLerp_Scalar(at::TensorList self, at::TensorList other,
                            const at::Scalar& weight) {
  auto promoted_weight = PromoteScalar(weight);
  TT_KERNEL(OpName::kForeachLerp_Scalar, _, (self, other, promoted_weight), {
    TT_THROW_IF_ERROR(CheckNotIntegral(self, /* arg_name= */ "self"));
    TT_ASSIGN_OR_THROW(auto out_dtypes, GetOutputDtypes(self));
    std::vector<at::Tensor> weight_list;
    weight_list.reserve(self.size());
    for (size_t i = 0; i < self.size(); ++i) {
      TT_ASSIGN_OR_THROW(at::Tensor weight_tensor,
                         promoted_weight.GetTensor(self[i].scalar_type()));
      weight_list.push_back(weight_tensor);
    }
    TT_THROW_IF_ERROR(ForeachAssignToTensor(
        ForeachLerp(self, other, weight_list, out_dtypes), self, out_dtypes));
  });
}

void AtenForeachLerp_ScalarList(at::TensorList self, at::TensorList other,
                                at::ArrayRef<at::Scalar> scalars) {
  auto promoted_scalars = PromoteScalar(scalars);
  TT_KERNEL(OpName::kForeachLerp_ScalarList, _, (self, other, promoted_scalars),
            {
              TT_THROW_IF_ERROR(CheckNotIntegral(self, /* arg_name= */ "self"));
              TT_ASSIGN_OR_THROW(auto out_dtypes, GetOutputDtypes(self));
              std::vector<at::Tensor> weight_list;
              weight_list.reserve(self.size());
              for (size_t i = 0; i < self.size(); ++i) {
                TT_ASSIGN_OR_THROW(
                    at::Tensor weight_tensor,
                    promoted_scalars[i].GetTensor(self[i].scalar_type()));
                weight_list.push_back(weight_tensor);
              }
              TT_THROW_IF_ERROR(ForeachAssignToTensor(
                  ForeachLerp(self, other, weight_list, out_dtypes), self,
                  out_dtypes));
            });
}

std::vector<at::Tensor> AtenForeachMulList(at::TensorList self,
                                           at::TensorList other) {
  TT_KERNEL(OpName::kForeachMulList, _, (self, other), {
    TT_ASSIGN_OR_THROW(auto out_dtypes, GetOutputDtypes(self, other));
    return ForeachConvertToTensor(ForeachMulList(self, other, out_dtypes),
                                  out_dtypes);
  });
}

std::vector<at::Tensor> AtenForeachMulScalar(at::TensorList self,
                                             const at::Scalar& scalar) {
  auto promoted_scalar = PromoteScalar(scalar);
  TT_KERNEL(OpName::kForeachMulScalar, _, (self, promoted_scalar), {
    TT_ASSIGN_OR_THROW(auto out_dtypes, GetOutputDtypes(self, scalar));
    std::vector<at::Tensor> other;
    other.reserve(self.size());
    for (size_t i = 0; i < self.size(); ++i) {
      at::ScalarType scalar_type = ConvertTo<at::ScalarType>(out_dtypes[i]);
      TT_ASSIGN_OR_THROW(at::Tensor scalar_tensor,
                         promoted_scalar.GetTensor(scalar_type));
      other.push_back(scalar_tensor);
    }
    return ForeachConvertToTensor(ForeachMulList(self, other, out_dtypes),
                                  out_dtypes);
  });
}

std::vector<at::Tensor> AtenForeachMulScalarList(
    at::TensorList self, at::ArrayRef<at::Scalar> scalars) {
  auto promoted_scalars = PromoteScalar(scalars);
  TT_KERNEL(OpName::kForeachMulScalarList, _, (self, promoted_scalars), {
    TT_ASSIGN_OR_THROW(auto out_dtypes, GetOutputDtypes(self, scalars));
    std::vector<at::Tensor> other;
    other.reserve(self.size());
    for (size_t i = 0; i < self.size(); ++i) {
      at::ScalarType scalar_type = ConvertTo<at::ScalarType>(out_dtypes[i]);
      TT_ASSIGN_OR_THROW(at::Tensor scalar_tensor,
                         promoted_scalars[i].GetTensor(scalar_type));
      other.push_back(scalar_tensor);
    }
    return ForeachConvertToTensor(ForeachMulList(self, other, out_dtypes),
                                  out_dtypes);
  });
}

std::vector<at::Tensor> AtenForeachMulTensor(at::TensorList self,
                                             const at::Tensor& other) {
  TT_KERNEL(OpName::kForeachMulTensor, _, (self, other), {
    std::vector<at::Tensor> other_list(self.size(), other);
    TT_ASSIGN_OR_THROW(auto out_dtypes, GetOutputDtypes(self, other_list));
    return ForeachConvertToTensor(ForeachMulList(self, other_list, out_dtypes),
                                  out_dtypes);
  });
}

void AtenForeachMul_List(at::TensorList self, at::TensorList other) {
  TT_KERNEL(OpName::kForeachMul_List, _, (self, other), {
    TT_ASSIGN_OR_THROW(auto out_dtypes, GetOutputDtypes(self));
    TT_THROW_IF_ERROR(ForeachAssignToTensor(
        ForeachMulList(self, other, out_dtypes), self, out_dtypes));
  });
}

void AtenForeachMul_Scalar(at::TensorList self, const at::Scalar& scalar) {
  auto promoted_scalar = PromoteScalar(scalar);
  TT_KERNEL(OpName::kForeachMul_Scalar, _, (self, promoted_scalar), {
    TT_THROW_IF_ERROR(CheckInplaceScalarType(self, scalar));
    TT_ASSIGN_OR_THROW(auto out_dtypes, GetOutputDtypes(self));
    std::vector<at::Tensor> other;
    other.reserve(self.size());
    for (size_t i = 0; i < self.size(); ++i) {
      TT_ASSIGN_OR_THROW(at::Tensor scalar_tensor,
                         promoted_scalar.GetTensor(self[i].scalar_type()));
      other.push_back(scalar_tensor);
    }
    TT_THROW_IF_ERROR(ForeachAssignToTensor(
        ForeachMulList(self, other, out_dtypes), self, out_dtypes));
  });
}

void AtenForeachMul_ScalarList(at::TensorList self,
                               at::ArrayRef<at::Scalar> scalars) {
  auto promoted_scalars = PromoteScalar(scalars);
  TT_KERNEL(OpName::kForeachMul_ScalarList, _, (self, promoted_scalars), {
    TT_THROW_IF_ERROR(CheckInplaceScalarType(self, scalars));
    TT_ASSIGN_OR_THROW(auto out_dtypes, GetOutputDtypes(self));
    std::vector<at::Tensor> other;
    other.reserve(self.size());
    for (size_t i = 0; i < self.size(); ++i) {
      TT_ASSIGN_OR_THROW(at::Tensor scalar_tensor,
                         promoted_scalars[i].GetTensor(self[i].scalar_type()));
      other.push_back(scalar_tensor);
    }
    TT_THROW_IF_ERROR(ForeachAssignToTensor(
        ForeachMulList(self, other, out_dtypes), self, out_dtypes));
  });
}

void AtenForeachMul_Tensor(at::TensorList self, const at::Tensor& other) {
  TT_KERNEL(OpName::kForeachMul_Tensor, _, (self, other), {
    std::vector<at::Tensor> other_list(self.size(), other);
    TT_ASSIGN_OR_THROW(auto out_dtypes, GetOutputDtypes(self));
    TT_THROW_IF_ERROR(ForeachAssignToTensor(
        ForeachMulList(self, other_list, out_dtypes), self, out_dtypes));
  });
}

std::vector<at::Tensor> AtenForeachSubList(at::TensorList self,
                                           at::TensorList other,
                                           const at::Scalar& alpha) {
  if (self.empty()) {
    return {};
  }
  auto promoted_neg_alpha = PromoteScalar(-alpha).AvoidPromoting(
      ScalarValue::kOne, ScalarValue::kMinusOne);
  TT_KERNEL(
      OpName::kForeachSubList, param_keys, (self, other, promoted_neg_alpha), {
        // _foreach_add supports bool, but _foreach_sub does not.
        TT_THROW_IF_ERROR(CheckNotBool(alpha, /* arg_name= */ "alpha"));
        TT_THROW_IF_ERROR(CheckNotBool(self, /* arg_name= */ "self"));
        TT_THROW_IF_ERROR(CheckNotBool(other, /* arg_name= */ "other"));
        TT_ASSIGN_OR_THROW(auto out_dtypes, GetOutputDtypes(self, other));

        // Check for invalid input types.
        size_t num_tensors = self.size();
        for (size_t i = 0; i < num_tensors; ++i) {
          TT_CHECK_THROW(!(IsIntegral(self[i]) && IsIntegral(other[i]) &&
                           !IsIntegral(alpha)),
                         error::kInvalidArgument)
              << "expected alpha to be integral for integral input tensors, "
                 "got "
              << ToString(alpha.type());
        }

        return ForeachConvertToTensor(
            ForeachAddList(self, other, std::move(promoted_neg_alpha),
                           out_dtypes, std::move(param_keys)),
            out_dtypes);
      });
}

void AtenForeachSub_List(at::TensorList self, at::TensorList other,
                         const at::Scalar& alpha) {
  if (self.empty()) {
    return;
  }
  auto promoted_neg_alpha = PromoteScalar(-alpha).AvoidPromoting(
      ScalarValue::kOne, ScalarValue::kMinusOne);
  TT_KERNEL(
      OpName::kForeachSub_List, param_keys, (self, other, promoted_neg_alpha), {
        // _foreach_add supports bool, but _foreach_sub does not.
        TT_THROW_IF_ERROR(CheckNotBool(alpha, /* arg_name= */ "alpha"));
        TT_THROW_IF_ERROR(CheckNotBool(self, /* arg_name= */ "self"));
        TT_THROW_IF_ERROR(CheckNotBool(other, /* arg_name= */ "other"));
        TT_ASSIGN_OR_THROW(auto out_dtypes, GetOutputDtypes(self, other));

        // Check for invalid input types.
        size_t num_tensors = self.size();
        for (size_t i = 0; i < num_tensors; ++i) {
          TT_CHECK_THROW(!(IsIntegral(self[i]) && IsIntegral(other[i]) &&
                           !IsIntegral(alpha)),
                         error::kInvalidArgument)
              << "expected alpha to be integral for integral input tensors, "
                 "got "
              << ToString(alpha.type());
        }

        TT_THROW_IF_ERROR(ForeachAssignToTensor(
            ForeachAddList(self, other, std::move(promoted_neg_alpha),
                           out_dtypes, std::move(param_keys)),
            self, out_dtypes));
      });
}

std::vector<at::Tensor> AtenForeachSubScalar(at::TensorList self,
                                             const at::Scalar& scalar) {
  if (self.empty()) {
    return {};
  }
  auto promoted_scalar = PromoteScalar(scalar);
  TT_KERNEL(OpName::kForeachSubScalar, _, (self, promoted_scalar), {
    // _foreach_add supports bool, but _foreach_sub does not.
    TT_THROW_IF_ERROR(CheckNotBool(scalar, /* arg_name= */ "scalar"));
    TT_THROW_IF_ERROR(CheckNotBool(self, /* arg_name= */ "self"));
    TT_ASSIGN_OR_THROW(auto out_dtypes, GetOutputDtypes(self, scalar));
    std::vector<at::Tensor> other;
    other.reserve(self.size());
    for (size_t i = 0; i < self.size(); ++i) {
      at::ScalarType scalar_type = ConvertTo<at::ScalarType>(out_dtypes[i]);
      TT_ASSIGN_OR_THROW(at::Tensor scalar_tensor,
                         promoted_scalar.GetTensor(scalar_type));
      other.push_back(scalar_tensor);
    }
    return ForeachConvertToTensor(
        ForeachAddList(
            self, other,
            PromoteScalar(at::Scalar(-1.0)).AvoidPromoting(ScalarValue::kOne),
            out_dtypes),
        out_dtypes);
  });
}

void AtenForeachSub_Scalar(at::TensorList self, const at::Scalar& scalar) {
  if (self.empty()) {
    return;
  }
  auto promoted_scalar = PromoteScalar(scalar);
  TT_KERNEL(OpName::kForeachSub_Scalar, _, (self, promoted_scalar), {
    // _foreach_add supports bool, but _foreach_sub does not.
    TT_THROW_IF_ERROR(CheckNotBool(scalar, /* arg_name= */ "scalar"));
    TT_THROW_IF_ERROR(CheckNotBool(self, /* arg_name= */ "self"));
    TT_ASSIGN_OR_THROW(auto out_dtypes, GetOutputDtypes(self, scalar));
    std::vector<at::Tensor> other;
    other.reserve(self.size());
    for (size_t i = 0; i < self.size(); ++i) {
      at::ScalarType scalar_type = ConvertTo<at::ScalarType>(out_dtypes[i]);
      TT_ASSIGN_OR_THROW(at::Tensor scalar_tensor,
                         promoted_scalar.GetTensor(scalar_type));
      other.push_back(scalar_tensor);
    }
    TT_THROW_IF_ERROR(ForeachAssignToTensor(
        ForeachAddList(
            self, other,
            PromoteScalar(at::Scalar(-1.0)).AvoidPromoting(ScalarValue::kOne),
            out_dtypes),
        self, out_dtypes));
  });
}

std::vector<at::Tensor> AtenForeachSubScalarList(
    at::TensorList self, at::ArrayRef<at::Scalar> scalars) {
  if (self.empty()) {
    return {};
  }
  auto promoted_scalars = PromoteScalar(scalars);
  TT_KERNEL(OpName::kForeachSubScalarList, _, (self, promoted_scalars), {
    // _foreach_add supports bool, but _foreach_sub does not.
    TT_THROW_IF_ERROR(CheckNotBool(scalars, /* arg_name= */ "scalars"));
    TT_THROW_IF_ERROR(CheckNotBool(self, /* arg_name= */ "self"));
    TT_ASSIGN_OR_THROW(auto out_dtypes, GetOutputDtypes(self, scalars));
    std::vector<at::Tensor> other;
    other.reserve(self.size());
    for (size_t i = 0; i < self.size(); ++i) {
      at::ScalarType scalar_type = ConvertTo<at::ScalarType>(out_dtypes[i]);
      TT_ASSIGN_OR_THROW(at::Tensor scalar_tensor,
                         promoted_scalars[i].GetTensor(scalar_type));
      other.push_back(scalar_tensor);
    }
    return ForeachConvertToTensor(
        ForeachAddList(
            self, other,
            PromoteScalar(at::Scalar(-1.0)).AvoidPromoting(ScalarValue::kOne),
            out_dtypes),
        out_dtypes);
  });
}

void AtenForeachSub_ScalarList(at::TensorList self,
                               at::ArrayRef<at::Scalar> scalars) {
  if (self.empty()) {
    return;
  }
  auto promoted_scalars = PromoteScalar(scalars);
  TT_KERNEL(OpName::kForeachSub_ScalarList, _, (self, promoted_scalars), {
    // _foreach_add supports bool, but _foreach_sub does not.
    TT_THROW_IF_ERROR(CheckNotBool(scalars, /* arg_name= */ "scalars"));
    TT_THROW_IF_ERROR(CheckNotBool(self, /* arg_name= */ "self"));
    TT_ASSIGN_OR_THROW(auto out_dtypes, GetOutputDtypes(self, scalars));
    std::vector<at::Tensor> other;
    other.reserve(self.size());
    for (size_t i = 0; i < self.size(); ++i) {
      at::ScalarType scalar_type = ConvertTo<at::ScalarType>(out_dtypes[i]);
      TT_ASSIGN_OR_THROW(at::Tensor scalar_tensor,
                         promoted_scalars[i].GetTensor(scalar_type));
      other.push_back(scalar_tensor);
    }
    TT_THROW_IF_ERROR(ForeachAssignToTensor(
        ForeachAddList(
            self, other,
            PromoteScalar(at::Scalar(-1.0)).AvoidPromoting(ScalarValue::kOne),
            out_dtypes),
        self, out_dtypes));
  });
}

absl::StatusOr<at::Tensor> AtenClampMax(const at::Tensor& self,
                                        const at::Scalar& alpha) {
  TT_ASSIGN_OR_RETURN(
      at::Tensor out,
      MakeEmptyTensor(self.sizes(), self.scalar_type(), self.device()));
  AtenClampMaxOut(self, alpha, out);
  return out;
}

absl::StatusOr<at::Tensor> AtenClampMax(const at::Tensor& self,
                                        const at::Tensor& other) {
  TT_ASSIGN_OR_RETURN(
      at::Tensor out,
      MakeEmptyTensor(self.sizes(), self.scalar_type(), self.device()));
  AtenClampMaxTensorOut(self, other, out);
  return out;
}

std::vector<at::Tensor> AtenForeachClampMaxScalar(at::TensorList self,
                                                  const at::Scalar& scalar) {
  auto promoted_scalar = PromoteScalar(scalar);
  TT_KERNEL(OpName::kForeachClampMaxScalar, _, (self, promoted_scalar), {
    std::vector<at::Tensor> result;
    result.reserve(self.size());
    for (const auto& tensor : self) {
      TT_ASSIGN_OR_THROW(at::Tensor scalar_tensor,
                         promoted_scalar.GetTensor(tensor.scalar_type()));
      TT_ASSIGN_OR_THROW(auto out, AtenClampMax(tensor, scalar_tensor));
      result.push_back(out);
    }
    return result;
  });
}

void AtenForeachClampMax_Scalar(at::TensorList self, const at::Scalar& scalar) {
  auto promoted_scalar = PromoteScalar(scalar);
  TT_KERNEL(OpName::kForeachClampMax_Scalar, _, (self, promoted_scalar), {
    for (const auto& tensor : self) {
      TT_ASSIGN_OR_THROW(at::Tensor scalar_tensor,
                         promoted_scalar.GetTensor(tensor.scalar_type()));
      AtenClampMaxTensorOut(tensor, scalar_tensor,
                            const_cast<at::Tensor&>(tensor));
    }
  });
}

std::vector<at::Tensor> AtenForeachClampMaxList(at::TensorList self,
                                                at::TensorList other) {
  TT_KERNEL(OpName::kForeachClampMaxList, _, (self, other), {
    std::vector<at::Tensor> result;
    result.reserve(self.size());
    for (size_t i = 0; i < self.size(); ++i) {
      TT_ASSIGN_OR_THROW(auto out, AtenClampMax(self[i], other[i]));
      result.push_back(out);
    }
    return result;
  });
}

void AtenForeachClampMax_List(at::TensorList self, at::TensorList other) {
  TT_KERNEL(OpName::kForeachClampMax_List, _, (self, other), {
    for (size_t i = 0; i < self.size(); ++i) {
      AtenClampMaxTensorOut(self[i], other[i],
                            const_cast<at::Tensor&>(self[i]));
    }
  });
}

std::vector<at::Tensor> AtenForeachClampMaxScalarList(
    at::TensorList self, at::ArrayRef<at::Scalar> scalars) {
  auto promoted_scalars = PromoteScalar(scalars);
  TT_KERNEL(OpName::kForeachClampMaxScalarList, _, (self, promoted_scalars), {
    std::vector<at::Tensor> result;
    result.reserve(self.size());
    for (size_t i = 0; i < self.size(); ++i) {
      TT_ASSIGN_OR_THROW(at::Tensor scalar_tensor,
                         promoted_scalars[i].GetTensor(self[i].scalar_type()));
      TT_ASSIGN_OR_THROW(auto out, AtenClampMax(self[i], scalar_tensor));
      result.push_back(out);
    }
    return result;
  });
}

void AtenForeachClampMax_ScalarList(at::TensorList self,
                                    at::ArrayRef<at::Scalar> scalars) {
  auto promoted_scalars = PromoteScalar(scalars);
  TT_KERNEL(OpName::kForeachClampMax_ScalarList, _, (self, promoted_scalars), {
    for (size_t i = 0; i < self.size(); ++i) {
      TT_ASSIGN_OR_THROW(at::Tensor scalar_tensor,
                         promoted_scalars[i].GetTensor(self[i].scalar_type()));
      AtenClampMaxTensorOut(self[i], scalar_tensor,
                            const_cast<at::Tensor&>(self[i]));
    }
  });
}

absl::StatusOr<at::Tensor> AtenClampMin(const at::Tensor& self,
                                        const at::Scalar& alpha) {
  TT_ASSIGN_OR_RETURN(
      at::Tensor out,
      MakeEmptyTensor(self.sizes(), self.scalar_type(), self.device()));
  AtenClampMinOut(self, alpha, out);
  return out;
}

absl::StatusOr<at::Tensor> AtenClampMin(const at::Tensor& self,
                                        const at::Tensor& other) {
  TT_ASSIGN_OR_RETURN(
      at::Tensor out,
      MakeEmptyTensor(self.sizes(), self.scalar_type(), self.device()));
  AtenClampMinTensorOut(self, other, out);
  return out;
}

std::vector<at::Tensor> AtenForeachClampMinScalar(at::TensorList self,
                                                  const at::Scalar& scalar) {
  auto promoted_scalar = PromoteScalar(scalar);
  TT_KERNEL(OpName::kForeachClampMinScalar, _, (self, promoted_scalar), {
    std::vector<at::Tensor> result;
    result.reserve(self.size());
    for (const auto& tensor : self) {
      TT_ASSIGN_OR_THROW(at::Tensor scalar_tensor,
                         promoted_scalar.GetTensor(tensor.scalar_type()));
      TT_ASSIGN_OR_THROW(auto out, AtenClampMin(tensor, scalar_tensor));
      result.push_back(out);
    }
    return result;
  });
}

void AtenForeachClampMin_Scalar(at::TensorList self, const at::Scalar& scalar) {
  auto promoted_scalar = PromoteScalar(scalar);
  TT_KERNEL(OpName::kForeachClampMin_Scalar, _, (self, promoted_scalar), {
    for (const auto& tensor : self) {
      TT_ASSIGN_OR_THROW(at::Tensor scalar_tensor,
                         promoted_scalar.GetTensor(tensor.scalar_type()));
      AtenClampMinTensorOut(tensor, scalar_tensor,
                            const_cast<at::Tensor&>(tensor));
    }
  });
}

std::vector<at::Tensor> AtenForeachClampMinList(at::TensorList self,
                                                at::TensorList other) {
  TT_KERNEL(OpName::kForeachClampMinList, _, (self, other), {
    std::vector<at::Tensor> result;
    result.reserve(self.size());
    for (size_t i = 0; i < self.size(); ++i) {
      TT_ASSIGN_OR_THROW(auto out, AtenClampMin(self[i], other[i]));
      result.push_back(out);
    }
    return result;
  });
}

void AtenForeachClampMin_List(at::TensorList self, at::TensorList other) {
  TT_KERNEL(OpName::kForeachClampMin_List, _, (self, other), {
    for (size_t i = 0; i < self.size(); ++i) {
      AtenClampMinTensorOut(self[i], other[i],
                            const_cast<at::Tensor&>(self[i]));
    }
  });
}

std::vector<at::Tensor> AtenForeachClampMinScalarList(
    at::TensorList self, at::ArrayRef<at::Scalar> scalars) {
  auto promoted_scalars = PromoteScalar(scalars);
  TT_KERNEL(OpName::kForeachClampMinScalarList, _, (self, promoted_scalars), {
    std::vector<at::Tensor> result;
    result.reserve(self.size());
    for (size_t i = 0; i < self.size(); ++i) {
      TT_ASSIGN_OR_THROW(at::Tensor scalar_tensor,
                         promoted_scalars[i].GetTensor(self[i].scalar_type()));
      TT_ASSIGN_OR_THROW(auto out, AtenClampMin(self[i], scalar_tensor));
      result.push_back(out);
    }
    return result;
  });
}

void AtenForeachClampMin_ScalarList(at::TensorList self,
                                    at::ArrayRef<at::Scalar> scalars) {
  auto promoted_scalars = PromoteScalar(scalars);
  TT_KERNEL(OpName::kForeachClampMin_ScalarList, _, (self, promoted_scalars), {
    for (size_t i = 0; i < self.size(); ++i) {
      TT_ASSIGN_OR_THROW(at::Tensor scalar_tensor,
                         promoted_scalars[i].GetTensor(self[i].scalar_type()));
      AtenClampMinTensorOut(self[i], scalar_tensor,
                            const_cast<at::Tensor&>(self[i]));
    }
  });
}

void AtenForeachCopy_(at::TensorList self, at::TensorList src,
                      bool non_blocking) {
  TT_KERNEL(
      OpName::kForeachCopy_, _,
      (self, src, IgnoreInCacheKey(non_blocking, "Delegates to AtenCopy_")), {
        for (size_t i = 0; i < self.size(); ++i) {
          AtenCopy_(const_cast<at::Tensor&>(self[i]), src[i], non_blocking);
        }
      });
}

std::vector<at::Tensor> AtenForeachNormScalar(
    at::TensorList self, const at::Scalar& ord,
    const std::optional<c10::ScalarType> dtype) {
  auto promoted_ord = PromoteScalar(ord);
  TT_KERNEL(
      OpName::kForeachNormScalar, _,
      (self, promoted_ord,
       IgnoreInCacheKey(dtype, "Delegates to AtenLinalgVectorNormOut")),
      {
        TT_THROW_IF_ERROR(CheckNotIntegral(self, /* arg_name= */ "self"));
        TT_ASSIGN_OR_THROW(at::Tensor ord_tensor, promoted_ord.GetTensor());
        std::vector<at::Tensor> result;
        result.reserve(self.size());
        for (const auto& tensor : self) {
          auto out_dtype =
              at::toRealValueType(dtype.value_or(tensor.scalar_type()));
          TT_ASSIGN_OR_THROW(auto out,
                             MakeEmptyTensor({}, out_dtype, tensor.device()));
          // If the tensor is empty, return zero scalar.
          // If ord < 0, to avoid inf caused by pow(0, ord), check for 0s
          // in the tensor and return zero directly, which is the correct
          // result.
          if (tensor.numel() == 0 || (promoted_ord.scalar().to<double>() < 0 &&
                                      tensor.eq(0).any().item<bool>())) {
            out.fill_(0);
            result.push_back(out);
            continue;
          }
          AtenLinalgVectorNormOut(tensor.to(out_dtype), promoted_ord.scalar(),
                                  /*dim=*/{},
                                  /*keepdim=*/false, dtype, out);
          result.push_back(out);
        }
        return result;
      });
}

std::vector<at::Tensor> AtenForeachMax(at::TensorList self) {
  TT_KERNEL(OpName::kForeachMax, _, (self), {
    std::vector<at::Tensor> result;
    result.reserve(self.size());
    for (const auto& tensor : self) {
      auto out = AtenMax(tensor);
      result.push_back(out);
    }
    return result;
  });
}

absl::StatusOr<at::Tensor> AtenMaximum(const at::Tensor& self,
                                       const at::Tensor& other) {
  TT_ASSIGN_OR_RETURN(
      at::Tensor out,
      MakeEmptyTensor(self.sizes(), self.scalar_type(), self.device()));
  AtenMaximumOut(self, other, out);
  return out;
}

std::vector<at::Tensor> AtenForeachMaximumScalar(at::TensorList self,
                                                 const at::Scalar& scalar) {
  auto promoted_scalar = PromoteScalar(scalar);
  TT_KERNEL(OpName::kForeachMaximumScalar, _, (self, promoted_scalar), {
    TT_ASSIGN_OR_THROW(auto scalar_tensor, promoted_scalar.GetTensor());
    std::vector<at::Tensor> result;
    result.reserve(self.size());
    for (const auto& tensor : self) {
      TT_ASSIGN_OR_THROW(auto out, AtenMaximum(tensor, scalar_tensor));
      result.push_back(out);
    }
    return result;
  });
}

void AtenForeachMaximum_Scalar(at::TensorList self, const at::Scalar& scalar) {
  auto promoted_scalar = PromoteScalar(scalar);
  TT_KERNEL(OpName::kForeachMaximum_Scalar, _, (self, promoted_scalar), {
    TT_ASSIGN_OR_THROW(auto scalar_tensor, promoted_scalar.GetTensor());
    for (const auto& tensor : self) {
      AtenMaximumOut(tensor, scalar_tensor, const_cast<at::Tensor&>(tensor));
    }
  });
}

std::vector<at::Tensor> AtenForeachMaximumList(at::TensorList self,
                                               at::TensorList other) {
  TT_KERNEL(OpName::kForeachMaximumList, _, (self, other), {
    std::vector<at::Tensor> result;
    result.reserve(self.size());
    for (size_t i = 0; i < self.size(); ++i) {
      TT_ASSIGN_OR_THROW(auto out, AtenMaximum(self[i], other[i]));
      result.push_back(out);
    }
    return result;
  });
}

void AtenForeachMaximum_List(at::TensorList self, at::TensorList other) {
  TT_KERNEL(OpName::kForeachMaximum_List, _, (self, other), {
    for (size_t i = 0; i < self.size(); ++i) {
      AtenMaximumOut(self[i], other[i], const_cast<at::Tensor&>(self[i]));
    }
  });
}

std::vector<at::Tensor> AtenForeachMaximumScalarList(
    at::TensorList self, at::ArrayRef<at::Scalar> scalars) {
  auto promoted_scalars = PromoteScalar(scalars);
  TT_KERNEL(OpName::kForeachMaximumScalarList, _, (self, promoted_scalars), {
    std::vector<at::Tensor> result;
    result.reserve(self.size());
    for (size_t i = 0; i < self.size(); ++i) {
      TT_ASSIGN_OR_THROW(auto scalar_tensor, promoted_scalars[i].GetTensor());
      TT_ASSIGN_OR_THROW(auto out, AtenMaximum(self[i], scalar_tensor));
      result.push_back(out);
    }
    return result;
  });
}

void AtenForeachMaximum_ScalarList(at::TensorList self,
                                   at::ArrayRef<at::Scalar> scalars) {
  auto promoted_scalars = PromoteScalar(scalars);
  TT_KERNEL(OpName::kForeachMaximum_ScalarList, _, (self, promoted_scalars), {
    for (size_t i = 0; i < self.size(); ++i) {
      TT_ASSIGN_OR_THROW(auto scalar_tensor, promoted_scalars[i].GetTensor());
      AtenMaximumOut(self[i], scalar_tensor, const_cast<at::Tensor&>(self[i]));
    }
  });
}

absl::StatusOr<at::Tensor> AtenMinimum(const at::Tensor& self,
                                       const at::Tensor& other) {
  TT_ASSIGN_OR_RETURN(
      at::Tensor out,
      MakeEmptyTensor(self.sizes(), self.scalar_type(), self.device()));
  AtenMinimumOut(self, other, out);
  return out;
}

std::vector<at::Tensor> AtenForeachMinimumScalar(at::TensorList self,
                                                 const at::Scalar& scalar) {
  auto promoted_scalar = PromoteScalar(scalar);
  TT_KERNEL(OpName::kForeachMinimumScalar, _, (self, promoted_scalar), {
    TT_ASSIGN_OR_THROW(auto scalar_tensor, promoted_scalar.GetTensor());
    std::vector<at::Tensor> result;
    result.reserve(self.size());
    for (const auto& tensor : self) {
      TT_ASSIGN_OR_THROW(auto out, AtenMinimum(tensor, scalar_tensor));
      result.push_back(out);
    }
    return result;
  });
}

void AtenForeachMinimum_Scalar(at::TensorList self, const at::Scalar& scalar) {
  auto promoted_scalar = PromoteScalar(scalar);
  TT_KERNEL(OpName::kForeachMinimum_Scalar, _, (self, promoted_scalar), {
    TT_ASSIGN_OR_THROW(auto scalar_tensor, promoted_scalar.GetTensor());
    for (const auto& tensor : self) {
      AtenMinimumOut(tensor, scalar_tensor, const_cast<at::Tensor&>(tensor));
    }
  });
}

std::vector<at::Tensor> AtenForeachMinimumList(at::TensorList self,
                                               at::TensorList other) {
  TT_KERNEL(OpName::kForeachMinimumList, _, (self, other), {
    std::vector<at::Tensor> result;
    result.reserve(self.size());
    for (size_t i = 0; i < self.size(); ++i) {
      TT_ASSIGN_OR_THROW(auto out, AtenMinimum(self[i], other[i]));
      result.push_back(out);
    }
    return result;
  });
}

void AtenForeachMinimum_List(at::TensorList self, at::TensorList other) {
  TT_KERNEL(OpName::kForeachMinimum_List, _, (self, other), {
    for (size_t i = 0; i < self.size(); ++i) {
      AtenMinimumOut(self[i], other[i], const_cast<at::Tensor&>(self[i]));
    }
  });
}

std::vector<at::Tensor> AtenForeachMinimumScalarList(
    at::TensorList self, at::ArrayRef<at::Scalar> scalars) {
  auto promoted_scalars = PromoteScalar(scalars);
  TT_KERNEL(OpName::kForeachMinimumScalarList, _, (self, promoted_scalars), {
    std::vector<at::Tensor> result;
    result.reserve(self.size());
    for (size_t i = 0; i < self.size(); ++i) {
      TT_ASSIGN_OR_THROW(auto scalar_tensor, promoted_scalars[i].GetTensor());
      TT_ASSIGN_OR_THROW(auto out, AtenMinimum(self[i], scalar_tensor));
      result.push_back(out);
    }
    return result;
  });
}

void AtenForeachMinimum_ScalarList(at::TensorList self,
                                   at::ArrayRef<at::Scalar> scalars) {
  auto promoted_scalars = PromoteScalar(scalars);
  TT_KERNEL(OpName::kForeachMinimum_ScalarList, _, (self, promoted_scalars), {
    for (size_t i = 0; i < self.size(); ++i) {
      TT_ASSIGN_OR_THROW(auto scalar_tensor, promoted_scalars[i].GetTensor());
      AtenMinimumOut(self[i], scalar_tensor, const_cast<at::Tensor&>(self[i]));
    }
  });
}

absl::StatusOr<at::Tensor> AtenPow(const at::Tensor& self,
                                   const at::Tensor& exponent) {
  TT_ASSIGN_OR_RETURN(
      at::Tensor out,
      MakeEmptyTensor(self.sizes(), self.scalar_type(), self.device()));
  AtenPowTensorTensorOut(self, exponent, out);
  return out;
}

absl::StatusOr<at::Tensor> AtenPow(const at::Tensor& self,
                                   const at::Scalar& exponent) {
  TT_ASSIGN_OR_RETURN(
      at::Tensor out,
      MakeEmptyTensor(self.sizes(), self.scalar_type(), self.device()));
  AtenPowTensorScalarOut(self, exponent, out);
  return out;
}

absl::StatusOr<at::Tensor> AtenPow(const at::Scalar& self,
                                   const at::Tensor& exponent) {
  TT_ASSIGN_OR_RETURN(at::Tensor out,
                      MakeEmptyTensor(exponent.sizes(), exponent.scalar_type(),
                                      exponent.device()));
  AtenPowScalarOut(self, exponent, out);
  return out;
}

std::vector<at::Tensor> AtenForeachPowList(at::TensorList self,
                                           at::TensorList exponent) {
  TT_KERNEL(OpName::kForeachPowList, _, (self, exponent), {
    std::vector<at::Tensor> result;
    result.reserve(self.size());
    for (size_t i = 0; i < self.size(); ++i) {
      const at::Tensor exp_i = BoolTensorToInt(exponent[i]);
      TT_ASSIGN_OR_THROW(auto out, AtenPow(self[i], exp_i));
      result.push_back(out);
    }
    return result;
  });
}

void AtenForeachPow_List(at::TensorList self, at::TensorList exponent) {
  TT_KERNEL(OpName::kForeachPow_List, _, (self, exponent), {
    for (size_t i = 0; i < self.size(); ++i) {
      const at::Tensor exp_i = BoolTensorToInt(exponent[i]);
      AtenPowTensorTensorOut(self[i], exp_i, const_cast<at::Tensor&>(self[i]));
    }
  });
}

std::vector<at::Tensor> AtenForeachPowScalar(at::TensorList self,
                                             const at::Scalar& exponent) {
  auto promoted_exponent = PromoteScalar(BoolScalarToInt(exponent));
  TT_KERNEL(OpName::kForeachPowScalar, _, (self, promoted_exponent), {
    std::vector<at::Tensor> result;
    result.reserve(self.size());
    TT_ASSIGN_OR_THROW(at::Tensor exponent_tensor,
                       promoted_exponent.GetTensor());
    for (const auto& tensor : self) {
      TT_ASSIGN_OR_THROW(auto out, AtenPow(tensor, exponent_tensor));
      result.push_back(out);
    }
    return result;
  });
}

void AtenForeachPow_Scalar(at::TensorList self, const at::Scalar& exponent) {
  auto promoted_exponent = PromoteScalar(BoolScalarToInt(exponent));
  TT_KERNEL(OpName::kForeachPow_Scalar, _, (self, promoted_exponent), {
    TT_ASSIGN_OR_THROW(at::Tensor exponent_tensor,
                       promoted_exponent.GetTensor());
    for (const auto& tensor : self) {
      AtenPowTensorTensorOut(tensor, exponent_tensor,
                             const_cast<at::Tensor&>(tensor));
    }
  });
}

std::vector<at::Tensor> AtenForeachPowScalarList(
    at::TensorList self, at::ArrayRef<at::Scalar> exponent) {
  std::vector<PromotedScalar> promoted_exponent;
  promoted_exponent.reserve(exponent.size());
  for (const auto& scalar : exponent) {
    promoted_exponent.push_back(PromoteScalar(BoolScalarToInt(scalar)));
  }
  TT_KERNEL(OpName::kForeachPowScalarList, _, (self, promoted_exponent), {
    std::vector<at::Tensor> result;
    result.reserve(self.size());
    for (size_t i = 0; i < self.size(); ++i) {
      TT_ASSIGN_OR_THROW(at::Tensor exponent_tensor_i,
                         promoted_exponent[i].GetTensor());
      TT_ASSIGN_OR_THROW(auto out, AtenPow(self[i], exponent_tensor_i));
      result.push_back(out);
    }
    return result;
  });
}

void AtenForeachPow_ScalarList(at::TensorList self,
                               at::ArrayRef<at::Scalar> exponent) {
  std::vector<PromotedScalar> promoted_exponent;
  promoted_exponent.reserve(exponent.size());
  for (const auto& scalar : exponent) {
    promoted_exponent.push_back(PromoteScalar(BoolScalarToInt(scalar)));
  }
  TT_KERNEL(OpName::kForeachPow_ScalarList, _, (self, promoted_exponent), {
    for (size_t i = 0; i < self.size(); ++i) {
      TT_ASSIGN_OR_THROW(at::Tensor exponent_tensor_i,
                         promoted_exponent[i].GetTensor());
      AtenPowTensorTensorOut(self[i], exponent_tensor_i,
                             const_cast<at::Tensor&>(self[i]));
    }
  });
}

std::vector<at::Tensor> AtenForeachPowScalarAndTensor(const at::Scalar& self,
                                                      at::TensorList exponent) {
  auto promoted_self = PromoteScalar(self);
  TT_KERNEL(OpName::kForeachPowScalarAndTensor, _, (promoted_self, exponent), {
    std::vector<at::Tensor> result;
    result.reserve(exponent.size());
    TT_ASSIGN_OR_THROW(at::Tensor base_tensor, promoted_self.GetTensor());
    for (const auto& tensor : exponent) {
      const at::Tensor exp_i = BoolTensorToInt(tensor);
      TT_ASSIGN_OR_THROW(auto out, AtenPow(base_tensor, exp_i));
      result.push_back(out);
    }
    return result;
  });
}

}  // namespace torch_tpu
