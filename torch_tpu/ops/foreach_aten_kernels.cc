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
#include <memory>
#include <optional>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/functional/any_invocable.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "mlir/Support/LLVM.h"
#include "ATen/core/ATen_fwd.h"
#include "c10/core/DefaultDtype.h"
#include "c10/core/ScalarType.h"
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
#include "torch_tpu/ops/unary.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/ChloBuilder.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"

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

std::vector<at::Tensor> ForeachConvertToTensor(
    std::vector<DeviceBufferRef> result_buffers) {
  std::vector<at::Tensor> result;
  result.reserve(result_buffers.size());
  for (auto& result_buffer : result_buffers) {
    result.push_back(MakeTensor(std::move(result_buffer)));
  }
  return result;
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

using DtypeVec = std::vector<mlir::ElementType>;
using UniqueDtypeVec = absl_nonnull std::unique_ptr<DtypeVec>;
using DtypeSpan = absl::Span<const mlir::ElementType>;

absl::StatusOr<UniqueDtypeVec> GetOutputDtypes(
    at::TensorList self, bool cast_integral_to_float = false,
    bool cast_complex_to_float = false) {
  const at::ScalarType default_dtype = c10::get_default_dtype_as_scalartype();
  UniqueDtypeVec out_dtypes_vec = std::make_unique<DtypeVec>();
  out_dtypes_vec->reserve(self.size());
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
    out_dtypes_vec->push_back(output_element_type);
  }
  return out_dtypes_vec;
}

absl::StatusOr<UniqueDtypeVec> GetOutputDtypes(at::TensorList self,
                                               at::TensorList other) {
  UniqueDtypeVec out_dtypes_vec = std::make_unique<DtypeVec>();
  out_dtypes_vec->reserve(self.size());
  for (size_t i = 0; i < self.size(); ++i) {
    const at::ScalarType output_scalar_type =
        c10::promoteTypes(self[i].scalar_type(), other[i].scalar_type());
    TT_ASSIGN_OR_RETURN(const auto output_element_type,
                        ConvertTo<mlir::ElementType>(output_scalar_type));
    out_dtypes_vec->push_back(output_element_type);
  }
  return out_dtypes_vec;
}

absl::StatusOr<UniqueDtypeVec> GetOutputDtypes(at::TensorList self,
                                               const at::Scalar& scalar) {
  UniqueDtypeVec out_dtypes_vec = std::make_unique<DtypeVec>();
  out_dtypes_vec->reserve(self.size());
  for (size_t i = 0; i < self.size(); ++i) {
    const at::ScalarType output_scalar_type =
        GetOutputDtypeFromTensorAndScalar(self[i], scalar);
    TT_ASSIGN_OR_RETURN(const auto output_element_type,
                        ConvertTo<mlir::ElementType>(output_scalar_type));
    out_dtypes_vec->push_back(output_element_type);
  }
  return out_dtypes_vec;
}

absl::StatusOr<UniqueDtypeVec> GetOutputDtypes(
    at::TensorList self, at::ArrayRef<at::Scalar> scalars) {
  UniqueDtypeVec out_dtypes_vec = std::make_unique<DtypeVec>();
  out_dtypes_vec->reserve(self.size());
  for (size_t i = 0; i < self.size(); ++i) {
    const at::ScalarType output_scalar_type =
        GetOutputDtypeFromTensorAndScalar(self[i], scalars[i]);
    TT_ASSIGN_OR_RETURN(const auto output_element_type,
                        ConvertTo<mlir::ElementType>(output_scalar_type));
    out_dtypes_vec->push_back(output_element_type);
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
  const DtypeSpan out_dtypes_span = *out_dtypes;
  const DtypeSpan result_out_dtypes_span = *result_out_dtypes;
  for (size_t i = 0; i < self.size(); ++i) {
    TT_RETURN_IF_ERROR(CheckScalarType(out_dtypes_span[i],
                                       result_out_dtypes_span[i],
                                       self[i].scalar_type(), scalar.type()));
  }
  return absl::OkStatus();
}

absl::Status CheckInplaceScalarType(at::TensorList self,
                                    at::ArrayRef<at::Scalar> scalars) {
  TT_ASSIGN_OR_RETURN(auto out_dtypes, GetOutputDtypes(self));
  TT_ASSIGN_OR_RETURN(const auto result_out_dtypes,
                      GetOutputDtypes(self, scalars));
  const DtypeSpan out_dtypes_span = *out_dtypes;
  const DtypeSpan result_out_dtypes_span = *result_out_dtypes;
  for (size_t i = 0; i < self.size(); ++i) {
    TT_RETURN_IF_ERROR(
        CheckScalarType(out_dtypes_span[i], result_out_dtypes_span[i],
                        self[i].scalar_type(), scalars[i].type()));
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
  TT_RETURN_IF_ERROR(CheckTensorsNotTypeImpl(tensors, arg_name,
                                             /* is_type= */ IsIntegral,
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

std::vector<at::Tensor> GetOtherTensors(at::TensorList tensor_list,
                                        const DtypeVec& out_dtypes) {
  std::vector<at::Tensor> other;
  other.reserve(tensor_list.size());
  for (size_t i = 0; i < tensor_list.size(); ++i) {
    at::ScalarType scalar_type = ConvertTo<at::ScalarType>(out_dtypes[i]);
    other.push_back(tensor_list[i].to(scalar_type));
  }
  return other;
}

std::vector<at::Tensor> GetOtherTensors(
    size_t size, internal::PromotedScalar& promoted_scalar,
    const DtypeVec& out_dtypes) {
  std::vector<at::Tensor> other;
  other.reserve(size);
  for (size_t i = 0; i < size; ++i) {
    at::ScalarType scalar_type = ConvertTo<at::ScalarType>(out_dtypes[i]);
    TT_ASSIGN_OR_THROW(at::Tensor scalar_tensor,
                       promoted_scalar.GetTensor(scalar_type));
    other.push_back(scalar_tensor);
  }
  return other;
}

std::vector<at::Tensor> GetOtherTensors(
    size_t size, std::vector<internal::PromotedScalar>& promoted_scalars,
    const DtypeVec& out_dtypes) {
  std::vector<at::Tensor> other;
  other.reserve(size);
  for (size_t i = 0; i < size; ++i) {
    at::ScalarType scalar_type = ConvertTo<at::ScalarType>(out_dtypes[i]);
    TT_ASSIGN_OR_THROW(at::Tensor scalar_tensor,
                       promoted_scalars[i].GetTensor(scalar_type));
    other.push_back(scalar_tensor);
  }
  return other;
}

std::vector<at::Tensor> GetOtherTensorsInplace(
    size_t size, internal::PromotedScalar& promoted_scalar,
    at::TensorList self) {
  std::vector<at::Tensor> other;
  other.reserve(size);
  for (size_t i = 0; i < size; ++i) {
    TT_ASSIGN_OR_THROW(at::Tensor scalar_tensor,
                       promoted_scalar.GetTensor(self[i].scalar_type()));
    other.push_back(scalar_tensor);
  }
  return other;
}

std::vector<at::Tensor> GetOtherTensorsInplace(
    size_t size, std::vector<internal::PromotedScalar>& promoted_scalars,
    at::TensorList self) {
  std::vector<at::Tensor> other;
  other.reserve(size);
  for (size_t i = 0; i < size; ++i) {
    TT_ASSIGN_OR_THROW(at::Tensor scalar_tensor,
                       promoted_scalars[i].GetTensor(self[i].scalar_type()));
    other.push_back(scalar_tensor);
  }
  return other;
}

template <typename F>
std::vector<at::Tensor> ForeachOpFunctional(at::TensorList self, F f) {
  std::vector<at::Tensor> result;
  result.reserve(self.size());
  for (const auto& tensor : self) {
    result.push_back(f(tensor));
  }
  return result;
}

template <typename T, typename F>
std::vector<at::Tensor> ForeachOpFunctional(at::TensorList self,
                                            at::ArrayRef<T> other, F f) {
  std::vector<at::Tensor> result;
  result.reserve(self.size());
  for (size_t i = 0; i < self.size(); ++i) {
    result.push_back(f(self[i], other[i]));
  }
  return result;
}

template <typename T, typename F>
std::vector<at::Tensor> ForeachOpFunctional(at::TensorList self,
                                            std::vector<T>& other, F f) {
  std::vector<at::Tensor> result;
  result.reserve(self.size());
  for (size_t i = 0; i < self.size(); ++i) {
    result.push_back(f(self[i], other[i]));
  }
  return result;
}

template <typename F>
void ForeachOpInplace(at::TensorList self, F f) {
  for (const auto& tensor : self) {
    f(tensor);
  }
}

template <typename T, typename F>
void ForeachOpInplace(at::TensorList self, at::ArrayRef<T> other, F f) {
  for (size_t i = 0; i < self.size(); ++i) {
    f(self[i], other[i]);
  }
}

template <typename T, typename F>
void ForeachOpInplace(at::TensorList self, std::vector<T>& other, F f) {
  for (size_t i = 0; i < self.size(); ++i) {
    f(self[i], other[i]);
  }
}

#define TT_FOREACH_BINARY_ALPHA(op_name_suffix, build_list)                    \
  std::vector<at::Tensor> AtenForeach##op_name_suffix##List(                   \
      at::TensorList self, at::TensorList other, const at::Scalar& alpha) {    \
    auto promoted_alpha = PromoteScalar(alpha);                                \
    TT_KERNEL(OpName::kForeach##op_name_suffix##List, _,                       \
              (self, other, promoted_alpha), {                                 \
                (void)promoted_alpha.GetTensor();                              \
                TT_ASSIGN_OR_THROW(auto out_dtypes,                            \
                                   GetOutputDtypes(self, other));              \
                return ForeachConvertToTensor(                                 \
                    build_list(self, other, alpha, std::move(out_dtypes)));    \
              });                                                              \
  }                                                                            \
  std::vector<at::Tensor> AtenForeach##op_name_suffix##Scalar(                 \
      at::TensorList self, const at::Scalar& scalar) {                         \
    auto promoted_scalar = PromoteScalar(scalar);                              \
    TT_KERNEL(                                                                 \
        OpName::kForeach##op_name_suffix##Scalar, _, (self, promoted_scalar),  \
        {                                                                      \
          TT_ASSIGN_OR_THROW(auto out_dtypes, GetOutputDtypes(self, scalar));  \
          auto other =                                                         \
              GetOtherTensors(self.size(), promoted_scalar, *out_dtypes);      \
          return ForeachConvertToTensor(                                       \
              build_list(self, other, 1.0, std::move(out_dtypes)));            \
        });                                                                    \
  }                                                                            \
  std::vector<at::Tensor> AtenForeach##op_name_suffix##ScalarList(             \
      at::TensorList self, at::ArrayRef<at::Scalar> scalars) {                 \
    auto promoted_scalars = PromoteScalar(scalars);                            \
    TT_KERNEL(                                                                 \
        OpName::kForeach##op_name_suffix##ScalarList, _,                       \
        (self, promoted_scalars), {                                            \
          TT_ASSIGN_OR_THROW(auto out_dtypes, GetOutputDtypes(self, scalars)); \
          auto other =                                                         \
              GetOtherTensors(self.size(), promoted_scalars, *out_dtypes);     \
          return ForeachConvertToTensor(                                       \
              build_list(self, other, 1.0, std::move(out_dtypes)));            \
        });                                                                    \
  }                                                                            \
  std::vector<at::Tensor> AtenForeach##op_name_suffix##Tensor(                 \
      at::TensorList self, const at::Tensor& other, const at::Scalar& alpha) { \
    auto promoted_alpha = PromoteScalar(alpha);                                \
    TT_KERNEL(OpName::kForeach##op_name_suffix##List, _,                       \
              (self, other, promoted_alpha), {                                 \
                (void)promoted_alpha.GetTensor();                              \
                std::vector<at::Tensor> other_list(self.size(), other);        \
                TT_ASSIGN_OR_THROW(auto out_dtypes,                            \
                                   GetOutputDtypes(self, other_list));         \
                return ForeachConvertToTensor(build_list(                      \
                    self, other_list, alpha, std::move(out_dtypes)));          \
              });                                                              \
  }                                                                            \
  void AtenForeach##op_name_suffix##_List(                                     \
      at::TensorList self, at::TensorList other, const at::Scalar& alpha) {    \
    auto promoted_alpha = PromoteScalar(alpha);                                \
    TT_KERNEL(                                                                 \
        OpName::kForeach##op_name_suffix##_List, _,                            \
        (self, other, promoted_alpha), {                                       \
          (void)promoted_alpha.GetTensor();                                    \
          TT_ASSIGN_OR_THROW(auto out_dtypes, GetOutputDtypes(self));          \
          TT_THROW_IF_ERROR(ForeachAssignToTensor(                             \
              build_list(self, other, alpha, std::move(out_dtypes)), self));   \
        });                                                                    \
  }                                                                            \
  void AtenForeach##op_name_suffix##_Scalar(at::TensorList self,               \
                                            const at::Scalar& scalar) {        \
    auto promoted_scalar = PromoteScalar(scalar);                              \
    TT_KERNEL(                                                                 \
        OpName::kForeach##op_name_suffix##_Scalar, _, (self, promoted_scalar), \
        {                                                                      \
          TT_THROW_IF_ERROR(CheckInplaceScalarType(self, scalar));             \
          auto other =                                                         \
              GetOtherTensorsInplace(self.size(), promoted_scalar, self);      \
          TT_ASSIGN_OR_THROW(auto out_dtypes, GetOutputDtypes(self));          \
          TT_THROW_IF_ERROR(ForeachAssignToTensor(                             \
              build_list(self, other, 1.0, std::move(out_dtypes)), self));     \
        });                                                                    \
  }                                                                            \
  void AtenForeach##op_name_suffix##_ScalarList(                               \
      at::TensorList self, at::ArrayRef<at::Scalar> scalars) {                 \
    auto promoted_scalars = PromoteScalar(scalars);                            \
    TT_KERNEL(                                                                 \
        OpName::kForeach##op_name_suffix##_ScalarList, _,                      \
        (self, promoted_scalars), {                                            \
          TT_THROW_IF_ERROR(CheckInplaceScalarType(self, scalars));            \
          auto other =                                                         \
              GetOtherTensorsInplace(self.size(), promoted_scalars, self);     \
          TT_ASSIGN_OR_THROW(auto out_dtypes, GetOutputDtypes(self));          \
          TT_THROW_IF_ERROR(ForeachAssignToTensor(                             \
              build_list(self, other, 1.0, std::move(out_dtypes)), self));     \
        });                                                                    \
  }                                                                            \
  void AtenForeach##op_name_suffix##_Tensor(                                   \
      at::TensorList self, const at::Tensor& other, const at::Scalar& alpha) { \
    auto promoted_alpha = PromoteScalar(alpha);                                \
    TT_KERNEL(                                                                 \
        OpName::kForeach##op_name_suffix##_List, _,                            \
        (self, other, promoted_alpha), {                                       \
          (void)promoted_alpha.GetTensor();                                    \
          std::vector<at::Tensor> other_list(self.size(), other);              \
          TT_ASSIGN_OR_THROW(auto out_dtypes, GetOutputDtypes(self));          \
          TT_THROW_IF_ERROR(ForeachAssignToTensor(                             \
              build_list(self, other_list, alpha, std::move(out_dtypes)),      \
              self));                                                          \
        });                                                                    \
  }

#define TT_FOREACH_BINARY_DIV(op_name_suffix, build_list)                    \
  std::vector<at::Tensor> AtenForeach##op_name_suffix##List(                 \
      at::TensorList self, at::TensorList other) {                           \
    TT_KERNEL(OpName::kForeach##op_name_suffix##List, _, (self, other), {    \
      TT_ASSIGN_OR_THROW(                                                    \
          auto out_dtypes,                                                   \
          GetOutputDtypes(self, /*cast_integral_to_float=*/true));           \
      auto other_promoted = GetOtherTensors(other, *out_dtypes);             \
      return ForeachConvertToTensor(                                         \
          build_list(self, other_promoted, std::move(out_dtypes)));          \
    });                                                                      \
  }                                                                          \
  std::vector<at::Tensor> AtenForeach##op_name_suffix##Scalar(               \
      at::TensorList self, const at::Scalar& scalar) {                       \
    auto promoted_scalar = PromoteScalar(scalar);                            \
    TT_KERNEL(OpName::kForeach##op_name_suffix##Scalar, _,                   \
              (self, promoted_scalar), {                                     \
                TT_ASSIGN_OR_THROW(                                          \
                    auto out_dtypes,                                         \
                    GetOutputDtypes(self, /*cast_integral_to_float=*/true)); \
                auto other = GetOtherTensors(self.size(), promoted_scalar,   \
                                             *out_dtypes);                   \
                return ForeachConvertToTensor(                               \
                    build_list(self, other, std::move(out_dtypes)));         \
              });                                                            \
  }                                                                          \
  std::vector<at::Tensor> AtenForeach##op_name_suffix##ScalarList(           \
      at::TensorList self, at::ArrayRef<at::Scalar> scalars) {               \
    auto promoted_scalars = PromoteScalar(scalars);                          \
    TT_KERNEL(OpName::kForeach##op_name_suffix##ScalarList, _,               \
              (self, promoted_scalars), {                                    \
                TT_ASSIGN_OR_THROW(                                          \
                    auto out_dtypes,                                         \
                    GetOutputDtypes(self, /*cast_integral_to_float=*/true)); \
                auto other = GetOtherTensors(self.size(), promoted_scalars,  \
                                             *out_dtypes);                   \
                return ForeachConvertToTensor(                               \
                    build_list(self, other, std::move(out_dtypes)));         \
              });                                                            \
  }                                                                          \
  std::vector<at::Tensor> AtenForeach##op_name_suffix##Tensor(               \
      at::TensorList self, const at::Tensor& other) {                        \
    TT_KERNEL(OpName::kForeach##op_name_suffix##List, _, (self, other), {    \
      std::vector<at::Tensor> other_list(self.size(), other);                \
      TT_ASSIGN_OR_THROW(                                                    \
          auto out_dtypes,                                                   \
          GetOutputDtypes(self, /*cast_integral_to_float=*/true));           \
      auto other_promoted = GetOtherTensors(other_list, *out_dtypes);        \
      return ForeachConvertToTensor(                                         \
          build_list(self, other_promoted, std::move(out_dtypes)));          \
    });                                                                      \
  }                                                                          \
  void AtenForeach##op_name_suffix##_List(at::TensorList self,               \
                                          at::TensorList other) {            \
    TT_KERNEL(OpName::kForeach##op_name_suffix##_List, _, (self, other), {   \
      TT_ASSIGN_OR_THROW(auto out_dtypes, GetOutputDtypes(self));            \
      TT_THROW_IF_ERROR(ForeachAssignToTensor(                               \
          build_list(self, other, std::move(out_dtypes)), self));            \
    });                                                                      \
  }                                                                          \
  void AtenForeach##op_name_suffix##_Scalar(at::TensorList self,             \
                                            const at::Scalar& scalar) {      \
    auto promoted_scalar = PromoteScalar(scalar);                            \
    TT_KERNEL(OpName::kForeach##op_name_suffix##_Scalar, _,                  \
              (self, promoted_scalar), {                                     \
                TT_THROW_IF_ERROR(CheckInplaceScalarType(self, scalar));     \
                auto other = GetOtherTensorsInplace(self.size(),             \
                                                    promoted_scalar, self);  \
                TT_ASSIGN_OR_THROW(auto out_dtypes, GetOutputDtypes(self));  \
                TT_THROW_IF_ERROR(ForeachAssignToTensor(                     \
                    build_list(self, other, std::move(out_dtypes)), self));  \
              });                                                            \
  }                                                                          \
  void AtenForeach##op_name_suffix##_ScalarList(                             \
      at::TensorList self, at::ArrayRef<at::Scalar> scalars) {               \
    auto promoted_scalars = PromoteScalar(scalars);                          \
    TT_KERNEL(OpName::kForeach##op_name_suffix##_ScalarList, _,              \
              (self, promoted_scalars), {                                    \
                TT_THROW_IF_ERROR(CheckInplaceScalarType(self, scalars));    \
                auto other = GetOtherTensorsInplace(self.size(),             \
                                                    promoted_scalars, self); \
                TT_ASSIGN_OR_THROW(auto out_dtypes, GetOutputDtypes(self));  \
                TT_THROW_IF_ERROR(ForeachAssignToTensor(                     \
                    build_list(self, other, std::move(out_dtypes)), self));  \
              });                                                            \
  }                                                                          \
  void AtenForeach##op_name_suffix##_Tensor(at::TensorList self,             \
                                            const at::Tensor& other) {       \
    TT_KERNEL(OpName::kForeach##op_name_suffix##_List, _, (self, other), {   \
      std::vector<at::Tensor> other_list(self.size(), other);                \
      TT_ASSIGN_OR_THROW(auto out_dtypes, GetOutputDtypes(self));            \
      TT_THROW_IF_ERROR(ForeachAssignToTensor(                               \
          build_list(self, other_list, std::move(out_dtypes)), self));       \
    });                                                                      \
  }

#define TT_FOREACH_BINARY_LEGACY(op_name_suffix, build_list)                   \
  std::vector<at::Tensor> AtenForeach##op_name_suffix##List(                   \
      at::TensorList self, at::TensorList other) {                             \
    TT_KERNEL(OpName::kForeach##op_name_suffix##List, _, (self, other), {      \
      TT_ASSIGN_OR_THROW(auto out_dtypes, GetOutputDtypes(self, other));       \
      return ForeachConvertToTensor(                                           \
          build_list(self, other, std::move(out_dtypes)));                     \
    });                                                                        \
  }                                                                            \
  std::vector<at::Tensor> AtenForeach##op_name_suffix##Scalar(                 \
      at::TensorList self, const at::Scalar& scalar) {                         \
    auto promoted_scalar = PromoteScalar(scalar);                              \
    TT_KERNEL(                                                                 \
        OpName::kForeach##op_name_suffix##Scalar, _, (self, promoted_scalar),  \
        {                                                                      \
          TT_ASSIGN_OR_THROW(auto out_dtypes, GetOutputDtypes(self, scalar));  \
          auto other =                                                         \
              GetOtherTensors(self.size(), promoted_scalar, *out_dtypes);      \
          return ForeachConvertToTensor(                                       \
              build_list(self, other, std::move(out_dtypes)));                 \
        });                                                                    \
  }                                                                            \
  std::vector<at::Tensor> AtenForeach##op_name_suffix##ScalarList(             \
      at::TensorList self, at::ArrayRef<at::Scalar> scalars) {                 \
    auto promoted_scalars = PromoteScalar(scalars);                            \
    TT_KERNEL(                                                                 \
        OpName::kForeach##op_name_suffix##ScalarList, _,                       \
        (self, promoted_scalars), {                                            \
          TT_ASSIGN_OR_THROW(auto out_dtypes, GetOutputDtypes(self, scalars)); \
          auto other =                                                         \
              GetOtherTensors(self.size(), promoted_scalars, *out_dtypes);     \
          return ForeachConvertToTensor(                                       \
              build_list(self, other, std::move(out_dtypes)));                 \
        });                                                                    \
  }                                                                            \
  std::vector<at::Tensor> AtenForeach##op_name_suffix##Tensor(                 \
      at::TensorList self, const at::Tensor& other) {                          \
    TT_KERNEL(OpName::kForeach##op_name_suffix##List, _, (self, other), {      \
      std::vector<at::Tensor> other_list(self.size(), other);                  \
      TT_ASSIGN_OR_THROW(auto out_dtypes, GetOutputDtypes(self, other_list));  \
      return ForeachConvertToTensor(                                           \
          build_list(self, other_list, std::move(out_dtypes)));                \
    });                                                                        \
  }                                                                            \
  void AtenForeach##op_name_suffix##_List(at::TensorList self,                 \
                                          at::TensorList other) {              \
    TT_KERNEL(OpName::kForeach##op_name_suffix##_List, _, (self, other), {     \
      TT_ASSIGN_OR_THROW(auto out_dtypes, GetOutputDtypes(self));              \
      TT_THROW_IF_ERROR(ForeachAssignToTensor(                                 \
          build_list(self, other, std::move(out_dtypes)), self));              \
    });                                                                        \
  }                                                                            \
  void AtenForeach##op_name_suffix##_Scalar(at::TensorList self,               \
                                            const at::Scalar& scalar) {        \
    auto promoted_scalar = PromoteScalar(scalar);                              \
    TT_KERNEL(OpName::kForeach##op_name_suffix##_Scalar, _,                    \
              (self, promoted_scalar), {                                       \
                TT_THROW_IF_ERROR(CheckInplaceScalarType(self, scalar));       \
                auto other = GetOtherTensorsInplace(self.size(),               \
                                                    promoted_scalar, self);    \
                TT_ASSIGN_OR_THROW(auto out_dtypes, GetOutputDtypes(self));    \
                TT_THROW_IF_ERROR(ForeachAssignToTensor(                       \
                    build_list(self, other, std::move(out_dtypes)), self));    \
              });                                                              \
  }                                                                            \
  void AtenForeach##op_name_suffix##_ScalarList(                               \
      at::TensorList self, at::ArrayRef<at::Scalar> scalars) {                 \
    auto promoted_scalars = PromoteScalar(scalars);                            \
    TT_KERNEL(OpName::kForeach##op_name_suffix##_ScalarList, _,                \
              (self, promoted_scalars), {                                      \
                TT_THROW_IF_ERROR(CheckInplaceScalarType(self, scalars));      \
                auto other = GetOtherTensorsInplace(self.size(),               \
                                                    promoted_scalars, self);   \
                TT_ASSIGN_OR_THROW(auto out_dtypes, GetOutputDtypes(self));    \
                TT_THROW_IF_ERROR(ForeachAssignToTensor(                       \
                    build_list(self, other, std::move(out_dtypes)), self));    \
              });                                                              \
  }                                                                            \
  void AtenForeach##op_name_suffix##_Tensor(at::TensorList self,               \
                                            const at::Tensor& other) {         \
    TT_KERNEL(OpName::kForeach##op_name_suffix##_List, _, (self, other), {     \
      std::vector<at::Tensor> other_list(self.size(), other);                  \
      TT_ASSIGN_OR_THROW(auto out_dtypes, GetOutputDtypes(self));              \
      TT_THROW_IF_ERROR(ForeachAssignToTensor(                                 \
          build_list(self, other_list, std::move(out_dtypes)), self));         \
    });                                                                        \
  }

absl::StatusOr<std::vector<DeviceBufferRef>> ForeachUnaryOp(
    at::TensorList self, UniqueDtypeVec out_dtypes,
    absl::AnyInvocable<absl::StatusOr<mlir::MlirOp>(mlir::MlirOp,
                                                    mlir::ElementType) const>
        tensor_transform,
    // OpName to use for dispatching. If omitted, use the op name from the
    // active TT_KERNEL() context.
    std::optional<OpName> op_name = std::nullopt,
    // If true, cast all inputs to the corresponding output dtype before
    // applying the tensor_transform.
    bool cast_inputs = true) {
  const DtypeSpan out_dtypes_span = *out_dtypes;
  auto op_builder =
      [out_dtypes = std::move(out_dtypes), out_dtypes_span, cast_inputs,
       tensor_transform = std::move(tensor_transform)](
          absl::Span<mlir::MlirOp> inputs, mlir::MlirBuilder& builder)
      -> absl::StatusOr<mlir::SmallVector<mlir::MlirOp>> {
    mlir::SmallVector<mlir::MlirOp> results;
    results.reserve(inputs.size());
    for (int i = 0; i < inputs.size(); ++i) {
      mlir::MlirOp input = inputs[i];
      if (cast_inputs) {
        TT_ASSIGN_OR_RETURN(input, CastIfNeeded(input, out_dtypes_span[i]));
      }
      TT_ASSIGN_OR_RETURN(mlir::MlirOp result,
                          tensor_transform(input, out_dtypes_span[i]));
      results.push_back(result);
    }
    return results;
  };

  std::vector<at::Tensor> inputs(self.begin(), self.end());
  const auto out_dims_list = GetDimsList(self);
  DispatchOpOptions<torch_tpu::kDynamicSize> options = {
      .op_name = op_name,
      .out_dtypes = out_dtypes_span,
      .out_dims_list = out_dims_list,
      .op_param_cache_keys = OpParamCacheKeys::Empty()};

  TT_ASSIGN_OR_RETURN(std::vector<DeviceBufferRef> result_buffers,
                      (DispatchOp<kDynamicSize, kDynamicSize>(
                          std::move(op_builder), inputs, std::move(options))));
  return result_buffers;
}

absl::StatusOr<std::vector<DeviceBufferRef>> ForeachUnaryOp(
    at::TensorList self, UniqueDtypeVec out_dtypes,
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
  return ForeachUnaryOp(self, std::move(out_dtypes),
                        std::move(tmp_tensor_transform), op_name, cast_inputs);
}

enum class UnaryCheck { kNone, kNotIntegral, kNotComplex, kNotBool };

template <UnaryCheck check>
absl::Status PerformUnaryCheck(at::TensorList self) {
  if constexpr (check == UnaryCheck::kNotIntegral) {
    return CheckNotIntegral(self, "self");
  } else if constexpr (check == UnaryCheck::kNotComplex) {
    return CheckNotComplex(self, "self");
  } else if constexpr (check == UnaryCheck::kNotBool) {
    return CheckNotBool(self, "self");
  }
  return absl::OkStatus();
}

template <UnaryCheck functional_check = UnaryCheck::kNone,
          bool cast_integral_to_float = true,
          bool cast_complex_to_float = false, bool cast_inputs = true,
          typename BuildShlo>
std::vector<at::Tensor> ForeachUnary(at::TensorList self, OpName op_name,
                                     BuildShlo build_shlo) {
  TT_THROW_IF_ERROR(PerformUnaryCheck<functional_check>(self));
  TT_ASSIGN_OR_THROW(
      auto out_dtypes,
      GetOutputDtypes(self, cast_integral_to_float, cast_complex_to_float));
  TT_ASSIGN_OR_THROW(auto result_buffers,
                     ForeachUnaryOp(self, std::move(out_dtypes), build_shlo,
                                    op_name, cast_inputs));
  return ForeachConvertToTensor(result_buffers);
}

template <UnaryCheck inplace_check = UnaryCheck::kNotIntegral,
          bool cast_inputs = true, typename BuildShlo>
void ForeachUnaryInplace(at::TensorList self, OpName op_name,
                         BuildShlo build_shlo) {
  TT_THROW_IF_ERROR(PerformUnaryCheck<inplace_check>(self));
  TT_ASSIGN_OR_THROW(auto out_dtypes, GetOutputDtypes(self));
  TT_ASSIGN_OR_THROW(auto result_buffers,
                     ForeachUnaryOp(self, std::move(out_dtypes), build_shlo,
                                    op_name, cast_inputs));
  TT_THROW_IF_ERROR(ForeachAssignToTensor(result_buffers, self));
}

#define TT_FOREACH_UNARY_CUSTOM(op_name_suffix, build_shlo, functional_check,  \
                                cast_integral_to_float, cast_complex_to_float, \
                                functional_cast_inputs, inplace_check,         \
                                inplace_cast_inputs)                           \
  std::vector<at::Tensor> AtenForeach##op_name_suffix(at::TensorList self) {   \
    TT_KERNEL(OpName::kForeach##op_name_suffix, _, (self), {                   \
      return ForeachUnary<functional_check, cast_integral_to_float,            \
                          cast_complex_to_float, functional_cast_inputs>(      \
          self, OpName::kForeach##op_name_suffix, build_shlo);                 \
    });                                                                        \
  }                                                                            \
  void AtenForeach##op_name_suffix##_(at::TensorList self) {                   \
    TT_KERNEL(OpName::kForeach##op_name_suffix##_, _, (self), {                \
      ForeachUnaryInplace<inplace_check, inplace_cast_inputs>(                 \
          self, OpName::kForeach##op_name_suffix, build_shlo);                 \
    });                                                                        \
  }

#define TT_FOREACH_UNARY(op_name_suffix, build_shlo)                      \
  TT_FOREACH_UNARY_CUSTOM(                                                \
      op_name_suffix, build_shlo, /*functional_check=*/UnaryCheck::kNone, \
      /*cast_integral_to_float=*/true, /*cast_complex_to_float=*/false,   \
      /*functional_cast_inputs=*/true,                                    \
      /*inplace_check=*/UnaryCheck::kNotIntegral,                         \
      /*inplace_cast_inputs=*/true)

std::vector<DeviceBufferRef> ForeachAddList(at::TensorList self,
                                            at::TensorList other,
                                            const at::Scalar& alpha,
                                            UniqueDtypeVec out_dtypes) {
  // self and other are guaranteed to have the same size.
  // The error is handled by the upstream torch.
  size_t num_tensors = self.size();

  // The op builder.
  const DtypeSpan out_dtypes_span = *out_dtypes;
  std::vector<at::Tensor> inputs(self.begin(), self.end());
  inputs.insert(inputs.end(), other.begin(), other.end());
  bool alpha_is_one = (alpha.isIntegral(true) && alpha.to<int64_t>() == 1) ||
                      (alpha.isFloatingPoint() && alpha.to<double>() == 1.0);
  auto param_keys = OpParamCacheKeys::Empty();
  // We create different Shlo based on whether alpha is one or not.
  // Alpha is not multiplied if equal to 1.0.
  TT_THROW_IF_ERROR(param_keys.SetParam("alpha_is_one", alpha_is_one));
  if (!alpha_is_one) {
    for (int i = 0; i < num_tensors; ++i) {
      at::ScalarType scalar_type = ConvertTo<at::ScalarType>((*out_dtypes)[i]);
      TT_ASSIGN_OR_THROW(at::Tensor alpha_tensor,
                         MakeTensor(alpha, scalar_type));
      inputs.push_back(alpha_tensor);
    }
  }

  auto op_builder = [alpha_is_one, num_tensors,
                     out_dtypes = std::move(out_dtypes),
                     out_dtypes_span](absl::Span<mlir::MlirOp> inputs,
                                      mlir::MlirBuilder& builder)
      -> absl::StatusOr<mlir::SmallVector<mlir::MlirOp>> {
    absl::Span<mlir::MlirOp> self_ops = inputs.subspan(0, num_tensors);
    absl::Span<mlir::MlirOp> other_ops =
        inputs.subspan(num_tensors, num_tensors);

    // If alpha is 1.0, do a simple addition without multiplying by alpha.
    if (alpha_is_one) {
      return BuildForeachShlo(self_ops, other_ops, out_dtypes_span,
                              mlir::stablehlo::Add, builder);
    }

    absl::Span<mlir::MlirOp> alpha_ops =
        inputs.subspan(2 * num_tensors, 3 * num_tensors);
    TT_ASSIGN_OR_RETURN(auto new_other_ops,
                        BuildForeachShlo(other_ops, alpha_ops, out_dtypes_span,
                                         mlir::stablehlo::Mul, builder));
    return BuildForeachShlo(self_ops, absl::MakeSpan(new_other_ops),
                            out_dtypes_span, mlir::stablehlo::Add, builder);
  };

  // Dispatch the op and prepare results.
  const auto out_dims_list = GetDimsList(self);
  DispatchOpOptions<kDynamicSize> options = {
      // Share the same OpName for all ForeachAddList() calls, as the underlying
      // Shlo is the same.
      .op_name = OpName::kForeachAddList,
      .out_dtypes = out_dtypes_span,
      .out_dims_list = absl::MakeConstSpan(out_dims_list),
      .op_param_cache_keys = std::move(param_keys),
  };
  TT_ASSIGN_OR_THROW(auto result_buffers,
                     (DispatchOp<kDynamicSize, kDynamicSize>(
                         std::move(op_builder), inputs, std::move(options))));
  return result_buffers;
}

std::vector<DeviceBufferRef> ForeachMulList(at::TensorList self,
                                            at::TensorList other,
                                            UniqueDtypeVec out_dtypes) {
  const DtypeSpan out_dtypes_span = *out_dtypes;
  // self and other are guaranteed to have the same size.
  // The error is handled by the upstream torch.
  const size_t num_tensors = self.size();
  auto op_builder = [num_tensors, out_dtypes = std::move(out_dtypes),
                     out_dtypes_span](absl::Span<mlir::MlirOp> inputs,
                                      mlir::MlirBuilder& builder)
      -> absl::StatusOr<mlir::SmallVector<mlir::MlirOp>> {
    absl::Span<mlir::MlirOp> self_ops = inputs.subspan(0, num_tensors);
    absl::Span<mlir::MlirOp> other_ops =
        inputs.subspan(num_tensors, num_tensors);
    return BuildForeachShlo(self_ops, other_ops, out_dtypes_span,
                            mlir::stablehlo::Mul, builder);
  };

  std::vector<at::Tensor> inputs(self.begin(), self.end());
  inputs.insert(inputs.end(), other.begin(), other.end());
  const auto out_dims_list = GetDimsList(self);
  DispatchOpOptions<kDynamicSize> options = {
      // Share the same OpName for all ForeachMulList() calls, as the underlying
      // Shlo is the same.
      .op_name = OpName::kForeachMulList,
      .out_dtypes = out_dtypes_span,
      .out_dims_list = out_dims_list,
      .op_param_cache_keys = OpParamCacheKeys::Empty(),
  };

  // Dispatch the op and prepare results.
  TT_ASSIGN_OR_THROW(auto result_buffers,
                     (DispatchOp<kDynamicSize, kDynamicSize>(
                         std::move(op_builder), inputs, std::move(options))));
  return result_buffers;
}

std::vector<DeviceBufferRef> ForeachSubList(at::TensorList self,
                                            at::TensorList other,
                                            const at::Scalar& alpha,
                                            UniqueDtypeVec out_dtypes) {
  return ForeachAddList(self, other, -alpha, std::move(out_dtypes));
}

std::vector<DeviceBufferRef> ForeachDivList(at::TensorList self,
                                            at::TensorList other,
                                            UniqueDtypeVec out_dtypes) {
  const DtypeSpan out_dtypes_span = *out_dtypes;
  // self and other are guaranteed to have the same size.
  // The error is handled by the upstream torch.
  const size_t num_tensors = self.size();
  auto op_builder = [num_tensors, out_dtypes = std::move(out_dtypes),
                     out_dtypes_span](absl::Span<mlir::MlirOp> inputs,
                                      mlir::MlirBuilder& builder)
      -> absl::StatusOr<mlir::SmallVector<mlir::MlirOp>> {
    absl::Span<mlir::MlirOp> self_ops = inputs.subspan(0, num_tensors);
    absl::Span<mlir::MlirOp> other_ops =
        inputs.subspan(num_tensors, num_tensors);
    return BuildForeachShlo(self_ops, other_ops, out_dtypes_span,
                            mlir::stablehlo::Div, builder);
  };

  std::vector<at::Tensor> inputs(self.begin(), self.end());
  inputs.insert(inputs.end(), other.begin(), other.end());
  const auto out_dims_list = GetDimsList(self);
  DispatchOpOptions<kDynamicSize> options = {
      // Share the same OpName for all ForeachDivList() calls, as the underlying
      // Shlo is the same.
      .op_name = OpName::kForeachDivList,
      .out_dtypes = out_dtypes_span,
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

TT_FOREACH_UNARY_CUSTOM(Abs, BuildAbsShlo,
                        /*functional_check=*/UnaryCheck::kNone,
                        /*cast_integral_to_float=*/false,
                        /*cast_complex_to_float=*/true,
                        /*functional_cast_inputs=*/false,
                        /*inplace_check=*/UnaryCheck::kNotComplex,
                        /*inplace_cast_inputs=*/false)
TT_FOREACH_UNARY(Acos, BuildAcosShlo)
TT_FOREACH_UNARY(Asin, BuildAsinShlo)
TT_FOREACH_UNARY(Atan, BuildAtanShlo)
TT_FOREACH_UNARY(Ceil, BuildCeilShlo)
TT_FOREACH_UNARY(Cos, BuildCosShlo)
TT_FOREACH_UNARY(Cosh, BuildCoshShlo)
TT_FOREACH_UNARY(Erf, BuildErfShlo)
TT_FOREACH_UNARY(Erfc, BuildErfcShlo)
TT_FOREACH_UNARY(Exp, BuildExpShlo)
TT_FOREACH_UNARY(Expm1, BuildExpm1Shlo)
TT_FOREACH_UNARY(Floor, BuildFloorShlo)
TT_FOREACH_UNARY_CUSTOM(Frac, BuildFracShlo,
                        /*functional_check=*/UnaryCheck::kNotIntegral,
                        /*cast_integral_to_float=*/true,
                        /*cast_complex_to_float=*/false,
                        /*functional_cast_inputs=*/true,
                        /*inplace_check=*/UnaryCheck::kNotIntegral,
                        /*inplace_cast_inputs=*/true)
TT_FOREACH_UNARY(Lgamma, BuildLgammaShlo)
TT_FOREACH_UNARY(Log, BuildLogShlo)
TT_FOREACH_UNARY(Log10, BuildLog10Shlo)
TT_FOREACH_UNARY(Log1p, BuildLog1pShlo)
TT_FOREACH_UNARY(Log2, BuildLog2Shlo)
TT_FOREACH_UNARY_CUSTOM(Neg, BuildNegShlo,
                        /*functional_check=*/UnaryCheck::kNone,
                        /*cast_integral_to_float=*/false,
                        /*cast_complex_to_float=*/false,
                        /*functional_cast_inputs=*/true,
                        /*inplace_check=*/UnaryCheck::kNone,
                        /*inplace_cast_inputs=*/true)
TT_FOREACH_UNARY(Reciprocal, BuildReciprocalShlo)

absl::StatusOr<std::vector<DeviceBufferRef>> ForeachRound(at::TensorList self) {
  // BuildRoundShlo has a different signature from the other unary transforms.
  auto tensor_transform = [](mlir::MlirOp input, mlir::ElementType) {
    return BuildRoundShlo(input, 0);
  };
  TT_RETURN_IF_ERROR(CheckNotBool(self, "self"));
  TT_ASSIGN_OR_RETURN(auto out_dtypes, GetOutputDtypes(self));
  return ForeachUnaryOp(self, std::move(out_dtypes),
                        std::move(tensor_transform), OpName::kForeachRound);
}

std::vector<at::Tensor> AtenForeachRound(at::TensorList self) {
  TT_KERNEL(OpName::kForeachRound, _, (self), {
    TT_ASSIGN_OR_THROW(auto result_buffers, ForeachRound(self));
    return ForeachConvertToTensor(result_buffers);
  });
}

void AtenForeachRound_(at::TensorList self) {
  TT_KERNEL(OpName::kForeachRound_, _, (self), {
    TT_ASSIGN_OR_THROW(auto result_buffers, ForeachRound(self));
    TT_THROW_IF_ERROR(ForeachAssignToTensor(result_buffers, self));
  });
}

TT_FOREACH_UNARY(Rsqrt, BuildRsqrtShlo)
TT_FOREACH_UNARY(Sigmoid, BuildSigmoidShlo)
TT_FOREACH_UNARY_CUSTOM(Sign, BuildSignShlo,
                        /*functional_check=*/UnaryCheck::kNotComplex,
                        /*cast_integral_to_float=*/false,
                        /*cast_complex_to_float=*/false,
                        /*functional_cast_inputs=*/true,
                        /*inplace_check=*/UnaryCheck::kNotComplex,
                        /*inplace_cast_inputs=*/true)
TT_FOREACH_UNARY(Sin, BuildSinShlo)
TT_FOREACH_UNARY(Sinh, BuildSinhShlo)
TT_FOREACH_UNARY(Sqrt, BuildSqrtShlo)
TT_FOREACH_UNARY(Tan, BuildTanShlo)
TT_FOREACH_UNARY(Tanh, BuildTanhShlo)
TT_FOREACH_UNARY_CUSTOM(Trunc, BuildTruncShlo,
                        /*functional_check=*/UnaryCheck::kNotBool,
                        /*cast_integral_to_float=*/false,
                        /*cast_complex_to_float=*/false,
                        /*functional_cast_inputs=*/true,
                        /*inplace_check=*/UnaryCheck::kNotBool,
                        /*inplace_cast_inputs=*/true)

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

TT_FOREACH_BINARY_ALPHA(Add, ForeachAddList)
TT_FOREACH_BINARY_LEGACY(Mul, ForeachMulList)

std::vector<at::Tensor> AtenForeachAddcdivScalar(at::TensorList self,
                                                 at::TensorList tensor1,
                                                 at::TensorList tensor2,
                                                 const at::Scalar& value) {
  TT_KERNEL(
      OpName::kForeachAddcdivScalar, _,
      (self, tensor1, tensor2, IgnoreInCacheKey(value, "Legacy usage")), {
        // _foreach_div supports two integral tensors, but _foreach_addcdiv
        // doesn't.
        TT_THROW_IF_ERROR(
            CheckPairwiseAddcdivAtLeastOneNotIntegral(tensor1, tensor2));
        std::vector<at::Tensor> quotient = AtenForeachDivList(tensor1, tensor2);
        std::vector<at::Tensor> product = AtenForeachMulScalar(quotient, value);
        return AtenForeachAddList(self, product, 1.0);
      });
}

std::vector<at::Tensor> AtenForeachAddcdivScalarList(
    at::TensorList self, at::TensorList tensor1, at::TensorList tensor2,
    at::ArrayRef<at::Scalar> scalars) {
  TT_KERNEL(
      OpName::kForeachAddcdivScalarList, _,
      (self, tensor1, tensor2, IgnoreInCacheKey(scalars, "Legacy usage")), {
        // _foreach_div supports two integral tensors, but
        // _foreach_addcdiv doesn't.
        TT_THROW_IF_ERROR(
            CheckPairwiseAddcdivAtLeastOneNotIntegral(tensor1, tensor2));
        std::vector<at::Tensor> quotient = AtenForeachDivList(tensor1, tensor2);
        std::vector<at::Tensor> product =
            AtenForeachMulScalarList(quotient, scalars);
        return AtenForeachAddList(self, product, 1.0);
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
        std::vector<at::Tensor> quotient = AtenForeachDivList(tensor1, tensor2);
        std::vector<at::Tensor> product =
            AtenForeachMulTensor(quotient, scalars);
        return AtenForeachAddList(self, product, 1.0);
      });
}

void AtenForeachAddcdiv_Scalar(at::TensorList self, at::TensorList tensor1,
                               at::TensorList tensor2,
                               const at::Scalar& value) {
  TT_KERNEL(
      OpName::kForeachAddcdiv_Scalar, _,
      (self, tensor1, tensor2, IgnoreInCacheKey(value, "Legacy usage")), {
        // _foreach_div supports two integral tensors, but _foreach_addcdiv
        // doesn't.
        TT_THROW_IF_ERROR(
            CheckPairwiseAddcdivAtLeastOneNotIntegral(tensor1, tensor2));
        std::vector<at::Tensor> quotient = AtenForeachDivList(tensor1, tensor2);
        std::vector<at::Tensor> product = AtenForeachMulScalar(quotient, value);
        AtenForeachAdd_List(self, product, 1.0);
      });
}

void AtenForeachAddcdiv_ScalarList(at::TensorList self, at::TensorList tensor1,
                                   at::TensorList tensor2,
                                   at::ArrayRef<at::Scalar> scalars) {
  TT_KERNEL(
      OpName::kForeachAddcdiv_ScalarList, _,
      (self, tensor1, tensor2, IgnoreInCacheKey(scalars, "Legacy usage")), {
        // _foreach_div supports two integral tensors, but
        // _foreach_addcdiv doesn't.
        TT_THROW_IF_ERROR(
            CheckPairwiseAddcdivAtLeastOneNotIntegral(tensor1, tensor2));
        std::vector<at::Tensor> quotient = AtenForeachDivList(tensor1, tensor2);
        std::vector<at::Tensor> product =
            AtenForeachMulScalarList(quotient, scalars);
        AtenForeachAdd_List(self, product, 1.0);
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
        std::vector<at::Tensor> quotient = AtenForeachDivList(tensor1, tensor2);
        std::vector<at::Tensor> product =
            AtenForeachMulTensor(quotient, scalars);
        AtenForeachAdd_List(self, product, 1.0);
      });
}

std::vector<at::Tensor> AtenForeachAddcmulScalar(at::TensorList self,
                                                 at::TensorList tensor1,
                                                 at::TensorList tensor2,
                                                 const at::Scalar& value) {
  TT_KERNEL(OpName::kForeachAddcmulScalar, _,
            (self, tensor1, tensor2, IgnoreInCacheKey(value, "Legacy usage")), {
              // _foreach_mul and _foreach_add supports bool tensors, but
              // _foreach_addcmul doesn't.
              TT_THROW_IF_ERROR(CheckNotBool(self, /* arg_name= */ "self"));
              std::vector<at::Tensor> product =
                  AtenForeachMulList(tensor1, tensor2);
              std::vector<at::Tensor> scaled_product =
                  AtenForeachMulScalar(product, value);
              return AtenForeachAddList(self, scaled_product, 1);
            });
}

std::vector<at::Tensor> AtenForeachAddcmulScalarList(
    at::TensorList self, at::TensorList tensor1, at::TensorList tensor2,
    at::ArrayRef<at::Scalar> scalars) {
  TT_KERNEL(
      OpName::kForeachAddcmulScalarList, _,
      (self, tensor1, tensor2, IgnoreInCacheKey(scalars, "Legacy usage")), {
        // _foreach_mul and _foreach_add supports bool tensors, but
        // _foreach_addcmul doesn't.
        TT_THROW_IF_ERROR(CheckNotBool(self, /* arg_name= */ "self"));
        std::vector<at::Tensor> product = AtenForeachMulList(tensor1, tensor2);
        std::vector<at::Tensor> scaled_product =
            AtenForeachMulScalarList(product, scalars);
        return AtenForeachAddList(self, scaled_product, 1);
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
        std::vector<at::Tensor> product = AtenForeachMulList(tensor1, tensor2);
        std::vector<at::Tensor> scaled_product =
            AtenForeachMulTensor(product, scalars);
        return AtenForeachAddList(self, scaled_product, 1);
      });
}

void AtenForeachAddcmul_Scalar(at::TensorList self, at::TensorList tensor1,
                               at::TensorList tensor2,
                               const at::Scalar& value) {
  TT_KERNEL(OpName::kForeachAddcmul_Scalar, _,
            (self, tensor1, tensor2, IgnoreInCacheKey(value, "Legacy usage")), {
              // _foreach_mul and _foreach_add supports bool tensors, but
              // _foreach_addcmul doesn't.
              TT_THROW_IF_ERROR(CheckNotBool(self, /* arg_name= */ "self"));
              std::vector<at::Tensor> product =
                  AtenForeachMulList(tensor1, tensor2);
              std::vector<at::Tensor> scaled_product =
                  AtenForeachMulScalar(product, value);
              AtenForeachAdd_List(self, scaled_product, 1);
            });
}

void AtenForeachAddcmul_ScalarList(at::TensorList self, at::TensorList tensor1,
                                   at::TensorList tensor2,
                                   at::ArrayRef<at::Scalar> scalars) {
  TT_KERNEL(
      OpName::kForeachAddcmul_ScalarList, _,
      (self, tensor1, tensor2, IgnoreInCacheKey(scalars, "Legacy usage")), {
        // _foreach_mul and _foreach_add supports bool tensors, but
        // _foreach_addcmul doesn't.
        TT_THROW_IF_ERROR(CheckNotBool(self, /* arg_name= */ "self"));
        std::vector<at::Tensor> product = AtenForeachMulList(tensor1, tensor2);
        std::vector<at::Tensor> scaled_product =
            AtenForeachMulScalarList(product, scalars);
        AtenForeachAdd_List(self, scaled_product, 1);
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
        std::vector<at::Tensor> product = AtenForeachMulList(tensor1, tensor2);
        std::vector<at::Tensor> scaled_product =
            AtenForeachMulTensor(product, scalars);
        AtenForeachAdd_List(self, scaled_product, 1);
      });
}

std::vector<at::Tensor> AtenForeachLerpList(at::TensorList self,
                                            at::TensorList other,
                                            at::TensorList weight) {
  TT_KERNEL(OpName::kForeachLerpList, _, (self, other, weight), {
    auto diff = AtenForeachSubList(other, self, 1.0);
    auto weighted_diff = AtenForeachMulList(weight, diff);
    return AtenForeachAddList(self, weighted_diff, 1.0);
  });
}

std::vector<at::Tensor> AtenForeachLerpScalar(at::TensorList self,
                                              at::TensorList other,
                                              const at::Scalar& weight) {
  TT_KERNEL(OpName::kForeachLerpScalar, _,
            (self, other, IgnoreInCacheKey(weight, "Legacy usage")), {
              auto diff = AtenForeachSubList(other, self, 1.0);
              auto weighted_diff = AtenForeachMulScalar(diff, weight);
              return AtenForeachAddList(self, weighted_diff, 1.0);
            });
}

std::vector<at::Tensor> AtenForeachLerpScalarList(
    at::TensorList self, at::TensorList other,
    at::ArrayRef<at::Scalar> scalars) {
  TT_KERNEL(OpName::kForeachLerpScalarList, _,
            (self, other, IgnoreInCacheKey(scalars, "Legacy usage")), {
              auto diff = AtenForeachSubList(other, self, 1.0);
              auto weighted_diff = AtenForeachMulScalarList(diff, scalars);
              return AtenForeachAddList(self, weighted_diff, 1.0);
            });
}

void AtenForeachLerp_List(at::TensorList self, at::TensorList other,
                          at::TensorList weight) {
  TT_KERNEL(OpName::kForeachLerp_List, _, (self, other, weight), {
    auto diff = AtenForeachSubList(other, self, 1.0);
    auto weighted_diff = AtenForeachMulList(diff, weight);
    AtenForeachAdd_List(self, weighted_diff, 1.0);
  });
}

void AtenForeachLerp_Scalar(at::TensorList self, at::TensorList other,
                            const at::Scalar& weight) {
  TT_KERNEL(OpName::kForeachLerp_Scalar, _,
            (self, other, IgnoreInCacheKey(weight, "Legacy usage")), {
              auto diff = AtenForeachSubList(other, self, 1.0);
              auto weighted_diff = AtenForeachMulScalar(diff, weight);
              AtenForeachAdd_List(self, weighted_diff, 1.0);
            });
}

void AtenForeachLerp_ScalarList(at::TensorList self, at::TensorList other,
                                at::ArrayRef<at::Scalar> scalars) {
  TT_KERNEL(OpName::kForeachLerp_ScalarList, _,
            (self, other, IgnoreInCacheKey(scalars, "Legacy usage")), {
              auto diff = AtenForeachSubList(other, self, 1.0);
              auto weighted_diff = AtenForeachMulScalarList(diff, scalars);
              AtenForeachAdd_List(self, weighted_diff, 1.0);
            });
}

std::vector<at::Tensor> AtenForeachSubList(at::TensorList self,
                                           at::TensorList other,
                                           const at::Scalar& alpha) {
  auto promoted_alpha = PromoteScalar(alpha);
  TT_KERNEL(OpName::kForeachSubList, _, (self, other, promoted_alpha), {
    (void)promoted_alpha.GetTensor();
    // _foreach_add supports bool, but _foreach_sub does not.
    TT_THROW_IF_ERROR(CheckNotBool(alpha, /* arg_name= */ "alpha"));
    TT_THROW_IF_ERROR(CheckNotBool(self, /* arg_name= */ "self"));
    TT_THROW_IF_ERROR(CheckNotBool(other, /* arg_name= */ "other"));
    TT_ASSIGN_OR_THROW(auto out_dtypes, GetOutputDtypes(self, other));
    return ForeachConvertToTensor(
        ForeachSubList(self, other, alpha, std::move(out_dtypes)));
  });
}

void AtenForeachSub_List(at::TensorList self, at::TensorList other,
                         const at::Scalar& alpha) {
  auto promoted_alpha = PromoteScalar(alpha);
  TT_KERNEL(OpName::kForeachSub_List, _, (self, other, promoted_alpha), {
    (void)promoted_alpha.GetTensor();
    // _foreach_add supports bool, but _foreach_sub does not.
    TT_THROW_IF_ERROR(CheckNotBool(alpha, /* arg_name= */ "alpha"));
    TT_THROW_IF_ERROR(CheckNotBool(self, /* arg_name= */ "self"));
    TT_THROW_IF_ERROR(CheckNotBool(other, /* arg_name= */ "other"));
    TT_ASSIGN_OR_THROW(auto out_dtypes, GetOutputDtypes(self));
    TT_THROW_IF_ERROR(ForeachAssignToTensor(
        ForeachSubList(self, other, alpha, std::move(out_dtypes)), self));
  });
}

std::vector<at::Tensor> AtenForeachSubScalar(at::TensorList self,
                                             const at::Scalar& scalar) {
  auto promoted_scalar = PromoteScalar(scalar);
  TT_KERNEL(OpName::kForeachSubScalar, _, (self, promoted_scalar), {
    // _foreach_add supports bool, but _foreach_sub does not.
    TT_THROW_IF_ERROR(CheckNotBool(scalar, /* arg_name= */ "scalar"));
    TT_THROW_IF_ERROR(CheckNotBool(self, /* arg_name= */ "self"));
    TT_ASSIGN_OR_THROW(auto out_dtypes, GetOutputDtypes(self, scalar));
    auto other = GetOtherTensors(self.size(), promoted_scalar, *out_dtypes);
    return ForeachConvertToTensor(
        ForeachSubList(self, other, 1.0, std::move(out_dtypes)));
  });
}

void AtenForeachSub_Scalar(at::TensorList self, const at::Scalar& scalar) {
  auto promoted_scalar = PromoteScalar(scalar);
  TT_KERNEL(OpName::kForeachSub_Scalar, _, (self, promoted_scalar), {
    // _foreach_add supports bool, but _foreach_sub does not.
    TT_THROW_IF_ERROR(CheckNotBool(scalar, /* arg_name= */ "scalar"));
    TT_THROW_IF_ERROR(CheckNotBool(self, /* arg_name= */ "self"));
    TT_THROW_IF_ERROR(CheckInplaceScalarType(self, scalar));
    auto other = GetOtherTensorsInplace(self.size(), promoted_scalar, self);
    TT_ASSIGN_OR_THROW(auto out_dtypes, GetOutputDtypes(self));
    TT_THROW_IF_ERROR(ForeachAssignToTensor(
        ForeachSubList(self, other, 1.0, std::move(out_dtypes)), self));
  });
}

std::vector<at::Tensor> AtenForeachSubScalarList(
    at::TensorList self, at::ArrayRef<at::Scalar> scalars) {
  auto promoted_scalars = PromoteScalar(scalars);
  TT_KERNEL(OpName::kForeachSubScalarList, _, (self, promoted_scalars), {
    // _foreach_add supports bool, but _foreach_sub does not.
    TT_THROW_IF_ERROR(CheckNotBool(scalars, /* arg_name= */ "scalars"));
    TT_THROW_IF_ERROR(CheckNotBool(self, /* arg_name= */ "self"));
    TT_ASSIGN_OR_THROW(auto out_dtypes, GetOutputDtypes(self, scalars));
    auto other = GetOtherTensors(self.size(), promoted_scalars, *out_dtypes);
    return ForeachConvertToTensor(
        ForeachSubList(self, other, 1.0, std::move(out_dtypes)));
  });
}

void AtenForeachSub_ScalarList(at::TensorList self,
                               at::ArrayRef<at::Scalar> scalars) {
  auto promoted_scalars = PromoteScalar(scalars);
  TT_KERNEL(OpName::kForeachSub_ScalarList, _, (self, promoted_scalars), {
    // _foreach_add supports bool, but _foreach_sub does not.
    TT_THROW_IF_ERROR(CheckNotBool(scalars, /* arg_name= */ "scalars"));
    TT_THROW_IF_ERROR(CheckNotBool(self, /* arg_name= */ "self"));
    TT_THROW_IF_ERROR(CheckInplaceScalarType(self, scalars));
    auto other = GetOtherTensorsInplace(self.size(), promoted_scalars, self);
    TT_ASSIGN_OR_THROW(auto out_dtypes, GetOutputDtypes(self));
    TT_THROW_IF_ERROR(ForeachAssignToTensor(
        ForeachSubList(self, other, 1.0, std::move(out_dtypes)), self));
  });
}

std::vector<at::Tensor> AtenForeachSubTensor(at::TensorList self,
                                             const at::Tensor& other,
                                             const at::Scalar& alpha) {
  auto promoted_alpha = PromoteScalar(alpha);
  TT_KERNEL(OpName::kForeachSubList, _, (self, other, promoted_alpha), {
    (void)promoted_alpha.GetTensor();
    // _foreach_add supports bool, but _foreach_sub does not.
    TT_THROW_IF_ERROR(CheckNotBool(alpha, /* arg_name= */ "alpha"));
    TT_THROW_IF_ERROR(CheckNotBool(self, /* arg_name= */ "self"));
    TT_THROW_IF_ERROR(CheckNotBool(other, /* arg_name= */ "other"));
    std::vector<at::Tensor> other_list(self.size(), other);
    TT_ASSIGN_OR_THROW(auto out_dtypes, GetOutputDtypes(self, other_list));
    return ForeachConvertToTensor(
        ForeachSubList(self, other_list, alpha, std::move(out_dtypes)));
  });
}

void AtenForeachSub_Tensor(at::TensorList self, const at::Tensor& other,
                           const at::Scalar& alpha) {
  auto promoted_alpha = PromoteScalar(alpha);
  TT_KERNEL(OpName::kForeachSub_List, _, (self, other, promoted_alpha), {
    (void)promoted_alpha.GetTensor();
    // _foreach_add supports bool, but _foreach_sub does not.
    TT_THROW_IF_ERROR(CheckNotBool(alpha, /* arg_name= */ "alpha"));
    TT_THROW_IF_ERROR(CheckNotBool(self, /* arg_name= */ "self"));
    TT_THROW_IF_ERROR(CheckNotBool(other, /* arg_name= */ "other"));
    std::vector<at::Tensor> other_list(self.size(), other);
    TT_ASSIGN_OR_THROW(auto out_dtypes, GetOutputDtypes(self));
    TT_THROW_IF_ERROR(ForeachAssignToTensor(
        ForeachSubList(self, other_list, alpha, std::move(out_dtypes)), self));
  });
}

TT_FOREACH_BINARY_DIV(Div, ForeachDivList)

absl::StatusOr<at::Tensor> AtenClampMax(const at::Tensor& self,
                                        const at::Scalar& alpha) {
  at::Tensor out =
      MakeEmptyTensor(self.sizes(), self.scalar_type(), self.device());
  AtenClampMaxOut(self, alpha, out);
  return out;
}

absl::StatusOr<at::Tensor> AtenClampMax(const at::Tensor& self,
                                        const at::Tensor& other) {
  at::Tensor out =
      MakeEmptyTensor(self.sizes(), self.scalar_type(), self.device());
  AtenClampMaxTensorOut(self, other, out);
  return out;
}

std::vector<at::Tensor> AtenForeachClampMaxScalar(at::TensorList self,
                                                  const at::Scalar& scalar) {
  TT_KERNEL(OpName::kForeachClampMaxScalar, _,
            (self, IgnoreInCacheKey(scalar, "Legacy usage")), {
              return ForeachOpFunctional(
                  self, [&](const at::Tensor& t) -> at::Tensor {
                    TT_ASSIGN_OR_THROW(auto out, AtenClampMax(t, scalar));
                    return out;
                  });
            });
}

void AtenForeachClampMax_Scalar(at::TensorList self, const at::Scalar& scalar) {
  TT_KERNEL(OpName::kForeachClampMax_Scalar, _,
            (self, IgnoreInCacheKey(scalar, "Legacy usage")), {
              ForeachOpInplace(self, [&](const at::Tensor& t) {
                AtenClampMaxOut(t, scalar, const_cast<at::Tensor&>(t));
              });
            });
}

std::vector<at::Tensor> AtenForeachClampMaxList(at::TensorList self,
                                                at::TensorList other) {
  TT_KERNEL(OpName::kForeachClampMaxList, _, (self, other), {
    return ForeachOpFunctional(
        self, other,
        [&](const at::Tensor& s, const at::Tensor& o) -> at::Tensor {
          TT_ASSIGN_OR_THROW(auto out, AtenClampMax(s, o));
          return out;
        });
  });
}

void AtenForeachClampMax_List(at::TensorList self, at::TensorList other) {
  TT_KERNEL(OpName::kForeachClampMax_List, _, (self, other), {
    ForeachOpInplace(self, other,
                     [&](const at::Tensor& s, const at::Tensor& o) {
                       AtenClampMaxTensorOut(s, o, const_cast<at::Tensor&>(s));
                     });
  });
}

std::vector<at::Tensor> AtenForeachClampMaxScalarList(
    at::TensorList self, at::ArrayRef<at::Scalar> scalars) {
  TT_KERNEL(OpName::kForeachClampMaxScalarList, _,
            (self, IgnoreInCacheKey(scalars, "Legacy usage")), {
              return ForeachOpFunctional(
                  self, scalars,
                  [&](const at::Tensor& t, const at::Scalar& s) -> at::Tensor {
                    TT_ASSIGN_OR_THROW(auto out, AtenClampMax(t, s));
                    return out;
                  });
            });
}

void AtenForeachClampMax_ScalarList(at::TensorList self,
                                    at::ArrayRef<at::Scalar> scalars) {
  TT_KERNEL(OpName::kForeachClampMax_ScalarList, _,
            (self, IgnoreInCacheKey(scalars, "Legacy usage")), {
              ForeachOpInplace(
                  self, scalars, [&](const at::Tensor& t, const at::Scalar& s) {
                    AtenClampMaxOut(t, s, const_cast<at::Tensor&>(t));
                  });
            });
}

absl::StatusOr<at::Tensor> AtenClampMin(const at::Tensor& self,
                                        const at::Scalar& alpha) {
  at::Tensor out =
      MakeEmptyTensor(self.sizes(), self.scalar_type(), self.device());
  AtenClampMinOut(self, alpha, out);
  return out;
}

absl::StatusOr<at::Tensor> AtenClampMin(const at::Tensor& self,
                                        const at::Tensor& other) {
  at::Tensor out =
      MakeEmptyTensor(self.sizes(), self.scalar_type(), self.device());
  AtenClampMinTensorOut(self, other, out);
  return out;
}

std::vector<at::Tensor> AtenForeachClampMinScalar(at::TensorList self,
                                                  const at::Scalar& scalar) {
  TT_KERNEL(OpName::kForeachClampMinScalar, _,
            (self, IgnoreInCacheKey(scalar, "Legacy usage")), {
              return ForeachOpFunctional(
                  self, [&](const at::Tensor& t) -> at::Tensor {
                    TT_ASSIGN_OR_THROW(auto out, AtenClampMin(t, scalar));
                    return out;
                  });
            });
}

void AtenForeachClampMin_Scalar(at::TensorList self, const at::Scalar& scalar) {
  TT_KERNEL(OpName::kForeachClampMin_Scalar, _,
            (self, IgnoreInCacheKey(scalar, "Legacy usage")), {
              ForeachOpInplace(self, [&](const at::Tensor& t) {
                AtenClampMinOut(t, scalar, const_cast<at::Tensor&>(t));
              });
            });
}

std::vector<at::Tensor> AtenForeachClampMinList(at::TensorList self,
                                                at::TensorList other) {
  TT_KERNEL(OpName::kForeachClampMinList, _, (self, other), {
    return ForeachOpFunctional(
        self, other,
        [&](const at::Tensor& s, const at::Tensor& o) -> at::Tensor {
          TT_ASSIGN_OR_THROW(auto out, AtenClampMin(s, o));
          return out;
        });
  });
}

void AtenForeachClampMin_List(at::TensorList self, at::TensorList other) {
  TT_KERNEL(OpName::kForeachClampMin_List, _, (self, other), {
    ForeachOpInplace(self, other,
                     [&](const at::Tensor& s, const at::Tensor& o) {
                       AtenClampMinTensorOut(s, o, const_cast<at::Tensor&>(s));
                     });
  });
}

std::vector<at::Tensor> AtenForeachClampMinScalarList(
    at::TensorList self, at::ArrayRef<at::Scalar> scalars) {
  TT_KERNEL(OpName::kForeachClampMinScalarList, _,
            (self, IgnoreInCacheKey(scalars, "Legacy usage")), {
              return ForeachOpFunctional(
                  self, scalars,
                  [&](const at::Tensor& t, const at::Scalar& s) -> at::Tensor {
                    TT_ASSIGN_OR_THROW(auto out, AtenClampMin(t, s));
                    return out;
                  });
            });
}

void AtenForeachClampMin_ScalarList(at::TensorList self,
                                    at::ArrayRef<at::Scalar> scalars) {
  TT_KERNEL(OpName::kForeachClampMin_ScalarList, _,
            (self, IgnoreInCacheKey(scalars, "Legacy usage")), {
              ForeachOpInplace(
                  self, scalars, [&](const at::Tensor& t, const at::Scalar& s) {
                    AtenClampMinOut(t, s, const_cast<at::Tensor&>(t));
                  });
            });
}

void AtenForeachCopy_(at::TensorList self, at::TensorList src,
                      bool non_blocking) {
  TT_KERNEL(OpName::kForeachCopy_, _,
            (self, src, IgnoreInCacheKey(non_blocking, "Legacy usage")), {
              ForeachOpInplace(
                  self, src, [&](const at::Tensor& s, const at::Tensor& o) {
                    AtenCopy_(const_cast<at::Tensor&>(s), o, non_blocking);
                  });
            });
}

std::vector<at::Tensor> AtenForeachNormScalar(
    at::TensorList self, const at::Scalar& ord,
    const std::optional<c10::ScalarType> dtype) {
  TT_KERNEL(
      OpName::kForeachNormScalar, _,
      (self, IgnoreInCacheKey(ord, "Legacy usage"),
       IgnoreInCacheKey(dtype, "Legacy usage")),
      {
        TT_THROW_IF_ERROR(CheckNotIntegral(self, /* arg_name= */ "self"));
        return ForeachOpFunctional(
            self, [&](const at::Tensor& tensor) -> at::Tensor {
              auto out_dtype =
                  at::toRealValueType(dtype.value_or(tensor.scalar_type()));
              auto out = MakeEmptyTensor({}, out_dtype, tensor.device());
              // If the tensor is empty, return zero scalar.
              // If ord < 0, to avoid inf caused by pow(0, ord), check for
              // 0s in the tensor and return zero directly, which is the
              // correct result.
              if (tensor.numel() == 0 ||
                  (ord.to<double>() < 0 && tensor.eq(0).any().item<bool>())) {
                out.fill_(0);
                return out;
              }
              AtenLinalgVectorNormOut(tensor.to(out_dtype), ord,
                                      /*dim=*/{},
                                      /*keepdim=*/false, dtype, out);
              return out;
            });
      });
}

std::vector<at::Tensor> AtenForeachMax(at::TensorList self) {
  TT_KERNEL(OpName::kForeachMax, _, (self), {
    return ForeachOpFunctional(
        self, [](const at::Tensor& t) -> at::Tensor { return AtenMax(t); });
  });
}

absl::StatusOr<at::Tensor> AtenMaximum(const at::Tensor& self,
                                       const at::Tensor& other) {
  at::Tensor out =
      MakeEmptyTensor(self.sizes(), self.scalar_type(), self.device());
  AtenMaximumOut(self, other, out);
  return out;
}

std::vector<at::Tensor> AtenForeachMaximumScalar(at::TensorList self,
                                                 const at::Scalar& scalar) {
  auto promoted_scalar = PromoteScalar(scalar);
  TT_ASSIGN_OR_THROW(auto scalar_tensor, promoted_scalar.GetTensor());
  TT_KERNEL(OpName::kForeachMaximumScalar, _,
            (self, IgnoreInCacheKey(scalar, "Legacy usage")), {
              return ForeachOpFunctional(
                  self, [&](const at::Tensor& t) -> at::Tensor {
                    TT_ASSIGN_OR_THROW(auto out, AtenMaximum(t, scalar_tensor));
                    return out;
                  });
            });
}

void AtenForeachMaximum_Scalar(at::TensorList self, const at::Scalar& scalar) {
  auto promoted_scalar = PromoteScalar(scalar);
  TT_ASSIGN_OR_THROW(auto scalar_tensor, promoted_scalar.GetTensor());
  TT_KERNEL(OpName::kForeachMaximum_Scalar, _,
            (self, IgnoreInCacheKey(scalar, "Legacy usage")), {
              ForeachOpInplace(self, [&](const at::Tensor& t) {
                AtenMaximumOut(t, scalar_tensor, const_cast<at::Tensor&>(t));
              });
            });
}

std::vector<at::Tensor> AtenForeachMaximumList(at::TensorList self,
                                               at::TensorList other) {
  TT_KERNEL(OpName::kForeachMaximumList, _, (self, other), {
    return ForeachOpFunctional(
        self, other,
        [&](const at::Tensor& s, const at::Tensor& o) -> at::Tensor {
          TT_ASSIGN_OR_THROW(auto out, AtenMaximum(s, o));
          return out;
        });
  });
}

void AtenForeachMaximum_List(at::TensorList self, at::TensorList other) {
  TT_KERNEL(OpName::kForeachMaximum_List, _, (self, other), {
    ForeachOpInplace(self, other,
                     [&](const at::Tensor& s, const at::Tensor& o) {
                       AtenMaximumOut(s, o, const_cast<at::Tensor&>(s));
                     });
  });
}

std::vector<at::Tensor> AtenForeachMaximumScalarList(
    at::TensorList self, at::ArrayRef<at::Scalar> scalars) {
  auto promoted_scalars = PromoteScalar(scalars);
  std::vector<at::Tensor> scalar_tensors;
  scalar_tensors.reserve(scalars.size());
  for (auto& ps : promoted_scalars) {
    TT_ASSIGN_OR_THROW(auto st, ps.GetTensor());
    scalar_tensors.push_back(st);
  }
  TT_KERNEL(OpName::kForeachMaximumScalarList, _,
            (self, IgnoreInCacheKey(scalars, "Legacy usage")), {
              return ForeachOpFunctional(
                  self, scalar_tensors,
                  [&](const at::Tensor& t, const at::Tensor& st) -> at::Tensor {
                    TT_ASSIGN_OR_THROW(auto out, AtenMaximum(t, st));
                    return out;
                  });
            });
}

void AtenForeachMaximum_ScalarList(at::TensorList self,
                                   at::ArrayRef<at::Scalar> scalars) {
  auto promoted_scalars = PromoteScalar(scalars);
  std::vector<at::Tensor> scalar_tensors;
  scalar_tensors.reserve(scalars.size());
  for (auto& ps : promoted_scalars) {
    TT_ASSIGN_OR_THROW(auto st, ps.GetTensor());
    scalar_tensors.push_back(st);
  }
  TT_KERNEL(OpName::kForeachMaximum_ScalarList, _,
            (self, IgnoreInCacheKey(scalars, "Legacy usage")), {
              ForeachOpInplace(self, scalar_tensors,
                               [&](const at::Tensor& t, const at::Tensor& st) {
                                 AtenMaximumOut(t, st,
                                                const_cast<at::Tensor&>(t));
                               });
            });
}

absl::StatusOr<at::Tensor> AtenMinimum(const at::Tensor& self,
                                       const at::Tensor& other) {
  at::Tensor out =
      MakeEmptyTensor(self.sizes(), self.scalar_type(), self.device());
  AtenMinimumOut(self, other, out);
  return out;
}

std::vector<at::Tensor> AtenForeachMinimumScalar(at::TensorList self,
                                                 const at::Scalar& scalar) {
  auto promoted_scalar = PromoteScalar(scalar);
  TT_ASSIGN_OR_THROW(auto scalar_tensor, promoted_scalar.GetTensor());
  TT_KERNEL(OpName::kForeachMinimumScalar, _,
            (self, IgnoreInCacheKey(scalar, "Legacy usage")), {
              return ForeachOpFunctional(
                  self, [&](const at::Tensor& t) -> at::Tensor {
                    TT_ASSIGN_OR_THROW(auto out, AtenMinimum(t, scalar_tensor));
                    return out;
                  });
            });
}

void AtenForeachMinimum_Scalar(at::TensorList self, const at::Scalar& scalar) {
  auto promoted_scalar = PromoteScalar(scalar);
  TT_ASSIGN_OR_THROW(auto scalar_tensor, promoted_scalar.GetTensor());
  TT_KERNEL(OpName::kForeachMinimum_Scalar, _,
            (self, IgnoreInCacheKey(scalar, "Legacy usage")), {
              ForeachOpInplace(self, [&](const at::Tensor& t) {
                AtenMinimumOut(t, scalar_tensor, const_cast<at::Tensor&>(t));
              });
            });
}

std::vector<at::Tensor> AtenForeachMinimumList(at::TensorList self,
                                               at::TensorList other) {
  TT_KERNEL(OpName::kForeachMinimumList, _, (self, other), {
    return ForeachOpFunctional(
        self, other,
        [&](const at::Tensor& s, const at::Tensor& o) -> at::Tensor {
          TT_ASSIGN_OR_THROW(auto out, AtenMinimum(s, o));
          return out;
        });
  });
}

void AtenForeachMinimum_List(at::TensorList self, at::TensorList other) {
  TT_KERNEL(OpName::kForeachMinimum_List, _, (self, other), {
    ForeachOpInplace(self, other,
                     [&](const at::Tensor& s, const at::Tensor& o) {
                       AtenMinimumOut(s, o, const_cast<at::Tensor&>(s));
                     });
  });
}

std::vector<at::Tensor> AtenForeachMinimumScalarList(
    at::TensorList self, at::ArrayRef<at::Scalar> scalars) {
  auto promoted_scalars = PromoteScalar(scalars);
  std::vector<at::Tensor> scalar_tensors;
  scalar_tensors.reserve(scalars.size());
  for (auto& ps : promoted_scalars) {
    TT_ASSIGN_OR_THROW(auto st, ps.GetTensor());
    scalar_tensors.push_back(st);
  }
  TT_KERNEL(OpName::kForeachMinimumScalarList, _,
            (self, IgnoreInCacheKey(scalars, "Legacy usage")), {
              return ForeachOpFunctional(
                  self, scalar_tensors,
                  [&](const at::Tensor& t, const at::Tensor& st) -> at::Tensor {
                    TT_ASSIGN_OR_THROW(auto out, AtenMinimum(t, st));
                    return out;
                  });
            });
}

void AtenForeachMinimum_ScalarList(at::TensorList self,
                                   at::ArrayRef<at::Scalar> scalars) {
  auto promoted_scalars = PromoteScalar(scalars);
  std::vector<at::Tensor> scalar_tensors;
  scalar_tensors.reserve(scalars.size());
  for (auto& ps : promoted_scalars) {
    TT_ASSIGN_OR_THROW(auto st, ps.GetTensor());
    scalar_tensors.push_back(st);
  }
  TT_KERNEL(OpName::kForeachMinimum_ScalarList, _,
            (self, IgnoreInCacheKey(scalars, "Legacy usage")), {
              ForeachOpInplace(self, scalar_tensors,
                               [&](const at::Tensor& t, const at::Tensor& st) {
                                 AtenMinimumOut(t, st,
                                                const_cast<at::Tensor&>(t));
                               });
            });
}

absl::StatusOr<at::Tensor> AtenPow(const at::Tensor& self,
                                   const at::Tensor& exponent) {
  at::Tensor out =
      MakeEmptyTensor(self.sizes(), self.scalar_type(), self.device());
  AtenPowTensorTensorOut(self, exponent, out);
  return out;
}

absl::StatusOr<at::Tensor> AtenPow(const at::Tensor& self,
                                   const at::Scalar& exponent) {
  at::Tensor out =
      MakeEmptyTensor(self.sizes(), self.scalar_type(), self.device());
  AtenPowTensorScalarOut(self, exponent, out);
  return out;
}

absl::StatusOr<at::Tensor> AtenPow(const at::Scalar& self,
                                   const at::Tensor& exponent) {
  at::Tensor out = MakeEmptyTensor(exponent.sizes(), exponent.scalar_type(),
                                   exponent.device());
  AtenPowScalarOut(self, exponent, out);
  return out;
}

std::vector<at::Tensor> AtenForeachPowList(at::TensorList self,
                                           at::TensorList exponent) {
  TT_KERNEL(OpName::kForeachPowList, _, (self, exponent), {
    return ForeachOpFunctional(
        self, exponent,
        [&](const at::Tensor& s, const at::Tensor& e) -> at::Tensor {
          TT_ASSIGN_OR_THROW(auto out, AtenPow(s, e));
          return out;
        });
  });
}

void AtenForeachPow_List(at::TensorList self, at::TensorList exponent) {
  TT_KERNEL(OpName::kForeachPow_List, _, (self, exponent), {
    ForeachOpInplace(self, exponent,
                     [&](const at::Tensor& s, const at::Tensor& e) {
                       AtenPowTensorTensorOut(s, e, const_cast<at::Tensor&>(s));
                     });
  });
}

std::vector<at::Tensor> AtenForeachPowScalar(at::TensorList self,
                                             const at::Scalar& exponent) {
  TT_KERNEL(OpName::kForeachPowScalar, _,
            (self, IgnoreInCacheKey(exponent, "Legacy usage")), {
              return ForeachOpFunctional(
                  self, [&](const at::Tensor& t) -> at::Tensor {
                    TT_ASSIGN_OR_THROW(auto out, AtenPow(t, exponent));
                    return out;
                  });
            });
}

void AtenForeachPow_Scalar(at::TensorList self, const at::Scalar& exponent) {
  TT_KERNEL(OpName::kForeachPow_Scalar, _,
            (self, IgnoreInCacheKey(exponent, "Legacy usage")), {
              ForeachOpInplace(self, [&](const at::Tensor& t) {
                AtenPowTensorScalarOut(t, exponent, const_cast<at::Tensor&>(t));
              });
            });
}

std::vector<at::Tensor> AtenForeachPowScalarList(
    at::TensorList self, at::ArrayRef<at::Scalar> exponent) {
  TT_KERNEL(OpName::kForeachPowScalarList, _,
            (self, IgnoreInCacheKey(exponent, "Legacy usage")), {
              return ForeachOpFunctional(
                  self, exponent,
                  [&](const at::Tensor& t, const at::Scalar& e) -> at::Tensor {
                    TT_ASSIGN_OR_THROW(auto out, AtenPow(t, e));
                    return out;
                  });
            });
}

void AtenForeachPow_ScalarList(at::TensorList self,
                               at::ArrayRef<at::Scalar> scalars) {
  TT_KERNEL(OpName::kForeachPow_ScalarList, _,
            (self, IgnoreInCacheKey(scalars, "Legacy usage")), {
              ForeachOpInplace(
                  self, scalars, [&](const at::Tensor& t, const at::Scalar& e) {
                    AtenPowTensorScalarOut(t, e, const_cast<at::Tensor&>(t));
                  });
            });
}

std::vector<at::Tensor> AtenForeachPowScalarAndTensor(const at::Scalar& self,
                                                      at::TensorList exponent) {
  TT_KERNEL(OpName::kForeachPowScalarAndTensor, _,
            (IgnoreInCacheKey(self, "Legacy usage"), exponent), {
              return ForeachOpFunctional(
                  exponent, [&](const at::Tensor& e) -> at::Tensor {
                    TT_ASSIGN_OR_THROW(auto out, AtenPow(self, e));
                    return out;
                  });
            });
}

}  // namespace torch_tpu
