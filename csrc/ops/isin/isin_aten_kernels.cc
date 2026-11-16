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

#include "csrc/ops/isin/isin_aten_kernels.h"

#include <cstdint>
#include <utility>

#include "ATen/core/ATen_fwd.h"
#include "ATen/ops/promote_types.h"
#include "absl/log/absl_log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "csrc/common/aten_utils.h"
#include "csrc/common/cache_key.h"
#include "csrc/common/dimension_types.h"
#include "csrc/common/dtype.h"
#include "csrc/common/error_utils.h"
#include "csrc/common/fixed_size_span.h"
#include "csrc/eager/op_dispatcher.h"
#include "csrc/ops/macros/kernel.h"
#include "csrc/ops/op_builder_utils.h"
#include "csrc/ops/op_names.h"
#include "mlir/IR/BuiltinTypes.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"
#include "torch/headeronly/core/ScalarType.h"

namespace torch_tpu {
namespace {

// Specifies whether elements are assumed to be unique to optimize search.
enum class IsUnique {
  kNo,
  kYes,
};

// Specifies whether matches should be inverted in the final predicate output.
enum class IsInverted {
  kNo,
  kYes,
};

absl::StatusOr<mlir::MlirOp> BuildIsInShlo(mlir::MlirOp elements,
                                           mlir::MlirOp test_elements,
                                           IsUnique uniqueness,
                                           IsInverted inversion) {
  ABSL_VLOG(3) << "BuildIsInShlo elements=" << elements.ToString()
               << " test_elements=" << test_elements.ToString()
               << " uniqueness="
               << (uniqueness == IsUnique::kYes ? "true" : "false")
               << " inversion="
               << (inversion == IsInverted::kYes ? "true" : "false");
  auto& builder = elements.getBuilder();

  // TODO: for now we ignore assume_unique. In principle, taking that into
  // account can speed up the computation.

  // The algorithm below is based on the SHLO expansion of jnp.isin, for
  // instance:
  //
  // def foo():
  //   elements = jnp.array([[1, 2.2], [3, 4]])
  //   test_elements = jnp.array([2, 3, 7])
  //   return jnp.isin(elements, test_elements, assume_unique=False,
  //   invert=False)
  //
  // print(jax.jit(foo).lower()._lowering.stablehlo())

  const mlir::RankedTensorType elements_type = GetTensorTypeOrDie(elements);
  const mlir::RankedTensorType test_elements_type =
      GetTensorTypeOrDie(test_elements);

  TT_ASSIGN_OR_RETURN(const int64_t num_elements,
                      NumElements(elements_type.getShape()));
  TT_ASSIGN_OR_RETURN(const int64_t num_test_elements,
                      NumElements(test_elements_type.getShape()));

  auto elements_flat = mlir::stablehlo::Reshape(elements, {num_elements});
  auto test_elements_flat =
      mlir::stablehlo::Reshape(test_elements, {num_test_elements});

  TT_ASSIGN_OR_RETURN(auto elements_bcast,
                      Broadcast(elements_flat, {num_elements, 1}, {0}));
  TT_ASSIGN_OR_RETURN(
      auto test_elements_bcast,
      Broadcast(test_elements_flat, {1, num_test_elements}, {1}));

  TT_ASSIGN_OR_RETURN(
      elements_bcast,
      Broadcast(elements_bcast, {num_elements, num_test_elements}, {0, 1}));
  TT_ASSIGN_OR_RETURN(test_elements_bcast,
                      Broadcast(test_elements_bcast,
                                {num_elements, num_test_elements}, {0, 1}));

  auto cmp = mlir::stablehlo::Compare(elements_bcast, test_elements_bcast,
                                      mlir::stablehlo::ComparisonDirection::EQ);
  auto init = MakeScalarConstant(builder, false, mlir::ElementType::PRED);

  const mlir::RankedTensorType cmp_type = GetTensorTypeOrDie(cmp);

  auto reduce_builder =
      [dtype = cmp_type.getElementType()](mlir::RegionBuilder& rb) {
        mlir::stablehlo::buildReduceBody<mlir::stablehlo::OrOp>(
            dtype, rb.getRegion(), rb.getOpBuilder());
      };

  auto res =
      mlir::stablehlo::Reduce(builder, cmp, init, reduce_builder, {1})[0];

  if (inversion == IsInverted::kYes) {
    res = mlir::stablehlo::Not(res);
  }

  return mlir::stablehlo::Reshape(res, elements_type.getShape());
}

IsUnique ToIsUnique(bool assume_unique) {
  return assume_unique ? IsUnique::kYes : IsUnique::kNo;
}

IsInverted ToIsInverted(bool invert) {
  return invert ? IsInverted::kYes : IsInverted::kNo;
}

absl::Status AtenIsInHelper(const at::Tensor& elements,
                            const at::Tensor& test_elements,
                            IsUnique uniqueness, IsInverted inversion,
                            at::Tensor& out, OpParamCacheKeys param_keys) {
  TT_RETURN_IF_ERROR(ValidateOutDtype(out, at::ScalarType::Bool,
                                      /*allow_out_dtype_cast=*/false));
  ABSL_VLOG(3) << "AtenIsInHelper uniqueness="
               << (uniqueness == IsUnique::kYes ? "true" : "false")
               << " inversion="
               << (inversion == IsInverted::kYes ? "true" : "false");

  const at::ScalarType computation_scalar_type =
      at::promote_types(elements.scalar_type(), test_elements.scalar_type());
  TT_ASSIGN_OR_RETURN(const auto computation_dtype,
                      ConvertTo<mlir::ElementType>(computation_scalar_type));

  auto op_builder = [uniqueness,
                     inversion](FixedSizeSpan<mlir::MlirOp, 2> inputs) {
    auto& [elements, test_elements] = inputs;
    return BuildIsInShlo(elements, test_elements, uniqueness, inversion);
  };

  const absl::Span<const int64_t> output_dims = elements.sizes();
  const mlir::ElementType output_dtype =
      mlir::ElementType::PRED;  // The output is always Boolean.

  return DispatchOpOut<2>(std::move(op_builder), {elements, test_elements}, out,
                          {.out_dtype = output_dtype,
                           .out_dims = output_dims,
                           .computation_dtype = computation_dtype,
                           .op_param_cache_keys = std::move(param_keys)});
}

}  // namespace

at::Tensor& AtenIsInScalarTensorOut(const at::Scalar& element,
                                    const at::Tensor& test_elements,
                                    bool assume_unique, bool invert,
                                    at::Tensor& out) {
  PromotedScalar promoted_element = PromoteScalar(element);
  TT_KERNEL(OpName::kIsInScalarTensorOut, param_keys,
            (promoted_element, test_elements, assume_unique, invert, out), {
              TT_ASSIGN_OR_THROW(at::Tensor element_tensor,
                                 promoted_element.GetTensor());
              TT_THROW_IF_ERROR(AtenIsInHelper(
                  element_tensor, test_elements, ToIsUnique(assume_unique),
                  ToIsInverted(invert), out, std::move(param_keys)));
              return out;
            });
}

at::Tensor& AtenIsInTensorTensorOut(const at::Tensor& elements,
                                    const at::Tensor& test_elements,
                                    bool assume_unique, bool invert,
                                    at::Tensor& out) {
  TT_KERNEL(OpName::kIsInTensorTensorOut, param_keys,
            (elements, test_elements, assume_unique, invert, out), {
              TT_THROW_IF_ERROR(AtenIsInHelper(
                  elements, test_elements, ToIsUnique(assume_unique),
                  ToIsInverted(invert), out, std::move(param_keys)));
              return out;
            });
}

at::Tensor& AtenIsInTensorScalarOut(const at::Tensor& elements,
                                    const at::Scalar& test_element,
                                    bool assume_unique, bool invert,
                                    at::Tensor& out) {
  PromotedScalar promoted_test_element = PromoteScalar(test_element);
  TT_KERNEL(OpName::kIsInTensorScalarOut, param_keys,
            (elements, promoted_test_element, assume_unique, invert, out), {
              TT_ASSIGN_OR_THROW(at::Tensor test_element_tensor,
                                 promoted_test_element.GetTensor());
              TT_THROW_IF_ERROR(AtenIsInHelper(
                  elements, test_element_tensor, ToIsUnique(assume_unique),
                  ToIsInverted(invert), out, std::move(param_keys)));
              return out;
            });
}

}  // namespace torch_tpu
