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
#include "csrc/ops/isin/isin.h"
#include "csrc/ops/macros/kernel.h"
#include "csrc/ops/op_names.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "torch/headeronly/core/ScalarType.h"

namespace torch_tpu {
namespace {

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
