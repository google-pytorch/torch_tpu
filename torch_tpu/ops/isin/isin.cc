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

#include "torch_tpu/ops/isin/isin.h"

#include <cstdint>

#include "absl/log/absl_log.h"
#include "absl/status/statusor.h"
#include "mlir/IR/BuiltinTypes.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/ops/op_builder_utils.h"

namespace torch_tpu {

absl::StatusOr<mlir::MlirOp> BuildIsInShlo(mlir::MlirOp elements,
                                           mlir::MlirOp test_elements,
                                           bool assume_unique, bool invert) {
  ABSL_VLOG(1) << "BuildIsInShlo elements=" << elements.ToString()
               << " test_elements=" << test_elements.ToString()
               << " assume_unique=" << assume_unique << " invert=" << invert;
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

  if (invert) {
    res = mlir::stablehlo::Not(res);
  }

  return mlir::stablehlo::Reshape(res, elements_type.getShape());
}

}  // namespace torch_tpu
