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

#include "csrc/ops/isin/isin.h"

#include <cstdint>

#include "absl/log/absl_log.h"
#include "absl/status/statusor.h"
#include "csrc/common/error_utils.h"
#include "csrc/ops/op_builder_utils.h"
#include "mlir/IR/BuiltinTypes.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"

namespace torch_tpu {

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
  const int64_t elements_rank = elements_type.getRank();

  // Unsqueeze elements to add a trailing dimension of size 1.
  TT_ASSIGN_OR_RETURN(const mlir::MlirOp elements_unsqueeze,
                      Unsqueeze(elements, elements_rank));

  // Flatten test_elements into a 1D tensor.
  const mlir::MlirOp test_elements_flat = Flatten(test_elements);

  // Broadcast elements_unsqueeze and test_elements_flat together.
  TT_ASSIGN_OR_RETURN(
      (auto [elements_bcast, test_elements_bcast]),
      ApplyBroadcastIfNeeded(elements_unsqueeze, test_elements_flat));

  auto cmp = mlir::stablehlo::Compare(elements_bcast, test_elements_bcast,
                                      mlir::stablehlo::ComparisonDirection::EQ);
  auto init = MakeScalarConstant(builder, false, mlir::ElementType::PRED);

  const mlir::RankedTensorType cmp_type = GetTensorTypeOrDie(cmp);

  auto reduce_builder =
      [dtype = cmp_type.getElementType()](mlir::RegionBuilder& rb) {
        mlir::stablehlo::buildReduceBody<mlir::stablehlo::OrOp>(
            dtype, rb.getRegion(), rb.getOpBuilder());
      };

  auto res = mlir::stablehlo::Reduce(builder, {cmp}, {init}, reduce_builder,
                                     /*dimensions=*/{elements_rank})[0];

  if (inversion == IsInverted::kYes) {
    res = mlir::stablehlo::Not(res);
  }

  return res;
}

}  // namespace torch_tpu
