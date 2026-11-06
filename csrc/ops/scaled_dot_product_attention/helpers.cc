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

#include "csrc/ops/scaled_dot_product_attention/helpers.h"

#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>
#include <vector>

#include "ATen/ExpandUtils.h"
#include "ATen/core/ATen_fwd.h"
#include "ATen/core/TensorBase.h"
#include "absl/algorithm/container.h"
#include "absl/log/absl_check.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "c10/util/ArrayRef.h"
#include "csrc/common/dimension_types.h"
#include "csrc/common/utils.h"
#include "csrc/ops/op_builder_utils.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Types.h"
#include "mlir/Support/LLVM.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"

namespace torch_tpu {

namespace {

// Batch dim (un)flattening is a valid reshape reassociation as long as at most
// one non-1 batch dim exists when a batch dim is dynamic, which
// PrepareAttentionLogits enforces, so failures here are internal errors.
mlir::MlirOp ReshapeBatchDimsOrDie(mlir::MlirOp mlir_op,
                                   absl::Span<const int64_t> shape) {
  absl::StatusOr<mlir::MlirOp> reshaped =
      ReshapeToStaticDimensions(mlir_op, CopyIntVector(shape));
  ABSL_CHECK_OK(reshaped.status());  // CRASH_OK
  return *reshaped;
}

}  // namespace

mlir::MlirOp flatten_batch_dims(mlir::MlirOp mlir_op, int batch_size,
                                int num_batch_dims) {
  const Dimensions shape = GetStaticShape(mlir_op);
  int rank = shape.size();
  int num_non_batch_dims = rank - num_batch_dims;
  Dimensions new_dims(num_non_batch_dims + 1);
  new_dims[0] = batch_size;
  for (int i = 0; i < num_non_batch_dims; i++)
    new_dims[i + 1] = shape[i + num_batch_dims];

  return ReshapeBatchDimsOrDie(mlir_op, new_dims);
}

mlir::MlirOp unflatten_batch_dims(mlir::MlirOp mlir_op,
                                  absl::Span<const int64_t> shape) {
  return ReshapeBatchDimsOrDie(mlir_op, shape);
}

mlir::MlirOp unflatten_batch_dims(mlir::MlirOp mlir_op,
                                  mlir::MlirOp mlir_op_with_target_dims) {
  return ReshapeBatchDimsOrDie(mlir_op,
                               GetStaticShape(mlir_op_with_target_dims));
}

int get_batch_size(absl::Span<const int64_t> shape) {
  ABSL_CHECK(shape.size() >= 3)  // CRASH_OK
      << "Shape must have at least 3 dimensions";
  int batch_size = 1;
  for (int i = 0; i < shape.size() - 3; i++) {
    batch_size *= shape[i];
  }
  return batch_size;
}

mlir::MlirOp GetScaleDefaulted(mlir::MlirBuilder& builder,
                               std::optional<double> maybe_scale,
                               int64_t head_dim, mlir::Type element_type) {
  double scale = maybe_scale.value_or(1.0 / std::sqrt(head_dim));

  if (element_type == nullptr) {
    element_type = builder.getOpBuilder().getF32Type();
  }
  return MakeScalarConstant(builder, scale, element_type);
}

Strides EmptyLikeStrides(const at::TensorBase& tensor) {
  if (tensor.is_non_overlapping_and_dense()) {
    return CopyIntVector(tensor.strides());
  }
  // `at::infer_dense_strides` returns a `std::vector`.
  const std::vector<int64_t> strides =  // INT_VEC_OK
      at::infer_dense_strides(tensor.sizes(), tensor.strides());
  return CopyIntVector(strides);
}

Strides MatchingLayoutStrides(const at::TensorBase& query,
                              absl::Span<const int64_t> result_sizes) {
  const at::IntArrayRef query_strides = query.strides();
  ABSL_DCHECK_EQ(query_strides.size(), result_sizes.size());
  if (absl::c_equal(query.sizes(), result_sizes)) {
    return EmptyLikeStrides(query);
  }

  // Fill the dims from innermost to outermost in ascending order of query
  // stride. Zero-stride (expanded) dims go last, i.e. outermost, and the stable
  // sort makes the lower dim innermost on ties.
  const auto fill_key = [&query_strides](int dim) {
    return query_strides[dim] == 0 ? std::numeric_limits<int64_t>::max()
                                   : query_strides[dim];
  };
  std::vector<int> fill_order(result_sizes.size());
  absl::c_iota(fill_order, 0);
  absl::c_stable_sort(fill_order, [&fill_key](int a, int b) {
    return fill_key(a) < fill_key(b);
  });

  Strides result_strides(result_sizes.size());
  int64_t stride = 1;
  for (const int dim : fill_order) {
    result_strides[dim] = stride;
    stride *= result_sizes[dim];
  }
  return result_strides;
}

}  // namespace torch_tpu
