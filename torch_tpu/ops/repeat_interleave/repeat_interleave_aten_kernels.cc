// Copyright 2026 Google LLC
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

#include "torch_tpu/ops/repeat_interleave/repeat_interleave_aten_kernels.h"

#include <cstdint>
#include <optional>
#include <utility>

#include "ATen/core/TensorBody.h"
#include "absl/status/statusor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Types.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"
#include "torch_tpu/common/dimension_types.h"
#include "torch_tpu/common/dtype.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/common/fixed_size_span.h"
#include "torch_tpu/common/utils.h"
#include "torch_tpu/eager/op_dispatcher.h"
#include "torch_tpu/eager/tensor_to_buffer.h"
#include "torch_tpu/ops/cumsum/cumsum.h"
#include "torch_tpu/ops/index_select/index_select.h"
#include "torch_tpu/ops/macros/kernel.h"
#include "torch_tpu/ops/op_builder_utils.h"
#include "torch_tpu/ops/op_names.h"

namespace torch_tpu {

namespace {

// Lowers repeat_interleave to StableHLO using cumsum, broadcasting compare, and
// reduction.
//
// Example: self=[A, B, C], repeats=[2, 1, 3], output_size=6
//  1. cumsum_exclusive = cumsum(repeats) - repeats = [2,3,6] - [2,1,3] =
//  [0,2,3]
//  2. Compare iota(6) with cumsum_exclusive via broadcasting GE:
//       iota_bcst[6,1] >= cumsum_ex_bcst[1,3]
//       [[0],[1],[2],[3],[4],[5]] >= [0,2,3]
//       => comp[6,3] = [[T,F,F], [T,F,F], [T,T,F], [T,T,T], [T,T,T], [T,T,T]]
//  3. Reduce sum along rows to get cumulative counts:
//       sum_indices = [1, 1, 2, 3, 3, 3]
//  4. Subtract 1 for 0-based indices:
//       indices = sum_indices - 1 = [0, 0, 1, 2, 2, 2]
//  5. Select: self[indices] = [A, A, B, C, C, C]
absl::StatusOr<mlir::MlirOp> BuildRepeatInterleaveShlo(
    mlir::MlirOp self, mlir::MlirOp repeats, std::optional<int64_t> dim,
    int64_t output_size) {
  auto& builder = self.getBuilder();
  auto& op_builder = builder.getOpBuilder();
  auto i64_type = op_builder.getI64Type();

  mlir::MlirOp self_op = self;
  mlir::MlirOp repeats_op = repeats;
  int64_t dim_val = dim.value_or(0);

  if (!dim.has_value()) {
    self_op = Flatten(self);
    repeats_op = Flatten(repeats);
  }

  const mlir::RankedTensorType repeats_type = GetTensorTypeOrDie(repeats_op);
  const int64_t r_size = repeats_type.getDimSize(0);

  // Convert repeats to i64 if not already
  repeats_op = mlir::stablehlo::ConvertElementType(repeats_op, i64_type);

  // 1. Compute cumsum of repeats
  TT_ASSIGN_OR_RETURN(auto cumsum,
                      BuildCumsumShlo(0, std::nullopt, repeats_op));

  // 2. Compute cumsum_exclusive = cumsum - repeats
  auto cumsum_exclusive = mlir::stablehlo::Subtract(cumsum, repeats_op);

  // 3. Generate indices of size output_size (M)
  // indices = (iota(M)[:, None] >= cumsum_exclusive[None, :]).sum(dim=-1) - 1
  auto iota_type =
      mlir::makeTensorType(builder.getContext(), {output_size}, i64_type);
  auto iota = mlir::stablehlo::Iota(builder, iota_type, 0);

  auto iota_reshaped = mlir::stablehlo::Reshape(iota, {output_size, 1});
  auto cumsum_ex_reshaped =
      mlir::stablehlo::Reshape(cumsum_exclusive, {1, r_size});

  // TODO: This broadcasting compare-all approach has O(M*N) memory
  // overhead. Refactor to use searchsorted later for better scaling.
  TT_ASSIGN_OR_RETURN(auto iota_bcst,
                      BroadcastIfNeeded(iota_reshaped, {output_size, r_size}));
  TT_ASSIGN_OR_RETURN(
      auto cumsum_bcst,
      BroadcastIfNeeded(cumsum_ex_reshaped, {output_size, r_size}));

  auto comp = mlir::stablehlo::Compare(
      iota_bcst, cumsum_bcst, mlir::stablehlo::ComparisonDirection::GE);
  auto comp_i64 = mlir::stablehlo::ConvertElementType(comp, i64_type);

  auto zero_init = MakeScalarConstant(builder, 0, i64_type);
  auto sum_indices = mlir::stablehlo::Reduce(
      builder, {comp_i64}, {zero_init},
      [](mlir::RegionBuilder& rb) {
        auto& body_builder = rb.getOpBuilder();
        mlir::stablehlo::buildReduceBody<mlir::stablehlo::AddOp>(
            body_builder.getI64Type(), rb.getRegion(), body_builder);
      },
      {1})[0];

  auto one_cst = MakeConstantLike(sum_indices, 1LL);
  auto indices = mlir::stablehlo::Subtract(sum_indices, one_cst);

  // 4. Select indices along dim_val
  return BuildIndexSelectShlo(self_op, dim_val, indices);
}
}  // namespace

at::Tensor AtenRepeatInterleave(const at::Tensor& self,
                                const at::Tensor& repeats,
                                std::optional<int64_t> dim,
                                std::optional<int64_t> output_size) {
  TT_KERNEL(
      OpName::kRepeatInterleaveSelfTensor, param_keys,
      (self, repeats, dim, output_size), {
        at::Tensor resolved_repeats = repeats;
        if (repeats.numel() == 1) {
          int64_t dim_size = dim.has_value() ? self.size(*dim) : self.numel();
          resolved_repeats = repeats.expand({dim_size});
        }

        int64_t resolved_output_size;
        if (output_size.has_value()) {
          resolved_output_size = *output_size;
        } else {
          // Host sync to get the resolved output size when it is not provided
          // explicitly.
          //
          // StableHLO compilation requires knowing the output shape statically
          // (or its bounds) at compile time. Since the output size of
          // repeat_interleave depends on the *values* of the `repeats` tensor,
          // and those values are on the device, we must perform a synchronous
          // host copy (`.cpu().item<int64_t>()`) to compute `repeats.sum()` on
          // the host.
          //
          // WARNING: This causes a blocking stall on the GPU/TPU pipeline,
          // degrading eager-mode performance. To avoid this overhead, users are
          // strongly encouraged to pass `output_size` explicitly (which
          // bypasses this block).
          resolved_output_size = resolved_repeats.sum().cpu().item<int64_t>();
        }

        std::optional<int64_t> resolved_dim = std::nullopt;
        if (dim.has_value()) {
          TT_ASSIGN_OR_THROW(int64_t wrapped_dim,
                             SafeWrapDim(*dim, self.dim()));
          resolved_dim = wrapped_dim;
        }

        TT_ASSIGN_OR_THROW(mlir::ElementType out_dtype,
                           ConvertTo<mlir::ElementType>(self.scalar_type()));

        Dimensions out_dims;
        if (resolved_dim.has_value()) {
          out_dims = CopyIntVector(self.sizes());
          out_dims[*resolved_dim] = resolved_output_size;
        } else {
          out_dims = Dimensions({resolved_output_size});
        }

        auto shlo_builder = [resolved_dim, resolved_output_size](
                                FixedSizeSpan<mlir::MlirOp, 2> inputs)
            -> absl::StatusOr<mlir::MlirOp> {
          return BuildRepeatInterleaveShlo(inputs[0], inputs[1], resolved_dim,
                                           resolved_output_size);
        };

        TT_ASSIGN_OR_THROW(
            auto result,
            (DispatchOp<2>(std::move(shlo_builder), {self, resolved_repeats},
                           {.out_dtype = out_dtype,
                            .out_dims = out_dims,
                            .op_param_cache_keys = std::move(param_keys)})));
        return MakeTensor(result);
      });
}

}  // namespace torch_tpu
