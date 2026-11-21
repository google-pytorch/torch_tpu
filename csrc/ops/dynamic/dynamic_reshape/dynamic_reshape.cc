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

#include "csrc/ops/dynamic/dynamic_reshape/dynamic_reshape.h"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <utility>
#include <vector>

#include "ATen/core/ATen_fwd.h"
#include "ATen/core/TensorBody.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "c10/core/ScalarType.h"
#include "csrc/common/dimension_types.h"
#include "csrc/common/dtype.h"
#include "csrc/common/error_utils.h"
#include "csrc/common/to_string.h"
#include "csrc/common/utils.h"
#include "csrc/eager/device_buffer.h"
#include "csrc/eager/op_dispatcher.h"
#include "csrc/eager/tensor_to_buffer.h"
#include "csrc/ops/cat/cat.h"
#include "csrc/ops/macros/kernel.h"
#include "csrc/ops/op_builder_utils.h"
#include "csrc/ops/op_names.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Types.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"
#include "stablehlo/transforms/StablehloBroadcastLowering.h"

namespace torch_tpu {
namespace {

// Returns true when flattening `input` to a 1D bounded dynamic tensor of bound
// `out_bound` would require XLA's `dynamic_padder` to compact non-major
// dynamic dimensions via an O(N log N) stable sort + gather.
//
// In that case, we can compute the exact source indices in closed form via
// integer division/modulo and emit a direct 1D gather, avoiding massive sort
// codegen and compile-time blowups.
bool ShouldUseGatherForDynamicFlatten(const mlir::MlirOp& input,
                                      const Dimensions& out_dims,
                                      const std::vector<bool>& is_dynamic) {
  if (out_dims.size() != 1 || !is_dynamic[0]) {
    return false;
  }
  const mlir::RankedTensorType input_type = GetTensorTypeOrDie(input);
  const int64_t rank = input_type.getRank();
  if (rank < 2) {
    return false;
  }

  const mlir::stablehlo::Dimensions input_dims = GetDimensions(input);
  int64_t input_numel = 1;
  bool has_leading_extent = false;
  bool has_non_major_dynamic_dim = false;
  for (int64_t dim = 0; dim < rank; ++dim) {
    const int64_t bound = input_dims[dim].size;
    if (bound <= 0 ||
        input_numel > std::numeric_limits<int32_t>::max() / bound) {
      return false;
    }
    if (input_type.isDynamicDim(dim) && has_leading_extent) {
      has_non_major_dynamic_dim = true;
    }
    if (bound > 1) {
      has_leading_extent = true;
    }
    input_numel *= bound;
  }

  const int64_t out_bound = out_dims[0];
  return has_non_major_dynamic_dim && out_bound > 0 &&
         out_bound <= std::numeric_limits<int32_t>::max();
}

// Computes exact `(rem / d, rem % d)` for non-negative `rem < out_bound` and
// `d >= 1`. When `out_bound <= (1 << 22)`, uses a scalar f32 reciprocal plus
// a single-step +/-1 correction to avoid expensive TPU s32 vector division.
struct DivModResult {
  mlir::MlirOp quotient;
  mlir::MlirOp remainder;
};

DivModResult ComputeDivMod(mlir::MlirOp rem, mlir::MlirOp dim_scalar,
                           mlir::MlirOp dim_vec,
                           mlir::RankedTensorType idx_tensor_type,
                           const int64_t out_bound,
                           mlir::MlirBuilder& builder) {
  constexpr int64_t kMaxExactF32RecipBound = 1LL << 22;
  if (out_bound > kMaxExactF32RecipBound) {
    mlir::MlirOp quotient = mlir::stablehlo::Div(rem, dim_vec);
    mlir::MlirOp product = mlir::stablehlo::Mul(quotient, dim_vec);
    mlir::MlirOp remainder = mlir::stablehlo::Subtract(rem, product);
    return {.quotient = quotient, .remainder = remainder};
  }

  mlir::Type f32_type = builder.getOpBuilder().getF32Type();
  mlir::RankedTensorType f32_scalar_type =
      mlir::RankedTensorType::get({}, f32_type);
  mlir::RankedTensorType f32_vec_type =
      mlir::RankedTensorType::get({out_bound}, f32_type);

  mlir::MlirOp one_f32 = MakeScalarConstant(builder, 1.0f, f32_type);
  mlir::MlirOp dim_f32 = mlir::stablehlo::Convert(f32_scalar_type, dim_scalar);
  mlir::MlirOp recip_scalar = mlir::stablehlo::Div(one_f32, dim_f32);
  mlir::MlirOp recip_vec = mlir::stablehlo::BroadcastInDim(
      f32_vec_type, recip_scalar, /*broadcast_dimensions=*/{});

  mlir::MlirOp rem_f32 = mlir::stablehlo::Convert(f32_vec_type, rem);
  mlir::MlirOp quot_f32 = mlir::stablehlo::Mul(rem_f32, recip_vec);
  mlir::MlirOp quotient = mlir::stablehlo::Convert(idx_tensor_type, quot_f32);
  mlir::MlirOp product = mlir::stablehlo::Mul(quotient, dim_vec);
  mlir::MlirOp remainder = mlir::stablehlo::Subtract(rem, product);

  // Correct off-by-one rounding from f32 reciprocal multiplication.
  mlir::MlirOp zero_vec = MakeConstant(builder, int32_t{0}, idx_tensor_type);
  mlir::MlirOp one_vec = MakeConstant(builder, int32_t{1}, idx_tensor_type);

  mlir::MlirOp underflow = mlir::stablehlo::Compare(
      remainder, zero_vec, mlir::stablehlo::ComparisonDirection::LT);
  mlir::MlirOp quot_dec = mlir::stablehlo::Subtract(quotient, one_vec);
  mlir::MlirOp rem_inc = mlir::stablehlo::Add(remainder, dim_vec);
  quotient = mlir::stablehlo::Select(underflow, quot_dec, quotient);
  remainder = mlir::stablehlo::Select(underflow, rem_inc, remainder);

  mlir::MlirOp overflow = mlir::stablehlo::Compare(
      remainder, dim_vec, mlir::stablehlo::ComparisonDirection::GE);
  mlir::MlirOp quot_inc = mlir::stablehlo::Add(quotient, one_vec);
  mlir::MlirOp rem_dec = mlir::stablehlo::Subtract(remainder, dim_vec);
  quotient = mlir::stablehlo::Select(overflow, quot_inc, quotient);
  remainder = mlir::stablehlo::Select(overflow, rem_dec, remainder);

  return {.quotient = quotient, .remainder = remainder};
}

// Accumulates `coord * stride` into `acc`, avoiding redundant multiplies or
// adds for unit stride or the first term.
void AddScaledCoordinate(std::optional<mlir::MlirOp>& acc, mlir::MlirOp coord,
                         const int64_t stride,
                         mlir::RankedTensorType idx_tensor_type,
                         mlir::MlirBuilder& builder) {
  mlir::MlirOp term = coord;
  if (stride != 1) {
    mlir::MlirOp stride_vec =
        MakeConstant(builder, static_cast<int32_t>(stride), idx_tensor_type);
    term = mlir::stablehlo::Mul(coord, stride_vec);
  }
  acc = acc.has_value() ? mlir::stablehlo::Add(*acc, term) : term;
}

// Flattens a bounded-dynamic tensor `input` into a 1D bounded-dynamic tensor of
// bound `out_bound` and runtime size `out_size` using closed-form index
// arithmetic and a single 1D `stablehlo.gather`.
mlir::MlirOp BuildDynamicFlattenWithGather(mlir::MlirOp input,
                                           mlir::MlirOp out_size,
                                           const int64_t out_bound,
                                           mlir::MlirBuilder& builder) {
  const mlir::stablehlo::Dimensions input_dims = GetDimensions(input);
  const int64_t rank = static_cast<int64_t>(input_dims.size());
  int64_t input_numel = 1;
  for (const mlir::stablehlo::DimensionInfo& dim_info : input_dims) {
    input_numel *= dim_info.size;
  }

  // Strip dynamic bounds to get the static padded buffer and capture each
  // dynamic dimension's runtime size.
  StaticAtBounds static_input = MakeStaticAtBounds(input);
  mlir::MlirOp flat_static =
      mlir::stablehlo::Reshape(static_input.op, {input_numel});

  mlir::Type i32_type = builder.getOpBuilder().getI32Type();
  mlir::RankedTensorType idx_tensor_type =
      mlir::RankedTensorType::get({out_bound}, i32_type);
  mlir::MlirOp rem =
      mlir::stablehlo::Iota(builder, idx_tensor_type, /*iota_dimension=*/0);
  mlir::MlirOp one_scalar = MakeScalarConstant(builder, int32_t{1}, i32_type);

  // Map each output flat index `k` in `[0, out_bound)` to its logical
  // coordinates `(i_0, ..., i_{rank-1})` under the runtime sizes
  // `(d_0, ..., d_{rank-1})`, then linearize into the padded static buffer
  // using static strides `stride_d = prod_{j > d} B_j`.
  std::optional<mlir::MlirOp> src_idx;
  int64_t stride = 1;
  for (int64_t dim = rank - 1; dim >= 1; --dim) {
    const int64_t bound = input_dims[dim].size;
    if (bound == 1) {
      continue;
    }
    mlir::MlirOp dim_scalar;
    mlir::MlirOp dim_size_vec;
    if (static_input.runtime_sizes[dim].has_value()) {
      // Clamp runtime size to >= 1 to avoid division by zero when a dynamic
      // dimension has runtime size 0 (in which case `out_size == 0` and all
      // output elements are padding).
      dim_scalar =
          mlir::stablehlo::Max(*static_input.runtime_sizes[dim], one_scalar);
      dim_size_vec = mlir::stablehlo::BroadcastInDim(
          idx_tensor_type, dim_scalar, /*broadcast_dimensions=*/{});
    } else {
      dim_scalar =
          MakeScalarConstant(builder, static_cast<int32_t>(bound), i32_type);
      dim_size_vec =
          MakeConstant(builder, static_cast<int32_t>(bound), idx_tensor_type);
    }
    DivModResult div_mod = ComputeDivMod(rem, dim_scalar, dim_size_vec,
                                         idx_tensor_type, out_bound, builder);
    rem = div_mod.quotient;
    AddScaledCoordinate(src_idx, div_mod.remainder, stride, idx_tensor_type,
                        builder);
    stride *= bound;
  }

  // Clamp the major-most coordinate to `[0, B_0 - 1]` so padding indices
  // (`k >= out_size`) remain strictly in bounds for the gather.
  const int64_t bound_0 = input_dims[0].size;
  if (bound_0 > 1) {
    mlir::MlirOp max_coord_0 = MakeConstant(
        builder, static_cast<int32_t>(bound_0 - 1), idx_tensor_type);
    mlir::MlirOp coord_0 = mlir::stablehlo::Min(rem, max_coord_0);
    AddScaledCoordinate(src_idx, coord_0, stride, idx_tensor_type, builder);
  }

  const mlir::stablehlo::GatherDimensionNumbersAttr gather_dims =
      mlir::stablehlo::GatherDimensionNumbersAttr::get(
          &builder.getContext(),
          /*offset_dims=*/{},
          /*collapsed_slice_dims=*/{0},
          /*operand_batching_dims=*/{},
          /*start_indices_batching_dims=*/{},
          /*start_index_map=*/{0},
          /*index_vector_dim=*/1);
  mlir::MlirOp gathered = mlir::stablehlo::Gather(
      flat_static, *src_idx, gather_dims, /*slice_sizes=*/{1},
      /*indices_are_sorted=*/false);
  return mlir::stablehlo::SetDimensionSize(gathered, out_size, /*dimension=*/0);
}

}  // namespace

at::Tensor DynamicReshape(const at::Tensor& input, at::TensorList shape,
                          at::IntArrayRef static_shape,
                          c10::List<bool> is_dynamic) {
  TT_KERNEL(
      OpName::kDynamicReshape, param_keys,
      (input, shape, static_shape, is_dynamic), {
        const size_t rank = static_shape.size();
        TT_CHECK_THROW(shape.size() == rank, error::kInvalidArgument)
            << "expected shape list size to match static_shape size, got shape "
               "list "
               "size "
            << shape.size() << " and static_shape size " << rank;
        TT_CHECK_THROW(is_dynamic.size() == rank, error::kInvalidArgument)
            << "expected is_dynamic size to match static_shape size, got "
               "is_dynamic "
               "size "
            << is_dynamic.size() << " and static_shape size " << rank;

        for (size_t i = 0; i < shape.size(); ++i) {
          const auto& t = shape[i];
          TT_CHECK_THROW(t.dim() == 0, error::kInvalidArgument)
              << "expected shape tensor at index " << i
              << " to be a 0D (scalar) tensor, got " << t.dim() << "D tensor";
          TT_CHECK_THROW(t.scalar_type() == at::kInt, error::kInvalidArgument)
              << "expected shape to be a list of int32 tensors, got "
              << ToString(t.scalar_type()) << " tensor at index " << i;
        }

        TT_ASSIGN_OR_THROW(const mlir::ElementType mlir_dtype,
                           ConvertTo<mlir::ElementType>(input.scalar_type()));

        const auto out_dims = CopyIntVector(static_shape);
        const std::vector<bool> is_dynamic_vec(is_dynamic.begin(),
                                               is_dynamic.end());

        // Concatenate input tensor and shape descriptors
        std::vector<at::Tensor> all_inputs;
        all_inputs.reserve(1 + shape.size());
        all_inputs.push_back(input);
        for (const auto& t : shape) {
          all_inputs.push_back(t);
        }

        auto builder =
            [shape_len = shape.size(), out_dims, is_dynamic_vec](
                absl::Span<mlir::MlirOp> inputs,
                mlir::MlirBuilder& builder) -> absl::StatusOr<mlir::MlirOp> {
          if (ShouldUseGatherForDynamicFlatten(inputs[0], out_dims,
                                               is_dynamic_vec)) {
            return BuildDynamicFlattenWithGather(inputs[0], inputs[1],
                                                 out_dims[0], builder);
          }

          // 1. Process and concatenate shape inputs to a single 1D tensor
          std::vector<mlir::MlirOp> shape_ops;
          shape_ops.reserve(shape_len);

          for (size_t i = 0; i < shape_len; ++i) {
            mlir::MlirOp dim_op = inputs[1 + i];
            auto dim_type = GetTensorTypeOrDie(dim_op);
            if (dim_type.getRank() == 0) {
              dim_op = mlir::stablehlo::Reshape(dim_op, {1});
            }
            shape_ops.push_back(dim_op);
          }

          TT_ASSIGN_OR_RETURN(mlir::MlirOp concatenated_shape,
                              BuildCatShlo(absl::MakeSpan(shape_ops), 0));

          // 2. Prepare MLIR dimensions and bounds
          Dimensions mlir_out_dims;
          mlir_out_dims.reserve(out_dims.size());
          Dimensions bounds;
          bounds.reserve(out_dims.size());

          for (size_t i = 0; i < out_dims.size(); ++i) {
            if (is_dynamic_vec[i]) {
              // Dynamic dimension: placeholder in shape, bounded by
              // static_shape
              mlir_out_dims.push_back(mlir::ShapedType::kDynamic);
              bounds.push_back(out_dims[i]);
            } else {
              // Static dimension: explicit size, no variable upper bound
              mlir_out_dims.push_back(out_dims[i]);
              bounds.push_back(mlir::ShapedType::kDynamic);
            }
          }

          // 3. Attach Bounds as Type Extensions Encoding
          auto bounds_attr = mlir::stablehlo::TypeExtensionsAttr::get(
              &builder.getContext(), bounds);

          auto result_type = mlir::RankedTensorType::get(
              mlir_out_dims, GetTensorTypeOrDie(inputs[0]).getElementType(),
              bounds_attr);

          // 4. Emit the StableHLO op
          return mlir::stablehlo::DynamicReshape(result_type, inputs[0],
                                                 concatenated_shape);
        };

        TT_ASSIGN_OR_THROW(auto result_buf,
                           DispatchOp<kDynamicSize>(
                               std::move(builder), all_inputs,
                               {.out_dtype = mlir_dtype,
                                .out_dims = out_dims,
                                .op_param_cache_keys = std::move(param_keys)}));
        return MakeTensor(std::move(result_buf));
      });
}

}  // namespace torch_tpu
