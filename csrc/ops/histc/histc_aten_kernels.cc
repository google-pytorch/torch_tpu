/*
 * Copyright 2025 Google LLC
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

#include "csrc/ops/histc/histc_aten_kernels.h"

#include <cmath>
#include <cstdint>
#include <optional>
#include <utility>

#include "ATen/core/ATen_fwd.h"
#include "ATen/core/ScalarType.h"
#include "ATen/core/TensorBody.h"
#include "ATen/ops/empty.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "c10/core/ScalarType.h"
#include "csrc/common/aten_utils.h"
#include "csrc/common/cache_key.h"
#include "csrc/common/dimension_types.h"
#include "csrc/common/dtype.h"
#include "csrc/common/error_utils.h"
#include "csrc/common/fixed_size_span.h"
#include "csrc/common/to_string.h"
#include "csrc/common/warnings.h"
#include "csrc/eager/op_dispatcher.h"
#include "csrc/ops/macros/kernel.h"
#include "csrc/ops/op_builder_utils.h"
#include "csrc/ops/op_names.h"
#include "csrc/ops/resize/resize_aten_kernels.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Support/LLVM.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"
#include "torch/headeronly/core/ScalarType.h"

namespace torch_tpu {

namespace {

struct HistcBounds {
  mlir::MlirOp min;
  mlir::MlirOp max;
};

namespace stablehlo = mlir::stablehlo;

// Builds a reduction over all dimensions of the input tensor, resulting in a
// rank 0 tensor with a single element.
template <typename ReduceOp>
mlir::MlirOp BuildReduce(mlir::MlirOp input_op, mlir::Attribute init_attr) {
  mlir::MlirBuilder& builder = input_op.getBuilder();
  const mlir::RankedTensorType input_type = GetTensorTypeOrDie(input_op);

  mlir::DenseElementsAttr value_init_attr = mlir::DenseElementsAttr::get(
      mlir::RankedTensorType::get({}, input_type.getElementType()), init_attr);
  mlir::MlirOp value_init = stablehlo::Constant(builder, value_init_attr);

  auto reduce_builder =
      [&](mlir::RegionBuilder& rb) {  // REF_CAPTURE_OK=called synchronously
        stablehlo::buildReduceBody<ReduceOp>(input_type.getElementType(),
                                             rb.getRegion(), rb.getOpBuilder());
      };
  auto result = stablehlo::Reduce(builder, {input_op}, {value_init},
                                  reduce_builder, GetAllDimensions(input_op));
  // Reducing over all dimensions yields a single result.
  return result[0];
}

// Builds a reduction to compute the minimum value over all elements of
// input_op.
mlir::MlirOp BuildReduceMinShlo(mlir::MlirOp input_op) {
  mlir::MlirBuilder& builder = input_op.getBuilder();
  const mlir::RankedTensorType input_type = GetTensorTypeOrDie(input_op);
  mlir::Attribute init_attr = GetMaxFiniteValueAttr(input_type.getElementType(),
                                                    builder.getOpBuilder());
  return BuildReduce<stablehlo::MinOp>(input_op, init_attr);
}

// Builds a reduction to compute the maximum value over all elements of
// input_op.
mlir::MlirOp BuildReduceMaxShlo(mlir::MlirOp input_op) {
  mlir::MlirBuilder& builder = input_op.getBuilder();
  const mlir::RankedTensorType input_type = GetTensorTypeOrDie(input_op);
  mlir::Attribute init_attr = GetMinFiniteValueAttr(input_type.getElementType(),
                                                    builder.getOpBuilder());
  return BuildReduce<stablehlo::MaxOp>(input_op, init_attr);
}

// Adjusts the min and max to account for the case where min == max.
// If min == max, set min = min - 1 and max = max + 1.
void AdjustMinMaxForEq(mlir::MlirOp& min_op, mlir::MlirOp& max_op) {
  mlir::Type dtype = GetTensorTypeOrDie(min_op).getElementType();

  mlir::MlirBuilder& builder = min_op.getBuilder();
  mlir::MlirOp one = MakeScalarConstant(builder, 1, dtype);
  mlir::MlirOp min_op_minus_one = stablehlo::Subtract(min_op, one);
  mlir::MlirOp max_op_plus_one = stablehlo::Add(max_op, one);

  mlir::MlirOp cond_op =
      stablehlo::Compare(min_op, max_op, stablehlo::ComparisonDirection::EQ);
  min_op = stablehlo::Select(cond_op, min_op_minus_one, min_op);
  max_op = stablehlo::Select(cond_op, max_op_plus_one, max_op);
}

// Computes bin indices for each element in input_op.
mlir::MlirOp BuildIndices(mlir::MlirOp input_op, mlir::MlirOp min_op,
                          mlir::MlirOp max_op, int64_t bins) {
  // floor((input_op - min) * bins / (max - min))
  mlir::MlirOp numerator = stablehlo::Subtract(input_op, min_op);
  mlir::MlirOp denominator = stablehlo::Subtract(max_op, min_op);
  mlir::MlirOp bins_op = MakeConstantLike(input_op, bins);
  mlir::MlirOp mul_bins = stablehlo::Mul(numerator, bins_op);
  mlir::MlirOp div = stablehlo::Div(mul_bins, denominator);
  if (GetTensorTypeOrDie(input_op).getElementType().isInteger()) {
    return stablehlo::ConvertElementType(div, mlir::ElementType::I32);
  }
  mlir::MlirOp indices = stablehlo::Floor(div);
  return stablehlo::ConvertElementType(indices, mlir::ElementType::I32);
}

// Clamps the indices to the range [0, bins - 1].
mlir::MlirOp BuildClampedIndices(mlir::MlirOp indices, int64_t bins) {
  // clamped_indices = clamp(indices, 0, bins - 1)
  // Use Min and Max instead of Clamp because stablehlo::Clamp drops dynamic
  // bounds.
  mlir::MlirOp zero_indices = MakeConstantLike(indices, 0);
  mlir::MlirOp bins_minus_1_indices = MakeConstantLike(indices, bins - 1);
  mlir::MlirOp min_indices = stablehlo::Min(indices, bins_minus_1_indices);
  return stablehlo::Max(min_indices, zero_indices);
}

// Returns a boolean tensor that is true for elements in [min_op, max_op].
mlir::MlirOp BuildInRange(mlir::MlirOp input_op, mlir::MlirOp min_op,
                          mlir::MlirOp max_op) {
  // in_range = (input >= min) & (input <= max)
  mlir::MlirOp ge_min =
      stablehlo::Compare(input_op, min_op, stablehlo::ComparisonDirection::GE);
  mlir::MlirOp le_max =
      stablehlo::Compare(input_op, max_op, stablehlo::ComparisonDirection::LE);
  return stablehlo::And(ge_min, le_max);
}

// Builds a scatter-add operation to accumulate histogram counts.
// Counts are only incremented for elements where in_range is true.
// This implementation was adapted from:
// torch_tpu/ops/bincount/bincount_aten_kernels.cc
mlir::MlirOp BuildScatterAdd(mlir::MlirOp clamped_indices,
                             mlir::MlirOp in_range, int64_t bins,
                             mlir::Type out_dtype, mlir::MlirBuilder& builder) {
  // Flatten indices and in_range tensors to 1D to be used in scatter.
  mlir::MlirOp clamped_indices_flat = clamped_indices;
  mlir::MlirOp in_range_flat = in_range;
  if (GetTensorTypeOrDie(clamped_indices).getRank() != 1) {
    clamped_indices_flat = Flatten(clamped_indices);
    in_range_flat = Flatten(in_range);
  }

  // The output histogram tensor of size `bins` is initialized to zeros.
  Dimensions output_dims = {bins};
  mlir::MlirOp empty_output = MakeConstant(builder, 0, out_dtype, output_dims);
  // Scatter indices can be 1D when index_vector_dim = 1.
  mlir::MlirOp scatter_indices = clamped_indices_flat;

  // Contribution to histogram is 1 if element is in range, 0 otherwise.
  mlir::MlirOp scatter_update =
      stablehlo::ConvertElementType(in_range_flat, out_dtype);

  // Create a region builder callback for scatter-add.
  // The scatter operation will compute:
  //   output[scatter_indices[i]] += scatter_update[i]
  auto scatter_builder =
      [&](mlir::RegionBuilder& builder) {  // REF_CAPTURE_OK=called locally
        stablehlo::buildReduceBody<stablehlo::AddOp>(
            out_dtype, builder.getRegion(), builder.getOpBuilder());
      };

  auto scatter_dimension_numbers =
      mlir::stablehlo::ScatterDimensionNumbersAttr::get(
          &builder.getContext(),
          /*updateWindowDims=*/{},
          /*inserted_window_dims=*/{0},
          /*input_batching_dims=*/{},
          /*scatter_indices_batching_dims=*/{},
          /*scatter_dims_to_operand_dims=*/{0},
          /*index_vector_dim=*/1);

  mlir::MlirOp hist =
      mlir::stablehlo::Scatter(empty_output, scatter_indices, scatter_update,
                               scatter_builder, scatter_dimension_numbers)[0];
  return hist;
}

absl::StatusOr<HistcBounds> BuildHistcBoundsShlo(mlir::MlirOp input_op) {
  // Below code mirrors the PyTorch CUDA implementation:

  // TODO(b/474571547): Replace with PyTorch min/max operation once nans are
  // propagated.
  mlir::MlirOp min_op = BuildReduceMinShlo(input_op);
  mlir::MlirOp max_op = BuildReduceMaxShlo(input_op);

  // Account for the case where min == max by adjusting the min and max.
  AdjustMinMaxForEq(min_op, max_op);

  return HistcBounds{.min = min_op, .max = max_op};
}

absl::StatusOr<mlir::MlirOp> BuildHistcShlo(mlir::MlirOp input_op, int64_t bins,
                                            mlir::MlirOp min_op,
                                            mlir::MlirOp max_op) {
  mlir::MlirBuilder& builder = input_op.getBuilder();
  mlir::Type input_dtype = GetTensorTypeOrDie(input_op).getElementType();
  mlir::MlirOp compute_input_op = input_op;
  if (input_dtype.isInteger()) {
    compute_input_op =
        stablehlo::ConvertElementType(input_op, mlir::ElementType::I64);
    min_op = stablehlo::ConvertElementType(min_op, mlir::ElementType::I64);
    max_op = stablehlo::ConvertElementType(max_op, mlir::ElementType::I64);
  }

  // Broadcast min and max to the shape of the input tensor.
  TT_ASSIGN_OR_RETURN(min_op, BroadcastIfNeeded(min_op, compute_input_op));
  TT_ASSIGN_OR_RETURN(max_op, BroadcastIfNeeded(max_op, compute_input_op));

  // Compute bin indices:
  //   indices = floor((input - min) * bins / (max - min))
  mlir::MlirOp indices = BuildIndices(compute_input_op, min_op, max_op, bins);
  // Then clamp indices to [0, bins - 1].
  mlir::MlirOp clamped_indices = BuildClampedIndices(indices, bins);

  // Values outside of [min_op, max_op] are ignored, which mirrors the PyTorch
  // implementation.
  mlir::MlirOp in_range = BuildInRange(compute_input_op, min_op, max_op);
  // Scatter-add to accumulate counts.
  return BuildScatterAdd(clamped_indices, in_range, bins, input_dtype, builder);
}

absl::StatusOr<bool> lt(const at::Scalar& a, const at::Scalar& b) {
  if (a.isFloatingPoint() && b.isFloatingPoint()) {
    return a.to<double>() < b.to<double>();
  }
  if (a.isIntegral(/*includeBool=*/false) &&
      b.isIntegral(/*includeBool=*/false)) {
    if (a.type() == c10::ScalarType::UInt64 &&
        b.type() == c10::ScalarType::UInt64) {
      return a.to<uint64_t>() < b.to<uint64_t>();
    }
    if (a.type() == c10::ScalarType::Long &&
        b.type() == c10::ScalarType::Long) {
      return a.to<int64_t>() < b.to<int64_t>();
    }
  }
  return TT_ERROR(error::kInvalidArgument)
         << "expected min and max to be float or int type, got " << a.type()
         << " and " << b.type();
}

absl::StatusOr<bool> eq(const at::Scalar& a, const at::Scalar& b) {
  if (a.isFloatingPoint() && b.isFloatingPoint()) {
    return a.to<double>() == b.to<double>();
  }
  if (a.isIntegral(/*includeBool=*/false) &&
      b.isIntegral(/*includeBool=*/false)) {
    if (a.type() == c10::ScalarType::UInt64 &&
        b.type() == c10::ScalarType::UInt64) {
      return a.to<uint64_t>() == b.to<uint64_t>();
    }
    if (a.type() == c10::ScalarType::Long &&
        b.type() == c10::ScalarType::Long) {
      return a.to<int64_t>() == b.to<int64_t>();
    }
  }
  return TT_ERROR(error::kInvalidArgument)
         << "expected min and max to be float or int type, got " << a.type()
         << " and " << b.type();
}

// Validates whether the given min and max values are valid. If min_prev
// and max_prev are provided, they contain the min and max values
// provided to torch.histc, while min and max contain values computed from
// the input data. Note that this case only occurs when torch.histc computed
// bounds from the input data.
absl::Status ValidateBounds(
    const at::Scalar& min, const at::Scalar& max,
    const std::optional<at::Scalar>& min_prev = std::nullopt,
    const std::optional<at::Scalar>& max_prev = std::nullopt) {
  // The below code for bounds validation mirrors the logic in the PyTorch GPU
  // kernel implementation.

  // Check for finite.
  if (min.isFloatingPoint() && max.isFloatingPoint()) {
    TT_RET_CHECK(
        !(std::isinf(min.to<double>()) || std::isinf(max.to<double>()) ||
          std::isnan(min.to<double>()) || std::isnan(max.to<double>())),
        error::kInvalidArgument)
        << "expected min and max to be finite, got " << min << " and " << max
        << ". Either make sure that the input data is finite, or provide valid "
           "finite bounds";
  }

  // Check for overflows.
  if (min_prev.has_value() && max_prev.has_value()) {
    // If min_prev < max_prev and min > max, there's an overflow.
    TT_ASSIGN_OR_RETURN(const bool min_prev_lt_max_prev,
                        lt(*min_prev, *max_prev));
    TT_ASSIGN_OR_RETURN(const bool max_lt_min, lt(max, min));
    TT_RET_CHECK(min_prev_lt_max_prev && max_lt_min, error::kInvalidArgument)
        << "expected min and max to be within the range of their data "
           "types, got min = "
        << min << " and max = " << max << ". "
        << "This happened because min and max were adjusted by one (due to min "
           "== max), which resulted in an overflow";
  }

  // Check that min < max.
  TT_ASSIGN_OR_RETURN(const bool min_lt_max, lt(min, max));
  TT_RET_CHECK(min_lt_max, error::kInvalidArgument)
      << "expected min <= max, got " << min << " vs " << max;

  return absl::OkStatus();
}

absl::Status HistcHelperOut(const at::Tensor& self, const int64_t bins,
                            const at::Scalar& min, const at::Scalar& max,
                            PromotedScalar& promoted_min,
                            PromotedScalar& promoted_max, at::Tensor& out,
                            OpParamCacheKeys param_keys) {
  TT_RET_CHECK(!IsComplex(self), error::kInvalidArgument)
      << "expected the first argument not to be complex, got "
      << ToString(self.scalar_type());

  TT_RET_CHECK(self.scalar_type() != at::kBFloat16 && !IsBool(self),
               error::kPythonNotImplementedError)
      << "not implemented for " << ToString(self.scalar_type());

  TT_RETURN_IF_ERROR(
      ValidateOutDtype(out, self.scalar_type(), /*allow_out_dtype_cast=*/true));
  const c10::ScalarType input_dtype = self.scalar_type();
  TT_ASSIGN_OR_RETURN(at::Tensor min_val, promoted_min.GetTensor(input_dtype));
  TT_ASSIGN_OR_RETURN(at::Tensor max_val, promoted_max.GetTensor(input_dtype));

  const int64_t numel = self.numel();

  // If input is empty, return tensor of zeros.
  if (numel == 0) {
    TT_RETURN_IF_ERROR(ResizeTensorIfShapeDiffers(out, {bins}));
    out.zero_();
    return absl::OkStatus();
  }

  TT_ASSIGN_OR_RETURN(auto mlir_input_dtype,
                      ConvertTo<mlir::ElementType>(input_dtype));

  TT_ASSIGN_OR_RETURN(const bool min_eq_max, eq(min, max));
  if (!(min_eq_max && numel > 0)) {
    // Validate the bounds provided as arguments to torch.histc.
    TT_RETURN_IF_ERROR(ValidateBounds(min, max));
  } else {
    // Compute min/max from input data. This is done when min == max and
    // numel > 0, which happens when the min and max are not provided as
    // arguments to torch.histc and the input tensor is not empty. The
    // PyTorch CPU and GPU kernels differ from the PyTorch documentation,
    // which states that min/max are computed when min == max == 0. To
    // maintain parity, the TPU kernel implementation follows the logic of
    // the GPU/CPU kernels. See the PyTorch GPU kernel implementation for
    // more details:
    // https://github.com/pytorch/pytorch/blob/v2.9.1/aten/src/ATen/native/cuda/SummaryOps.cu#L328
    auto op_builder =
        [](mlir::MlirOp input) -> absl::StatusOr<MlirOpResults<2>> {
      TT_ASSIGN_OR_RETURN(auto min_max_outputs, BuildHistcBoundsShlo(input));
      return {{min_max_outputs.min, min_max_outputs.max}};
    };

    // Distinguish the bound computation from the main computation
    // in the dispatcher.
    const bool compute_bounds = true;
    TT_ASSIGN_OR_RETURN(auto compute_bounds_param_keys,
                        TT_MAKE_OP_PARAM_CACHE_KEYS(compute_bounds));

    TT_RETURN_IF_ERROR((DispatchOpOut<1, 2>(
        std::move(op_builder), self, {min_val, max_val},
        {.out_dtypes = {mlir_input_dtype, mlir_input_dtype},
         .out_dims_list = {Dimensions(), Dimensions()},
         .op_param_cache_keys = std::move(compute_bounds_param_keys)})));

    // Validate computed bounds, as they may be invalid due to underflow
    // and overflow. Because this needs an expensive transfer to CPU, this
    // is only done when debug checks are enabled.
    if (GetEnableDebugChecks()) {
      TT_RETURN_IF_ERROR(
          ValidateBounds(min_val.cpu().item(), max_val.cpu().item(), min, max));
    } else {
      TT_WARN_ONCE(
          WarningCategory::kCorrectness,
          "Bounds computed from input data are not validated by default on "
          "TorchTPU (unlike CUDA/CPU) to prevent an expensive CPU transfer. "
          "Set the TORCH_TPU_ENABLE_DEBUG_CHECKS environment variable to \"1\" "
          "to enable this check, which incurs a performance penalty.");
    }
  }

  // Perform the actual histogram computation.
  TT_ASSIGN_OR_RETURN(auto mlir_out_dtype,
                      ConvertTo<mlir::ElementType>(out.scalar_type()));
  auto op_builder = [bins](FixedSizeSpan<mlir::MlirOp, 3> inputs)
      -> absl::StatusOr<mlir::MlirOp> {
    auto& [input, min, max] = inputs;
    return BuildHistcShlo(input, bins, min, max);
  };
  TT_RETURN_IF_ERROR(
      DispatchOpOut<3>(std::move(op_builder), {self, min_val, max_val}, out,
                       {.out_dtype = mlir_out_dtype,
                        .out_dims = {bins},
                        .op_param_cache_keys = std::move(param_keys)}));

  return absl::OkStatus();
}

}  // namespace

at::Tensor AtenHistc(const at::Tensor& self, const int64_t bins,
                     const at::Scalar& min, const at::Scalar& max) {
  auto promoted_min = PromoteScalar(min);
  auto promoted_max = PromoteScalar(max);
  TT_KERNEL(
      OpName::kHistc, param_keys, (self, bins, promoted_min, promoted_max), {
        c10::ScalarType out_dtype = self.scalar_type();
        at::Tensor out = at::empty({bins}, self.options().dtype(out_dtype));
        TT_THROW_IF_ERROR(HistcHelperOut(self, bins, min, max, promoted_min,
                                         promoted_max, out,
                                         std::move(param_keys)));
        return out;
      });
}

at::Tensor& AtenHistcOut(const at::Tensor& self, const int64_t bins,
                         const at::Scalar& min, const at::Scalar& max,
                         at::Tensor& out) {
  auto promoted_min = PromoteScalar(min);
  auto promoted_max = PromoteScalar(max);
  TT_KERNEL(OpName::kHistcOut, param_keys,
            (self, bins, promoted_min, promoted_max, out), {
              TT_THROW_IF_ERROR(HistcHelperOut(self, bins, min, max,
                                               promoted_min, promoted_max, out,
                                               std::move(param_keys)));
              return out;
            });
}

}  // namespace torch_tpu
