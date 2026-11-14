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
#include "absl/log/absl_log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "c10/core/ScalarType.h"
#include "csrc/common/aten_utils.h"
#include "csrc/common/cache_key.h"
#include "csrc/common/device_type.h"
#include "csrc/common/dimension_types.h"
#include "csrc/common/dtype.h"
#include "csrc/common/error_utils.h"
#include "csrc/common/fixed_size_span.h"
#include "csrc/common/op_name_stack.h"
#include "csrc/common/to_string.h"
#include "csrc/common/warnings.h"
#include "csrc/eager/device_buffer.h"
#include "csrc/eager/op_dispatcher.h"
#include "csrc/eager/tensor_to_buffer.h"
#include "csrc/ops/macros/kernel.h"
#include "csrc/ops/nullary_aten_kernels.h"
#include "csrc/ops/op_builder_utils.h"
#include "csrc/ops/op_names.h"
#include "csrc/ops/python_context.h"
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
  mlir::Attribute init_attr = GetMinReductionIdentityAttr(
      input_type.getElementType(), builder.getOpBuilder());
  return BuildReduce<stablehlo::MinOp>(input_op, init_attr);
}

// Builds a reduction to compute the maximum value over all elements of
// input_op.
mlir::MlirOp BuildReduceMaxShlo(mlir::MlirOp input_op) {
  mlir::MlirBuilder& builder = input_op.getBuilder();
  const mlir::RankedTensorType input_type = GetTensorTypeOrDie(input_op);
  mlir::Attribute init_attr = GetMaxReductionIdentityAttr(
      input_type.getElementType(), builder.getOpBuilder());
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
  const mlir::Type input_dtype = GetTensorTypeOrDie(input_op).getElementType();
  const mlir::Type bounds_dtype = GetTensorTypeOrDie(min_op).getElementType();
  mlir::MlirOp compute_input_op = input_op;
  if (input_dtype.isInteger()) {
    compute_input_op =
        stablehlo::ConvertElementType(input_op, mlir::ElementType::I64);
    min_op = stablehlo::ConvertElementType(min_op, mlir::ElementType::I64);
    max_op = stablehlo::ConvertElementType(max_op, mlir::ElementType::I64);
  } else if (bounds_dtype != input_dtype) {
    compute_input_op = stablehlo::ConvertElementType(input_op, bounds_dtype);
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

enum class BoundsMode {
  kExplicit,
  kFromInput,
};

struct ValidatedHistcBounds {
  BoundsMode mode;
  at::Scalar acc_min;
  at::Scalar acc_max;
};

struct HistcBoundTensors {
  at::Tensor min_val;  // UNINITIALIZED_TENSOR_OK: Aggregate struct field.
  at::Tensor max_val;  // UNINITIALIZED_TENSOR_OK: Aggregate struct field.
};

absl::Status ValidateScalarBoundsTypes(const at::Scalar& min,
                                       const at::Scalar& max) {
  const bool min_valid =
      min.isFloatingPoint() || min.isIntegral(/*includeBool=*/false);
  const bool max_valid =
      max.isFloatingPoint() || max.isIntegral(/*includeBool=*/false);
  TT_RET_CHECK(min_valid && max_valid, error::kInvalidArgument)
      << "expected min and max to be float or int type, got "
      << ToString(min.type()) << " and " << ToString(max.type());
  return absl::OkStatus();
}

// Precondition: ValidateScalarBoundsTypes succeeded for `scalar`.
absl::StatusOr<at::Scalar> CastScalarToAccumulateType(
    const at::Scalar& scalar, const at::ScalarType acc_dtype) {
  TT_ASSIGN_OR_RETURN(const bool can_cast,
                      CanCastScalarWithoutOverflow(scalar, acc_dtype));
  TT_RET_CHECK(can_cast, error::kInvalidArgument)
      << "value cannot be converted to type " << ToString(acc_dtype)
      << " without overflow";
  switch (acc_dtype) {
    case at::ScalarType::Long:
      return at::Scalar(scalar.toLong());
    case at::ScalarType::UInt64:
      return at::Scalar(scalar.toUInt64());
    case at::ScalarType::Float:
      return at::Scalar(static_cast<double>(scalar.toFloat()));
    case at::ScalarType::Double:
      return at::Scalar(scalar.toDouble());
    default:
      ABSL_LOG(FATAL)  // CRASH_OK
          << "Unexpected accumulate dtype for histc: " << ToString(acc_dtype);
  }
}

// Precondition: ValidateScalarBoundsTypes(a, b) succeeded.
bool ScalarLt(const at::Scalar& a, const at::Scalar& b) {
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
    if (a.type() == c10::ScalarType::Long) {
      const int64_t a_val = a.to<int64_t>();
      return a_val < 0 || static_cast<uint64_t>(a_val) < b.to<uint64_t>();
    }
    const int64_t b_val = b.to<int64_t>();
    return b_val >= 0 && a.to<uint64_t>() < static_cast<uint64_t>(b_val);
  }
  return a.to<double>() < b.to<double>();
}

// Precondition: ValidateScalarBoundsTypes(a, b) succeeded.
bool ScalarEq(const at::Scalar& a, const at::Scalar& b) {
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
    if (a.type() == c10::ScalarType::Long) {
      const int64_t a_val = a.to<int64_t>();
      return a_val >= 0 && static_cast<uint64_t>(a_val) == b.to<uint64_t>();
    }
    const int64_t b_val = b.to<int64_t>();
    return b_val >= 0 && a.to<uint64_t>() == static_cast<uint64_t>(b_val);
  }
  return a.to<double>() == b.to<double>();
}

absl::Status ValidateScalarBoundsFinite(const at::Scalar& min,
                                        const at::Scalar& max) {
  if (min.isFloatingPoint() || max.isFloatingPoint()) {
    const double min_val = min.to<double>();
    const double max_val = max.to<double>();
    TT_RET_CHECK(!(std::isinf(min_val) || std::isinf(max_val) ||
                   std::isnan(min_val) || std::isnan(max_val)),
                 error::kInvalidArgument)
        << "expected min and max to be finite, got " << min << " and " << max
        << ". Either make sure that the input data is finite, or provide valid "
           "finite bounds";
  }
  return absl::OkStatus();
}

// Validates whether the given min and max values are valid. When `bounds_mode`
// is `BoundsMode::kFromInput`, `min` and `max` are the bounds computed from
// the input data (after adjusting `[x, x]` to `[x - 1, x + 1]`).
absl::Status ValidateBounds(
    const at::Scalar& min, const at::Scalar& max,
    const BoundsMode bounds_mode = BoundsMode::kExplicit) {
  // The below code for bounds validation mirrors the logic in the PyTorch GPU
  // kernel implementation.
  TT_RETURN_IF_ERROR(ValidateScalarBoundsTypes(min, max));
  TT_RETURN_IF_ERROR(ValidateScalarBoundsFinite(min, max));

  // Check for overflows.
  if (bounds_mode == BoundsMode::kFromInput) {
    // Before adjustment, min <= max holds for any non-empty input, so max < min
    // after adjustment means +/- 1 overflowed or underflowed.
    TT_RET_CHECK(!ScalarLt(max, min), error::kInvalidArgument)
        << "expected min and max to be within the range of their data "
           "types, got min = "
        << min << " and max = " << max << ". "
        << "This happened because min and max were adjusted by one (due to min "
           "== max), which resulted in an overflow";
  }

  // Check that min < max.
  TT_RET_CHECK(ScalarLt(min, max), error::kInvalidArgument)
      << "expected min <= max, got " << min << " vs " << max;

  return absl::OkStatus();
}

absl::Status ValidateHistcInputTensor(const at::Tensor& self) {
  TT_RET_CHECK(IsPrivateUse1Device(self), error::kInvalidArgument)
      << "expected tensor to be on " << GetPrivateUse1DeviceDebugName()
      << ", got " << self.device();

  TT_RET_CHECK(!IsComplex(self), error::kInvalidArgument)
      << "expected the first argument not to be complex, got "
      << ToString(self.scalar_type());

  TT_RET_CHECK(self.scalar_type() != at::kBFloat16 && !IsBool(self),
               error::kPythonNotImplementedError)
      << "not implemented for " << ToString(self.scalar_type());

  return absl::OkStatus();
}

absl::StatusOr<ValidatedHistcBounds> ValidateHistcBinsAndBounds(
    const at::Tensor& self, const int64_t bins, const at::Scalar& min,
    const at::Scalar& max) {
  TT_RET_CHECK(bins > 0, error::kInvalidArgument)
      << "expected bins > 0, got " << bins;

  TT_RETURN_IF_ERROR(ValidateScalarBoundsTypes(min, max));
  const at::ScalarType acc_dtype = ToAccumulateType(self.scalar_type());
  TT_ASSIGN_OR_RETURN(const at::Scalar acc_min,
                      CastScalarToAccumulateType(min, acc_dtype));
  TT_ASSIGN_OR_RETURN(const at::Scalar acc_max,
                      CastScalarToAccumulateType(max, acc_dtype));
  const bool min_eq_max = ScalarEq(acc_min, acc_max);
  if (!min_eq_max) {
    // Validate the bounds provided as arguments to torch.histc.
    TT_RETURN_IF_ERROR(ValidateBounds(acc_min, acc_max));
    return ValidatedHistcBounds{
        .mode = BoundsMode::kExplicit,
        .acc_min = acc_min,
        .acc_max = acc_max,
    };
  }
  if (self.numel() == 0) {
    // On empty input with min == max, PyTorch adjusts to [min - 1, max + 1],
    // which still requires min and max to be finite. We return `acc_min` and
    // `acc_max` unadjusted (instead of `[acc_min - 1, acc_max + 1]`) because
    // AtenHistc and AtenHistcOut early-return a zero tensor when
    // `self.numel() == 0` without reading `mode`, `acc_min`, or `acc_max`.
    TT_RETURN_IF_ERROR(ValidateScalarBoundsFinite(acc_min, acc_max));
    return ValidatedHistcBounds{
        .mode = BoundsMode::kExplicit,
        .acc_min = acc_min,
        .acc_max = acc_max,
    };
  }
  // When `mode == BoundsMode::kFromInput`, callers compute bounds from `self`
  // via HistcFromInput / HistcOutFromInput, so `acc_min` and `acc_max` are
  // unused placeholders.
  return ValidatedHistcBounds{
      .mode = BoundsMode::kFromInput,
      .acc_min = acc_min,
      .acc_max = acc_max,
  };
}

ValidatedHistcBounds ValidateHistcOrThrow(const OpName op_name,
                                          const at::Tensor& self,
                                          const int64_t bins,
                                          const at::Scalar& min,
                                          const at::Scalar& max) {
  internal::ScopedOpName scoped_op_name(op_name);
  ScopedPythonContextCapturer capturer(op_name);
  try {
    TT_THROW_IF_ERROR(ValidateHistcInputTensor(self));
    TT_ASSIGN_OR_THROW(const ValidatedHistcBounds validated,
                       ValidateHistcBinsAndBounds(self, bins, min, max));
    return validated;
  } catch (const TtError& e) {
    TranslateToC10ErrorAndThrow(e);
  }
}

void ValidateHistcOutTensorOrThrow(const at::Tensor& self,
                                   const at::Tensor& out) {
  internal::ScopedOpName scoped_op_name(OpName::kHistcOut);
  ScopedPythonContextCapturer capturer(OpName::kHistcOut);
  try {
    TT_CHECK_THROW(IsPrivateUse1Device(out), error::kInvalidArgument)
        << "expected out tensor to be on " << GetPrivateUse1DeviceDebugName()
        << ", got " << out.device();
    TT_THROW_IF_ERROR(ValidateOutDtype(out, self.scalar_type(),
                                       /*allow_out_dtype_cast=*/false));
  } catch (const TtError& e) {
    TranslateToC10ErrorAndThrow(e);
  }
}

absl::StatusOr<HistcBoundTensors> ComputeHistcBoundsFromInput(
    const at::Tensor& self) {
  // Compute min/max from input data. This is done when min == max and
  // numel > 0, which happens when the min and max are not provided as
  // arguments to torch.histc and the input tensor is not empty. The
  // PyTorch CPU and GPU kernels differ from the PyTorch documentation,
  // which states that min/max are computed when min == max == 0. To
  // maintain parity, the TPU kernel implementation follows the logic of
  // the GPU/CPU kernels. See the PyTorch GPU kernel implementation for
  // more details:
  // https://github.com/pytorch/pytorch/blob/v2.9.1/aten/src/ATen/native/cuda/SummaryOps.cu#L328
  TT_ASSIGN_OR_RETURN(const auto mlir_input_dtype,
                      ConvertTo<mlir::ElementType>(self.scalar_type()));
  auto op_builder = [](mlir::MlirOp input) -> absl::StatusOr<MlirOpResults<2>> {
    TT_ASSIGN_OR_RETURN(auto min_max_outputs, BuildHistcBoundsShlo(input));
    return {{min_max_outputs.min, min_max_outputs.max}};
  };

  // Distinguish the bound computation from the main computation
  // in the dispatcher.
  const bool compute_bounds = true;
  TT_ASSIGN_OR_RETURN(auto compute_bounds_param_keys,
                      TT_MAKE_OP_PARAM_CACHE_KEYS(compute_bounds));

  TT_ASSIGN_OR_RETURN(
      (auto [min_buf, max_buf]),
      (DispatchOp<1, 2>(
          std::move(op_builder), self,
          {.out_dtypes = {mlir_input_dtype, mlir_input_dtype},
           .out_dims_list = {Dimensions(), Dimensions()},
           .op_param_cache_keys = std::move(compute_bounds_param_keys)})));
  at::Tensor min_val = MakeTensor(std::move(min_buf));
  at::Tensor max_val = MakeTensor(std::move(max_buf));

  // Validate computed bounds, as they may be invalid due to underflow
  // and overflow. Because this needs an expensive transfer to CPU, this
  // is only done when debug checks are enabled.
  if (GetEnableDebugChecks()) {
    TT_RETURN_IF_ERROR(ValidateBounds(
        min_val.cpu().item(), max_val.cpu().item(), BoundsMode::kFromInput));
  } else {
    TT_WARN_ONCE(
        WarningCategory::kCorrectness,
        "Bounds computed from input data are not validated by default on "
        "TorchTPU (unlike CUDA/CPU) to prevent an expensive CPU transfer. "
        "Set the TORCH_TPU_ENABLE_DEBUG_CHECKS environment variable to \"1\" "
        "to enable this check, which incurs a performance penalty.");
  }

  return HistcBoundTensors{
      .min_val = std::move(min_val),
      .max_val = std::move(max_val),
  };
}

auto MakeHistcOpBuilder(const int64_t bins) {
  return [bins](FixedSizeSpan<mlir::MlirOp, 3> inputs)
             -> absl::StatusOr<mlir::MlirOp> {
    auto& [input, min, max] = inputs;
    return BuildHistcShlo(input, bins, min, max);
  };
}

absl::StatusOr<at::Tensor> DispatchHistc(const at::Tensor& self,
                                         const int64_t bins,
                                         const at::Tensor& min_val,
                                         const at::Tensor& max_val,
                                         OpParamCacheKeys param_keys) {
  TT_ASSIGN_OR_RETURN(const auto mlir_out_dtype,
                      ConvertTo<mlir::ElementType>(self.scalar_type()));
  TT_ASSIGN_OR_RETURN(
      DeviceBufferRef out_buf,
      DispatchOp<3>(MakeHistcOpBuilder(bins), {self, min_val, max_val},
                    {.out_dtype = mlir_out_dtype,
                     .out_dims = {bins},
                     .op_param_cache_keys = std::move(param_keys)}));
  return MakeTensor(std::move(out_buf));
}

absl::Status DispatchHistcOut(const at::Tensor& self, const int64_t bins,
                              const at::Tensor& min_val,
                              const at::Tensor& max_val, at::Tensor& out,
                              OpParamCacheKeys param_keys) {
  TT_ASSIGN_OR_RETURN(const auto mlir_out_dtype,
                      ConvertTo<mlir::ElementType>(out.scalar_type()));
  return DispatchOpOut<3>(MakeHistcOpBuilder(bins), {self, min_val, max_val},
                          out,
                          {.out_dtype = mlir_out_dtype,
                           .out_dims = {bins},
                           .op_param_cache_keys = std::move(param_keys)});
}

at::Tensor HistcFromInput(const at::Tensor& self, const int64_t bins) {
  TT_KERNEL(OpName::kHistc, param_keys, (self, bins), {
    TT_ASSIGN_OR_THROW(const HistcBoundTensors bounds,
                       ComputeHistcBoundsFromInput(self));
    TT_ASSIGN_OR_THROW(at::Tensor out,
                       DispatchHistc(self, bins, bounds.min_val, bounds.max_val,
                                     std::move(param_keys)));
    return out;
  });
}

at::Tensor HistcExplicit(const at::Tensor& self, const int64_t bins,
                         const at::Scalar& acc_min, const at::Scalar& acc_max) {
  auto promoted_min = PromoteScalar(acc_min);
  auto promoted_max = PromoteScalar(acc_max);
  TT_KERNEL(
      OpName::kHistc, param_keys, (self, bins, promoted_min, promoted_max), {
        const at::ScalarType acc_dtype = ToAccumulateType(self.scalar_type());
        TT_ASSIGN_OR_THROW(const at::Tensor min_val,
                           promoted_min.GetTensor(acc_dtype));
        TT_ASSIGN_OR_THROW(const at::Tensor max_val,
                           promoted_max.GetTensor(acc_dtype));
        TT_ASSIGN_OR_THROW(
            at::Tensor out,
            DispatchHistc(self, bins, min_val, max_val, std::move(param_keys)));
        return out;
      });
}

at::Tensor& HistcOutFromInput(const at::Tensor& self, const int64_t bins,
                              at::Tensor& out) {
  TT_KERNEL(OpName::kHistcOut, param_keys, (self, bins, out), {
    TT_ASSIGN_OR_THROW(const HistcBoundTensors bounds,
                       ComputeHistcBoundsFromInput(self));
    TT_THROW_IF_ERROR(DispatchHistcOut(self, bins, bounds.min_val,
                                       bounds.max_val, out,
                                       std::move(param_keys)));
    return out;
  });
}

at::Tensor& HistcOutExplicit(const at::Tensor& self, const int64_t bins,
                             const at::Scalar& acc_min,
                             const at::Scalar& acc_max, at::Tensor& out) {
  auto promoted_min = PromoteScalar(acc_min);
  auto promoted_max = PromoteScalar(acc_max);
  TT_KERNEL(OpName::kHistcOut, param_keys,
            (self, bins, promoted_min, promoted_max, out), {
              const at::ScalarType acc_dtype =
                  ToAccumulateType(self.scalar_type());
              TT_ASSIGN_OR_THROW(const at::Tensor min_val,
                                 promoted_min.GetTensor(acc_dtype));
              TT_ASSIGN_OR_THROW(const at::Tensor max_val,
                                 promoted_max.GetTensor(acc_dtype));
              TT_THROW_IF_ERROR(DispatchHistcOut(self, bins, min_val, max_val,
                                                 out, std::move(param_keys)));
              return out;
            });
}

at::Tensor HistcEmpty(const at::Tensor& self, const int64_t bins) {
  TT_KERNEL(
      OpName::kHistc, _,
      (self, IgnoreInCacheKey(bins, "Delegates to AtenEfficientZeroTensor")), {
        return AtenEfficientZeroTensor(
            {bins}, self.scalar_type(), /*layout_opt=*/std::nullopt,
            self.device(), /*pin_memory_opt=*/std::nullopt);
      });
}

at::Tensor& HistcOutEmpty(const at::Tensor& self, const int64_t bins,
                          at::Tensor& out) {
  TT_KERNEL(OpName::kHistcOut, _,
            (self, IgnoreInCacheKey(bins, "Doesn't affect SHLO"), out), {
              TT_THROW_IF_ERROR(ResizeTensorIfShapeDiffers(out, {bins}));
              out.zero_();
              return out;
            });
}

}  // namespace

at::Tensor AtenHistc(const at::Tensor& self, const int64_t bins,
                     const at::Scalar& min, const at::Scalar& max) {
  const ValidatedHistcBounds validated =
      ValidateHistcOrThrow(OpName::kHistc, self, bins, min, max);
  if (self.numel() == 0) {
    return HistcEmpty(self, bins);
  }
  if (validated.mode == BoundsMode::kFromInput) {
    return HistcFromInput(self, bins);
  }
  return HistcExplicit(self, bins, validated.acc_min, validated.acc_max);
}

at::Tensor& AtenHistcOut(const at::Tensor& self, const int64_t bins,
                         const at::Scalar& min, const at::Scalar& max,
                         at::Tensor& out) {
  const ValidatedHistcBounds validated =
      ValidateHistcOrThrow(OpName::kHistcOut, self, bins, min, max);
  ValidateHistcOutTensorOrThrow(self, out);
  if (self.numel() == 0) {
    return HistcOutEmpty(self, bins, out);
  }
  if (validated.mode == BoundsMode::kFromInput) {
    return HistcOutFromInput(self, bins, out);
  }
  return HistcOutExplicit(self, bins, validated.acc_min, validated.acc_max,
                          out);
}

}  // namespace torch_tpu
