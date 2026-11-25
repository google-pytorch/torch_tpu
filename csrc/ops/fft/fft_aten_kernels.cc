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

#include "csrc/ops/fft/fft_aten_kernels.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string_view>
#include <utility>
#include <vector>

#include "ATen/core/ATen_fwd.h"
#include "ATen/core/TensorBody.h"
#include "ATen/ops/fft_fft_native.h"
#include "ATen/ops/fft_ifft_native.h"
#include "ATen/ops/fft_irfft_native.h"
#include "ATen/ops/fft_rfft_native.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/absl_check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "c10/core/ScalarType.h"
#include "csrc/common/aten_utils.h"
#include "csrc/common/dimension_types.h"
#include "csrc/common/dtype.h"
#include "csrc/common/error_utils.h"
#include "csrc/common/static_shape_check.h"
#include "csrc/common/to_string.h"
#include "csrc/common/utils.h"
#include "csrc/eager/device_buffer.h"
#include "csrc/eager/op_dispatcher.h"
#include "csrc/eager/tensor_to_buffer.h"
#include "csrc/ops/macros/kernel.h"
#include "csrc/ops/op_builder_utils.h"
#include "csrc/ops/op_names.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Support/LLVM.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"
#include "torch/headeronly/core/ScalarType.h"

namespace torch_tpu {
namespace {

// FftNormalization encapsulates PyTorch's three FFT scaling modes.
// PyTorch maps string options to integer modes:
// - "backward" -> 0: No scaling in forward transform,
//                  1/N in backward transform.
// - "ortho"    -> 1: 1/sqrt(N) scaling in both directions.
// - "forward"  -> 2: 1/N scaling in forward, no scaling in backward transform.
enum class FftNormalization {
  kNone = 0,
  kByRootN = 1,
  kByN = 2,
};

constexpr int64_t kMaxStablehloFftRank = 3;

enum class AllowEmptyDim {
  kNo,
  kYes,
};

// Normalizes the dimensions for an FFT operation, checking bounds and
// uniqueness. When `allow_empty` is `kYes`, an empty `dim` returns empty
// (meaning no axes are transformed).
absl::StatusOr<Dimensions> GetNormalizedDims(const at::Tensor& self,
                                             const at::IntArrayRef dim,
                                             const AllowEmptyDim allow_empty) {
  const int64_t num_dims = self.dim();
  if (dim.empty()) {
    TT_RET_CHECK(allow_empty == AllowEmptyDim::kYes, error::kInvalidArgument)
        << "expected at least 1 transform dimension, got 0";
    return Dimensions{};
  }
  // SafeWrapDim allows wrapping dim 0/-1 for 0D scalars, which is invalid for
  // FFT transforms because there is no axis to transform.
  TT_RET_CHECK(num_dims > 0, error::kPythonIndexError)
      << "expected input tensor to have at least 1 dimension, got 0D";

  Dimensions normalized_dims;
  normalized_dims.reserve(dim.size());
  absl::flat_hash_set<int64_t> seen_dims;
  seen_dims.reserve(dim.size());
  for (const int64_t d : dim) {
    TT_ASSIGN_OR_RETURN(const int64_t wrapped_d, SafeWrapDim(d, num_dims));
    TT_RET_CHECK(seen_dims.insert(wrapped_d).second, error::kInvalidArgument)
        << "dim " << wrapped_d << " appears multiple times in the list of dims";
    normalized_dims.push_back(wrapped_d);
  }

  return normalized_dims;
}

absl::StatusOr<at::ScalarType> GetFftR2cResultDtype(const at::Tensor& self) {
  const at::ScalarType scalar_type = self.scalar_type();
  TT_RET_CHECK(scalar_type == at::kHalf || scalar_type == at::kBFloat16 ||
                   scalar_type == at::kFloat || scalar_type == at::kDouble,
               error::kInvalidArgument)
      << "expected float16, bfloat16, float32, or float64 input type, got "
      << ToString(scalar_type);
  const at::ScalarType compute_scalar_type =
      scalar_type == at::kBFloat16 ? at::kFloat : scalar_type;
  return c10::toComplexType(compute_scalar_type);
}

absl::Status ValidateFftC2rLastDimSize(const at::Tensor& self,
                                       const int64_t last_dim,
                                       const int64_t last_dim_size) {
  TT_RET_CHECK(last_dim_size >= 1, error::kInvalidArgument)
      << "expected last_dim_size to be >= 1, got " << last_dim_size;
  const int64_t expected_size = last_dim_size / 2 + 1;
  const int64_t actual_size = self.size(last_dim);
  TT_RET_CHECK(actual_size == expected_size, error::kInvalidArgument)
      << "expected input size along dim " << last_dim << " to be "
      << expected_size << " for last_dim_size " << last_dim_size << ", got "
      << actual_size;
  return absl::OkStatus();
}

[[nodiscard]] Dimensions ComputeFftLengths(
    const mlir::RankedTensorType input_type,
    const absl::Span<const int64_t> dims,
    const std::optional<int64_t> last_dim_size = std::nullopt) {
  const mlir::ArrayRef<int64_t> input_shape = input_type.getShape();
  Dimensions fft_length;
  fft_length.reserve(dims.size());
  for (size_t i = 0; i < dims.size(); ++i) {
    const int64_t dim = dims[i];
    if (last_dim_size.has_value() && i == dims.size() - 1) {
      fft_length.push_back(*last_dim_size);
    } else {
      fft_length.push_back(input_shape[dim]);
    }
  }
  return fft_length;
}

// Applies a StableHLO FFT over the innermost `fft_length.size()` axes of
// `input`. StableHLO's FftOp verifier requires `1 <= fft_length.size() <= 3`,
// so higher-rank transforms are decomposed into chunks of at most 3 axes
// (matching JAX's `_fft_core_nd`), splitting 4D into 2D+2D to avoid 1D tails.
[[nodiscard]] mlir::MlirOp ApplyChunkedFft(
    mlir::MlirOp input, const mlir::stablehlo::FftType fft_type,
    const absl::Span<const int64_t> fft_length) {
  const int64_t fft_rank = static_cast<int64_t>(fft_length.size());
  ABSL_CHECK_GE(fft_rank, 1)  // CRASH_OK=Callers validate non-empty dims.
      << "ApplyChunkedFft requires at least 1 FFT dimension. This is a "
         "TorchTPU bug.";
  if (fft_rank <= kMaxStablehloFftRank) {
    return mlir::stablehlo::Fft(input, fft_type, AsArrayRef(fft_length));
  }

  const int64_t rank = GetTensorTypeOrDie(input).getRank();
  const int64_t chunk_rank = (fft_rank == 4) ? 2 : kMaxStablehloFftRank;
  const int64_t outer_rank = fft_rank - chunk_rank;
  const int64_t batch_rank = rank - fft_rank;

  // Rotate the last `fft_rank` axes so the `outer_rank` outer FFT axes become
  // innermost, and build the inverse rotation to restore axis order.
  Dimensions move_outer_to_back;
  move_outer_to_back.reserve(rank);
  Dimensions restore_axes;
  restore_axes.reserve(rank);
  for (int64_t i = 0; i < batch_rank; ++i) {
    move_outer_to_back.push_back(i);
    restore_axes.push_back(i);
  }
  for (int64_t i = rank - chunk_rank; i < rank; ++i) {
    move_outer_to_back.push_back(i);
  }
  for (int64_t i = batch_rank; i < rank - chunk_rank; ++i) {
    move_outer_to_back.push_back(i);
  }
  for (int64_t i = rank - outer_rank; i < rank; ++i) {
    restore_axes.push_back(i);
  }
  for (int64_t i = batch_rank; i < rank - outer_rank; ++i) {
    restore_axes.push_back(i);
  }

  const absl::Span<const int64_t> outer_lengths =
      fft_length.subspan(0, outer_rank);
  const absl::Span<const int64_t> inner_lengths =
      fft_length.subspan(outer_rank, chunk_rank);

  if (fft_type == mlir::stablehlo::FftType::FFT ||
      fft_type == mlir::stablehlo::FftType::RFFT) {
    // Transform the innermost `chunk_rank` axes first (handling real-to-complex
    // on the final axis for RFFT), then apply complex FFT to the outer axes.
    mlir::MlirOp result =
        mlir::stablehlo::Fft(input, fft_type, AsArrayRef(inner_lengths));
    result = mlir::stablehlo::Transpose(result, move_outer_to_back);
    result =
        ApplyChunkedFft(result, mlir::stablehlo::FftType::FFT, outer_lengths);
    return mlir::stablehlo::Transpose(result, restore_axes);
  }

  // For inverse / C2R transforms, apply complex IFFT to the outer axes while
  // the tensor is still complex, then transform the innermost `chunk_rank`
  // axes (handling complex-to-real on the final axis for IRFFT).
  mlir::MlirOp result = mlir::stablehlo::Transpose(input, move_outer_to_back);
  result =
      ApplyChunkedFft(result, mlir::stablehlo::FftType::IFFT, outer_lengths);
  result = mlir::stablehlo::Transpose(result, restore_axes);
  return mlir::stablehlo::Fft(result, fft_type, AsArrayRef(inner_lengths));
}

struct TransposePermutations {
  Dimensions permutation;
  Dimensions inverse_permutation;
  bool requires_transpose;
};

// Computes dimension permutations for PyTorch-to-StableHLO FFT mapping.
// StableHLO Fft strictly transforms innermost dimensions. To transform a custom
// set of dimensions in a specific order, we:
// 1. Place all non-transform dimensions first (preserving coordinate order).
// 2. Place all transform dimensions at the end in the exact order requested.
[[nodiscard]] TransposePermutations ComputeTransposePermutations(
    const int64_t num_dims, const absl::Span<const int64_t> dims) {
  std::vector<bool> is_fft_dim(num_dims, false);
  for (int64_t d : dims) {
    is_fft_dim[d] = true;
  }

  Dimensions p_vec;
  p_vec.reserve(num_dims);
  for (int i = 0; i < num_dims; ++i) {
    if (!is_fft_dim[i]) {
      p_vec.push_back(i);
    }
  }
  for (int64_t d : dims) {
    p_vec.push_back(d);
  }

  bool requires_transpose = false;
  for (int i = 0; i < num_dims; ++i) {
    if (p_vec[i] != i) {
      requires_transpose = true;
      break;
    }
  }

  Dimensions inv_p_vec;
  if (requires_transpose) {
    inv_p_vec.resize(num_dims);
    for (int i = 0; i < num_dims; ++i) {
      inv_p_vec[p_vec[i]] = i;
    }
  }

  return {std::move(p_vec), std::move(inv_p_vec), requires_transpose};
}

enum class ScaleMode {
  kNone = 0,
  kDivSqrtN = 1,  // 1 / sqrt(N)
  kDivN = 2,      // 1 / N
  kMulSqrtN = 3,  // sqrt(N)
  kMulN = 4,      // N
};

absl::StatusOr<mlir::MlirOp> ComputeScaleOp(mlir::MlirBuilder& builder,
                                            mlir::MlirOp num_elements_op,
                                            mlir::Type float_type,
                                            ScaleMode scale_mode) {
  auto one_op = MakeScalarConstant(builder, 1.0, float_type);
  switch (scale_mode) {
    case ScaleMode::kNone:
      return one_op;
    case ScaleMode::kDivSqrtN: {
      auto sqrt_op = mlir::stablehlo::Sqrt(num_elements_op);
      return mlir::stablehlo::Div(one_op, sqrt_op);
    }
    case ScaleMode::kDivN:
      return mlir::stablehlo::Div(one_op, num_elements_op);
    case ScaleMode::kMulSqrtN:
      return mlir::stablehlo::Sqrt(num_elements_op);
    case ScaleMode::kMulN:
      return num_elements_op;
  }
}

absl::StatusOr<mlir::MlirOp> ApplyComplexScaling(mlir::MlirOp fft_op,
                                                 mlir::MlirOp scale_op,
                                                 mlir::Type float_type) {
  TT_ASSIGN_OR_RETURN(auto real_op, BroadcastIfNeeded(scale_op, fft_op));
  auto imag_op = MakeConstantLike(fft_op, 0.0, float_type);
  auto complex_op = mlir::stablehlo::Complex(real_op, imag_op);
  return mlir::stablehlo::Mul(fft_op, complex_op);
}

absl::StatusOr<mlir::MlirOp> BuildFftR2cShlo(
    mlir::MlirOp input, const absl::Span<const int64_t> dims,
    const FftNormalization normalization, const bool onesided) {
  mlir::RankedTensorType input_type = GetTensorTypeOrDie(input);

  // StableHLO RFFT on f16/bf16 produces complex<f16>/complex<bf16>, which XLA
  // TPU does not support; promote to f32 before transforming.
  if (input_type.getElementType().isBF16() ||
      input_type.getElementType().isF16()) {
    mlir::Builder builder(input_type.getContext());
    input = mlir::stablehlo::ConvertElementType(input, builder.getF32Type());
    input_type = GetTensorTypeOrDie(input);
  }

  const mlir::Type float_type = input_type.getElementType();

  const Dimensions fft_length = ComputeFftLengths(input_type, dims);
  const TransposePermutations transpose_perms =
      ComputeTransposePermutations(input_type.getRank(), dims);

  mlir::MlirOp fft_input = input;
  if (transpose_perms.requires_transpose) {
    fft_input = mlir::stablehlo::Transpose(input, transpose_perms.permutation);
  }

  // Perform the FFT operation.
  mlir::MlirOp fft_op;
  if (onesided) {
    fft_op =
        ApplyChunkedFft(fft_input, mlir::stablehlo::FftType::RFFT, fft_length);
  } else {
    mlir::MlirOp zero_imag = MakeConstantLike(fft_input, 0.0, float_type);
    mlir::MlirOp complex_input = mlir::stablehlo::Complex(fft_input, zero_imag);
    fft_op = ApplyChunkedFft(complex_input, mlir::stablehlo::FftType::FFT,
                             fft_length);
  }

  // Apply normalization: stablehlo::Fft does not support normalization modes,
  // so we scale the result manually based on the 'normalization' parameter
  // to match PyTorch's behavior.
  ScaleMode scale_mode = ScaleMode::kNone;
  if (normalization == FftNormalization::kByN) {
    scale_mode = ScaleMode::kDivN;
  } else if (normalization == FftNormalization::kByRootN) {
    scale_mode = ScaleMode::kDivSqrtN;
  }

  if (scale_mode != ScaleMode::kNone) {
    mlir::MlirOp num_elements_op =
        GetNumElements(input, float_type, AsArrayRef(dims));
    TT_ASSIGN_OR_RETURN(mlir::MlirOp scale_op,
                        ComputeScaleOp(fft_op.getBuilder(), num_elements_op,
                                       float_type, scale_mode));

    TT_ASSIGN_OR_RETURN(fft_op,
                        ApplyComplexScaling(fft_op, scale_op, float_type));
  }

  if (transpose_perms.requires_transpose) {
    fft_op =
        mlir::stablehlo::Transpose(fft_op, transpose_perms.inverse_permutation);
  }

  return fft_op;
}

absl::StatusOr<mlir::MlirOp> BuildFftC2cShlo(
    mlir::MlirOp input, const absl::Span<const int64_t> dims,
    const FftNormalization normalization, const bool forward) {
  const mlir::RankedTensorType input_type = GetTensorTypeOrDie(input);

  const mlir::Type elem_type = input_type.getElementType();
  if (!mlir::isa<mlir::ComplexType>(elem_type)) {
    TT_ASSIGN_OR_RETURN(const at::ScalarType scalar_type,
                        ConvertTo<at::ScalarType>(elem_type));
    return TT_ERROR(  // ERROR_COV_INFEASIBLE=Checked by caller before
                      // DispatchOp.
               error::kInvalidArgument)
           << "expected complex input type, got " << ToString(scalar_type);
  }
  if (dims.empty()) {
    return input;
  }
  const mlir::Type float_type =
      mlir::cast<mlir::ComplexType>(elem_type).getElementType();

  const mlir::stablehlo::FftType fft_type =
      forward ? mlir::stablehlo::FftType::FFT : mlir::stablehlo::FftType::IFFT;

  // StableHLO's FftOp does not support normalization modes, so we scale the
  // result manually based on the 'normalization' and 'forward' parameters to
  // match PyTorch's behavior.
  // Stablehlo's ifft precisely inverses the fft operation, which means that it
  // includes 1/N scaling in the backward transform. Thus, we have to apply
  // scaling in the opposite direction to obtain the same result as PyTorch
  // expects.
  ScaleMode scale_mode = ScaleMode::kNone;
  if (forward) {
    if (normalization == FftNormalization::kByRootN) {
      scale_mode = ScaleMode::kDivSqrtN;
    } else if (normalization == FftNormalization::kByN) {
      scale_mode = ScaleMode::kDivN;
    }
  } else {
    if (normalization == FftNormalization::kNone) {
      scale_mode = ScaleMode::kMulN;
    } else if (normalization == FftNormalization::kByRootN) {
      scale_mode = ScaleMode::kMulSqrtN;
    }
  }

  const Dimensions fft_length = ComputeFftLengths(input_type, dims);
  const TransposePermutations transpose_perms =
      ComputeTransposePermutations(input_type.getRank(), dims);

  mlir::MlirOp fft_input = input;
  if (transpose_perms.requires_transpose) {
    fft_input = mlir::stablehlo::Transpose(input, transpose_perms.permutation);
  }

  // Perform the FFT operation.
  mlir::MlirOp fft_op = ApplyChunkedFft(fft_input, fft_type, fft_length);

  // Apply scaling manually.
  if (scale_mode != ScaleMode::kNone) {
    mlir::MlirOp num_elements_op =
        GetNumElements(input, float_type, AsArrayRef(dims));
    TT_ASSIGN_OR_RETURN(mlir::MlirOp scale_op,
                        ComputeScaleOp(fft_op.getBuilder(), num_elements_op,
                                       float_type, scale_mode));

    TT_ASSIGN_OR_RETURN(fft_op,
                        ApplyComplexScaling(fft_op, scale_op, float_type));
  }

  if (transpose_perms.requires_transpose) {
    fft_op =
        mlir::stablehlo::Transpose(fft_op, transpose_perms.inverse_permutation);
  }

  return fft_op;
}

absl::StatusOr<mlir::MlirOp> BuildFftC2rShlo(
    mlir::MlirOp input, const absl::Span<const int64_t> dims,
    const FftNormalization normalization, const int64_t last_dim_size) {
  const mlir::RankedTensorType input_type = GetTensorTypeOrDie(input);

  const mlir::Type elem_type = input_type.getElementType();
  if (!mlir::isa<mlir::ComplexType>(elem_type)) {
    TT_ASSIGN_OR_RETURN(const at::ScalarType scalar_type,
                        ConvertTo<at::ScalarType>(elem_type));
    return TT_ERROR(  // ERROR_COV_INFEASIBLE=Checked by caller before
                      // DispatchOp.
               error::kInvalidArgument)
           << "expected complex input type, got " << ToString(scalar_type);
  }
  const mlir::Type float_type =
      mlir::cast<mlir::ComplexType>(elem_type).getElementType();

  const mlir::stablehlo::FftType fft_type = mlir::stablehlo::FftType::IRFFT;

  ScaleMode scale_mode = ScaleMode::kNone;
  if (normalization == FftNormalization::kNone) {
    scale_mode = ScaleMode::kMulN;
  } else if (normalization == FftNormalization::kByRootN) {
    scale_mode = ScaleMode::kMulSqrtN;
  }

  const Dimensions fft_length =
      ComputeFftLengths(input_type, dims, last_dim_size);
  const TransposePermutations transpose_perms =
      ComputeTransposePermutations(input_type.getRank(), dims);

  mlir::MlirOp fft_input = input;
  if (transpose_perms.requires_transpose) {
    fft_input = mlir::stablehlo::Transpose(input, transpose_perms.permutation);
  }

  // Perform the FFT operation.
  mlir::MlirOp fft_op = ApplyChunkedFft(fft_input, fft_type, fft_length);

  // Apply scaling manually.
  // NOTE: BuildFftC2rShlo scales based on raw transform size
  // (fft_length product) rather than input tensor sizes, because for
  // real-to-complex boundary, input shape has size N/2 + 1 but transform
  // has logical size N.
  if (scale_mode != ScaleMode::kNone) {
    // Compute the number of elements in the FFT dimensions (statically).
    int64_t num_elements = 1;
    for (const int64_t len : fft_length) {
      num_elements *= len;
    }
    mlir::MlirOp num_elements_op = MakeScalarConstant(
        input.getBuilder(), static_cast<double>(num_elements), float_type);

    TT_ASSIGN_OR_RETURN(mlir::MlirOp scale_op,
                        ComputeScaleOp(fft_op.getBuilder(), num_elements_op,
                                       float_type, scale_mode));

    TT_ASSIGN_OR_RETURN(mlir::MlirOp scaled_op,
                        BroadcastIfNeeded(scale_op, fft_op));
    fft_op = mlir::stablehlo::Mul(fft_op, scaled_op);
  }

  if (transpose_perms.requires_transpose) {
    fft_op =
        mlir::stablehlo::Transpose(fft_op, transpose_perms.inverse_permutation);
  }

  return fft_op;
}

absl::Status FFTValidateStaticShape(const at::Tensor& tensor,
                                    const std::string_view arg_name) {
  TT_ASSIGN_OR_RETURN(DeviceBufferRef buffer_ref, GetBaseBuffer(tensor));
  return ValidateStaticShape(tensor, buffer_ref, arg_name);
}

[[nodiscard]] at::Tensor PromoteToFloatIfBFloat16(const at::Tensor& self) {
  return self.scalar_type() == at::kBFloat16 ? self.to(at::kFloat) : self;
}

}  // namespace

at::Tensor AtenFftR2c(const at::Tensor& self, const at::IntArrayRef dim,
                      const int64_t normalization, const bool onesided) {
  TT_KERNEL(OpName::kFftR2c, param_keys, (self, dim, normalization, onesided), {
    const FftNormalization norm_enum =
        static_cast<FftNormalization>(normalization);

    // Explicitly check for static shape.
    // Shape introspection is unsafe with dynamism.
    // NOTE: This implementation is not dynamism safe. It currently
    // uses static shape indices, more work is needed to determine how
    // this op supports bounded dynamic values.
    TT_THROW_IF_ERROR(FFTValidateStaticShape(self, "input"));

    TT_ASSIGN_OR_THROW(const at::ScalarType out_scalar_type,
                       GetFftR2cResultDtype(self));
    TT_ASSIGN_OR_THROW(const mlir::ElementType output_dtype,
                       ConvertTo<mlir::ElementType>(out_scalar_type));

    TT_ASSIGN_OR_THROW(Dimensions normalized_dims,
                       GetNormalizedDims(self, dim, AllowEmptyDim::kNo));
    Dimensions out_sizes = CopyIntVector(self.sizes());
    if (onesided) {
      const int64_t last_dim = normalized_dims.back();
      out_sizes[last_dim] = self.size(last_dim) / 2 + 1;
    }

    auto op_builder =
        [normalized_dims = std::move(normalized_dims), norm_enum,
         onesided](mlir::MlirOp input) -> absl::StatusOr<mlir::MlirOp> {
      return BuildFftR2cShlo(input, normalized_dims, norm_enum, onesided);
    };

    TT_ASSIGN_OR_THROW(
        DeviceBufferRef result,
        DispatchOp<1>(std::move(op_builder), self,
                      {.out_dtype = output_dtype,
                       .out_dims = out_sizes,
                       .op_param_cache_keys = std::move(param_keys)}));

    TT_ASSIGN_OR_THROW(
        at::Tensor out,
        MakeEmptyTensor(out_sizes, out_scalar_type, self.device()));
    TT_THROW_IF_ERROR(AssignBufferToAtTensor(std::move(result), out));
    return out;
  });
}

at::Tensor& AtenFftR2cOut(const at::Tensor& self, const at::IntArrayRef dim,
                          const int64_t normalization, const bool onesided,
                          at::Tensor& out) {
  TT_KERNEL(
      OpName::kFftR2cOut, param_keys, (self, dim, normalization, onesided, out),
      {
        const FftNormalization norm_enum =
            static_cast<FftNormalization>(normalization);

        // Explicitly check for static shape.
        // Shape introspection is unsafe with dynamism.
        // NOTE: This implementation is not dynamism safe. It currently uses
        // static shape indices, more work is needed to determine how this op
        // supports bounded dynamic values.
        TT_THROW_IF_ERROR(FFTValidateStaticShape(self, "input"));

        TT_ASSIGN_OR_THROW(const at::ScalarType result_dtype,
                           GetFftR2cResultDtype(self));
        TT_THROW_IF_ERROR(ValidateOutDtype(out, result_dtype,
                                           /*allow_out_dtype_cast=*/true));
        TT_ASSIGN_OR_THROW(const mlir::ElementType output_dtype,
                           ConvertTo<mlir::ElementType>(out.scalar_type()));

        TT_ASSIGN_OR_THROW(Dimensions normalized_dims,
                           GetNormalizedDims(self, dim, AllowEmptyDim::kNo));
        Dimensions out_sizes = CopyIntVector(self.sizes());
        if (onesided) {
          const int64_t last_dim = normalized_dims.back();
          out_sizes[last_dim] = self.size(last_dim) / 2 + 1;
        }

        auto op_builder =
            [normalized_dims = std::move(normalized_dims), norm_enum,
             onesided](mlir::MlirOp input) -> absl::StatusOr<mlir::MlirOp> {
          return BuildFftR2cShlo(input, normalized_dims, norm_enum, onesided);
        };

        TT_THROW_IF_ERROR(
            DispatchOpOut<1>(std::move(op_builder), self, out,
                             {.out_dtype = output_dtype,
                              .out_dims = out_sizes,
                              .op_param_cache_keys = std::move(param_keys)}));
        return out;
      });
}

at::Tensor AtenFftC2c(const at::Tensor& self, const at::IntArrayRef dim,
                      const int64_t normalization, const bool forward) {
  TT_KERNEL(OpName::kFftC2c, param_keys, (self, dim, normalization, forward), {
    const FftNormalization norm_enum =
        static_cast<FftNormalization>(normalization);

    // Explicitly check for static shape.
    // Shape introspection is unsafe with dynamism.
    // NOTE: This implementation is not dynamism safe. It currently uses
    // static shape indices, more work is needed to determine how this op
    // supports bounded dynamic values.
    TT_THROW_IF_ERROR(FFTValidateStaticShape(self, "input"));

    TT_ASSIGN_OR_THROW(const mlir::ElementType output_dtype,
                       ConvertTo<mlir::ElementType>(self.scalar_type()));
    TT_CHECK_THROW(self.is_complex(), error::kInvalidArgument)
        << "expected complex input type, got " << ToString(self.scalar_type());

    TT_ASSIGN_OR_THROW(Dimensions normalized_dims,
                       GetNormalizedDims(self, dim, AllowEmptyDim::kYes));

    auto op_builder =
        [normalized_dims = std::move(normalized_dims), norm_enum,
         forward](mlir::MlirOp input) -> absl::StatusOr<mlir::MlirOp> {
      return BuildFftC2cShlo(input, normalized_dims, norm_enum, forward);
    };

    TT_ASSIGN_OR_THROW(
        DeviceBufferRef result,
        DispatchOp<1>(std::move(op_builder), self,
                      {.out_dtype = output_dtype,
                       .out_dims = CopyIntVector(self.sizes()),
                       .op_param_cache_keys = std::move(param_keys)}));

    TT_ASSIGN_OR_THROW(
        at::Tensor out,
        MakeEmptyTensor(self.sizes(), self.scalar_type(), self.device()));
    TT_THROW_IF_ERROR(AssignBufferToAtTensor(std::move(result), out));
    return out;
  });
}

at::Tensor& AtenFftC2cOut(const at::Tensor& self, const at::IntArrayRef dim,
                          const int64_t normalization, const bool forward,
                          at::Tensor& out) {
  TT_KERNEL(
      OpName::kFftC2cOut, param_keys, (self, dim, normalization, forward, out),
      {
        const FftNormalization norm_enum =
            static_cast<FftNormalization>(normalization);

        // Explicitly check for static shape.
        // Shape introspection is unsafe with dynamism.
        TT_THROW_IF_ERROR(FFTValidateStaticShape(self, "input"));
        TT_THROW_IF_ERROR(ValidateOutDtype(out, self.scalar_type(),
                                           /*allow_out_dtype_cast=*/true));

        TT_ASSIGN_OR_THROW(const mlir::ElementType output_dtype,
                           ConvertTo<mlir::ElementType>(out.scalar_type()));
        TT_CHECK_THROW(  // ERROR_COV_INFEASIBLE=torch.aten.ops._fft_c2c_out
                         // does not exist
            self.is_complex(), error::kInvalidArgument)
            << "expected complex input type, got "
            << ToString(self.scalar_type());

        TT_ASSIGN_OR_THROW(Dimensions normalized_dims,
                           GetNormalizedDims(self, dim, AllowEmptyDim::kYes));

        auto op_builder =
            [normalized_dims = std::move(normalized_dims), norm_enum,
             forward](mlir::MlirOp input) -> absl::StatusOr<mlir::MlirOp> {
          return BuildFftC2cShlo(input, normalized_dims, norm_enum, forward);
        };

        TT_THROW_IF_ERROR(
            DispatchOpOut<1>(std::move(op_builder), self, out,
                             {.out_dtype = output_dtype,
                              .out_dims = self.sizes(),
                              .op_param_cache_keys = std::move(param_keys)}));
        return out;
      });
}

at::Tensor AtenFftC2r(const at::Tensor& self, const at::IntArrayRef dim,
                      const int64_t normalization,
                      const int64_t last_dim_size) {
  TT_KERNEL(
      OpName::kFftC2r, param_keys, (self, dim, normalization, last_dim_size), {
        const FftNormalization norm_enum =
            static_cast<FftNormalization>(normalization);

        // Explicitly check for static shape.
        // Shape introspection is unsafe with dynamism.
        // NOTE: This implementation is not dynamism safe. It currently uses
        // static shape indices, more work is needed to determine how this op
        // supports bounded dynamic values.
        TT_THROW_IF_ERROR(FFTValidateStaticShape(self, "input"));

        TT_CHECK_THROW(self.is_complex(), error::kInvalidArgument)
            << "expected complex input type, got "
            << ToString(self.scalar_type());
        TT_ASSIGN_OR_THROW(const mlir::ElementType output_dtype,
                           ConvertTo<mlir::ElementType>(
                               c10::toRealValueType(self.scalar_type())));

        TT_ASSIGN_OR_THROW(Dimensions normalized_dims,
                           GetNormalizedDims(self, dim, AllowEmptyDim::kNo));
        const int64_t last_dim = normalized_dims.back();
        TT_THROW_IF_ERROR(
            ValidateFftC2rLastDimSize(self, last_dim, last_dim_size));
        Dimensions out_sizes = CopyIntVector(self.sizes());
        out_sizes[last_dim] = last_dim_size;

        auto op_builder =
            [normalized_dims = std::move(normalized_dims), norm_enum,
             last_dim_size](
                mlir::MlirOp input) -> absl::StatusOr<mlir::MlirOp> {
          return BuildFftC2rShlo(input, normalized_dims, norm_enum,
                                 last_dim_size);
        };

        TT_ASSIGN_OR_THROW(
            DeviceBufferRef result,
            DispatchOp<1>(std::move(op_builder), self,
                          {.out_dtype = output_dtype,
                           .out_dims = out_sizes,
                           .op_param_cache_keys = std::move(param_keys)}));

        TT_ASSIGN_OR_THROW(
            at::Tensor out,
            MakeEmptyTensor(out_sizes, c10::toRealValueType(self.scalar_type()),
                            self.device()));
        TT_THROW_IF_ERROR(AssignBufferToAtTensor(std::move(result), out));
        return out;
      });
}

at::Tensor& AtenFftC2rOut(const at::Tensor& self, const at::IntArrayRef dim,
                          const int64_t normalization,
                          const int64_t last_dim_size, at::Tensor& out) {
  TT_KERNEL(
      OpName::kFftC2rOut, param_keys,
      (self, dim, normalization, last_dim_size, out), {
        const FftNormalization norm_enum =
            static_cast<FftNormalization>(normalization);

        // Explicitly check for static shape.
        // Shape introspection is unsafe with dynamism.
        // NOTE: This implementation is not dynamism safe. It currently uses
        // static shape indices, more work is needed to determine how this op
        // supports bounded dynamic values.
        TT_THROW_IF_ERROR(FFTValidateStaticShape(self, "input"));

        const at::ScalarType result_dtype =
            c10::toRealValueType(self.scalar_type());
        TT_THROW_IF_ERROR(ValidateOutDtype(out, result_dtype,
                                           /*allow_out_dtype_cast=*/true));
        TT_ASSIGN_OR_THROW(const mlir::ElementType output_dtype,
                           ConvertTo<mlir::ElementType>(out.scalar_type()));
        TT_CHECK_THROW(  // ERROR_COV_INFEASIBLE=torch.aten.ops._fft_c2r_out
                         // does not exist
            self.is_complex(), error::kInvalidArgument)
            << "expected complex input type, got "
            << ToString(self.scalar_type());

        TT_ASSIGN_OR_THROW(Dimensions normalized_dims,
                           GetNormalizedDims(self, dim, AllowEmptyDim::kNo));
        const int64_t last_dim = normalized_dims.back();
        TT_THROW_IF_ERROR(
            ValidateFftC2rLastDimSize(self, last_dim, last_dim_size));
        Dimensions out_sizes = CopyIntVector(self.sizes());
        out_sizes[last_dim] = last_dim_size;

        auto op_builder =
            [normalized_dims = std::move(normalized_dims), norm_enum,
             last_dim_size](
                mlir::MlirOp input) -> absl::StatusOr<mlir::MlirOp> {
          return BuildFftC2rShlo(input, normalized_dims, norm_enum,
                                 last_dim_size);
        };

        TT_THROW_IF_ERROR(
            DispatchOpOut<1>(std::move(op_builder), self, out,
                             {.out_dtype = output_dtype,
                              .out_dims = out_sizes,
                              .op_param_cache_keys = std::move(param_keys)}));
        return out;
      });
}

at::Tensor AtenFftFft(const at::Tensor& self, std::optional<c10::SymInt> n,
                      int64_t dim, std::optional<std::string_view> norm) {
  return at::native::fft_fft_symint(PromoteToFloatIfBFloat16(self), n, dim,
                                    norm);
}

at::Tensor& AtenFftFftOut(const at::Tensor& self, std::optional<c10::SymInt> n,
                          int64_t dim, std::optional<std::string_view> norm,
                          at::Tensor& out) {
  return at::native::fft_fft_symint_out(PromoteToFloatIfBFloat16(self), n, dim,
                                        norm, out);
}

at::Tensor AtenFftIfft(const at::Tensor& self, std::optional<c10::SymInt> n,
                       int64_t dim, std::optional<std::string_view> norm) {
  return at::native::fft_ifft_symint(PromoteToFloatIfBFloat16(self), n, dim,
                                     norm);
}

at::Tensor& AtenFftIfftOut(const at::Tensor& self, std::optional<c10::SymInt> n,
                           int64_t dim, std::optional<std::string_view> norm,
                           at::Tensor& out) {
  return at::native::fft_ifft_symint_out(PromoteToFloatIfBFloat16(self), n, dim,
                                         norm, out);
}

at::Tensor AtenFftRfft(const at::Tensor& self, std::optional<c10::SymInt> n,
                       int64_t dim, std::optional<std::string_view> norm) {
  return at::native::fft_rfft_symint(PromoteToFloatIfBFloat16(self), n, dim,
                                     norm);
}

at::Tensor& AtenFftRfftOut(const at::Tensor& self, std::optional<c10::SymInt> n,
                           int64_t dim, std::optional<std::string_view> norm,
                           at::Tensor& out) {
  return at::native::fft_rfft_symint_out(PromoteToFloatIfBFloat16(self), n, dim,
                                         norm, out);
}

at::Tensor AtenFftIrfft(const at::Tensor& self, std::optional<c10::SymInt> n,
                        int64_t dim, std::optional<std::string_view> norm) {
  return at::native::fft_irfft_symint(PromoteToFloatIfBFloat16(self), n, dim,
                                      norm);
}

at::Tensor& AtenFftIrfftOut(const at::Tensor& self,
                            std::optional<c10::SymInt> n, int64_t dim,
                            std::optional<std::string_view> norm,
                            at::Tensor& out) {
  return at::native::fft_irfft_symint_out(PromoteToFloatIfBFloat16(self), n,
                                          dim, norm, out);
}

}  // namespace torch_tpu
