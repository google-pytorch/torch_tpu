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

#include "torch_tpu/ops/fft/fft_aten_kernels.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string_view>
#include <utility>
#include <vector>

#include "ATen/core/ATen_fwd.h"
#include "ATen/core/TensorBody.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "c10/core/ScalarType.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Support/LLVM.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"
#include "torch/headeronly/core/ScalarType.h"
#include "torch_tpu/common/dimension_types.h"
#include "torch_tpu/common/dtype.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/common/static_shape_check.h"
#include "torch_tpu/common/to_string.h"
#include "torch_tpu/common/utils.h"
#include "torch_tpu/eager/device_buffer.h"
#include "torch_tpu/eager/op_dispatcher.h"
#include "torch_tpu/eager/tensor_to_buffer.h"
#include "torch_tpu/ops/macros/kernel.h"
#include "torch_tpu/ops/nullary_aten_kernels.h"
#include "torch_tpu/ops/op_builder_utils.h"
#include "torch_tpu/ops/op_names.h"
#include "torch_tpu/ops/resize/resize_aten_kernels.h"

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

// Normalizes the dimensions for the FFT operation
absl::StatusOr<Dimensions> GetNormalizedDims(const at::Tensor& self,
                                             at::IntArrayRef dim) {
  const int64_t num_dims = self.dim();
  Dimensions dims_vec = CopyIntVector(dim);

  // If no dimensions are specified, perform FFT on all dimensions
  if (dims_vec.empty()) {
    dims_vec.reserve(num_dims);
    for (int i = 0; i < num_dims; ++i) {
      dims_vec.push_back(i);
    }
  }

  // Normalize the dimensions to be positive and perform bounds-checking
  Dimensions normalized_dims;
  normalized_dims.reserve(dims_vec.size());
  for (int64_t d : dims_vec) {
    TT_ASSIGN_OR_RETURN(int64_t wrapped_d, SafeWrapDim(d, num_dims));
    normalized_dims.push_back(wrapped_d);
  }

  return normalized_dims;
}

struct FftParams {
  Dimensions fft_length;
  std::vector<bool> is_fft_dim;
};

FftParams ComputeFftParams(
    mlir::RankedTensorType input_type, Dimensions dims,
    std::optional<int64_t> last_dim_size = std::nullopt) {
  auto input_shape = input_type.getShape();
  const int64_t num_dims = input_type.getRank();

  Dimensions fft_length;
  fft_length.reserve(dims.size());
  for (size_t i = 0; i < dims.size(); ++i) {
    int64_t dim = dims[i];
    if (last_dim_size.has_value() && i == dims.size() - 1) {
      fft_length.push_back(*last_dim_size);
    } else {
      fft_length.push_back(input_shape[dim]);
    }
  }

  std::vector<bool> is_fft_dim(num_dims, false);
  for (int64_t dim : dims) {
    is_fft_dim[dim] = true;
  }

  return {std::move(fft_length), std::move(is_fft_dim)};
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
TransposePermutations ComputeTransposePermutations(int64_t num_dims,
                                                   const Dimensions& dims) {
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
    mlir::MlirOp input, Dimensions dims, const FftNormalization normalization,
    bool onesided) {
  auto input_type = GetTensorTypeOrDie(input);

  const auto float_type = input_type.getElementType();

  auto fft_params = ComputeFftParams(input_type, dims);
  auto transpose_perms =
      ComputeTransposePermutations(input_type.getRank(), dims);

  mlir::MlirOp fft_input = input;
  if (transpose_perms.requires_transpose) {
    fft_input = mlir::stablehlo::Transpose(input, transpose_perms.permutation);
  }

  // Perform the FFT operation
  mlir::MlirOp fft_op;
  if (onesided) {
    fft_op = mlir::stablehlo::Fft(fft_input, mlir::stablehlo::FftType::RFFT,
                                  fft_params.fft_length);
  } else {
    auto zero_imag = MakeConstantLike(fft_input, 0.0, float_type);
    auto complex_input = mlir::stablehlo::Complex(fft_input, zero_imag);
    fft_op = mlir::stablehlo::Fft(complex_input, mlir::stablehlo::FftType::FFT,
                                  fft_params.fft_length);
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
    mlir::MlirOp num_elements_op = GetNumElements(input, float_type, dims);
    TT_ASSIGN_OR_RETURN(auto scale_op,
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
    mlir::MlirOp input, Dimensions dims, const FftNormalization normalization,
    bool forward) {
  auto input_type = GetTensorTypeOrDie(input);

  auto elem_type = input_type.getElementType();
  if (!mlir::isa<mlir::ComplexType>(elem_type)) {
    TT_ASSIGN_OR_RETURN(auto scalar_type, ConvertTo<at::ScalarType>(elem_type));
    return TT_ERROR(error::kInvalidArgument)
           << "expected complex input type, got " << ToString(scalar_type);
  }
  auto float_type = mlir::cast<mlir::ComplexType>(elem_type).getElementType();

  mlir::stablehlo::FftType fft_type =
      forward ? mlir::stablehlo::FftType::FFT : mlir::stablehlo::FftType::IFFT;

  // StableHLO's FftOp does not support normalization modes, so we scale the
  // result manually based on the 'normalization' and 'forward' parameters to
  // match PyTorch's behavior.
  // Stablehlo's ifft precisely inverses the fft operation, which means that it
  // includes 1/N scaling in the backward transform. Thus, we ha ve to apply
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

  auto fft_params = ComputeFftParams(input_type, dims);
  auto transpose_perms =
      ComputeTransposePermutations(input_type.getRank(), dims);

  mlir::MlirOp fft_input = input;
  if (transpose_perms.requires_transpose) {
    fft_input = mlir::stablehlo::Transpose(input, transpose_perms.permutation);
  }

  // Perform the FFT operation
  auto fft_op =
      mlir::stablehlo::Fft(fft_input, fft_type, fft_params.fft_length);

  // Apply scaling manually
  if (scale_mode != ScaleMode::kNone) {
    mlir::MlirOp num_elements_op = GetNumElements(input, float_type, dims);
    TT_ASSIGN_OR_RETURN(auto scale_op,
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
    mlir::MlirOp input, Dimensions dims, const FftNormalization normalization,
    int64_t last_dim_size) {
  auto input_type = GetTensorTypeOrDie(input);

  auto elem_type = input_type.getElementType();
  if (!mlir::isa<mlir::ComplexType>(elem_type)) {
    TT_ASSIGN_OR_RETURN(auto scalar_type, ConvertTo<at::ScalarType>(elem_type));
    return TT_ERROR(error::kInvalidArgument)
           << "expected complex input type, got " << ToString(scalar_type);
  }
  auto float_type = mlir::cast<mlir::ComplexType>(elem_type).getElementType();

  mlir::stablehlo::FftType fft_type = mlir::stablehlo::FftType::IRFFT;

  ScaleMode scale_mode = ScaleMode::kNone;
  if (normalization == FftNormalization::kNone) {
    scale_mode = ScaleMode::kMulN;
  } else if (normalization == FftNormalization::kByRootN) {
    scale_mode = ScaleMode::kMulSqrtN;
  }

  auto fft_params = ComputeFftParams(input_type, dims, last_dim_size);
  auto transpose_perms =
      ComputeTransposePermutations(input_type.getRank(), dims);

  mlir::MlirOp fft_input = input;
  if (transpose_perms.requires_transpose) {
    fft_input = mlir::stablehlo::Transpose(input, transpose_perms.permutation);
  }

  // Perform the FFT operation
  auto fft_op =
      mlir::stablehlo::Fft(fft_input, fft_type, fft_params.fft_length);

  // Apply scaling manually.
  // NOTE: BuildFftC2rShlo scales based on raw transform size
  // (fft_length product) rather than input tensor sizes, because for
  // real-to-complex boundary, input shape has size N/2 + 1 but transform
  // has logical size N.
  if (scale_mode != ScaleMode::kNone) {
    // Compute the number of elements in the FFT dimensions (statically).
    int64_t num_elements = 1;
    for (int64_t len : fft_params.fft_length) {
      num_elements *= len;
    }
    mlir::MlirOp num_elements_op = MakeScalarConstant(
        input.getBuilder(), static_cast<double>(num_elements), float_type);

    TT_ASSIGN_OR_RETURN(auto scale_op,
                        ComputeScaleOp(fft_op.getBuilder(), num_elements_op,
                                       float_type, scale_mode));

    TT_ASSIGN_OR_RETURN(auto scaled_op, BroadcastIfNeeded(scale_op, fft_op));
    fft_op = mlir::stablehlo::Mul(fft_op, scaled_op);
  }

  if (transpose_perms.requires_transpose) {
    fft_op =
        mlir::stablehlo::Transpose(fft_op, transpose_perms.inverse_permutation);
  }

  return fft_op;
}

absl::Status FFTCheckStaticShape(const at::Tensor& tensor,
                                 const std::string_view arg_name) {
  TT_ASSIGN_OR_RETURN(DeviceBufferRef buffer_ref, GetBaseBuffer(tensor));
  return CheckStaticShape(tensor, buffer_ref, arg_name);
}

}  // namespace

at::Tensor AtenFftR2c(const at::Tensor& self, at::IntArrayRef dim,
                      int64_t normalization, bool onesided) {
  TT_KERNEL(OpName::kFftR2c, param_keys, (self, dim, normalization, onesided), {
    FftNormalization norm_enum = static_cast<FftNormalization>(normalization);

    // Explicitly check for static shape.
    // Shape introspection is unsafe with dynamism.
    // NOTE: This implementation is not dynamism safe. It currently
    // uses static shape indices, more work is needed to determine how
    // this op supports bounded dynamic values.
    TT_THROW_IF_ERROR(FFTCheckStaticShape(self, "input"));

    TT_ASSIGN_OR_THROW(auto normalized_dims, GetNormalizedDims(self, dim));
    auto out_sizes = CopyIntVector(self.sizes());
    if (onesided) {
      const int64_t last_dim = normalized_dims.back();
      out_sizes[last_dim] = self.size(last_dim) / 2 + 1;
    }

    const auto out_scalar_type = c10::toComplexType(self.scalar_type());
    TT_ASSIGN_OR_THROW(const auto output_dtype,
                       ConvertTo<mlir::ElementType>(out_scalar_type));

    auto op_builder = [normalized_dims, norm_enum, onesided](
                          mlir::MlirOp input) -> absl::StatusOr<mlir::MlirOp> {
      return BuildFftR2cShlo(input, normalized_dims, norm_enum, onesided);
    };

    TT_ASSIGN_OR_THROW(auto result, DispatchOp<1>(std::move(op_builder), self,
                                                  {.out_dtype = output_dtype,
                                                   .out_dims = out_sizes,
                                                   .op_param_cache_keys =
                                                       std::move(param_keys)}));

    TT_ASSIGN_OR_THROW(
        auto out, MakeEmptyTensor(out_sizes, out_scalar_type, self.device()));
    TT_THROW_IF_ERROR(AssignBufferToAtTensor(std::move(result), out));
    return out;
  });
}

at::Tensor& AtenFftR2cOut(const at::Tensor& self, at::IntArrayRef dim,
                          int64_t normalization, bool onesided,
                          at::Tensor& out) {
  TT_KERNEL(
      OpName::kFftR2cOut, param_keys, (self, dim, normalization, onesided, out),
      {
        FftNormalization norm_enum =
            static_cast<FftNormalization>(normalization);

        // Explicitly check for static shape.
        // Shape introspection is unsafe with dynamism.
        // NOTE: This implementation is not dynamism safe. It currently uses
        // static shape indices, more work is needed to determine how this op
        // supports bounded dynamic values.
        TT_THROW_IF_ERROR(FFTCheckStaticShape(self, "input"));

        TT_ASSIGN_OR_THROW(auto normalized_dims, GetNormalizedDims(self, dim));
        auto out_sizes = CopyIntVector(self.sizes());
        if (onesided) {
          const int64_t last_dim = normalized_dims.back();
          out_sizes[last_dim] = self.size(last_dim) / 2 + 1;
        }

        TT_THROW_IF_ERROR(ResizeTensorIfShapeDiffers(out, out_sizes));

        TT_ASSIGN_OR_THROW(const auto output_dtype,
                           ConvertTo<mlir::ElementType>(out.scalar_type()));

        auto op_builder =
            [normalized_dims, norm_enum,
             onesided](mlir::MlirOp input) -> absl::StatusOr<mlir::MlirOp> {
          return BuildFftR2cShlo(input, normalized_dims, norm_enum, onesided);
        };

        TT_ASSIGN_OR_THROW(
            auto result,
            DispatchOp<1>(std::move(op_builder), self,
                          {.out_dtype = output_dtype,
                           .out_dims = std::move(out_sizes),
                           .op_param_cache_keys = std::move(param_keys)}));

        TT_THROW_IF_ERROR(AssignBufferToAtTensor(std::move(result), out));
        return out;
      });
}

at::Tensor AtenFftC2c(const at::Tensor& self, at::IntArrayRef dim,
                      int64_t normalization, bool forward) {
  TT_KERNEL(OpName::kFftC2c, param_keys, (self, dim, normalization, forward), {
    FftNormalization norm_enum = static_cast<FftNormalization>(normalization);

    // Explicitly check for static shape.
    // Shape introspection is unsafe with dynamism.
    // NOTE: This implementation is not dynamism safe. It currently uses
    // static shape indices, more work is needed to determine how this op
    // supports bounded dynamic values.
    TT_THROW_IF_ERROR(FFTCheckStaticShape(self, "input"));

    TT_ASSIGN_OR_THROW(auto normalized_dims, GetNormalizedDims(self, dim));

    TT_ASSIGN_OR_THROW(const auto output_dtype,
                       ConvertTo<mlir::ElementType>(self.scalar_type()));

    auto op_builder = [normalized_dims, norm_enum, forward](
                          mlir::MlirOp input) -> absl::StatusOr<mlir::MlirOp> {
      return BuildFftC2cShlo(input, normalized_dims, norm_enum, forward);
    };

    TT_ASSIGN_OR_THROW(
        auto result,
        DispatchOp<1>(std::move(op_builder), self,
                      {.out_dtype = output_dtype,
                       .out_dims = CopyIntVector(self.sizes()),
                       .op_param_cache_keys = std::move(param_keys)}));

    TT_ASSIGN_OR_THROW(
        auto out,
        MakeEmptyTensor(self.sizes(), self.scalar_type(), self.device()));
    TT_THROW_IF_ERROR(AssignBufferToAtTensor(std::move(result), out));
    return out;
  });
}

at::Tensor& AtenFftC2cOut(const at::Tensor& self, at::IntArrayRef dim,
                          int64_t normalization, bool forward,
                          at::Tensor& out) {
  TT_KERNEL(
      OpName::kFftC2cOut, param_keys, (self, dim, normalization, forward, out),
      {
        FftNormalization norm_enum =
            static_cast<FftNormalization>(normalization);

        // Explicitly check for static shape.
        // Shape introspection is unsafe with dynamism.
        // NOTE: This implementation is not dynamism safe. It currently uses
        // static shape indices, more work is needed to determine how this op
        // supports bounded dynamic values.
        TT_THROW_IF_ERROR(FFTCheckStaticShape(self, "input"));

        TT_THROW_IF_ERROR(ResizeTensorIfShapeDiffers(out, self.sizes()));

        TT_ASSIGN_OR_THROW(auto normalized_dims, GetNormalizedDims(self, dim));

        TT_ASSIGN_OR_THROW(const auto output_dtype,
                           ConvertTo<mlir::ElementType>(out.scalar_type()));

        auto op_builder =
            [normalized_dims, norm_enum,
             forward](mlir::MlirOp input) -> absl::StatusOr<mlir::MlirOp> {
          return BuildFftC2cShlo(input, normalized_dims, norm_enum, forward);
        };

        TT_ASSIGN_OR_THROW(
            auto result,
            DispatchOp<1>(std::move(op_builder), self,
                          {.out_dtype = output_dtype,
                           .out_dims = CopyIntVector(out.sizes()),
                           .op_param_cache_keys = std::move(param_keys)}));

        TT_THROW_IF_ERROR(AssignBufferToAtTensor(std::move(result), out));
        return out;
      });
}

at::Tensor AtenFftC2r(const at::Tensor& self, at::IntArrayRef dim,
                      int64_t normalization, int64_t last_dim_size) {
  TT_KERNEL(
      OpName::kFftC2r, param_keys, (self, dim, normalization, last_dim_size), {
        FftNormalization norm_enum =
            static_cast<FftNormalization>(normalization);

        // Explicitly check for static shape.
        // Shape introspection is unsafe with dynamism.
        // NOTE: This implementation is not dynamism safe. It currently uses
        // static shape indices, more work is needed to determine how this op
        // supports bounded dynamic values.
        TT_THROW_IF_ERROR(FFTCheckStaticShape(self, "input"));

        TT_ASSIGN_OR_THROW(auto normalized_dims, GetNormalizedDims(self, dim));
        auto out_sizes = CopyIntVector(self.sizes());
        const int64_t last_dim = normalized_dims.back();
        out_sizes[last_dim] = last_dim_size;

        TT_ASSIGN_OR_THROW(const auto output_dtype,
                           ConvertTo<mlir::ElementType>(
                               c10::toRealValueType(self.scalar_type())));

        auto op_builder =
            [normalized_dims, norm_enum, last_dim_size](
                mlir::MlirOp input) -> absl::StatusOr<mlir::MlirOp> {
          return BuildFftC2rShlo(input, normalized_dims, norm_enum,
                                 last_dim_size);
        };

        TT_ASSIGN_OR_THROW(
            auto result,
            DispatchOp<1>(std::move(op_builder), self,
                          {.out_dtype = output_dtype,
                           .out_dims = out_sizes,
                           .op_param_cache_keys = std::move(param_keys)}));

        TT_ASSIGN_OR_THROW(
            auto out,
            MakeEmptyTensor(out_sizes, c10::toRealValueType(self.scalar_type()),
                            self.device()));
        TT_THROW_IF_ERROR(AssignBufferToAtTensor(std::move(result), out));
        return out;
      });
}

at::Tensor& AtenFftC2rOut(const at::Tensor& self, at::IntArrayRef dim,
                          int64_t normalization, int64_t last_dim_size,
                          at::Tensor& out) {
  TT_KERNEL(
      OpName::kFftC2rOut, param_keys,
      (self, dim, normalization, last_dim_size, out), {
        FftNormalization norm_enum =
            static_cast<FftNormalization>(normalization);

        // Explicitly check for static shape.
        // Shape introspection is unsafe with dynamism.
        // NOTE: This implementation is not dynamism safe. It currently uses
        // static shape indices, more work is needed to determine how this op
        // supports bounded dynamic values.
        TT_THROW_IF_ERROR(FFTCheckStaticShape(self, "input"));

        TT_ASSIGN_OR_THROW(auto normalized_dims, GetNormalizedDims(self, dim));
        auto out_sizes = CopyIntVector(self.sizes());
        const int64_t last_dim = normalized_dims.back();
        out_sizes[last_dim] = last_dim_size;

        TT_THROW_IF_ERROR(ResizeTensorIfShapeDiffers(out, out_sizes));

        TT_ASSIGN_OR_THROW(const auto output_dtype,
                           ConvertTo<mlir::ElementType>(out.scalar_type()));

        auto op_builder =
            [normalized_dims, norm_enum, last_dim_size](
                mlir::MlirOp input) -> absl::StatusOr<mlir::MlirOp> {
          return BuildFftC2rShlo(input, normalized_dims, norm_enum,
                                 last_dim_size);
        };

        TT_ASSIGN_OR_THROW(
            auto result,
            DispatchOp<1>(std::move(op_builder), self,
                          {.out_dtype = output_dtype,
                           .out_dims = out_sizes,
                           .op_param_cache_keys = std::move(param_keys)}));

        TT_THROW_IF_ERROR(AssignBufferToAtTensor(std::move(result), out));
        return out;
      });
}

}  // namespace torch_tpu
