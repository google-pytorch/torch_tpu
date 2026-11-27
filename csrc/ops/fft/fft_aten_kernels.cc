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
#include "ATen/ops/empty.h"
#include "ATen/ops/fft_fft_native.h"
#include "ATen/ops/fft_ifft_native.h"
#include "ATen/ops/fft_irfft_native.h"
#include "ATen/ops/fft_rfft_native.h"
#include "absl/base/nullability.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "c10/core/ScalarType.h"
#include "c10/core/SymInt.h"
#include "c10/core/SymIntArrayRef.h"
#include "c10/util/DimVector.h"
#include "csrc/common/aten_utils.h"
#include "csrc/common/cache_key.h"
#include "csrc/common/const_tensor.h"
#include "csrc/common/dimension_types.h"
#include "csrc/common/dtype.h"
#include "csrc/common/error_utils.h"
#include "csrc/common/fake_tensor_utils.h"
#include "csrc/common/static_shape_check.h"
#include "csrc/common/to_string.h"
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

// Normalizes the FFT dimension indices into `c10::SymDimVector` (expanding an
// empty `dim` to `[0, ..., num_dims - 1]` and wrapping negative indices)
// without specializing symbolic input shapes or symbolic `SymInt` indices.
absl::StatusOr<c10::SymDimVector> NormalizeDims(const ConstTensor& self,
                                                const c10::SymIntArrayRef dim) {
  TT_ASSIGN_OR_RETURN(const int64_t num_dims, self.dim());

  c10::SymDimVector normalized_dims;
  if (dim.empty()) {
    normalized_dims.reserve(num_dims);
    for (int64_t i = 0; i < num_dims; ++i) {
      normalized_dims.emplace_back(i);
    }
    return normalized_dims;
  }

  normalized_dims.reserve(dim.size());
  for (const c10::SymInt& sym_d : dim) {
    TT_ASSIGN_OR_RETURN(c10::SymInt wrapped_d, SafeWrapDim(sym_d, num_dims));
    normalized_dims.push_back(std::move(wrapped_d));
  }
  return normalized_dims;
}

// Overload for `at::IntArrayRef`.
// Converts into `c10::SymIntArrayRef` by calling `c10::fromIntArrayRefSlow`,
// which errors if any integer element is not representable by a `SymInt`.
absl::StatusOr<c10::SymDimVector> NormalizeDims(const ConstTensor& self,
                                                const at::IntArrayRef dim) {
  return NormalizeDims(self, c10::fromIntArrayRefSlow(dim));
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
    mlir::MlirOp input, const c10::SymIntArrayRef sym_dims,
    const FftNormalization normalization, bool onesided) {
  TT_ASSIGN_OR_RETURN(const Dimensions dims,
                      ConcreteSizeToDimensions(sym_dims));
  auto input_type = GetTensorTypeOrDie(input);

  if (input_type.getElementType().isBF16()) {
    mlir::Builder builder(input_type.getContext());
    input = mlir::stablehlo::ConvertElementType(input, builder.getF32Type());
    input_type = GetTensorTypeOrDie(input);
  }

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
    mlir::MlirOp input, const c10::SymIntArrayRef sym_dims,
    const FftNormalization normalization, bool forward) {
  TT_ASSIGN_OR_RETURN(const Dimensions dims,
                      ConcreteSizeToDimensions(sym_dims));
  auto input_type = GetTensorTypeOrDie(input);

  auto elem_type = input_type.getElementType();
  if (!mlir::isa<mlir::ComplexType>(elem_type)) {
    TT_ASSIGN_OR_RETURN(auto scalar_type, ConvertTo<at::ScalarType>(elem_type));
    return TT_ERROR(  // ERROR_COV_INFEASIBLE=Checked by caller before
                      // DispatchOp.
               error::kInvalidArgument)
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
    return TT_ERROR(  // ERROR_COV_INFEASIBLE=Checked by caller before
                      // DispatchOp.
               error::kInvalidArgument)
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

absl::Status FFTValidateStaticShape(const at::Tensor& tensor,
                                    const std::string_view arg_name) {
  TT_ASSIGN_OR_RETURN(DeviceBufferRef buffer_ref, GetBaseBuffer(tensor));
  return ValidateStaticShape(tensor, buffer_ref, arg_name);
}

[[nodiscard]] at::Tensor PromoteToFloatIfBFloat16(const at::Tensor& self) {
  return self.scalar_type() == at::kBFloat16 ? self.to(at::kFloat) : self;
}

// Computes the symbolic output sizes for an R2C FFT, halving the last
// transformed dimension to `n / 2 + 1` when `onesided` is true without
// specializing symbolic dimensions.
[[nodiscard]] c10::SymDimVector ComputeR2cOutSymSizes(
    const c10::SymIntArrayRef sym_sizes,
    const c10::SymIntArrayRef normalized_dims, const bool onesided) {
  c10::SymDimVector out_sym_sizes(sym_sizes.begin(), sym_sizes.end());
  if (onesided && !normalized_dims.empty()) {
    const int64_t last_dim =
        normalized_dims.back().guard_int(__FILE__, __LINE__);
    out_sym_sizes[last_dim] = sym_sizes[last_dim] / 2 + 1;
  }
  return out_sym_sizes;
}

// Computes the symbolic output sizes for a C2R FFT, setting the last
// transformed dimension to `last_dim_size` without specializing other symbolic
// dimensions.
[[nodiscard]] c10::SymDimVector ComputeC2rOutSymSizes(
    const c10::SymIntArrayRef sym_sizes,
    const c10::SymIntArrayRef normalized_dims,
    const c10::SymInt& last_dim_size) {
  c10::SymDimVector out_sym_sizes(sym_sizes.begin(), sym_sizes.end());
  if (!normalized_dims.empty()) {
    const int64_t last_dim =
        normalized_dims.back().guard_int(__FILE__, __LINE__);
    out_sym_sizes[last_dim] = last_dim_size;
  }
  return out_sym_sizes;
}

// Validates that `self` has a static shape and dispatches an FFT op on a real
// TPU tensor. When `out` is non-null, writes the result into `*out` via
// `DispatchOpOut` and returns `*out`; otherwise allocates and returns a new
// output tensor.
template <typename BuildShloFn>
absl::StatusOr<at::Tensor> DispatchFftImpl(
    const at::Tensor& self, const c10::SymIntArrayRef out_sym_sizes,
    const mlir::ElementType output_dtype, OpParamCacheKeys param_keys,
    const BuildShloFn& build_shlo, at::Tensor* absl_nullable out = nullptr) {
  // Explicitly check for static shape.
  // Shape introspection is unsafe with dynamism.
  // NOTE: This implementation is not dynamism safe. It currently
  // uses static shape indices, more work is needed to determine how
  // this op supports bounded dynamic values.
  TT_RETURN_IF_ERROR(FFTValidateStaticShape(self, "input"));
  TT_ASSIGN_OR_RETURN(const Dimensions out_dims,
                      ConcreteSizeToDimensions(out_sym_sizes));

  auto op_builder = [build_shlo, out_dims](
                        mlir::MlirOp input) -> absl::StatusOr<mlir::MlirOp> {
    return build_shlo(input, out_dims);
  };

  DispatchOpOptions<1> options = {.out_dtype = output_dtype,
                                  .out_dims = out_dims,
                                  .op_param_cache_keys = std::move(param_keys)};
  if (out != nullptr) {
    TT_RETURN_IF_ERROR(DispatchOpOut<1>(std::move(op_builder), self, *out,
                                        std::move(options)));
    return *out;
  }

  TT_ASSIGN_OR_RETURN(
      DeviceBufferRef result,
      DispatchOp<1>(std::move(op_builder), self, std::move(options)));
  return MakeTensor(std::move(result));
}

// Dispatches an allocating FFT kernel, returning a symbolic tensor via
// `at::empty_symint` for fake/functorch inputs or invoking `DispatchFftImpl`
// for real TPU tensors.
template <typename BuildShloFn>
absl::StatusOr<at::Tensor> DispatchFft(const at::Tensor& self,
                                       const c10::SymIntArrayRef out_sym_sizes,
                                       const at::ScalarType out_scalar_type,
                                       const mlir::ElementType output_dtype,
                                       OpParamCacheKeys param_keys,
                                       const BuildShloFn& build_shlo) {
  if (IsFakeOrFunctorchTensor(self)) {
    return at::empty_symint(out_sym_sizes,
                            self.options().dtype(out_scalar_type));
  }
  return DispatchFftImpl(self, out_sym_sizes, output_dtype,
                         std::move(param_keys), build_shlo);
}

// Dispatches an out-variant FFT kernel, resizing `out` symbolically for
// fake/functorch inputs or invoking `DispatchFftImpl` into `out` for real TPU
// tensors.
template <typename BuildShloFn>
absl::Status DispatchFftOut(const at::Tensor& self, at::Tensor& out,
                            const c10::SymIntArrayRef out_sym_sizes,
                            const mlir::ElementType output_dtype,
                            OpParamCacheKeys param_keys,
                            const BuildShloFn& build_shlo) {
  if (IsFakeOrFunctorchTensor(self) || IsFakeOrFunctorchTensor(out)) {
    out.resize__symint(out_sym_sizes);
  } else {
    TT_RETURN_IF_ERROR(DispatchFftImpl(self, out_sym_sizes, output_dtype,
                                       std::move(param_keys), build_shlo,
                                       &out));
  }
  return absl::OkStatus();
}

}  // namespace

at::Tensor AtenFftR2c(const at::Tensor& self, at::IntArrayRef dim,
                      int64_t normalization, bool onesided) {
  TT_KERNEL(OpName::kFftR2c, param_keys, (self, dim, normalization, onesided), {
    const FftNormalization norm_enum =
        static_cast<FftNormalization>(normalization);
    TT_ASSIGN_OR_THROW(const c10::SymDimVector normalized_dims,
                       NormalizeDims(self, dim));
    const c10::SymDimVector out_sym_sizes =
        ComputeR2cOutSymSizes(self.sym_sizes(), normalized_dims, onesided);

    const auto compute_scalar_type =
        self.scalar_type() == at::kBFloat16 ? at::kFloat : self.scalar_type();
    const auto out_scalar_type = c10::toComplexType(compute_scalar_type);
    TT_ASSIGN_OR_THROW(const auto output_dtype,
                       ConvertTo<mlir::ElementType>(out_scalar_type));

    TT_ASSIGN_OR_THROW(
        at::Tensor out,
        DispatchFft(self, out_sym_sizes, out_scalar_type, output_dtype,
                    std::move(param_keys),
                    [normalized_dims, norm_enum, onesided](
                        mlir::MlirOp input, const Dimensions& /*out_dims*/) {
                      return BuildFftR2cShlo(input, normalized_dims, norm_enum,
                                             onesided);
                    }));
    return out;
  });
}

at::Tensor& AtenFftR2cOut(const at::Tensor& self, at::IntArrayRef dim,
                          int64_t normalization, bool onesided,
                          at::Tensor& out) {
  TT_KERNEL(
      OpName::kFftR2cOut, param_keys, (self, dim, normalization, onesided, out),
      {
        const FftNormalization norm_enum =
            static_cast<FftNormalization>(normalization);
        TT_ASSIGN_OR_THROW(const c10::SymDimVector normalized_dims,
                           NormalizeDims(self, dim));
        const c10::SymDimVector out_sym_sizes =
            ComputeR2cOutSymSizes(self.sym_sizes(), normalized_dims, onesided);

        const auto result_dtype = c10::toComplexType(self.scalar_type());
        TT_THROW_IF_ERROR(ValidateOutDtype(out, result_dtype,
                                           /*allow_out_dtype_cast=*/true));

        TT_ASSIGN_OR_THROW(const auto output_dtype,
                           ConvertTo<mlir::ElementType>(out.scalar_type()));

        TT_THROW_IF_ERROR(DispatchFftOut(
            self, out, out_sym_sizes, output_dtype, std::move(param_keys),
            [normalized_dims, norm_enum, onesided](
                mlir::MlirOp input, const Dimensions& /*out_dims*/) {
              return BuildFftR2cShlo(input, normalized_dims, norm_enum,
                                     onesided);
            }));
        return out;
      });
}

at::Tensor AtenFftC2c(const at::Tensor& self, c10::SymIntArrayRef dim,
                      int64_t normalization, bool forward) {
  TT_KERNEL(OpName::kFftC2c, param_keys, (self, dim, normalization, forward), {
    const FftNormalization norm_enum =
        static_cast<FftNormalization>(normalization);
    TT_ASSIGN_OR_THROW(const c10::SymDimVector normalized_dims,
                       NormalizeDims(self, dim));

    TT_ASSIGN_OR_THROW(const auto output_dtype,
                       ConvertTo<mlir::ElementType>(self.scalar_type()));
    TT_CHECK_THROW(self.is_complex(), error::kInvalidArgument)
        << "expected complex input type, got " << ToString(self.scalar_type());
    TT_ASSIGN_OR_THROW(
        at::Tensor out,
        DispatchFft(self, self.sym_sizes(), self.scalar_type(), output_dtype,
                    std::move(param_keys),
                    [normalized_dims, norm_enum, forward](
                        mlir::MlirOp input, const Dimensions& /*out_dims*/) {
                      return BuildFftC2cShlo(input, normalized_dims, norm_enum,
                                             forward);
                    }));
    return out;
  });
}

at::Tensor& AtenFftC2cOut(const at::Tensor& self, c10::SymIntArrayRef dim,
                          int64_t normalization, bool forward,
                          at::Tensor& out) {
  TT_KERNEL(
      OpName::kFftC2cOut, param_keys, (self, dim, normalization, forward, out),
      {
        const FftNormalization norm_enum =
            static_cast<FftNormalization>(normalization);
        TT_THROW_IF_ERROR(ValidateOutDtype(out, self.scalar_type(),
                                           /*allow_out_dtype_cast=*/true));

        TT_ASSIGN_OR_THROW(const c10::SymDimVector normalized_dims,
                           NormalizeDims(self, dim));

        TT_ASSIGN_OR_THROW(const auto output_dtype,
                           ConvertTo<mlir::ElementType>(out.scalar_type()));
        TT_CHECK_THROW(self.is_complex(), error::kInvalidArgument)
            << "expected complex input type, got "
            << ToString(self.scalar_type());
        TT_THROW_IF_ERROR(DispatchFftOut(
            self, out, self.sym_sizes(), output_dtype, std::move(param_keys),
            [normalized_dims, norm_enum, forward](
                mlir::MlirOp input, const Dimensions& /*out_dims*/) {
              return BuildFftC2cShlo(input, normalized_dims, norm_enum,
                                     forward);
            }));
        return out;
      });
}

at::Tensor AtenFftC2r(const at::Tensor& self, at::IntArrayRef dim,
                      int64_t normalization, c10::SymInt last_dim_size) {
  TT_KERNEL(
      OpName::kFftC2r, param_keys, (self, dim, normalization, last_dim_size), {
        const FftNormalization norm_enum =
            static_cast<FftNormalization>(normalization);
        TT_ASSIGN_OR_THROW(const c10::SymDimVector sym_normalized_dims,
                           NormalizeDims(self, dim));
        const c10::SymDimVector out_sym_sizes = ComputeC2rOutSymSizes(
            self.sym_sizes(), sym_normalized_dims, last_dim_size);

        const auto out_scalar_type = c10::toRealValueType(self.scalar_type());
        TT_ASSIGN_OR_THROW(const auto output_dtype,
                           ConvertTo<mlir::ElementType>(out_scalar_type));
        TT_CHECK_THROW(self.is_complex(), error::kInvalidArgument)
            << "expected complex input type, got "
            << ToString(self.scalar_type());

        // Get concrete size for normalized dimensions. This won't error.
        TT_ASSIGN_OR_THROW(const Dimensions normalized_dims,
                           ConcreteSizeToDimensions(sym_normalized_dims));

        TT_ASSIGN_OR_THROW(
            at::Tensor out,
            DispatchFft(self, out_sym_sizes, out_scalar_type, output_dtype,
                        std::move(param_keys),
                        [normalized_dims = std::move(normalized_dims),
                         norm_enum, last_dim_size](mlir::MlirOp input,
                                                   const Dimensions& out_dims) {
                          return BuildFftC2rShlo(
                              input, std::move(normalized_dims), norm_enum,
                              last_dim_size.expect_int());
                        }));
        return out;
      });
}

at::Tensor& AtenFftC2rOut(const at::Tensor& self, at::IntArrayRef dim,
                          int64_t normalization, c10::SymInt last_dim_size,
                          at::Tensor& out) {
  TT_KERNEL(
      OpName::kFftC2rOut, param_keys,
      (self, dim, normalization, last_dim_size, out), {
        const FftNormalization norm_enum =
            static_cast<FftNormalization>(normalization);
        TT_ASSIGN_OR_THROW(const c10::SymDimVector sym_normalized_dims,
                           NormalizeDims(self, dim));
        const c10::SymDimVector out_sym_sizes = ComputeC2rOutSymSizes(
            self.sym_sizes(), sym_normalized_dims, last_dim_size);

        const auto result_dtype = c10::toRealValueType(self.scalar_type());
        TT_THROW_IF_ERROR(ValidateOutDtype(out, result_dtype,
                                           /*allow_out_dtype_cast=*/true));

        TT_ASSIGN_OR_THROW(const auto output_dtype,
                           ConvertTo<mlir::ElementType>(out.scalar_type()));
        TT_CHECK_THROW(self.is_complex(), error::kInvalidArgument)
            << "expected complex input type, got "
            << ToString(self.scalar_type());

        // Get concrete size for normalized dimensions. This won't error.
        TT_ASSIGN_OR_THROW(const Dimensions normalized_dims,
                           ConcreteSizeToDimensions(sym_normalized_dims));

        TT_THROW_IF_ERROR(DispatchFftOut(
            self, out, out_sym_sizes, output_dtype, std::move(param_keys),
            [normalized_dims = std::move(normalized_dims), norm_enum,
             last_dim_size](mlir::MlirOp input, const Dimensions& out_dims) {
              return BuildFftC2rShlo(input, std::move(normalized_dims),
                                     norm_enum, last_dim_size.expect_int());
            }));
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
