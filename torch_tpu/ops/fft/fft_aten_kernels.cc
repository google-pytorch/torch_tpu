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

#include <algorithm>
#include <cstdint>
#include <numeric>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Support/LLVM.h"
#include "ATen/core/ATen_fwd.h"
#include "ATen/core/TensorBody.h"
#include "ATen/ops/empty.h"
#include "c10/core/ScalarType.h"
#include "torch_tpu/common/aten_utils.h"
#include "torch_tpu/common/cache_key.h"
#include "torch_tpu/common/dimension_types.h"
#include "torch_tpu/common/dtype.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/common/static_shape_check.h"
#include "torch_tpu/eager/device_buffer.h"
#include "torch_tpu/eager/op_dispatcher.h"
#include "torch_tpu/eager/tensor_to_buffer.h"
#include "torch_tpu/ops/macros/kernel.h"
#include "torch_tpu/ops/op_builder_utils.h"
#include "torch_tpu/ops/op_names.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"

namespace torch_tpu {
namespace {

// Normalizes the dimensions for the FFT operation
Dimensions GetNormalizedDims(const at::Tensor& self, at::IntArrayRef dim) {
  const int64_t num_dims = self.dim();
  Dimensions dims_vec = CopyIntVector(dim);

  // If no dimensions are specified, perform FFT on all dimensions
  if (dims_vec.empty()) {
    dims_vec.reserve(num_dims);
    for (int i = 0; i < num_dims; ++i) {
      dims_vec.push_back(i);
    }
  }

  // Normalize the dimensions to be positive
  Dimensions normalized_dims;
  normalized_dims.reserve(dims_vec.size());
  for (int64_t d : dims_vec) {
    normalized_dims.push_back(d < 0 ? d + num_dims : d);
  }

  return normalized_dims;
}

absl::StatusOr<mlir::MlirOp> BuildFftR2cShlo(mlir::MlirOp input,
                                             Dimensions dims,
                                             const int64_t normalization,
                                             bool onesided) {
  auto input_type = GetTensorTypeOrDie(input);

  // Explicitly check for static shape.
  // Shape introspection is unsafe with dynamism.
  // NOTE: This implementation is not dynamism safe. It currently uses static
  // shape indices, more work is needed to determine how this op supports
  // bounded dynamic values.
  TT_RETURN_IF_ERROR(CheckStaticShape(input_type, "input"));
  auto input_shape = input_type.getShape();
  const int64_t num_dims = input_type.getRank();
  const auto float_type = input_type.getElementType();

  // Compute the FFT length.
  Dimensions fft_length;
  for (int64_t dim : dims) {
    fft_length.push_back(input_shape[dim]);
  }

  // Compute the number of elements in the FFT dimensions.
  mlir::MlirOp num_elements_op = GetNumElements(input, float_type, dims);

  // Transpose the input tensor to move the FFT dimensions to the end
  std::vector<bool> is_fft_dim(num_dims, false);
  for (int64_t dim : dims) {
    is_fft_dim[dim] = true;
  }

  Dimensions p_vec(num_dims);
  std::iota(p_vec.begin(), p_vec.end(), 0);
  std::stable_partition(p_vec.begin(), p_vec.end(),
                        /*pred=*/[&](int64_t i) { return !is_fft_dim[i]; });
  Dimensions permutation_vec(p_vec.begin(), p_vec.end());

  bool requires_transpose = false;
  for (int i = 0; i < num_dims; ++i) {
    if (permutation_vec[i] != i) {
      requires_transpose = true;
      break;
    }
  }

  mlir::MlirOp fft_input = input;
  if (requires_transpose) {
    fft_input = mlir::stablehlo::Transpose(input, permutation_vec);
  }

  // Perform the FFT operation
  auto fft_op = mlir::stablehlo::Fft(fft_input, mlir::stablehlo::FftType::RFFT,
                                     fft_length);

  // Apply normalization: stablehlo::Fft does not support normalization modes,
  // so we scale the result manually based on the 'normalization' parameter
  // to match PyTorch's behavior.
  // 0 for 'backward', no normalization
  // 1 for 'ortho', 1 / sqrt(N)
  // 2 for 'forward', 1 / N
  if (normalization == 1 || normalization == 2) {
    mlir::MlirOp scale_op =
        MakeScalarConstant(fft_op.getBuilder(), 1.0, float_type);

    if (normalization == 1) {
      auto sqrt_op = mlir::stablehlo::Sqrt(num_elements_op);
      scale_op = mlir::stablehlo::Div(scale_op, sqrt_op);
    } else if (normalization == 2) {
      scale_op = mlir::stablehlo::Div(scale_op, num_elements_op);
    }

    // Create complex attribute {scale, 0.0} with same shape as fft_op
    TT_ASSIGN_OR_RETURN(auto real_op, BroadcastIfNeeded(scale_op, fft_op));
    auto imag_op = MakeConstantLike(fft_op, 0.0, float_type);
    auto complex_op = mlir::stablehlo::Complex(real_op, imag_op);

    fft_op = mlir::stablehlo::Mul(fft_op, complex_op);
  }

  // Transpose the result back to the original order of dimensions
  if (requires_transpose) {
    Dimensions inv_permutation_vec(num_dims);
    for (int i = 0; i < num_dims; ++i) {
      inv_permutation_vec[permutation_vec[i]] = i;
    }
    fft_op = mlir::stablehlo::Transpose(fft_op, inv_permutation_vec);
  }

  return fft_op;
}

absl::Status CheckStaticShape(const at::Tensor& tensor,
                              const std::string_view arg_name) {
  TT_ASSIGN_OR_RETURN(DeviceBufferRef buffer_ref,
                      GetBaseBufferFromAtTensor(tensor));
  return CheckStaticShape(tensor, buffer_ref, arg_name);
}

}  // namespace

at::Tensor AtenFftR2c(const at::Tensor& self, at::IntArrayRef dim,
                      int64_t normalization, bool onesided) {
  TT_KERNEL(OpName::kFftR2c, _,
            (self, IgnoreInCacheKey(dim, "Legacy usage"),
             IgnoreInCacheKey(normalization, "Legacy usage"),
             IgnoreInCacheKey(onesided, "Legacy usage")),
            {
              TT_THROW_IF_ERROR(CheckStaticShape(self, "input"));

              auto normalized_dims = GetNormalizedDims(self, dim);
              auto out_sizes = CopyIntVector(self.sizes());
              if (onesided) {
                const int64_t last_dim = normalized_dims.back();
                out_sizes[last_dim] = self.size(last_dim) / 2 + 1;
              }

              auto out = at::empty(
                  out_sizes,
                  self.options().dtype(c10::toComplexType(self.scalar_type())));

              AtenFftR2cOut(self, dim, normalization, onesided, out);
              return out;
            });
}

at::Tensor& AtenFftR2cOut(const at::Tensor& self, at::IntArrayRef dim,
                          int64_t normalization, bool onesided,
                          at::Tensor& out) {
  TT_KERNEL(
      OpName::kFftR2cOut, param_keys, (self, dim, normalization, onesided, out),
      {
        auto normalized_dims = GetNormalizedDims(self, dim);

        TT_ASSIGN_OR_THROW(const auto output_dtype,
                           ConvertTo<mlir::ElementType>(out.scalar_type()));

        auto op_builder =
            [normalized_dims, normalization,
             onesided](mlir::MlirOp input) -> absl::StatusOr<mlir::MlirOp> {
          return BuildFftR2cShlo(input, normalized_dims, normalization,
                                 onesided);
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
}  // namespace torch_tpu
