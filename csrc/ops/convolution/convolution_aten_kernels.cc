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

#include "csrc/ops/convolution/convolution_aten_kernels.h"

#include <array>
#include <cstdint>
#include <optional>
#include <string_view>
#include <tuple>
#include <utility>

#include "ATen/core/ATen_fwd.h"
#include "ATen/core/TensorBody.h"
#include "ATen/ops/promote_types.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "c10/core/ScalarType.h"
#include "csrc/common/aten_utils.h"
#include "csrc/common/cache_key.h"
#include "csrc/common/dimension_types.h"
#include "csrc/common/dtype.h"
#include "csrc/common/error_utils.h"
#include "csrc/common/fixed_size_span.h"
#include "csrc/common/to_string.h"
#include "csrc/eager/device_buffer.h"
#include "csrc/eager/op_dispatcher.h"
#include "csrc/eager/tensor_to_buffer.h"
#include "csrc/ops/convolution/convolution.h"
#include "csrc/ops/convolution/convolution_checks.h"
#include "csrc/ops/macros/kernel.h"
#include "csrc/ops/op_builder_utils.h"
#include "csrc/ops/op_names.h"
#include "csrc/ops/precision_context.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "torch/headeronly/core/ScalarType.h"
#include "xla/xla_data.pb.h"

namespace torch_tpu {

namespace {

// This helper function handles PyTorch's parameter broadcasting for
// convolutions. It ensures that convolution parameters (specifically stride,
// padding, dilation, and output_padding) have the correct number of elements to
// match the input's spatial dimensions (e.g., 2 elements for conv2d, 1 for
// conv1d). This function is needed because in PyTorch a user can provide
// convolution parameters in two ways:
//
//   1. A tuple matching the rank: For conv2d, providing stride=(2, 2).
//   2. A single integer (broadcasted): Providing stride=2.
//
//  When the PyTorch dispatcher calls the backend kernel, these are often passed
//  as at::IntArrayRef. If a single value is passed (like [2]), the backend must
//  "expand" it to match the spatial dimensions (becoming [2, 2]) before it can
//  be used for output shape calculations or used to generate StableHLO, which
//  expects the full array. Without this expansion, the code below would see an
//  input with 2 spatial dimensions but a stride with only 1 dimension.
Dimensions ExpandIfNecessary(at::IntArrayRef param, int num_spatial_dims) {
  if (param.size() == 1) {
    return Dimensions(num_spatial_dims, param[0]);
  }
  return Dimensions(param.begin(), param.end());
}

absl::Status ValidateConvolutionInputs(
    const at::Tensor& input, const at::Tensor& weight,
    absl::Span<const int64_t> bias, absl::Span<const int64_t> stride,
    absl::Span<const int64_t> padding, absl::Span<const int64_t> dilation,
    bool transposed, absl::Span<const int64_t> output_padding, int64_t groups) {
  TT_RETURN_IF_ERROR(ValidateConvolutionInput(input.sizes()));

  // Number of dimensions, excluding the batch and channel dimensions.
  const int64_t num_spatial_dims = input.dim() - 2;
  const int64_t in_channels = input.size(1);

  TT_RETURN_IF_ERROR(ValidateConvolutionSpatialDimensionsMatch(
      num_spatial_dims, stride, "stride"));
  TT_RETURN_IF_ERROR(ValidateConvolutionSpatialDimensionsMatch(
      num_spatial_dims, padding, "padding"));
  TT_RETURN_IF_ERROR(ValidateConvolutionSpatialDimensionsMatch(
      num_spatial_dims, dilation, "dilation"));
  TT_RETURN_IF_ERROR(ValidateConvolutionSpatialDimensionsMatch(
      num_spatial_dims, output_padding, "output_padding"));

  TT_RETURN_IF_ERROR(ValidateConvolutionWeight(
      weight.sizes(), num_spatial_dims, in_channels, groups, transposed));

  if (!bias.empty()) {
    const int64_t out_channels =
        transposed ? weight.size(1) * groups : weight.size(0);
    TT_RETURN_IF_ERROR(ValidateConvolutionBias(bias, out_channels));
  }

  return absl::OkStatus();
}

absl::Status ValidateConvolutionInputs(
    const at::Tensor& input, const at::Tensor& weight,
    const std::optional<at::Tensor>& bias_opt, absl::Span<const int64_t> stride,
    absl::Span<const int64_t> padding, absl::Span<const int64_t> dilation,
    bool transposed, absl::Span<const int64_t> output_padding, int64_t groups) {
  auto bias_dimensions =
      bias_opt.has_value() ? bias_opt->sizes() : at::IntArrayRef();
  return ValidateConvolutionInputs(input, weight, bias_dimensions, stride,
                                   padding, dilation, transposed,
                                   output_padding, groups);
}

absl::StatusOr<Dimensions> GetOutputDimensions(
    const at::Tensor& input, const at::Tensor& weight,
    const std::optional<at::Tensor>& bias_opt, absl::Span<const int64_t> stride,
    absl::Span<const int64_t> padding, absl::Span<const int64_t> dilation,
    bool transposed, absl::Span<const int64_t> output_padding, int64_t groups) {
  // Number of dimensions, excluding the batch and channel dimensions.
  const int64_t num_spatial_dims = input.dim() - 2;
  Dimensions output_sizes;

  output_sizes.reserve(num_spatial_dims + 2);
  output_sizes.push_back(input.size(0));  // Batch
  if (transposed) {
    output_sizes.push_back(weight.size(1) * groups);  // Output features
  } else {
    output_sizes.push_back(weight.size(0));  // Output features
  }

  for (int i = 0; i < num_spatial_dims; ++i) {
    if (transposed) {
      // H_out = (H_in - 1) * stride - 2 * padding + dilation * (kernel_size -
      // 1)
      // + output_padding + 1
      int64_t output_dim_size = (input.size(i + 2) - 1) * stride[i];
      output_dim_size -= 2 * padding[i];
      output_dim_size += dilation[i] * (weight.size(i + 2) - 1);
      output_dim_size += output_padding[i] + 1;
      output_sizes.push_back(output_dim_size);
    } else {
      // Output shape formula for spatial dimensions:
      // https://docs.pytorch.org/docs/stable/generated/torch.nn.Conv2d.html#torch.nn.Conv2d
      int64_t output_dim_size = input.size(i + 2);
      output_dim_size += padding[i] * 2;
      output_dim_size += stride[i] - 1;
      output_dim_size -= (weight.size(i + 2) - 1) * dilation[i];
      // Truncating integer division is intended
      output_dim_size /= stride[i];
      output_sizes.push_back(output_dim_size);
    }
  }
  return output_sizes;
}

absl::Status IsTypeSupported(const at::Tensor& tensor,
                             const std::string_view arg_name) {
  TT_RET_CHECK(!IsBool(tensor) && !IsLong(tensor), error::kInvalidArgument)
      << "expected the dtype of the " << arg_name
      << " tensor to be neither long nor bool, got "
      << ToString(tensor.scalar_type());
  return absl::OkStatus();
}

absl::StatusOr<at::ScalarType> GetPromotedType(
    const at::Tensor& input, const at::Tensor& weight,
    const std::optional<at::Tensor>& bias_opt) {
  TT_RETURN_IF_ERROR(IsTypeSupported(input, "input"));
  TT_RETURN_IF_ERROR(IsTypeSupported(weight, "weight"));

  at::ScalarType promoted_type =
      at::promote_types(input.scalar_type(), weight.scalar_type());

  // CUDA runs zero-batch and zero-channel convolutions on a dtype-agnostic
  // path, so only non-empty int32 inputs fail.
  TT_RET_CHECK(
      input.size(0) == 0 || input.size(1) == 0 || promoted_type != at::kInt,
      error::kPythonNotImplementedError)
      << "not implemented for " << ToString(promoted_type);

  if (bias_opt.has_value()) {
    // TODO: native PyTorch does not errors on boolean bias.
    TT_RETURN_IF_ERROR(IsTypeSupported(*bias_opt, "bias"));
    promoted_type = at::promote_types(promoted_type, bias_opt->scalar_type());
  }

  return promoted_type;
}

struct ConvolutionDispatchParams {
  Dimensions expanded_stride;
  Dimensions expanded_padding;
  Dimensions expanded_dilation;
  Dimensions expanded_output_padding;
  Dimensions output_dims;
  mlir::ElementType mlir_dtype;
  mlir::stablehlo::Precision current_precision;
};

absl::StatusOr<ConvolutionDispatchParams> PrepareConvolutionDispatch(
    const at::Tensor& input, const at::Tensor& weight,
    const std::optional<at::Tensor>& bias_opt, at::IntArrayRef stride,
    at::IntArrayRef padding, at::IntArrayRef dilation, bool transposed,
    at::IntArrayRef output_padding, int64_t groups,
    OpParamCacheKeys& param_keys) {
  const int num_spatial_dims = input.dim() - 2;
  Dimensions expanded_stride = ExpandIfNecessary(stride, num_spatial_dims);
  Dimensions expanded_padding = ExpandIfNecessary(padding, num_spatial_dims);
  Dimensions expanded_dilation = ExpandIfNecessary(dilation, num_spatial_dims);
  Dimensions expanded_output_padding =
      ExpandIfNecessary(output_padding, num_spatial_dims);

  const auto current_precision = GetAndAddPrecisionTo(param_keys);
  TT_RETURN_IF_ERROR(ValidateConvolutionInputs(
      input, weight, bias_opt, expanded_stride, expanded_padding,
      expanded_dilation, transposed, expanded_output_padding, groups));
  TT_ASSIGN_OR_RETURN(
      Dimensions output_dims,
      GetOutputDimensions(input, weight, bias_opt, expanded_stride,
                          expanded_padding, expanded_dilation, transposed,
                          expanded_output_padding, groups));
  TT_ASSIGN_OR_RETURN(at::ScalarType promoted_dtype,
                      GetPromotedType(input, weight, bias_opt));
  TT_ASSIGN_OR_RETURN(const auto mlir_dtype,
                      ConvertTo<mlir::ElementType>(promoted_dtype));

  return ConvolutionDispatchParams{
      .expanded_stride = std::move(expanded_stride),
      .expanded_padding = std::move(expanded_padding),
      .expanded_dilation = std::move(expanded_dilation),
      .expanded_output_padding = std::move(expanded_output_padding),
      .output_dims = std::move(output_dims),
      .mlir_dtype = mlir_dtype,
      .current_precision = current_precision,
  };
}

absl::Status ConvolutionBinaryOut(
    const at::Tensor& input, const at::Tensor& weight, at::IntArrayRef stride,
    at::IntArrayRef padding, at::IntArrayRef dilation, bool transposed,
    at::IntArrayRef output_padding, int64_t groups, at::Tensor& out,
    OpParamCacheKeys param_keys) {
  TT_ASSIGN_OR_RETURN(
      ConvolutionDispatchParams params,
      PrepareConvolutionDispatch(input, weight, std::nullopt, stride, padding,
                                 dilation, transposed, output_padding, groups,
                                 param_keys));

  auto op_builder =
      [stride = Strides(params.expanded_stride.begin(),
                        params.expanded_stride.end()),
       padding = Dimensions(params.expanded_padding.begin(),
                            params.expanded_padding.end()),
       dilation = Dimensions(params.expanded_dilation.begin(),
                             params.expanded_dilation.end()),
       transposed,
       output_padding = Dimensions(params.expanded_output_padding.begin(),
                                   params.expanded_output_padding.end()),
       groups,
       output_dims =
           Dimensions(params.output_dims.begin(), params.output_dims.end()),
       mlir_dtype = params.mlir_dtype,
       current_precision =
           params.current_precision](FixedSizeSpan<mlir::MlirOp, 2> inputs) {
        auto& [input, weight] = inputs;
        return BuildConvolution(input, weight, std::nullopt, stride, padding,
                                dilation, transposed, output_padding, groups,
                                output_dims, mlir_dtype, current_precision);
      };

  DispatchOpOptions<1> options = {
      .out_dtype = params.mlir_dtype,
      .out_dims = std::move(params.output_dims),
      .op_param_cache_keys = std::move(param_keys),
  };
  return DispatchOpOut<2>(std::move(op_builder), {input, weight}, out,
                          std::move(options));
}

absl::Status ConvolutionTernaryOut(
    const at::Tensor& input, const at::Tensor& weight, const at::Tensor& bias,
    at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation,
    bool transposed, at::IntArrayRef output_padding, int64_t groups,
    at::Tensor& out, OpParamCacheKeys param_keys) {
  TT_ASSIGN_OR_RETURN(ConvolutionDispatchParams params,
                      PrepareConvolutionDispatch(
                          input, weight, bias, stride, padding, dilation,
                          transposed, output_padding, groups, param_keys));

  auto op_builder =
      [stride = Strides(params.expanded_stride.begin(),
                        params.expanded_stride.end()),
       padding = Dimensions(params.expanded_padding.begin(),
                            params.expanded_padding.end()),
       dilation = Dimensions(params.expanded_dilation.begin(),
                             params.expanded_dilation.end()),
       transposed,
       output_padding = Dimensions(params.expanded_output_padding.begin(),
                                   params.expanded_output_padding.end()),
       groups,
       output_dims =
           Dimensions(params.output_dims.begin(), params.output_dims.end()),
       mlir_dtype = params.mlir_dtype,
       current_precision =
           params.current_precision](FixedSizeSpan<mlir::MlirOp, 3> inputs) {
        auto& [input, weight, bias] = inputs;
        return BuildConvolution(input, weight, bias, stride, padding, dilation,
                                transposed, output_padding, groups, output_dims,
                                mlir_dtype, current_precision);
      };

  DispatchOpOptions<1> options = {
      .out_dtype = params.mlir_dtype,
      .out_dims = std::move(params.output_dims),
      .op_param_cache_keys = std::move(param_keys),
  };
  return DispatchOpOut<3>(std::move(op_builder), {input, weight, bias}, out,
                          std::move(options));
}

absl::Status ConvolutionOut(const at::Tensor& input, const at::Tensor& weight,
                            const std::optional<at::Tensor>& bias_opt,
                            at::IntArrayRef stride, at::IntArrayRef padding,
                            at::IntArrayRef dilation, bool transposed,
                            at::IntArrayRef output_padding, int64_t groups,
                            at::Tensor& out, OpParamCacheKeys param_keys) {
  if (bias_opt.has_value() && bias_opt->defined()) {
    TT_RETURN_IF_ERROR(param_keys.SetParam("ternary", true));
    return ConvolutionTernaryOut(input, weight, bias_opt.value(), stride,
                                 padding, dilation, transposed, output_padding,
                                 groups, out, std::move(param_keys));
  }
  return ConvolutionBinaryOut(input, weight, stride, padding, dilation,
                              transposed, output_padding, groups, out,
                              std::move(param_keys));
}

absl::StatusOr<at::ScalarType> GetPromotedTypeBackward(
    const at::Tensor& grad_output, const at::Tensor& input,
    const at::Tensor& weight) {
  TT_RETURN_IF_ERROR(IsTypeSupported(grad_output, "grad"));
  TT_RETURN_IF_ERROR(IsTypeSupported(input, "input"));
  TT_RETURN_IF_ERROR(IsTypeSupported(weight, "weight"));
  return at::promote_types(
      at::promote_types(input.scalar_type(), weight.scalar_type()),
      grad_output.scalar_type());
}

absl::StatusOr<DeviceBufferRefArray<3>> ConvolutionBackward(
    const at::Tensor& grad_output, const at::Tensor& input,
    const at::Tensor& weight, at::OptionalIntArrayRef bias_sizes,
    at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation,
    bool transposed, at::IntArrayRef output_padding, int64_t groups,
    std::array<bool, 3> output_mask, OpParamCacheKeys param_keys) {
  const int num_spatial_dims = input.dim() - 2;
  Dimensions expanded_stride = ExpandIfNecessary(stride, num_spatial_dims);
  Dimensions expanded_padding = ExpandIfNecessary(padding, num_spatial_dims);
  Dimensions expanded_dilation = ExpandIfNecessary(dilation, num_spatial_dims);
  Dimensions expanded_output_padding =
      ExpandIfNecessary(output_padding, num_spatial_dims);

  const auto current_precision = GetAndAddPrecisionTo(param_keys);

  // A non-empty `bias_dimensions` will trigger a bias dimensions check.
  // This should only be run if we are computing the backwards w.r.t. the
  // bias tensor. Otherwise, do not check it.
  auto bias_dimensions = (output_mask[2] && bias_sizes.has_value())
                             ? bias_sizes.value()
                             : at::IntArrayRef();

  TT_RETURN_IF_ERROR(ValidateConvolutionInputs(
      input, weight, bias_dimensions, expanded_stride, expanded_padding,
      expanded_dilation, transposed, expanded_output_padding, groups));
  TT_ASSIGN_OR_RETURN(at::ScalarType promoted_dtype,
                      GetPromotedTypeBackward(grad_output, input, weight));
  TT_ASSIGN_OR_RETURN(const auto output_dtype,
                      ConvertTo<mlir::ElementType>(promoted_dtype));

  auto input_sizes = input.sizes();
  auto weight_sizes = weight.sizes();

  auto op_builder =
      [stride = expanded_stride, padding = expanded_padding,
       dilation = expanded_dilation, output_padding = expanded_output_padding,
       groups, transposed, output_mask,
       input_dims = Dimensions(input_sizes.begin(), input_sizes.end()),
       weight_dims = Dimensions(weight_sizes.begin(), weight_sizes.end()),
       output_dtype, current_precision](FixedSizeSpan<mlir::MlirOp, 3> inputs)
      -> absl::StatusOr<MlirOpResults<3>> {
    auto& [grad_out, in, w] = inputs;
    mlir::MlirOp grad_in, grad_w, grad_b;

    auto make_undefined = [&grad_out,
                           output_dtype]() -> absl::StatusOr<mlir::MlirOp> {
      return MakeZeroSizedTensor(
          grad_out.getBuilder(),
          mlir::getElementType(grad_out.getContext(), output_dtype));
    };

    if (output_mask[0]) {
      TT_ASSIGN_OR_RETURN(
          grad_in,
          BuildConvolutionBackwardInput(
              grad_out, w, input_dims, stride, padding, dilation, groups,
              transposed, output_padding, output_dtype, current_precision));
    } else {
      TT_ASSIGN_OR_RETURN(grad_in, make_undefined());
    }

    if (output_mask[1]) {
      TT_ASSIGN_OR_RETURN(
          grad_w,
          BuildConvolutionBackwardWeight(
              in, grad_out, weight_dims, stride, padding, dilation, groups,
              transposed, output_padding, output_dtype, current_precision));
    } else {
      TT_ASSIGN_OR_RETURN(grad_w, make_undefined());
    }

    if (output_mask[2]) {
      TT_ASSIGN_OR_RETURN(grad_b, BuildConvolutionBackwardBias(
                                      grad_out, output_padding, output_dtype));
    } else {
      TT_ASSIGN_OR_RETURN(grad_b, make_undefined());
    }

    return MlirOpResults<3>{grad_in, grad_w, grad_b};
  };

  std::array<mlir::ElementType, 3> out_dtypes = {output_dtype, output_dtype,
                                                 output_dtype};
  Dimensions bias_dims = {transposed ? groups * weight.size(1)
                                     : weight.size(0)};
  Dimensions empty_dims = {0};

  std::array<absl::Span<const int64_t>, 3> out_dims_list = {
      output_mask[0] ? absl::Span<const int64_t>(input_sizes)
                     : absl::Span<const int64_t>(empty_dims),
      output_mask[1] ? absl::Span<const int64_t>(weight_sizes)
                     : absl::Span<const int64_t>(empty_dims),
      output_mask[2] ? absl::Span<const int64_t>(bias_dims)
                     : absl::Span<const int64_t>(empty_dims)};

  TT_ASSIGN_OR_RETURN(
      auto results,
      (DispatchOp<3, 3>(
          std::move(op_builder), {grad_output, input, weight},
          {.out_dtypes = FixedSizeSpan<const mlir::ElementType, 3>(out_dtypes),
           .out_dims_list =
               FixedSizeSpan<const absl::Span<const int64_t>, 3>(out_dims_list),
           .op_param_cache_keys = std::move(param_keys)})));
  return results;
}

}  // namespace

at::Tensor AtenConvolution(const at::Tensor& input, const at::Tensor& weight,
                           const std::optional<at::Tensor>& bias_opt,
                           at::IntArrayRef stride, at::IntArrayRef padding,
                           at::IntArrayRef dilation, bool transposed,
                           at::IntArrayRef output_padding, int64_t groups) {
  TT_KERNEL(
      OpName::kConvolution, param_keys,
      (input, weight, bias_opt, stride, padding, dilation, transposed,
       output_padding, groups),
      {
        TT_ASSIGN_OR_THROW(
            at::ScalarType promoted_dtype,
            GetPromotedType(input, weight,
                            bias_opt.has_value() && bias_opt->defined()
                                ? bias_opt
                                : std::nullopt));
        TT_ASSIGN_OR_THROW(
            at::Tensor out,
            MakeEmptyTensor(/*size=*/{0}, promoted_dtype, input.device()));
        TT_THROW_IF_ERROR(ConvolutionOut(
            input, weight, bias_opt, stride, padding, dilation, transposed,
            output_padding, groups, out, std::move(param_keys)));
        return out;
      });
}

at::Tensor& AtenConvolutionOut(const at::Tensor& input,
                               const at::Tensor& weight,
                               const std::optional<at::Tensor>& bias_opt,
                               at::IntArrayRef stride, at::IntArrayRef padding,
                               at::IntArrayRef dilation, bool transposed,
                               at::IntArrayRef output_padding, int64_t groups,
                               at::Tensor& out) {
  TT_KERNEL(OpName::kConvolutionOut, param_keys,
            (input, weight, bias_opt, stride, padding, dilation, transposed,
             output_padding, groups, out),
            {
              TT_THROW_IF_ERROR(ConvolutionOut(input, weight, bias_opt, stride,
                                               padding, dilation, transposed,
                                               output_padding, groups, out,
                                               std::move(param_keys)));
              return out;
            });
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> AtenConvolutionBackward(
    const at::Tensor& grad_output, const at::Tensor& input,
    const at::Tensor& weight, at::OptionalIntArrayRef bias_sizes,
    at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation,
    bool transposed, at::IntArrayRef output_padding, int64_t groups,
    std::array<bool, 3> output_mask) {
  TT_KERNEL(
      OpName::kConvolutionBackward, param_keys,
      (grad_output, input, weight, bias_sizes, stride, padding, dilation,
       transposed, output_padding, groups, output_mask),
      {
        TT_ASSIGN_OR_THROW(
            auto results,
            ConvolutionBackward(grad_output, input, weight, bias_sizes, stride,
                                padding, dilation, transposed, output_padding,
                                groups, output_mask, std::move(param_keys)));
        auto to_tensor = [&](int idx) -> at::Tensor {
          if (!output_mask[idx]) return at::Tensor();
          return MakeTensor(std::move(results[idx]));
        };

        return std::make_tuple(to_tensor(0), to_tensor(1), to_tensor(2));
      });
}

}  // namespace torch_tpu
