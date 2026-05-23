// Copyright 2025 Google LLC
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

#include "torch_tpu/ops/nll_loss/nll_loss_aten_kernels.h"

#include <array>
#include <cstdint>
#include <optional>
#include <tuple>
#include <utility>
#include <vector>

#include "ATen/core/ATen_fwd.h"
#include "ATen/core/LegacyTypeDispatch.h"
#include "ATen/core/Reduction.h"
#include "ATen/native/Resize.h"
#include "ATen/ops/nll_loss_forward.h"
#include "ATen/ops/permute.h"
#include "ATen/ops/reshape.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "c10/util/Optional.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"
#include "torch/headeronly/core/ScalarType.h"
#include "torch_tpu/common/cache_key.h"
#include "torch_tpu/common/dimension_types.h"
#include "torch_tpu/common/dtype.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/common/to_string.h"
#include "torch_tpu/common/utils.h"
#include "torch_tpu/eager/device_buffer.h"
#include "torch_tpu/eager/op_dispatcher.h"
#include "torch_tpu/eager/tensor_to_buffer.h"
#include "torch_tpu/ops/macros/kernel.h"
#include "torch_tpu/ops/nll_loss/nll_loss.h"
#include "torch_tpu/ops/op_builder_utils.h"
#include "torch_tpu/ops/op_names.h"
#include "xla/xla_data.pb.h"

namespace torch_tpu {
namespace {

bool IsLongOrByte(const at::Tensor& tensor) {
  const at::ScalarType scalar_type = tensor.scalar_type();
  return scalar_type == at::ScalarType::Long ||
         scalar_type == at::ScalarType::Byte;
}

absl::StatusOr<mlir::MlirOp> BuildNllLoss2dBackwardGradInputShlo(
    mlir::MlirOp grad_output, mlir::MlirOp self, mlir::MlirOp target,
    std::optional<mlir::MlirOp> weight, int64_t reduction, int64_t ignore_index,
    mlir::MlirOp total_weight, mlir::MlirBuilder& builder) {
  auto self_type = GetTensorTypeOrDie(self);
  auto self_shape = self_type.getShape();
  int64_t n = self_shape[0];
  int64_t c = self_shape[1];
  int64_t h = self_shape[2];
  int64_t w = self_shape[3];

  // Transpose `self` to map `(n, c, h, w)` to `(n, h, w, c)`
  mlir::MlirOp transposed_self = mlir::stablehlo::Transpose(self, {0, 2, 3, 1});

  // Reshape the transposed `self` to flatten spatial dims to `(n * h * w, c)`
  mlir::MlirOp flattened_self =
      mlir::stablehlo::Reshape(transposed_self, {n * h * w, c});

  // Reshape `target` to flatten spatial dims to `(n * h * w)`
  mlir::MlirOp flattened_target = mlir::stablehlo::Reshape(target, {n * h * w});

  // Reshape `grad_output` to flatten to `(n * h * w)` if `reduction == None`
  mlir::MlirOp flattened_grad_output = grad_output;
  if (reduction == at::Reduction::None) {
    flattened_grad_output = mlir::stablehlo::Reshape(grad_output, {n * h * w});
  }

  TT_ASSIGN_OR_RETURN(
      mlir::MlirOp grad_input_flattened,
      BuildNllLossBackwardGradInputShlo(flattened_grad_output, flattened_self,
                                        flattened_target, weight, reduction,
                                        ignore_index, total_weight, builder));

  // Reshape the returned `(n * h * w, c)` gradient back to `(n, h, w, c)`
  mlir::MlirOp reshaped_grad_input =
      mlir::stablehlo::Reshape(grad_input_flattened, {n, h, w, c});

  // Transpose `(0, 3, 1, 2)` to reconstruct the spatial format `(n, c, h, w)`
  mlir::MlirOp final_grad_input =
      mlir::stablehlo::Transpose(reshaped_grad_input, {0, 3, 1, 2});

  return final_grad_input;
}

}  // namespace

std::tuple<at::Tensor&, at::Tensor&> AtenNllLossForwardOut(
    const at::Tensor& self, const at::Tensor& target,
    const c10::optional<at::Tensor>& weight, int64_t reduction,
    int64_t ignore_index, at::Tensor& output, at::Tensor& total_weight) {
  TT_KERNEL(
      OpName::kNllLossForwardOut, param_keys,
      (self, target, weight, reduction, ignore_index, output, total_weight), {
        // self.scalar_type() must be a real-valued float type.
        TT_CHECK_THROW(
            self.scalar_type() == at::ScalarType::Half ||
                self.scalar_type() == at::ScalarType::BFloat16 ||
                self.scalar_type() == at::ScalarType::Float ||
                self.scalar_type() == at::ScalarType::Double ||
                self.scalar_type() == at::ScalarType::Float8_e5m2 ||
                self.scalar_type() == at::ScalarType::Float8_e4m3fn ||
                self.scalar_type() == at::ScalarType::Float8_e5m2fnuz ||
                self.scalar_type() == at::ScalarType::Float8_e4m3fnuz ||
                self.scalar_type() == at::ScalarType::Float8_e8m0fnu ||
                self.scalar_type() == at::ScalarType::Float4_e2m1fn_x2,
            error::kInvalidArgument)
            << "unsupported input dtype: " << ToString(self.scalar_type());
        TT_CHECK_THROW(IsLongOrByte(target), error::kInvalidArgument)
            << "expected the target dtype to be either int64 or uint8, got "
            << ToString(target.scalar_type());

        // Get output dtype and output dims.
        TT_ASSIGN_OR_THROW(mlir::ElementType output_dtype,
                           ConvertTo<mlir::ElementType>(self.scalar_type()));
        std::array<mlir::ElementType, 2> out_dtypes = {output_dtype,
                                                       output_dtype};
        std::array<absl::Span<const int64_t>, 2> out_dims_list;
        Dimensions empty_dims = {};
        if (reduction == at::Reduction::None) {
          out_dims_list = {target.sizes(), absl::MakeConstSpan(empty_dims)};
        } else {
          out_dims_list = {absl::MakeConstSpan(empty_dims),
                           absl::MakeConstSpan(empty_dims)};
        }

        DispatchOpOptions<2> options = {
            .out_dtypes = out_dtypes,
            .out_dims_list = out_dims_list,
            .op_param_cache_keys = std::move(param_keys),
        };

        bool has_weight = weight.has_value() && weight->defined();
        auto op_builder = [reduction, ignore_index, has_weight](
                              absl::Span<const mlir::MlirOp> inputs,
                              mlir::MlirBuilder& builder) {
          mlir::MlirOp self_op = inputs[0];
          mlir::MlirOp target_op = inputs[1];
          std::optional<mlir::MlirOp> weight_op;
          if (has_weight) {
            weight_op = inputs[2];
          }

          return BuildNllLossForwardOutShlo(self_op, target_op, weight_op,
                                            reduction, ignore_index, builder);
        };

        // Build inputs for the op.
        std::vector<at::Tensor> inputs = {self, target};
        if (has_weight) {
          inputs.push_back(weight.value());
        }

        TT_ASSIGN_OR_THROW(auto output_bufs, (DispatchOp<kDynamicSize, 2>(
                                                 std::move(op_builder), inputs,
                                                 std::move(options))));

        at::native::resize_output(output, output_bufs[0].dimensions());
        TT_THROW_IF_ERROR(
            AssignBufferToAtTensor(std::move(output_bufs[0]), output));
        at::native::resize_output(total_weight, output_bufs[1].dimensions());
        TT_THROW_IF_ERROR(
            AssignBufferToAtTensor(std::move(output_bufs[1]), total_weight));

        return std::tie(output, total_weight);
      });
}

std::tuple<at::Tensor, at::Tensor> AtenNllLoss2dForward(
    const at::Tensor& self, const at::Tensor& target,
    const c10::optional<at::Tensor>& weight, int64_t reduction,
    int64_t ignore_index) {
  TT_KERNEL(
      OpName::kNllLoss2dForward, _,
      (self, target,
       IgnoreInCacheKey(weight, "delegates to inner at::nll_loss_forward"),
       IgnoreInCacheKey(reduction, "delegates to inner at::nll_loss_forward"),
       IgnoreInCacheKey(ignore_index,
                        "delegates to inner at::nll_loss_forward")),
      {
        TT_CHECK_THROW(self.dim() >= 3 && self.dim() == target.dim() + 1 &&
                           self.size(0) == target.size(0) &&
                           self.sizes().slice(2) == target.sizes().slice(1),
                       error::kInvalidArgument)
            << "expect the shapes of the input [N, C, d1, ..., dk] and the "
               "target [N, d1, ..., dk] (k >= 1) to match, got input: "
            << self.sizes() << ", target: " << target.sizes();

        const int64_t n_classes = self.size(1);
        // (N, C, H, W) -> (N * H * W, C)
        at::Tensor input_reshaped =
            at::reshape(at::permute(self, {0, 2, 3, 1}), {-1, n_classes});
        // (N, H, W) -> (N * H * W)
        at::Tensor target_reshaped = at::reshape(target, {-1});

        at::Tensor loss_output_inner;   // UNINITIALIZED_TENSOR_OK=return value
                                        // in a tuple
        at::Tensor total_weight_inner;  // UNINITIALIZED_TENSOR_OK=return value
                                        // in a tuple
        {
          // All compositional ATen calls must be within a guard to prevent
          // re-dispatching to the Autograd layer, which would cause infinite
          // recursion. This ensures we are calling the primitive PrivateUse1
          // kernels.

          at::AutoDispatchBelowAutograd guard;
          std::tie(loss_output_inner, total_weight_inner) =
              at::nll_loss_forward(input_reshaped, target_reshaped, weight,
                                   reduction, ignore_index);
        }

        if (reduction == at::Reduction::None) {
          return {loss_output_inner.view(
                      {self.size(0), self.size(2), self.size(3)}),
                  total_weight_inner};
        }
        return {loss_output_inner, total_weight_inner};
      });
}

std::tuple<at::Tensor&, at::Tensor&> AtenNllLoss2dForwardOut(
    const at::Tensor& self, const at::Tensor& target,
    const c10::optional<at::Tensor>& weight, int64_t reduction,
    int64_t ignore_index, at::Tensor& output, at::Tensor& total_weight) {
  TT_KERNEL(
      OpName::kNllLoss2dForwardOut, _,
      (self, target,
       IgnoreInCacheKey(weight, "delegates to inner at::nll_loss_forward"),
       IgnoreInCacheKey(reduction, "delegates to inner at::nll_loss_forward"),
       IgnoreInCacheKey(ignore_index,
                        "delegates to inner at::nll_loss_forward"),
       output, total_weight),
      {
        at::Tensor loss_output_inner;   // UNINITIALIZED_TENSOR_OK=return value
                                        // in a tuple
        at::Tensor total_weight_inner;  // UNINITIALIZED_TENSOR_OK=return value
                                        // in a tuple
        std::tie(loss_output_inner, total_weight_inner) =
            AtenNllLoss2dForward(self, target, weight, reduction, ignore_index);

        at::native::resize_output(output, loss_output_inner.sizes());
        output.copy_(loss_output_inner);
        at::native::resize_output(total_weight, total_weight_inner.sizes());
        total_weight.copy_(total_weight_inner);

        return std::tie(output, total_weight);
      });
}

at::Tensor& AtenNllLossBackwardGradInput(
    const at::Tensor& grad_output, const at::Tensor& self,
    const at::Tensor& target, const c10::optional<at::Tensor>& weight,
    int64_t reduction, int64_t ignore_index, const at::Tensor& total_weight,
    at::Tensor& grad_input) {
  TT_KERNEL(
      OpName::kNllLossBackwardGradInput, param_keys,
      (grad_output, self, target, weight, reduction, ignore_index, total_weight,
       grad_input),
      {
        bool has_weight = weight.has_value() && weight->defined();
        auto op_builder =
            [reduction, ignore_index, has_weight](
                absl::Span<const mlir::MlirOp> inputs,
                mlir::MlirBuilder& builder) -> absl::StatusOr<mlir::MlirOp> {
          mlir::MlirOp grad_output_op = inputs[0];
          mlir::MlirOp self_op = inputs[1];
          mlir::MlirOp target_op = inputs[2];
          mlir::MlirOp total_weight_op = inputs[3];
          std::optional<mlir::MlirOp> weight_op;
          if (has_weight) {
            weight_op = inputs[4];
          }

          TT_ASSIGN_OR_RETURN(
              mlir::MlirOp grad_input_op,
              BuildNllLossBackwardGradInputShlo(
                  grad_output_op, self_op, target_op, weight_op, reduction,
                  ignore_index, total_weight_op, builder));
          return grad_input_op;
        };

        // Get output dtype and output dims.
        TT_ASSIGN_OR_THROW(const auto output_dtype,
                           ConvertTo<mlir::ElementType>(self.scalar_type()));
        const auto grad_input_sizes = self.sizes();
        Dimensions output_dims = CopyIntVector(grad_input_sizes);

        // Build inputs for the op.
        std::vector<at::Tensor> inputs = {grad_output, self, target,
                                          total_weight};
        if (has_weight) {
          inputs.push_back(weight.value());
        }

        DispatchOpOptions<1> options = {
            .out_dtype = output_dtype,
            .out_dims = output_dims,
            .op_param_cache_keys = std::move(param_keys)};

        TT_ASSIGN_OR_THROW(
            auto grad_input_buf,
            (DispatchOp<kDynamicSize, 1>(std::move(op_builder), inputs,
                                         std::move(options))));

        at::IntArrayRef output_sizes_array_ref(grad_input_buf.dimensions());
        at::native::resize_output(grad_input, grad_input_buf.dimensions());
        TT_THROW_IF_ERROR(
            AssignBufferToAtTensor(std::move(grad_input_buf), grad_input));
        return grad_input;
      });
}

at::Tensor AtenNllLoss2dBackward(const at::Tensor& grad_output,
                                 const at::Tensor& self,
                                 const at::Tensor& target,
                                 const c10::optional<at::Tensor>& weight,
                                 int64_t reduction, int64_t ignore_index,
                                 const at::Tensor& total_weight) {
  TT_KERNEL(
      OpName::kNllLoss2dBackward, param_keys,
      (grad_output, self, target, weight, reduction, ignore_index,
       total_weight),
      {
        TT_CHECK_THROW(self.dim() == 4, error::kInvalidArgument)
            << "expected input to be a 4D tensor, got " << self.dim() << "D";
        TT_CHECK_THROW(target.dim() == 3, error::kInvalidArgument)
            << "expected target to be a 3D tensor, got " << target.dim() << "D";

        bool has_weight = weight.has_value() && weight->defined();
        auto op_builder =
            [reduction, ignore_index, has_weight](
                absl::Span<const mlir::MlirOp> inputs,
                mlir::MlirBuilder& builder) -> absl::StatusOr<mlir::MlirOp> {
          mlir::MlirOp grad_output_op = inputs[0];
          mlir::MlirOp self_op = inputs[1];
          mlir::MlirOp target_op = inputs[2];
          mlir::MlirOp total_weight_op = inputs[3];
          std::optional<mlir::MlirOp> weight_op;
          if (has_weight) {
            weight_op = inputs[4];
          }

          return BuildNllLoss2dBackwardGradInputShlo(
              grad_output_op, self_op, target_op, weight_op, reduction,
              ignore_index, total_weight_op, builder);
        };

        TT_ASSIGN_OR_THROW(const auto output_dtype,
                           ConvertTo<mlir::ElementType>(self.scalar_type()));
        const auto grad_input_sizes = self.sizes();
        Dimensions output_dims = CopyIntVector(grad_input_sizes);

        std::vector<at::Tensor> inputs = {grad_output, self, target,
                                          total_weight};
        if (has_weight) {
          inputs.push_back(weight.value());
        }

        DispatchOpOptions<1> options = {
            .out_dtype = output_dtype,
            .out_dims = output_dims,
            .op_param_cache_keys = std::move(param_keys)};

        TT_ASSIGN_OR_THROW(
            auto grad_input_buf,
            (DispatchOp<kDynamicSize, 1>(std::move(op_builder), inputs,
                                         std::move(options))));

        return MakeTensor(std::move(grad_input_buf));
      });
}

at::Tensor& AtenNllLoss2dBackwardGradInput(
    const at::Tensor& grad_output, const at::Tensor& self,
    const at::Tensor& target, const c10::optional<at::Tensor>& weight,
    int64_t reduction, int64_t ignore_index, const at::Tensor& total_weight,
    at::Tensor& grad_input) {
  TT_KERNEL(
      OpName::kNllLoss2dBackwardGradInput, _,
      (grad_output, self, target,
       IgnoreInCacheKey(weight, "delegates to inner AtenNllLoss2dBackward"),
       IgnoreInCacheKey(reduction, "delegates to inner AtenNllLoss2dBackward"),
       IgnoreInCacheKey(ignore_index,
                        "delegates to inner AtenNllLoss2dBackward"),
       total_weight, grad_input),
      {
        at::Tensor grad_input_inner =
            AtenNllLoss2dBackward(grad_output, self, target, weight, reduction,
                                  ignore_index, total_weight);

        at::native::resize_output(grad_input, grad_input_inner.sizes());
        grad_input.copy_(grad_input_inner);
        return grad_input;
      });
}

}  // namespace torch_tpu
