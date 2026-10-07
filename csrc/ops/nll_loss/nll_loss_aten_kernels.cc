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

#include "csrc/ops/nll_loss/nll_loss_aten_kernels.h"

#include <array>
#include <cstdint>
#include <optional>
#include <tuple>
#include <utility>
#include <vector>

#include "ATen/core/ATen_fwd.h"
#include "ATen/core/LegacyTypeDispatch.h"
#include "ATen/core/Reduction.h"
#include "ATen/ops/nll_loss_nd_native.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "c10/core/DispatchKey.h"
#include "c10/core/GradMode.h"
#include "c10/core/impl/LocalDispatchKeySet.h"
#include "c10/util/Optional.h"
#include "csrc/common/cache_key.h"
#include "csrc/common/dimension_types.h"
#include "csrc/common/dtype.h"
#include "csrc/common/error_utils.h"
#include "csrc/common/to_string.h"
#include "csrc/common/utils.h"
#include "csrc/eager/device_buffer.h"
#include "csrc/eager/op_dispatcher.h"
#include "csrc/eager/tensor_to_buffer.h"
#include "csrc/ops/macros/kernel.h"
#include "csrc/ops/nll_loss/nll_loss.h"
#include "csrc/ops/op_builder_utils.h"
#include "csrc/ops/op_names.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "torch/csrc/autograd/custom_function.h"
#include "torch/headeronly/core/ScalarType.h"

namespace torch_tpu {
namespace {

bool IsFunctorchActive() {
  const auto include_set = c10::impl::tls_local_dispatch_key_set().included_;
  return include_set.has(c10::DispatchKey::FuncTorchDynamicLayerFrontMode) ||
         include_set.has(c10::DispatchKey::FuncTorchDynamicLayerBackMode);
}

bool IsFakeOrFunctorchTensor(const at::Tensor& tensor) {
  if (IsFunctorchActive() || !tensor.has_storage()) {
    return true;
  }
  return tensor.storage().data_ptr() == nullptr;
}

absl::Status ValidateNllLossCommon(const at::Tensor& self,
                                   const at::Tensor& target,
                                   const c10::optional<at::Tensor>& weight,
                                   int64_t reduction) {
  TT_RET_CHECK(IsSupportedLossFloatType(self.scalar_type()),
               error::kInvalidArgument)
      << "unsupported input dtype: " << ToString(self.scalar_type());
  TT_RET_CHECK(IsLongOrByteTarget(target), error::kInvalidArgument)
      << "expected the target dtype to be either int64 or uint8, got "
      << ToString(target.scalar_type());
  TT_RET_CHECK(reduction == at::Reduction::None ||
                   reduction == at::Reduction::Mean ||
                   reduction == at::Reduction::Sum,
               error::kInvalidArgument)
      << "expected reduction to be one of ['none', 'mean', 'sum'], got "
      << reduction;
  if (weight.has_value() && weight->defined() && self.dim() >= 1) {
    const int64_t n_classes = self.size(self.dim() == 1 ? 0 : 1);
    TT_RET_CHECK(weight->dim() == 1 && weight->numel() == n_classes,
                 error::kInvalidArgument)
        << "expected weight tensor to be 1D with size " << n_classes << ", got "
        << ToString(weight->sizes());
  }
  return absl::OkStatus();
}

absl::Status ValidateNllLoss1dOr2dShapes(const at::Tensor& self,
                                         const at::Tensor& target) {
  TT_RET_CHECK(self.dim() >= 1 && self.dim() <= 2, error::kInvalidArgument)
      << "expected 1D or 2D input tensor, got " << self.dim() << "D";
  TT_RET_CHECK(target.dim() <= 1, error::kInvalidArgument)
      << "expected 0D or 1D target tensor, got " << target.dim() << "D";
  if (self.dim() == 1 && target.dim() == 1) {
    TT_RET_CHECK(target.size(0) == 1, error::kInvalidArgument)
        << "expected 1D target tensor to have size 1 for 1D input, got "
        << target.size(0);
  }
  TT_RET_CHECK(self.dim() == 1 || self.size(0) == target.size(0),
               error::kInvalidArgument)
      << "expected input batch_size (" << self.size(0)
      << ") to match target batch_size (" << target.size(0)
      << "), got input: " << ToString(self.sizes())
      << ", target: " << ToString(target.sizes());
  return absl::OkStatus();
}

absl::Status ValidateNllLoss2dForwardShapes(const at::Tensor& self,
                                            const at::Tensor& target) {
  TT_RET_CHECK(self.dim() == 4 && self.dim() == target.dim() + 1 &&
                   self.size(0) == target.size(0) &&
                   self.sizes().slice(2) == target.sizes().slice(1),
               error::kInvalidArgument)
      << "expected the shapes of the input [N, C, d1, ..., dk] and the "
         "target [N, d1, ..., dk] (k >= 1) to match, got input: "
      << ToString(self.sizes()) << ", target: " << ToString(target.sizes());
  return absl::OkStatus();
}

absl::Status ValidateNllLossNdShapes(const at::Tensor& self,
                                     const at::Tensor& target) {
  TT_RET_CHECK(self.dim() >= 1, error::kInvalidArgument)
      << "expected 1 or more dimensions for input, got " << self.dim();
  if (self.dim() <= 2) {
    return ValidateNllLoss1dOr2dShapes(self, target);
  }
  TT_RET_CHECK(self.dim() == target.dim() + 1 &&
                   self.size(0) == target.size(0) &&
                   self.sizes().slice(2) == target.sizes().slice(1),
               error::kInvalidArgument)
      << "expected the shapes of the input [N, C, d1, ..., dk] and the "
         "target [N, d1, ..., dk] (k >= 1) to match, got input: "
      << ToString(self.sizes()) << ", target: " << ToString(target.sizes());
  return absl::OkStatus();
}

absl::StatusOr<std::array<DeviceBufferRef, 2>> DispatchNllLossForwardBuffers(
    const at::Tensor& self, const at::Tensor& target,
    const c10::optional<at::Tensor>& weight, int64_t reduction,
    int64_t ignore_index, OpParamCacheKeys param_keys) {
  TT_ASSIGN_OR_RETURN(const mlir::ElementType output_dtype,
                      ConvertTo<mlir::ElementType>(self.scalar_type()));
  const std::array<mlir::ElementType, 2> out_dtypes = {output_dtype,
                                                       output_dtype};
  std::array<absl::Span<const int64_t>, 2> out_dims_list;
  const Dimensions empty_dims = {};
  if (reduction == at::Reduction::None && self.dim() >= 2) {
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

  const bool has_weight = weight.has_value() && weight->defined();
  auto op_builder = [reduction, ignore_index, has_weight](
                        absl::Span<const mlir::MlirOp> inputs,
                        mlir::MlirBuilder& builder) {
    mlir::MlirOp self_op = inputs[0];
    mlir::MlirOp target_op = inputs[1];
    std::optional<mlir::MlirOp> weight_op;
    if (has_weight) {
      weight_op = inputs[2];
    }
    return BuildNllLossForwardOutShlo(self_op, target_op, weight_op, reduction,
                                      ignore_index, builder);
  };

  std::vector<at::Tensor> inputs = {self, target};
  if (has_weight) {
    inputs.push_back(weight.value());
  }

  return DispatchOp<kDynamicSize, 2>(std::move(op_builder), inputs,
                                     std::move(options));
}

absl::Status DispatchNllLossForwardOut(const at::Tensor& self,
                                       const at::Tensor& target,
                                       const c10::optional<at::Tensor>& weight,
                                       int64_t reduction, int64_t ignore_index,
                                       at::Tensor& output,
                                       at::Tensor& total_weight,
                                       OpParamCacheKeys param_keys) {
  TT_ASSIGN_OR_RETURN(const mlir::ElementType output_dtype,
                      ConvertTo<mlir::ElementType>(self.scalar_type()));
  const std::array<mlir::ElementType, 2> out_dtypes = {output_dtype,
                                                       output_dtype};
  std::array<absl::Span<const int64_t>, 2> out_dims_list;
  const Dimensions empty_dims = {};
  if (reduction == at::Reduction::None && self.dim() >= 2) {
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

  const bool has_weight = weight.has_value() && weight->defined();
  auto op_builder = [reduction, ignore_index, has_weight](
                        absl::Span<const mlir::MlirOp> inputs,
                        mlir::MlirBuilder& builder) {
    mlir::MlirOp self_op = inputs[0];
    mlir::MlirOp target_op = inputs[1];
    std::optional<mlir::MlirOp> weight_op;
    if (has_weight) {
      weight_op = inputs[2];
    }
    return BuildNllLossForwardOutShlo(self_op, target_op, weight_op, reduction,
                                      ignore_index, builder);
  };

  std::vector<at::Tensor> inputs = {self, target};
  if (has_weight) {
    inputs.push_back(weight.value());
  }

  OpOutputs<2> outputs = {output, total_weight};
  return DispatchOpOut<kDynamicSize, 2>(std::move(op_builder), inputs, outputs,
                                        std::move(options));
}

absl::StatusOr<DeviceBufferRef> DispatchNllLossBackwardBuffer(
    const at::Tensor& grad_output, const at::Tensor& self,
    const at::Tensor& target, const c10::optional<at::Tensor>& weight,
    int64_t reduction, int64_t ignore_index, const at::Tensor& total_weight,
    OpParamCacheKeys param_keys) {
  const bool has_weight = weight.has_value() && weight->defined();
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
    return BuildNllLossBackwardGradInputShlo(grad_output_op, self_op, target_op,
                                             weight_op, reduction, ignore_index,
                                             total_weight_op, builder);
  };

  TT_ASSIGN_OR_RETURN(const auto output_dtype,
                      ConvertTo<mlir::ElementType>(self.scalar_type()));
  const Dimensions output_dims = CopyIntVector(self.sizes());

  std::vector<at::Tensor> inputs = {grad_output, self, target, total_weight};
  if (has_weight) {
    inputs.push_back(weight.value());
  }

  DispatchOpOptions<1> options = {
      .out_dtype = output_dtype,
      .out_dims = output_dims,
      .op_param_cache_keys = std::move(param_keys),
  };

  return DispatchOp<kDynamicSize, 1>(std::move(op_builder), inputs,
                                     std::move(options));
}

absl::Status DispatchNllLossBackwardOut(
    const at::Tensor& grad_output, const at::Tensor& self,
    const at::Tensor& target, const c10::optional<at::Tensor>& weight,
    int64_t reduction, int64_t ignore_index, const at::Tensor& total_weight,
    at::Tensor& grad_input, OpParamCacheKeys param_keys) {
  const bool has_weight = weight.has_value() && weight->defined();
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
    return BuildNllLossBackwardGradInputShlo(grad_output_op, self_op, target_op,
                                             weight_op, reduction, ignore_index,
                                             total_weight_op, builder);
  };

  TT_ASSIGN_OR_RETURN(const auto output_dtype,
                      ConvertTo<mlir::ElementType>(self.scalar_type()));
  const Dimensions output_dims = CopyIntVector(self.sizes());

  std::vector<at::Tensor> inputs = {grad_output, self, target, total_weight};
  if (has_weight) {
    inputs.push_back(weight.value());
  }

  DispatchOpOptions<1> options = {
      .out_dtype = output_dtype,
      .out_dims = output_dims,
      .op_param_cache_keys = std::move(param_keys),
  };

  return DispatchOpOut<kDynamicSize>(std::move(op_builder), inputs, grad_input,
                                     std::move(options));
}

std::tuple<at::Tensor, at::Tensor> NllLossNdForwardWithTotalWeight(
    const at::Tensor& self, const at::Tensor& target,
    const c10::optional<at::Tensor>& weight, int64_t reduction,
    int64_t ignore_index) {
  TT_KERNEL(OpName::kNllLossNd, param_keys,
            (self, target, weight, reduction, ignore_index), {
              TT_THROW_IF_ERROR(
                  ValidateNllLossCommon(self, target, weight, reduction));
              TT_THROW_IF_ERROR(ValidateNllLossNdShapes(self, target));
              TT_ASSIGN_OR_THROW(auto output_bufs,
                                 DispatchNllLossForwardBuffers(
                                     self, target, weight, reduction,
                                     ignore_index, std::move(param_keys)));
              return {MakeTensor(std::move(output_bufs[0])),
                      MakeTensor(std::move(output_bufs[1]))};
            });
}

at::Tensor NllLossBackwardKernel(const at::Tensor& grad_output,
                                 const at::Tensor& self,
                                 const at::Tensor& target,
                                 const c10::optional<at::Tensor>& weight,
                                 int64_t reduction, int64_t ignore_index,
                                 const at::Tensor& total_weight) {
  TT_KERNEL(OpName::kNllLossBackward, param_keys,
            (grad_output, self, target, weight, reduction, ignore_index,
             total_weight),
            {
              TT_THROW_IF_ERROR(
                  ValidateNllLossCommon(self, target, weight, reduction));
              TT_THROW_IF_ERROR(ValidateNllLossNdShapes(self, target));
              TT_ASSIGN_OR_THROW(
                  auto grad_input_buf,
                  DispatchNllLossBackwardBuffer(
                      grad_output, self, target, weight, reduction,
                      ignore_index, total_weight, std::move(param_keys)));
              return MakeTensor(std::move(grad_input_buf));
            });
}

class NllLossNdFunction : public torch::autograd::Function<NllLossNdFunction> {
 public:
  static at::Tensor forward(torch::autograd::AutogradContext* ctx,
                            const at::Tensor& self, const at::Tensor& target,
                            const c10::optional<at::Tensor>& weight,
                            int64_t reduction, int64_t ignore_index) {
    at::AutoDispatchBelowAutograd guard;
    auto [output, total_weight] = NllLossNdForwardWithTotalWeight(
        self, target, weight, reduction, ignore_index);
    const at::Tensor weight_tensor =
        (weight.has_value() && weight->defined()) ? *weight : at::Tensor();
    ctx->save_for_backward({self, target, weight_tensor, total_weight});
    ctx->saved_data["reduction"] = reduction;
    ctx->saved_data["ignore_index"] = ignore_index;
    return output;
  }

  static torch::autograd::variable_list backward(
      torch::autograd::AutogradContext* ctx,
      torch::autograd::variable_list grad_outputs) {
    const auto saved = ctx->get_saved_variables();
    const at::Tensor& self = saved[0];
    const at::Tensor& target = saved[1];
    const at::Tensor& weight_tensor = saved[2];
    const at::Tensor& total_weight = saved[3];
    const int64_t reduction = ctx->saved_data["reduction"].toInt();
    const int64_t ignore_index = ctx->saved_data["ignore_index"].toInt();

    const c10::optional<at::Tensor> weight =
        weight_tensor.defined() ? c10::optional<at::Tensor>(weight_tensor)
                                : c10::nullopt;
    at::Tensor grad_self;  // UNINITIALIZED_TENSOR_OK=Assigned conditionally or
                           // returned as undefined gradient.
    if (ctx->needs_input_grad(0)) {
      grad_self = NllLossBackwardKernel(grad_outputs[0], self, target, weight,
                                        reduction, ignore_index, total_weight);
    }
    return {grad_self, at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor()};
  }
};

}  // namespace

bool IsFakeOrFunctorchLossInputs(const at::Tensor& self,
                                 const at::Tensor& target,
                                 const c10::optional<at::Tensor>& weight) {
  return IsFakeOrFunctorchTensor(self) || IsFakeOrFunctorchTensor(target) ||
         (weight.has_value() && weight->defined() &&
          IsFakeOrFunctorchTensor(*weight));
}

bool IsSupportedLossFloatType(at::ScalarType scalar_type) {
  return scalar_type == at::ScalarType::Half ||
         scalar_type == at::ScalarType::BFloat16 ||
         scalar_type == at::ScalarType::Float ||
         scalar_type == at::ScalarType::Double ||
         scalar_type == at::ScalarType::Float8_e5m2 ||
         scalar_type == at::ScalarType::Float8_e4m3fn ||
         scalar_type == at::ScalarType::Float8_e5m2fnuz ||
         scalar_type == at::ScalarType::Float8_e4m3fnuz ||
         scalar_type == at::ScalarType::Float8_e8m0fnu ||
         scalar_type == at::ScalarType::Float4_e2m1fn_x2;
}

bool IsLongOrByteTarget(const at::Tensor& tensor) {
  const at::ScalarType scalar_type = tensor.scalar_type();
  return scalar_type == at::ScalarType::Long ||
         scalar_type == at::ScalarType::Byte;
}

std::tuple<at::Tensor&, at::Tensor&> AtenNllLossForwardOut(
    const at::Tensor& self, const at::Tensor& target,
    const c10::optional<at::Tensor>& weight, int64_t reduction,
    int64_t ignore_index, at::Tensor& output, at::Tensor& total_weight) {
  TT_KERNEL(
      OpName::kNllLossForwardOut, param_keys,
      (self, target, weight, reduction, ignore_index, output, total_weight), {
        TT_THROW_IF_ERROR(
            ValidateNllLossCommon(self, target, weight, reduction));
        TT_THROW_IF_ERROR(ValidateNllLoss1dOr2dShapes(self, target));
        TT_THROW_IF_ERROR(DispatchNllLossForwardOut(
            self, target, weight, reduction, ignore_index, output, total_weight,
            std::move(param_keys)));
        return std::tie(output, total_weight);
      });
}

std::tuple<at::Tensor, at::Tensor> AtenNllLoss2dForward(
    const at::Tensor& self, const at::Tensor& target,
    const c10::optional<at::Tensor>& weight, int64_t reduction,
    int64_t ignore_index) {
  TT_KERNEL(OpName::kNllLoss2dForward, param_keys,
            (self, target, weight, reduction, ignore_index), {
              TT_THROW_IF_ERROR(
                  ValidateNllLossCommon(self, target, weight, reduction));
              TT_THROW_IF_ERROR(ValidateNllLoss2dForwardShapes(self, target));
              TT_ASSIGN_OR_THROW(auto output_bufs,
                                 DispatchNllLossForwardBuffers(
                                     self, target, weight, reduction,
                                     ignore_index, std::move(param_keys)));
              return {MakeTensor(std::move(output_bufs[0])),
                      MakeTensor(std::move(output_bufs[1]))};
            });
}

std::tuple<at::Tensor&, at::Tensor&> AtenNllLoss2dForwardOut(
    const at::Tensor& self, const at::Tensor& target,
    const c10::optional<at::Tensor>& weight, int64_t reduction,
    int64_t ignore_index, at::Tensor& output, at::Tensor& total_weight) {
  TT_KERNEL(
      OpName::kNllLoss2dForwardOut, param_keys,
      (self, target, weight, reduction, ignore_index, output, total_weight), {
        TT_THROW_IF_ERROR(
            ValidateNllLossCommon(self, target, weight, reduction));
        TT_THROW_IF_ERROR(ValidateNllLoss2dForwardShapes(self, target));
        TT_THROW_IF_ERROR(DispatchNllLossForwardOut(
            self, target, weight, reduction, ignore_index, output, total_weight,
            std::move(param_keys)));
        return std::tie(output, total_weight);
      });
}

at::Tensor& AtenNllLossBackwardGradInput(
    const at::Tensor& grad_output, const at::Tensor& self,
    const at::Tensor& target, const c10::optional<at::Tensor>& weight,
    int64_t reduction, int64_t ignore_index, const at::Tensor& total_weight,
    at::Tensor& grad_input) {
  TT_KERNEL(OpName::kNllLossBackwardGradInput, param_keys,
            (grad_output, self, target, weight, reduction, ignore_index,
             total_weight, grad_input),
            {
              TT_THROW_IF_ERROR(
                  ValidateNllLossCommon(self, target, weight, reduction));
              TT_THROW_IF_ERROR(ValidateNllLoss1dOr2dShapes(self, target));
              TT_THROW_IF_ERROR(DispatchNllLossBackwardOut(
                  grad_output, self, target, weight, reduction, ignore_index,
                  total_weight, grad_input, std::move(param_keys)));
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
        TT_THROW_IF_ERROR(
            ValidateNllLossCommon(self, target, weight, reduction));
        TT_THROW_IF_ERROR(ValidateNllLoss2dForwardShapes(self, target));
        TT_ASSIGN_OR_THROW(
            auto grad_input_buf,
            DispatchNllLossBackwardBuffer(grad_output, self, target, weight,
                                          reduction, ignore_index, total_weight,
                                          std::move(param_keys)));
        return MakeTensor(std::move(grad_input_buf));
      });
}

at::Tensor& AtenNllLoss2dBackwardGradInput(
    const at::Tensor& grad_output, const at::Tensor& self,
    const at::Tensor& target, const c10::optional<at::Tensor>& weight,
    int64_t reduction, int64_t ignore_index, const at::Tensor& total_weight,
    at::Tensor& grad_input) {
  TT_KERNEL(OpName::kNllLoss2dBackwardGradInput, param_keys,
            (grad_output, self, target, weight, reduction, ignore_index,
             total_weight, grad_input),
            {
              TT_CHECK_THROW(self.dim() == 4, error::kInvalidArgument)
                  << "expected input to be a 4D tensor, got " << self.dim()
                  << "D";
              TT_CHECK_THROW(target.dim() == 3, error::kInvalidArgument)
                  << "expected target to be a 3D tensor, got " << target.dim()
                  << "D";
              TT_THROW_IF_ERROR(
                  ValidateNllLossCommon(self, target, weight, reduction));
              TT_THROW_IF_ERROR(ValidateNllLoss2dForwardShapes(self, target));
              TT_THROW_IF_ERROR(DispatchNllLossBackwardOut(
                  grad_output, self, target, weight, reduction, ignore_index,
                  total_weight, grad_input, std::move(param_keys)));
              return grad_input;
            });
}

at::Tensor AtenNllLossNd(const at::Tensor& self, const at::Tensor& target,
                         const c10::optional<at::Tensor>& weight,
                         int64_t reduction, int64_t ignore_index) {
  if (IsFakeOrFunctorchLossInputs(self, target, weight)) {
    return at::native::nll_loss_nd_symint(self, target, weight, reduction,
                                          ignore_index);
  }
  return std::get<0>(NllLossNdForwardWithTotalWeight(self, target, weight,
                                                     reduction, ignore_index));
}

at::Tensor AtenNllLossNdAutograd(const at::Tensor& self,
                                 const at::Tensor& target,
                                 const c10::optional<at::Tensor>& weight,
                                 int64_t reduction, int64_t ignore_index) {
  if (IsFakeOrFunctorchLossInputs(self, target, weight)) {
    return at::native::nll_loss_nd_symint(self, target, weight, reduction,
                                          ignore_index);
  }
  if (!c10::GradMode::is_enabled() || !self.requires_grad()) {
    at::AutoDispatchBelowAutograd guard;
    return AtenNllLossNd(self, target, weight, reduction, ignore_index);
  }
  return NllLossNdFunction::apply(self, target, weight, reduction,
                                  ignore_index);
}

}  // namespace torch_tpu
