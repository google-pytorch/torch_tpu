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

#include "csrc/ops/matmul/matmul_aten_kernels.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <optional>
#include <tuple>
#include <utility>

#include "ATen/EmptyTensor.h"
#include "ATen/core/LegacyTypeDispatch.h"
#include "ATen/core/TensorBody.h"
#include "ATen/ops/add.h"
#include "ATen/ops/matmul.h"
#include "ATen/ops/matmul_backward.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "c10/core/ScalarType.h"
#include "c10/util/ArrayRef.h"
#include "c10/util/intrusive_ptr.h"
#include "csrc/common/cache_key.h"
#include "csrc/common/dimension_types.h"
#include "csrc/common/dtype.h"
#include "csrc/common/error_utils.h"
#include "csrc/common/fixed_size_span.h"
#include "csrc/common/to_string.h"
#include "csrc/eager/device_buffer.h"
#include "csrc/eager/op_dispatcher.h"
#include "csrc/eager/tensor_to_buffer.h"
#include "csrc/ops/macros/kernel.h"
#include "csrc/ops/matmul/matmul.h"
#include "csrc/ops/op_builder_utils.h"
#include "csrc/ops/op_names.h"
#include "csrc/ops/precision_context.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "torch/csrc/autograd/function.h"
#include "torch/csrc/autograd/function_hook.h"
#include "torch/csrc/autograd/functions/utils.h"
#include "torch/csrc/autograd/saved_variable.h"
#include "torch/headeronly/core/ScalarType.h"

namespace torch_tpu {
namespace {

bool IsSupportedScalarType(const at::ScalarType scalar_type) {
  return scalar_type != at::kBool && scalar_type != at::kInt &&
         scalar_type != at::kLong;
}

absl::Status ValidateDtypes(const at::Tensor& self, const at::Tensor& other,
                            const std::optional<at::Tensor>& out) {
  TT_RET_CHECK(self.scalar_type() == other.scalar_type(),
               error::kInvalidArgument)
      << "expected both arguments to have the same dtype, got "
      << ToString(self.scalar_type()) << " vs "
      << ToString(other.scalar_type());
  if (out.has_value()) {
    TT_RET_CHECK(out->scalar_type() == self.scalar_type(),
                 error::kInvalidArgument)
        << "expected out tensor to have dtype " << ToString(self.scalar_type())
        << ", got " << ToString(out->scalar_type());
  }
  return absl::OkStatus();
}

absl::StatusOr<Dimensions> InferMatmulShape(const at::Tensor& self,
                                            const at::Tensor& other) {
  const int64_t self_rank = self.dim();
  const int64_t other_rank = other.dim();
  TT_RET_CHECK(self_rank >= 1 && other_rank >= 1, error::kInvalidArgument)
      << "both arguments to matmul need to be at least 1D, but they are "
      << self_rank << "D and " << other_rank << "D";

  if (self_rank == 1 && other_rank == 1) {
    TT_RET_CHECK(self.size(0) == other.size(0), error::kInvalidArgument)
        << "expected both 1D tensors to have the same number of elements, got "
        << self.size(0) << " and " << other.size(0);
    return Dimensions{};
  }
  if (self_rank == 2 && other_rank == 1) {
    TT_RET_CHECK(self.size(1) == other.size(0), error::kInvalidArgument)
        << "expected the last dimension of the first argument (" << self.size(0)
        << "x" << self.size(1)
        << ") to match the dimension of the second argument (" << other.size(0)
        << "), got " << self.size(1) << " vs " << other.size(0);
    return Dimensions{self.size(0)};
  }
  if (self_rank == 1 && other_rank == 2) {
    TT_RET_CHECK(self.size(0) == other.size(0), error::kInvalidArgument)
        << "mat1 and mat2 shapes cannot be multiplied (1x" << self.size(0)
        << " and " << other.size(0) << "x" << other.size(1) << ")";
    return Dimensions{other.size(1)};
  }
  if (self_rank == 2 && other_rank == 2) {
    TT_RET_CHECK(self.size(1) == other.size(0), error::kInvalidArgument)
        << "mat1 and mat2 shapes cannot be multiplied (" << self.size(0) << "x"
        << self.size(1) << " and " << other.size(0) << "x" << other.size(1)
        << ")";
    return Dimensions{self.size(0), other.size(1)};
  }
  if (self_rank >= 3 && other_rank == 1) {
    const int64_t self_k = self.size(self_rank - 1);
    const int64_t other_k = other.size(0);
    TT_RET_CHECK(self_k == other_k, error::kInvalidArgument)
        << "expected the last dimension of the first argument "
        << ToString(self.sizes())
        << " to match the dimension of the second argument "
        << ToString(other.sizes()) << ", got " << self_k << " vs " << other_k;
    return Dimensions(self.sizes().begin(), self.sizes().end() - 1);
  }

  // At least one of self or other has rank >= 3, and other_rank >= 2.
  const auto self_batch =
      self.sizes().slice(0, std::max<int64_t>(0, self_rank - 2));
  const auto other_batch =
      other.sizes().slice(0, std::max<int64_t>(0, other_rank - 2));
  TT_ASSIGN_OR_RETURN(Dimensions output_dims,
                      InferSize(self_batch, other_batch));
  const int64_t self_m = self_rank == 1 ? 1 : self.size(self_rank - 2);
  const int64_t self_k = self.size(self_rank - 1);
  const int64_t other_k = other.size(other_rank - 2);
  const int64_t other_n = other.size(other_rank - 1);
  TT_RET_CHECK(self_k == other_k, error::kInvalidArgument)
      << "mat1 and mat2 shapes cannot be multiplied (" << self_m << "x"
      << self_k << " and " << other_k << "x" << other_n << ")";
  if (self_rank >= 2) {
    output_dims.push_back(self_m);
  }
  output_dims.push_back(other_n);
  return output_dims;
}

absl::StatusOr<Dimensions> ValidateAndInferMatmulOutputShape(
    const at::Tensor& self, const at::Tensor& other,
    const std::optional<at::Tensor>& out) {
  TT_RETURN_IF_ERROR(ValidateDtypes(self, other, out));
  TT_ASSIGN_OR_RETURN(Dimensions output_dims, InferMatmulShape(self, other));

  // For non-dot rank combinations (anything other than 1D x 1D), CUDA
  // short-circuits zero-element inputs before checking supported scalar types;
  // for non-empty inputs and 1D x 1D, CUDA rejects bool, int32, and int64.
  const bool is_empty_non_dot = (self.numel() == 0 || other.numel() == 0) &&
                                !(self.dim() == 1 && other.dim() == 1);
  TT_RET_CHECK(is_empty_non_dot || IsSupportedScalarType(self.scalar_type()),
               error::kPythonNotImplementedError)
      << "not implemented for " << ToString(self.scalar_type());
  return output_dims;
}

absl::Status ValidateMatmulBackwardInputs(
    const at::Tensor& grad, const at::Tensor& self, const at::Tensor& other,
    const std::array<bool, 2> mask,
    const std::optional<at::Tensor>& grad_self_out,
    const std::optional<at::Tensor>& grad_other_out) {
  TT_ASSIGN_OR_RETURN(
      const Dimensions expected_grad_dims,
      ValidateAndInferMatmulOutputShape(self, other, /*out=*/std::nullopt));
  TT_RET_CHECK(grad.scalar_type() == self.scalar_type(),
               error::kInvalidArgument)
      << "expected grad and self to have the same dtype, got "
      << ToString(grad.scalar_type()) << " vs " << ToString(self.scalar_type());
  const c10::IntArrayRef expected_grad_sizes(expected_grad_dims.data(),
                                             expected_grad_dims.size());
  TT_RET_CHECK(grad.sizes() == expected_grad_sizes, error::kInvalidArgument)
      << "expected grad to have shape " << ToString(expected_grad_dims)
      << ", got " << ToString(grad.sizes());
  if (mask[0] && grad_self_out.has_value() && grad_self_out->defined()) {
    TT_RET_CHECK(grad_self_out->scalar_type() == self.scalar_type(),
                 error::kInvalidArgument)
        << "expected grad_self out tensor to have dtype "
        << ToString(self.scalar_type()) << ", got "
        << ToString(grad_self_out->scalar_type());
  }
  if (mask[1] && grad_other_out.has_value() && grad_other_out->defined()) {
    TT_RET_CHECK(grad_other_out->scalar_type() == other.scalar_type(),
                 error::kInvalidArgument)
        << "expected grad_other out tensor to have dtype "
        << ToString(other.scalar_type()) << ", got "
        << ToString(grad_other_out->scalar_type());
  }
  return absl::OkStatus();
}

template <typename ReturnType, typename DispatchFn>
ReturnType DispatchMatmul(const at::Tensor& self, const at::Tensor& other,
                          const std::optional<at::Tensor>& out,
                          OpParamCacheKeys param_keys,
                          DispatchFn&& dispatch_fn) {
  TT_ASSIGN_OR_RETURN(Dimensions output_dims,
                      ValidateAndInferMatmulOutputShape(self, other, out));
  TT_ASSIGN_OR_RETURN(const mlir::ElementType out_dtype_mlir,
                      ConvertTo<mlir::ElementType>(self.scalar_type()));
  TT_ASSIGN_OR_RETURN(const int64_t output_num_elements,
                      NumElements(output_dims));
  const mlir::stablehlo::Precision current_precision =
      GetAndAddPrecisionTo(param_keys);
  auto op_builder = [current_precision, output_num_elements, output_dims,
                     out_dtype_mlir](FixedSizeSpan<mlir::MlirOp, 2> inputs)
      -> absl::StatusOr<mlir::MlirOp> {
    if (output_num_elements == 0) {
      return MakeZeroSizedTensor(inputs[0].getBuilder(), out_dtype_mlir,
                                 AsArrayRef(absl::MakeConstSpan(output_dims)));
    }
    return BuildMatmulShlo(inputs[0], inputs[1], current_precision);
  };
  return std::forward<DispatchFn>(dispatch_fn)(
      std::move(op_builder),
      DispatchOpOptions<1>{.out_dtype = out_dtype_mlir,
                           .out_dims = std::move(output_dims),
                           .op_param_cache_keys = std::move(param_keys)});
}

absl::StatusOr<std::tuple<at::Tensor, at::Tensor>> MatmulBackwardImpl(
    const at::Tensor& grad, const at::Tensor& self, const at::Tensor& other,
    const std::array<bool, 2> mask, OpParamCacheKeys param_keys) {
  TT_RETURN_IF_ERROR(ValidateMatmulBackwardInputs(
      grad, self, other, mask, /*grad_self_out=*/std::nullopt,
      /*grad_other_out=*/std::nullopt));
  if (!mask[0] && !mask[1]) {
    return std::make_tuple(at::Tensor(), at::Tensor());
  }
  TT_ASSIGN_OR_RETURN(const mlir::ElementType self_dtype_mlir,
                      ConvertTo<mlir::ElementType>(self.scalar_type()));
  TT_ASSIGN_OR_RETURN(const mlir::ElementType other_dtype_mlir,
                      ConvertTo<mlir::ElementType>(other.scalar_type()));
  const mlir::stablehlo::Precision current_precision =
      GetAndAddPrecisionTo(param_keys);
  if (mask[0] && !mask[1]) {
    auto op_builder = [current_precision](FixedSizeSpan<mlir::MlirOp, 3> inputs)
        -> absl::StatusOr<mlir::MlirOp> {
      return BuildMatmulGradSelfShlo(inputs[0], inputs[1], inputs[2],
                                     current_precision);
    };
    TT_ASSIGN_OR_RETURN(
        DeviceBufferRef grad_self_buf,
        DispatchOp<3>(std::move(op_builder), {grad, self, other},
                      {.out_dtype = self_dtype_mlir,
                       .out_dims = self.sizes(),
                       .op_param_cache_keys = std::move(param_keys)}));
    return std::make_tuple(MakeTensor(std::move(grad_self_buf)), at::Tensor());
  }
  if (!mask[0] && mask[1]) {
    auto op_builder = [current_precision](FixedSizeSpan<mlir::MlirOp, 3> inputs)
        -> absl::StatusOr<mlir::MlirOp> {
      return BuildMatmulGradOtherShlo(inputs[0], inputs[1], inputs[2],
                                      current_precision);
    };
    TT_ASSIGN_OR_RETURN(
        DeviceBufferRef grad_other_buf,
        DispatchOp<3>(std::move(op_builder), {grad, self, other},
                      {.out_dtype = other_dtype_mlir,
                       .out_dims = other.sizes(),
                       .op_param_cache_keys = std::move(param_keys)}));
    return std::make_tuple(at::Tensor(), MakeTensor(std::move(grad_other_buf)));
  }

  auto op_builder = [current_precision](FixedSizeSpan<mlir::MlirOp, 3> inputs)
      -> absl::StatusOr<MlirOpResults<2>> {
    TT_ASSIGN_OR_RETURN(mlir::MlirOp grad_self_op,
                        BuildMatmulGradSelfShlo(inputs[0], inputs[1], inputs[2],
                                                current_precision));
    TT_ASSIGN_OR_RETURN(mlir::MlirOp grad_other_op,
                        BuildMatmulGradOtherShlo(inputs[0], inputs[1],
                                                 inputs[2], current_precision));
    return MlirOpResults<2>{grad_self_op, grad_other_op};
  };
  const std::array<mlir::ElementType, 2> out_dtypes = {
      self_dtype_mlir,
      other_dtype_mlir,
  };
  const std::array<absl::Span<const int64_t>, 2> out_dims_list = {
      self.sizes(),
      other.sizes(),
  };
  TT_ASSIGN_OR_RETURN(
      auto output_bufs,
      (DispatchOp<3, 2>(std::move(op_builder), {grad, self, other},
                        {.out_dtypes = out_dtypes,
                         .out_dims_list = out_dims_list,
                         .op_param_cache_keys = std::move(param_keys)})));
  return std::make_tuple(MakeTensor(std::move(output_bufs[0])),
                         MakeTensor(std::move(output_bufs[1])));
}

absl::Status MatmulBackwardOutImpl(
    const at::Tensor& grad, const at::Tensor& self, const at::Tensor& other,
    const std::array<bool, 2> mask, at::Tensor& grad_self,
    at::Tensor& grad_other, OpParamCacheKeys param_keys) {
  TT_RETURN_IF_ERROR(ValidateMatmulBackwardInputs(grad, self, other, mask,
                                                  grad_self, grad_other));
  if (!mask[0] && !mask[1]) {
    return absl::OkStatus();
  }
  if (mask[0] && !grad_self.defined()) {
    TT_ASSIGN_OR_RETURN(
        grad_self,
        MakeEmptyTensor(/*size=*/{0}, self.scalar_type(), self.device()));
  }
  if (mask[1] && !grad_other.defined()) {
    TT_ASSIGN_OR_RETURN(
        grad_other,
        MakeEmptyTensor(/*size=*/{0}, other.scalar_type(), other.device()));
  }
  TT_ASSIGN_OR_RETURN(const mlir::ElementType self_dtype_mlir,
                      ConvertTo<mlir::ElementType>(self.scalar_type()));
  TT_ASSIGN_OR_RETURN(const mlir::ElementType other_dtype_mlir,
                      ConvertTo<mlir::ElementType>(other.scalar_type()));
  const mlir::stablehlo::Precision current_precision =
      GetAndAddPrecisionTo(param_keys);
  if (mask[0] && !mask[1]) {
    auto op_builder = [current_precision](FixedSizeSpan<mlir::MlirOp, 3> inputs)
        -> absl::StatusOr<mlir::MlirOp> {
      return BuildMatmulGradSelfShlo(inputs[0], inputs[1], inputs[2],
                                     current_precision);
    };
    return DispatchOpOut<3>(std::move(op_builder), {grad, self, other},
                            grad_self,
                            {.out_dtype = self_dtype_mlir,
                             .out_dims = self.sizes(),
                             .op_param_cache_keys = std::move(param_keys)});
  }
  if (!mask[0] && mask[1]) {
    auto op_builder = [current_precision](FixedSizeSpan<mlir::MlirOp, 3> inputs)
        -> absl::StatusOr<mlir::MlirOp> {
      return BuildMatmulGradOtherShlo(inputs[0], inputs[1], inputs[2],
                                      current_precision);
    };
    return DispatchOpOut<3>(std::move(op_builder), {grad, self, other},
                            grad_other,
                            {.out_dtype = other_dtype_mlir,
                             .out_dims = other.sizes(),
                             .op_param_cache_keys = std::move(param_keys)});
  }

  auto op_builder = [current_precision](FixedSizeSpan<mlir::MlirOp, 3> inputs)
      -> absl::StatusOr<MlirOpResults<2>> {
    TT_ASSIGN_OR_RETURN(mlir::MlirOp grad_self_op,
                        BuildMatmulGradSelfShlo(inputs[0], inputs[1], inputs[2],
                                                current_precision));
    TT_ASSIGN_OR_RETURN(mlir::MlirOp grad_other_op,
                        BuildMatmulGradOtherShlo(inputs[0], inputs[1],
                                                 inputs[2], current_precision));
    return MlirOpResults<2>{grad_self_op, grad_other_op};
  };
  const std::array<mlir::ElementType, 2> out_dtypes = {
      self_dtype_mlir,
      other_dtype_mlir,
  };
  const std::array<absl::Span<const int64_t>, 2> out_dims_list = {
      self.sizes(),
      other.sizes(),
  };
  return DispatchOpOut<3, 2>(std::move(op_builder), {grad, self, other},
                             {grad_self, grad_other},
                             {.out_dtypes = out_dtypes,
                              .out_dims_list = out_dims_list,
                              .op_param_cache_keys = std::move(param_keys)});
}

class MatmulBackwardAutogradNode : public torch::autograd::Node {
 public:
  MatmulBackwardAutogradNode(const at::Tensor& grad, const at::Tensor& self,
                             const at::Tensor& other,
                             const std::array<bool, 2> mask)
      : grad_(grad, /*is_output=*/false),
        self_(self, /*is_output=*/false),
        other_(other, /*is_output=*/false),
        mask_(mask) {}

  static at::Tensor ComputeGradGrad(const at::Tensor& self,
                                    const at::Tensor& other,
                                    const at::Tensor& gg_self,
                                    const at::Tensor& gg_other, bool mask0,
                                    bool mask1) {
    const bool has_first = mask0 && gg_self.defined();
    const bool has_second = mask1 && gg_other.defined();
    if (has_first && has_second) {
      return at::add(at::matmul(gg_self, other), at::matmul(self, gg_other));
    }
    if (has_first) {
      return at::matmul(gg_self, other);
    }
    if (has_second) {
      return at::matmul(self, gg_other);
    }
    return at::Tensor();
  }

  torch::autograd::variable_list apply(
      torch::autograd::variable_list&& grads) override {
    if (grads.size() < 2) {
      return {at::Tensor(), at::Tensor(), at::Tensor()};
    }
    const at::Tensor grad = grad_.unpack();
    const at::Tensor self = self_.unpack();
    const at::Tensor other = other_.unpack();
    const at::Tensor& gg_self = grads[0];
    const at::Tensor& gg_other = grads[1];

    at::Tensor grad_grad = should_compute_output(0)
                               ? ComputeGradGrad(self, other, gg_self, gg_other,
                                                 mask_[0], mask_[1])
                               : at::Tensor();

    at::Tensor grad_self =
        (should_compute_output(1) && mask_[1] && gg_other.defined())
            ? std::get<0>(
                  at::matmul_backward(grad, self, gg_other, {true, false}))
            : at::Tensor();

    at::Tensor grad_other =
        (should_compute_output(2) && mask_[0] && gg_self.defined())
            ? std::get<1>(
                  at::matmul_backward(grad, gg_self, other, {false, true}))
            : at::Tensor();

    return {std::move(grad_grad), std::move(grad_self), std::move(grad_other)};
  }

  void release_variables() override {
    grad_.reset_data();
    self_.reset_data();
    other_.reset_data();
  }

 private:
  torch::autograd::SavedVariable grad_;
  torch::autograd::SavedVariable self_;
  torch::autograd::SavedVariable other_;
  std::array<bool, 2> mask_;
};

class MatmulAutogradNode : public torch::autograd::Node {
 public:
  MatmulAutogradNode(const at::Tensor& self, const at::Tensor& other)
      : self_(self, /*is_output=*/false), other_(other, /*is_output=*/false) {}

  torch::autograd::variable_list apply(
      torch::autograd::variable_list&& grads) override {
    if (grads.empty()) {
      return {at::Tensor(), at::Tensor()};
    }
    const at::Tensor self = self_.unpack();
    const at::Tensor other = other_.unpack();
    const std::array<bool, 2> mask = {should_compute_output(0),
                                      should_compute_output(1)};
    if (!mask[0] && !mask[1]) {
      return {at::Tensor(), at::Tensor()};
    }
    auto [grad_self, grad_other] =
        at::matmul_backward(grads[0], self, other, mask);
    return {std::move(grad_self), std::move(grad_other)};
  }

  void release_variables() override {
    self_.reset_data();
    other_.reset_data();
  }

 private:
  torch::autograd::SavedVariable self_;
  torch::autograd::SavedVariable other_;
};

}  // namespace

at::Tensor AtenMatmul(const at::Tensor& self, const at::Tensor& other) {
  TT_KERNEL(OpName::kMatmul, param_keys, (self, other), {
    TT_ASSIGN_OR_THROW(
        DeviceBufferRef result_buf,
        DispatchMatmul<absl::StatusOr<DeviceBufferRef>>(
            self, other, /*out=*/std::nullopt, std::move(param_keys),
            [&](auto op_builder, auto options) {
              return DispatchOp<2>(std::move(op_builder), {self, other},
                                   std::move(options));
            }));
    return MakeTensor(std::move(result_buf));
  });
}

at::Tensor AtenMatmulAutograd(const at::Tensor& self, const at::Tensor& other) {
  c10::intrusive_ptr<MatmulAutogradNode> grad_fn;
  if (torch::autograd::compute_requires_grad(self, other)) {
    grad_fn = c10::make_intrusive<MatmulAutogradNode>(self, other);
    grad_fn->set_next_edges(torch::autograd::collect_next_edges(self, other));
  }
  at::Tensor result = [&]() {
    at::AutoDispatchBelowAutograd guard;
    return at::matmul(self, other);
  }();
  if (grad_fn) {
    torch::autograd::set_history(result, std::move(grad_fn));
  }
  return result;
}

at::Tensor& AtenMatmulOut(const at::Tensor& self, const at::Tensor& other,
                          at::Tensor& out) {
  TT_KERNEL(OpName::kMatmulOut, param_keys, (self, other, out), {
    TT_THROW_IF_ERROR(DispatchMatmul<absl::Status>(
        self, other, out, std::move(param_keys),
        [&](auto op_builder, auto options) {
          return DispatchOpOut<2>(std::move(op_builder), {self, other}, out,
                                  std::move(options));
        }));
    return out;
  });
}

std::tuple<at::Tensor, at::Tensor> AtenMatmulBackward(
    const at::Tensor& grad, const at::Tensor& self, const at::Tensor& other,
    const std::array<bool, 2> mask) {
  TT_KERNEL(OpName::kMatmulBackward, param_keys, (grad, self, other, mask), {
    TT_ASSIGN_OR_THROW(auto results, MatmulBackwardImpl(grad, self, other, mask,
                                                        std::move(param_keys)));
    return results;
  });
}

std::tuple<at::Tensor&, at::Tensor&> AtenMatmulBackwardOut(
    const at::Tensor& grad, const at::Tensor& self, const at::Tensor& other,
    const std::array<bool, 2> mask, at::Tensor& grad_self,
    at::Tensor& grad_other) {
  TT_KERNEL(OpName::kMatmulBackwardOut, param_keys,
            (grad, self, other, mask, grad_self, grad_other), {
              TT_THROW_IF_ERROR(MatmulBackwardOutImpl(grad, self, other, mask,
                                                      grad_self, grad_other,
                                                      std::move(param_keys)));
              return std::forward_as_tuple(grad_self, grad_other);
            });
}

std::tuple<at::Tensor, at::Tensor> AtenMatmulBackwardAutograd(
    const at::Tensor& grad, const at::Tensor& self, const at::Tensor& other,
    const std::array<bool, 2> mask) {
  c10::intrusive_ptr<MatmulBackwardAutogradNode> grad_fn;
  if (torch::autograd::compute_requires_grad(grad, self, other)) {
    grad_fn = c10::make_intrusive<MatmulBackwardAutogradNode>(grad, self, other,
                                                              mask);
    grad_fn->set_next_edges(
        torch::autograd::collect_next_edges(grad, self, other));
  }
  auto [grad_self, grad_other] = [&]() {
    at::AutoDispatchBelowAutograd guard;
    return at::matmul_backward(grad, self, other, mask);
  }();
  if (grad_fn) {
    torch::autograd::set_history(grad_self, grad_fn);
    torch::autograd::set_history(grad_other, grad_fn);
  }
  return std::make_tuple(std::move(grad_self), std::move(grad_other));
}

std::tuple<at::Tensor, at::Tensor> MetaMatmulBackward(
    const at::Tensor& grad, const at::Tensor& self, const at::Tensor& other,
    const std::array<bool, 2> mask) {
  TT_KERNEL(
      OpName::kMatmulBackward, _,
      (grad, self, other,
       IgnoreInCacheKey(mask, "Meta kernel does not dispatch to XLA")),
      {
        TT_THROW_IF_ERROR(
            ValidateMatmulBackwardInputs(grad, self, other, mask,
                                         /*grad_self_out=*/std::nullopt,
                                         /*grad_other_out=*/std::nullopt));
        at::Tensor grad_self =
            mask[0] ? at::detail::empty_meta(self.sizes(), self.scalar_type(),
                                             std::nullopt, self.device(),
                                             std::nullopt, std::nullopt)
                    : at::Tensor();
        at::Tensor grad_other =
            mask[1] ? at::detail::empty_meta(other.sizes(), other.scalar_type(),
                                             std::nullopt, other.device(),
                                             std::nullopt, std::nullopt)
                    : at::Tensor();
        return std::make_tuple(std::move(grad_self), std::move(grad_other));
      });
}

std::tuple<at::Tensor&, at::Tensor&> MetaMatmulBackwardOut(
    const at::Tensor& grad, const at::Tensor& self, const at::Tensor& other,
    const std::array<bool, 2> mask, at::Tensor& grad_self,
    at::Tensor& grad_other) {
  TT_KERNEL(OpName::kMatmulBackwardOut, _,
            (grad, self, other,
             IgnoreInCacheKey(mask, "Meta kernel does not dispatch to XLA"),
             grad_self, grad_other),
            {
              TT_THROW_IF_ERROR(ValidateMatmulBackwardInputs(
                  grad, self, other, mask, grad_self, grad_other));
              if (mask[0] && grad_self.defined()) {
                grad_self.resize_(self.sizes());
              }
              if (mask[1] && grad_other.defined()) {
                grad_other.resize_(other.sizes());
              }
              return std::forward_as_tuple(grad_self, grad_other);
            });
}

}  // namespace torch_tpu
