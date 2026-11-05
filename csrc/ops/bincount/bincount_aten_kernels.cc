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

#include "csrc/ops/bincount/bincount_aten_kernels.h"

#include <algorithm>
#include <cstdint>
#include <optional>
#include <utility>

#include "ATen/core/ATen_fwd.h"
#include "ATen/core/TensorBody.h"
#include "ATen/ops/zeros.h"
#include "absl/log/absl_check.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "c10/core/ScalarType.h"
#include "c10/core/SymInt.h"
#include "c10/util/Optional.h"
#include "csrc/common/aten_utils.h"
#include "csrc/common/cache_key.h"
#include "csrc/common/dimension_types.h"
#include "csrc/common/error_utils.h"
#include "csrc/common/fixed_size_span.h"
#include "csrc/common/to_string.h"
#include "csrc/eager/op_dispatcher.h"
#include "csrc/eager/tensor_to_buffer.h"
#include "csrc/ops/a_min_max/a_min_max.h"
#include "csrc/ops/copy_from/copy_common.h"
#include "csrc/ops/copy_from/tpu_to_cpu.h"
#include "csrc/ops/macros/kernel.h"
#include "csrc/ops/op_builder_utils.h"
#include "csrc/ops/op_names.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"
#include "torch/headeronly/core/ScalarType.h"

namespace torch_tpu {

namespace {

absl::StatusOr<mlir::MlirOp> BuildBinCount(
    mlir::MlirOp input, int64_t num_of_bins,
    mlir::ElementType computation_dtype,
    std::optional<mlir::MlirOp> weights = std::nullopt) {
  auto& builder = input.getBuilder();
  auto& ctx = builder.getContext();

  const mlir::RankedTensorType input_type = GetTensorTypeOrDie(input);
  auto input_shape = input_type.getShape();
  ABSL_CHECK_EQ(input_shape.size(), 1)  // CRASH_OK=checked in AtenBinCount
      << "unexpected dimension of input tensor: " << ToString(input_shape);
  int64_t num_inputs = input_shape[0];

  mlir::Type input_dtype = input_type.getElementType();

  // We first clamp the input in [0, num_of_bins - 1].
  auto min = MakeConstantLike(input, 0);
  auto max = MakeConstantLike(input, num_of_bins - 1);
  auto clamped_inputs = mlir::stablehlo::Clamp(min, input, max);

  // Then we use the clamped input as indices for scattering an summable value
  // (1 or weights[i], if weights is defined) to an initially 0 output tensor.
  Dimensions output_dims = {num_of_bins};
  auto empty_output = MakeConstant(builder, 0, computation_dtype, output_dims);
  auto scatter_indices = mlir::stablehlo::BroadcastInDim(
      mlir::RankedTensorType::get({num_inputs, 1}, input_dtype), clamped_inputs,
      {0});
  mlir::MlirOp scatter_update =
      weights ? mlir::stablehlo::ConvertElementType(*weights, computation_dtype)
              : MakeConstantLike(input, 1, computation_dtype);

  // Create a region builder callback for the scatter operation below.
  auto block_type =
      input_type.clone({}, mlir::getElementType(ctx, computation_dtype));
  auto scatter_builder = [block_type](mlir::RegionBuilder& builder) {
    auto arg0 = mlir::Argument(builder, block_type);
    auto arg1 = mlir::Argument(builder, block_type);
    mlir::MlirOp result = mlir::stablehlo::Add(arg0, arg1);
    mlir::stablehlo::Return(builder, {result});
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

  return mlir::stablehlo::Scatter(empty_output, scatter_indices, scatter_update,
                                  scatter_builder,
                                  scatter_dimension_numbers)[0];
};

}  // namespace

at::Tensor AtenBinCount(const at::Tensor& self,
                        const c10::optional<at::Tensor>& weights,
                        c10::SymInt minlength) {
  TT_KERNEL(
      OpName::kBinCount, _,
      (self,
       IgnoreInCacheKey(weights,
                        "Tensors are implicitly tracked by DispatchOp inputs"),
       IgnoreInCacheKey(minlength, "Captured via nbins in param_keys")),
      {
        auto weights_ = SanitizeOptionalTensor(weights);
        int64_t minlength_val = minlength.guard_int(__FILE__, __LINE__);

        TT_CHECK_THROW(minlength_val >= 0, error::kInvalidArgument)
            << "expected non-negative minlength, got " << minlength_val;

        TT_CHECK_THROW(self.dim() == 1, error::kInvalidArgument)
            << "expected one-dimensional input tensor, got " << self.dim()
            << "-dimensional";

        TT_CHECK_THROW(IsInteger(self), error::kInvalidArgument)
            << "bincount only supports 1-d non-negative integral inputs";

        // Shortcut optimization for empty input tensors.
        if (self.defined() && self.numel() == 0) {
          return at::zeros({minlength_val}, self.options().dtype(at::kLong));
        }

        if (weights_) {
          TT_CHECK_THROW(weights_->dim() == 1, error::kInvalidArgument)
              << "expected one-dimensional weights, got " << weights_->dim()
              << "-dimensional";
          TT_CHECK_THROW(weights_->size(0) == self.size(0),
                         error::kInvalidArgument)
              << "expected input and weights lengths to match, got "
              << self.size(0) << " and " << weights_->size(0);
        }

        // Compute the minimum and maximum element in the input using a single
        // fused TPU kernel and copy directly into a stack buffer. This avoids
        // redispatching through the PyTorch dispatcher, avoids creating CPU
        // tensors, and requires only a single host synchronization.
        auto min_max_builder = [](FixedSizeSpan<mlir::MlirOp, 1> inputs)
            -> absl::StatusOr<mlir::MlirOp> {
          auto& builder = inputs[0].getBuilder();
          TT_ASSIGN_OR_RETURN(
              auto min_max_ops,
              BuildFusedAMinMaxShlo(/*dims=*/{0}, ReductionMode::kDropDims,
                                    inputs[0]));
          auto min_i64 = mlir::stablehlo::ConvertElementType(
              min_max_ops[0], mlir::ElementType::I64);
          auto max_i64 = mlir::stablehlo::ConvertElementType(
              min_max_ops[1], mlir::ElementType::I64);
          auto min_1d = mlir::stablehlo::Reshape(builder, {1}, min_i64);
          auto max_1d = mlir::stablehlo::Reshape(builder, {1}, max_i64);
          return mlir::stablehlo::Concatenate(builder, {min_1d, max_1d},
                                              /*dimension=*/0);
        };

        TT_ASSIGN_OR_THROW(DeviceBufferRef min_max_buf,
                           DispatchOp<1>(std::move(min_max_builder), {self},
                                         {.op_name = OpName::kAminmaxOut,
                                          .out_dtype = mlir::ElementType::I64,
                                          .out_dims = {2}}));

        int64_t min_max_vals[2] = {0, 0};
        TT_THROW_IF_ERROR(CopyDeviceBufferRefToCpuPtr(
            min_max_buf, min_max_vals,
            GetTransferSyncMode(/*non_blocking=*/false)));
        int64_t min_val = min_max_vals[0];
        int64_t max_val = min_max_vals[1];

        TT_CHECK_THROW(min_val >= 0, error::kInvalidArgument)
            << "expected non-negative inputs, but got a minimum value of "
            << min_val;

        // Compute the number of bins, i.e., the output size.
        int64_t nbins = std::max(max_val + 1, minlength_val);

        if (weights_) {
          // If weights are provided, PyTorch expects the computation/output
          // dtype to always be torch.float64, unless the weights are
          // torch.float32, in which case that's what's being used.
          mlir::ElementType cdtype =
              weights_->scalar_type() == at::ScalarType::Float
                  ? mlir::ElementType::F32
                  : mlir::ElementType::F64;
          mlir::ElementType output_dtype = cdtype;
          TT_ASSIGN_OR_THROW(auto param_keys,
                             TT_MAKE_OP_PARAM_CACHE_KEYS(nbins, cdtype));
          auto op_builder = [nbins,
                             cdtype](FixedSizeSpan<mlir::MlirOp, 2> inputs)
              -> absl::StatusOr<mlir::MlirOp> {
            auto& [input, weights] = inputs;
            return BuildBinCount(input, nbins, cdtype, weights);
          };
          TT_ASSIGN_OR_THROW(
              auto result_buf,
              DispatchOp<2>(std::move(op_builder), {self, *weights_},
                            {.out_dtype = output_dtype,
                             .out_dims = {nbins},
                             .op_param_cache_keys = std::move(param_keys)}));
          return MakeTensor(std::move(result_buf));
        }

        // If no weights are provided, the computation/output dtype is always
        // torch.int64.
        mlir::ElementType computation_dtype = mlir::ElementType::I64;
        mlir::ElementType output_dtype = computation_dtype;
        TT_ASSIGN_OR_THROW(auto param_keys, TT_MAKE_OP_PARAM_CACHE_KEYS(nbins));
        MlirUnaryOpBuilder op_builder =
            [nbins, computation_dtype](
                mlir::MlirOp input) -> absl::StatusOr<mlir::MlirOp> {
          return BuildBinCount(input, nbins, computation_dtype);
        };
        TT_ASSIGN_OR_THROW(
            auto result_buf,
            DispatchOp<1>(std::move(op_builder), self,
                          {.out_dtype = output_dtype,
                           .out_dims = {nbins},
                           .op_param_cache_keys = std::move(param_keys)}));
        return MakeTensor(std::move(result_buf));
      });
}

}  // namespace torch_tpu
