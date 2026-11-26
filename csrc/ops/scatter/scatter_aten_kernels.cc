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

#include "csrc/ops/scatter/scatter_aten_kernels.h"

#include <cstdint>
#include <optional>
#include <string_view>
#include <utility>

#include "ATen/MemoryOverlap.h"
#include "ATen/core/ATen_fwd.h"
#include "ATen/core/TensorBody.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "c10/core/TensorImpl.h"
#include "csrc/common/aten_utils.h"
#include "csrc/common/cache_key.h"
#include "csrc/common/dimension_types.h"
#include "csrc/common/dtype.h"
#include "csrc/common/error_utils.h"
#include "csrc/common/fixed_size_span.h"
#include "csrc/common/to_string.h"
#include "csrc/common/utils.h"
#include "csrc/eager/op_dispatcher.h"
#include "csrc/ops/copy_from/tpu_to_tpu.h"
#include "csrc/ops/index_utils.h"
#include "csrc/ops/macros/kernel.h"
#include "csrc/ops/op_builder_utils.h"
#include "csrc/ops/op_names.h"
#include "csrc/ops/resize/resize_aten_kernels.h"
#include "csrc/ops/scatter/scatter.h"
#include "mlir/IR/BuiltinTypes.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"

namespace torch_tpu {
namespace {

enum class ScatterSrcKind { kTensor, kScalar };

absl::Status Scatter(const at::Tensor& self, int64_t dim,
                     const at::Tensor& index, const at::Tensor& src,
                     ScatterSrcKind src_kind, ScatterOp reduction_op,
                     ScatterIncludeSelf include_self, at::Tensor& out) {
  TT_RETURN_IF_ERROR(ValidateOutDtype(out, self.scalar_type(),
                                      /*allow_out_dtype_cast=*/false));
  TT_ASSIGN_OR_RETURN(dim, SafeWrapDim(dim, self.dim()));
  if (src_kind == ScatterSrcKind::kTensor) {
    TT_RET_CHECK(self.scalar_type() == src.scalar_type(),
                 error::kInvalidArgument)
        << "expected self and src to have the same dtype, got "
        << ToString(self.scalar_type()) << " and "
        << ToString(src.scalar_type());
  }

  at::assert_no_internal_overlap(out);
  at::assert_no_overlap(out, index);
  if (src_kind == ScatterSrcKind::kTensor) {
    at::assert_no_overlap(out, src);
  }

  // Upstream scatter_shape_check returns early when index is empty, skipping
  // rank and shape validation and leaving out as a copy of self.
  if (index.numel() == 0) {
    TT_RETURN_IF_ERROR(ResizeTensorIfShapeDiffers(out, self.sizes()));
    if (!self.is_same(out) && self.numel() > 0) {
      TT_RETURN_IF_ERROR(CopyTpuToTpu(self, out));
    }
    return absl::OkStatus();
  }

  TT_ASSIGN_OR_RETURN(const auto output_dtype,
                      ConvertTo<mlir::ElementType>(out.scalar_type()));
  Dimensions output_dims = CopyIntVector(self.sizes());

  // 0-dimensional self and index arguments will be cast to 1-dimensional.
  const auto self_rank = self.dim() == 0 ? 1 : self.dim();
  const auto index_rank = index.dim() == 0 ? 1 : index.dim();

  // 0-dimensional src arguments will be broadcast to match the rank of `index`.
  const auto src_rank = (src.dim() == 0) ? index_rank : src.dim();

  TT_RET_CHECK(self_rank == src_rank, error::kInvalidArgument)
      << "expected the self tensor of shape " << ToString(self.sizes())
      << " to have the same rank as the src tensor of shape "
      << ToString(src.sizes()) << ", got " << self.dim() << " vs. "
      << src.dim();
  TT_RET_CHECK(self_rank == index_rank, error::kInvalidArgument)
      << "expected the self tensor of shape " << ToString(self.sizes())
      << " to have the same rank as the index tensor of shape "
      << ToString(index.sizes()) << ", got " << self.dim() << " vs. "
      << index.dim();

  if (auto unexpanded = TryUnexpandZeroStrideIndex(index)) {
    Dimensions squeezed_dims = std::move(unexpanded->squeezed_dims);
    Dimensions bcast_dims = std::move(unexpanded->bcast_dims);
    Dimensions logical_dims = std::move(unexpanded->logical_dims);
    auto scatter_op_builder = [dim, reduction_op, output_dtype, include_self,
                               squeezed_dims, bcast_dims, logical_dims](
                                  FixedSizeSpan<mlir::MlirOp, 3> inputs) {
      auto& [self_op, index_op, src_op] = inputs;
      const auto index_type = GetTensorTypeOrDie(index_op);
      if (index_type.getRank() != static_cast<int64_t>(squeezed_dims.size())) {
        index_op = mlir::stablehlo::Reshape(index_op, squeezed_dims);
      }
      const auto logical_type = mlir::RankedTensorType::get(
          logical_dims, index_type.getElementType());
      mlir::MlirOp expanded_index =
          mlir::stablehlo::BroadcastInDim(logical_type, index_op, bcast_dims);
      return BuildScatterShlo(self_op, dim, expanded_index, src_op,
                              reduction_op, output_dtype, include_self);
    };
    TT_ASSIGN_OR_RETURN(
        auto param_keys,
        TT_MAKE_OP_PARAM_CACHE_KEYS(dim, reduction_op, include_self,
                                    squeezed_dims, bcast_dims, logical_dims));
    return DispatchOpOut<3>(std::move(scatter_op_builder),
                            {self, unexpanded->tensor, src}, out,
                            {.out_dtype = output_dtype,
                             .out_dims = output_dims,
                             .op_param_cache_keys = std::move(param_keys)});
  }

  auto scatter_op_builder = [dim, reduction_op, output_dtype, include_self](
                                FixedSizeSpan<mlir::MlirOp, 3> inputs) {
    auto& [self, index, src] = inputs;
    return BuildScatterShlo(self, dim, index, src, reduction_op, output_dtype,
                            include_self);
  };

  TT_ASSIGN_OR_RETURN(auto param_keys, TT_MAKE_OP_PARAM_CACHE_KEYS(
                                           dim, reduction_op, include_self));
  return DispatchOpOut<3>(std::move(scatter_op_builder), {self, index, src},
                          out,
                          {.out_dtype = output_dtype,
                           .out_dims = output_dims,
                           .op_param_cache_keys = std::move(param_keys)});
}

absl::StatusOr<ScatterOp> ParseScatterOp(
    std::string_view reduction_op,
    ScatterVersion version = ScatterVersion::kV1) {
  if (version == ScatterVersion::kV2) {
    // scatter_reduce.two op only supports sum, prod, mean, amax, and amin
    if (reduction_op == "sum") {
      return ScatterOp::kSum;
    } else if (reduction_op == "prod") {
      return ScatterOp::kProd;
    } else if (reduction_op == "mean") {
      return ScatterOp::kMean;
    } else if (reduction_op == "amax") {
      return ScatterOp::kAmax;
    } else if (reduction_op == "amin") {
      return ScatterOp::kAmin;
    } else {
      return TT_ERROR(error::kInvalidArgument)
             << "expected the reduction_op to be 'sum', 'prod', 'mean', "
                "'amax', or 'amin', got '"
             << reduction_op << "'";
    }
  } else {
    // scatter supports add and multiply.
    if (reduction_op == "add") {
      return ScatterOp::kAdd;
    } else if (reduction_op == "multiply") {
      return ScatterOp::kMul;
    } else {
      return TT_ERROR(error::kInvalidArgument)
             << "expected the reduction_op to be 'add' or 'multiply', got '"
             << reduction_op << "'";
    }
  }
}

}  // namespace

at::Tensor& AtenScatterSrcOut(const at::Tensor& self, int64_t dim,
                              const at::Tensor& index, const at::Tensor& src,
                              at::Tensor& out) {
  TT_KERNEL(
      OpName::kScatterSrcOut, _,
      (self, IgnoreInCacheKey(dim, "delegates to Scatter()"), index, src, out),
      {
        TT_THROW_IF_ERROR(Scatter(self, dim, index, src,
                                  ScatterSrcKind::kTensor, ScatterOp::kReplace,
                                  ScatterIncludeSelf::kYes, out));
        return out;
      });
}

at::Tensor& AtenScatterValueOut(const at::Tensor& self, int64_t dim,
                                const at::Tensor& index,
                                const at::Scalar& value, at::Tensor& out) {
  auto promoted_value = PromoteScalar(value);
  TT_KERNEL(OpName::kScatterValueOut, _,
            (self, IgnoreInCacheKey(dim, "delegates to Scatter()"), index,
             promoted_value, out),
            {
              auto promoted_type = self.scalar_type();
              TT_ASSIGN_OR_THROW(at::Tensor value_tensor,
                                 promoted_value.GetTensor(promoted_type));
              TT_THROW_IF_ERROR(Scatter(
                  self, dim, index, value_tensor, ScatterSrcKind::kScalar,
                  ScatterOp::kReplace, ScatterIncludeSelf::kYes, out));
              return out;
            });
}

at::Tensor& AtenScatterReduceOut(const at::Tensor& self, int64_t dim,
                                 const at::Tensor& index, const at::Tensor& src,
                                 std::string_view reduction_op,
                                 at::Tensor& out) {
  TT_KERNEL(OpName::kScatterReduceOut, _,
            (self, IgnoreInCacheKey(dim, "delegates to Scatter()"), index, src,
             IgnoreInCacheKey(reduction_op, "delegates to Scatter()"), out),
            {
              TT_ASSIGN_OR_THROW(ScatterOp scatter_op,
                                 ParseScatterOp(reduction_op));
              TT_THROW_IF_ERROR(Scatter(self, dim, index, src,
                                        ScatterSrcKind::kTensor, scatter_op,
                                        ScatterIncludeSelf::kYes, out));
              return out;
            });
}

at::Tensor& AtenScatterReduceTwoOut(const at::Tensor& self, int64_t dim,
                                    const at::Tensor& index,
                                    const at::Tensor& src,
                                    std::string_view reduction,
                                    bool include_self, at::Tensor& out) {
  TT_KERNEL(
      OpName::kScatterReduceTwoOut, _,
      (self, IgnoreInCacheKey(dim, "delegates to Scatter()"), index, src,
       IgnoreInCacheKey(reduction, "delegates to Scatter()"),
       IgnoreInCacheKey(include_self, "delegates to Scatter()"), out),
      {
        TT_ASSIGN_OR_THROW(ScatterOp scatter_op,
                           ParseScatterOp(reduction, ScatterVersion::kV2));
        TT_THROW_IF_ERROR(Scatter(
            self, dim, index, src, ScatterSrcKind::kTensor, scatter_op,
            include_self ? ScatterIncludeSelf::kYes : ScatterIncludeSelf::kNo,
            out));
        return out;
      });
}

at::Tensor& AtenScatterValueReduceOut(const at::Tensor& self, int64_t dim,
                                      const at::Tensor& index,
                                      const at::Scalar& value,
                                      std::string_view reduction_op,
                                      at::Tensor& out) {
  auto promoted_value = PromoteScalar(value);
  TT_KERNEL(OpName::kScatterValueReduceOut, _,
            (self, IgnoreInCacheKey(dim, "delegates to Scatter()"), index,
             promoted_value,
             IgnoreInCacheKey(reduction_op, "delegates to Scatter()"), out),
            {
              auto promoted_type = self.scalar_type();
              TT_ASSIGN_OR_THROW(at::Tensor value_tensor,
                                 promoted_value.GetTensor(promoted_type));
              TT_ASSIGN_OR_THROW(ScatterOp scatter_op,
                                 ParseScatterOp(reduction_op));
              TT_THROW_IF_ERROR(Scatter(self, dim, index, value_tensor,
                                        ScatterSrcKind::kScalar, scatter_op,
                                        ScatterIncludeSelf::kYes, out));
              return out;
            });
}

at::Tensor& AtenScatterAddOut(const at::Tensor& self, int64_t dim,
                              const at::Tensor& index, const at::Tensor& src,
                              at::Tensor& out) {
  TT_KERNEL(
      OpName::kScatterAddOut, _,
      (self, IgnoreInCacheKey(dim, "delegates to AtenScatterReduceOut"), index,
       src, out),
      { return AtenScatterReduceOut(self, dim, index, src, "add", out); });
}

}  // namespace torch_tpu
