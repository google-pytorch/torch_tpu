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

#include "torch_tpu/ops/histc/histc_aten_kernels.h"

#include <cmath>
#include <cstdint>
#include <optional>
#include <utility>

#include "ATen/core/ATen_fwd.h"
#include "ATen/core/ScalarType.h"
#include "ATen/core/TensorBody.h"
#include "ATen/ops/empty.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "c10/util/Exception.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "torch/headeronly/core/ScalarType.h"
#include "torch_tpu/common/aten_utils.h"
#include "torch_tpu/common/cache_key.h"
#include "torch_tpu/common/dimension_types.h"
#include "torch_tpu/common/dtype.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/common/fixed_size_span.h"
#include "torch_tpu/common/to_string.h"
#include "torch_tpu/eager/device_buffer.h"
#include "torch_tpu/eager/op_dispatcher.h"
#include "torch_tpu/eager/tensor_to_buffer.h"
#include "torch_tpu/ops/histc/histc.h"
#include "torch_tpu/ops/macros/kernel.h"
#include "torch_tpu/ops/op_builder_utils.h"
#include "torch_tpu/ops/op_names.h"
#include "torch_tpu/ops/resize/resize_aten_kernels.h"

namespace torch_tpu {

namespace {

absl::StatusOr<bool> lt(const at::Scalar& a, const at::Scalar& b) {
  if (a.isFloatingPoint() && b.isFloatingPoint()) {
    return a.to<double>() < b.to<double>();
  }
  if (a.isIntegral(/*includeBool=*/false) &&
      b.isIntegral(/*includeBool=*/false)) {
    if (a.type() == c10::ScalarType::UInt64 &&
        b.type() == c10::ScalarType::UInt64) {
      return a.to<uint64_t>() < b.to<uint64_t>();
    }
    if (a.type() == c10::ScalarType::Long &&
        b.type() == c10::ScalarType::Long) {
      return a.to<int64_t>() < b.to<int64_t>();
    }
  }
  return TT_ERROR(error::kInvalidArgument)
         << "expected min and max to be float or int type, got " << a.type()
         << " and " << b.type();
}

absl::StatusOr<bool> eq(const at::Scalar& a, const at::Scalar& b) {
  if (a.isFloatingPoint() && b.isFloatingPoint()) {
    return a.to<double>() == b.to<double>();
  }
  if (a.isIntegral(/*includeBool=*/false) &&
      b.isIntegral(/*includeBool=*/false)) {
    if (a.type() == c10::ScalarType::UInt64 &&
        b.type() == c10::ScalarType::UInt64) {
      return a.to<uint64_t>() == b.to<uint64_t>();
    }
    if (a.type() == c10::ScalarType::Long &&
        b.type() == c10::ScalarType::Long) {
      return a.to<int64_t>() == b.to<int64_t>();
    }
  }
  return TT_ERROR(error::kInvalidArgument)
         << "expected min and max to be float or int type, got " << a.type()
         << " and " << b.type();
}

// Validates whether the given min and max values are valid. If min_prev
// and max_prev are provided, they contain the min and max values
// provided to torch.histc, while min and max contain values computed from
// the input data. Note that this case only occurs when torch.histc computed
// bounds from the input data.
absl::Status ValidateBounds(
    const at::Scalar& min, const at::Scalar& max,
    const std::optional<at::Scalar>& min_prev = std::nullopt,
    const std::optional<at::Scalar>& max_prev = std::nullopt) {
  // The below code for bounds validation mirrors the logic in the PyTorch GPU
  // kernel implementation.

  // Check for finite.
  if (min.isFloatingPoint() && max.isFloatingPoint()) {
    TT_RET_CHECK(
        !(std::isinf(min.to<double>()) || std::isinf(max.to<double>()) ||
          std::isnan(min.to<double>()) || std::isnan(max.to<double>())),
        error::kInvalidArgument)
        << "expected min and max to be finite, got " << min << " and " << max
        << ". Either make sure that the input data is finite, or provide valid "
           "finite bounds";
  }

  // Check for overflows.
  if (min_prev.has_value() && max_prev.has_value()) {
    // If min_prev < max_prev and min > max, there's an overflow.
    TT_ASSIGN_OR_RETURN(const bool min_prev_lt_max_prev,
                        lt(*min_prev, *max_prev));
    TT_ASSIGN_OR_RETURN(const bool max_lt_min, lt(max, min));
    TT_RET_CHECK(min_prev_lt_max_prev && max_lt_min, error::kInvalidArgument)
        << "expected min and max to be within the range of their data "
           "types, but got min = "
        << min << " and max = " << max << ". "
        << "This happened because min and max were adjusted by one (due to min "
           "== max), which resulted in an overflow";
  }

  // Check that min < max.
  TT_ASSIGN_OR_RETURN(const bool min_lt_max, lt(min, max));
  TT_RET_CHECK(min_lt_max, error::kInvalidArgument)
      << "expected min <= max, got " << min << " vs " << max;

  return absl::OkStatus();
}

absl::Status HistcHelperOut(const at::Tensor& self, const int64_t bins,
                            const at::Scalar& min, const at::Scalar& max,
                            PromotedScalar& promoted_min,
                            PromotedScalar& promoted_max, at::Tensor& out,
                            OpParamCacheKeys param_keys) {
  TT_RETURN_IF_ERROR(ResizeTensorIfShapeDiffers(out, {bins}));
  const c10::ScalarType out_dtype = self.scalar_type();
  TT_ASSIGN_OR_RETURN(at::Tensor min_val, promoted_min.GetTensor(out_dtype));
  TT_ASSIGN_OR_RETURN(at::Tensor max_val, promoted_max.GetTensor(out_dtype));

  TT_RET_CHECK(!IsComplex(self), error::kInvalidArgument)
      << "expected the first argument not to be complex, got "
      << ToString(self.scalar_type());

  const int64_t numel = self.numel();

  // If input is empty, return tensor of zeros.
  if (numel == 0) {
    out.zero_();
    return absl::OkStatus();
  }

  TT_ASSIGN_OR_RETURN(auto mlir_out_dtype,
                      ConvertTo<mlir::ElementType>(out_dtype));

  TT_ASSIGN_OR_RETURN(const bool min_eq_max, eq(min, max));
  if (!(min_eq_max && numel > 0)) {
    // Validate the bounds provided as arguments to torch.histc.
    TT_RETURN_IF_ERROR(ValidateBounds(min, max));
  } else {
    // Compute min/max from input data. This is done when min == max and
    // numel > 0, which happens when the min and max are not provided as
    // arguments to torch.histc and the input tensor is not empty. The
    // PyTorch CPU and GPU kernels differ from the PyTorch documentation,
    // which states that min/max are computed when min == max == 0. To
    // maintain parity, the TPU kernel implementation follows the logic of
    // the GPU/CPU kernels. See the PyTorch GPU kernel implementation for
    // more details:
    // https://github.com/pytorch/pytorch/blob/v2.9.1/aten/src/ATen/native/cuda/SummaryOps.cu#L328
    auto op_builder =
        [](mlir::MlirOp input) -> absl::StatusOr<MlirOpResults<2>> {
      TT_ASSIGN_OR_RETURN(auto min_max_outputs, BuildHistcBoundsShlo(input));
      return {{min_max_outputs.min, min_max_outputs.max}};
    };

    // Distinguish the bound computation from the main computation
    // in the dispatcher.
    const bool compute_bounds = true;
    TT_ASSIGN_OR_RETURN(auto compute_bounds_param_keys,
                        TT_MAKE_OP_PARAM_CACHE_KEYS(compute_bounds));

    TT_ASSIGN_OR_RETURN(
        (auto [input_min, input_max]),
        (DispatchOp<1, 2>(
            std::move(op_builder), self,
            {.out_dtypes = {mlir_out_dtype, mlir_out_dtype},
             .out_dims_list = {Dimensions(), Dimensions()},
             .op_param_cache_keys = std::move(compute_bounds_param_keys)})));
    TT_RETURN_IF_ERROR(AssignBufferToAtTensor(std::move(input_min), min_val));
    TT_RETURN_IF_ERROR(AssignBufferToAtTensor(std::move(input_max), max_val));

    // Validate computed bounds, as they may be invalid due to underflow
    // and overflow. Because this needs an expensive transfer to CPU, this
    // is only done when debug checks are enabled.
    if (GetEnableDebugChecks()) {
      TT_RETURN_IF_ERROR(
          ValidateBounds(min_val.cpu().item(), max_val.cpu().item(), min, max));
    } else {
      TORCH_WARN_ONCE(
          "Bounds computed from input data are not validated by default "
          "on "
          "TorchTPU (unlike CUDA/CPU) to prevent an expensive CPU "
          "transfer. "
          "Set the TORCH_TPU_INTERNAL_ENABLE_DEBUG_CHECKS environment "
          "variable to \"1\" to enable this check, which incurs a "
          "performance penalty.");
    }
  }

  // Perform the actual histogram computation.
  auto op_builder = [bins](FixedSizeSpan<mlir::MlirOp, 3> inputs)
      -> absl::StatusOr<mlir::MlirOp> {
    auto& [input, min, max] = inputs;
    return BuildHistcShlo(input, bins, min, max);
  };
  TT_ASSIGN_OR_RETURN(
      auto result_buf,
      DispatchOp<3>(std::move(op_builder), {self, min_val, max_val},
                    {.out_dtype = mlir_out_dtype,
                     .out_dims = {bins},
                     .op_param_cache_keys = std::move(param_keys)}));

  TT_RETURN_IF_ERROR(AssignBufferToAtTensor(std::move(result_buf), out));
  return absl::OkStatus();
}

}  // namespace

at::Tensor AtenHistc(const at::Tensor& self, const int64_t bins,
                     const at::Scalar& min, const at::Scalar& max) {
  auto promoted_min = PromoteScalar(min);
  auto promoted_max = PromoteScalar(max);
  TT_KERNEL(
      OpName::kHistc, param_keys, (self, bins, promoted_min, promoted_max), {
        c10::ScalarType out_dtype = self.scalar_type();
        at::Tensor out = at::empty({bins}, self.options().dtype(out_dtype));
        TT_THROW_IF_ERROR(HistcHelperOut(self, bins, min, max, promoted_min,
                                         promoted_max, out,
                                         std::move(param_keys)));
        return out;
      });
}

at::Tensor& AtenHistcOut(const at::Tensor& self, const int64_t bins,
                         const at::Scalar& min, const at::Scalar& max,
                         at::Tensor& out) {
  auto promoted_min = PromoteScalar(min);
  auto promoted_max = PromoteScalar(max);
  TT_KERNEL(OpName::kHistcOut, param_keys,
            (self, bins, promoted_min, promoted_max, out), {
              TT_THROW_IF_ERROR(HistcHelperOut(self, bins, min, max,
                                               promoted_min, promoted_max, out,
                                               std::move(param_keys)));
              return out;
            });
}

}  // namespace torch_tpu
