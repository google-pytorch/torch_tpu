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

#include "csrc/ops/histc/histc_aten_kernels.h"

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
#include "c10/core/ScalarType.h"
#include "csrc/common/aten_utils.h"
#include "csrc/common/cache_key.h"
#include "csrc/common/device_type.h"
#include "csrc/common/dimension_types.h"
#include "csrc/common/dtype.h"
#include "csrc/common/error_utils.h"
#include "csrc/common/fixed_size_span.h"
#include "csrc/common/to_string.h"
#include "csrc/common/warnings.h"
#include "csrc/eager/op_dispatcher.h"
#include "csrc/ops/histc/histc.h"
#include "csrc/ops/macros/kernel.h"
#include "csrc/ops/op_builder_utils.h"
#include "csrc/ops/op_names.h"
#include "csrc/ops/resize/resize_aten_kernels.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "torch/headeronly/core/ScalarType.h"

namespace torch_tpu {

namespace {

enum class BoundsMode {
  kExplicit,
  kFromInput,
};

absl::Status ValidateScalarBoundsTypes(const at::Scalar& min,
                                       const at::Scalar& max) {
  const bool min_valid =
      min.isFloatingPoint() || min.isIntegral(/*includeBool=*/false);
  const bool max_valid =
      max.isFloatingPoint() || max.isIntegral(/*includeBool=*/false);
  TT_RET_CHECK(min_valid && max_valid, error::kInvalidArgument)
      << "expected min and max to be float or int type, got " << min.type()
      << " and " << max.type();
  return absl::OkStatus();
}

// Precondition: ValidateScalarBoundsTypes(a, b) succeeded.
bool ScalarLt(const at::Scalar& a, const at::Scalar& b) {
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
    if (a.type() == c10::ScalarType::Long) {
      const int64_t a_val = a.to<int64_t>();
      return a_val < 0 || static_cast<uint64_t>(a_val) < b.to<uint64_t>();
    }
    const int64_t b_val = b.to<int64_t>();
    return b_val >= 0 && a.to<uint64_t>() < static_cast<uint64_t>(b_val);
  }
  return a.to<double>() < b.to<double>();
}

// Precondition: ValidateScalarBoundsTypes(a, b) succeeded.
bool ScalarEq(const at::Scalar& a, const at::Scalar& b) {
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
    if (a.type() == c10::ScalarType::Long) {
      const int64_t a_val = a.to<int64_t>();
      return a_val >= 0 && static_cast<uint64_t>(a_val) == b.to<uint64_t>();
    }
    const int64_t b_val = b.to<int64_t>();
    return b_val >= 0 && a.to<uint64_t>() == static_cast<uint64_t>(b_val);
  }
  return a.to<double>() == b.to<double>();
}

absl::Status ValidateScalarBoundsFinite(const at::Scalar& min,
                                        const at::Scalar& max) {
  if (min.isFloatingPoint() || max.isFloatingPoint()) {
    const double min_val = min.to<double>();
    const double max_val = max.to<double>();
    TT_RET_CHECK(!(std::isinf(min_val) || std::isinf(max_val) ||
                   std::isnan(min_val) || std::isnan(max_val)),
                 error::kInvalidArgument)
        << "expected min and max to be finite, got " << min << " and " << max
        << ". Either make sure that the input data is finite, or provide valid "
           "finite bounds";
  }
  return absl::OkStatus();
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
  TT_RETURN_IF_ERROR(ValidateScalarBoundsTypes(min, max));
  TT_RETURN_IF_ERROR(ValidateScalarBoundsFinite(min, max));

  // Check for overflows.
  if (min_prev.has_value() && max_prev.has_value()) {
    TT_RETURN_IF_ERROR(ValidateScalarBoundsTypes(*min_prev, *max_prev));
    // If min_prev < max_prev and min > max, there's an overflow.
    const bool min_prev_lt_max_prev = ScalarLt(*min_prev, *max_prev);
    const bool max_lt_min = ScalarLt(max, min);
    TT_RET_CHECK(min_prev_lt_max_prev && max_lt_min, error::kInvalidArgument)
        << "expected min and max to be within the range of their data "
           "types, got min = "
        << min << " and max = " << max << ". "
        << "This happened because min and max were adjusted by one (due to min "
           "== max), which resulted in an overflow";
  }

  // Check that min < max.
  TT_RET_CHECK(ScalarLt(min, max), error::kInvalidArgument)
      << "expected min <= max, got " << min << " vs " << max;

  return absl::OkStatus();
}

absl::Status ValidateHistcInputTensor(const at::Tensor& self) {
  TT_RET_CHECK(IsPrivateUse1Device(self), error::kInvalidArgument)
      << "expected tensor to be on " << GetPrivateUse1DeviceDebugName()
      << ", got " << self.device();

  TT_RET_CHECK(!IsComplex(self), error::kInvalidArgument)
      << "expected the first argument not to be complex, got "
      << ToString(self.scalar_type());

  TT_RET_CHECK(self.scalar_type() != at::kBFloat16 && !IsBool(self),
               error::kPythonNotImplementedError)
      << "not implemented for " << ToString(self.scalar_type());

  return absl::OkStatus();
}

absl::StatusOr<BoundsMode> ValidateHistcBinsAndBounds(const at::Tensor& self,
                                                      const int64_t bins,
                                                      const at::Scalar& min,
                                                      const at::Scalar& max) {
  TT_RET_CHECK(bins > 0, error::kInvalidArgument)
      << "expected bins > 0, got " << bins;

  TT_RETURN_IF_ERROR(ValidateScalarBoundsTypes(min, max));
  const bool min_eq_max = ScalarEq(min, max);
  if (!min_eq_max) {
    // Validate the bounds provided as arguments to torch.histc.
    TT_RETURN_IF_ERROR(ValidateBounds(min, max));
    return BoundsMode::kExplicit;
  }
  if (self.numel() == 0) {
    // On empty input with min == max, PyTorch adjusts to [min - 1, max + 1],
    // which still requires min and max to be finite.
    TT_RETURN_IF_ERROR(ValidateScalarBoundsFinite(min, max));
    return BoundsMode::kExplicit;
  }
  return BoundsMode::kFromInput;
}

absl::Status HistcComputeIntoOut(const at::Tensor& self, const int64_t bins,
                                 const at::Scalar& min, const at::Scalar& max,
                                 const BoundsMode bounds_mode,
                                 PromotedScalar& promoted_min,
                                 PromotedScalar& promoted_max, at::Tensor& out,
                                 OpParamCacheKeys param_keys) {
  const c10::ScalarType input_dtype = self.scalar_type();
  TT_ASSIGN_OR_RETURN(at::Tensor min_val, promoted_min.GetTensor(input_dtype));
  TT_ASSIGN_OR_RETURN(at::Tensor max_val, promoted_max.GetTensor(input_dtype));

  // If input is empty, return tensor of zeros.
  if (self.numel() == 0) {
    TT_RETURN_IF_ERROR(ResizeTensorIfShapeDiffers(out, {bins}));
    out.zero_();
    return absl::OkStatus();
  }

  TT_ASSIGN_OR_RETURN(const auto mlir_input_dtype,
                      ConvertTo<mlir::ElementType>(input_dtype));

  if (bounds_mode == BoundsMode::kFromInput) {
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

    TT_RETURN_IF_ERROR((DispatchOpOut<1, 2>(
        std::move(op_builder), self, {min_val, max_val},
        {.out_dtypes = {mlir_input_dtype, mlir_input_dtype},
         .out_dims_list = {Dimensions(), Dimensions()},
         .op_param_cache_keys = std::move(compute_bounds_param_keys)})));

    // Validate computed bounds, as they may be invalid due to underflow
    // and overflow. Because this needs an expensive transfer to CPU, this
    // is only done when debug checks are enabled.
    if (GetEnableDebugChecks()) {
      TT_RETURN_IF_ERROR(
          ValidateBounds(min_val.cpu().item(), max_val.cpu().item(), min, max));
    } else {
      TT_WARN_ONCE(
          WarningCategory::kCorrectness,
          "Bounds computed from input data are not validated by default on "
          "TorchTPU (unlike CUDA/CPU) to prevent an expensive CPU transfer. "
          "Set the TORCH_TPU_ENABLE_DEBUG_CHECKS environment variable to \"1\" "
          "to enable this check, which incurs a performance penalty.");
    }
  }

  // Perform the actual histogram computation.
  TT_ASSIGN_OR_RETURN(const auto mlir_out_dtype,
                      ConvertTo<mlir::ElementType>(out.scalar_type()));
  auto op_builder = [bins](FixedSizeSpan<mlir::MlirOp, 3> inputs)
      -> absl::StatusOr<mlir::MlirOp> {
    auto& [input, min, max] = inputs;
    return BuildHistcShlo(input, bins, min, max);
  };
  TT_RETURN_IF_ERROR(
      DispatchOpOut<3>(std::move(op_builder), {self, min_val, max_val}, out,
                       {.out_dtype = mlir_out_dtype,
                        .out_dims = {bins},
                        .op_param_cache_keys = std::move(param_keys)}));

  return absl::OkStatus();
}

}  // namespace

at::Tensor AtenHistc(const at::Tensor& self, const int64_t bins,
                     const at::Scalar& min, const at::Scalar& max) {
  auto promoted_min = PromoteScalar(min);
  auto promoted_max = PromoteScalar(max);
  TT_KERNEL(
      OpName::kHistc, param_keys, (self, bins, promoted_min, promoted_max), {
        TT_THROW_IF_ERROR(ValidateHistcInputTensor(self));
        TT_ASSIGN_OR_THROW(const BoundsMode bounds_mode,
                           ValidateHistcBinsAndBounds(self, bins, min, max));
        at::Tensor out =
            at::empty({bins}, self.options().dtype(self.scalar_type()));
        TT_THROW_IF_ERROR(HistcComputeIntoOut(self, bins, min, max, bounds_mode,
                                              promoted_min, promoted_max, out,
                                              std::move(param_keys)));
        return out;
      });
}

at::Tensor& AtenHistcOut(const at::Tensor& self, const int64_t bins,
                         const at::Scalar& min, const at::Scalar& max,
                         at::Tensor& out) {
  auto promoted_min = PromoteScalar(min);
  auto promoted_max = PromoteScalar(max);
  TT_KERNEL(
      OpName::kHistcOut, param_keys,
      (self, bins, promoted_min, promoted_max, out), {
        TT_THROW_IF_ERROR(ValidateHistcInputTensor(self));
        TT_CHECK_THROW(IsPrivateUse1Device(out), error::kInvalidArgument)
            << "expected out tensor to be on "
            << GetPrivateUse1DeviceDebugName() << ", got " << out.device();
        TT_THROW_IF_ERROR(ValidateOutDtype(out, self.scalar_type(),
                                           /*allow_out_dtype_cast=*/false));
        TT_ASSIGN_OR_THROW(const BoundsMode bounds_mode,
                           ValidateHistcBinsAndBounds(self, bins, min, max));
        TT_THROW_IF_ERROR(HistcComputeIntoOut(self, bins, min, max, bounds_mode,
                                              promoted_min, promoted_max, out,
                                              std::move(param_keys)));
        return out;
      });
}

}  // namespace torch_tpu
