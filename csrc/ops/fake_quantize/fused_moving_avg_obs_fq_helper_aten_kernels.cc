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

#include "csrc/ops/fake_quantize/fused_moving_avg_obs_fq_helper_aten_kernels.h"

#include <array>
#include <cstdint>
#include <limits>
#include <numeric>
#include <optional>
#include <tuple>
#include <utility>

#include "ATen/core/ATen_fwd.h"
#include "ATen/core/TensorBody.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "c10/core/ScalarType.h"
#include "c10/util/ArrayRef.h"
#include "csrc/common/cache_key.h"
#include "csrc/common/dimension_types.h"
#include "csrc/common/dtype.h"
#include "csrc/common/error_utils.h"
#include "csrc/common/fixed_size_span.h"
#include "csrc/common/to_string.h"
#include "csrc/eager/op_dispatcher.h"
#include "csrc/eager/tensor_to_buffer.h"
#include "csrc/ops/a_min_max/a_min_max.h"
#include "csrc/ops/macros/kernel.h"
#include "csrc/ops/op_builder_utils.h"
#include "csrc/ops/op_names.h"
#include "csrc/ops/reductions/reductions.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Types.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/ChloBuilder.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"

namespace torch_tpu {

namespace {

// Validates input tensor preconditions and normalizes ch_axis.
absl::StatusOr<int64_t> ValidateFusedMovingAvgObsFqInputs(
    const at::Tensor& self, int64_t quant_min, int64_t quant_max,
    int64_t ch_axis, bool per_row_fake_quant) {
  TT_RET_CHECK(self.is_floating_point(), error::kInvalidArgument)
      << "expected floating point tensor for self, got "
      << ToString(self.scalar_type());

  TT_RET_CHECK(quant_min < quant_max, error::kInvalidArgument)
      << "expected quant_min to be strictly less than quant_max, got quant_min="
      << quant_min << " and quant_max=" << quant_max;

  int64_t norm_ch_axis = 0;
  if (per_row_fake_quant) {
    TT_RET_CHECK(self.dim() > 0, error::kInvalidArgument)
        << "expected positive tensor rank when per_row_fake_quant is true, got "
           "rank 0";
    TT_ASSIGN_OR_RETURN(norm_ch_axis, SafeWrapDim(ch_axis, self.dim()));
  }
  return norm_ch_axis;
}

// Builds the StableHLO graph for updating running min and max using exponential
// moving average when observer_on is true.
absl::StatusOr<std::array<mlir::MlirOp, 2>> BuildMinMaxObservationShlo(
    mlir::MlirOp self_op, mlir::MlirOp observer_on_op,
    mlir::MlirOp running_min_op, mlir::MlirOp running_max_op,
    double averaging_const, int64_t norm_ch_axis, bool per_row_fake_quant,
    bool is_uninitialized_observer) {
  const mlir::RankedTensorType self_type = GetTensorTypeOrDie(self_op);
  const int64_t rank = self_type.getRank();

  Dimensions reduce_dims;
  if (!per_row_fake_quant || rank == 0) {
    reduce_dims.resize(rank);
    std::iota(reduce_dims.begin(), reduce_dims.end(), 0);
  } else {
    reduce_dims.reserve(rank - 1);
    for (int64_t d = 0; d < rank; ++d) {
      if (d != norm_ch_axis) {
        reduce_dims.push_back(d);
      }
    }
  }

  // Compute min/max of the input tensor along the reduction dimensions.
  mlir::MlirOp curr_min = self_op;
  mlir::MlirOp curr_max = self_op;
  if (!reduce_dims.empty()) {
    TT_ASSIGN_OR_RETURN(
        auto min_max_results,
        BuildFusedAMinMaxShlo(reduce_dims, ReductionMode::kDropDims, self_op));
    curr_min = min_max_results[0];
    curr_max = min_max_results[1];
  }

  if (GetTensorTypeOrDie(curr_min).getRank() !=
      GetTensorTypeOrDie(running_min_op).getRank()) {
    curr_min = mlir::stablehlo::BroadcastInDim(
        GetTensorTypeOrDie(running_min_op), curr_min, {});
    curr_max = mlir::stablehlo::BroadcastInDim(
        GetTensorTypeOrDie(running_max_op), curr_max, {});
  }

  const mlir::ElementType running_elem_type =
      GetElementTypeOrDie(running_min_op);
  TT_ASSIGN_OR_RETURN(curr_min, CastIfNeeded(curr_min, running_elem_type));
  TT_ASSIGN_OR_RETURN(curr_max, CastIfNeeded(curr_max, running_elem_type));

  mlir::MlirOp rmin_input = running_min_op;
  mlir::MlirOp rmax_input = running_max_op;
  // Initialize running statistics if this is the first execution
  // (uninitialized state).
  if (is_uninitialized_observer) {
    rmin_input = MakeConstantLike(running_min_op,
                                  std::numeric_limits<double>::infinity());
    rmax_input = MakeConstantLike(running_max_op,
                                  -std::numeric_limits<double>::infinity());
  }

  // Compute the exponential moving average update.
  auto avg_cst = MakeConstantLike(rmin_input, averaging_const);
  auto min_diff = mlir::stablehlo::Subtract(curr_min, rmin_input);
  auto min_step = mlir::stablehlo::Mul(avg_cst, min_diff);
  auto ema_min = mlir::stablehlo::Add(rmin_input, min_step);

  auto max_diff = mlir::stablehlo::Subtract(curr_max, rmax_input);
  auto max_step = mlir::stablehlo::Mul(avg_cst, max_diff);
  auto ema_max = mlir::stablehlo::Add(rmax_input, max_step);

  // If running statistics are infinite, seed them directly with
  // current min/max.
  auto is_inf_min = mlir::chlo::IsInf(rmin_input);
  auto is_inf_max = mlir::chlo::IsInf(rmax_input);
  auto updated_min = mlir::stablehlo::Select(is_inf_min, curr_min, ema_min);
  auto updated_max = mlir::stablehlo::Select(is_inf_max, curr_max, ema_max);

  // Select updated running statistics only if observer_on is true.
  auto zero_obs = MakeConstantLike(observer_on_op, 0);
  auto is_observer_on = mlir::stablehlo::Compare(
      observer_on_op, zero_obs, mlir::stablehlo::ComparisonDirection::NE);
  auto is_obs_on_bcast = mlir::stablehlo::BroadcastInDim(
      mlir::RankedTensorType::get(
          GetTensorTypeOrDie(rmin_input).getShape(),
          GetTensorTypeOrDie(is_observer_on).getElementType()),
      is_observer_on, {});
  auto final_min =
      mlir::stablehlo::Select(is_obs_on_bcast, updated_min, rmin_input);
  auto final_max =
      mlir::stablehlo::Select(is_obs_on_bcast, updated_max, rmax_input);

  return std::array<mlir::MlirOp, 2>{final_min, final_max};
}

// Builds the StableHLO graph for computing quantization scale and zero point
// from observed minimum and maximum values when fake_quant_on is true.
absl::StatusOr<std::array<mlir::MlirOp, 2>> BuildChooseQParamsShlo(
    mlir::MlirOp min_op, mlir::MlirOp max_op, mlir::MlirOp scale_op,
    mlir::MlirOp zero_point_op, mlir::MlirOp fake_quant_on_op,
    int64_t quant_min, int64_t quant_max, bool symmetric_quant,
    bool is_uninitialized_observer) {
  mlir::MlirOp scale_input = scale_op;
  mlir::MlirOp zp_input = zero_point_op;
  if (is_uninitialized_observer) {
    scale_input = MakeConstantLike(scale_op, 1.0);
    zp_input = MakeConstantLike(zero_point_op, 0);
  }

  auto min_f64 =
      mlir::stablehlo::ConvertElementType(min_op, mlir::ElementType::F64);
  auto max_f64 =
      mlir::stablehlo::ConvertElementType(max_op, mlir::ElementType::F64);
  auto zero_f64 = MakeConstantLike(min_f64, 0.0);

  // Symmetrize the range around zero if symmetric quantization is
  // enabled and range crosses zero.
  if (symmetric_quant) {
    int64_t sym_qmin = -((quant_max - quant_min) / 2 + 1);
    int64_t sym_qmax = (quant_max - quant_min) / 2;
    auto sym_qmin_op = MakeConstantLike(min_f64, static_cast<double>(sym_qmin));
    auto sym_qmax_op = MakeConstantLike(min_f64, static_cast<double>(sym_qmax));

    auto div_min = mlir::stablehlo::Div(min_f64, sym_qmin_op);
    auto abs_min_div = mlir::stablehlo::Abs(div_min);
    auto div_max = mlir::stablehlo::Div(max_f64, sym_qmax_op);
    auto abs_max_div = mlir::stablehlo::Abs(div_max);
    auto max_scale = mlir::stablehlo::Max(abs_min_div, abs_max_div);

    auto sym_min_val = mlir::stablehlo::Mul(max_scale, sym_qmin_op);
    auto sym_max_val = mlir::stablehlo::Mul(max_scale, sym_qmax_op);

    auto min_lt_zero = mlir::stablehlo::Compare(
        min_f64, zero_f64, mlir::stablehlo::ComparisonDirection::LT);
    auto max_gt_zero = mlir::stablehlo::Compare(
        max_f64, zero_f64, mlir::stablehlo::ComparisonDirection::GT);
    auto sym_cond = mlir::stablehlo::And(min_lt_zero, max_gt_zero);

    min_f64 = mlir::stablehlo::Select(sym_cond, sym_min_val, min_f64);
    max_f64 = mlir::stablehlo::Select(sym_cond, sym_max_val, max_f64);
  }

  // Extend the range boundaries to ensure they contain zero.
  min_f64 = mlir::stablehlo::Min(min_f64, zero_f64);
  max_f64 = mlir::stablehlo::Max(max_f64, zero_f64);

  // Compute the quantization scale.
  auto qmin_f64 = MakeConstantLike(min_f64, static_cast<double>(quant_min));
  auto qmax_f64 = MakeConstantLike(min_f64, static_cast<double>(quant_max));
  auto qrange_f64 = mlir::stablehlo::Subtract(qmax_f64, qmin_f64);
  auto range_f64 = mlir::stablehlo::Subtract(max_f64, min_f64);
  auto scale_f64 = mlir::stablehlo::Div(range_f64, qrange_f64);

  const mlir::Type scale_elem_type =
      GetTensorTypeOrDie(scale_op).getElementType();
  auto scale_computed =
      mlir::stablehlo::ConvertElementType(scale_f64, scale_elem_type);

  // Guard scale against zero or division overflow, defaulting to 0.1.
  auto zero_scale = MakeConstantLike(scale_computed, 0.0);
  auto one_scale = MakeConstantLike(scale_computed, 1.0);
  auto is_zero = mlir::stablehlo::Compare(
      scale_computed, zero_scale, mlir::stablehlo::ComparisonDirection::EQ);
  auto recip = mlir::stablehlo::Div(one_scale, scale_computed);
  auto is_recip_inf = mlir::chlo::IsInf(recip);
  auto guard_cond = mlir::stablehlo::Or(is_zero, is_recip_inf);

  auto default_scale = MakeConstantLike(scale_computed, 0.1);
  scale_computed =
      mlir::stablehlo::Select(guard_cond, default_scale, scale_computed);

  // Compute the initial zero point by selecting the endpoint that
  // minimizes arithmetic error.
  auto scale_for_zp = mlir::stablehlo::ConvertElementType(
      scale_computed, mlir::ElementType::F64);
  auto min_div_scale = mlir::stablehlo::Div(min_f64, scale_for_zp);
  auto max_div_scale = mlir::stablehlo::Div(max_f64, scale_for_zp);

  auto zp_from_min = mlir::stablehlo::Subtract(qmin_f64, min_div_scale);
  auto zp_from_max = mlir::stablehlo::Subtract(qmax_f64, max_div_scale);

  auto abs_qmin = mlir::stablehlo::Abs(qmin_f64);
  auto abs_qmax = mlir::stablehlo::Abs(qmax_f64);
  auto abs_min_div = mlir::stablehlo::Abs(min_div_scale);
  auto abs_max_div = mlir::stablehlo::Abs(max_div_scale);

  auto err_min = mlir::stablehlo::Add(abs_qmin, abs_min_div);
  auto err_max = mlir::stablehlo::Add(abs_qmax, abs_max_div);
  auto min_err_lt_max_err = mlir::stablehlo::Compare(
      err_min, err_max, mlir::stablehlo::ComparisonDirection::LT);
  auto initial_zp =
      mlir::stablehlo::Select(min_err_lt_max_err, zp_from_min, zp_from_max);

  // Force the zero point to the midpoint if symmetric quantization is
  // active and range crosses zero.
  if (symmetric_quant) {
    double mid_zp_val = static_cast<double>(quant_min + quant_max) / 2.0;
    auto mid_zp = MakeConstantLike(initial_zp, mid_zp_val);
    auto min_lt_zero = mlir::stablehlo::Compare(
        min_f64, zero_f64, mlir::stablehlo::ComparisonDirection::LT);
    auto max_gt_zero = mlir::stablehlo::Compare(
        max_f64, zero_f64, mlir::stablehlo::ComparisonDirection::GT);
    auto sym_zp_cond = mlir::stablehlo::And(min_lt_zero, max_gt_zero);
    initial_zp = mlir::stablehlo::Select(sym_zp_cond, mid_zp, initial_zp);
  }

  // Round the zero point to nearest integer and clamp to quantization range.
  auto rounded_zp = mlir::stablehlo::RoundNearestEven(initial_zp);
  auto clamped_zp_f64 = mlir::stablehlo::Clamp(qmin_f64, rounded_zp, qmax_f64);
  const mlir::Type zp_elem_type =
      GetTensorTypeOrDie(zero_point_op).getElementType();
  auto zp_computed =
      mlir::stablehlo::ConvertElementType(clamped_zp_f64, zp_elem_type);

  // Select calculated qparams only if fake_quant_on is true.
  auto zero_fq_on = MakeConstantLike(fake_quant_on_op, 0);
  auto is_fake_quant_on = mlir::stablehlo::Compare(
      fake_quant_on_op, zero_fq_on, mlir::stablehlo::ComparisonDirection::NE);
  auto is_fq_on_bcast = mlir::stablehlo::BroadcastInDim(
      mlir::RankedTensorType::get(
          GetTensorTypeOrDie(scale_input).getShape(),
          GetTensorTypeOrDie(is_fake_quant_on).getElementType()),
      is_fake_quant_on, {});
  auto final_scale =
      mlir::stablehlo::Select(is_fq_on_bcast, scale_computed, scale_input);
  auto final_zp =
      mlir::stablehlo::Select(is_fq_on_bcast, zp_computed, zp_input);

  return std::array<mlir::MlirOp, 2>{final_scale, final_zp};
}

// Builds the StableHLO graph for fake quantization and boolean mask generation.
absl::StatusOr<std::array<mlir::MlirOp, 2>> BuildFakeQuantizeAndMaskShlo(
    mlir::MlirOp self_op, mlir::MlirOp scale_op, mlir::MlirOp zero_point_op,
    mlir::MlirOp fake_quant_on_op, int64_t quant_min, int64_t quant_max,
    int64_t norm_ch_axis, bool per_row_fake_quant) {
  const mlir::RankedTensorType self_type = GetTensorTypeOrDie(self_op);
  const mlir::ElementType self_elem_type = GetElementTypeOrDie(self_op);

  // Broadcast scale and zero-point tensors to match the input tensor shape.
  mlir::MlirOp scale_bcast = scale_op;
  mlir::MlirOp zp_bcast = zero_point_op;
  if (per_row_fake_quant && self_type.getRank() > 0) {
    mlir::RankedTensorType scale_type = GetTensorTypeOrDie(scale_op);
    mlir::RankedTensorType scale_bcast_type =
        scale_type.clone(self_type.getShape());
    scale_bcast = mlir::stablehlo::BroadcastInDim(
        scale_bcast_type, scale_op, c10::IntArrayRef{norm_ch_axis});

    mlir::RankedTensorType zp_type = GetTensorTypeOrDie(zero_point_op);
    mlir::RankedTensorType zp_bcast_type = zp_type.clone(self_type.getShape());
    zp_bcast = mlir::stablehlo::BroadcastInDim(zp_bcast_type, zero_point_op,
                                               c10::IntArrayRef{norm_ch_axis});
  } else {
    mlir::MlirOp scale_scalar = scale_op;
    if (GetTensorTypeOrDie(scale_op).getRank() > 0) {
      scale_scalar = mlir::stablehlo::Reshape(
          mlir::RankedTensorType::get(
              {}, GetTensorTypeOrDie(scale_op).getElementType()),
          scale_op);
    }
    scale_bcast = mlir::stablehlo::BroadcastInDim(
        GetTensorTypeOrDie(scale_op).clone(self_type.getShape()), scale_scalar,
        {});

    mlir::MlirOp zp_scalar = zero_point_op;
    if (GetTensorTypeOrDie(zero_point_op).getRank() > 0) {
      zp_scalar = mlir::stablehlo::Reshape(
          mlir::RankedTensorType::get(
              {}, GetTensorTypeOrDie(zero_point_op).getElementType()),
          zero_point_op);
    }
    zp_bcast = mlir::stablehlo::BroadcastInDim(
        GetTensorTypeOrDie(zero_point_op).clone(self_type.getShape()),
        zp_scalar, {});
  }

  TT_ASSIGN_OR_RETURN(scale_bcast, CastIfNeeded(scale_bcast, self_elem_type));
  TT_ASSIGN_OR_RETURN(zp_bcast, CastIfNeeded(zp_bcast, self_elem_type));

  // Perform the fake quantization (scale, round, clamp, and unscale).
  auto x_scaled = mlir::stablehlo::Div(self_op, scale_bcast);
  auto x_rounded = mlir::stablehlo::RoundNearestEven(x_scaled);
  auto x_zp = mlir::stablehlo::Add(x_rounded, zp_bcast);

  auto qmin_cst = MakeConstantLike(self_op, static_cast<double>(quant_min));
  auto qmax_cst = MakeConstantLike(self_op, static_cast<double>(quant_max));
  auto x_clamped = mlir::stablehlo::Clamp(qmin_cst, x_zp, qmax_cst);

  auto x_unscaled = mlir::stablehlo::Subtract(x_clamped, zp_bcast);
  auto x_fq = mlir::stablehlo::Mul(x_unscaled, scale_bcast);

  // Generate boolean mask indicating which elements fell within the
  // quantization range.
  auto ge_qmin = mlir::stablehlo::Compare(
      x_zp, qmin_cst, mlir::stablehlo::ComparisonDirection::GE);
  auto le_qmax = mlir::stablehlo::Compare(
      x_zp, qmax_cst, mlir::stablehlo::ComparisonDirection::LE);
  auto fq_mask = mlir::stablehlo::And(ge_qmin, le_qmax);

  // Select fake quantized output and computed mask only if
  // fake_quant_on is true.
  auto zero_fq_on = MakeConstantLike(fake_quant_on_op, 0);
  auto is_fake_quant_on = mlir::stablehlo::Compare(
      fake_quant_on_op, zero_fq_on, mlir::stablehlo::ComparisonDirection::NE);
  auto is_fq_on_bcast = mlir::stablehlo::BroadcastInDim(
      mlir::RankedTensorType::get(
          self_type.getShape(),
          GetTensorTypeOrDie(is_fake_quant_on).getElementType()),
      is_fake_quant_on, {});

  auto true_mask = MakeConstantLike(self_op, true, mlir::ElementType::PRED);
  auto final_output = mlir::stablehlo::Select(is_fq_on_bcast, x_fq, self_op);
  auto final_mask = mlir::stablehlo::Select(is_fq_on_bcast, fq_mask, true_mask);

  return std::array<mlir::MlirOp, 2>{final_output, final_mask};
}

// Master StableHLO builder composing min/max observation, qparam calculation,
// and fake quantization.
absl::StatusOr<MlirOpResults<6>> BuildFusedMovingAvgObsFqHelperShlo(
    FixedSizeSpan<mlir::MlirOp, 7> inputs, double averaging_const,
    int64_t quant_min, int64_t quant_max, int64_t norm_ch_axis,
    bool per_row_fake_quant, bool symmetric_quant,
    bool is_uninitialized_observer) {
  auto& [self_op, observer_on_op, fake_quant_on_op, running_min_op,
         running_max_op, scale_op, zero_point_op] = inputs;

  if (GetTensorTypeOrDie(observer_on_op).getRank() > 0) {
    observer_on_op = mlir::stablehlo::Reshape(
        mlir::RankedTensorType::get(
            {}, GetTensorTypeOrDie(observer_on_op).getElementType()),
        observer_on_op);
  }
  if (GetTensorTypeOrDie(fake_quant_on_op).getRank() > 0) {
    fake_quant_on_op = mlir::stablehlo::Reshape(
        mlir::RankedTensorType::get(
            {}, GetTensorTypeOrDie(fake_quant_on_op).getElementType()),
        fake_quant_on_op);
  }

  TT_ASSIGN_OR_RETURN(auto min_max_res,
                      BuildMinMaxObservationShlo(
                          self_op, observer_on_op, running_min_op,
                          running_max_op, averaging_const, norm_ch_axis,
                          per_row_fake_quant, is_uninitialized_observer));
  mlir::MlirOp new_running_min = min_max_res[0];
  mlir::MlirOp new_running_max = min_max_res[1];

  TT_ASSIGN_OR_RETURN(auto qparams_res,
                      BuildChooseQParamsShlo(
                          new_running_min, new_running_max, scale_op,
                          zero_point_op, fake_quant_on_op, quant_min, quant_max,
                          symmetric_quant, is_uninitialized_observer));
  mlir::MlirOp new_scale = qparams_res[0];
  mlir::MlirOp new_zero_point = qparams_res[1];

  TT_ASSIGN_OR_RETURN(
      auto fq_res, BuildFakeQuantizeAndMaskShlo(
                       self_op, new_scale, new_zero_point, fake_quant_on_op,
                       quant_min, quant_max, norm_ch_axis, per_row_fake_quant));
  mlir::MlirOp output = fq_res[0];
  mlir::MlirOp mask = fq_res[1];

  return MlirOpResults<6>{output,          mask,      new_running_min,
                          new_running_max, new_scale, new_zero_point};
}

absl::StatusOr<std::tuple<at::Tensor, at::Tensor>>
FusedMovingAvgObsFqHelperImpl(const at::Tensor& self,
                              const at::Tensor& observer_on,
                              const at::Tensor& fake_quant_on,
                              at::Tensor& running_min, at::Tensor& running_max,
                              at::Tensor& scale, at::Tensor& zero_point,
                              double averaging_const, int64_t quant_min,
                              int64_t quant_max, int64_t norm_ch_axis,
                              bool per_row_fake_quant, bool symmetric_quant,
                              OpParamCacheKeys param_keys) {
  bool is_uninitialized = (running_min.numel() == 0);
  Dimensions qparam_shape =
      per_row_fake_quant ? Dimensions{self.size(norm_ch_axis)} : Dimensions{1};
  at::Tensor rmin_in = running_min;
  at::Tensor rmax_in = running_max;
  at::Tensor scale_in = scale;
  at::Tensor zp_in = zero_point;
  // Initialize temporary empty input buffers if running statistics
  // are uninitialized.
  if (is_uninitialized) {
    TT_ASSIGN_OR_RETURN(rmin_in,
                        MakeEmptyTensor(qparam_shape, running_min.scalar_type(),
                                        self.device()));
    TT_ASSIGN_OR_RETURN(rmax_in,
                        MakeEmptyTensor(qparam_shape, running_max.scalar_type(),
                                        self.device()));
    TT_ASSIGN_OR_RETURN(
        scale_in,
        MakeEmptyTensor(qparam_shape, scale.scalar_type(), self.device()));
    TT_ASSIGN_OR_RETURN(
        zp_in,
        MakeEmptyTensor(qparam_shape, zero_point.scalar_type(), self.device()));
  }

  auto op_builder = [averaging_const, quant_min, quant_max, norm_ch_axis,
                     per_row_fake_quant, symmetric_quant,
                     is_uninitialized](FixedSizeSpan<mlir::MlirOp, 7> inputs)
      -> absl::StatusOr<MlirOpResults<6>> {
    return BuildFusedMovingAvgObsFqHelperShlo(
        inputs, averaging_const, quant_min, quant_max, norm_ch_axis,
        per_row_fake_quant, symmetric_quant, is_uninitialized);
  };

  TT_ASSIGN_OR_RETURN(const auto self_dtype,
                      ConvertTo<mlir::ElementType>(self.scalar_type()));
  TT_ASSIGN_OR_RETURN(const auto rmin_dtype,
                      ConvertTo<mlir::ElementType>(running_min.scalar_type()));
  TT_ASSIGN_OR_RETURN(const auto rmax_dtype,
                      ConvertTo<mlir::ElementType>(running_max.scalar_type()));
  TT_ASSIGN_OR_RETURN(const auto scale_dtype,
                      ConvertTo<mlir::ElementType>(scale.scalar_type()));
  TT_ASSIGN_OR_RETURN(const auto zp_dtype,
                      ConvertTo<mlir::ElementType>(zero_point.scalar_type()));

  TT_ASSIGN_OR_RETURN(
      at::Tensor output,
      MakeEmptyTensor(self.sizes(), self.scalar_type(), self.device()));
  TT_ASSIGN_OR_RETURN(at::Tensor mask,
                      MakeEmptyTensor(self.sizes(), at::kBool, self.device()));

  TT_RETURN_IF_ERROR((DispatchOpOut<7, 6>(
      std::move(op_builder),
      {self, observer_on, fake_quant_on, rmin_in, rmax_in, scale_in, zp_in},
      {output, mask, running_min, running_max, scale, zero_point},
      {.out_dtypes = {self_dtype, mlir::ElementType::PRED, rmin_dtype,
                      rmax_dtype, scale_dtype, zp_dtype},
       .out_dims_list = {self.sizes(), self.sizes(), qparam_shape, qparam_shape,
                         qparam_shape, qparam_shape},
       .op_param_cache_keys = std::move(param_keys)})));

  return std::make_tuple(output, mask);
}

}  // namespace

std::tuple<at::Tensor, at::Tensor> FusedMovingAvgObsFqHelper(
    const at::Tensor& self, const at::Tensor& observer_on,
    const at::Tensor& fake_quant_on, at::Tensor& running_min,
    at::Tensor& running_max, at::Tensor& scale, at::Tensor& zero_point,
    double averaging_const, int64_t quant_min, int64_t quant_max,
    int64_t ch_axis, bool per_row_fake_quant, bool symmetric_quant) {
  TT_KERNEL(OpName::kFusedMovingAvgObsFqHelper, param_keys,
            (self, observer_on, fake_quant_on, running_min, running_max, scale,
             zero_point, averaging_const, quant_min, quant_max, ch_axis,
             per_row_fake_quant, symmetric_quant),
            {
              TT_ASSIGN_OR_THROW(
                  int64_t norm_ch_axis,
                  ValidateFusedMovingAvgObsFqInputs(
                      self, quant_min, quant_max, ch_axis, per_row_fake_quant));
              TT_ASSIGN_OR_THROW(
                  auto res,
                  FusedMovingAvgObsFqHelperImpl(
                      self, observer_on, fake_quant_on, running_min,
                      running_max, scale, zero_point, averaging_const,
                      quant_min, quant_max, norm_ch_axis, per_row_fake_quant,
                      symmetric_quant, std::move(param_keys)));
              return res;
            });
}

}  // namespace torch_tpu
