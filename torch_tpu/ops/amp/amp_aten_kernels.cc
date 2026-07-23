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

#include "torch_tpu/ops/amp/amp_aten_kernels.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <utility>
#include <vector>

#include "ATen/core/ATen_fwd.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "c10/core/Device.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Support/LLVM.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"
#include "torch/headeronly/core/Layout.h"
#include "torch/headeronly/core/ScalarType.h"
#include "torch_tpu/common/cache_key.h"
#include "torch_tpu/common/device_type.h"
#include "torch_tpu/common/dimension_types.h"
#include "torch_tpu/common/dtype.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/common/fixed_size_span.h"
#include "torch_tpu/common/to_string.h"
#include "torch_tpu/eager/device_buffer.h"
#include "torch_tpu/eager/op_dispatcher.h"
#include "torch_tpu/eager/tensor_to_buffer.h"
#include "torch_tpu/ops/all_any/all_any.h"
#include "torch_tpu/ops/macros/kernel.h"
#include "torch_tpu/ops/op_builder_utils.h"
#include "torch_tpu/ops/op_names.h"
#include "torch_tpu/ops/reductions/reductions.h"

namespace torch_tpu {
namespace {

absl::StatusOr<std::vector<mlir::MlirOp>>
BuildAmpForeachNonFiniteCheckAndUnscaleShlo(
    absl::Span<const mlir::MlirOp> scaled_grads, mlir::MlirOp found_inf,
    mlir::MlirOp inv_scale) {
  const size_t num_grads = scaled_grads.size();
  std::vector<mlir::MlirOp> results;
  results.reserve(num_grads + 1);

  mlir::MlirOp updated_found_inf = found_inf;

  for (size_t i = 0; i < num_grads; ++i) {
    mlir::MlirOp grad_i = scaled_grads[i];
    const mlir::RankedTensorType grad_type = GetTensorTypeOrDie(grad_i);
    TT_ASSIGN_OR_RETURN(
        mlir::ElementType element_type,
        ConvertTo<mlir::ElementType>(grad_type.getElementType()));

    // Check for non-finite elements.
    mlir::MlirOp is_finite = mlir::stablehlo::IsFinite(grad_i);
    mlir::MlirOp is_non_finite = mlir::stablehlo::Not(is_finite);

    const int64_t rank = grad_type.getRank();
    mlir::MlirOp any_non_finite_float;
    if (rank > 0) {
      Dimensions reduce_dims(rank);
      std::iota(reduce_dims.begin(), reduce_dims.end(), 0);
      TT_ASSIGN_OR_RETURN(
          mlir::MlirOp any_non_finite_bool,
          BuildAnyShlo(is_non_finite, reduce_dims, ReductionMode::kDropDims));
      any_non_finite_float = mlir::stablehlo::ConvertElementType(
          any_non_finite_bool, mlir::ElementType::F32);
    } else {
      any_non_finite_float = mlir::stablehlo::ConvertElementType(
          is_non_finite, mlir::ElementType::F32);
    }

    TT_ASSIGN_OR_RETURN(
        (auto [bcast_found_inf, bcast_any]),
        ApplyBroadcastIfNeeded(updated_found_inf, any_non_finite_float));
    updated_found_inf = mlir::stablehlo::Max(bcast_found_inf, bcast_any);

    // Unscale grad_i.
    TT_ASSIGN_OR_RETURN(mlir::MlirOp inv_scale_cast,
                        CastIfNeeded(inv_scale, element_type));
    TT_ASSIGN_OR_RETURN((auto [bcast_grad, bcast_inv_scale]),
                        ApplyBroadcastIfNeeded(grad_i, inv_scale_cast));
    mlir::MlirOp unscaled_grad =
        mlir::stablehlo::Mul(bcast_grad, bcast_inv_scale);
    results.push_back(unscaled_grad);
  }

  TT_ASSIGN_OR_RETURN((auto [bcast_updated_found_inf, _]),
                      ApplyBroadcastIfNeeded(updated_found_inf, found_inf));
  results.push_back(bcast_updated_found_inf);

  return results;
}

absl::StatusOr<std::vector<mlir::MlirOp>> BuildAmpUpdateScaleShlo(
    mlir::MlirOp current_scale, mlir::MlirOp growth_tracker,
    mlir::MlirOp found_inf, double scale_growth_factor,
    double scale_backoff_factor, int64_t growth_interval) {
  mlir::MlirBuilder& builder = current_scale.getBuilder();

  mlir::MlirOp zero_float =
      MakeScalarConstant(builder, 0.0f, mlir::ElementType::F32);
  mlir::MlirOp backoff_factor_const =
      MakeScalarConstant(builder, static_cast<float>(scale_backoff_factor),
                         mlir::ElementType::F32);
  mlir::MlirOp growth_factor_const = MakeScalarConstant(
      builder, static_cast<float>(scale_growth_factor), mlir::ElementType::F32);
  mlir::MlirOp zero_int = MakeScalarConstant(builder, static_cast<int32_t>(0),
                                             mlir::ElementType::I32);
  mlir::MlirOp one_int = MakeScalarConstant(builder, static_cast<int32_t>(1),
                                            mlir::ElementType::I32);
  mlir::MlirOp growth_interval_const = MakeScalarConstant(
      builder, static_cast<int32_t>(growth_interval), mlir::ElementType::I32);

  // has_inf = found_inf != 0.0
  TT_ASSIGN_OR_RETURN((auto [bcast_found_inf, bcast_zero_float]),
                      ApplyBroadcastIfNeeded(found_inf, zero_float));
  mlir::MlirOp has_inf =
      mlir::stablehlo::Compare(bcast_found_inf, bcast_zero_float,
                               mlir::stablehlo::ComparisonDirection::NE);

  // Branch IF inf found:
  TT_ASSIGN_OR_RETURN(
      (auto [bcast_scale1, bcast_backoff]),
      ApplyBroadcastIfNeeded(current_scale, backoff_factor_const));
  mlir::MlirOp scale_if_inf = mlir::stablehlo::Mul(bcast_scale1, bcast_backoff);
  TT_ASSIGN_OR_RETURN(
      mlir::MlirOp tracker_if_inf,
      BroadcastIfNeeded(zero_int,
                        GetTensorTypeOrDie(growth_tracker).getShape()));

  // Branch IF NO inf found:
  TT_ASSIGN_OR_RETURN((auto [bcast_tracker, bcast_one_int]),
                      ApplyBroadcastIfNeeded(growth_tracker, one_int));
  mlir::MlirOp successful = mlir::stablehlo::Add(bcast_tracker, bcast_one_int);

  TT_ASSIGN_OR_RETURN(
      (auto [bcast_succ, bcast_interval]),
      ApplyBroadcastIfNeeded(successful, growth_interval_const));
  mlir::MlirOp reached_interval = mlir::stablehlo::Compare(
      bcast_succ, bcast_interval, mlir::stablehlo::ComparisonDirection::EQ);

  // Sub-branch IF reached_interval:
  TT_ASSIGN_OR_RETURN(
      (auto [bcast_scale2, bcast_growth]),
      ApplyBroadcastIfNeeded(current_scale, growth_factor_const));
  mlir::MlirOp grew_scale = mlir::stablehlo::Mul(bcast_scale2, bcast_growth);
  mlir::MlirOp is_grew_finite = mlir::stablehlo::IsFinite(grew_scale);

  TT_ASSIGN_OR_RETURN((auto [bcast_grew, bcast_curr]),
                      ApplyBroadcastIfNeeded(grew_scale, current_scale));
  TT_ASSIGN_OR_RETURN((auto [bcast_finite, bcast_grew_sel]),
                      ApplyBroadcastIfNeeded(is_grew_finite, bcast_grew));
  mlir::MlirOp scale_if_reached =
      mlir::stablehlo::Select(bcast_finite, bcast_grew_sel, bcast_curr);
  TT_ASSIGN_OR_RETURN(
      mlir::MlirOp tracker_if_reached,
      BroadcastIfNeeded(zero_int,
                        GetTensorTypeOrDie(growth_tracker).getShape()));

  // Sub-branch IF NOT reached_interval:
  mlir::MlirOp scale_if_not_reached = current_scale;
  mlir::MlirOp tracker_if_not_reached = successful;

  // Combine reached / not reached:
  TT_ASSIGN_OR_RETURN(
      (auto [bcast_reached1, bcast_scale_r]),
      ApplyBroadcastIfNeeded(reached_interval, scale_if_reached));
  TT_ASSIGN_OR_RETURN(
      (auto [bcast_reached1_b, bcast_scale_nr]),
      ApplyBroadcastIfNeeded(bcast_reached1, scale_if_not_reached));
  mlir::MlirOp scale_if_no_inf =
      mlir::stablehlo::Select(bcast_reached1_b, bcast_scale_r, bcast_scale_nr);

  TT_ASSIGN_OR_RETURN(
      (auto [bcast_reached2, bcast_tracker_r]),
      ApplyBroadcastIfNeeded(reached_interval, tracker_if_reached));
  TT_ASSIGN_OR_RETURN(
      (auto [bcast_reached2_b, bcast_tracker_nr]),
      ApplyBroadcastIfNeeded(bcast_reached2, tracker_if_not_reached));
  mlir::MlirOp tracker_if_no_inf = mlir::stablehlo::Select(
      bcast_reached2_b, bcast_tracker_r, bcast_tracker_nr);

  // Combine has_inf / no_inf:
  TT_ASSIGN_OR_RETURN((auto [bcast_has_inf1, bcast_scale_inf]),
                      ApplyBroadcastIfNeeded(has_inf, scale_if_inf));
  TT_ASSIGN_OR_RETURN((auto [bcast_has_inf1_b, bcast_scale_no_inf]),
                      ApplyBroadcastIfNeeded(bcast_has_inf1, scale_if_no_inf));
  mlir::MlirOp new_current_scale = mlir::stablehlo::Select(
      bcast_has_inf1_b, bcast_scale_inf, bcast_scale_no_inf);

  TT_ASSIGN_OR_RETURN((auto [bcast_has_inf2, bcast_tracker_inf]),
                      ApplyBroadcastIfNeeded(has_inf, tracker_if_inf));
  TT_ASSIGN_OR_RETURN(
      (auto [bcast_has_inf2_b, bcast_tracker_no_inf]),
      ApplyBroadcastIfNeeded(bcast_has_inf2, tracker_if_no_inf));
  mlir::MlirOp new_growth_tracker = mlir::stablehlo::Select(
      bcast_has_inf2_b, bcast_tracker_inf, bcast_tracker_no_inf);

  TT_ASSIGN_OR_RETURN((auto [bcast_new_scale, _1]),
                      ApplyBroadcastIfNeeded(new_current_scale, current_scale));
  TT_ASSIGN_OR_RETURN(
      (auto [bcast_new_tracker, _2]),
      ApplyBroadcastIfNeeded(new_growth_tracker, growth_tracker));

  return std::vector<mlir::MlirOp>{bcast_new_scale, bcast_new_tracker};
}

absl::Status AmpForeachNonFiniteCheckAndUnscaleImpl(
    at::TensorList self, at::Tensor& found_inf, const at::Tensor& inv_scale) {
  if (self.empty()) {
    return absl::OkStatus();
  }

  TT_RET_CHECK(inv_scale.device().type() == GetPrivateUse1DeviceType(),
               error::kInvalidArgument)
      << "expected inv_scale to be on tpu device, got " << inv_scale.device();
  TT_RET_CHECK(found_inf.device().type() == GetPrivateUse1DeviceType(),
               error::kInvalidArgument)
      << "expected found_inf to be on tpu device, got " << found_inf.device();
  TT_RET_CHECK(inv_scale.numel() == 1, error::kInvalidArgument)
      << "expected inv_scale to have 1 element, got " << inv_scale.numel();
  TT_RET_CHECK(found_inf.numel() == 1, error::kInvalidArgument)
      << "expected found_inf to have 1 element, got " << found_inf.numel();
  TT_RET_CHECK(inv_scale.scalar_type() == at::ScalarType::Float,
               error::kInvalidArgument)
      << "expected inv_scale dtype to be float, got "
      << ToString(inv_scale.scalar_type());
  TT_RET_CHECK(found_inf.scalar_type() == at::ScalarType::Float,
               error::kInvalidArgument)
      << "expected found_inf dtype to be float, got "
      << ToString(found_inf.scalar_type());

  const at::Device expected_device = self[0].device();
  for (const at::Tensor& t : self) {
    TT_RET_CHECK(t.device().type() == GetPrivateUse1DeviceType(),
                 error::kInvalidArgument)
        << "expected self tensor to be on tpu device, got " << t.device();
    TT_RET_CHECK(  // ERROR_COV_INFEASIBLE=Multi-device checks cannot be
                   // triggered in a single-device environment because PyTorch
                   // device parsing validates device ordinals first.
        t.device() == expected_device, error::kInvalidArgument)
        << "expected self tensors to be on the same device, got " << t.device();
    TT_RET_CHECK(t.layout() == at::kStrided, error::kInvalidArgument)
        << "expected self tensor layout to be strided, got "
        << ToString(t.layout());
  }
  TT_RET_CHECK(  // ERROR_COV_INFEASIBLE=Multi-device checks cannot be triggered
                 // in a single-device environment because PyTorch device
                 // parsing validates device ordinals first.
      inv_scale.device() == expected_device, error::kInvalidArgument)
      << "expected inv_scale to be on the same device as self, got "
      << inv_scale.device();
  TT_RET_CHECK(  // ERROR_COV_INFEASIBLE=Multi-device checks cannot be triggered
                 // in a single-device environment because PyTorch device
                 // parsing validates device ordinals first.
      found_inf.device() == expected_device, error::kInvalidArgument)
      << "expected found_inf to be on the same device as self, got "
      << found_inf.device();

  const size_t num_grads = self.size();
  std::vector<at::Tensor> inputs;
  inputs.reserve(num_grads + 2);
  std::vector<mlir::ElementType> out_dtypes;
  out_dtypes.reserve(num_grads + 1);
  std::vector<absl::Span<const int64_t>> out_dims_list;
  out_dims_list.reserve(num_grads + 1);

  for (const at::Tensor& t : self) {
    inputs.push_back(t);
    TT_ASSIGN_OR_RETURN(mlir::ElementType dt,
                        ConvertTo<mlir::ElementType>(t.scalar_type()));
    out_dtypes.push_back(dt);
    out_dims_list.push_back(t.sizes());
  }

  inputs.push_back(found_inf);
  inputs.push_back(inv_scale);

  TT_ASSIGN_OR_RETURN(mlir::ElementType float_dt,
                      ConvertTo<mlir::ElementType>(at::ScalarType::Float));
  out_dtypes.push_back(float_dt);
  out_dims_list.push_back(found_inf.sizes());

  DispatchOpOptions<kDynamicSize> options = {
      .out_dtypes = out_dtypes,
      .out_dims_list = out_dims_list,
      .op_param_cache_keys = OpParamCacheKeys::Empty()};

  auto op_builder = [num_grads](absl::Span<mlir::MlirOp> op_inputs,
                                mlir::MlirBuilder& builder)
      -> absl::StatusOr<mlir::SmallVector<mlir::MlirOp>> {
    TT_ASSIGN_OR_RETURN(std::vector<mlir::MlirOp> res,
                        BuildAmpForeachNonFiniteCheckAndUnscaleShlo(
                            op_inputs.subspan(0, num_grads),
                            op_inputs[num_grads], op_inputs[num_grads + 1]));
    return mlir::SmallVector<mlir::MlirOp>(res.begin(), res.end());
  };

  TT_ASSIGN_OR_RETURN(std::vector<DeviceBufferRef> result_buffers,
                      (DispatchOp<kDynamicSize, kDynamicSize>(
                          std::move(op_builder), inputs, std::move(options))));

  for (size_t i = 0; i < num_grads; ++i) {
    TT_RETURN_IF_ERROR(AssignBufferToAtTensor(
        std::move(result_buffers[i]), const_cast<at::Tensor&>(self[i])));
  }
  TT_RETURN_IF_ERROR(
      AssignBufferToAtTensor(std::move(result_buffers[num_grads]), found_inf));

  return absl::OkStatus();
}

absl::Status AmpUpdateScaleImpl(at::Tensor& self, at::Tensor& growth_tracker,
                                const at::Tensor& found_inf,
                                double scale_growth_factor,
                                double scale_backoff_factor,
                                int64_t growth_interval,
                                OpParamCacheKeys param_keys) {
  TT_RET_CHECK(self.device().type() == GetPrivateUse1DeviceType(),
               error::kInvalidArgument)
      << "expected self to be on tpu device, got " << self.device();
  TT_RET_CHECK(growth_tracker.device().type() == GetPrivateUse1DeviceType(),
               error::kInvalidArgument)
      << "expected growth_tracker to be on tpu device, got "
      << growth_tracker.device();
  TT_RET_CHECK(found_inf.device().type() == GetPrivateUse1DeviceType(),
               error::kInvalidArgument)
      << "expected found_inf to be on tpu device, got " << found_inf.device();

  TT_RET_CHECK(  // ERROR_COV_INFEASIBLE=Multi-device checks cannot be triggered
                 // in a single-device environment because PyTorch device
                 // parsing validates device ordinals first.
      growth_tracker.device() == self.device(), error::kInvalidArgument)
      << "expected growth_tracker to be on the same device as self, got "
      << growth_tracker.device();
  TT_RET_CHECK(  // ERROR_COV_INFEASIBLE=Multi-device checks cannot be triggered
                 // in a single-device environment because PyTorch device
                 // parsing validates device ordinals first.
      found_inf.device() == self.device(), error::kInvalidArgument)
      << "expected found_inf to be on the same device as self, got "
      << found_inf.device();

  TT_RET_CHECK(self.numel() == 1, error::kInvalidArgument)
      << "expected self to have 1 element, got " << self.numel();
  TT_RET_CHECK(growth_tracker.numel() == 1, error::kInvalidArgument)
      << "expected growth_tracker to have 1 element, got "
      << growth_tracker.numel();
  TT_RET_CHECK(found_inf.numel() == 1, error::kInvalidArgument)
      << "expected found_inf to have 1 element, got " << found_inf.numel();

  TT_RET_CHECK(self.scalar_type() == at::ScalarType::Float,
               error::kInvalidArgument)
      << "expected self dtype to be float, got "
      << ToString(self.scalar_type());
  TT_RET_CHECK(growth_tracker.scalar_type() == at::ScalarType::Int,
               error::kInvalidArgument)
      << "expected growth_tracker dtype to be int32, got "
      << ToString(growth_tracker.scalar_type());
  TT_RET_CHECK(found_inf.scalar_type() == at::ScalarType::Float,
               error::kInvalidArgument)
      << "expected found_inf dtype to be float, got "
      << ToString(found_inf.scalar_type());

  TT_ASSIGN_OR_RETURN(mlir::ElementType float_dt,
                      ConvertTo<mlir::ElementType>(at::ScalarType::Float));
  TT_ASSIGN_OR_RETURN(mlir::ElementType int_dt,
                      ConvertTo<mlir::ElementType>(at::ScalarType::Int));

  const std::array<mlir::ElementType, 2> out_dtypes = {float_dt, int_dt};
  const std::array<absl::Span<const int64_t>, 2> out_dims_list = {
      self.sizes(), growth_tracker.sizes()};

  TT_RETURN_IF_ERROR(
      param_keys.SetParam("scale_growth_factor", scale_growth_factor));
  TT_RETURN_IF_ERROR(
      param_keys.SetParam("scale_backoff_factor", scale_backoff_factor));
  TT_RETURN_IF_ERROR(param_keys.SetParam("growth_interval", growth_interval));

  DispatchOpOptions<2> options = {.out_dtypes = out_dtypes,
                                  .out_dims_list = out_dims_list,
                                  .op_param_cache_keys = std::move(param_keys)};

  auto op_builder = [scale_growth_factor, scale_backoff_factor,
                     growth_interval](FixedSizeSpan<mlir::MlirOp, 3> op_inputs)
      -> absl::StatusOr<std::array<mlir::MlirOp, 2>> {
    TT_ASSIGN_OR_RETURN(
        std::vector<mlir::MlirOp> results,
        BuildAmpUpdateScaleShlo(op_inputs[0], op_inputs[1], op_inputs[2],
                                scale_growth_factor, scale_backoff_factor,
                                growth_interval));
    return std::array<mlir::MlirOp, 2>{results[0], results[1]};
  };

  TT_ASSIGN_OR_RETURN((auto result_buffers),
                      (DispatchOp<3, 2>(std::move(op_builder),
                                        {self, growth_tracker, found_inf},
                                        std::move(options))));

  TT_RETURN_IF_ERROR(
      AssignBufferToAtTensor(std::move(result_buffers[0]), self));
  TT_RETURN_IF_ERROR(
      AssignBufferToAtTensor(std::move(result_buffers[1]), growth_tracker));

  return absl::OkStatus();
}

}  // namespace

void AtenAmpForeachNonFiniteCheckAndUnscale_(at::TensorList self,
                                             at::Tensor& found_inf,
                                             const at::Tensor& inv_scale) {
  TT_KERNEL(OpName::kAmpForeachNonFiniteCheckAndUnscale_, _,
            (self, found_inf, inv_scale), {
              TT_THROW_IF_ERROR(AmpForeachNonFiniteCheckAndUnscaleImpl(
                  self, found_inf, inv_scale));
            });
}

at::Tensor& AtenAmpUpdateScale_(at::Tensor& self, at::Tensor& growth_tracker,
                                const at::Tensor& found_inf,
                                double scale_growth_factor,
                                double scale_backoff_factor,
                                int64_t growth_interval) {
  TT_KERNEL(
      OpName::kAmpUpdateScale_, param_keys,
      (self, growth_tracker, found_inf, scale_growth_factor,
       scale_backoff_factor, growth_interval),
      {
        TT_THROW_IF_ERROR(AmpUpdateScaleImpl(
            self, growth_tracker, found_inf, scale_growth_factor,
            scale_backoff_factor, growth_interval, std::move(param_keys)));
      });
  return self;
}

}  // namespace torch_tpu
