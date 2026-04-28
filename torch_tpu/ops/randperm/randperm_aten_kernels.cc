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

#include "torch_tpu/ops/randperm/randperm_aten_kernels.h"

#include <cstdint>
#include <mutex>
#include <optional>
#include <utility>

#include "absl/status/statusor.h"
#include "mlir/Support/LLVM.h"
#include "ATen/core/ATen_fwd.h"
#include "ATen/core/Generator.h"
#include "ATen/core/TensorBody.h"
#include "torch_tpu/common/dtype.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/eager/device_buffer.h"
#include "torch_tpu/eager/device_gen_impl.h"
#include "torch_tpu/eager/op_dispatcher.h"
#include "torch_tpu/eager/tensor_to_buffer.h"
#include "torch_tpu/ops/macros/kernel.h"
#include "torch_tpu/ops/op_builder_utils.h"
#include "torch_tpu/ops/op_names.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"

namespace torch_tpu {
namespace {

absl::StatusOr<mlir::MlirOp> BuildRandpermShlo(mlir::MlirOp state, int64_t n,
                                               mlir::ElementType output_dtype) {
  const auto rng_input_state_type = GetTensorTypeOrDie(state);
  auto& builder = state.getBuilder();
  auto& op_builder = builder.getOpBuilder();
  auto& ctx = builder.getContext();

  // 1. Iota: [0, 1, ..., n-1], use I64 to avoid overflow
  auto iota_type =
      makeTensorType(ctx, {n}, builder.getOpBuilder().getI64Type());
  auto iota = mlir::stablehlo::Iota(builder, iota_type, 0);

  // 2. Prepare keys: Generate UI64 random numbers as keys for sorting.
  // UI64 is the only dtype StableHLO RngBitGenerator supports.
  auto key_type = makeTensorType(ctx, {n}, mlir::ElementType::UI64);

  // Call rng bit generator using Philox algorithm to generate UI64.
  // We ignore the updated state returned by this op because we compute it
  // independently in BuildRngStateUpdateShlo.
  auto rng_op = mlir::stablehlo::RngBitGeneratorOp::create(
      op_builder, state.getValue().getLoc(), rng_input_state_type, key_type,
      mlir::stablehlo::RngAlgorithmAttr::get(
          op_builder.getContext(), mlir::stablehlo::RngAlgorithm::PHILOX),
      state.getValue());

  mlir::MlirOp keys = mlir::MlirOp(builder, rng_op.getOutput());

  // 3. Sort the keys and values
  // Input: [keys, values] -> Output: [sorted_keys, sorted_values]
  auto comparator = [key_type, iota_type](mlir::RegionBuilder& rb) {
    auto op_builder = rb.getOpBuilder();
    mlir::stablehlo::buildSortComparisonBody(
        {key_type.getElementType(), iota_type.getElementType()},
        mlir::stablehlo::ComparisonDirection::LT,
        /*compareType=*/std::nullopt, &rb.getRegion(), &op_builder);
  };

  auto sorted_keys_and_values = mlir::stablehlo::Sort(
      builder, {keys, iota}, comparator, /*dim*/ 0, /*is_stable=*/false);

  auto result_op = sorted_keys_and_values[1];
  if (output_dtype != mlir::ElementType::I64) {
    auto result_tensor_type = makeTensorType(ctx, {n}, output_dtype);
    result_op =
        mlir::stablehlo::Convert(result_tensor_type, sorted_keys_and_values[1]);
  }

  return result_op;
}
}  // namespace

at::Tensor& AtenRandpermGeneratorOut(c10::SymInt n,
                                     std::optional<at::Generator> generator,
                                     at::Tensor& out) {
  TT_KERNEL(OpName::kRandpermGeneratorOut, param_keys, (n, generator, out), {
    const int64_t n_int = n.guard_int(__FILE__, __LINE__);

    TT_ASSIGN_OR_THROW(const auto output_dtype,
                       ConvertTo<mlir::ElementType>(out.scalar_type()));

    auto gen = at::get_generator_or_default<DeviceGeneratorImpl>(
        generator, GetDefaultDeviceGenerator());

    at::Tensor rng_input_state;  // UNINITIALIZED_TENSOR_OK
    {
      // See Note [Acquire lock when using random generators]
      // NOLINTNEXTLINE
      std::scoped_lock<std::mutex> lock(gen->mutex_);
      // Query the current RNG state and advance it.
      TT_ASSIGN_OR_THROW(rng_input_state,
                         gen->GetAndAdvanceDeviceStateTensor(n_int, 64));
    }

    // Dispatch the actual randperm.
    auto randperm_op_builder =
        [n_int, output_dtype](
            mlir::MlirOp rng_input_state) -> absl::StatusOr<mlir::MlirOp> {
      return BuildRandpermShlo(rng_input_state, n_int, output_dtype);
    };

    TT_ASSIGN_OR_THROW(
        auto output_buf,
        (DispatchOp<1>(std::move(randperm_op_builder), {rng_input_state},
                       {.out_dtype = output_dtype,
                        .out_dims = out.sizes(),
                        .op_param_cache_keys = std::move(param_keys)})));

    TT_THROW_IF_ERROR(AssignBufferToAtTensor(std::move(output_buf), out));
    return out;
  });
}

}  // namespace torch_tpu
