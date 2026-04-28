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

#include "torch_tpu/ops/random/random_aten_kernels.h"

#include <cstdint>
#include <limits>
#include <mutex>
#include <optional>
#include <type_traits>
#include <utility>

#include "absl/log/absl_log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "ATen/Context.h"
#include "ATen/Dispatch.h"
#include "ATen/Dispatch_v2.h"
#include "ATen/core/ATen_fwd.h"
#include "ATen/core/Generator.h"
#include "ATen/native/DistributionTemplates.h"
#include "c10/core/ScalarType.h"
#include "c10/core/ScalarTypeToTypeMeta.h"
#include "c10/util/Optional.h"
#include "torch/headeronly/core/Dispatch_v2.h"
#include "torch/headeronly/core/ScalarType.h"
#include "torch_tpu/common/cache_key.h"
#include "torch_tpu/common/dimension_types.h"
#include "torch_tpu/common/dtype.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/common/fixed_size_span.h"
#include "torch_tpu/common/utils.h"
#include "torch_tpu/eager/device_buffer.h"
#include "torch_tpu/eager/device_gen_impl.h"
#include "torch_tpu/eager/op_dispatcher.h"
#include "torch_tpu/eager/tensor_to_buffer.h"
#include "torch_tpu/ops/macros/kernel.h"
#include "torch_tpu/ops/op_builder_utils.h"
#include "torch_tpu/ops/op_names.h"
#include "torch_tpu/ops/random/random.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"

namespace torch_tpu {

namespace {

NAryMlirOpBuilder<1, 2> GetRandomFunctional(Dimensions dims,
                                            mlir::ElementType output_dtype,
                                            int64_t from, int64_t to) {
  return [dims, output_dtype, from,
          to](mlir::MlirOp rng_state) -> absl::StatusOr<MlirOpResults<2>> {
    return BuildRandomShlo(rng_state, dims, output_dtype, from, to);
  };
}

// See https://docs.pytorch.org/docs/stable/generated/torch.Tensor.random_.html
// for the logic behind this.
// Reusing logic from
// torch/aten/src/ATen/native/DistributionTemplates.h
absl::StatusOr<int64_t> ComputeToValue(at::ScalarType dtype) {
  ABSL_VLOG(3) << "ComputeToValue: " << dtype;
  absl::StatusOr<int64_t> maybe_to = TT_ERROR(error::kInvalidArgument)
                                     << "Unsupported dtype for random: "
                                     << dtype;
  if (at::isFloatingType(dtype)) {
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16, dtype,
        "random_float_to_value_calc", [&] {
          constexpr int64_t scalar_t_max =
              static_cast<int64_t>(1) << std::numeric_limits<scalar_t>::digits;
          maybe_to = scalar_t_max > std::numeric_limits<int64_t>::max()
                         ? std::numeric_limits<int64_t>::max()
                         : static_cast<int64_t>(scalar_t_max);
        });
  } else if (at::isIntegralType(dtype, /*includeBool=*/true)) {
    AT_DISPATCH_V2(dtype, "random_integral_to_value_calc", AT_WRAP([&] {
                     if constexpr (std::is_same_v<scalar_t, bool>) {
                       maybe_to = static_cast<int64_t>(true);
                     } else {
                       maybe_to = static_cast<int64_t>(
                           std::numeric_limits<scalar_t>::max());
                     }
                   }),
                   AT_EXPAND(AT_INTEGRAL_TYPES_V2), at::kBool);
  }
  return maybe_to;
}

absl::Status FromToInRange(int64_t from, int64_t to, at::ScalarType dtype) {
  TT_RET_CHECK(from < to, error::kInvalidArgument)
      << "expected 'from' to be < 'to', got " << from << " vs " << to;
  at::native::templates::check_from_to_in_range(
      from, to - 1, c10::scalarTypeToTypeMeta(dtype));
  return absl::OkStatus();
}

absl::Status Random(at::Tensor& self, c10::optional<at::Generator> generator,
                    int64_t from, c10::optional<int64_t> to_opt,
                    OpParamCacheKeys param_keys) {
  if (self.numel() == 0) {
    return absl::OkStatus();
  }
  int64_t to;
  if (to_opt.has_value()) {
    to = to_opt.value();
  } else {
    TT_ASSIGN_OR_RETURN(to, ComputeToValue(self.scalar_type()));
  }
  TT_RETURN_IF_ERROR(FromToInRange(from, to, self.scalar_type()));
  TT_ASSIGN_OR_RETURN(mlir::ElementType output_dtype,
                      ConvertTo<mlir::ElementType>(self.scalar_type()));
  auto gen = at::get_generator_or_default<DeviceGeneratorImpl>(
      generator, GetDefaultDeviceGenerator());

  // See Note [Acquire lock when using random generators]
  // NOLINTNEXTLINE
  std::scoped_lock<std::mutex> lock(gen->mutex_);

  // Since we need to generate random bits, we query for the rng state tensor.
  at::Tensor rng_input_state = gen->DeviceStateTensor();

  auto dims = CopyIntVector(self.sizes());
  TT_ASSIGN_OR_RETURN(
      (auto [rng_output_state_buf, output_buf]),
      (DispatchOp<1, 2>(GetRandomFunctional(dims, output_dtype, from, to),
                        {rng_input_state},
                        {.out_dtypes = {mlir::ElementType::UI64, output_dtype},
                         .out_dims_list = {{2}, self.sizes()},
                         .op_param_cache_keys = std::move(param_keys),
                         .split_mode = OpSplitMode::kSplitAfter})));
  // After the state has been used (and updated) after generating random bits,
  // we give it back to the generator, so that it can be used by other ops in
  // the same graph.
  auto rng_output_state = MakeTensor(std::move(rng_output_state_buf));
  TT_RETURN_IF_ERROR(gen->SetDeviceStateTensor(rng_output_state));
  return AssignBufferToAtTensor(std::move(output_buf), self);
}

}  // namespace

at::Tensor& AtenRandom_(at::Tensor& self,
                        c10::optional<at::Generator> generator) {
  TT_KERNEL(OpName::kRandom_, param_keys, (self, generator), {
    TT_THROW_IF_ERROR(
        Random(self, generator, 0, std::nullopt, std::move(param_keys)));
    return self;
  });
}

at::Tensor& AtenRandom_From(at::Tensor& self, int64_t from,
                            c10::optional<int64_t> to,
                            c10::optional<at::Generator> generator) {
  TT_KERNEL(OpName::kRandom_From, param_keys, (self, from, to, generator), {
    TT_THROW_IF_ERROR(Random(self, generator, from, to, std::move(param_keys)));
    return self;
  });
}

at::Tensor& AtenRandom_To(at::Tensor& self, int64_t to,
                          c10::optional<at::Generator> generator) {
  TT_KERNEL(OpName::kRandom_To, param_keys, (self, to, generator), {
    TT_THROW_IF_ERROR(Random(self, generator, 0, to, std::move(param_keys)));
    return self;
  });
}

}  // namespace torch_tpu
