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

#include "torch_tpu/csrc/ops/precision_context.h"

#include <atomic>
#include <optional>
#include <string_view>

#include "absl/log/absl_check.h"
#include "absl/strings/str_cat.h"
#include "c10/util/Exception.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "torch_tpu/csrc/common/cache_key.h"
#include "torch_tpu/csrc/common/context_manager.h"
#include "torch_tpu/csrc/common/context_states.h"
#include "torch_tpu/csrc/common/to_string.h"

namespace torch_tpu {

static std::atomic<mlir::stablehlo::Precision> g_global_precision{
    mlir::stablehlo::Precision::DEFAULT};  // EXPLICIT_PRECISION_OK=global state

void PySetGlobalPrecision(mlir::stablehlo::Precision precision) {
  g_global_precision.store(precision);
}

mlir::stablehlo::Precision PyGetGlobalPrecision() {
  return g_global_precision.load();
}

void PyPushPrecision(mlir::stablehlo::Precision precision) {
  PushContextState<PrecisionContextState>(precision);
}

void PyPopPrecision() { PopContextState<PrecisionContextState>(); }

// Returns the current python thread's precision. It's unsafe in the sense that
// it doesn't ensure that the precision is included in an op's cache key.
[[nodiscard]] mlir::stablehlo::Precision PyUnsafeGetPrecision() {
  const auto state = GetContextState<PrecisionContextState>();
  if (state.has_value()) {
    return state.value();
  }

  const auto current_global_precision = PyGetGlobalPrecision();

  if (current_global_precision !=
      mlir::stablehlo::Precision::DEFAULT) {  // EXPLICIT_PRECISION_OK=root
                                              // usage
    TORCH_WARN_ONCE(
        absl::StrCat("TPU is using ", ToString(current_global_precision),
                     " precision as requested, but performance may be "
                     "significantly lower. Consider using ",
                     "torch.set_float32_matmul_precision('medium') for better ",
                     "performance."));
  }

  return current_global_precision;
}

[[nodiscard]] mlir::stablehlo::Precision GetAndAddPrecisionTo(
    OpParamCacheKeys& param_keys, const std::string_view param_name) {
  const auto precision = PyUnsafeGetPrecision();
  ABSL_CHECK_OK(param_keys.SetParam(param_name, precision));  // CRASH_OK
  return precision;
}

}  // namespace torch_tpu
