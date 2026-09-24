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

#include "csrc/mlir/instrumentation.h"

#include <atomic>
#include <cstdlib>
#include <memory>
#include <optional>
#include <string>
#include <system_error>  // NOLINT
#include <utility>

#include "absl/log/absl_log.h"
#include "absl/strings/str_cat.h"
#include "csrc/common/env_vars.h"
#include "csrc/common/utils.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/Timing.h"

namespace torch_tpu {
namespace {

static std::atomic<int> g_module_counter{0};

int NextModuleId() { return g_module_counter.fetch_add(1); }

}  // namespace

void InstrumentationState::Apply(mlir::PassManager& pm) {
  if (ir_os) {
    pm.enableIRPrinting(
        /*shouldPrintBeforePass=*/[](mlir::Pass*,
                                     mlir::Operation*) { return true; },
        /*shouldPrintAfterPass=*/
        [](mlir::Pass*, mlir::Operation*) { return true; },
        /*printModuleScope=*/true,
        /*printAfterOnlyOnChange=*/true,
        /*printAfterOnlyOnFailure=*/false, *ir_os);
  } else if (dump_to_print) {
    pm.enableIRPrinting(
        /*shouldPrintBeforePass=*/[](mlir::Pass*,
                                     mlir::Operation*) { return true; },
        /*shouldPrintAfterPass=*/
        [](mlir::Pass*, mlir::Operation*) { return true; },
        /*printModuleScope=*/true,
        /*printAfterOnlyOnChange=*/true,
        /*printAfterOnlyOnFailure=*/false);
  }
  if (timing_os) {
    auto tm = std::make_unique<mlir::DefaultTimingManager>();
    tm->setEnabled(true);
    tm->setOutput(mlir::createOutputStrategy(
        mlir::DefaultTimingManager::OutputFormat::Text, *timing_os));
    pm.enableTiming(std::move(tm));
  } else if (dump_to_print) {
    pm.enableTiming();
  }
}

std::optional<InstrumentationState> ConfigureInstrumentation(
    mlir::MLIRContext* context, mlir::ModuleOp module) {
  const auto& env_dump_to =
      GetEnvOnce<kTorchTpuInternalDumpMlirPassesToEnvVar>();
  if (!env_dump_to.has_value() || env_dump_to->empty()) {
    return std::nullopt;
  }
  const std::string& dump_to = *env_dump_to;

  if (dump_to == "print") {
    InstrumentationState state;
    state.dump_to_print = true;
    return state;
  }

  std::string dump_dir;
#if TT_IS_INTERNAL_TORCH_TPU
  if (dump_to == "sponge") {
    const char* sponge_dir =
        std::getenv("TEST_UNDECLARED_OUTPUTS_DIR");  // GETENV_OK=Implicit
    if (sponge_dir != nullptr && sponge_dir[0] != '\0') {
      dump_dir = sponge_dir;
    } else {
      return std::nullopt;
    }
  } else {
    dump_dir = dump_to;
  }
#else
  dump_dir = dump_to;
#endif

  if (dump_dir.empty()) {
    return std::nullopt;
  }

  int module_id = NextModuleId();
  InstrumentationState state;

  std::string module_name = "unknown";
  if (module.getName().has_value()) {
    module_name = module.getName()->str();
  }

  std::string ir_file_name =
      absl::StrCat(module_name, "_", module_id, "_pass_dumps.mlir");
  std::string ir_file_path = absl::StrCat(dump_dir, "/", ir_file_name);
  std::error_code ir_ec;
  auto ir_stream = std::make_unique<llvm::raw_fd_ostream>(ir_file_path, ir_ec);
  if (!ir_ec) {
    state.ir_os = std::move(ir_stream);
  } else {
    ABSL_LOG(WARNING) << "Failed to open IR dump file " << ir_file_path << ": "
                      << ir_ec.message();
  }

  std::string timing_file_name =
      absl::StrCat(module_name, "_", module_id, "_timing.txt");
  std::string timing_file_path = absl::StrCat(dump_dir, "/", timing_file_name);
  std::error_code timing_ec;
  auto timing_stream =
      std::make_unique<llvm::raw_fd_ostream>(timing_file_path, timing_ec);
  if (!timing_ec) {
    state.timing_os = std::move(timing_stream);
  } else {
    ABSL_LOG(WARNING) << "Failed to open timing dump file " << timing_file_path
                      << ": " << timing_ec.message();
  }

  return state;
}

}  // namespace torch_tpu
