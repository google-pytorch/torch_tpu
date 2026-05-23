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

#include "torch_tpu/ops/python_context.h"

#include <array>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/flags/flag.h"
#include "absl/log/absl_check.h"
#include "absl/log/absl_log.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Support/DebugStringHelper.h"
#include "mlir/Support/LLVM.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "torch/csrc/profiler/combined_traceback.h"
#include "torch/csrc/profiler/unwind/unwind.h"
#include "torch_tpu/common/context_manager.h"
#include "torch_tpu/common/context_states.h"
#include "torch_tpu/common/flags.h"
#include "torch_tpu/eager/eager_mode.h"
#include "torch_tpu/ops/op_names.h"

ABSL_FLAG(std::optional<bool>, torch_tpu_internal_mlir_tracebacks,
          std::nullopt,  //
          "If set, MLIR location tracebacks will be enabled or disabled by "
          "default depending on the Boolean value. Otherwise, the default is "
          "disabling tracebacks in the eager mode and enabling them in the "
          "compiled mode. This can be overridden by the enable_tracebacks() "
          "context manager.");

namespace torch_tpu {

// Returns the traceback mode in the current context.
TracebackMode GetTracebackMode() {
  // If there's an override, use it.
  const auto mode = GetContextState<EnableTracebacksContextState>(std::nullopt);
  if (mode.has_value()) return mode.value();

  // Otherwise, use the flag value if set. Read the flag once for consistent
  // behavior in case the flag is changed while the program is running.
  static const auto maybe_flag =
      GetFlagOnce<std::optional<bool>,
                  &FLAGS_torch_tpu_internal_mlir_tracebacks>();
  if (maybe_flag.has_value()) {
    return (*maybe_flag ? TracebackMode::kEnabled : TracebackMode::kDisabled);
  }

  // The flag is unset, check if we are in compiled mode.
  return GetEagerMode() == EagerMode::kInternalDeferAll
             ? TracebackMode::kEnabled
             : TracebackMode::kDisabled;
}

namespace {

std::string_view GetCanonicalFilename(std::string_view filename) {
  // We use rfind since the generated build path may include multiple
  // references to the filter prefix.
  constexpr std::string_view kFilterPrefix = "/bazel-out/";
  auto it = filename.rfind(kFilterPrefix);
  if (it != std::string_view::npos) {
    return filename.substr(it + kFilterPrefix.size());
  }
  return filename;
}

bool IsUserFrame(const PythonStackFrame& frame) {
  if (frame.file_name == "??" && frame.func_name == "??") return false;

  // TODO: This filters functions starting with _ or <
  // This probably isn't always something we want to omit.
  std::string_view funcname = frame.func_name;
  if (absl::StartsWith(funcname, "<")) return false;

  // TODO: This needs to be more scalable than hard-coded.
  using namespace std::string_view_literals;  // NOLINT needed for sv literals
  constexpr std::array<std::string_view, 7> hide_frames{
      "/absl/testing/"sv,
      "/torch/_ops.py"sv,
      "/torch/fx/interpreter.py"sv,
      "/torch/testing/_internal/"sv,
      "/torch_tpu/export/export.py"sv,
      "/unittest/main.py"sv,
      "/<"sv};

  for (const auto& hide_frame : hide_frames) {
    if (absl::StrContains(frame.file_name, hide_frame)) return false;
  }
  return true;
}

// Creates an MLIR NamedLocation class for the given string. Useful for
// capturing the name of a function and its file location in `loc`. If the name
// is "??", meaning a symbol couldn't be resolved when C++ stack info is
// enabled, don't nest the location as this isn't meaningful debuginfo.
mlir::Location NamedLocation(std::string_view name, mlir::Location loc) {
  if (name == "??") return loc;
  return mlir::NameLoc::get(mlir::StringAttr::get(loc.getContext(), name), loc);
}

// Creates an MLIR FileLineColLoc class for the given string. Useful for
// capturing the file location of a function in `loc`. If the name is "??",
// meaning a symbol couldn't be resolved when C++ stack info is enabled, return
// an unknown location.
mlir::Location FileLineColLoc(mlir::MLIRContext& ctx, std::string_view filename,
                              unsigned lineno, unsigned column) {
  if (filename == "??") return mlir::UnknownLoc::get(&ctx);
  return mlir::FileLineColLoc::get(&ctx, GetCanonicalFilename(filename), lineno,
                                   column);
}

// Utility function for creating Function & FileLineCol debuginfo in the proper
// format. Emits a `NamedLocation(func_name, FileLineColLoc(filename, lineno))`.
// Converts `??` filenames to unknown locations and skips `??` function names.
mlir::Location MakeFuncFileLineColLoc(mlir::MLIRContext& ctx,
                                      const PythonStackFrame& frame) {
  auto file_loc = FileLineColLoc(ctx, frame.file_name,
                                 static_cast<unsigned>(frame.line_number), 0);
  return NamedLocation(frame.func_name, file_loc);
}

// Returns a location of the callstack->func->filename as described in the
// header file of `MakeMlirLocation`.
//
// If no python info is available, returns an unknown location.
mlir::Location MakeLocFromPythonTraceback(mlir::MLIRContext& ctx,
                                          MaybeSharedTraceback tb) {
  // If no python info is available, return an unknown location.
  if (!tb) return mlir::UnknownLoc::get(&ctx);

  // TODO(gleasonk): Possible perf optimization, only take the top 2 frames,
  // that's all XProf uses today.
  mlir::SmallVector<mlir::Location> frame_locs;
  for (const auto& frame : tb->frames) {
    if (!IsUserFrame(frame)) continue;

    auto named_loc = MakeFuncFileLineColLoc(ctx, frame);
    ABSL_VLOG(3) << "  Adding location: " << mlir::debugString(named_loc);
    frame_locs.push_back(named_loc);
  }

  if (frame_locs.empty()) return mlir::UnknownLoc::get(&ctx);
  if (frame_locs.size() == 1) return frame_locs.front();

  // Return nested CallSiteLoc of entire stack trace.
  mlir::ArrayRef<mlir::Location> frame_locs_ref(frame_locs);
  return mlir::CallSiteLoc::get(frame_locs_ref[0], frame_locs_ref.drop_front());
}

// Create a name scope / op loc for the given call chain.
// In xProf this frame is `NAME/STACK/opname`.
//
// In our case we will use:
//   CALL/CHAIN/opname
// Where opname is the back of the call chain.
// If chain is empty, return an unknown location.
// If chain is size 1, return opname/opname for consistency in xprof.
std::string MakeNameScopeOpStringFromCallChain(
    const PythonContext& python_context) {
  // Unreachable?
  if (python_context.op_call_chain().empty()) return "";

  // Curr op is the last op in the call chain.
  mlir::ArrayRef<std::string> op_call_chain(python_context.op_call_chain());
  auto curr_op = op_call_chain.back();

  // Name scope is everything but the current op.
  // For single op call chains, the name scope is the op itself.
  auto name_scope =
      op_call_chain.size() == 1 ? op_call_chain : op_call_chain.drop_back();
  return absl::StrCat(absl::StrJoin(name_scope, "/"), "/", curr_op);
}

}  // namespace

PythonContext::PythonContext(std::vector<std::string> op_call_chain,
                             MaybeSharedTraceback traceback)
    : op_call_chain_(std::move(op_call_chain)),
      traceback_(std::move(traceback)) {}

SharedTraceback CaptureCurrentPythonTrackback(PythonStackLocationOptions opts) {
  const bool include_cpp = opts == PythonStackLocationOptions::kPythonAndCpp;
  auto captured = torch::CapturedTraceback::gather(/*python=*/true,
                                                   /*script=*/false,
                                                   /*cpp=*/include_cpp);

  // We want to symbolize at dispatch time, because symbolization requires the
  // GIL, and the PyTorch dispatcher is already holding the GIL. If we wait
  // until later, this means we can't do it on a background thread.
  auto symbolized = torch::symbolize({captured.get()});

  // Convert to local Traceback struct.
  auto traceback = std::make_shared<PythonTraceback>();
  // We only care about the first traceback (since we passed a single captured
  // traceback).
  if (!symbolized.tracebacks.empty()) {
    const auto& indices = symbolized.tracebacks[0];
    traceback->frames.reserve(indices.size());
    for (auto idx : indices) {
      if (idx < symbolized.all_frames.size()) {
        const auto& frame = symbolized.all_frames[idx];
        traceback->frames.push_back({frame.filename, frame.funcname,
                                     static_cast<int64_t>(frame.lineno)});
      }
    }
  }
  return traceback;
}

// See header file for format of locations
mlir::Location MakeMlirLocation(mlir::MLIRContext& ctx,
                                const PythonContext& python_context) {
  // Build the callsite->func->file location using the traceback if available.
  auto python_stack_loc =
      MakeLocFromPythonTraceback(ctx, python_context.traceback());

  // Build the root frame using the op call chain.
  auto name_scope_op_str = MakeNameScopeOpStringFromCallChain(python_context);
  auto root_loc = NamedLocation(name_scope_op_str, python_stack_loc);
  ABSL_VLOG(3) << "  Final location: " << mlir::debugString(root_loc);

  return root_loc;
}

ScopedPythonContextCapturer::ScopedPythonContextCapturer(const OpName op_name) {
  op_names_.push_back(std::string(ToBaseName(op_name)));
  if (GetNumAliveForThread() == 1) {
    ABSL_CHECK(traceback_ == nullptr)  // CRASH_OK
        << "While creating the first alive ScopedPythonContextCapturer, we "
           "found an element in the context capturer stack. This is a "
           "torch_tpu bug.";
    if (GetTracebackMode() == TracebackMode::kEnabled) {
      traceback_ = CaptureCurrentPythonTrackback();
    }
  }
  ABSL_VLOG(1) << "[ScopedPythonContextCapturer] Created for op: " << op_name;
  ABSL_VLOG(3) << "[ScopedPythonContextCapturer] Current call chain: "
               << absl::StrJoin(op_names_, ", ");
}

ScopedPythonContextCapturer::~ScopedPythonContextCapturer() {
  ABSL_CHECK(!op_names_.empty())  // CRASH_OK
      << "ScopedPythonContextCapturer::~ScopedPythonContextCapturer() called "
         "when the context capturer stack is empty. This is a torch_tpu bug.";
  op_names_.pop_back();
  if (op_names_.empty()) {
    traceback_ = nullptr;
  }
}

PythonContext ScopedPythonContextCapturer::GetContext() {
  ABSL_CHECK(!op_names_.empty())  // CRASH_OK
      << "ScopedPythonContextCapturer::GetContext() called when no "
         "ScopedPythonContextCapturer is alive. This is a torch_tpu bug. Make "
         "sure a ScopedPythonContextCapturer object is defined "
         "somewhere up the stack. See go/catto-design for details.";
  return PythonContext(op_names_, traceback_);
}

std::optional<PythonContext> ScopedPythonContextCapturer::MaybeGetContext() {
  if (op_names_.empty()) return std::nullopt;
  return PythonContext(op_names_, traceback_);
}

thread_local  // CPP_THREAD_LOCAL_OK=needed only in the dispatching thread.
    std::vector<std::string>
        ScopedPythonContextCapturer::op_names_;  // NOLINT
thread_local  // CPP_THREAD_LOCAL_OK=needed only in the dispatching thread.
    MaybeSharedTraceback ScopedPythonContextCapturer::traceback_;  // NOLINT

ScopedPythonContextProvider::ScopedPythonContextProvider(
    PythonContext context, mlir::MlirBuilder* absl_nullable const builder) {
  if (builder != nullptr) {
    scoped_builder_loc_.emplace(
        *builder, MakeMlirLocation(builder->getContext(), context));
  }
  previous_context_ = std::move(top_context_);
  top_context_ = std::move(context);
}

ScopedPythonContextProvider::~ScopedPythonContextProvider() {
  top_context_ = std::move(previous_context_);
}

thread_local  // CPP_THREAD_LOCAL_OK=needed only in the dispatching thread.
    std::optional<PythonContext>
        ScopedPythonContextProvider::top_context_;  // NOLINT

std::string GetRootOpName(const std::optional<OpName> current_op_name) {
  if (const auto& context = ScopedPythonContextProvider::MaybeGetContext();
      context.has_value()) {
    return context->op_call_chain().front();
  }
  if (const auto& context = ScopedPythonContextCapturer::MaybeGetContext();
      context.has_value()) {
    return context->op_call_chain().front();
  }

  // No ScopedPythonContextProvider or ScopedPythonContextCapturer is alive.
  // This can only happen if:
  // 1. The user directly calls the current op, and
  // 2. The kernel hasn't defined a ScopedPythonContextCapturer yet or the
  //    ScopedPythonContextCapturer has already been destroyed.
  // In this case, the current op is the root op.
  return current_op_name.has_value() ? std::string(ToString(*current_op_name))
                                     : "";
}

}  // namespace torch_tpu
