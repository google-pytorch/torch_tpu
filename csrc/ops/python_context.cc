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

#include "csrc/ops/python_context.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/base/nullability.h"
#include "absl/flags/flag.h"
#include "absl/log/absl_check.h"
#include "absl/log/absl_log.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_replace.h"
#include "absl/types/span.h"
#include "csrc/common/context_manager.h"
#include "csrc/common/context_states.h"
#include "csrc/common/flags.h"
#include "csrc/eager/eager_mode.h"
#include "csrc/ops/op_names.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/Support/DebugStringHelper.h"
#include "mlir/Support/LLVM.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "torch/csrc/profiler/combined_traceback.h"
#include "torch/csrc/profiler/unwind/unwind.h"

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
  const auto mode =
      GetContextState<EnableTracebacksContextState>().value_or(std::nullopt);
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
  return GetEagerMode() == EagerMode::kInternalCompileFxGraph
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

  mlir::ArrayRef<std::string> op_call_chain(python_context.op_call_chain());
  // `tpu_torch_pallas.call_custom_kernel` pushes
  // `PythonApiName{"call_custom_kernel"}` at the pybind boundary (used by
  // `GetRootOpName` for error messages) before `TT_KERNEL` pushes the custom
  // kernel op name. Drop the outer pybind wrapper when formatting XProf scopes
  // so the kernel name sits directly under the caller's module and
  // `named_scope` hierarchy.
  if (op_call_chain.size() > 1 &&
      op_call_chain.front() == "call_custom_kernel") {
    op_call_chain = op_call_chain.drop_front();
  }
  auto curr_op = op_call_chain.back();

  // Name scope is everything but the current op.
  // For single op call chains, the name scope is the op itself.
  auto name_scope =
      op_call_chain.size() == 1 ? op_call_chain : op_call_chain.drop_back();
  const std::string op_str =
      absl::StrCat(absl::StrJoin(name_scope, "/"), "/", curr_op);
  if (python_context.name_scopes().empty()) {
    return op_str;
  }
  return absl::StrCat(absl::StrJoin(python_context.name_scopes(), "/"), "/",
                      op_str);
}

MaybeSharedNameScopes GetActiveNameScopes(
    absl::Span<const std::string> op_names = {}) {
  MaybeSharedNameScopes active =
      GetContextState<NameScopeContextState>().value_or(nullptr);
  if (active == nullptr || active->empty()) {
    return nullptr;
  }
  // Embedded constants (`kTorchTpuInternalConstant`) are module-level tensor
  // attributes created once per GraphModule attribute and shared across uses,
  // so they should not capture thread-local name scopes.
  if (!op_names.empty() &&
      op_names.back() == ToBaseName(OpName::kTorchTpuInternalConstant)) {
    return nullptr;
  }
  return active;
}

bool IsFrameNameLocation(mlir::Location location) {
  auto name_loc = mlir::dyn_cast<mlir::NameLoc>(location);
  return name_loc && mlir::isa<mlir::FileLineColLoc>(name_loc.getChildLoc());
}

bool IsTracebackLocation(mlir::Location location) {
  if (IsFrameNameLocation(location) ||
      mlir::isa<mlir::FileLineColLoc>(location)) {
    return true;
  }
  if (auto call_loc = mlir::dyn_cast<mlir::CallSiteLoc>(location)) {
    return IsTracebackLocation(call_loc.getCallee()) &&
           IsTracebackLocation(call_loc.getCaller());
  }
  return false;
}

// `MakeNameScopeOpStringFromCallChain` duplicates the leaf op name for
// single-op call chains (`<scopes>/<op>/<op>`) so XProf sees `<scopes>/<op>`
// as the scope prefix and `<op>` as the leaf op. When that op is an inlined
// `func.call`, strip the trailing duplicate segment so the caller scope prefix
// is `<scopes>/<op>` rather than `<scopes>/<op>/<op>`.
std::string_view StripTrailingDuplicateScope(std::string_view name) {
  const size_t last_slash = name.rfind('/');
  if (last_slash == std::string_view::npos) {
    return name;
  }
  const std::string_view last_seg = name.substr(last_slash + 1);
  const std::string_view prefix = name.substr(0, last_slash);
  const size_t prev_slash = prefix.rfind('/');
  const std::string_view prev_seg = prev_slash == std::string_view::npos
                                        ? prefix
                                        : prefix.substr(prev_slash + 1);
  if (!last_seg.empty() && prev_seg == last_seg) {
    return prefix;
  }
  return name;
}

// Unpacked components of an op's MLIR location (`FusedLoc` or `NameLoc`),
// separating an optional `:`-suffixed XLA op-type `NameLoc` prefix from the
// hierarchical op name and child source/traceback location. `op_type_prefix`
// and `op_name` are non-owning views into interned `mlir::StringAttr` storage
// owned by the `mlir::MLIRContext`.
struct OpLocationInfo {
  std::optional<std::string_view> op_type_prefix;
  std::string_view op_name;
  mlir::Location child_loc;
};

OpLocationInfo ExtractNameLocationInfo(mlir::NameLoc name_loc) {
  if (IsFrameNameLocation(name_loc)) {
    return {std::nullopt, "", name_loc};
  }
  const std::string_view name = name_loc.getName().strref();
  if (!absl::EndsWith(name, ":")) {
    return {std::nullopt, name, name_loc.getChildLoc()};
  }
  if (auto inner_name_loc =
          mlir::dyn_cast<mlir::NameLoc>(name_loc.getChildLoc());
      inner_name_loc && !IsFrameNameLocation(inner_name_loc)) {
    return {name, inner_name_loc.getName().strref(),
            inner_name_loc.getChildLoc()};
  }
  return {name, "", name_loc.getChildLoc()};
}

mlir::Location FirstNonUnknownSibling(mlir::FusedLoc fused_loc,
                                      mlir::Location skip_loc) {
  for (mlir::Location other_loc : fused_loc.getLocations()) {
    if (other_loc != skip_loc && !mlir::isa<mlir::UnknownLoc>(other_loc)) {
      return other_loc;
    }
  }
  return mlir::UnknownLoc::get(fused_loc.getContext());
}

OpLocationInfo ExtractOpLocationInfo(mlir::Location loc) {
  if (auto name_loc = mlir::dyn_cast<mlir::NameLoc>(loc)) {
    return ExtractNameLocationInfo(name_loc);
  }
  auto fused_loc = mlir::dyn_cast<mlir::FusedLoc>(loc);
  if (!fused_loc) {
    return {std::nullopt, "", loc};
  }
  for (mlir::Location sub_loc : fused_loc.getLocations()) {
    auto name_loc = mlir::dyn_cast<mlir::NameLoc>(sub_loc);
    if (!name_loc) {
      continue;
    }
    OpLocationInfo info = ExtractNameLocationInfo(name_loc);
    if (!info.op_name.empty()) {
      if (mlir::isa<mlir::UnknownLoc>(info.child_loc)) {
        info.child_loc = FirstNonUnknownSibling(fused_loc, sub_loc);
      }
      return info;
    }
  }
  return {std::nullopt, "", loc};
}

mlir::Location BuildOpLocation(mlir::MLIRContext* ctx,
                               std::optional<std::string_view> op_type_prefix,
                               std::string_view op_name,
                               mlir::Location child_loc) {
  const bool fuse_child = mlir::isa<mlir::FileLineColLoc>(child_loc);
  mlir::Location result =
      mlir::NameLoc::get(mlir::StringAttr::get(ctx, op_name),
                         fuse_child ? mlir::UnknownLoc::get(ctx) : child_loc);
  if (op_type_prefix.has_value()) {
    result =
        mlir::NameLoc::get(mlir::StringAttr::get(ctx, *op_type_prefix), result);
  }
  return fuse_child ? mlir::FusedLoc::get(ctx, {result, child_loc}) : result;
}

std::string MergeInlinedOpName(std::string_view caller_scope,
                               std::string_view callee_name) {
  std::string merged_name;
  if (callee_name.empty()) {
    // XProf's `ParseTfNameScopes` pops the final `/`-delimited segment as the
    // leaf op name, so duplicate the trailing caller segment to preserve the
    // full `caller_scope` in the scope hierarchy.
    const size_t last_slash = caller_scope.rfind('/');
    const std::string_view caller_leaf =
        last_slash == std::string_view::npos
            ? caller_scope
            : caller_scope.substr(last_slash + 1);
    merged_name = absl::StrCat(caller_scope, "/", caller_leaf);
  } else {
    const std::string prefix = absl::StrCat(caller_scope, "/");
    merged_name = absl::StartsWith(callee_name, prefix)
                      ? std::string(callee_name)
                      : absl::StrCat(prefix, callee_name);
  }
  // XProf formats `StatType::kTfOp` as `<op_name>:<op_type>` and splits on the
  // first `:`, so sanitize any `:` in `merged_name` to avoid truncating scopes.
  return absl::StrReplaceAll(merged_name, {{"::", "_"}, {":", "_"}});
}

mlir::Location CombineInlinedChildLocations(mlir::MLIRContext* ctx,
                                            const OpLocationInfo& callee_info,
                                            const OpLocationInfo& caller_info,
                                            std::string_view caller_scope) {
  if (mlir::isa<mlir::UnknownLoc>(callee_info.child_loc)) {
    return caller_info.child_loc;
  }
  if (mlir::isa<mlir::UnknownLoc>(caller_info.child_loc)) {
    return callee_info.child_loc;
  }
  mlir::Location callee_frame = callee_info.child_loc;
  // XLA's `StackFrameIndexBuilder` requires every frame inside a `CallSiteLoc`
  // to be a `NameLoc(func_name, FileLineColLoc)`. Wrap a bare `FileLineColLoc`
  // callee location (from `FusedLoc`) so XLA retains it alongside caller
  // frames.
  if (mlir::isa<mlir::FileLineColLoc>(callee_frame)) {
    const std::string_view frame_name =
        callee_info.op_name.empty() ? caller_scope : callee_info.op_name;
    callee_frame = mlir::NameLoc::get(mlir::StringAttr::get(ctx, frame_name),
                                      callee_frame);
  }
  return mlir::CallSiteLoc::get(callee_frame, caller_info.child_loc);
}

mlir::Location ResolveInlinedLocation(mlir::Location loc) {
  auto call_loc = mlir::dyn_cast<mlir::CallSiteLoc>(loc);
  if (!call_loc || IsTracebackLocation(loc)) {
    return loc;
  }

  const mlir::Location callee_loc =
      ResolveInlinedLocation(call_loc.getCallee());
  const mlir::Location caller_loc =
      ResolveInlinedLocation(call_loc.getCaller());

  const OpLocationInfo caller_info = ExtractOpLocationInfo(caller_loc);
  if (caller_info.op_name.empty()) {
    return callee_loc;
  }
  const OpLocationInfo callee_info = ExtractOpLocationInfo(callee_loc);
  const std::string_view caller_scope =
      StripTrailingDuplicateScope(caller_info.op_name);
  const std::string merged_name =
      MergeInlinedOpName(caller_scope, callee_info.op_name);

  mlir::MLIRContext* ctx = loc.getContext();
  const mlir::Location combined_child =
      CombineInlinedChildLocations(ctx, callee_info, caller_info, caller_scope);
  const std::optional<std::string_view> op_type_prefix =
      callee_info.op_type_prefix.has_value() ? callee_info.op_type_prefix
                                             : caller_info.op_type_prefix;
  return BuildOpLocation(ctx, op_type_prefix, merged_name, combined_child);
}

}  // namespace

PythonContext::PythonContext(std::vector<std::string> op_call_chain,
                             MaybeSharedTraceback traceback,
                             MaybeSharedNameScopes name_scopes)
    : op_call_chain_(std::move(op_call_chain)),
      traceback_(std::move(traceback)),
      name_scopes_(std::move(name_scopes)) {}

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

ScopedPythonContextCapturer::ScopedPythonContextCapturer(
    const std::string_view op_name) {
  op_names_.push_back(std::string(op_name));
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

ScopedPythonContextCapturer::ScopedPythonContextCapturer(
    const PythonApiName op_name)
    : ScopedPythonContextCapturer(op_name.name) {}

ScopedPythonContextCapturer::ScopedPythonContextCapturer(const OpName op_name)
    : ScopedPythonContextCapturer(ToBaseName(op_name)) {}

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
  return PythonContext(op_names_, traceback_, GetActiveNameScopes(op_names_));
}

std::optional<PythonContext> ScopedPythonContextCapturer::MaybeGetContext() {
  if (op_names_.empty()) return std::nullopt;
  return PythonContext(op_names_, traceback_, GetActiveNameScopes(op_names_));
}

void ScopedPythonContextCapturer::AppendNameScope(std::string scope) {
  std::vector<std::string> scopes;
  if (const MaybeSharedNameScopes active = GetActiveNameScopes();
      active != nullptr) {
    scopes = *active;
  }
  if (!absl::c_linear_search(scopes, scope)) {
    scopes.push_back(std::move(scope));
  }
  PushContextState<NameScopeContextState>(
      std::make_shared<const std::vector<std::string>>(std::move(scopes)));
}

void ScopedPythonContextCapturer::PushNameScopes(
    std::vector<std::string> scopes) {
  PushContextState<NameScopeContextState>(
      std::make_shared<const std::vector<std::string>>(std::move(scopes)));
}

void ScopedPythonContextCapturer::PopNameScopes() {
  PopContextState<NameScopeContextState>();
}

void ScopedPythonContextCapturer::OverrideCurrentOpName(
    const std::string_view op_name) {
  ABSL_CHECK(!op_names_.empty())  // CRASH_OK
      << "ScopedPythonContextCapturer::OverrideCurrentOpName() called when "
         "the context capturer stack is empty. This is a torch_tpu bug.";
  if (op_name.empty()) {
    return;
  }
  op_names_.back() =
      absl::StrReplaceAll(op_name, {{"::", "_"}, {":", "_"}, {"/", "_"}});
}

void PropagateInlinedCallerLocations(mlir::ModuleOp module) {
  module.walk([](mlir::Operation* op) {
    if (mlir::isa<mlir::CallSiteLoc>(op->getLoc())) {
      op->setLoc(ResolveInlinedLocation(op->getLoc()));
    }
  });
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
