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

#ifndef TORCH_TPU_OPS_python_context_H_
#define TORCH_TPU_OPS_python_context_H_

// This library provides utilities for capturing the python context of an op
// and using it to implement the op's behavior correctly. See go/catto-design
// for its design.
//
// The core concept here is the PythonContext type, which contains information
// on how an op is used in user code. An op's implementation interacts with
// PythonContext in two ways:
//
// 1. Capturing the python context. This is done by defining an object of type
//    `ScopedPythonContextCapturer` while the aten kernel is still running.
// 2. Providing the python context to an `MlirOp` builder. This is done by
//    defining an object of type `ScopedPythonContextProvider` before building
//    the `MlirOp`.

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "torch_tpu/common/context_states.h"
#include "torch_tpu/ops/op_names.h"

namespace torch_tpu {

// We define our own traceback format here instead of using
// torch::SymbolizedTracebacks because it can interact with python objects,
// which can cause GIL contention. These structs are pure C++ and safe to pass
// between threads.
// A python traceback.
struct PythonStackFrame {
  std::string file_name;
  std::string func_name;
  // 1-based line number.
  int64_t line_number = 0;
};

struct PythonTraceback {
  std::vector<PythonStackFrame> frames;
};

using SharedTraceback = absl_nonnull std::shared_ptr<const PythonTraceback>;
using MaybeSharedTraceback =
    absl_nullable std::shared_ptr<const PythonTraceback>;

// Returns the active traceback mode.
[[nodiscard]] TracebackMode GetTracebackMode();

// Options for specifying what stack traces to include in the MLIR Location.
// kPythonOnly: Only include Python stack frames.
// kPythonAndCpp: Include both Python and C++ stack frames, more costly and
//   resolution of C++ frames requires parsing dawrf files to figure out proper
//   symbolization.
enum class PythonStackLocationOptions {
  kPythonOnly,
  kPythonAndCpp,
};

// Used in deferred op builders to capture stack traces while building
// deferred op graphs.
// At this phase of the pipeline we don't have an MLIR Context since we aren't
// actively building a graph, so we split APIs for capturing the stack and
// and converting them to an MLIR Location.
// See `PythonStackLocationOptions` for specifying what stack traces to include.
[[nodiscard]] SharedTraceback CaptureCurrentPythonTrackback(
    PythonStackLocationOptions opts = PythonStackLocationOptions::kPythonOnly);

// The python context of an op.
class PythonContext {
 public:
  PythonContext() = default;
  PythonContext(std::vector<std::string> op_call_chain,
                MaybeSharedTraceback traceback);
  // This type is move-only.
  PythonContext(PythonContext&& other) = default;
  PythonContext& operator=(PythonContext&& other) = default;
  PythonContext(const PythonContext&) = delete;
  PythonContext& operator=(const PythonContext&) = delete;

  // Returns a copy of this PythonContext.
  [[nodiscard]] PythonContext Copy() const {
    return PythonContext(op_call_chain_, traceback_);
  }

  // Returns the names of the ops in the op call chain leading to the current
  // op.
  [[nodiscard]] const std::vector<std::string>& op_call_chain() const {
    return op_call_chain_;
  }

  // Returns the python traceback of the op, if one has been captured.
  [[nodiscard]] const MaybeSharedTraceback& traceback() const {
    return traceback_;
  }

 private:
  // Names of the ops in the op call chain leading to the current op (for
  // instance, if op foo is implemented by callingcalls op bar, in the context
  // of bar, this field will be {"foo", "bar"}).
  //
  // Neither the vector nor its elements may be empty.
  std::vector<std::string> op_call_chain_;
  // The python traceback of the op, if one has been captured.
  MaybeSharedTraceback traceback_;
};

// RAII class to capture the python context for an op. Conceptually, it
// maintains a per-thread singleton stack of PythonContext objects (the context
// capturer stack).
//
// Usage:
//
// 1. If an op is implemented via an MlirOp build function (the common case),
//    define a ScopedPythonContextCapturer object inside the DispatchOp
//    function. This is done once for all ops.
//
// 2. If an op is implemented by calling other ops, define a
//    ScopedPythonContextCapturer object in the op's kernel before it calls the
//    other ops.
//
// The ScopedPythonContextCapturer object must be defined *before* the
// kernel returns in order to capture the correct context. In particular,
// if an op is deferred, its MlirOp build function will be called
// *after* the kernel returns. Therefore we shouldn't define a
// ScopedPythonContextCapturer object inside the op build function.
class ScopedPythonContextCapturer {
 public:
  // Pushes the current python context onto the singleton context capturer
  // stack for the current thread. `op_name` is the name of the op currently
  // running.
  // If GetTracebackMode() == kEnabled and this is the root
  // ScopedPythonContextCapturer for the current thread, then a traceback will
  // be captured.
  explicit ScopedPythonContextCapturer(OpName op_name);

  // This class is neither copyable nor movable.
  ScopedPythonContextCapturer(const ScopedPythonContextCapturer&) = delete;
  ScopedPythonContextCapturer& operator=(const ScopedPythonContextCapturer&) =
      delete;
  ScopedPythonContextCapturer(ScopedPythonContextCapturer&&) = delete;
  ScopedPythonContextCapturer& operator=(ScopedPythonContextCapturer&&) =
      delete;

  // Pops the top python context from the singleton context capturer stack for
  // the current thread.
  ~ScopedPythonContextCapturer();

  // Returns the top python context from the singleton context capturer stack
  // for the current thread. Must be called when the stack is not empty, i.e.,
  // when at least one ScopedPythonContextCapturer is alive for the current
  // thread.
  [[nodiscard]] static PythonContext GetContext();

  // Like GetContext(), but returns std::nullopt if no
  // ScopedPythonContextCapturer is alive for the current thread.
  [[nodiscard]] static std::optional<PythonContext> MaybeGetContext();

  // For testing only: override the traceback of all currently alive contexts.
  static void SetTracebackForTesting(MaybeSharedTraceback traceback) {
    traceback_ = std::move(traceback);
  }

 private:
  // Returns the number of alive ScopedPythonContextCapturer instances for the
  // current thread. This is the size of the context capturer stack for the
  // current thread.
  [[nodiscard]] static int64_t GetNumAliveForThread() {
    return op_names_.size();
  }

  // While the state of this class is *conceptually* a per-thread stack, the
  // actual implementation takes advantage of one fact for efficiency:
  //
  // All contexts in the stack for the same thread have the same traceback.
  // Therefore we only need to maintain one copy of the traceback per thread.

  // The names of the ops in the context capturer stack for the current thread,
  // from bottom to top.
  static thread_local  // CPP_THREAD_LOCAL_OK=needed only in the dispatching
                       // thread.
      std::vector<std::string>
          op_names_;
  // nullptr when the context capturer stack for the current thread is
  // empty, or when traceback capture is disabled.
  // Otherwise, the traceback of all contexts in the stack.
  static thread_local  // CPP_THREAD_LOCAL_OK=needed only in the dispatching
                       // thread.
      MaybeSharedTraceback traceback_;
};

// RAII class to provide a PythonContext captured earlier to an MlirOp build
// function. Conceptually, it maintains a per-thread singleton stack of
// PythonContext objects (the context provider stack).
//
// Usage: define a ScopedPythonContextProvider object with the context for an
// op before building the MlirOp(s) for the op. Since we only call the MlirOp
// build function in a small number of places, this work is already done and
// op authors don't need to worry about it.
class ScopedPythonContextProvider {
 public:
  // Pushes the python context onto the singleton context
  // provider stack for the current thread. If `builder` is provided, sets the
  // location attribute of `builder` to the MLIR Location corresponding to the
  // python context.
  ScopedPythonContextProvider(
      PythonContext context,
      mlir::MlirBuilder* absl_nullable builder = nullptr);

  // This class is neither copyable nor movable.
  ScopedPythonContextProvider(const ScopedPythonContextProvider&) = delete;
  ScopedPythonContextProvider& operator=(const ScopedPythonContextProvider&) =
      delete;
  ScopedPythonContextProvider(ScopedPythonContextProvider&&) = delete;
  ScopedPythonContextProvider& operator=(ScopedPythonContextProvider&&) =
      delete;

  // Restores the location attribute of `builder` to its previous value, and
  // pops the top python context from the singleton context provider stack for
  // the current thread.
  ~ScopedPythonContextProvider();

  // Returns the top python context on the singleton context provider stack for
  // the current thread, or std::nullopt if the stack is empty.
  static const std::optional<PythonContext>& MaybeGetContext() {
    return top_context_;
  }

 private:
  // The top python context on the singleton context provider stack, one for
  // each thread, or std::nullopt if the stack is empty.
  static thread_local  // CPP_THREAD_LOCAL_OK=needed only in the dispatching
                       // thread.
      std::optional<PythonContext>
          top_context_;

  // Each alive ScopedPythonContextProvider instance corresponds to an element
  // in the context provider stack for the current thread. This field is the
  // previous top of the stack before the current instance is created.
  std::optional<PythonContext> previous_context_;

  // The RAII object that sets the location attribute of `builder` in its
  // constructor and restores the location in its destructor.
  std::optional<mlir::ScopedBuilderLocation> scoped_builder_loc_;
};

// Returns the op name without the suffix (e.g. "add.out" -> "add").
// If the op name doesn't have a suffix, it is returned unchanged.
[[nodiscard]] std::string_view RemoveOpSuffix(std::string_view op_name);

// Returns the root op name for the current op.
//
// The root op for an op X is defined as:
//
// 1. the first op in the op call chain that leads to X when X is called to
//    implement some other op, or
// 2. X itself if X is called by the user directly.
//
// For instance, if the user calls op foo, which calls op bar, which calls op
// baz, then the root op for foo, bar, and baz are all foo.
//
// This is usually used for reporting errors, where we want to report the
// root op name as it appears in the user's PyTorch code.
[[nodiscard]] std::string GetRootOpName(std::optional<OpName> op_name);

// Converts a traceback into an MLIR Location. This uses `torch::symbolize`
// which can be slow especially if CPP stack info is included.
//
// Note the format of locations must be fairly specific to allow xProf to parse
// it correctly - the root frame with named scopes, the callstack, the funcname,
// and the file-line-col.
//
// Example, `#locRoot` is returned and attached to the MlirOp:
//   #locFile = loc("/path/to/file.py":10:11 to :16)
//   #locFunc = loc("python_funcname"(#locFile))
//   #locCallstack = loc(callsite(#locFunc at #locCallsite)) <- can be chained
//   #locRoot = loc("NAME/SCOPES/op_name"(#locCallstack))
[[nodiscard]] mlir::Location MakeMlirLocation(
    mlir::MLIRContext& ctx, const PythonContext& python_context);

}  // namespace torch_tpu

#endif  // TORCH_TPU_OPS_python_context_H_
