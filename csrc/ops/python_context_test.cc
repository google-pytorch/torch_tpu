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

#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "c10/util/ThreadLocalDebugInfo.h"
#include "csrc/common/context_manager.h"
#include "csrc/common/context_states.h"
#include "csrc/ops/op_names.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Support/DebugStringHelper.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"

namespace torch_tpu {
namespace {

using ::testing::ElementsAre;
using ::testing::HasSubstr;
using ::testing::IsEmpty;
using ::testing::Not;

// RAII test helper that pushes or appends name scopes on construction via
// `ScopedPythonContextCapturer` and pops them on destruction.
class RaiiContextCapturer {
 public:
  explicit RaiiContextCapturer(std::string scope) {
    ScopedPythonContextCapturer::AppendNameScope(std::move(scope));
  }

  explicit RaiiContextCapturer(std::vector<std::string> scopes) {
    ScopedPythonContextCapturer::PushNameScopes(std::move(scopes));
  }

  RaiiContextCapturer(const RaiiContextCapturer&) = delete;
  RaiiContextCapturer& operator=(const RaiiContextCapturer&) = delete;

  ~RaiiContextCapturer() { ScopedPythonContextCapturer::PopNameScopes(); }
};

TEST(CaptureCurrentPythonTrackback, CapturesPythonStack) {
  mlir::MLIRContext ctx;
  PythonContext python_context(
      {},
      CaptureCurrentPythonTrackback(PythonStackLocationOptions::kPythonOnly));
  auto loc = MakeMlirLocation(ctx, python_context);
  EXPECT_THAT(mlir::debugString(loc),
              Not(HasSubstr("CaptureCurrentPythonTrackback")));
}

TEST(CaptureCurrentPythonTrackback, CapturesCppStackWhenRequested) {
  mlir::MLIRContext ctx;
  PythonContext python_context(
      {},
      CaptureCurrentPythonTrackback(PythonStackLocationOptions::kPythonAndCpp));
  auto loc = MakeMlirLocation(ctx, python_context);
  EXPECT_THAT(mlir::debugString(loc),
              HasSubstr("CapturesCppStackWhenRequested"));
}

TEST(ScopedPythonContextCapturer, NoObjectIsAlive) {
  const auto& context = ScopedPythonContextCapturer::MaybeGetContext();
  EXPECT_FALSE(context.has_value());
}

TEST(ScopedPythonContextCapturer, OneObjectIsAlive) {
  {
    ScopedPythonContextCapturer capturer(OpName::kAdd);
    EXPECT_THAT(ScopedPythonContextCapturer::GetContext().op_call_chain(),
                ElementsAre("add"));
    const auto& context = ScopedPythonContextCapturer::MaybeGetContext();
    ASSERT_TRUE(context.has_value());
    EXPECT_THAT(context->op_call_chain(), ElementsAre("add"));
  }
  const auto& context = ScopedPythonContextCapturer::MaybeGetContext();
  EXPECT_FALSE(context.has_value());
}

TEST(ScopedPythonContextCapturer, TwoObjectsAreAlive) {
  {
    ScopedPythonContextCapturer capturer(OpName::kAdd);
    {
      ScopedPythonContextCapturer capturer(OpName::kBmm);
      EXPECT_THAT(ScopedPythonContextCapturer::GetContext().op_call_chain(),
                  ElementsAre("add", "bmm"));
      const auto& context = ScopedPythonContextCapturer::MaybeGetContext();
      ASSERT_TRUE(context.has_value());
      EXPECT_THAT(context->op_call_chain(), ElementsAre("add", "bmm"));
    }
    EXPECT_THAT(ScopedPythonContextCapturer::GetContext().op_call_chain(),
                ElementsAre("add"));
    const auto& context = ScopedPythonContextCapturer::MaybeGetContext();
    ASSERT_TRUE(context.has_value());
    EXPECT_THAT(context->op_call_chain(), ElementsAre("add"));
  }
  const auto& context = ScopedPythonContextCapturer::MaybeGetContext();
  EXPECT_FALSE(context.has_value());
}

class ScopedPythonContextCapturerTrackebackTest : public testing::Test {
 protected:
  ~ScopedPythonContextCapturerTrackebackTest() override {
    // Restore the default state.
    PopContextState<EnableTracebacksContextState>();
  }
};

TEST_F(ScopedPythonContextCapturerTrackebackTest,
       CapturesTracebackWhenEnabled) {
  PushContextState<EnableTracebacksContextState>(TracebackMode::kEnabled);
  {
    ScopedPythonContextCapturer capturer(OpName::kAdd);
    auto context = ScopedPythonContextCapturer::GetContext();

    // The traceback is not nullptr, but we don't check its contents because
    // this created context has no python frames (and hence, is empty).
    ASSERT_NE(context.traceback(), nullptr);
  }
}

TEST_F(ScopedPythonContextCapturerTrackebackTest,
       DoesNotCaptureTracebackWhenDisabled) {
  PushContextState<EnableTracebacksContextState>(TracebackMode::kDisabled);
  {
    ScopedPythonContextCapturer capturer(OpName::kAdd);
    auto context = ScopedPythonContextCapturer::GetContext();
    EXPECT_TRUE(context.traceback() == nullptr);
  }
}

TEST(ScopedPythonContextProvider, NoObjectIsAlive) {
  const auto& context = ScopedPythonContextProvider::MaybeGetContext();
  EXPECT_FALSE(context.has_value());
}

TEST(ScopedPythonContextProvider, OneObjectIsAlive) {
  mlir::MLIRContext ctx;
  mlir::Location loc = mlir::FileLineColLoc::get(&ctx, "add", 1, 2);
  mlir::MlirBuilder builder(ctx, loc);
  {
    ScopedPythonContextProvider provider(
        PythonContext({"add"}, CaptureCurrentPythonTrackback()), &builder);
    EXPECT_NE(builder.getLoc(), loc);
    const auto& context = ScopedPythonContextProvider::MaybeGetContext();
    ASSERT_TRUE(context.has_value());
    EXPECT_THAT(context->op_call_chain(), ElementsAre("add"));
  }
  EXPECT_EQ(builder.getLoc(), loc);
  const auto& context = ScopedPythonContextProvider::MaybeGetContext();
  EXPECT_FALSE(context.has_value());
}

TEST(ScopedPythonContextProvider, TwoObjectsAreAlive) {
  mlir::MLIRContext ctx;
  mlir::Location loc = mlir::FileLineColLoc::get(&ctx, "add", 1, 2);
  mlir::MlirBuilder builder(ctx, loc);
  {
    ScopedPythonContextProvider provider(
        PythonContext({"add"}, CaptureCurrentPythonTrackback()), &builder);
    {
      ScopedPythonContextProvider provider(
          PythonContext({"sub"}, CaptureCurrentPythonTrackback()), &builder);
      EXPECT_NE(builder.getLoc(), loc);
      const auto& context = ScopedPythonContextProvider::MaybeGetContext();
      ASSERT_TRUE(context.has_value());
      EXPECT_THAT(context->op_call_chain(), ElementsAre("sub"));
    }
    EXPECT_NE(builder.getLoc(), loc);
    const auto& context = ScopedPythonContextProvider::MaybeGetContext();
    ASSERT_TRUE(context.has_value());
    EXPECT_THAT(context->op_call_chain(), ElementsAre("add"));
  }
  EXPECT_EQ(builder.getLoc(), loc);
  const auto& context = ScopedPythonContextProvider::MaybeGetContext();
  EXPECT_FALSE(context.has_value());
}

TEST(GetOriginalOpName, ReturnsCurrentOpNameWhenNoContextIsAlive) {
  EXPECT_EQ(GetRootOpName(OpName::kAdd), "add");
}

TEST(GetOriginalOpName, ReturnsFirstOpNameWhenCapturerIsAlive) {
  {
    ScopedPythonContextCapturer capturer(OpName::kAdd);
    EXPECT_EQ(GetRootOpName(OpName::kSub), "add");
    {
      ScopedPythonContextCapturer capturer(OpName::kCatOut);
      EXPECT_EQ(GetRootOpName(OpName::kSub), "add");
    }
    EXPECT_EQ(GetRootOpName(OpName::kSub), "add");
  }
  EXPECT_EQ(GetRootOpName(OpName::kSub), "sub");
}

TEST(GetOriginalOpName, ReturnsFirstOpNameWhenProviderIsAlive) {
  mlir::MLIRContext ctx;
  mlir::Location loc = mlir::FileLineColLoc::get(&ctx, "add", 1, 2);
  mlir::MlirBuilder builder(ctx, loc);
  {
    ScopedPythonContextProvider provider(
        PythonContext({"add"}, CaptureCurrentPythonTrackback()), &builder);
    EXPECT_EQ(GetRootOpName(OpName::kSub), "add");
    {
      ScopedPythonContextProvider provider(
          PythonContext({"baz", "add"}, CaptureCurrentPythonTrackback()),
          &builder);
      EXPECT_EQ(GetRootOpName(OpName::kSub), "baz");
    }
    EXPECT_EQ(GetRootOpName(OpName::kSub), "add");
  }
  EXPECT_EQ(GetRootOpName(OpName::kSub), "sub");
}

TEST(ScopedPythonContextCapturer, CapturesNameScopesAndFormatsMlirLocation) {
  mlir::MLIRContext ctx;
  RaiiContextCapturer block_scope("layers.0[SimpleTransformerBlock]");
  RaiiContextCapturer attn_scope("attn[SimpleAttention]");
  // Duplicate nested scopes collapse into one while keeping PopNameScopes
  // balanced.
  RaiiContextCapturer dup_scope("attn[SimpleAttention]");

  ScopedPythonContextCapturer capturer(OpName::kBmm);
  const PythonContext context = ScopedPythonContextCapturer::GetContext();
  EXPECT_THAT(
      context.name_scopes(),
      ElementsAre("layers.0[SimpleTransformerBlock]", "attn[SimpleAttention]"));
  EXPECT_EQ(GetRootOpName(OpName::kAdd), "bmm");
  const mlir::Location loc = MakeMlirLocation(ctx, context);
  EXPECT_THAT(mlir::debugString(loc),
              HasSubstr("layers.0[SimpleTransformerBlock]/"
                        "attn[SimpleAttention]/bmm/bmm"));
}

TEST(ScopedPythonContextCapturer, InternalConstantSuppressesActiveNameScopes) {
  RaiiContextCapturer block_scope("layers.0[SimpleTransformerBlock]");

  ScopedPythonContextCapturer capturer(OpName::kTorchTpuInternalConstant);
  EXPECT_THAT(ScopedPythonContextCapturer::GetContext().name_scopes(),
              IsEmpty());
}

TEST(ScopedPythonContextCapturer, PushEmptyNameScopesIsolatesFromOuterScopes) {
  RaiiContextCapturer block_scope("layers.0[SimpleTransformerBlock]");
  RaiiContextCapturer clear_scopes(std::vector<std::string>{});

  ScopedPythonContextCapturer capturer(OpName::kBmm);
  EXPECT_THAT(ScopedPythonContextCapturer::GetContext().name_scopes(),
              IsEmpty());
}

TEST(ScopedPythonContextCapturer, PushNameScopesReplacesActiveHierarchy) {
  RaiiContextCapturer outer_scope("outer_scope");
  RaiiContextCapturer mlp_scopes(std::vector<std::string>{
      "layers.0[SimpleTransformerBlock]", "mlp[SimpleMLP]"});

  ScopedPythonContextCapturer capturer(OpName::kBmm);
  EXPECT_THAT(
      ScopedPythonContextCapturer::GetContext().name_scopes(),
      ElementsAre("layers.0[SimpleTransformerBlock]", "mlp[SimpleMLP]"));
}

TEST(ScopedPythonContextCapturer, NameScopesRestoreViaDebugInfoGuard) {
  std::shared_ptr<c10::ThreadLocalDebugInfo> saved_debug_info;
  {
    RaiiContextCapturer backward_scope("backward_phase");
    saved_debug_info = c10::ThreadLocalDebugInfo::current();
  }
  // Scope has been popped.
  PythonContext unguarded_context;
  {
    ScopedPythonContextCapturer capturer(OpName::kAdd);
    unguarded_context = ScopedPythonContextCapturer::GetContext();
  }
  // Restoring saved_debug_info via DebugInfoGuard (as PyTorch's C++ autograd
  // engine does when dispatching backward tasks) restores the backward
  // caller's name scopes.
  PythonContext guarded_context;
  {
    c10::DebugInfoGuard guard(saved_debug_info);
    ScopedPythonContextCapturer capturer(OpName::kMul);
    guarded_context = ScopedPythonContextCapturer::GetContext();
  }

  EXPECT_THAT(unguarded_context.name_scopes(), IsEmpty());
  EXPECT_THAT(guarded_context.name_scopes(), ElementsAre("backward_phase"));
}

TEST(ScopedPythonContextCapturer,
     OverrideCurrentOpNameSanitizesDelimitersAndPreservesRootOpName) {
  mlir::MLIRContext ctx;
  RaiiContextCapturer block_scope(std::vector<std::string>{"layers.0[Block]"});

  ScopedPythonContextCapturer api_capturer(PythonApiName{"call_custom_kernel"});
  ScopedPythonContextCapturer kernel_capturer(OpName::kCustomKernel);
  ScopedPythonContextCapturer::OverrideCurrentOpName("pallas::ns/my_kernel");
  // Empty override is a no-op and preserves the existing op name.
  ScopedPythonContextCapturer::OverrideCurrentOpName("");

  const PythonContext context = ScopedPythonContextCapturer::GetContext();
  const mlir::Location loc = MakeMlirLocation(ctx, context);

  EXPECT_THAT(context.op_call_chain(),
              ElementsAre("call_custom_kernel", "pallas_ns_my_kernel"));
  // Root op name for error messages remains `call_custom_kernel`, while XProf
  // location formatting drops the pybind wrapper and uses the kernel name.
  EXPECT_EQ(GetRootOpName(std::nullopt), "call_custom_kernel");
  EXPECT_EQ(mlir::debugString(loc),
            "loc(\"layers.0[Block]/pallas_ns_my_kernel/pallas_ns_my_kernel\")");
}

TEST(PropagateInlinedCallerLocationsTest,
     PrependsCallerScopeAndPreservesStackFrames) {
  mlir::MLIRContext ctx;
  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceString<mlir::ModuleOp>(
          R"mlir(
            module {} loc(callsite(
              "jit(add_vectors)/pallas_call"("jax_kernel"("kernel.py":12:0))
              at "layers.0[Block]/pallas_add_vectors/pallas_add_vectors"(
                "forward"("model.py":42:0))))
          )mlir",
          &ctx);
  ASSERT_TRUE(module);

  PropagateInlinedCallerLocations(module.get());

  EXPECT_EQ(
      mlir::debugString(module->getLoc()),
      R"(loc("layers.0[Block]/pallas_add_vectors/jit(add_vectors)/pallas_call"(callsite("jax_kernel"("kernel.py":12:0) at "forward"("model.py":42:0)))))");
}

TEST(PropagateInlinedCallerLocationsTest,
     FallsBackToCallerNameWhenCalleeIsUnknownLoc) {
  mlir::MLIRContext ctx;
  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceString<mlir::ModuleOp>(
          R"mlir(
            module {} loc(callsite(
              unknown
              at "layers.1[Block]/pallas_kernel_add/pallas_kernel_add"))
          )mlir",
          &ctx);
  ASSERT_TRUE(module);

  PropagateInlinedCallerLocations(module.get());

  EXPECT_EQ(mlir::debugString(module->getLoc()),
            R"(loc("layers.1[Block]/pallas_kernel_add/pallas_kernel_add"))");
}

TEST(PropagateInlinedCallerLocationsTest,
     UnwrapsFusedLocAndPreservesOpTypePrefixAndSanitizesColons) {
  mlir::MLIRContext ctx;
  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceString<mlir::ModuleOp>(
          R"mlir(
            module {} loc(callsite(
              fused["custom_call:"("jit(ns::kernel)/pallas_call"), "kernel.py":10:0]
              at "layers.0[Block]/pallas_phase/pallas_phase"))
          )mlir",
          &ctx);
  ASSERT_TRUE(module);

  PropagateInlinedCallerLocations(module.get());

  EXPECT_EQ(
      mlir::debugString(module->getLoc()),
      R"(loc(fused["custom_call:"("layers.0[Block]/pallas_phase/jit(ns_kernel)/pallas_call"), "kernel.py":10:0]))");
}

TEST(PropagateInlinedCallerLocationsTest,
     WrapsBareFileLineColCalleeInFrameNameLocWhenCallerHasTraceback) {
  mlir::MLIRContext ctx;
  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceString<mlir::ModuleOp>(
          R"mlir(
            module {} loc(callsite(
              fused["jit(add_vectors)/pallas_call", "kernel.py":10:0]
              at "layers.0[Block]/pallas_add_vectors/pallas_add_vectors"(
                "forward"("model.py":42:0))))
          )mlir",
          &ctx);
  ASSERT_TRUE(module);

  PropagateInlinedCallerLocations(module.get());

  EXPECT_EQ(
      mlir::debugString(module->getLoc()),
      R"(loc("layers.0[Block]/pallas_add_vectors/jit(add_vectors)/pallas_call"(callsite("jit(add_vectors)/pallas_call"("kernel.py":10:0) at "forward"("model.py":42:0)))))");
}

TEST(PropagateInlinedCallerLocationsTest,
     ResolvesMultiLevelNestedCallSiteLocations) {
  mlir::MLIRContext ctx;
  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceString<mlir::ModuleOp>(
          R"mlir(
            module {} loc(callsite(
              callsite("add.1" at "helper_fn/helper_fn")
              at "layers.0[Block]/custom_op/custom_op"))
          )mlir",
          &ctx);
  ASSERT_TRUE(module);

  PropagateInlinedCallerLocations(module.get());

  EXPECT_EQ(mlir::debugString(module->getLoc()),
            R"(loc("layers.0[Block]/custom_op/helper_fn/add.1"))");
}

TEST(PropagateInlinedCallerLocationsTest,
     LeavesPureTracebackCallSiteLocUnchanged) {
  mlir::MLIRContext ctx;
  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceString<mlir::ModuleOp>(
          R"mlir(
            module {} loc(callsite(
              "callee_fn"("a.py":1:0) at "caller_fn"("b.py":2:0)))
          )mlir",
          &ctx);
  ASSERT_TRUE(module);

  PropagateInlinedCallerLocations(module.get());

  EXPECT_EQ(
      mlir::debugString(module->getLoc()),
      R"(loc(callsite("callee_fn"("a.py":1:0) at "caller_fn"("b.py":2:0))))");
}

}  // namespace
}  // namespace torch_tpu
