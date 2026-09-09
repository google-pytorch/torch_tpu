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

#include "csrc/experimental/precompiled_kernels/runtime/runtime.h"

#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "csrc/common/compilation.h"
#include "csrc/common/context_states.h"
#include "csrc/common/dimension_types.h"
#include "csrc/common/dtype.h"
#include "csrc/common/error_utils.h"
#include "csrc/common/shape.h"
#include "csrc/common/status_test_utils.h"
#include "csrc/eager/device_buffer.h"
#include "csrc/eager/eager_mode.h"
#include "csrc/eager/traversal.h"
#include "csrc/ops/op_names.h"
#include "gtest/gtest.h"
#include "mlir/IR/Types.h"

namespace torch_tpu {
namespace {
using precompiled_kernels::proto::Bundle;
using precompiled_kernels::proto::Tensor;

// Constructs a synthetic precompiled bundle for testing matching logic.
Bundle TestBundle() {
  Bundle bundle;
  bundle.set_format_version(1);
  for (auto op :
       {OpName::kRelu, OpName::kNegOut, OpName::kMulOut, OpName::kAddOut}) {
    for (auto dtype : {xla::F32, xla::BF16}) {
      for (int rank : {2, 3}) {
        auto* kernel = bundle.add_kernels();
        kernel->set_op(ToString(op));
        Tensor tensor;
        tensor.set_dtype(dtype);
        for (int i = 0; i < rank; ++i) {
          tensor.add_dims()->set_symbol("d" + std::to_string(i));
        }
        const int arity = op == OpName::kRelu || op == OpName::kNegOut ? 1 : 2;
        for (int i = 0; i < arity; ++i) *kernel->add_inputs() = tensor;
        *kernel->add_outputs() = tensor;
      }
    }
  }
  return bundle;
}

// Creates a deferred operation node for graph matching tests.
absl::StatusOr<std::vector<DeviceBufferRef>> Deferred(
    OpName op, std::vector<DeviceBufferRef> inputs,
    std::optional<Shape> shape = std::nullopt, Indices donated = {}) {
  ScopedPythonContextCapturer capture(op);
  Shape output = shape.value_or(inputs.front().shape());
  const auto previous = GetEagerMode();
  SetEagerMode(EagerMode::kInternalDeferAll);
  auto result = DeviceBufferList::CreateDeferred(
      op, {}, std::move(inputs), OpParamCacheKeys::Empty(), {output},
      OpSplitMode::kNone, std::move(donated));
  SetEagerMode(previous);
  return result;
}

TEST(Catalog, GroupsVariantsAndKeepsBundleEntries) {
  const auto bundle = TestBundle();
  const Catalog catalog(bundle);
  auto relu = catalog.Find("relu");
  ASSERT_EQ(relu.size(), 4);
  EXPECT_EQ(relu[0], &bundle.kernels(0));
  EXPECT_EQ(relu[3], &bundle.kernels(3));
  EXPECT_TRUE(catalog.Find("unknown").empty());
  EXPECT_EQ(catalog.Find("add.out").size(), 4);
}

TEST(Matching, DoesNotSubstituteAnotherOperation) {
  auto bundle = TestBundle();
  for (auto& kernel : *bundle.mutable_kernels()) {
    if (kernel.op() == "relu") kernel.set_op("unknown");
  }
  TT_ASSERT_OK_AND_ASSIGN(auto input, DeviceBufferList::CreatePlaceholder(
                                          {9, 259}, mlir::ElementType::F32));
  TT_ASSERT_OK_AND_ASSIGN(auto output, Deferred(OpName::kRelu, {input}));
  TT_ASSERT_OK_AND_ASSIGN(auto traversal, Traversal::Create(output));
  EXPECT_FALSE(MatchPrecompiledKernel(*traversal, Catalog(bundle)));
}

TEST(Matching, TriesLaterVariantsAfterInputSignatureMismatch) {
  auto bundle = TestBundle();
  auto valid = bundle.kernels(0);
  bundle.mutable_kernels(0)->mutable_inputs(0)->set_dtype(xla::S32);
  *bundle.add_kernels() = valid;
  TT_ASSERT_OK_AND_ASSIGN(auto input, DeviceBufferList::CreatePlaceholder(
                                          {9, 259}, mlir::ElementType::F32));
  TT_ASSERT_OK_AND_ASSIGN(auto output, Deferred(OpName::kRelu, {input}));
  TT_ASSERT_OK_AND_ASSIGN(auto traversal, Traversal::Create(output));
  EXPECT_TRUE(MatchPrecompiledKernel(*traversal, Catalog(bundle)));
}

TEST(Matching, CoversFamilySignatures) {
  const auto bundle = TestBundle();
  for (auto op :
       {OpName::kRelu, OpName::kNegOut, OpName::kMulOut, OpName::kAddOut}) {
    for (auto type : {mlir::ElementType::F32, mlir::ElementType::BF16}) {
      for (const Dimensions& dims :
           {Dimensions{9, 259}, Dimensions{3, 9, 259}}) {
        TT_ASSERT_OK_AND_ASSIGN(
            auto input, DeviceBufferList::CreatePlaceholder(dims, type));
        std::vector<DeviceBufferRef> inputs{input};
        if (op == OpName::kMulOut || op == OpName::kAddOut) {
          inputs.push_back(input);  // Repeated logical operand, deduped arg.
        }
        TT_ASSERT_OK_AND_ASSIGN(auto output, Deferred(op, inputs));
        TT_ASSERT_OK_AND_ASSIGN(auto traversal, Traversal::Create(output));
        EXPECT_TRUE(MatchPrecompiledKernel(*traversal, Catalog(bundle)));
      }
    }
  }
}

TEST(Matching, DeclinesFusionDonationAndScaledAdd) {
  const auto bundle = TestBundle();
  TT_ASSERT_OK_AND_ASSIGN(auto input, DeviceBufferList::CreatePlaceholder(
                                          {9, 259}, mlir::ElementType::F32));
  TT_ASSERT_OK_AND_ASSIGN(auto first, Deferred(OpName::kRelu, {input}));
  TT_ASSERT_OK_AND_ASSIGN(auto second, Deferred(OpName::kRelu, {first[0]}));
  TT_ASSERT_OK_AND_ASSIGN(auto fused, Traversal::Create(second));
  EXPECT_FALSE(MatchPrecompiledKernel(*fused, Catalog(bundle)));

  TT_ASSERT_OK_AND_ASSIGN(
      auto donated_neg, Deferred(OpName::kNegOut, {input}, std::nullopt, {0}));
  TT_ASSERT_OK_AND_ASSIGN(auto donated_traversal,
                          Traversal::Create(donated_neg));
  EXPECT_FALSE(MatchPrecompiledKernel(*donated_traversal, Catalog(bundle)));

  TT_ASSERT_OK_AND_ASSIGN(auto ternary_add,
                          Deferred(OpName::kAddOut, {input, input, input}));
  TT_ASSERT_OK_AND_ASSIGN(auto ternary_traversal,
                          Traversal::Create(ternary_add));
  EXPECT_FALSE(MatchPrecompiledKernel(*ternary_traversal, Catalog(bundle)));
}

TEST(Matching, DeclinesBroadcastEmptyAndUnsupportedRank) {
  const auto bundle = TestBundle();
  for (const Dimensions& dims : {Dimensions{0, 128}, Dimensions{128}}) {
    TT_ASSERT_OK_AND_ASSIGN(auto input, DeviceBufferList::CreatePlaceholder(
                                            dims, mlir::ElementType::F32));
    TT_ASSERT_OK_AND_ASSIGN(auto output, Deferred(OpName::kRelu, {input}));
    TT_ASSERT_OK_AND_ASSIGN(auto traversal, Traversal::Create(output));
    EXPECT_FALSE(MatchPrecompiledKernel(*traversal, Catalog(bundle)));
  }
  TT_ASSERT_OK_AND_ASSIGN(auto lhs, DeviceBufferList::CreatePlaceholder(
                                        {9, 259}, mlir::ElementType::F32));
  TT_ASSERT_OK_AND_ASSIGN(auto rhs, DeviceBufferList::CreatePlaceholder(
                                        {259}, mlir::ElementType::F32));
  TT_ASSERT_OK_AND_ASSIGN(auto output, Deferred(OpName::kMulOut, {lhs, rhs}));
  TT_ASSERT_OK_AND_ASSIGN(auto traversal, Traversal::Create(output));
  EXPECT_FALSE(MatchPrecompiledKernel(*traversal, Catalog(bundle)));
}

TEST(Policy, ReadySuccessAndErrorDeclineWithoutTouchingInputs) {
  const auto bundle = TestBundle();
  TT_ASSERT_OK_AND_ASSIGN(auto input, DeviceBufferList::CreatePlaceholder(
                                          {9, 259}, mlir::ElementType::F32));
  TT_ASSERT_OK_AND_ASSIGN(auto output, Deferred(OpName::kRelu, {input}));
  TT_ASSERT_OK_AND_ASSIGN(auto traversal, Traversal::Create(output));
  for (bool error : {false, true}) {
    auto run = MatchPrecompiledKernel(*traversal, Catalog(bundle));
    ASSERT_TRUE(run);
    LoadedExecutablePromise promise;
    CompiledKernel compiled{.fixed_shape_kernel = promise.get_future()};
    if (error) {
      promise.set_value(TT_ERROR(error::kInternal) << "compile failed");
    } else {
      promise.set_value(SharedLoadedExecutableWithMetadata{});
    }
    // Empty arguments deliberately prove no input access on ready XLA.
    TT_ASSERT_OK_AND_ASSIGN(auto result, std::move(run)(compiled, {}));
    EXPECT_FALSE(result.has_value());
  }
}

TEST(Policy, PendingCompilationPropagatesFailedInput) {
  const auto bundle = TestBundle();
  TT_ASSERT_OK_AND_ASSIGN(auto input, DeviceBufferList::CreatePlaceholder(
                                          {9, 259}, mlir::ElementType::F32));
  TT_ASSERT_OK_AND_ASSIGN(auto output, Deferred(OpName::kRelu, {input}));
  TT_ASSERT_OK_AND_ASSIGN(auto traversal, Traversal::Create(output));
  auto run = MatchPrecompiledKernel(*traversal, Catalog(bundle));
  ASSERT_TRUE(run);
  TT_ASSERT_OK_AND_ASSIGN(auto pending,
                          DeviceBufferList::CreatePending(input.shape(), 0, 0));
  pending.device_buffer_list()->SetAsError(TT_ERROR(error::kInternal)
                                           << "producer failed");
  LoadedExecutablePromise promise;
  CompiledKernel compiled{.fixed_shape_kernel = promise.get_future()};
  auto result = std::move(run)(compiled, {pending});
  EXPECT_EQ(result.status().message(), "producer failed");
}

TEST(Matching, StaticDimensionsConstrainOtherwiseMatchingKernels) {
  auto bundle = TestBundle();
  bundle.mutable_kernels(0)->mutable_inputs(0)->mutable_dims(0)->set_size(8);
  TT_ASSERT_OK_AND_ASSIGN(auto input, DeviceBufferList::CreatePlaceholder(
                                          {9, 259}, mlir::ElementType::F32));
  TT_ASSERT_OK_AND_ASSIGN(auto output, Deferred(OpName::kRelu, {input}));
  TT_ASSERT_OK_AND_ASSIGN(auto traversal, Traversal::Create(output));
  EXPECT_FALSE(MatchPrecompiledKernel(*traversal, Catalog(bundle)));
  bundle.mutable_kernels(0)->mutable_inputs(0)->mutable_dims(0)->set_size(9);
  EXPECT_TRUE(MatchPrecompiledKernel(*traversal, Catalog(bundle)));
}

TEST(Matching, SymbolicRelationsPermitDifferentInputAndOutputShapes) {
  auto bundle = TestBundle();
  // A synthetic shape-changing implementation exercises matching only.
  bundle.mutable_kernels(0)->mutable_outputs(0)->mutable_dims()->RemoveLast();
  TT_ASSERT_OK_AND_ASSIGN(auto input, DeviceBufferList::CreatePlaceholder(
                                          {9, 259}, mlir::ElementType::F32));
  TT_ASSERT_OK_AND_ASSIGN(
      auto output,
      Deferred(OpName::kRelu, {input}, Shape({9}, mlir::ElementType::F32)));
  TT_ASSERT_OK_AND_ASSIGN(auto traversal, Traversal::Create(output));
  EXPECT_TRUE(MatchPrecompiledKernel(*traversal, Catalog(bundle)));
}

TEST(Matching, UnknownBundleVersionDeclines) {
  auto bundle = TestBundle();
  bundle.set_format_version(999);
  TT_ASSERT_OK_AND_ASSIGN(auto input, DeviceBufferList::CreatePlaceholder(
                                          {9, 259}, mlir::ElementType::F32));
  TT_ASSERT_OK_AND_ASSIGN(auto output, Deferred(OpName::kRelu, {input}));
  TT_ASSERT_OK_AND_ASSIGN(auto traversal, Traversal::Create(output));
  EXPECT_FALSE(MatchPrecompiledKernel(*traversal, Catalog(bundle)));
}

TEST(Cache, ClearIsIdempotentAndResetsState) {
  ClearPrecompiledKernelCache();
  ClearPrecompiledKernelCache();
  auto bundle = TestBundle();
  TT_ASSERT_OK_AND_ASSIGN(auto input, DeviceBufferList::CreatePlaceholder(
                                          {9, 259}, mlir::ElementType::F32));
  TT_ASSERT_OK_AND_ASSIGN(auto output, Deferred(OpName::kRelu, {input}));
  TT_ASSERT_OK_AND_ASSIGN(auto traversal, Traversal::Create(output));
  EXPECT_TRUE(MatchPrecompiledKernel(*traversal, Catalog(bundle)));
  ClearPrecompiledKernelCache();
}

}  // namespace
}  // namespace torch_tpu
