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

#include "csrc/common/unbounded_dynamic_kernel_registry.h"

#include <cstdint>
#include <memory>
#include <string_view>

#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "csrc/common/cache_key.h"
#include "csrc/common/compilation.h"
#include "csrc/common/compilation_spec.h"
#include "csrc/common/compile_options_key.h"
#include "csrc/common/dimension_types.h"
#include "csrc/common/error_utils.h"
#include "csrc/common/fingerprint_utils.h"
#include "csrc/common/shape.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_executable.h"

namespace torch_tpu {
namespace {

using absl_testing::IsOk;
using absl_testing::StatusIs;
using testing::HasSubstr;

// The registry is a process-wide, append-only singleton, so each test
// registers dummy kernels under its own shapeless key to stay independent of
// the other tests.

// Never called: the tests only exercise the registry's bookkeeping.
absl::StatusOr<std::unique_ptr<xla::PjRtLoadedExecutable>> BuildDummyKernel(
    xla::PjRtClient&, UniqueCompileOptions) {
  return TT_ERROR(error::kInternal) << "the dummy kernel must not be built";
}

int64_t InputRank(absl::Span<const Shape> input_shapes) {
  return input_shapes.size() == 1 ? input_shapes[0].dimensions().size() : -1;
}

bool AcceptsRank1Input(absl::Span<const Shape> input_shapes,
                       absl::Span<const Shape> /*output_shapes*/) {
  return InputRank(input_shapes) == 1;
}

bool AcceptsRank2Input(absl::Span<const Shape> input_shapes,
                       absl::Span<const Shape> /*output_shapes*/) {
  return InputRank(input_shapes) == 2;
}

bool AcceptsAnyInput(absl::Span<const Shape> /*input_shapes*/,
                     absl::Span<const Shape> /*output_shapes*/) {
  return true;
}

ShapelessKey DummyShapelessKey(std::string_view name) {
  return ShapelessKey(Fingerprint(name));
}

// Returns a cache key under `shapeless_key`. Kernels sharing a shapeless key
// are told apart by `variant`, which goes into the dimensions key.
CompilationCacheKey DummyKey(const ShapelessKey& shapeless_key,
                             int64_t variant = 0) {
  return CompilationCacheKey(GraphKey(shapeless_key, DimensionsKey({variant})),
                             CompileOptionsKey({}));
}

UnboundedDynamicKernel DummyKernel(
    CompilationCacheKey key,
    UnboundedDynamicExecutionMetadata::CompatibilityFunc is_compatible) {
  return UnboundedDynamicKernel{
      .build = &BuildDummyKernel,
      .metadata =
          UnboundedDynamicExecutionMetadata{.is_compatible = is_compatible},
      .key = key,
  };
}

Shape Rank2Input() { return Shape(Dimensions({4, 4}), mlir::ElementType::F32); }
Shape Rank3Input() {
  return Shape(Dimensions({2, 4, 4}), mlir::ElementType::F32);
}
Shape Output() { return Shape(Dimensions({5, 2}), mlir::ElementType::I64); }

TEST(UnboundedDynamicKernelRegistryTest,
     FindUnboundedDynamicKernelSkipsIncompatibleKernels) {
  UnboundedDynamicKernelRegistry& registry =
      UnboundedDynamicKernelRegistry::GetInstance();
  const ShapelessKey shapeless_key = DummyShapelessKey("skips_incompatible");
  const CompilationCacheKey rank1_key = DummyKey(shapeless_key, 1);
  const CompilationCacheKey rank2_key = DummyKey(shapeless_key, 2);
  const CompilationCacheKey any_key = DummyKey(shapeless_key, 3);
  ASSERT_THAT(
      registry.RegisterKernel(DummyKernel(rank1_key, AcceptsRank1Input)),
      IsOk());
  ASSERT_THAT(
      registry.RegisterKernel(DummyKernel(rank2_key, AcceptsRank2Input)),
      IsOk());
  ASSERT_THAT(registry.RegisterKernel(DummyKernel(any_key, AcceptsAnyInput)),
              IsOk());

  // The incompatible rank-1 kernel is skipped, and the rank-2 kernel is
  // returned rather than the later, also compatible, catch-all kernel.
  const UnboundedDynamicKernel* kernel = registry.FindUnboundedDynamicKernel(
      shapeless_key, {Rank2Input()}, {Output()});
  ASSERT_NE(kernel, nullptr);
  EXPECT_EQ(kernel->key, rank2_key);
  EXPECT_EQ(kernel->build, &BuildDummyKernel);

  // Only the catch-all kernel accepts a rank-3 input.
  kernel = registry.FindUnboundedDynamicKernel(shapeless_key, {Rank3Input()},
                                               {Output()});
  ASSERT_NE(kernel, nullptr);
  EXPECT_EQ(kernel->key, any_key);
}

TEST(UnboundedDynamicKernelRegistryTest,
     FindUnboundedDynamicKernelReturnsNullIfNoMatch) {
  UnboundedDynamicKernelRegistry& registry =
      UnboundedDynamicKernelRegistry::GetInstance();
  const ShapelessKey shapeless_key = DummyShapelessKey("no_match");
  ASSERT_THAT(registry.RegisterKernel(
                  DummyKernel(DummyKey(shapeless_key), AcceptsRank1Input)),
              IsOk());

  // No kernel is registered under the shapeless key.
  EXPECT_EQ(registry.FindUnboundedDynamicKernel(
                DummyShapelessKey("no_match_unregistered"), {Rank2Input()},
                {Output()}),
            nullptr);
  // The only kernel under the shapeless key rejects the shapes.
  EXPECT_EQ(registry.FindUnboundedDynamicKernel(shapeless_key, {Rank2Input()},
                                                {Output()}),
            nullptr);
}

TEST(UnboundedDynamicKernelRegistryTest, GetUnboundedDynamicMetadata) {
  UnboundedDynamicKernelRegistry& registry =
      UnboundedDynamicKernelRegistry::GetInstance();
  const ShapelessKey shapeless_key = DummyShapelessKey("get_exact_key");
  const CompilationCacheKey key = DummyKey(shapeless_key, 1);
  ASSERT_THAT(registry.RegisterKernel(DummyKernel(key, AcceptsRank2Input)),
              IsOk());

  const UnboundedDynamicExecutionMetadata* metadata =
      registry.GetUnboundedDynamicMetadata(key);
  ASSERT_NE(metadata, nullptr);
  EXPECT_EQ(metadata->is_compatible, &AcceptsRank2Input);

  // Same shapeless key, different dimensions key.
  EXPECT_EQ(registry.GetUnboundedDynamicMetadata(DummyKey(shapeless_key, 2)),
            nullptr);
}

TEST(UnboundedDynamicKernelRegistryTest, RegisterKernelRejectsDuplicateKey) {
  UnboundedDynamicKernelRegistry& registry =
      UnboundedDynamicKernelRegistry::GetInstance();
  const CompilationCacheKey key = DummyKey(DummyShapelessKey("duplicate_key"));
  ASSERT_THAT(registry.RegisterKernel(DummyKernel(key, AcceptsRank1Input)),
              IsOk());

  EXPECT_THAT(
      registry.RegisterKernel(DummyKernel(key, AcceptsRank2Input)),
      StatusIs(error::kAlreadyExists,
               HasSubstr("an unbounded dynamic kernel is already registered "
                         "for key")));
  // The first registration is kept.
  const UnboundedDynamicExecutionMetadata* metadata =
      registry.GetUnboundedDynamicMetadata(key);
  ASSERT_NE(metadata, nullptr);
  EXPECT_EQ(metadata->is_compatible, &AcceptsRank1Input);
}

}  // namespace
}  // namespace torch_tpu
