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

#include <string_view>

#include "csrc/common/cache_key.h"
#include "csrc/common/dimension_types.h"
#include "csrc/common/fingerprint_utils.h"
#include "csrc/common/shape.h"
#include "csrc/common/unbounded_dynamic_kernel_registry.h"
#include "csrc/experimental/precompiled_kernels/nonzero_indices.h"
#include "gtest/gtest.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"

namespace torch_tpu {
namespace {

// Verifies what :kernel_registration, linked into this test, registers with
// the unbounded dynamic kernel registry during static initialization.

TEST(KernelRegistrationTest,
     FindUnboundedDynamicKernelReturnsNonzeroKernelForCompatibleInput) {
  const CompilationCacheKey nonzero_key = NonZeroCacheKey();

  const UnboundedDynamicKernel* kernel =
      UnboundedDynamicKernelRegistry::GetInstance().FindUnboundedDynamicKernel(
          nonzero_key.graph_key().shapeless_key(),
          {Shape(Dimensions({4, 4}), mlir::ElementType::F32)},
          {Shape(Dimensions({5, 2}), mlir::ElementType::I64)});
  ASSERT_NE(kernel, nullptr);
  EXPECT_EQ(kernel->key, nonzero_key);
  EXPECT_EQ(kernel->build, &BuildNonzeroIndicesExecutable);
}

TEST(KernelRegistrationTest,
     FindUnboundedDynamicKernelRejectsIncompatibleLookups) {
  UnboundedDynamicKernelRegistry& registry =
      UnboundedDynamicKernelRegistry::GetInstance();
  const ShapelessKey nonzero_shapeless_key =
      NonZeroCacheKey().graph_key().shapeless_key();
  const Shape output_shape(Dimensions({5, 2}), mlir::ElementType::I64);

  // Unrelated shapeless key.
  EXPECT_EQ(
      registry.FindUnboundedDynamicKernel(
          ShapelessKey(Fingerprint(std::string_view("dummykey"))),
          {Shape(Dimensions({4, 4}), mlir::ElementType::F32)}, {output_shape}),
      nullptr);
  // Input rank other than 2.
  EXPECT_EQ(registry.FindUnboundedDynamicKernel(
                nonzero_shapeless_key,
                {Shape(Dimensions({2, 4, 4}), mlir::ElementType::F32)},
                {output_shape}),
            nullptr);
  // Input dtype other than F32.
  EXPECT_EQ(
      registry.FindUnboundedDynamicKernel(
          nonzero_shapeless_key,
          {Shape(Dimensions({4, 4}), mlir::ElementType::BF16)}, {output_shape}),
      nullptr);
  // More than one input.
  EXPECT_EQ(registry.FindUnboundedDynamicKernel(
                nonzero_shapeless_key,
                {Shape(Dimensions({4, 4}), mlir::ElementType::F32),
                 Shape(Dimensions({4, 4}), mlir::ElementType::F32)},
                {output_shape}),
            nullptr);
  // Output dtype other than I64.
  EXPECT_EQ(registry.FindUnboundedDynamicKernel(
                nonzero_shapeless_key,
                {Shape(Dimensions({4, 4}), mlir::ElementType::F32)},
                {Shape(Dimensions({5, 2}), mlir::ElementType::I32)}),
            nullptr);
}

TEST(KernelRegistrationTest, KernelsAreBucketedByShapelessKey) {
  const CompilationCacheKey nonzero_key = NonZeroCacheKey();
  const UnboundedDynamicKernelRegistry::KernelMap& kernels =
      UnboundedDynamicKernelRegistry::GetInstance().kernels();

  const auto it = kernels.find(nonzero_key.graph_key().shapeless_key());
  ASSERT_NE(it, kernels.end());
  ASSERT_EQ(it->second.size(), 1);
  EXPECT_EQ(it->second[0]->key, nonzero_key);
  EXPECT_EQ(
      kernels.count(ShapelessKey(Fingerprint(std::string_view("dummykey")))),
      0);
}

}  // namespace
}  // namespace torch_tpu
