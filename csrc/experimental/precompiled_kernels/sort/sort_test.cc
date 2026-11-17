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

#include "csrc/experimental/precompiled_kernels/sort/sort.h"

#include <cstdint>
#include <vector>

#include "csrc/common/dimension_types.h"
#include "csrc/common/shape.h"
#include "csrc/common/unbounded_dynamic_kernel_registry.h"
#include "gtest/gtest.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"

namespace torch_tpu {
namespace {

using ::mlir::ElementType;

Shape WithTile(Shape shape, int64_t tile) {
  shape.set_layout({.minor_to_major = {0}, .tiles = {{tile}}});
  return shape;
}

Shape Keys() { return Shape({1024}, ElementType::I32); }
Shape Indices() { return Shape({1024}, ElementType::I64); }

struct TestCase {
  std::vector<Shape> inputs = {Keys()};
  std::vector<Shape> outputs = {Keys(), Indices()};
  bool expected = false;
};

using SortTest = ::testing::TestWithParam<TestCase>;

// Serving and dtypes are tested through torch.sort in
// tests/dynamic_ops_test.py, and the shapeless key fixes the number, ranks and
// dtypes of the operands; this covers the size and layout checks that Python
// can't reach.
TEST_P(SortTest, IsCompatible) {
  for (const UnboundedDynamicKernel& kernel :
       GetSortUnboundedDynamicKernels()) {
    EXPECT_EQ(
        kernel.metadata.is_compatible(GetParam().inputs, GetParam().outputs),
        GetParam().expected);
  }
}

INSTANTIATE_TEST_SUITE_P(SortTests, SortTest,
                         testing::ValuesIn<TestCase>({
                             // Default layouts, given implicitly or explicitly.
                             {.expected = true},
                             {.inputs = {WithTile(Keys(), 1024)},
                              .outputs = {WithTile(Keys(), 1024),
                                          WithTile(Indices(), 1024)},
                              .expected = true},
                             // Other sizes.
                             {.inputs = {Shape({1023}, ElementType::I32)},
                              .outputs = {Shape({1023}, ElementType::I32),
                                          Shape({1023}, ElementType::I64)}},
                             {.inputs = {Shape({2048}, ElementType::I32)},
                              .outputs = {Shape({2048}, ElementType::I32),
                                          Shape({2048}, ElementType::I64)}},
                             // Layouts other than the kernel's.
                             {.inputs = {WithTile(Keys(), 512)}},
                             {.outputs = {WithTile(Keys(), 512), Indices()}},
                             {.outputs = {Keys(), WithTile(Indices(), 512)}},
                         }));

}  // namespace
}  // namespace torch_tpu
