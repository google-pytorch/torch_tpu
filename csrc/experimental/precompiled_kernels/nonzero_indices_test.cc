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

#include "csrc/experimental/precompiled_kernels/nonzero_indices.h"

#include <vector>

#include "csrc/common/compilation.h"
#include "csrc/common/dimension_types.h"
#include "csrc/common/shape.h"
#include "gtest/gtest.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"

namespace torch_tpu {
namespace {

using ::mlir::ElementType;

Shape WithLayout(Shape shape, const Indices& order, const Indices& tile) {
  shape.set_layout({.minor_to_major = order, .tiles = {tile}});
  return shape;
}

struct TestCase {
  std::vector<Shape> inputs = {Shape({17, 257}, ElementType::F32)};
  std::vector<Shape> outputs = {Shape({5, 2}, ElementType::I64)};
  bool expected = false;
};

using NonzeroIndicesTest = ::testing::TestWithParam<TestCase>;

// Execution, caching and compatibility with default layouts are tested through
// torch.nonzero in tests/dynamic_ops_test.py, and the shapeless key fixes the
// number, ranks and dtypes of the operands; this covers the compatibility
// checks that neither can reach.
TEST_P(NonzeroIndicesTest, IsCompatible) {
  EXPECT_EQ(GetNonzeroUnboundedDynamicKernel().metadata.is_compatible(
                GetParam().inputs, GetParam().outputs),
            GetParam().expected);
}

INSTANTIATE_TEST_SUITE_P(
    NonzeroIndicesTests, NonzeroIndicesTest,
    testing::ValuesIn<TestCase>({
        // Default layouts differ from the kernel's, overridden to match.
        {.inputs = {WithLayout(Shape({4, 4}, ElementType::F32), {1, 0},
                               {8, 128})},
         .outputs = {WithLayout(Shape({2, 2}, ElementType::I64), {0, 1},
                                {2, 128})},
         .expected = true},
        {.outputs = {Shape({5, 3}, ElementType::I64)}},
        {.outputs = {WithLayout(Shape({5, 2}, ElementType::I64), {0, 1},
                                {4, 128})}},
    }));

}  // namespace
}  // namespace torch_tpu
