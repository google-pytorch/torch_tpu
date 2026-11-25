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

#include "csrc/experimental/precompiled_kernels/unique.h"

#include <cstdint>
#include <limits>
#include <vector>

#include "csrc/common/shape.h"
#include "csrc/common/unbounded_dynamic_kernel_registry.h"
#include "gtest/gtest.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"

namespace torch_tpu {
namespace {

using ::mlir::ElementType;

constexpr int64_t kMaxSupportedN = std::numeric_limits<int32_t>::max() - 1024;

Shape WithTile(Shape shape, const int64_t tile) {
  shape.set_layout({.minor_to_major = {0}, .tiles = {{tile}}});
  return shape;
}

Shape I32(const int64_t n) { return Shape({n}, ElementType::I32); }

struct TestCase {
  int kernel_idx = 0;  // 0: T(1024), 1: T(512), 2: T(256), 3: T(128).
  std::vector<Shape> inputs = {I32(3000)};
  std::vector<Shape> outputs = {I32(700)};
  bool expected = false;
};

using UniqueTest = ::testing::TestWithParam<TestCase>;

// Execution, caching and default-layout serving are tested through
// torch.unique in tests/dynamic_ops_test.py, and the extentless key fixes the
// number, ranks and dtypes of the operands; this covers the size bounds and
// non-default layout checks that Python tests can't reach.
TEST_P(UniqueTest, IsCompatible) {
  const auto& is_compatible =
      GetUniqueUnboundedDynamicKernels()[GetParam().kernel_idx]
          .kernel.metadata.is_compatible;
  EXPECT_EQ(is_compatible(GetParam().inputs, GetParam().outputs),
            GetParam().expected);
}

INSTANTIATE_TEST_SUITE_P(UniqueTests, UniqueTest,
                         testing::ValuesIn<TestCase>({
                             // Default layouts, implicit or explicit.
                             {.expected = true},
                             {.inputs = {WithTile(I32(3000), 1024)},
                              .outputs = {WithTile(I32(700), 1024)},
                              .expected = true},
                             {.inputs = {I32(kMaxSupportedN)},
                              .outputs = {I32(1)},
                              .expected = true},
                             {.kernel_idx = 1,
                              .inputs = {I32(300)},
                              .outputs = {I32(200)},
                              .expected = true},
                             // Out-of-range sizes.
                             {.inputs = {I32(0)}, .outputs = {I32(1)}},
                             {.inputs = {I32(kMaxSupportedN + 1)},
                              .outputs = {I32(1)}},
                             {.inputs = {I32(3000)}, .outputs = {I32(0)}},
                             // Non-default input or output layouts.
                             {.inputs = {WithTile(I32(3000), 512)}},
                             {.outputs = {WithTile(I32(700), 512)}},
                             {.outputs = {WithTile(I32(100), 256)}},
                         }));

}  // namespace
}  // namespace torch_tpu
