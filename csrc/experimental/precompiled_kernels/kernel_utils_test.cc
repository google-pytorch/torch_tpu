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

#include "csrc/experimental/precompiled_kernels/kernel_utils.h"

#include "csrc/common/cache_key.h"
#include "csrc/ops/op_names.h"
#include "gtest/gtest.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"

namespace torch_tpu {
namespace {

using ::mlir::ElementType;

TEST(KernelUtilsTest, UnaryOpExtentlessKeyDistinguishesSignature) {
  const ExtentlessKey nonzero_1d =
      UnaryOpExtentlessKey(OpName::kNonzero, /*in_rank=*/1, ElementType::F32,
                           /*out_rank=*/2, ElementType::I64);
  const ExtentlessKey nonzero_2d =
      UnaryOpExtentlessKey(OpName::kNonzero, /*in_rank=*/2, ElementType::F32,
                           /*out_rank=*/2, ElementType::I64);
  const ExtentlessKey unique_1d = UnaryOpExtentlessKey(
      OpName::kUniqueValuesFromSorted, /*in_rank=*/1, ElementType::I32,
      /*out_rank=*/1, ElementType::I32);
  EXPECT_EQ(nonzero_1d, UnaryOpExtentlessKey(OpName::kNonzero, /*in_rank=*/1,
                                             ElementType::F32, /*out_rank=*/2,
                                             ElementType::I64));
  EXPECT_NE(nonzero_1d, nonzero_2d);
  EXPECT_NE(nonzero_1d, unique_1d);
}

}  // namespace
}  // namespace torch_tpu
