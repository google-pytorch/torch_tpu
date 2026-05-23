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

#include "torch_tpu/eager/op_dispatcher.h"

#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "ATen/core/ATen_fwd.h"
#include "gtest/gtest.h"
#include "torch_tpu/common/cache_key.h"

namespace torch_tpu {
namespace {

TEST(PromoteScalar, Single) {
  at::Scalar s(1.0);
  auto ps = PromoteScalar(s);
  EXPECT_EQ(ps.scalar().toDouble(), 1.0);
  // Call tensor() to avoid dtor crash.
  ps.GetTensor().IgnoreError();
}

TEST(PromoteScalar, Optional) {
  std::optional<at::Scalar> os(2.0);
  auto ops = PromoteScalar(os);
  ASSERT_TRUE(ops.has_value());
  EXPECT_EQ(ops->scalar().toDouble(), 2.0);
  // Call tensor() to avoid dtor crash.
  ops->GetTensor().IgnoreError();

  std::optional<at::Scalar> empty_os;
  auto empty_ops = PromoteScalar(empty_os);
  EXPECT_FALSE(empty_ops.has_value());
}

TEST(PromoteScalar, Array) {
  std::vector<at::Scalar> vs = {at::Scalar(3.0), at::Scalar(4.0)};
  auto vps = PromoteScalar(vs);
  ASSERT_EQ(vps.size(), 2);
  // Call tensor() to avoid dtor crash.
  vps[0].GetTensor().IgnoreError();
  vps[1].GetTensor().IgnoreError();

  EXPECT_EQ(vps[0].scalar().toDouble(), 3.0);
  EXPECT_EQ(vps[1].scalar().toDouble(), 4.0);
}

TEST(FormatParamCacheKey, OptionalPromotedScalar) {
  at::Scalar s(1.0);
  auto ps = PromoteScalar(s);
  std::optional<PromotedScalar> ops = std::move(ps);
  EXPECT_EQ(internal::FormatParamCacheKey(ops), "s");
  // Call tensor() to avoid dtor crash.
  ops->GetTensor().IgnoreError();

  std::optional<PromotedScalar> empty;
  EXPECT_EQ(internal::FormatParamCacheKey(empty), "");
}

}  // namespace
}  // namespace torch_tpu
