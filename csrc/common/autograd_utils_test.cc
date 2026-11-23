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

#include "csrc/common/autograd_utils.h"

#include <memory>
#include <string>
#include <vector>

#include "ATen/core/TensorBody.h"
#include "ATen/ops/ones.h"
#include "c10/util/intrusive_ptr.h"
#include "csrc/common/macro_utils.h"
#include "gtest/gtest.h"
#if TT_TORCH_VERSION_GE(2, 13)
#include "torch/csrc/autograd/node.h"
#else
#include "torch/csrc/autograd/function.h"
#endif

namespace torch_tpu {
namespace {

class NegateNode : public torch::autograd::Node {
 public:
  ~NegateNode() override {
    if (destroyed_ != nullptr) {
      *destroyed_ = true;
    }
  }

  std::string name() const override { return "NegateNode0"; }

  void set_destroyed_flag(bool* destroyed) { destroyed_ = destroyed; }

 protected:
  std::vector<at::Tensor> apply(std::vector<at::Tensor>&& inputs) override {
    return {inputs[0].neg()};
  }

 private:
  bool* destroyed_ = nullptr;
};

class NonNodeIntrusiveTarget : public c10::intrusive_ptr_target {};

template <typename T>
concept CanFireNodeCreationHooks =
    requires(const T& node) { FireNodeCreationHooks(node); };

static_assert(
    CanFireNodeCreationHooks<decltype(MakeAutogradNode<NegateNode>())>);
static_assert(CanFireNodeCreationHooks<AutogradNodePtr<NegateNode>>);
static_assert(!CanFireNodeCreationHooks<int>);
static_assert(!CanFireNodeCreationHooks<NegateNode*>);
static_assert(!CanFireNodeCreationHooks<std::shared_ptr<int>>);
static_assert(!CanFireNodeCreationHooks<std::unique_ptr<NegateNode>>);
static_assert(
    !CanFireNodeCreationHooks<c10::intrusive_ptr<NonNodeIntrusiveTarget>>);

TEST(AutogradUtilsTest, MakeAutogradNodeRunsApplyAndDestroysNode) {
  bool destroyed = false;
  {
    auto node = MakeAutogradNode<NegateNode>();
    node->set_destroyed_flag(&destroyed);
    const at::Tensor input = at::ones({3});
    const std::vector<at::Tensor> outputs = (*node)({input});
    ASSERT_EQ(outputs.size(), 1);
    EXPECT_TRUE(outputs[0].equal(at::ones({3}).neg()));
    EXPECT_FALSE(destroyed);
  }
  EXPECT_TRUE(destroyed);
}

TEST(AutogradUtilsTest, FireNodeCreationHooksRunsOnNodePointer) {
  const auto node = MakeAutogradNode<NegateNode>();
#if TT_TORCH_VERSION_GE(2, 14)
  EXPECT_FALSE(node->node_creation_hooks_fired());
#endif
  FireNodeCreationHooks(node);
#if TT_TORCH_VERSION_GE(2, 14)
  EXPECT_TRUE(node->node_creation_hooks_fired());
#endif
}

}  // namespace
}  // namespace torch_tpu
