// Copyright 2026 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "torch_tpu/runtime/backend_registry.h"

#include <memory>

#include "absl/status/status.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "torch_tpu/runtime/backend.h"
#include "torch_tpu/runtime/mock_backend.h"

namespace torch_tpu {
namespace {

class BackendRegistryTest : public testing::Test {
 protected:
  void TearDown() override { ResetRegisteredBackendFactories(); }
};

TEST_F(BackendRegistryTest, RegisterAndCreateCustomBackend) {
  EXPECT_FALSE(HasRegisteredBackendFactory(BackendKind::kCerebro));

  RegisterBackendFactory(BackendKind::kCerebro,
                         [](const BackendOptions& options) {
                           return std::make_unique<MockBackend>();
                         });

  EXPECT_TRUE(HasRegisteredBackendFactory(BackendKind::kCerebro));

  BackendOptions options;
  ASSERT_OK_AND_ASSIGN(auto backend,
                       Backend::Create(BackendKind::kCerebro, options));
  ASSERT_NE(backend, nullptr);
  EXPECT_NE(dynamic_cast<MockBackend*>(backend.get()), nullptr);
}

TEST_F(BackendRegistryTest, CreateUnregisteredFails) {
  BackendOptions options;
  auto unregistered_or =
      CreateRegisteredBackend(BackendKind::kCerebro, options);
  EXPECT_FALSE(unregistered_or.ok());
}

}  // namespace
}  // namespace torch_tpu
