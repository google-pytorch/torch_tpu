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

#ifndef TORCH_TPU_CSRC_PJRT_FAKE_HLO_LOADED_EXECUTABLE_TEST_HELPER_H_
#define TORCH_TPU_CSRC_PJRT_FAKE_HLO_LOADED_EXECUTABLE_TEST_HELPER_H_

#include <memory>
#include <optional>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "csrc/common/error_utils.h"
#include "xla/future.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/parser/hlo_parser.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/pjrt/pjrt_layout.h"
#include "xla/service/device_assignment.h"
#include "xla/shape.h"

namespace torch_tpu {

// Fake `xla::PjRtLoadedExecutable` that only reports its optimized HLO: it has
// no layouts or output shapes, and cannot be executed.
class FakeHloLoadedExecutable final : public xla::PjRtLoadedExecutable {
 public:
  using HloModules = std::vector<std::shared_ptr<xla::HloModule>>;

  // `modules` is what GetHloModules() returns.
  explicit FakeHloLoadedExecutable(absl::StatusOr<HloModules> modules)
      : modules_(std::move(modules)) {}

  // Returns a fake with one module parsed from each of `hlo_texts`.
  static absl::StatusOr<std::unique_ptr<FakeHloLoadedExecutable>> FromHloTexts(
      const absl::Span<const std::string_view> hlo_texts) {
    HloModules modules;
    for (const std::string_view hlo_text : hlo_texts) {
      TT_ASSIGN_OR_RETURN(std::unique_ptr<xla::HloModule> module,
                          xla::ParseAndReturnUnverifiedModule(hlo_text));
      modules.push_back(std::move(module));
    }
    return std::make_unique<FakeHloLoadedExecutable>(std::move(modules));
  }

  absl::StatusOr<HloModules> GetHloModules() const override { return modules_; }
  absl::StatusOr<std::vector<xla::Shape>> GetOutputShapes() const override {
    return std::vector<xla::Shape>{};
  }
  absl::StatusOr<std::vector<std::shared_ptr<const xla::PjRtLayout>>>
  GetParameterLayouts() const override {
    return TT_ERROR(error::kPythonNotImplementedError)
           << "the fake executable has no layouts";
  }
  absl::StatusOr<std::vector<std::shared_ptr<const xla::PjRtLayout>>>
  GetOutputLayouts() const override {
    return TT_ERROR(error::kPythonNotImplementedError)
           << "the fake executable has no layouts";
  }

  xla::PjRtClient* client() const override { return nullptr; }
  const xla::DeviceAssignment& device_assignment() const override {
    return device_assignment_;
  }
  absl::Span<xla::PjRtDevice* const> addressable_devices() const override {
    return {};
  }
  void Delete() override {}
  bool IsDeleted() const override { return false; }

  absl::StatusOr<std::vector<std::vector<std::unique_ptr<xla::PjRtBuffer>>>>
  Execute(absl::Span<const std::vector<xla::PjRtBuffer*>> argument_handles,
          const xla::ExecuteOptions& options,
          std::optional<std::vector<xla::Future<>>>& returned_futures)
      const override {
    return TT_ERROR(error::kPythonNotImplementedError)
           << "the fake executable cannot run";
  }
  absl::StatusOr<std::vector<std::unique_ptr<xla::PjRtBuffer>>> ExecuteSharded(
      absl::Span<xla::PjRtBuffer* const> argument_handles,
      xla::PjRtDevice* device, const xla::ExecuteOptions& options,
      std::optional<xla::Future<>>& returned_future,
      bool fill_future) const override {
    return TT_ERROR(error::kPythonNotImplementedError)
           << "the fake executable cannot run";
  }
  absl::StatusOr<std::vector<std::unique_ptr<xla::PjRtBuffer>>> ExecutePortable(
      absl::Span<xla::PjRtBuffer* const> argument_handles,
      xla::PjRtDevice* device, const xla::ExecuteOptions& options,
      std::optional<xla::Future<>>& returned_future,
      bool fill_future) const override {
    return TT_ERROR(error::kPythonNotImplementedError)
           << "the fake executable cannot run";
  }

 private:
  absl::StatusOr<HloModules> modules_;
  xla::DeviceAssignment device_assignment_;
};

}  // namespace torch_tpu

#endif  // TORCH_TPU_CSRC_PJRT_FAKE_HLO_LOADED_EXECUTABLE_TEST_HELPER_H_
