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

#ifndef TORCH_TPU_COMMON_SYMBOL_STAGE_H_
#define TORCH_TPU_COMMON_SYMBOL_STAGE_H_

#include <stdexcept>
#include <string_view>

namespace torch_tpu {

// Represents the stability and visibility stage of a TorchTPU symbol.
//
// A SymbolStage can be one of:
// - kInternalImplementation: an internal implementation detail
// - kInternalApi: an internal API
// - kExperimental: an experimental API
// - kStable: a stable API
// - (kDeprecated, version): a deprecated API with a target deprecation version
//   (e.g. "2.13")
//
// The difference between kInternalImplementation and kInternalApi is that
// the former is used by TorchTPU to implement its functionalities, while the
// latter is for controlling TorchTPU's behavior.
//
// This class is fully constexpr and compliant with C++20.
class SymbolStage {
 public:
  // Observers
  [[nodiscard]] constexpr std::string_view version() const noexcept {
    return version_;
  }

  [[nodiscard]] constexpr bool is_internal_implementation() const noexcept {
    return stage_ == Stage::kInternalImplementation;
  }
  [[nodiscard]] constexpr bool is_internal_api() const noexcept {
    return stage_ == Stage::kInternalApi;
  }
  [[nodiscard]] constexpr bool is_experimental() const noexcept {
    return stage_ == Stage::kExperimental;
  }
  [[nodiscard]] constexpr bool is_stable() const noexcept {
    return stage_ == Stage::kStable;
  }
  [[nodiscard]] constexpr bool is_deprecated() const noexcept {
    return stage_ == Stage::kDeprecated;
  }

  // Equality comparison
  friend constexpr bool operator==(const SymbolStage& lhs,
                                   const SymbolStage& rhs) = default;

  // Factory methods.
  static constexpr SymbolStage InternalImplementation() {
    return SymbolStage(Stage::kInternalImplementation);
  };
  static constexpr SymbolStage InternalApi() {
    return SymbolStage(Stage::kInternalApi);
  };
  static constexpr SymbolStage Experimental() {
    return SymbolStage(Stage::kExperimental);
  };
  static constexpr SymbolStage Stable() { return SymbolStage(Stage::kStable); };
  static constexpr SymbolStage Deprecated(std::string_view version) {
    return SymbolStage(Stage::kDeprecated, version);
  };

 private:
  enum class Stage {
    kInternalImplementation,
    kInternalApi,
    kExperimental,
    kStable,
    kDeprecated,
  };

  // Constructor for non-deprecated stages: kInternalImplementation,
  // kExperimental, kStable. Throws if kDeprecated is passed (use the 2-argument
  // constructor with version).
  constexpr explicit SymbolStage(Stage stage) : stage_(stage), version_("") {
    if (stage == Stage::kDeprecated) {
      throw std::invalid_argument(  // throw needed for compile-time validation.
          "SymbolStage: kDeprecated requires a version string.");
    }
  }

  // Constructor for kDeprecated stage with the given version string_view.
  // Throws if a non-deprecated stage or an empty version is passed.
  // The caller must ensure that the version string_view outlives the
  // SymbolStage object.
  constexpr SymbolStage(Stage stage, std::string_view version)
      : stage_(stage), version_(version) {
    if (stage != Stage::kDeprecated) {
      throw std::invalid_argument(  // throw needed for compile-time validation.
          "SymbolStage: non-deprecated stages cannot have a version string.");
    }
    if (version.empty()) {
      throw std::invalid_argument(  // throw needed for compile-time validation.
          "SymbolStage: version string cannot be empty for kDeprecated stage.");
    }
  }

  Stage stage_{Stage::kInternalImplementation};
  std::string_view version_{};
};

}  // namespace torch_tpu

#endif  // TORCH_TPU_COMMON_SYMBOL_STAGE_H_
