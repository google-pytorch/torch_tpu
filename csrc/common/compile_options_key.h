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

#ifndef TORCH_TPU_CSRC_COMMON_COMPILE_OPTIONS_KEY_H_
#define TORCH_TPU_CSRC_COMMON_COMPILE_OPTIONS_KEY_H_

#include <cstddef>

#include "absl/strings/str_format.h"
#include "csrc/common/fingerprint_utils.h"

namespace torch_tpu {

class CompileOptionsKey {
 public:
  explicit CompileOptionsKey(FingerprintType key) : key_(key) {}

  struct Hash {
    [[nodiscard]] inline size_t operator()(const CompileOptionsKey key) const {
      return key.key_;
    }
  };

  [[nodiscard]] bool operator==(const CompileOptionsKey rhs) const {
    return key_ == rhs.key_;
  }

  [[nodiscard]] FingerprintType key() const { return key_; }

 private:
  FingerprintType key_;
};

// Formats a compile options key as a human-readable string.
template <typename Sink>
void AbslStringify(Sink& sink, const CompileOptionsKey key) {
  absl::Format(&sink, "%016x", key.key());
}

}  // namespace torch_tpu

#endif  // TORCH_TPU_CSRC_COMMON_COMPILE_OPTIONS_KEY_H_
