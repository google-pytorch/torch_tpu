/*
 * Copyright 2025 Google LLC
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

#include "csrc/ops/op_names.h"

#include <string_view>

#include "absl/strings/match.h"

namespace torch_tpu {

std::string_view ToBaseName(const OpName op_name) {
  if (op_name == OpName::kIndexPutImpl_) {
    return "index_put_";
  }
  if (op_name == OpName::kCtcLossTensor) {
    return "_ctc_loss";
  }

  // Remove leading underscore (internal op) if present.
  //
  // Exception: if op_name is a dunder method (e.g. __lshift__), do not remove
  // the leading underscore.
  auto full_name = ToString(op_name);
  if (!full_name.empty() && full_name.front() == '_' &&
      !full_name.starts_with("__")) {
    full_name.remove_prefix(1);
  }

  // Distributed ops are named like "distributed.foo" and base name is the same
  // as the op name.
  if (absl::StartsWith(full_name, "distributed.")) {
    return full_name;
  }
  const auto last_dot_pos = full_name.rfind('.');
  return last_dot_pos == std::string_view::npos
             ? full_name
             : full_name.substr(0, last_dot_pos);
}

}  // namespace torch_tpu
