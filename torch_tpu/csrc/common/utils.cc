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

#include "torch_tpu/csrc/common/utils.h"

#include <cstdint>
#include <string>
#include <string_view>

#include "ATen/core/TensorBody.h"
#include "absl/log/absl_log.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_split.h"
#include "torch/headeronly/util/complex.h"

namespace torch_tpu {

// TODO: b/441998432 - Use to_chars() once the OSS toolchain (currently clang
// 18) supports it for doubles.
std::string LosslessToString(double value) {
  // Precision of 17 is chosen because doubles max out at 15 to 17 digits
  // of precision, so take the higher precision.
  return absl::StrFormat("%.17g", value);
}

std::string LosslessToString(const c10::complex<double>& value) {
  return absl::StrCat(LosslessToString(value.real()), "+",
                      LosslessToString(value.imag()), "i");
}

// To prevent the log system from truncating the log messages, each
// message shouldn't exceed this number of bytes.
constexpr int kMaxLogMessageSize = 14000;

void LogLines(std::string_view s) {
  // For readability, we split the string to chunks of up to kMaxLogMessageSize
  // bytes and log each chunk separately. This is better than logging line by
  // line because it reduces the number of log message prefixes.
  const std::string_view delimiter = "\n";
  const auto lines = absl::StrSplit(s, delimiter);
  std::string chunk;
  for (const auto& line : lines) {
    if (chunk.size() + delimiter.size() + line.size() >= kMaxLogMessageSize) {
      if (!chunk.empty()) {
        ABSL_LOG(INFO) << chunk;
        chunk.clear();
      }
    }
    if (line.size() > kMaxLogMessageSize) {
      ABSL_LOG(INFO) << line;
      continue;
    }
    absl::StrAppend(&chunk, delimiter, line);
  }
  if (!chunk.empty()) {
    ABSL_LOG(INFO) << chunk;
  }
}

std::string PercAsStr(uint64_t num, uint64_t den) {
  if (den == 0) {
    return "NA";
  }
  auto k = num * 1000 / den;
  return absl::StrCat(k / 10, ".", k % 10, "%");
}

bool TensorHasTrivialLayout(const at::Tensor& tensor) {
  return tensor.is_contiguous() && tensor.storage_offset() == 0 &&
         tensor.numel() > 0;
}

}  // namespace torch_tpu
