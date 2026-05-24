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

#include "torch_tpu/eager/repeated_ops_heuristic.h"

#include <algorithm>
#include <cstddef>
#include <deque>
#include <optional>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/const_init.h"
#include "absl/base/no_destructor.h"
#include "absl/flags/declare.h"
#include "absl/log/absl_check.h"
#include "absl/log/absl_log.h"
#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "torch_tpu/common/env_vars.h"
#include "torch_tpu/eager/materialize.h"

namespace torch_tpu {

namespace {

constexpr int kMinRepeatedSubsequenceLength = 2;
constexpr int kMaxRepeatedSubsequenceLength = 128;
constexpr std::string_view kRepeatedOpSafeMode = "safe";
constexpr std::string_view kRepeatedOpAggressiveMode = "aggressive";

class OpWindow {
 public:
  OpWindow(size_t min_repeated_sequence_size, size_t max_repeated_sequence_size)
      : min_repeated_sequence_size_(min_repeated_sequence_size),
        // We need a history that is at least 2 times the maximum repeated
        // sequence size we want to detect.
        max_history_size_(2 * max_repeated_sequence_size) {
    Reset();
  }

  absl::Mutex& mutex() { return mutex_; }

  void Reset() {
    history_.clear();
    pos_ = 0;
    next_valid_pos_ = 0;
    last_repeated_sequence_size_ = 0;
  }

  void Append(const DeferredOp& op) {
    // Here we store only a quick hash of the op. At the op level, this will
    // lead to a high probability of collision, however, that probability
    // decreases quickly as we match longer sequences.
    history_.push_back(op.Hash());
    if (history_.size() > max_history_size_) {
      history_.pop_front();
    }
    // Increase the number of ops we have encountered so far.
    pos_++;
    // However, for very long running op sequences, we wrap the counters around
    // to avoid over overflow.
    const auto wrap = 2 * max_history_size_;
    if (pos_ > wrap) {
      pos_ -= wrap;
      if (next_valid_pos_ > wrap) {
        next_valid_pos_ -= wrap;
      } else {
        next_valid_pos_ = pos_;
      }
    }
  }

  // Return true if the last dispatched op is the end of a repeated suffix in
  // the sequence of ops that have been dispatched so far.
  bool FindRepeatedSequence() {
    if (pos_ < next_valid_pos_) {
      return false;
    }

    // Skip if the accumulated ops are not enough.
    auto n = history_.size();
    if (n < 2 * min_repeated_sequence_size_) {
      return false;
    }

    // If last time we found a repeated sequence of size S, chances are we are
    // going to find another one of the same size. Hence, speculatively check
    // that size first, and only later check for repeated sequences of all other
    // sizes.
    if (last_repeated_sequence_size_ > 0 &&
        // Ensure we have sufficient elements in the history to check for the
        // match.
        n >= 2 * last_repeated_sequence_size_) {
      auto i = (n - 1 - last_repeated_sequence_size_);
      if (IsMatch(i)) {
        return true;
      }
    }

    if (ABSL_VLOG_IS_ON(1)) {
      std::ostringstream os;
      for (auto x : history_) {
        os << static_cast<char>(x | 0x20);  // Make the value printable.
      }
      ABSL_VLOG(1) << os.str();
    }

    // Here we perform an exhaustive search for all possible sizes.
    for (auto i = n - 2; i >= n / 2; --i) {
      if (IsMatch(i)) {
        ABSL_VLOG(1) << std::string(i, ' ') << "^";
        return true;
      }
    }
    return false;
  }

 private:
  // Return true if there is a repeated sequence in the history at position i,
  // specifically if history[2i+2-n:i] = history[i+1:n-1].
  bool IsMatch(size_t i) {
    if (history_[i] == history_.back()) {
      auto n = history_.size();
      size_t repeated_sequence_size = (n - i - 1);
      if (repeated_sequence_size > min_repeated_sequence_size_) {
        ABSL_CHECK((i - repeated_sequence_size + 1) >= 0);  // CRASH_OK
        bool has_repeated_sequence =
            std::equal(history_.begin() + (i - repeated_sequence_size + 1),
                       history_.begin() + i + 1,
                       history_.end() - repeated_sequence_size, history_.end());
        if (has_repeated_sequence) {
          ABSL_VLOG(1) << "Found repeated subsequence of length "
                       << repeated_sequence_size << ", pos=" << pos_;
          // Now that we have found a repeated sequence, we need to skip the
          // next search to `repeated_sequence_size` ops ahead.
          next_valid_pos_ = pos_ + repeated_sequence_size;
          last_repeated_sequence_size_ = repeated_sequence_size;
          return true;
        }
      }
    }
    return false;
  }

  absl::Mutex mutex_;
  size_t min_repeated_sequence_size_;
  size_t max_history_size_;
  std::deque<size_t> history_;
  size_t pos_;
  size_t next_valid_pos_;
  size_t last_repeated_sequence_size_;
};

OpWindow& GetOpWindow() {
  static absl::NoDestructor<OpWindow> g_op_window(
      kMinRepeatedSubsequenceLength, kMaxRepeatedSubsequenceLength);
  return *g_op_window;
}

}  // namespace

bool MustApplyRepeatedOpsHeuristic() {
  const std::string& detect_repeated_ops =
      GetEnvOnce<kTorchTpuInternalDetectRepeatedOpsEnvVar>().value_or(
          std::string(kRepeatedOpSafeMode));
  return (detect_repeated_ops == kRepeatedOpSafeMode ||
          detect_repeated_ops == kRepeatedOpAggressiveMode);
}

absl::Status ApplyRepeatedOpsHeuristic(
    const SharedDeviceBufferList& device_buffer_list) {
  const std::string& detect_repeated_ops =
      GetEnvOnce<kTorchTpuInternalDetectRepeatedOpsEnvVar>().value_or(
          std::string(kRepeatedOpSafeMode));

  // Note that view operations (like reshapes and transposes) don't go
  // through the op_dispatcher sequence. So the heuristic below considers
  // only non-view ops.
  const auto op = device_buffer_list->deferred_op();
  ABSL_CHECK(op != nullptr);  // CRASH_OK
  bool found_repetition = false;
  {
    auto& op_window = GetOpWindow();
    absl::MutexLock lock(op_window.mutex());
    op_window.Append(*op);
    found_repetition = op_window.FindRepeatedSequence();
  }

  if (found_repetition) {
    auto materialization_mode =
        (detect_repeated_ops == kRepeatedOpAggressiveMode)
            ? MaterializationMode::kFullGraph
            : MaterializationMode::kSplitGraph;
    TT_RETURN_IF_ERROR(Materialize(device_buffer_list,
                                   MaterializationReason::kRepeatedOpDetection,
                                   materialization_mode));
  }

  return absl::OkStatus();
}

void ResetRepeatedOpsHeuristicState() {
  auto& op_window = GetOpWindow();
  absl::MutexLock lock(op_window.mutex());
  op_window.Reset();
}

}  // namespace torch_tpu
