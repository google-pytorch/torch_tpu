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

#ifndef TORCH_TPU_EAGER_STRUCTURED_LOG_BUFFER_H_
#define TORCH_TPU_EAGER_STRUCTURED_LOG_BUFFER_H_

#include <atomic>
#include <cstdint>
#include <deque>
#include <string>

#include "absl/base/no_destructor.h"
#include "absl/synchronization/mutex.h"

namespace torch_tpu {

enum class MaterializationReason {
  kCpuConversion,
  kScalarConversion,
  kExplicitSync,
  kAutogradSync,
  kRepeatedOpDetection,
  kDebugMode,
  kDistributedOp,
  kCompileModeExecution,
  kUnknown
};

inline const char* MaterializationReasonToString(MaterializationReason reason) {
  switch (reason) {
    case MaterializationReason::kCpuConversion:
      return ".cpu()";
    case MaterializationReason::kScalarConversion:
      return "item()";
    case MaterializationReason::kExplicitSync:
      return "synchronize()";
    case MaterializationReason::kAutogradSync:
      return "autograd_post_backward";
    case MaterializationReason::kRepeatedOpDetection:
      return "repeated_op_heuristic";
    case MaterializationReason::kDebugMode:
      return "debug_mode";
    case MaterializationReason::kDistributedOp:
      return "distributed_op";
    case MaterializationReason::kCompileModeExecution:
      return "compile_mode_execution";
    case MaterializationReason::kUnknown:
      return "unknown";
  }
  return "unknown";
}

struct EagerEvent {
  std::string name;
  int64_t ts_us;
  int64_t dur_us;
  std::string device;
  bool cache_hit;
  std::string aten_graph_payload;
  std::string mlir_payload;
  MaterializationReason reason;
};

class StructuredLogBuffer {
 public:
  static StructuredLogBuffer& Get();
  bool enabled() const { return enabled_; }
  void Push(EagerEvent event);
  std::deque<EagerEvent> Drain();
  void SetEnabledForTest(bool enabled) { enabled_ = enabled; }

 private:
  StructuredLogBuffer();
  friend class absl::NoDestructor<StructuredLogBuffer>;

  std::atomic<bool> enabled_;
  mutable absl::Mutex mu_;
  std::deque<EagerEvent> events_;
  static constexpr size_t kMaxEvents = 1000;
};

}  // namespace torch_tpu

#endif  // TORCH_TPU_EAGER_STRUCTURED_LOG_BUFFER_H_
