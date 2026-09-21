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

#ifndef TORCH_TPU_CSRC_COMMON_OP_CORRELATION_H_
#define TORCH_TPU_CSRC_COMMON_OP_CORRELATION_H_

#include <cstdint>
#include <string_view>
#include <vector>

#include "absl/types/span.h"

// Carries the PyTorch profiler's correlation ID from the thread that dispatches
// an ATen op to the thread that actually enqueues the resulting program to the
// TPU.
//
// Why this is needed
// ------------------
// TorchTPU execution is deferred. An ATen op dispatches on the caller's thread,
// records a DeferredOp, and returns; the DeferredOp is later compiled and
// enqueued from a different thread. By the time the program reaches the TPU,
// the RecordFunction that caused it is long gone, so nothing in the trace ties
// the device kernel back to the ATen op that asked for it.
//
// The profiler resolves that with a two-hop join, and this library supplies the
// piece that cannot be recovered from the trace geometry:
//
//   1. libkineto hands us the correlation ID of the currently-executing ATen op
//      via IActivityProfilerSession::pushCorrelationId(). We keep it in a
//      thread-local stack (`PushOpCorrelationId`). This is the *same* ID that
//      Kineto's `cpuActivity()` lookup is keyed on, so it is the only ID that
//      will actually resolve to a CPU activity later.
//   2. Each DeferredOp records the innermost such ID at construction time.
//   3. Just before a compiled program is executed, the execution thread
//      installs the correlation IDs of the ops in that program
//      (`ScopedLaunchCorrelation`), and the enqueue-side TraceMe stamps them
//      into the trace.
//
// Both stacks are thread-local and are only non-empty while a profiling session
// is active, so the cost when not profiling is a thread-local load.

namespace torch_tpu {

// The type a correlation ID must be stored and compared as.
//
// DO NOT widen this to uint64_t, however tempting, and however much the
// libkineto API appears to hand us 64 bits.
//
// libkineto delivers the ID to us as `uint64_t`, but the map we will eventually
// look it up in is keyed on a *narrowed* value:
//
//   kineto_shim.cpp:  act.id = static_cast<int32_t>(correlation_id);
//   GenericActivityProfiler.cpp:  activityMap_[act->correlationId()] = ...
//
// and the lookup callback itself takes an int32_t:
//
//   using getLinkedActivityCallback =
//       std::function<const ITraceActivity*(int32_t)>;
//
// PyTorch mints correlation IDs from a *global* monotonically increasing
// counter (`id_start_ = 1 + ChunkSize * counter_++`), so in a long-running
// process they do cross 2^31. Past that point a 64-bit key would miss every
// single lookup, silently and with no error -- correlation would simply stop
// working. Narrowing here, with exactly the same cast the shim uses, keeps our
// key bit-identical to the map's key at all magnitudes.
using CorrelationId = int32_t;

// The value used to mean "no correlation ID". PyTorch's IDs start at 1
// (`id_start_ = 1 + ...`), so 0 is never a legitimate ID.
inline constexpr CorrelationId kNoCorrelationId = 0;

// TraceMe metadata keys written by the PJRT enqueue path and read back by the
// profiler's two-hop join. Producer and consumer must agree exactly, so they
// share these constants rather than repeating string literals.
inline constexpr std::string_view kLaunchCorrelationIdStat =
    "pt_launch_correlation_id";
inline constexpr std::string_view kLaunchCorrelationIdsStat =
    "pt_launch_correlation_ids";
inline constexpr std::string_view kLaunchOpCountStat = "pt_launch_op_count";

// Narrows a correlation ID as libkineto's own activity map does.
//
// Must use exactly the cast at kineto_shim.cpp `act.id =
// static_cast<int32_t>(correlation_id)` so that the two agree bit for bit,
// including for IDs above INT32_MAX where the result is negative.
[[nodiscard]] inline CorrelationId NarrowCorrelationId(
    uint64_t correlation_id) {
  return static_cast<CorrelationId>(correlation_id);
}

// ---------------------------------------------------------------------------
// Dispatch side: the correlation ID of the ATen op currently being dispatched.
// ---------------------------------------------------------------------------

// Pushes the correlation ID of an ATen op that has just begun executing.
// Called from the profiler session, which receives these IDs from libkineto.
//
// Takes uint64_t to match the libkineto virtual it is called from; the value is
// narrowed to CorrelationId on the way in.
void PushOpCorrelationId(uint64_t correlation_id);

// Pops the most recently pushed correlation ID. Safe to call when the stack is
// empty; libkineto does not guarantee that a session observes the push that
// matches a given pop, because a session is only added to `sessions_` when the
// profiler is configured and may therefore attach part-way through a nest.
void PopOpCorrelationId();

// Returns the correlation ID of the innermost ATen op currently executing on
// this thread, or kNoCorrelationId if there is none.
//
// The innermost op is the right choice: for a nest such as
// `aten::matmul` -> `aten::mm`, it is `aten::mm` that actually dispatches work
// to the device, and PyTorch's own event tree rolls its device time up into
// `aten::matmul` afterwards.
[[nodiscard]] CorrelationId GetCurrentOpCorrelationId();

// Clears this thread's dispatch-side stack. Called when a profiling session
// ends so that IDs from a finished session cannot leak into the next one.
void ResetOpCorrelationIds();

// Returns how many correlation IDs have been pushed process-wide since the last
// ResetOpCorrelationPushCount().
//
// This exists to make a specific silent failure loud. Our session only receives
// pushes if libkineto actually added it to `sessions_`, which happens only if
// `TpuProfiler::configure()` returned a session. If it declines -- wrong
// activity types, outside the profiling window -- every push is discarded and
// the only evidence is a "Not running child profiler." log line. A zero count
// at the end of a profiling session means correlation was never wired up, and
// the profiler should say so rather than silently emitting unlinked events.
[[nodiscard]] int64_t GetOpCorrelationPushCount();

// Resets the process-wide push counter. Called when a profiling session starts.
void ResetOpCorrelationPushCount();

// ---------------------------------------------------------------------------
// Execution side: the correlation IDs of the ops in the program about to run.
// ---------------------------------------------------------------------------

// Returns the correlation IDs of the ATen ops whose work is contained in the
// program currently being executed on this thread. Empty if not inside a
// `ScopedLaunchCorrelation`, which is the normal case when not profiling.
//
// This may contain more than one ID: several ATen ops can be fused into a
// single compiled program and therefore a single launch.
//
// Ordering matters and is easy to get wrong. This is *not* a dispatch stack.
// `ExecutionTask::ExtractCorrelationIds` fills it by walking
// `Traversal::execution_order()`, so it runs earliest-executed to
// latest-executed within the fused graph: `front()` is typically an allocation
// or input materialisation, and `back()` is the op that produces the result.
// Callers that want a single representative op for the launch want `back()`.
// The dispatch-nest accessor is `GetCurrentOpCorrelationId()` above, which is a
// different vector entirely.
[[nodiscard]] absl::Span<const CorrelationId> GetCurrentLaunchCorrelationIds();

// Installs `ids` as the current launch's correlation IDs for the lifetime of
// this object. Nesting is supported; the innermost scope wins.
class [[nodiscard]] ScopedLaunchCorrelation {
 public:
  // Installs `ids` into thread-local storage for this thread, saving any
  // previous scope IDs to be restored on destruction.
  explicit ScopedLaunchCorrelation(std::vector<CorrelationId> ids);

  // Restores the previous scope's correlation IDs upon destruction (RAII).
  ~ScopedLaunchCorrelation();

  ScopedLaunchCorrelation(const ScopedLaunchCorrelation&) = delete;
  ScopedLaunchCorrelation& operator=(const ScopedLaunchCorrelation&) = delete;
  ScopedLaunchCorrelation(ScopedLaunchCorrelation&&) = delete;
  ScopedLaunchCorrelation& operator=(ScopedLaunchCorrelation&&) = delete;

 private:
  std::vector<CorrelationId> previous_;
};

}  // namespace torch_tpu

#endif  // TORCH_TPU_CSRC_COMMON_OP_CORRELATION_H_
