# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""DeviceOps: the small set of primitives that differ per device and per framework.

Everything backend-specific is funnelled through this protocol. This is to make
sure that everything framework specific is handled by the backend and not by
the measuring utilities.
"""

from collections.abc import Callable
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class DeviceOps(Protocol):
  """Backend primitives required by `measure()`."""

  def await_result(self, out: Any) -> None:
    """Block until the work that produced "out" has actually completed.

    Why await_result takes the step's output:
      - torch: torch.cuda.synchronize() is a device-level barrier. It takes
        nothing and blocks on everything queued. The out argument is ignored.
      - torchax: jax.block_until_ready(out) is value-level. It blocks on the
        futures reachable from out and nothing else. This creates an obligation:
        a torchax step fn must return everything it produced, not just the loss.
        block_until_ready(loss) completes the loss future while the
        parameter and optimizer-state updates are still in flight. This causes
        the clock to stop early and the training step time is silently
        under-reported.
    A single zero-arg sync() cannot express both. Passing the step's output
    through, untouched, lets each backend do the right thing.

    Implementations may ignore out if they use a device-level barrier instead
    of value-level futures.

    await_result is explicitly defined instead of being
    embedded in step to allow flexibility in device time measurement using
    events API in future.
    """
    ...

  def check_result(self, out: Any) -> None:
    """Raise if the work that produced `out` failed.

    measure() calls this after `await_result` for every step, outside the
    timed region, so it may be slower than `await_result` without affecting
    the measurement.

    Backends with asynchronous execution may not report a failed execution
    (e.g. a device OOM) from their device-level barrier: TorchTPU's
    torch.accelerator.synchronize() returns and only marks the outputs of the
    failed execution as failed. Without this check such a step would be timed
    as if it had run. Backends whose `await_result` already raises on failure
    may implement this as a no-op.

    The step output must therefore contain the backend's device outputs (e.g.
    the logits or the loss); backends that check them may raise if it does
    not.
    """
    ...

  def reset_peak_memory(self) -> None:
    """Reset the peak-memory counter."""
    ...

  def release_cached_memory(self) -> None:
    """Releases device memory that the backend caches but doesn't use.

    measure() calls this after the pipelined steps run out of memory, so that
    the memory they left behind does not affect the steps that follow. Backends
    without such a cache may implement this as a no-op.
    """
    ...

  def agree_on_step_count(self, steps: int) -> int:
    """Returns the step count that all the processes of the benchmark run.

    The processes of a distributed benchmark must run the same number of steps,
    as each step runs collectives across them, so measure() agrees on every
    step count that it derives from the timings of this process. Backends that
    only run in a single process may return `steps`.

    Args:
      steps: The step count that this process proposes.
    """
    ...

  def peak_memory_mb(self) -> float | None:
    """Peak device memory in MB since the last reset, or None if unsupported.

    Returning None (rather than raising or reporting 0.0) lets callers
    record what a backend can report without special-casing the ones that
    cannot.
    """
    ...

  def get_device_time(self, session_id: str | None = None) -> float:
    """Total active device time in seconds across profiling steps, or -1.0 if unsupported.

    WARNING: prefer `get_device_time_from_xspace`. Looking the trace up by
    session id can return another benchmark's device time, because xprof may
    assign a single session id to several benchmark processes.
    """
    ...

  def get_device_time_from_xspace(
      self,
      xspace,
      expected_steps: int | None = None,
  ) -> float:
    """Total active device time in seconds from a locally captured XSpace.

    Args:
      xspace: The XSpace captured by this process's own xprof session.
      expected_steps: Number of profiled steps, used only to sanity check the
        trace. Events are never truncated to this count, because a single step
        can execute several XLA modules.

    Returns:
      Total device time in seconds, or -1.0 if unsupported/unavailable.
    """
    ...

  def supports_event_device_time(self) -> bool:
    """Returns whether this backend can measure per-step device time via device events."""
    ...

  def measure_step_device_time(self, step_fn: Callable[[], Any]) -> float:
    """Executes one step and returns its active device time in seconds, or -1.0."""
    ...

  def compile_count(self) -> int:
    """Monotonic count of compilations performed so far in this process.

    Compilation is a framework-specific notion (torch: Dynamo frame counters;
    torchax: jit cache misses). Keeping it behind the protocol is what lets
    measure() import neither torch nor jax. Backends with no notion of
    recompilation return a constant (0).
    """
    ...
