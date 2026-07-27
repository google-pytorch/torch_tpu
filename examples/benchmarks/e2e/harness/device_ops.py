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

  def reset_peak_memory(self) -> None:
    """Reset the peak-memory counter."""
    ...

  def peak_memory_mb(self) -> float | None:
    """Peak device memory in MB since the last reset, or None if unsupported.

    Returning None (rather than raising or reporting 0.0) lets callers
    record what a backend can report without special-casing the ones that
    cannot.
    """
    ...

  def compile_count(self) -> int:
    """Monotonic count of compilations performed so far in this process.

    Compilation is a framework-specific notion (torch: Dynamo frame counters;
    torchax: jit cache misses). Keeping it behind the protocol is what lets
    measure() import neither torch nor jax. Backends with no notion of
    recompilation return a constant (0).
    """
    ...
