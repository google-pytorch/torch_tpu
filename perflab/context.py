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

"""Context captures the information available to a benchmark for a run.

Everything here describes the run, chosen per invocation, as opposed to
the benchmark, which is declared on the spec.
"""

import dataclasses
import enum

from absl import flags
from perflab import target as target_lib
from perflab import workload as workload_lib


class RunScope(enum.Enum):
  """Execution scopes.

  This allows us to run the same benchmark with different configurations.
  """

  FULL = "full"  # Real measurement.
  # Fast smoke test to prevent benchmarks from breaking (e.g., failing to
  # compile).
  PRESUBMIT = "presubmit"


RUN_SCOPE = flags.DEFINE_enum_class(  # PYTHON_FLAG_OK=benchmark
    "run_scope",
    RunScope.FULL,
    RunScope,
    "The execution scope of the benchmark run (full vs. presubmit).",
)


@dataclasses.dataclass(frozen=True)
class Context:
  """Context needed to build a benchmark workload.

  Context deliberately excludes the execution mode and framework to prevent
  the factory from building different models for those parameters.

  Attributes:
    target: The hardware target descriptor for the run.
    run_scope: The execution scope (e.g., FULL or PRESUBMIT).
    workload: The workload to build the benchmark at (see
      `BenchmarkSpec.workload`), or None for a benchmark that declares none. The
      harness resolves it for each case; factories read it with
      `require_workload`.
  """

  target: target_lib.Target
  run_scope: RunScope
  workload: workload_lib.Workload | None = None

  def require_workload(self) -> workload_lib.Workload:
    """Returns the workload to build the benchmark at.

    Raises:
      ValueError: If the context has no workload, i.e. it was not resolved for
        a benchmark that declares one.
    """
    if self.workload is None:
      raise ValueError(
          "Context has no workload; build benchmarks with a context from"
          " `registry.resolve_context`."
      )
    return self.workload

  @property
  def device_kind(self) -> target_lib.DeviceKind:
    """Returns the device kind of the target."""
    return self.target.device_kind

  @property
  def dtype(self) -> target_lib.DType:
    """Returns the data type of the target."""
    return self.target.dtype
