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

"""CompileConfig specifies how to run torch.compile() on the benchmark.

This is a declarative modifier on the benchmark spec, applied by the harness.
Supports two compilation scopes:

  scope="model"
      torch.compile(model). Compiles the forward pass. For training benchmarks,
      the backward pass is compiled via AOTAutograd. The optimizer step stays
      eager.

  scope="step"
      Wraps the whole forward (+ backward + optimizer.step() for training)
      region in one torch.compile.
"""

import dataclasses
from typing import Any, Callable
import torch
from perflab import target as target_lib
from perflab.steps import common

Scope = common.Scope


@dataclasses.dataclass(frozen=True)
class CompileConfig:
  """Settings for the compiled mode. Defaults to step-scope"""

  scope: Scope = Scope.STEP
  dynamic: bool | None = None
  fullgraph: bool = False

  def apply(self, obj: Any, target: target_lib.Target) -> Callable:
    """Compile `obj` based on the target.

    Picks a backend based on the target, and then compiles the object.

    Args:
      obj: The object to compile, either an nn.Module (Scope.MODEL) or the step
        callable (Scope.STEP).
      target: The target to compile for, used to pick the backend.

    Returns:
      The compiled object.
    """
    backend = "inductor"
    if target.device_kind is target_lib.DeviceKind.TPU:
      from torch_tpu._internal import compile as torch_tpu_compile  # pylint: disable=g-import-not-at-top

      backend = torch_tpu_compile.TpuBackend(dynamism=self.dynamic is not False)
    else:
      from perflab import device_utils  # pylint: disable=g-import-not-at-top

      device_utils.configure_inductor_for_benchmarks()

    return torch.compile(
        obj,
        backend=backend,
        dynamic=self.dynamic,
        fullgraph=self.fullgraph,
    )
