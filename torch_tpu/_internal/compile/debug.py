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

"""Log container for TorchTPU compilation artifacts."""

import dataclasses
from typing import Any


@dataclasses.dataclass
class TpuCompileDebug:
  """Log container for TorchTPU compilation artifacts."""

  # 1. Dynamo FX Graph (Pre-AOTAutograd)
  pre_autograd_fx_code: list[str] = dataclasses.field(default_factory=list)
  pre_autograd_fx_readable: list[str] = dataclasses.field(default_factory=list)
  # 2. AOTAutograd Forward FX Graph
  post_autograd_fx_forward_code: list[str] = dataclasses.field(
      default_factory=list
  )
  post_autograd_fx_forward_readable: list[str] = dataclasses.field(
      default_factory=list
  )
  # 3. AOTAutograd Backward FX Graph (empty for inference / trace_autograd_ops=True)
  post_autograd_fx_backward_code: list[str] = dataclasses.field(
      default_factory=list
  )
  post_autograd_fx_backward_readable: list[str] = dataclasses.field(
      default_factory=list
  )
  # 4. StableHLO IR (Note: hlo_text and llo_text may be added in the future)
  stablehlo_forward_text: list[str] = dataclasses.field(default_factory=list)
  stablehlo_backward_text: list[str] = dataclasses.field(default_factory=list)
  # 5. Compiled executables
  compiled_executables: list[Any] = dataclasses.field(default_factory=list)

  def __str__(self) -> str:
    sections = []
    if self.pre_autograd_fx_code:
      sections.append(
          "=== Pre-Autograd FX Graph ===\n"
          + "\n".join(self.pre_autograd_fx_code).strip()
      )
    if self.post_autograd_fx_forward_code:
      sections.append(
          "=== Post-Autograd Forward FX Graph ===\n"
          + "\n".join(self.post_autograd_fx_forward_code).strip()
      )
    if self.post_autograd_fx_backward_code:
      sections.append(
          "=== Post-Autograd Backward FX Graph ===\n"
          + "\n".join(self.post_autograd_fx_backward_code).strip()
      )
    if self.stablehlo_forward_text:
      sections.append(
          "=== StableHLO Forward ===\n"
          + "\n".join(self.stablehlo_forward_text).strip()
      )
    if self.stablehlo_backward_text:
      sections.append(
          "=== StableHLO Backward ===\n"
          + "\n".join(self.stablehlo_backward_text).strip()
      )

    return "\n\n".join(sections) if sections else "TpuCompileDebug(empty)"

  def __repr__(self) -> str:
    return (
        "TpuCompileDebug("
        f"pre_fx={len(self.pre_autograd_fx_code)}, "
        f"post_fwd_fx={len(self.post_autograd_fx_forward_code)}, "
        f"post_bwd_fx={len(self.post_autograd_fx_backward_code)}, "
        f"stablehlo_fwd={len(self.stablehlo_forward_text)}, "
        f"stablehlo_bwd={len(self.stablehlo_backward_text)}, "
        f"executables={len(self.compiled_executables)})"
    )
