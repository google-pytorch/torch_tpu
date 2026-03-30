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

"""SPMD Context Manager for fat collectives."""

from torch_tpu._internal import spmd_context_impl


class SPMDContext:
  """Context manager for SPMD regions.

  Anything inside this context will be treated as its own graph, and fat
  collectives will be allowed.
  """

  def __enter__(self):
    spmd_context_impl.enter_spmd_context()
    return self

  def __exit__(self, exc_type, exc_val, exc_tb):
    spmd_context_impl.exit_spmd_context()
    return False
