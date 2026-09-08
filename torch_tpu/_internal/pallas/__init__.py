# Copyright 2025 Google LLC
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
"""Pallas API for Torch-TPU."""

import importlib


def __getattr__(name):
  if name in ("_compat", "export", "pallas", "tpu_torch_pallas"):
    return importlib.import_module(f"{__name__}.{name}")
  if name in ("custom_jax_kernel", "jax_op"):
    pallas = importlib.import_module(f"{__name__}.pallas")

    return getattr(pallas, name)
  raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# PEP 8 requires this to be a list of strings, not a tuple or a list of objects.
__all__ = [
    # go/keep-sorted start
    "custom_jax_kernel",
    "jax_op",
    "pallas",
    # go/keep-sorted end
]
