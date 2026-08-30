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

"""Manual bounded dynamism API."""

from torch_tpu._internal.dynamism._tpu_torch_dynamism import get_dynamism_info
from torch_tpu._internal.dynamism._tpu_torch_dynamism import mark_dynamic

# PEP 8 requires this to be a list of strings, not a tuple or a list of objects.
__all__ = [
    # go/keep-sorted start
    "get_dynamism_info",
    "mark_dynamic",
    # go/keep-sorted end
]
