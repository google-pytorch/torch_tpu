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

"""Internal optimizer module."""

from torch_tpu._internal.optim import adamw
from torch_tpu._internal.optim import patch

patch_optimizer_graph_breaks = patch.patch_optimizer_graph_breaks
unpatch_optimizer_graph_breaks = patch.unpatch_optimizer_graph_breaks
AdamW = adamw.AdamW


__all__ = [
    "patch_optimizer_graph_breaks",
    "unpatch_optimizer_graph_breaks",
    "AdamW",
]
