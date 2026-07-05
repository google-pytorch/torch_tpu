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

"""TorchInductor backend for TPU that lowers fused kernels to Pallas.

Importing this package registers the ``tpu_inductor_pallas`` backend for
`torch.compile()`. See `pallas_backend.py` for the design.
"""

from torch_tpu._internal.inductor.pallas_backend import tpu_inductor_pallas

__all__ = ["tpu_inductor_pallas"]
