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

"""model_registry: data-driven replacement for tests/module_registry.py.

Named `model_registry` during the migration so it does not shadow the legacy
`tests/module_registry.py` module.
"""

from tests.model_registry.types import Modality

__all__ = [
    "Modality",
]
