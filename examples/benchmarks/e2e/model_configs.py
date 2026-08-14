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

"""Shared model configurations for benchmarks."""

import dataclasses


@dataclasses.dataclass
class ModelConfig:
  batch_size: int
  seq_len: int


GEMMA_4_TRAIN_CONFIGS = (
    ModelConfig(
        batch_size=4,
        seq_len=1024,
    ),
    ModelConfig(
        batch_size=1,
        seq_len=512,
    ),
)
