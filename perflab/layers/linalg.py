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
"""Layer benchmarks for LU linear algebra operations."""

from perflab.layer_utils import builders
from perflab.layer_utils import configs
from perflab.layer_utils import registration

registration.add_layer_sweep(
    benchmark="lu_unpack",
    configs=configs.LU_UNPACK_CONFIGS,
    to_layer_args=lambda config: builders.LayerArgs(
        model_name="lu_unpack",
        batch_size=config.batch_size,
        custom_kwargs={
            "matrix_size": config.matrix_size,
            "unpack_data": config.unpack_data,
            "unpack_pivots": config.unpack_pivots,
        },
    ),
    steps=frozenset({registration.Step.EVAL}),
)

registration.add_layer_sweep(
    benchmark="linalg_lu_solve",
    configs=configs.LINALG_LU_SOLVE_CONFIGS,
    to_layer_args=lambda config: builders.LayerArgs(
        model_name="linalg_lu_solve",
        batch_size=config.batch_size,
        custom_kwargs={
            "matrix_size": config.matrix_size,
            "nrhs": config.nrhs,
            "left": config.left,
            "adjoint": config.adjoint,
        },
    ),
    steps=frozenset({registration.Step.EVAL}),
)
