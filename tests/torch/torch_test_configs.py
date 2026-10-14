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

"""TorchTPU-specific configs for PyTorch device-generic tests.

For official documentation and guidance, see
https://github.com/pytorch/pytorch/blob/main/docs/source/accelerator/testing.md.
"""

from collections.abc import Mapping, Sequence
from typing import Final

# Generic test classes or methods excluded from instantiation for TPU.
#
# Keyed by test module, tests are identified by their generic class and method
# names, e.g. `TestOps` instead of `TestOpsTPU`. In this way, one entry covers
# all instantiated variants of a test. Use "*" to exclude a whole class.
TEST_EXCLUSIONS: Final[Mapping[str, Mapping[str, str | Sequence[str]]]] = {
    # TODO(b/573758862): track per-test exclusion reasons and triage.
    # go/keep-sorted start block=yes
    "distributed/fsdp/test_utils": {
        "TestUtils": ["test_apply_to_tensors"],
    },
    "dynamo/test_debug_utils": {
        "TestInductorConfigOverrideIntegration": [
            "test_config_override_backward_propagation",
        ],
    },
    "dynamo/test_fx_annotate": {
        "AnnotateTestsDevice": [
            "test_flex_attention_backward_tag_does_not_leak"
        ],
    },
    "dynamo/test_repros": {
        "ReproTestsDevice": [
            "test_custom_op_mutation_with_noop_add",
            "test_data_attr_mutation_with_noop_add",
            "test_layer_norm_mixed_dtype_aot_eager_decomp_partition_errors",
            "test_mem_leak_guards",
            "test_memleak_when_graph_input_has_tensor_attr",
            "test_tensor_set_data_cross_device",
            "test_tensor_set_data_cross_device_placeholder_metadata",
            "test_torch_device_is_initialized",
        ],
    },
    "inductor/test_slice_scatter_chunking": {
        "TestSliceScatterChunking": [
            "test_functionalized_copy_chain_compile",
            "test_nested_region_output_fusibility",
        ],
    },
    "test_custom_ops": {
        "TestCustomOpTesting": [
            "test_autograd_registration_check_autograd_kernel",
            "test_autograd_registration_check_compositeimplicitautograd",
            "test_autograd_registration_check_incorrect",
            "test_autograd_registration_check_incorrect_composite",
            "test_incorrect_schema_mutation",
        ],
    },
    "test_out_dtype_op": {
        "TestOutDtypeOpDevice": [
            "test_out_dtype_inductor_decomp",
            "test_out_dtype_inductor_decomp_trace",
        ],
    },
    # go/keep-sorted end
}
