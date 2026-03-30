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

import os
import pickle

from absl.testing import absltest
import torch
import torch.fx as fx
from torch_tpu import api
from torch_tpu._internal.compile import _backend


from torch._functorch._aot_autograd.aot_autograd_result import (
    deserialize_bundled_cache_entry,
)

from torch._dynamo.functional_export import dynamo_graph_capture_for_export


class BackendSerializationTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    os.environ["TORCHDYNAMO_VERBOSE"] = "1"
  
  def check_serialization(self, f, inputs):
    inputs_tpu = _backend.to_device(inputs, api.tpu_device())
    gm = dynamo_graph_capture_for_export(f)(*inputs_tpu)

    backend = _backend.TpuBackend(
        debug=True, enable_serialization=True,
    )
    compiled_fn = backend(gm, inputs_tpu)

    # Verify execution produces correct results.
    result = compiled_fn(*inputs_tpu)
    result = _backend.to_device(result, "cpu")
    torch.testing.assert_close(result[0], inputs[0] + inputs[1])
    torch.testing.assert_close(result[1], inputs[0] * inputs[1])

    # Round-trip through pickle (like TpuCompilerAdaptor would do).
    entry = compiled_fn.serialize()
    self.assertIsNotNone(entry)
    pickled = pickle.dumps(entry)
    restored_entry = pickle.loads(pickled)

    # Reconstruct a callable from the deserialized entry.
    restored_fn = deserialize_bundled_cache_entry(restored_entry)
    self.assertTrue(callable(restored_fn))

    # Verify the restored callable produces the same results.
    # deserialize_bundled_cache_entry returns a function with boxed
    # calling convention (takes a list of args).
    restored_result = restored_fn(*inputs_tpu)
    restored_result = _backend.to_device(restored_result, "cpu")
    torch.testing.assert_close(restored_result[0], inputs[0] + inputs[1])
    torch.testing.assert_close(restored_result[1], inputs[0] * inputs[1])

  def test_simple(self):
    def simple(x, y):
      return x + y, x * y

    inputs = [torch.randn(4, 4), torch.randn(4, 4)]
    self.check_serialization(simple, inputs)

  def test_int_input(self):
    def simple(x, y):
      return x + y, x * y

    inputs = [torch.randn(4, 4), 4]
    self.check_serialization(simple, inputs)

if __name__ == "__main__":
  absltest.main()
