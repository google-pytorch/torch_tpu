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

"""Ensure we error out at torch.compile() when "dynamic" is applyed"""

import os
from typing import Callable, List, Tuple

from absl import logging
from absl.testing import absltest
import torch
from torch._dynamo.backends.common import aot_autograd
from torch_tpu import api
from torch_tpu._internal import compile
from torch_tpu._internal.utils import utils


def _run_tpu_backend_with_injected_test_case(
    tpu_backend: compile.TpuBackend,
    inject_test_case: Callable[
        [torch.fx.GraphModule, List[torch.Tensor]],
        Tuple[torch.fx.GraphModule, List[torch.Tensor]],
    ],
    map_output: Callable[List[torch.Tensor], List[torch.Tensor]],
):
  """Wraps a TpuBackend to inject graph/output modifications for testing.

  This function returns a backend function compatible with `torch.compile`. It
  intercepts the compilation process to inject modification of the FX graph
  via `inject_test_case` before it is compiled by `tpu_backend`. It also
  wraps the resulting executable to modify its output via `map_output` before
  returning the results to the caller, so that the code runs correctly. This is
  useful for testing backend
  behavior with graph structures or outputs that are difficult to produce
  from Python source, such as outputs containing `None`.

  Args:
    tpu_backend: The `TpuBackend` instance to use for compilation.
    inject_test_case: A callable `fn(gm, example_inputs)` that modifies the FX
      graph or example inputs and returns `(new_gm, new_example_inputs)`.
    map_output: A callable `fn(results)` that transforms the output of the
      compiled executable before it is returned.

  Returns:
    A backend function compatible with `torch.compile`.
  """

  def _inner(gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]):
    logging.info("before injecting test case %s", gm.code)
    # 1. Inject test-specific modifications to the graph and/or inputs.
    # Modify the graph and example inputs to reproduce a test scenario.
    gm, example_inputs = inject_test_case(gm, example_inputs)
    # 2. Compile the graph using the TPU backend.
    # If inject_test_case modified outputs to include None, the backend handles
    # this via allow_graph_modification=True, and executable() will return
    # outputs including None.
    logging.info("after injecting test case %s", gm.code)
    # Use the modified graph to test our code
    executable = tpu_backend._compile_graph_module(gm, example_inputs)

    # 3. If needed, transform the results from executable() before returning.
    # For example, if inject_test_case added a None to the output which
    # the backend preserved, map_output can remove it to match the
    # return signature expected by torch.compile for the original function.
    def wrapped_executable(*args):
      results = executable(list(args))
      # inject_test_case can modify the output of the executable. To get the code running
      # We need to apply the counterpart before returning them.
      results = map_output(results)
      return results

    return wrapped_executable

  return _inner


class OutputModifyTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    os.environ["TORCHDYNAMO_VERBOSE"] = "1"
    os.environ["TORCH_LOGS"] = "+dynamo"
    torch.compiler.reset()

  def test_none_in_graph_output(self):
    def _my_function(x, y):
      a = x + y
      return a

    # Note that _add_none_to_output must be paired with _remove_none_from_output
    def _add_none_to_output(
        gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]
    ) -> Tuple[torch.fx.GraphModule, List[torch.Tensor]]:
      for node in reversed(gm.graph.nodes):
        if node.op == "output":
          # Add a None to the beginning of the output
          new_args = [None] + list(node.args[0])
          node.args = (tuple(new_args),)
          break

      gm.recompile()
      return gm, example_inputs

    def _remove_none_from_output(
        results: List[torch.Tensor],
    ) -> List[torch.Tensor]:
      # Ignore the None we added
      self.assertIsNone(results[0])
      return results[1]

    in_a = torch.randn(5).to(api.tpu_device())
    in_b = torch.randn(5).to(api.tpu_device())

    tpu_backend = compile.TpuBackend(debug=True)
    compiled = torch.compile(
        _my_function,
        backend=_run_tpu_backend_with_injected_test_case(
            tpu_backend,
            inject_test_case=_add_none_to_output,
            map_output=_remove_none_from_output,
        ),
    )
    # This should not raise an error.
    result = compiled(in_a, in_b)
    result.to("cpu")

  def test_duplicate_in_graph_output(self):
    def _my_function(x, y):
      a = x + y
      b = x - y
      return a, b, a

    def _assert_no_duplicate_output(
        results: List[torch.Tensor],
    ) -> List[torch.Tensor]:
      # The third output should be dropped internally.
      self.assertLen(results, 2)
      return results

    in_a = torch.randn(5).to(api.tpu_device())
    in_b = torch.randn(5).to(api.tpu_device())
    tpu_backend = compile.TpuBackend(debug=True)
    compiled = torch.compile(
        _my_function,
        backend=_run_tpu_backend_with_injected_test_case(
            tpu_backend,
            inject_test_case=lambda gm, example_inputs: (gm, example_inputs),
            map_output=_assert_no_duplicate_output,
        ),
    )
    results = compiled(in_a, in_b)
    # After the compiled function returns, the duplicate output should be
    # restored.
    self.assertLen(results, 3)
    utils.assert_close(actual=results[0], expected=results[2])


if __name__ == "__main__":
  absltest.main()
