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


from absl.testing import absltest
import torch
from torch_tpu._internal import compile as compile_lib
from torch_tpu._internal.utils import test_utils
from tests import seed_test_utils


class TpuCompileDebugTest(seed_test_utils.RepeatableTest):

  def setUp(self):
    super().setUp()
    if not torch.accelerator.is_available():
      self.skipTest("TPU accelerator not available in this test environment.")
    torch._dynamo.reset()

  def test_tpu_compile_debug_dataclass_initialization(self):
    debug = compile_lib.TpuCompileDebug()
    self.assertEqual(debug.pre_autograd_fx_code, [])
    self.assertEqual(debug.pre_autograd_fx_readable, [])
    self.assertEqual(debug.post_autograd_fx_forward_code, [])
    self.assertEqual(debug.post_autograd_fx_forward_readable, [])
    self.assertEqual(debug.post_autograd_fx_backward_code, [])
    self.assertEqual(debug.post_autograd_fx_backward_readable, [])
    self.assertEqual(debug.stablehlo_forward_text, [])
    self.assertEqual(debug.stablehlo_backward_text, [])
    self.assertEqual(debug.compiled_executables, [])

  def test_torch_tpu_debug_callback_option(self):
    def simple_fn(x, y):
      return (x + y) * 2.0

    device = torch.device("tpu")
    x = torch.tensor([1.0, 2.0, 3.0], device=device, requires_grad=True)
    y = torch.tensor([4.0, 5.0, 6.0], device=device, requires_grad=True)

    debugs = []
    compiled = torch.compile(
        simple_fn,
        backend="tpu",
        options={"debug_callback": debugs.append},
    )

    out = compiled(x, y)
    loss = out.sum()
    loss.backward()

    self.assertLen(debugs, 1)
    debug = debugs[0]
    self.assertIsInstance(debug, compile_lib.TpuCompileDebug)
    self.assertLen(debug.pre_autograd_fx_code, 1)
    self.assertLen(debug.pre_autograd_fx_readable, 1)
    self.assertLen(debug.post_autograd_fx_forward_code, 1)
    self.assertLen(debug.post_autograd_fx_forward_readable, 1)
    self.assertLen(debug.post_autograd_fx_backward_code, 1)
    self.assertLen(debug.post_autograd_fx_backward_readable, 1)
    self.assertLen(debug.stablehlo_forward_text, 1)
    self.assertLen(debug.stablehlo_backward_text, 1)
    self.assertLen(debug.compiled_executables, 2)

    self.assertIn("torch.ops.aten.add", debug.post_autograd_fx_forward_code[0])
    self.assertIn("stablehlo.add", debug.stablehlo_forward_text[0])

  def test_user_provided_debug_callback(self):
    def mul_fn(x):
      return x * 3.0

    device = torch.device("tpu")
    x = torch.tensor([2.0, 4.0], device=device)

    debugs = []
    compiled = torch.compile(
        mul_fn, backend="tpu", options={"debug_callback": debugs.append}
    )

    out = compiled(x)
    test_utils.assert_close(out.cpu(), torch.tensor([6.0, 12.0]))

    debug = debugs[0]
    self.assertLen(debug.pre_autograd_fx_code, 1)
    self.assertLen(debug.post_autograd_fx_forward_code, 1)
    self.assertLen(debug.stablehlo_forward_text, 1)
    self.assertLen(debug.compiled_executables, 1)
    self.assertIn("stablehlo.multiply", debug.stablehlo_forward_text[0])

  def test_serializable_disable_option(self):
    def fn(x):
      return x + 1.0

    device = torch.device("tpu")
    x = torch.tensor([1.0, 2.0], device=device)

    debugs = []
    compiled = torch.compile(
        fn,
        backend="tpu",
        options={
            "serializable": False,
            "debug_callback": debugs.append,
        },
    )

    # First execution
    _ = compiled(x)

    debug = debugs[0]
    self.assertLen(debug.compiled_executables, 1)

  def test_debug_str_and_repr(self):
    debug = compile_lib.TpuCompileDebug()
    self.assertEqual(str(debug), "TpuCompileDebug(empty)")
    self.assertIn("pre_fx=0", repr(debug))
    self.assertIn("executables=0", repr(debug))

    debug.pre_autograd_fx_code.append("def forward(self, x): return x")
    debug.stablehlo_forward_text.append("module { func.func @main() }")

    debug_str = str(debug)
    self.assertIn("=== Pre-Autograd FX Graph ===", debug_str)
    self.assertIn("def forward(self, x): return x", debug_str)
    self.assertIn("=== StableHLO Forward ===", debug_str)

    debug_repr = repr(debug)
    self.assertIn("pre_fx=1", debug_repr)
    self.assertIn("stablehlo_fwd=1", debug_repr)
    self.assertIn("executables=0", debug_repr)

  def test_compiler_init_accepts_debug_container(self):
    debug = compile_lib.TpuCompileDebug()
    static_comp = compile_lib.compiler.StaticCompiler(debug=debug)
    self.assertIs(static_comp._debug, debug)

    dyn_comp = compile_lib.dynamic_compiler.DynamicCompiler(debug=debug)
    self.assertIs(dyn_comp._debug, debug)
    self.assertIs(dyn_comp.static_compiler._debug, debug)

    split_comp = compile_lib.split_compiler.SplitCompiler(static_comp)
    self.assertIs(split_comp._debug, debug)


if __name__ == "__main__":
  absltest.main()
