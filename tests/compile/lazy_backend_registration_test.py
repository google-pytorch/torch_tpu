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

"""Regression tests for lazy Dynamo backend registration on TPU initialization.
"""

import os
import subprocess
import sys

from absl.testing import absltest


class LazyBackendRegistrationTest(absltest.TestCase):

  def test_simulated_tpu_init_defers_compile_backend(self) -> None:
    """Verifies that TPU device init does not eagerly load compiler modules."""
    script = (
        "import sys\n"
        "from torch.nn.attention import flex_attention\n"
        "orig_validate = flex_attention._validate_device\n"
        "from torch_tpu import _loader\n"
        "_loader._init_device('tpu')\n"
        "\n"
        "assert 'torch_tpu._loader' in sys.modules, (\n"
        "    'torch_tpu._loader not loaded'\n"
        ")\n"
        "assert 'torch._dynamo' not in sys.modules, (\n"
        "    'torch._dynamo unexpectedly loaded'\n"
        ")\n"
        "assert 'torch_tpu._internal.compile._backend' not in sys.modules, (\n"
        "    '_backend unexpectedly loaded'\n"
        ")\n"
        "assert 'torch_tpu._internal.compile.compiler' not in sys.modules, (\n"
        "    'compiler unexpectedly loaded'\n"
        ")\n"
        "\n"
        "assert flex_attention._validate_device is not orig_validate, (\n"
        "    'flex_attention._validate_device not patched'\n"
        ")\n"
        "\n"
        "from torch._dynamo.backends import registry\n"
        "backend = registry.lookup_backend('tpu')\n"
        "from torch_tpu._internal.compile import _backend\n"
        "assert isinstance(backend, _backend.TpuBackend), (\n"
        "    f'Expected TpuBackend, got {type(backend)}'\n"
        ")\n"
        "\n"
        "from torch._dynamo.device_interface import get_interface_for_device\n"
        "from torch_tpu._internal.device import _device_interface\n"
        "from torch_tpu._internal.device import _device_module\n"
        "iface = get_interface_for_device('tpu')\n"
        "assert iface is _device_interface.TpuDeviceInterface, (\n"
        "    f'Expected TpuDeviceInterface, got {iface}'\n"
        ")\n"
        "assert issubclass(iface, _device_module.TpuDeviceModule), (\n"
        "    f'Expected subclass of TpuDeviceModule, got {iface}'\n"
        ")\n"
        "\n"
        "import torch_tpu._internal.compile as compile_pkg\n"
        "compile_pkg.ensure_initialized()\n"
        "print('PASS')\n"
    )

    env = dict(os.environ, TORCH_DEVICE_BACKEND_AUTOLOAD="0")
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    self.assertEqual(
        result.returncode,
        0,
        f"Subprocess test failed:\nstdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}",
    )
    self.assertIn("PASS", result.stdout)

  def test_simulated_tpu_init_eager_hop_registrations(self) -> None:
    """Verifies eager HOP registrations on TPU init without loading compile."""
    script = (
        "import sys\n"
        "import torch\n"
        "import torch._C\n"
        "from torch._higher_order_ops import associative_scan\n"
        "from torch_tpu import _loader\n"
        "_loader._init_device('tpu')\n"
        "\n"
        "assert 'torch_tpu._loader' in sys.modules, (\n"
        "    'torch_tpu._loader not loaded'\n"
        ")\n"
        "assert torch._C.DispatchKey.PrivateUse1 in (\n"
        "    torch.ops.higher_order.scan.py_kernels\n"
        "), 'scan PrivateUse1 impl missing'\n"
        "assert torch._C.DispatchKey.PrivateUse1 in (\n"
        "    torch.ops.higher_order.invoke_subgraph.py_kernels\n"
        "), 'invoke_subgraph PrivateUse1 impl missing'\n"
        "assert getattr(associative_scan, '_tpu_patched', False), (\n"
        "    'associative_scan not patched'\n"
        ")\n"
        "assert 'torch_tpu._internal.compile.scan' not in sys.modules, (\n"
        "    'compile.scan unexpectedly loaded'\n"
        ")\n"
        "assert (\n"
        "    'torch_tpu._internal.compile.nested_compile_region' not in"
        " sys.modules\n"
        "), 'nested_compile_region unexpectedly loaded'\n"
        "assert 'torch_tpu._internal.compile._backend' not in sys.modules, (\n"
        "    '_backend unexpectedly loaded'\n"
        ")\n"
        "assert 'torch._dynamo' not in sys.modules, (\n"
        "    'torch._dynamo unexpectedly loaded'\n"
        ")\n"
        "print('PASS')\n"
    )

    env = dict(os.environ, TORCH_DEVICE_BACKEND_AUTOLOAD="0")
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    self.assertEqual(
        result.returncode,
        0,
        f"Subprocess test failed:\nstdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}",
    )
    self.assertIn("PASS", result.stdout)

  def test_no_entry_point_fallback_registration(self) -> None:
    """Verifies fallback registration via _default_tpu_compile."""
    script = (
        "import sys\n"
        "from torch_tpu import _loader\n"
        "_loader._init_device('tpu')\n"
        "assert 'torch_tpu._loader' in sys.modules, (\n"
        "    'torch_tpu._loader not loaded'\n"
        ")\n"
        "from torch._dynamo.backends import registry\n"
        "assert 'tpu' not in registry._COMPILER_FNS, (\n"
        "    'tpu unexpectedly registered before compile'\n"
        ")\n"
        "_loader._default_tpu_compile(lambda x: x, backend='tpu')\n"
        "assert 'tpu' in registry._COMPILER_FNS, (\n"
        "    'tpu not registered in _COMPILER_FNS after default compile'\n"
        ")\n"
        "from torch_tpu._internal.compile import _backend\n"
        "backend = registry._COMPILER_FNS['tpu']\n"
        "assert isinstance(backend, _backend.TpuBackend), (\n"
        "    f'Expected TpuBackend, got {type(backend)}'\n"
        ")\n"
        "print('PASS')\n"
    )

    env = dict(os.environ, TORCH_DEVICE_BACKEND_AUTOLOAD="0")
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    self.assertEqual(
        result.returncode,
        0,
        f"Subprocess test failed:\nstdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}",
    )
    self.assertIn("PASS", result.stdout)


if __name__ == "__main__":
  absltest.main()
