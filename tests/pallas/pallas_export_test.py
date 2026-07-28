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

"""Tests for Pallas CPU export helpers."""

import json
import os
import subprocess
import sys
import textwrap

from absl.testing import absltest


class PallasExportTest(absltest.TestCase):

  def _run_subprocess(self, script: str):
    env = os.environ.copy()
    env.pop("JAX_PLATFORMS", None)
    env.pop("XLA_FLAGS", None)
    completed = subprocess.run(
        [sys.executable, "-c", script],
        env=env,
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )
    self.assertEqual(
        completed.returncode,
        0,
        f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}",
    )
    return json.loads(completed.stdout.strip().splitlines()[-1])

  def test_export_module_does_not_import_jax(self):
    script = textwrap.dedent("""
        import json
        import sys
        import torch_tpu._internal.pallas.export as export

        export.configure_jax_cpu_export_env(4)
        print(json.dumps({
            "jax_imported": "jax" in sys.modules,
            "jax_platforms": export.os.environ.get("JAX_PLATFORMS"),
            "xla_flags": export.os.environ.get("XLA_FLAGS"),
        }))
    """)
    result = self._run_subprocess(script)

    self.assertFalse(result["jax_imported"])
    self.assertEqual(result["jax_platforms"], "cpu")
    self.assertIn(
        "--xla_force_host_platform_device_count=4", result["xla_flags"]
    )

  def test_cpu_export_mesh_uses_requested_logical_device_count(self):
    script = textwrap.dedent("""
        import json
        import torch_tpu._internal.pallas.export as export

        mesh = export.make_jax_cpu_export_mesh((1, 4), ("data", "model"))

        import jax
        from jax._src import xla_bridge as xb

        print(json.dumps({
            "backend_keys": sorted(xb._backends.keys()),
            "default_backend": xb._default_backend.platform,
            "device_ids": [device.id for device in mesh.devices.reshape(-1)],
            "mesh_shape": dict(mesh.shape),
            "mesh_platforms": sorted({device.platform for device in mesh.devices.reshape(-1)}),
        }))
    """)
    result = self._run_subprocess(script)

    self.assertEqual(result["backend_keys"], ["cpu"])
    self.assertEqual(result["default_backend"], "cpu")
    self.assertEqual(result["device_ids"], [0, 1, 2, 3])
    self.assertEqual(result["mesh_shape"], {"data": 1, "model": 4})
    self.assertEqual(result["mesh_platforms"], ["cpu"])


if __name__ == "__main__":
  absltest.main()
