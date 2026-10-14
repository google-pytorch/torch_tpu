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

"""Unit tests for public TorchTPU API exposure and registration."""

import importlib
import importlib.metadata
import runpy
import sys
import types
import warnings

from absl.testing import absltest
from packaging import version as packaging_version
import torch
from torch import torch_version
import torch_tpu
from torch_tpu import _loader  # build_cleaner: keep
from torch_tpu._internal import compiler_options as internal_compiler_options
from torch_tpu._internal import execution_mode as _internal_execution_mode
from torch_tpu._internal.device_utils import annotations as tpu_annotations
from torch_tpu._internal.distributed.distributed import (
    set_tpu_launch_env as internal_set_tpu_launch_env,
)
from torch_tpu._internal.ops import ops as internal_ops
from torch_tpu._internal.pallas.pallas import (
    custom_jax_kernel as internal_custom_jax_kernel,
    jax_op as internal_jax_op,
)
from torch_tpu._internal.profiler.profiler_config import (
    TpuProfilerConfig as InternalTpuProfilerConfig,
)
from torch_tpu._internal.utils import annotations
from tests import seed_test_utils


class ApiRegistryTest(seed_test_utils.RepeatableTest):

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    _loader._init_device("tpu")  # INIT_DEVICE_OK=CPU runner

  def test_torch_tpu_module_identity_and_dir(self):
    """Verifies torch.tpu is a real ModuleType with expected submodules and dir()."""
    self.assertIsInstance(torch.tpu, types.ModuleType)
    self.assertEqual(torch.tpu.__name__, "torch.tpu")
    self.assertIs(sys.modules["torch.tpu"], torch.tpu)
    self.assertIs(torch.get_device_module("tpu"), torch.tpu)

    tpu_dir = dir(torch.tpu)
    self.assertEmpty([name for name in tpu_dir if name.startswith("_")])

    for bad_attr in ("non_existent_attribute", "__mro__"):
      with self.assertRaises(  # ASSERT_RAISES_OK=Verifies torch.tpu raises AttributeError for unknown/dunder attributes
          AttributeError
      ):
        _ = getattr(torch.tpu, bad_attr)

  def test_version(self):
    """Verifies torch.tpu.__version__ is the installed torch_tpu version."""
    with warnings.catch_warnings(record=True) as caught:
      warnings.simplefilter("always")
      version = torch.tpu.__version__
    # A stable API doesn't warn.
    self.assertEmpty(caught)

    # Like `torch.__version__`, it's a `TorchVersion`.
    self.assertIsInstance(version, torch_version.TorchVersion)
    self.assertEqual(version, importlib.metadata.version("torch_tpu"))
    self.assertGreaterEqual(version, "0.1")
    self.assertEqual(
        packaging_version.Version(version),
        packaging_version.Version(importlib.metadata.version("torch_tpu")),
    )
    # Like other dunder attributes, it isn't listed.
    self.assertNotIn("__version__", torch.tpu.__all__)
    self.assertNotIn("__version__", dir(torch.tpu))

  def test_tpu_profiler_config_accessibility(self):
    """Verifies TpuProfilerConfig is accessible via torch.tpu.profiler."""
    self.assertTrue(hasattr(torch, "tpu"))
    self.assertTrue(hasattr(torch.tpu, "profiler"))
    self.assertTrue(hasattr(torch.tpu.profiler, "TpuProfilerConfig"))
    self.assertFalse(hasattr(torch_tpu, "profiler"))
    self.assertNotIn("torch_tpu.profiler", sys.modules)

  def test_tpu_profiler_config_identity(self):
    """Verifies identity between public export and internal class definition."""
    self.assertIs(
        torch.tpu.profiler.TpuProfilerConfig, InternalTpuProfilerConfig
    )

  def test_invalid_attribute_raises(self):
    """Verifies accessing non-existent attributes on proxy module raises AttributeError."""
    with self.assertRaises(  # ASSERT_RAISES_OK=Verifies proxy module raises AttributeError for non-existent attributes
        AttributeError
    ):
      _ = getattr(torch.tpu.profiler, "non_existent_attribute")

  def test_tpu_profiler_config_metadata(self):
    """Verifies lifecycle stage metadata attached to TpuProfilerConfig."""
    cls = torch.tpu.profiler.TpuProfilerConfig
    self.assertEqual(
        getattr(cls, annotations.TT_API_STAGE, None),
        annotations.Stage.EXPERIMENTAL.value,
    )
    self.assertEqual(
        getattr(cls, annotations.TT_API_STAGE_REASON, None),
        "torch.tpu.profiler.TpuProfilerConfig is experimental and subject to"
        " change.",
    )

  def test_tpu_profiler_config_instantiation_warning(self):
    """Verifies instantiating TpuProfilerConfig triggers UserWarning."""
    with warnings.catch_warnings(record=True) as w:
      warnings.simplefilter("always")
      torch.tpu.profiler.TpuProfilerConfig()
      self.assertNotEmpty(w)
      experimental_warnings = [
          item for item in w if issubclass(item.category, UserWarning)
      ]
      self.assertLen(experimental_warnings, 1)
      self.assertIn(
          "TpuProfilerConfig is experimental",
          str(experimental_warnings[0].message),
      )

  def test_profiler_module_dir(self):
    """Verifies dir(torch.tpu.profiler) returns public symbol list."""
    self.assertEqual(dir(torch.tpu.profiler), ["TpuProfilerConfig"])

  def test_pallas_api_registry(self):
    """Verifies public exposure, identity, metadata, and proxy behavior for pallas."""
    # 1. Symbol identity
    self.assertIs(torch.tpu.pallas.jax_op, internal_jax_op)

    # 2. Lifecycle stage metadata
    fn = torch.tpu.pallas.jax_op
    self.assertEqual(
        getattr(fn, annotations.TT_API_STAGE, None),
        annotations.Stage.EXPERIMENTAL.value,
    )

    # 3. Proxy dir and invalid attribute handling
    self.assertEqual(dir(torch.tpu.pallas), ["get_pallas_mesh", "jax_op"])
    with self.assertRaises(  # ASSERT_RAISES_OK=Verifies proxy module raises AttributeError for non-existent attributes
        AttributeError
    ):
      _ = getattr(torch.tpu.pallas, "non_existent_attribute")

    # 4. Verify torch_tpu does not expose pallas
    self.assertFalse(hasattr(torch_tpu, "pallas"))
    self.assertNotIn("torch_tpu.pallas", sys.modules)

    # 5. Verify torch_tpu._internal.pallas re-exports custom_jax_kernel and jax_op
    import torch_tpu._internal.pallas as internal_pallas  # pylint: disable=g-import-not-at-top

    internal_pallas_dir = dir(internal_pallas)
    self.assertIn("custom_jax_kernel", internal_pallas_dir)
    self.assertIn("jax_op", internal_pallas_dir)
    self.assertIs(internal_pallas.jax_op, internal_jax_op)
    self.assertIs(internal_pallas.custom_jax_kernel, internal_custom_jax_kernel)
    with self.assertRaises(  # ASSERT_RAISES_OK=Verifies non-exported attributes raise AttributeError
        AttributeError
    ):
      _ = getattr(internal_pallas, "non_existent_attribute")

  def test_ops_optimization_barrier_registry(self):
    """Verifies exposure, identity, metadata and proxy behavior for the barrier."""
    # 1. Submodule import and symbol identity
    ops_mod = importlib.import_module("torch.tpu.ops")
    self.assertIs(ops_mod, torch.tpu.ops)
    self.assertIs(
        torch.tpu.ops.optimization_barrier, internal_ops.optimization_barrier
    )

    # 2. Lifecycle stage metadata
    fn = torch.tpu.ops.optimization_barrier
    self.assertEqual(
        getattr(fn, annotations.TT_API_STAGE, None),
        annotations.Stage.EXPERIMENTAL.value,
    )
    self.assertEqual(
        getattr(fn, annotations.TT_API_STAGE_REASON, None),
        "torch.tpu.ops.optimization_barrier is experimental and subject to"
        " change.",
    )

    # 3. Invalid attribute handling
    with self.assertRaises(  # ASSERT_RAISES_OK=Verifies proxy module raises AttributeError for non-existent attributes
        AttributeError
    ):
      _ = getattr(torch.tpu.ops, "non_existent_attribute")

    # 4. The raw custom op is hidden behind a private name.
    self.assertFalse(hasattr(torch.ops.tpu, "optimization_barrier"))
    self.assertTrue(hasattr(torch.ops.tpu, "_optimization_barrier"))

  def test_ops_max_pool2d_registry(self):
    """Verifies exposure, identity and stage metadata for max_pool2d."""
    # 1. Symbol identity
    self.assertIs(torch.tpu.ops.max_pool2d, internal_ops.max_pool2d)

    # 2. Lifecycle stage metadata
    self.assertEqual(
        getattr(torch.tpu.ops.max_pool2d, annotations.TT_API_STAGE, None),
        annotations.Stage.STABLE.value,
    )

    # 3. The raw custom op is hidden behind a private name.
    self.assertFalse(hasattr(torch.ops.tpu, "max_pool2d"))
    self.assertTrue(hasattr(torch.ops.tpu, "_max_pool2d"))

  def test_ops_ragged_dot_registry(self):
    """Verifies exposure, identity and stage metadata for ragged_dot."""
    # 1. Symbol identity
    self.assertIs(torch.tpu.ops.ragged_dot, internal_ops.ragged_dot)

    # 2. Lifecycle stage metadata
    fn = torch.tpu.ops.ragged_dot
    self.assertEqual(
        getattr(fn, annotations.TT_API_STAGE, None),
        annotations.Stage.EXPERIMENTAL.value,
    )
    self.assertEqual(
        getattr(fn, annotations.TT_API_STAGE_REASON, None),
        "torch.tpu.ops.ragged_dot is experimental and subject to change.",
    )

    # 3. The raw custom op is hidden behind a private name.
    self.assertFalse(hasattr(torch.ops.tpu, "ragged_dot"))
    self.assertTrue(hasattr(torch.ops.tpu, "_ragged_dot"))

  def test_distributed_api_registry(self):
    """Verifies public exposure, identity, metadata, and proxy behavior for distributed."""
    # 1. Verify python -m torch.tpu.distributed resolves code cleanly without
    # RuntimeWarning before the submodule is imported into sys.modules.
    self.assertNotIn("torch.tpu.distributed", sys.modules)
    with warnings.catch_warnings():
      warnings.simplefilter("error", RuntimeWarning)
      _, mod_spec, code = runpy._get_module_details("torch.tpu.distributed")
    self.assertEqual(mod_spec.name, "torch.tpu.distributed")
    self.assertIsNotNone(code)

    # 2. Submodule import and symbol identity
    dist_mod = importlib.import_module("torch.tpu.distributed")
    from torch.tpu import distributed as from_imported_dist  # pylint: disable=g-import-not-at-top

    self.assertIs(dist_mod, torch.tpu.distributed)
    self.assertIs(from_imported_dist, torch.tpu.distributed)
    self.assertIs(
        torch.tpu.distributed.set_tpu_launch_env,
        internal_set_tpu_launch_env,
    )

    # 3. Lifecycle stage metadata
    fn = torch.tpu.distributed.set_tpu_launch_env
    self.assertEqual(
        getattr(fn, annotations.TT_API_STAGE, None),
        annotations.Stage.EXPERIMENTAL.value,
    )
    self.assertEqual(
        getattr(fn, annotations.TT_API_STAGE_REASON, None),
        "torch.tpu.distributed.set_tpu_launch_env is experimental"
        " and subject to change.",
    )

    # 4. Invalid attribute handling
    with self.assertRaises(  # ASSERT_RAISES_OK=Verifies proxy module raises AttributeError for non-existent attributes
        AttributeError
    ):
      _ = getattr(torch.tpu.distributed, "non_existent_attribute")
    with self.assertRaises(  # ASSERT_RAISES_OK=Verifies removed environment submodule raises AttributeError
        AttributeError
    ):
      _ = getattr(torch.tpu.distributed, "environment")

    # 5. Verify neither torch.tpu.distributed.environment nor torch_tpu.distributed exists
    self.assertFalse(hasattr(torch_tpu, "distributed"))
    self.assertNotIn("torch.tpu.distributed.environment", sys.modules)
    self.assertNotIn("torch_tpu.distributed", sys.modules)
    self.assertNotIn("torch_tpu.distributed.environment", sys.modules)
    for bad_mod in (
        "torch.tpu.distributed.environment",
        "torch_tpu.distributed",
        "torch_tpu.distributed.environment",
    ):
      with self.assertRaises(  # ASSERT_RAISES_OK=Verifies removed/non-public distributed modules cannot be imported
          ModuleNotFoundError
      ):
        importlib.import_module(bad_mod)

  def test_execution_mode_registry(self):
    """Verifies public exposure, identity, metadata, and functional behavior for execution mode."""
    # 1. Verify excluded / non-public symbols are not exposed
    self.assertFalse(hasattr(torch.tpu, "PublicEagerMode"))
    self.assertFalse(hasattr(torch.tpu.EagerMode, "INTERNAL_DEFER_ALL"))
    self.assertFalse(hasattr(torch.tpu.EagerMode, "INTERNAL_COMPILE_FX_GRAPH"))
    self.assertTrue(hasattr(torch.tpu.EagerMode, "_INTERNAL_COMPILE_FX_GRAPH"))

    # 2. Identity and enum members (accessed directly)
    self.assertIs(torch.tpu.EagerMode, _internal_execution_mode.EagerMode)
    self.assertLen(torch.tpu.EagerMode, 4)
    for mode in torch.tpu.EagerMode:
      raw_name = (
          "INTERNAL_COMPILE_FX_GRAPH"
          if mode.name == "_INTERNAL_COMPILE_FX_GRAPH"
          else mode.name
      )
      raw_mode = getattr(_internal_execution_mode._EagerMode, raw_name)
      self.assertEqual(mode.value, raw_mode)
      self.assertEqual(mode, raw_mode)
      self.assertEqual(hash(mode), hash(raw_mode))

    # 3. Lifecycle stage metadata
    self.assertEqual(
        getattr(torch.tpu.EagerMode, annotations.TT_API_STAGE, None),
        annotations.Stage.EXPERIMENTAL.value,
    )
    self.assertEqual(
        getattr(torch.tpu.set_eager_mode, annotations.TT_API_STAGE, None),
        annotations.Stage.EXPERIMENTAL.value,
    )

    # 4. Functional behavior (getter/setter & context manager)
    prev_mode = torch.tpu.eager_mode
    try:
      torch.tpu.eager_mode = torch.tpu.EagerMode.DEFER_NEVER
      self.assertEqual(torch.tpu.eager_mode, torch.tpu.EagerMode.DEFER_NEVER)
      torch.tpu.eager_mode = torch.tpu.EagerMode.DEFER_AND_FUSE
      self.assertEqual(torch.tpu.eager_mode, torch.tpu.EagerMode.DEFER_AND_FUSE)
      torch.tpu.eager_mode = torch.tpu.EagerMode._INTERNAL_COMPILE_FX_GRAPH
      self.assertEqual(
          torch.tpu.eager_mode, torch.tpu.EagerMode._INTERNAL_COMPILE_FX_GRAPH
      )

      with torch.tpu.set_eager_mode(torch.tpu.EagerMode.DEFER_NEVER):
        self.assertEqual(torch.tpu.eager_mode, torch.tpu.EagerMode.DEFER_NEVER)
      self.assertEqual(
          torch.tpu.eager_mode, torch.tpu.EagerMode._INTERNAL_COMPILE_FX_GRAPH
      )

      with torch.tpu.set_eager_mode(
          torch.tpu.EagerMode._INTERNAL_COMPILE_FX_GRAPH
      ):
        self.assertEqual(
            torch.tpu.eager_mode, torch.tpu.EagerMode._INTERNAL_COMPILE_FX_GRAPH
        )
      self.assertEqual(
          torch.tpu.eager_mode, torch.tpu.EagerMode._INTERNAL_COMPILE_FX_GRAPH
      )

      # Verify exception safety of context manager
      with self.assertRaises(  # ASSERT_RAISES_OK=Verifies context manager restores state upon exception
          RuntimeError
      ):
        with torch.tpu.set_eager_mode(torch.tpu.EagerMode.DEFER_NEVER):
          self.assertEqual(
              torch.tpu.eager_mode, torch.tpu.EagerMode.DEFER_NEVER
          )
          raise RuntimeError("Simulated failure")
      self.assertEqual(
          torch.tpu.eager_mode, torch.tpu.EagerMode._INTERNAL_COMPILE_FX_GRAPH
      )
    finally:
      torch.tpu.eager_mode = prev_mode

  def test_layout_registry(self):
    """Verifies exposure, identity and stage metadata for the layout API."""
    # 1. Symbol identity
    self.assertIs(torch.tpu.TpuLayout, tpu_annotations.TpuLayout)
    self.assertIs(torch.tpu.LayoutContext, tpu_annotations.LayoutContext)
    self.assertIs(
        torch.tpu.get_device_layout_if_materialized,
        tpu_annotations.get_device_layout_if_materialized,
    )

    # 2. Lifecycle stage metadata
    for api in (
        torch.tpu.TpuLayout,
        torch.tpu.LayoutContext,
        torch.tpu.get_device_layout_if_materialized,
    ):
      self.assertEqual(
          getattr(api, annotations.TT_API_STAGE, None),
          annotations.Stage.EXPERIMENTAL.value,
      )

  def test_compiler_api_registry(self):
    """Verifies public exposure, identity, metadata, and proxy behavior for compiler."""
    # 1. Symbol identity
    self.assertIs(
        torch.tpu.compiler.custom_compiler_options,
        internal_compiler_options.custom_compiler_options,
    )

    # 2. Lifecycle stage metadata
    fn = torch.tpu.compiler.custom_compiler_options
    self.assertEqual(
        getattr(fn, annotations.TT_API_STAGE, None),
        annotations.Stage.EXPERIMENTAL.value,
    )
    self.assertEqual(
        getattr(fn, annotations.TT_API_STAGE_REASON, None),
        "torch.tpu.compiler.custom_compiler_options is experimental and subject"
        " to change.",
    )

    # 3. Proxy dir and invalid attribute handling
    self.assertEqual(dir(torch.tpu.compiler), ["custom_compiler_options"])
    with self.assertRaises(  # ASSERT_RAISES_OK=Verifies proxy module raises AttributeError for non-existent attributes
        AttributeError
    ):
      _ = getattr(torch.tpu.compiler, "non_existent_attribute")

    # 4. Verify torch_tpu does not expose compiler
    self.assertFalse(hasattr(torch_tpu, "compiler"))
    self.assertNotIn("torch_tpu.compiler", sys.modules)
    with self.assertRaises(  # ASSERT_RAISES_OK=Verifies torch_tpu.compiler cannot be imported
        ModuleNotFoundError
    ):
      importlib.import_module("torch_tpu.compiler")

  def test_optimizers_api_registry(self):
    """Verifies public exposure and proxy behavior for optimizers."""
    # 1. Module accessibility
    self.assertTrue(hasattr(torch, "tpu"))
    self.assertTrue(hasattr(torch.tpu, "optimizers"))
    self.assertIn("torch.tpu.optimizers", sys.modules)

    # 2. Proxy invalid attribute handling
    with self.assertRaises(  # ASSERT_RAISES_OK=Verifies proxy module raises AttributeError for non-existent attributes
        AttributeError
    ):
      _ = getattr(torch.tpu.optimizers, "non_existent_attribute")

  def test_ops_api_registry(self):
    """Verifies public exposure, identity, metadata, and proxy behavior for ops."""
    # 1. Module accessibility
    self.assertTrue(hasattr(torch, "tpu"))
    self.assertTrue(hasattr(torch.tpu, "ops"))
    self.assertIn("torch.tpu.ops", sys.modules)

    # 2. Symbol accessibility
    self.assertTrue(hasattr(torch.tpu.ops, "align_jagged_to_multiple"))
    self.assertTrue(hasattr(torch.tpu.ops, "optimization_barrier"))
    self.assertTrue(hasattr(torch.tpu.ops, "strip_jagged_padding"))
    self.assertTrue(hasattr(torch.tpu.ops, "splash_sdpa"))


if __name__ == "__main__":
  absltest.main()
