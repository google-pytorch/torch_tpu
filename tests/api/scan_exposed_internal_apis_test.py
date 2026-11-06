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

"""Unit tests for `scan_exposed_internal_apis.py`."""

import ast
import json
import os
import textwrap

from absl.testing import absltest
from tests.api import scan_exposed_internal_apis

Stage = scan_exposed_internal_apis.Stage


class ScanExposedInternalApisTest(absltest.TestCase):  # ABSLTEST_OK=tool test

  def test_discover_python_files_includes_and_excludes_tests(self):
    temp_dir = self.create_tempdir().full_path
    with open(os.path.join(temp_dir, "regular.py"), "w") as f:
      f.write("# regular")
    with open(os.path.join(temp_dir, "test_helper.py"), "w") as f:
      f.write("# test")
    with open(os.path.join(temp_dir, "unit_test.py"), "w") as f:
      f.write("# unit test")
    with open(os.path.join(temp_dir, "notebook.ipynb"), "w") as f:
      f.write("{}")

    all_files = scan_exposed_internal_apis.discover_python_files(
        temp_dir, include_tests=True
    )
    prod_files = scan_exposed_internal_apis.discover_python_files(
        temp_dir, include_tests=False
    )

    self.assertLen(all_files, 4)
    self.assertLen(prod_files, 2)
    self.assertIn(os.path.join(temp_dir, "test_helper.py"), all_files)
    self.assertIn(os.path.join(temp_dir, "unit_test.py"), all_files)
    self.assertNotIn(os.path.join(temp_dir, "test_helper.py"), prod_files)
    self.assertNotIn(os.path.join(temp_dir, "unit_test.py"), prod_files)

  def test_visitor_extracts_internal_python_imports(self):
    code = textwrap.dedent("""
      import marimo as mo
      from torch_tpu._internal import execution_mode as em
      from torch_tpu._internal.utils import utils
      import torch.tpu._private_mod
      import torch_tpu._private_helper
      from torch.tpu import Event
    """)
    tree = ast.parse(code)
    visitor = scan_exposed_internal_apis.InternalApiVisitor(
        rel_path="docs/dummy_notebook.py",
        py_stages={"torch.tpu.Event": Stage.EXPERIMENTAL},
        env_stages={},
    )
    visitor.visit(tree)
    visitor.scan_text(code)

    self.assertIn("torch_tpu._internal.execution_mode", visitor.internal_py)
    self.assertEqual(
        visitor.internal_py["torch_tpu._internal.execution_mode"],
        {"docs/dummy_notebook.py"},
    )
    self.assertIn("torch_tpu._internal.utils.utils", visitor.internal_py)
    self.assertIn("torch.tpu._private_mod", visitor.internal_py)
    self.assertIn("torch_tpu._private_helper", visitor.internal_py)
    # Public APIs should be ignored
    self.assertNotIn("torch.tpu.Event", visitor.internal_py)

  def test_visitor_detects_unexposed_apis_without_leading_underscore(self):
    code = textwrap.dedent("""
      import torch.tpu
      torch.tpu.secret_unexposed_function()
      torch.tpu.Event.record()
    """)
    tree = ast.parse(code)
    visitor = scan_exposed_internal_apis.InternalApiVisitor(
        rel_path="examples/unexposed_test.py",
        py_stages={
            "torch.tpu.Event": Stage.EXPERIMENTAL,
            "torch.tpu.Event.record": Stage.EXPERIMENTAL,
        },
        env_stages={},
    )
    visitor.visit(tree)

    # secret_unexposed_function has no leading underscore but is absent from tt_apis.yaml
    self.assertIn("torch.tpu.secret_unexposed_function", visitor.internal_py)
    # Event.record is a registered public API
    self.assertNotIn("torch.tpu.Event.record", visitor.internal_py)

  def test_visitor_extracts_environment_variables(self):
    code = textwrap.dedent("""
      import os
      from torch_tpu._internal import env

      os.environ["TORCH_TPU_INTERNAL_HANDSHAKE_PORT"] = "12345"
      val = os.getenv("LIBTPU_INIT_ARGS", "")
      addr = env.get_str_env_once("MASTER_ADDR")
      if "TORCH_SHOW_CPP_STACKTRACES" in os.environ:
        pass
    """)
    tree = ast.parse(code)
    env_stages = {
        "TORCH_TPU_INTERNAL_HANDSHAKE_PORT": Stage.INTERNAL,
        "LIBTPU_INIT_ARGS": Stage.STABLE,
        "MASTER_ADDR": Stage.INTERNAL,
        "TORCH_SHOW_CPP_STACKTRACES": Stage.EXPERIMENTAL,
    }
    visitor = scan_exposed_internal_apis.InternalApiVisitor(
        rel_path="docs/env_test.py",
        py_stages={},
        env_stages=env_stages,
    )
    visitor.visit(tree)
    visitor.scan_text(code)

    self.assertIn("TORCH_TPU_INTERNAL_HANDSHAKE_PORT", visitor.internal_envs)
    self.assertIn("MASTER_ADDR", visitor.internal_envs)
    # Public / non-internal environment variables should be ignored
    self.assertNotIn("LIBTPU_INIT_ARGS", visitor.internal_envs)
    self.assertNotIn("TORCH_SHOW_CPP_STACKTRACES", visitor.internal_envs)

  def test_scan_markdown_strings_finds_backticked_and_cli_apis(self):
    source = textwrap.dedent("""
      # Overview
      Run `python -m torch_tpu._internal.distributed.launchers.singlehost_wrapper`
      Also inspect `TORCH_TPU_INTERNAL_XLA_OPTIONS` and `XLA_FLAGS`.
      python train.py --torch_tpu_internal_mlir_tracebacks=true
    """)
    env_stages = {
        "TORCH_TPU_INTERNAL_XLA_OPTIONS": Stage.INTERNAL,
        "XLA_FLAGS": Stage.EXPERIMENTAL,
    }
    visitor = scan_exposed_internal_apis.InternalApiVisitor(
        rel_path="docs/doc.py",
        py_stages={"torch_tpu_internal_mlir_tracebacks": Stage.INTERNAL},
        env_stages=env_stages,
    )
    visitor.scan_text(source)

    self.assertIn(
        "torch_tpu._internal.distributed.launchers.singlehost_wrapper",
        visitor.internal_py,
    )
    self.assertIn("torch_tpu_internal_mlir_tracebacks", visitor.internal_py)
    self.assertIn("TORCH_TPU_INTERNAL_XLA_OPTIONS", visitor.internal_envs)
    # Public XLA_FLAGS should not be tracked as an internal violation
    self.assertNotIn("XLA_FLAGS", visitor.internal_envs)

  def test_extract_code_from_ipynb(self):
    notebook = {
        "cells": [
            {
                "cell_type": "markdown",
                "source": ["# Title\n", "See `TORCH_TPU_INTERNAL_DEBUG`."],
            },
            {
                "cell_type": "code",
                "source": [
                    "import torch.tpu\n",
                    "from torch_tpu._internal import env\n",
                ],
            },
        ]
    }
    temp_dir = self.create_tempdir().full_path
    nb_path = os.path.join(temp_dir, "test_nb.ipynb")
    with open(nb_path, "w", encoding="utf-8") as f:
      json.dump(notebook, f)

    visitor = scan_exposed_internal_apis.InternalApiVisitor(
        rel_path="examples/test_nb.ipynb",
        py_stages={},
        env_stages={"TORCH_TPU_INTERNAL_DEBUG": Stage.INTERNAL},
    )
    scan_exposed_internal_apis.scan_file(nb_path, visitor)

    self.assertIn("torch_tpu._internal.env", visitor.internal_py)
    self.assertIn("TORCH_TPU_INTERNAL_DEBUG", visitor.internal_envs)

  def test_format_exposed_internal_apis_yaml_structure(self):
    internal_py = [
        scan_exposed_internal_apis.InternalApiEntry(
            name="torch_tpu._internal.execution_mode",
            category="Internal Python Module",
            stage=Stage.INTERNAL,
            used_in=["docs/eager.py", "examples/test.py"],
        )
    ]
    internal_env = [
        scan_exposed_internal_apis.InternalApiEntry(
            name="TORCH_TPU_INTERNAL_DEBUG",
            category="Environment Variable",
            stage=Stage.INTERNAL,
            used_in=["docs/env.py"],
        )
    ]

    yaml_text = scan_exposed_internal_apis.format_exposed_internal_apis_yaml(
        internal_py, internal_env
    )

    self.assertIn("# --- Internal Python API ---", yaml_text)
    self.assertIn("# --- Internal Environment Variable ---", yaml_text)
    self.assertNotIn("# --- Public Python API ---", yaml_text)
    self.assertNotIn("# --- Environment Variable ---", yaml_text)
    self.assertIn(
        "torch_tpu._internal.execution_mode: {used_in: [docs/eager.py,"
        " examples/test.py]}",
        yaml_text,
    )
    self.assertIn(
        "TORCH_TPU_INTERNAL_DEBUG: {used_in: [docs/env.py]}",
        yaml_text,
    )

  def test_scan_directories_end_to_end(self):
    temp_dir = self.create_tempdir().full_path
    docs_dir = os.path.join(temp_dir, "docs")
    examples_dir = os.path.join(temp_dir, "examples")
    os.makedirs(docs_dir)
    os.makedirs(examples_dir)

    with open(
        os.path.join(docs_dir, "notebook.py"), "w", encoding="utf-8"
    ) as f:
      f.write(textwrap.dedent("""
        from torch_tpu._internal import execution_mode
        import os
        os.environ["LIBTPU_INIT_ARGS"] = "foo"
      """))

    with open(
        os.path.join(examples_dir, "example.py"), "w", encoding="utf-8"
    ) as f:
      f.write(textwrap.dedent("""
        import os
        os.environ["TORCH_TPU_INTERNAL_TEST_FLAG"] = "1"
      """))

    tt_apis_path = os.path.join(temp_dir, "tt_apis.yaml")
    with open(tt_apis_path, "w", encoding="utf-8") as f:
      f.write(textwrap.dedent("""
        LIBTPU_INIT_ARGS: {category: Environment Variable, stage: Stable}
        TORCH_TPU_INTERNAL_TEST_FLAG: {category: Environment Variable, stage: Internal}
      """))

    internal_py, internal_env = scan_exposed_internal_apis.scan_directories(
        [docs_dir, examples_dir], tt_apis_path, repository_root=temp_dir
    )

    self.assertEqual(len(internal_py), 1)
    self.assertEqual(internal_py[0].name, "torch_tpu._internal.execution_mode")
    self.assertEqual(len(internal_env), 1)
    self.assertEqual(internal_env[0].name, "TORCH_TPU_INTERNAL_TEST_FLAG")

  def test_visitor_ignores_pytorch_native_attrs_on_call_results_and_instances(
      self,
  ):
    code = textwrap.dedent("""
      import torch.tpu
      from torch.tpu.ops import splash_sdpa

      op = torch.tpu.pallas.jax_op("test", lambda x: x)
      op.register_fake(lambda x: x)
      op.register_autograd(lambda x: x)

      loss = op(x, y).sum()
      val = op(x, y).cpu()

      res = splash_sdpa(q, k, v)
      m = res.mean()
      m.backward()
      s = res.shape
      t = res.transpose(1, 2).contiguous()
    """)
    tree = ast.parse(code)
    visitor = scan_exposed_internal_apis.InternalApiVisitor(
        rel_path="docs/custom_op.py",
        py_stages={
            "torch.tpu.pallas.jax_op": Stage.EXPERIMENTAL,
            "torch.tpu.ops.splash_sdpa": Stage.EXPERIMENTAL,
        },
        env_stages={},
    )
    visitor.visit(tree)

    self.assertNotIn("torch.tpu.pallas.jax_op.sum", visitor.internal_py)
    self.assertNotIn("torch.tpu.pallas.jax_op.cpu", visitor.internal_py)
    self.assertNotIn(
        "torch.tpu.pallas.jax_op.register_fake", visitor.internal_py
    )
    self.assertNotIn(
        "torch.tpu.pallas.jax_op.register_autograd", visitor.internal_py
    )
    self.assertNotIn("torch.tpu.ops.splash_sdpa.mean", visitor.internal_py)
    self.assertNotIn("torch.tpu.ops.splash_sdpa.shape", visitor.internal_py)
    self.assertNotIn(
        "torch.tpu.ops.splash_sdpa.transpose",
        visitor.internal_py,
    )

  def test_visitor_flags_internal_class_instance_method_calls(self):
    code = textwrap.dedent("""
      from torch_tpu._internal.models import InternalModel

      model = InternalModel()
      model.forward(x)
      model.train()
    """)
    tree = ast.parse(code)
    visitor = scan_exposed_internal_apis.InternalApiVisitor(
        rel_path="examples/internal_model_test.py",
        py_stages={},
        env_stages={},
    )
    visitor.visit(tree)

    self.assertIn(
        "torch_tpu._internal.models.InternalModel.forward", visitor.internal_py
    )
    self.assertIn(
        "torch_tpu._internal.models.InternalModel.train", visitor.internal_py
    )

  def test_visitor_flags_unexposed_attribute_on_public_class(self):
    code = textwrap.dedent("""
      from torch_tpu.extras.optimizers.single_trace_trainer import SingleTraceTrainer

      trainer = SingleTraceTrainer(model, optimizer)
      trainer.make_compiled_train_step()
      trainer.unexposed_secret_method()
    """)
    tree = ast.parse(code)
    visitor = scan_exposed_internal_apis.InternalApiVisitor(
        rel_path="examples/trainer_test.py",
        py_stages={
            "torch_tpu.extras.optimizers.single_trace_trainer.SingleTraceTrainer": (
                Stage.EXPERIMENTAL
            ),
            "torch_tpu.extras.optimizers.single_trace_trainer.SingleTraceTrainer.make_compiled_train_step": (
                Stage.EXPERIMENTAL
            ),
        },
        env_stages={},
    )
    visitor.visit(tree)

    self.assertNotIn(
        "torch_tpu.extras.optimizers.single_trace_trainer.SingleTraceTrainer.make_compiled_train_step",
        visitor.internal_py,
    )
    self.assertIn(
        "torch_tpu.extras.optimizers.single_trace_trainer.SingleTraceTrainer.unexposed_secret_method",
        visitor.internal_py,
    )

  def test_load_tt_apis_stages_keeps_names_in_multiple_sections(self):
    tt_apis_path = self.create_tempfile(content=textwrap.dedent("""\
            # --- Public Python API ---
            torch.tpu.op: {category: Public Python API, defined_in: runtime, stage: Experimental}

            # --- Duplicated Export ---
            torch.tpu.op: {also_exported_as: [torch_tpu.op], category: Duplicated Export}
            """)).full_path

    py_stages, env_stages = scan_exposed_internal_apis.load_tt_apis_stages(
        tt_apis_path
    )

    self.assertEqual(py_stages, {"torch.tpu.op": Stage.EXPERIMENTAL})
    self.assertEqual(env_stages, {})


if __name__ == "__main__":
  absltest.main()
