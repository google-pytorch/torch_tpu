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

"""Tests compiler-option propagation across the async compile thread pool."""

import concurrent.futures
import threading
from unittest import mock

from absl.testing import absltest
import torch
from torch_tpu._internal.compile import compiler as compiler_mod


class _AddOne(torch.nn.Module):

  def forward(self, x):
    return x + 1


class _CompiledArtifact:
  pass


class _FakeCompilerOptions:
  """Thread-local stand-in for the native compiler-options TLS stack."""

  def __init__(self):
    self._local = threading.local()

  def _stack(self):
    if not hasattr(self._local, "stack"):
      self._local.stack = []
    return self._local.stack

  def _get_compiler_options(self):
    stack = self._stack()
    return dict(stack[-1]) if stack else {}

  def push_compiler_options(self, options):
    merged = self._get_compiler_options()
    merged.update(options)
    self._stack().append(merged)

  def pop_compiler_options(self):
    self._stack().pop()


class AsyncCompileOptionsTest(absltest.TestCase):  # ABSLTEST_OK=no RNG used

  def setUp(self):
    super().setUp()
    self.fake_options = _FakeCompilerOptions()
    options_patcher = mock.patch.object(
        compiler_mod, "_compiler_options", self.fake_options
    )
    options_patcher.start()
    self.addCleanup(options_patcher.stop)

    self.executor = concurrent.futures.ThreadPoolExecutor(
        max_workers=1, thread_name_prefix="async-options-test"
    )
    self.addCleanup(self.executor.shutdown, wait=True)
    original_executor = compiler_mod.StaticCompiler._async_compile_executor
    compiler_mod.StaticCompiler._async_compile_executor = self.executor
    self.addCleanup(
        setattr,
        compiler_mod.StaticCompiler,
        "_async_compile_executor",
        original_executor,
    )

  def _make_compile(self, records, error=None):

    def compile_fn(*args, **kwargs):
      del args, kwargs
      records.append({
          "thread": threading.current_thread().name,
          "options": self.fake_options._get_compiler_options(),
      })
      if error is not None:
        raise error
      return _CompiledArtifact()

    return compile_fn

  def _submit(self, compile_fn):
    compiler = compiler_mod.StaticCompiler(async_compile=True)
    compiler._compile = compile_fn
    graph_module = torch.fx.symbolic_trace(_AddOne())
    return compiler(graph_module, [torch.ones(2)])

  def test_async_compile_uses_custom_options(self):
    records = []
    self.fake_options.push_compiler_options({"xla_optimization_level": "O1"})
    try:
      artifact = self._submit(self._make_compile(records))
    finally:
      self.fake_options.pop_compiler_options()

    artifact._future.result(timeout=10)

    self.assertEqual(
        records[0]["options"], {"xla_optimization_level": "O1"}
    )
    self.assertNotEqual(records[0]["thread"], threading.current_thread().name)

  def test_async_compile_uses_nested_custom_options(self):
    records = []
    self.fake_options.push_compiler_options({"xla_optimization_level": "O1"})
    self.fake_options.push_compiler_options(
        {"xla_memory_fitting_level": "O0"}
    )
    try:
      artifact = self._submit(self._make_compile(records))
    finally:
      self.fake_options.pop_compiler_options()
      self.fake_options.pop_compiler_options()

    artifact._future.result(timeout=10)

    self.assertEqual(
        records[0]["options"],
        {
            "xla_optimization_level": "O1",
            "xla_memory_fitting_level": "O0",
        },
    )

  def test_async_compile_restores_options_after_worker_error(self):
    self.executor.submit(
        self.fake_options.push_compiler_options, {"baseline": "1"}
    ).result(timeout=10)
    self.addCleanup(
        lambda: self.executor.submit(
            self.fake_options.pop_compiler_options
        ).result(timeout=10)
    )

    records = []
    self.fake_options.push_compiler_options({"xla_optimization_level": "O1"})
    try:
      artifact = self._submit(
          self._make_compile(records, error=RuntimeError("boom"))
      )
    finally:
      self.fake_options.pop_compiler_options()

    with self.assertRaisesRegex(  # ASSERT_RAISES_OK=worker cleanup
        RuntimeError, "boom"
    ):
      artifact._future.result(timeout=10)
    self.assertEqual(
        records[0]["options"],
        {"baseline": "1", "xla_optimization_level": "O1"},
    )

    cleanup_records = []
    cleanup_artifact = self._submit(self._make_compile(cleanup_records))
    cleanup_artifact._future.result(timeout=10)

    self.assertEqual(cleanup_records[0]["options"], {"baseline": "1"})
    self.assertEqual(cleanup_records[0]["thread"], records[0]["thread"])


if __name__ == "__main__":
  absltest.main()
