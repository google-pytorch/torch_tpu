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

"""Tests for async compiled artifact error handling."""

import concurrent.futures

from absl.testing import absltest
from torch_tpu._internal.compile import torch_tpu_compiled_executable
from tests import seed_test_utils

AsyncCompiledArtifact = torch_tpu_compiled_executable.AsyncCompiledArtifact


class AsyncCompiledArtifactTest(seed_test_utils.RepeatableTest):

  def test_magic_attributes_do_not_resolve_and_raise_attribute_error(self):
    future = concurrent.futures.Future()
    artifact = AsyncCompiledArtifact(future)

    with self.assertRaises(AttributeError):
      _ = artifact.__non_existent_magic_attr__

    self.assertFalse(artifact.is_resolved)

    # Special framework attributes return class defaults without resolving
    self.assertIsNone(artifact._fx_graph_cache_key)
    self.assertFalse(artifact._is_fallback)
    self.assertFalse(artifact.is_resolved)

  def test_exception_propagation_on_resolve(self):
    future = concurrent.futures.Future()
    future.set_exception(RuntimeError("Simulated XLA compilation failure"))

    artifact = AsyncCompiledArtifact(future)
    self.assertFalse(artifact.is_resolved)

    with self.assertRaisesRegex(
        RuntimeError, "Simulated XLA compilation failure"
    ):
      artifact.resolve()

    self.assertFalse(artifact.is_resolved)

  def test_exception_propagation_on_call(self):
    future = concurrent.futures.Future()
    future.set_exception(ValueError("Invalid tensor layout"))

    artifact = AsyncCompiledArtifact(future)

    with self.assertRaisesRegex(ValueError, "Invalid tensor layout"):
      artifact([])

    self.assertFalse(artifact.is_resolved)

  def test_exception_propagation_on_getattr(self):
    future = concurrent.futures.Future()
    future.set_exception(RuntimeError("Compile error"))

    artifact = AsyncCompiledArtifact(future)

    with self.assertRaisesRegex(RuntimeError, "Compile error"):
      _ = artifact.some_missing_attr


if __name__ == "__main__":
  absltest.main()
