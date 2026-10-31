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

"""Covers tcmalloc detection, the warning category, and the warning filters."""

from collections.abc import Iterator
import contextlib
import re
import types
from typing import Final
from unittest import mock
import warnings

from absl.testing import absltest
from absl.testing import parameterized
from torch_tpu import _tcmalloc_check
from torch_tpu._internal import env
from torch_tpu._internal import warnings as warning_categories
from torch_tpu._internal.distributed import multiprocessing
from tests import seed_test_utils

# The prefix users are told to filter on, via
# PYTHONWARNINGS="ignore:<TorchTpu/Performance/TcMalloc/".
_DOCUMENTED_FILTER_PREFIX: Final[str] = "<TorchTpu/Performance/TcMalloc/"

# A /proc/self/maps holding neither tcmalloc nor a sanitizer.
_MAPS_WITHOUT_TCMALLOC: Final[str] = "7f000000 /usr/lib/libc.so.6\n"


@contextlib.contextmanager
def _no_tcmalloc() -> Iterator[None]:
  """Mocks an environment with neither tcmalloc nor a sanitizer loaded."""
  with (
      mock.patch(
          "ctypes.CDLL", autospec=True, return_value=types.SimpleNamespace()
      ),
      # `autospec` cannot be combined with an explicit replacement object, but
      # `mock_open` already specs the file protocol it stands in for.
      mock.patch(
          "builtins.open", mock.mock_open(read_data=_MAPS_WITHOUT_TCMALLOC)
      ),
  ):
    yield


def _check_tcmalloc_in_child_worker(q) -> None:
  """Worker function executed in a spawned child process."""
  with _no_tcmalloc(), warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    _tcmalloc_check.check_tcmalloc()
  q.put(
      (env.has_torch_tpu_ancestor_process(), [str(w.message) for w in caught])
  )


class CheckTcmallocTest(seed_test_utils.RepeatableTest):

  def test_warns_when_tcmalloc_missing(self) -> None:
    with _no_tcmalloc(), warnings.catch_warnings(record=True) as caught:
      warnings.simplefilter("always")
      _tcmalloc_check.check_tcmalloc()

    self.assertLen(caught, 1)
    self.assertIs(caught[0].category, UserWarning)
    self.assertStartsWith(
        str(caught[0].message), _tcmalloc_check.TCMALLOC_WARNING_PREFIX
    )
    self.assertIn("LD_PRELOAD", str(caught[0].message))

  def test_prefix_matches_cpp_warning_category(self) -> None:
    # `_tcmalloc_check` can't get the prefix from C++, so it hard-codes it.
    self.assertEqual(
        _tcmalloc_check.TCMALLOC_WARNING_PREFIX,
        warning_categories.get_prefix(warning_categories.TCMALLOC),
    )
    self.assertIs(
        warning_categories.get_builtin_category(warning_categories.TCMALLOC),
        UserWarning,
    )

  def test_message_starts_with_the_documented_filter_prefix(self) -> None:
    # PYTHONWARNINGS matches the message as an anchored prefix, so rewording
    # the start of the message would silently break the documented incantation.
    with _no_tcmalloc(), warnings.catch_warnings(record=True) as caught:
      warnings.simplefilter("always")
      _tcmalloc_check.check_tcmalloc()

    self.assertStartsWith(str(caught[0].message), _DOCUMENTED_FILTER_PREFIX)
    self.assertIn(
        f'PYTHONWARNINGS="ignore:{_DOCUMENTED_FILTER_PREFIX}"',
        str(caught[0].message),
    )

  def test_warning_is_suppressed_by_documented_message_filter(self) -> None:
    # Mirrors the PYTHONWARNINGS filter suggested by the warning, which Python
    # turns into an anchored, re.escape()d message filter.
    with _no_tcmalloc(), warnings.catch_warnings(record=True) as caught:
      warnings.resetwarnings()
      warnings.filterwarnings(
          "ignore", message=re.escape(_DOCUMENTED_FILTER_PREFIX)
      )
      _tcmalloc_check.install_default_filters()
      _tcmalloc_check.check_tcmalloc()

    self.assertEmpty(caught)

  def test_warns_only_once_with_default_filters(self) -> None:
    # Test runners reset the process warning filters, so reinstall the ones
    # torch_tpu installs at import time.
    with _no_tcmalloc(), warnings.catch_warnings(record=True) as caught:
      warnings.resetwarnings()
      _tcmalloc_check.install_default_filters()
      _tcmalloc_check.check_tcmalloc()
      _tcmalloc_check.check_tcmalloc()

    self.assertLen(caught, 1)

  def test_user_filter_takes_precedence_over_default_filters(self) -> None:
    with _no_tcmalloc(), warnings.catch_warnings(record=True) as caught:
      warnings.resetwarnings()
      # Simulates PYTHONWARNINGS="always:<TorchTpu/Performance/".
      warnings.filterwarnings(
          "always", message=re.escape("<TorchTpu/Performance/")
      )
      _tcmalloc_check.install_default_filters()
      _tcmalloc_check.check_tcmalloc()
      _tcmalloc_check.check_tcmalloc()

    self.assertLen(caught, 2)

  @parameterized.named_parameters(
      ("tcmalloc_warning", "<TorchTpu/Performance/TcMalloc/"),
      ("tcmalloc_warning_only", "<TorchTpu/Performance/TcMalloc/>"),
      ("performance_warning", "<TorchTpu/Performance/"),
      ("torch_tpu_warning", "<TorchTpu/"),
  )
  def test_warning_is_suppressed_by_ignore_filter(self, prefix: str) -> None:
    with _no_tcmalloc(), warnings.catch_warnings(record=True) as caught:
      warnings.simplefilter("always")
      warnings.filterwarnings("ignore", message=re.escape(prefix))
      _tcmalloc_check.check_tcmalloc()

    self.assertEmpty(caught)

  def test_warning_is_escalated_by_error_filter(self) -> None:
    with _no_tcmalloc(), warnings.catch_warnings():
      warnings.filterwarnings("error", message="<TorchTpu/")
      with (
          self.assertRaisesRegex(  # ASSERT_RAISES_OK=Verifies the error filter.
              UserWarning,
              "^" + re.escape(_tcmalloc_check.TCMALLOC_WARNING_PREFIX),
          )
      ):
        _tcmalloc_check.check_tcmalloc()

  @parameterized.named_parameters(
      ("maps_has_tcmalloc", "7f000000 /workspace/libtcmalloc.so\n", ""),
      (
          "maps_has_tcmalloc_minimal",
          "7f000000 /usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4\n",
          "",
      ),
      ("maps_has_asan", "7f000000 /usr/lib/libasan.so.8\n", ""),
      ("maps_has_msan", "7f000000 /usr/lib/libmsan.so.0\n", ""),
      ("maps_has_tsan", "7f000000 /usr/lib/libtsan.so.0\n", ""),
      (
          "maps_has_clang_rt_asan",
          "7f000000 /usr/lib/clang/18/lib/linux/libclang_rt.asan-x86_64.so\n",
          "",
      ),
      (
          "maps_has_clang_rt_msan",
          "7f000000 /usr/lib/clang/18/lib/linux/libclang_rt.msan-x86_64.so\n",
          "",
      ),
      ("rtld_has_tcmalloc_symbol", "", "MallocExtension_GetAllocatedSize"),
      (
          "rtld_has_internal_tcmalloc_symbol",
          "",
          "MallocExtension_Internal_GetAllocatedSize",
      ),
      ("rtld_has_tc_version_symbol", "", "tc_version"),
      ("rtld_has_asan_symbol", "", "__asan_init"),
      ("rtld_has_msan_symbol", "", "__msan_init"),
      ("rtld_has_tsan_symbol", "", "__tsan_init"),
  )
  def test_does_not_warn_when_tcmalloc_present(
      self, maps_content: str, symbol_name: str
  ) -> None:
    syms = (
        types.SimpleNamespace(**{symbol_name: object()})
        if symbol_name
        else types.SimpleNamespace()
    )
    with (
        mock.patch("ctypes.CDLL", autospec=True, return_value=syms),
        mock.patch("builtins.open", mock.mock_open(read_data=maps_content)),
        warnings.catch_warnings(record=True) as caught,
    ):
      warnings.simplefilter("always")
      _tcmalloc_check.check_tcmalloc()

    self.assertEmpty(caught)

  def test_does_not_warn_when_maps_is_unreadable(self) -> None:
    with (
        mock.patch(
            "ctypes.CDLL", autospec=True, return_value=types.SimpleNamespace()
        ),
        mock.patch("builtins.open", autospec=True, side_effect=OSError),
        warnings.catch_warnings(record=True) as caught,
    ):
      warnings.simplefilter("always")
      _tcmalloc_check.check_tcmalloc()

    self.assertEmpty(caught)

  def test_current_process_has_tcmalloc_loaded(self) -> None:
    with warnings.catch_warnings(record=True) as caught:
      warnings.simplefilter("always")
      _tcmalloc_check.check_tcmalloc()

    self.assertEmpty(
        [w for w in caught if _DOCUMENTED_FILTER_PREFIX in str(w.message)]
    )

  def test_does_not_warn_when_has_torch_tpu_ancestor_process(self) -> None:
    with (
        _no_tcmalloc(),
        mock.patch.object(
            env,
            "has_torch_tpu_ancestor_process",
            autospec=True,
            return_value=True,
        ),
        warnings.catch_warnings(record=True) as caught,
    ):
      warnings.simplefilter("always")
      _tcmalloc_check.check_tcmalloc()

    self.assertEmpty(caught)

  def test_does_not_warn_in_spawned_child_process(self) -> None:
    self.assertFalse(env.has_torch_tpu_ancestor_process())
    ctx = multiprocessing.get_context("spawn")
    q = ctx.Queue()
    proc = ctx.Process(target=_check_tcmalloc_in_child_worker, args=(q,))
    proc.start()
    proc.join(timeout=60)
    self.assertEqual(proc.exitcode, 0)
    has_ancestor, caught_messages = q.get(timeout=5)
    self.assertTrue(has_ancestor)
    self.assertEmpty(caught_messages)


if __name__ == "__main__":
  multiprocessing.handle_test_main(absltest.main)
