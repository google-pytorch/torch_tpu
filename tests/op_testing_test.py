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

"""Unit tests for the op_testing framework."""

from collections.abc import Mapping, Sequence
import dataclasses
import gzip
import hashlib
import multiprocessing
import os
import pathlib
import plistlib
import sys
import tempfile
from typing import Any
import unittest
from unittest import mock

from absl.testing import absltest
from absl.testing import flagsaver
from absl.testing import parameterized
import torch
from torch.testing._internal import common_methods_invocations
from torch.utils import _pytree
from torch_tpu._internal.utils import test_utils
from tests import dynamism_test_utils
from tests import goldens_pin
from tests import op_testing
from tests import ops_test
from tests import seed_test_utils
from tests import test_utils as tests_test_utils

op_db = common_methods_invocations.op_db

# Capture Dynamo's default recompile_limit at import time before SeededTest.setUp
# overrides it via disable_cache_limit().
_DEFAULT_DYNAMO_RECOMPILE_LIMIT = (
    torch._dynamo.config.recompile_limit  # pylint: disable=protected-access
)


class FakeSample:
  """A fake test sample for an op."""

  def __init__(
      self,
      name: str,
      input_value: torch.Tensor,
      args: tuple[Any, ...],
      kwargs: Mapping[str, Any],
  ):
    self.name = name
    self.input = input_value
    self.args = args
    self.kwargs = kwargs


class OpTestingTest(op_testing.OpInfoTestBase):
  """Tests for the op_testing framework itself."""

  def test_torch_tpu_vs_gpu_max_samples(self):
    op = next(op for op in op_db if op.name == "add")
    fake_samples = [
        (
            op_testing.OpInput(FakeSample(f"s{i}", torch.zeros(1), (), {})),
            op_testing.OpOutput(torch.zeros(1)),
        )
        for i in range(5)
    ]
    golden_data = op_testing.GoldenGpuData()
    for op_input, op_output in fake_samples:
      golden_data.add(
          self._testMethodName,
          op_testing.OpVariant.BASE,
          torch.float32,
          op_input,
          op_output,
      )
    with (
        flagsaver.flagsaver(test_mode=op_testing.TestMode.TORCH_TPU_VS_GPU),
        mock.patch.object(op_testing, "_GOLDEN_GPU_DATA", golden_data),
    ):
      # 1. max_samples argument limits returned samples.
      res = self._get_golden_input_output_pairs(
          op=op,
          dtype=torch.float32,
          variant=op_testing.OpVariant.BASE,
          max_samples=2,
      )
      self.assertEqual(res, fake_samples[:2])

      # 2. When max_samples is None, --max_samples_per_op_dtype flag is respected.
      with flagsaver.flagsaver(max_samples_per_op_dtype=3):
        res = self._get_golden_input_output_pairs(
            op=op,
            dtype=torch.float32,
            variant=op_testing.OpVariant.BASE,
            max_samples=None,
        )
        self.assertEqual(res, fake_samples[:3])

      # 3. Default or negative max_samples returns all samples.
      with flagsaver.flagsaver(max_samples_per_op_dtype=-1):
        res = self._get_golden_input_output_pairs(
            op=op,
            dtype=torch.float32,
            variant=op_testing.OpVariant.BASE,
            max_samples=None,
        )
        self.assertEqual(res, fake_samples)

  def _make_golden_sample_tree(
      self, op_name: str, value: float
  ) -> _pytree.PyTree:
    data = op_testing.GoldenGpuData()
    sample = (
        op_testing.OpInput(
            FakeSample(f"s_{value}", torch.tensor([value]), (), {})
        ),
        op_testing.OpOutput(torch.tensor([value])),
    )
    data.add(op_name, op_testing.OpVariant.BASE, torch.float32, *sample)
    return data.to_plistlib_pytree()

  def test_golden_gpu_data_merge_is_lazy(self):
    tree = self._make_golden_sample_tree("test_op", 1.0)
    golden_data = op_testing.GoldenGpuData()

    # Merging alone must not trigger tensor deserialization.
    with mock.patch.object(
        op_testing.OpInput,
        "from_plistlib_pytree",
        wraps=op_testing.OpInput.from_plistlib_pytree,
    ) as mock_deser:
      golden_data.merge_plistlib_pytree(tree)
      mock_deser.assert_not_called()

  def test_golden_gpu_data_get_samples_deserializes_and_caches_in_place(self):
    tree = self._make_golden_sample_tree("test_op", 1.0)
    golden_data = op_testing.GoldenGpuData()
    golden_data.merge_plistlib_pytree(tree)

    with mock.patch.object(
        op_testing.OpInput,
        "from_plistlib_pytree",
        wraps=op_testing.OpInput.from_plistlib_pytree,
    ) as mock_deser:
      samples1 = golden_data.get_samples(
          "test_op", op_testing.OpVariant.BASE, torch.float32
      )
      samples2 = golden_data.get_samples(
          "test_op", op_testing.OpVariant.BASE, torch.float32
      )

    # Deserialized once, cached for subsequent calls, returns correct tensor.
    self.assertEqual(mock_deser.call_count, 1)
    self.assertIs(samples1[0], samples2[0])
    self.assertEqual(samples1[0][1].output_value.item(), 1.0)

  def test_golden_gpu_data_incremental_merge_does_not_redecode_existing_samples(
      self,
  ):
    # Load sample 1 and trigger its deserialization.
    tree1 = self._make_golden_sample_tree("test_op", 1.0)
    tree2 = self._make_golden_sample_tree("test_op", 2.0)
    golden_data = op_testing.GoldenGpuData()
    golden_data.merge_plistlib_pytree(tree1)
    golden_data.get_samples("test_op", op_testing.OpVariant.BASE, torch.float32)

    # Merge sample 2 and retrieve all samples.
    with mock.patch.object(
        op_testing.OpInput,
        "from_plistlib_pytree",
        wraps=op_testing.OpInput.from_plistlib_pytree,
    ) as mock_deser:
      golden_data.merge_plistlib_pytree(tree2)
      samples = golden_data.get_samples(
          "test_op", op_testing.OpVariant.BASE, torch.float32
      )

    # Only sample 2 was decoded (call count is 1, not 2).
    self.assertEqual(mock_deser.call_count, 1)
    self.assertLen(samples, 2)
    self.assertEqual(samples[0][1].output_value.item(), 1.0)
    self.assertEqual(samples[1][1].output_value.item(), 2.0)

  @parameterized.named_parameters(
      ("coo", lambda t: t.to_sparse()),
      ("csr", lambda t: t.to_sparse_csr()),
      ("csc", lambda t: t.to_sparse_csc()),
      ("bsr", lambda t: t.to_sparse_bsr(blocksize=(1, 1))),
      ("bsc", lambda t: t.to_sparse_bsc(blocksize=(1, 1))),
  )
  def test_plistlib_sparse_tensor_serialization(self, to_sparse_fn):
    dense = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    sparse = to_sparse_fn(dense)
    encoded = op_testing._to_plistlib_compatible(sparse)
    restored = op_testing._from_plistlib_compatible(encoded)
    if sparse.layout == torch.sparse_coo:
      self.assertEqual(restored.layout, torch.sparse_coo)
      test_utils.assert_close(restored, sparse)
    else:
      test_utils.assert_close(restored, dense)

  def test_plistlib_float8_e8m0fnu_serialization(self):
    # TODO: b/576283069 - Remove hasattr check once minimum OSS PyTorch is
    # >= 2.12.0.
    if not hasattr(torch, "float8_e8m0fnu"):
      # For pytorch < 2.12
      self.skipTest("float8_e8m0fnu is not supported")
    t = torch.ones((4, 4), dtype=torch.float8_e8m0fnu)
    encoded = op_testing._to_plistlib_compatible(t)
    restored = op_testing._from_plistlib_compatible(encoded)
    self.assertEqual(restored.dtype, torch.float8_e8m0fnu)
    self.assertTrue(
        torch.equal(restored.view(torch.uint8), t.view(torch.uint8))
    )

  def test_supports_out(self):
    add_op = next(op for op in op_db if op.name == "add")
    self.assertTrue(op_testing._supports_out(add_op))

    foreach_abs_op = op_testing._get_op("_foreach_abs")
    self.assertFalse(op_testing._supports_out(foreach_abs_op))

  def test_find_cast_pairs(self):
    # Distinct dtypes with both allowed upcast (int32 -> float32) and illegal downcast (float32 -> int32).
    pairs = op_testing.find_cast_pairs([torch.float32, torch.int32])
    self.assertEqual(pairs.allowed_pair, (torch.int32, torch.float32))
    self.assertEqual(pairs.illegal_pair, (torch.float32, torch.int32))

    # All pairs allowed (e.g. float32 <-> float64).
    pairs = op_testing.find_cast_pairs([torch.float32, torch.float64])
    self.assertEqual(pairs.allowed_pair, (torch.float32, torch.float64))
    self.assertIsNone(pairs.illegal_pair)

  def test_find_cast_pairs_duplicate_dtypes(self):
    # Duplicate entries of the same dtype should not yield identical pairs (e.g. float32 -> float32).
    pairs = op_testing.find_cast_pairs([torch.float32, torch.float32])
    self.assertIsNone(pairs.allowed_pair)
    self.assertIsNone(pairs.illegal_pair)

    # Duplicates mixed with other dtypes should yield distinct pairs.
    pairs = op_testing.find_cast_pairs(
        [torch.float32, torch.float32, torch.int32]
    )
    self.assertEqual(pairs.allowed_pair, (torch.int32, torch.float32))
    self.assertEqual(pairs.illegal_pair, (torch.float32, torch.int32))

  def test_find_cast_pairs_insufficient_dtypes(self):
    self.assertEqual(
        op_testing.find_cast_pairs([]),
        op_testing.CastPairs(illegal_pair=None, allowed_pair=None),
    )
    self.assertEqual(
        op_testing.find_cast_pairs([torch.float32]),
        op_testing.CastPairs(illegal_pair=None, allowed_pair=None),
    )

  @parameterized.parameters(
      ("_main",),
      ("torch_tpu",),
  )
  def test_load_golden_files_from_test_srcdir(self, workspace_name):
    with tempfile.TemporaryDirectory() as tmpdir:
      tmp_path = pathlib.Path(tmpdir)
      dummy_test_utils_path = (
          tmp_path / "site-packages" / "tests" / "test_utils.py"
      )
      dummy_test_utils_path.parent.mkdir(parents=True)

      test_srcdir = tmp_path / "runfiles"
      target_tests_dir = test_srcdir / workspace_name / "tests"
      target_tests_dir.mkdir(parents=True)

      prefix = op_testing._golden_file_prefix()
      golden_paths = [
          self._write_golden_file(target_tests_dir, f"{prefix}0.gz", [1.0]),
          self._write_golden_file(
              target_tests_dir,
              f"{prefix}{op_testing._GPU_GOLDEN_HOPPER_SUFFIX}0.gz",
              [2.0],
              test_case_name="test_fake_op_hopper",
          ),
      ]
      pin_data = goldens_pin.GoldenPathPinFile(
          version=1,
          gcs_bucket="torchtpu-goldens",
          goldens={
              path.name: goldens_pin.GoldenPath(
                  sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                  gcs_path=f"ops_test/test/{path.name}",
              )
              for path in golden_paths
          },
      )
      # Pre-existing samples must be cleared by the load.
      golden_data = op_testing.GoldenGpuData()
      stale = torch.full((1,), -1.0)
      golden_data.add(
          "test_stale_op",
          op_testing.OpVariant.BASE,
          torch.float32,
          op_testing.OpInput(FakeSample("stale", stale, (), {})),
          op_testing.OpOutput(stale),
      )
      with (
          flagsaver.flagsaver(golden_data_base_dir=""),
          mock.patch.dict(os.environ, {"TEST_SRCDIR": str(test_srcdir)}),
          mock.patch.object(
              tests_test_utils, "__file__", str(dummy_test_utils_path)
          ),
          mock.patch.object(
              pathlib.Path, "cwd", return_value=tmp_path / "empty_cwd"
          ),
          mock.patch.object(op_testing, "_GOLDEN_GPU_DATA", golden_data),
          mock.patch.object(
              tests_test_utils.goldens_pin,
              "parse_goldens_pin",
              return_value=pin_data,
          ),
      ):
        op_testing._load_golden_files()

      def loaded_values(test_case_name: str) -> list[float]:
        samples = golden_data.get_samples(
            test_case_name, op_testing.OpVariant.BASE, torch.float32
        )
        return [op_output.output_value.item() for _, op_output in samples]

      self.assertEqual(loaded_values("test_fake_op"), [1.0])
      self.assertEqual(loaded_values("test_fake_op_hopper"), [2.0])
      self.assertEqual(loaded_values("test_stale_op"), [])

  def test_load_golden_files_candidate_dirs(self):
    with tempfile.TemporaryDirectory() as tmpdir:
      tmp_path = pathlib.Path(tmpdir)
      dummy_test_utils_path = (
          tmp_path / "site-packages" / "tests" / "test_utils.py"
      )
      dummy_test_utils_path.parent.mkdir(parents=True)

      test_srcdir = tmp_path / "runfiles"

      with (
          flagsaver.flagsaver(golden_data_base_dir=""),
          mock.patch.dict(os.environ, {"TEST_SRCDIR": str(test_srcdir)}),
          mock.patch.object(
              tests_test_utils, "__file__", str(dummy_test_utils_path)
          ),
          mock.patch.object(
              pathlib.Path, "cwd", return_value=tmp_path / "empty_cwd"
          ),
      ):
        candidate_dirs = [
            str(d)
            for d in tests_test_utils.goldens_candidate_local_dirs(
                base_dir=op_testing._GOLDEN_DATA_BASE_DIR.value
            )
        ]
        self.assertIn(str(test_srcdir / "_main" / "tests"), candidate_dirs)
        self.assertIn(str(test_srcdir / "torch_tpu" / "tests"), candidate_dirs)
        self.assertNotIn("google3", "".join(candidate_dirs))

  def test_do_test_op_skips_when_all_dtypes_excluded(self):
    op = op_testing._get_op("abs")
    all_dtypes = tuple(
        dict.fromkeys((*op_testing.ALL_NUMERIC_DTYPES, *op.dtypes))
    )
    with self.assertRaises(  # ASSERT_RAISES_OK=Testing framework skip error.
        unittest.case.SkipTest
    ) as cm:
      self.do_test_op(
          "abs",
          exclude_dtypes=all_dtypes,
      )
    self.assertIn("All dtypes excluded for abs", str(cm.exception))

  def test_do_test_op_fails_when_zero_subtests_executed(self):
    with mock.patch.object(
        self, "_get_golden_input_output_pairs", return_value=[]
    ):
      with self.assertRaises(  # ASSERT_RAISES_OK=Testing zero subtest failure.
          self.failureException
      ) as cm:
        self.do_test_op("abs")
      self.assertIn("executed 0 subtests for golden_device", str(cm.exception))

  @parameterized.parameters(
      (256, 8),  # Aligned: M=256, K_log=8 (K=256)
      (64, 4),  # Unaligned M < 128
      (1, 4),  # Edge case: M=1
      (128, 2),  # Unaligned K_log < 4
      (32, 2),  # Both unaligned
  )
  def test_swizzle_unswizzle_scale_roundtrip(self, m, k_logical):
    # LHS scale: [M, K_logical]
    lhs_scale = torch.randn(m, k_logical, dtype=torch.float32)
    lhs_swizzled = op_testing._swizzle_lhs_scale(lhs_scale)
    m_blocks = (m + 127) // 128
    k_blocks = (k_logical + 3) // 4
    self.assertEqual(lhs_swizzled.numel(), m_blocks * 128 * k_blocks * 4)
    lhs_restored = op_testing._unswizzle_lhs_scale(lhs_swizzled, m, k_logical)
    test_utils.assert_close(lhs_restored, lhs_scale)

    # RHS scale: [K_logical, N]
    n = m
    rhs_scale = torch.randn(k_logical, n, dtype=torch.float32)
    rhs_swizzled = op_testing._swizzle_rhs_scale(rhs_scale)
    n_blocks = (n + 127) // 128
    self.assertEqual(rhs_swizzled.numel(), n_blocks * 128 * k_blocks * 4)
    rhs_restored = op_testing._unswizzle_rhs_scale(rhs_swizzled, k_logical, n)
    test_utils.assert_close(rhs_restored, rhs_scale)

  @parameterized.named_parameters(
      ("sm90", 9, op_testing.GPU_SM90),
      ("sm100", 10, op_testing.GPU_SM100),
  )
  def test_golden_target_gpu_arch(self, major, expected_target):
    mock_prop = mock.MagicMock(major=major)
    with (
        mock.patch.object(torch.cuda, "is_available", return_value=True),
        mock.patch.object(
            torch.cuda, "get_device_properties", return_value=mock_prop
        ),
    ):
      self.assertEqual(op_testing.golden_target_gpu_arch(), expected_target)

  @parameterized.named_parameters(
      ("no_cuda", False, 0, "CUDA is not available"),
      (
          "unsupported_major_11",
          True,
          11,
          "Unsupported GPU compute capability 11",
      ),
  )
  def test_golden_target_gpu_arch_error(
      self, cuda_available, major, error_regex
  ):
    mock_prop = mock.MagicMock(major=major)
    with (
        mock.patch.object(
            torch.cuda, "is_available", return_value=cuda_available
        ),
        mock.patch.object(
            torch.cuda, "get_device_properties", return_value=mock_prop
        ),
    ):
      with self.assertRaisesRegex(  # ASSERT_RAISES_OK=Testing GPU arch validation error.
          RuntimeError, error_regex
      ):
        op_testing.golden_target_gpu_arch()

  @parameterized.named_parameters(
      ("sm90", op_testing.GPU_SM90, "ops_test_gpu_golden_hopper2.gz"),
      ("sm100", op_testing.GPU_SM100, "ops_test_gpu_golden2.gz"),
  )
  def test_golden_file_basename(self, target_gpu, expected_name):
    with mock.patch.object(
        op_testing,
        "golden_target_gpu_arch",
        return_value=target_gpu,
    ):
      self.assertEqual(op_testing._golden_file_basename(2), expected_name)

  def _run_golden_arch_test(
      self, method_name: str
  ) -> tuple[Sequence[str], unittest.TestResult]:
    """Runs one method of a test class using record_gpu_goldens_on."""
    ran = []

    class GoldenArchTest(op_testing.OpInfoTestBase):

      @op_testing.record_gpu_goldens_on(op_testing.GPU_SM90)
      def test_tagged(self):
        ran.append(self._testMethodName)

      def test_untagged(self):
        ran.append(self._testMethodName)

    result = unittest.TestResult()
    GoldenArchTest(method_name).run(result)
    self.assertEmpty(result.errors)
    self.assertEmpty(result.failures)
    return ran, result

  @parameterized.named_parameters(
      ("tagged_on_hopper", "test_tagged", op_testing.GPU_SM90, True),
      ("tagged_on_blackwell", "test_tagged", op_testing.GPU_SM100, False),
      ("untagged_on_hopper", "test_untagged", op_testing.GPU_SM90, False),
      ("untagged_on_blackwell", "test_untagged", op_testing.GPU_SM100, True),
  )
  def test_record_gpu_goldens_on_gen_mode(
      self, method_name, current_arch, expect_run
  ):
    with (
        flagsaver.flagsaver(test_mode=op_testing.TestMode.GEN_GPU_GOLDEN),
        mock.patch.object(
            op_testing, "golden_target_gpu_arch", return_value=current_arch
        ),
    ):
      ran, result = self._run_golden_arch_test(method_name)
    self.assertEqual(ran, [method_name] if expect_run else [])
    self.assertLen(result.skipped, 0 if expect_run else 1)

  @parameterized.parameters("test_tagged", "test_untagged")
  def test_record_gpu_goldens_on_other_modes(self, method_name):
    # Outside gen_gpu_golden mode, the GPU is never queried and every test runs.
    with (
        flagsaver.flagsaver(test_mode=op_testing.TestMode.TORCH_TPU_VS_GPU),
        mock.patch.object(
            op_testing,
            "golden_target_gpu_arch",
            side_effect=AssertionError("unexpected GPU query"),
        ),
    ):
      ran, result = self._run_golden_arch_test(method_name)
    self.assertEqual(ran, [method_name])
    self.assertEmpty(result.skipped)

  def test_record_gpu_goldens_on_tags_method_in_place(self):
    def test_method(self):
      del self

    decorated = op_testing.record_gpu_goldens_on(op_testing.GPU_SM90)(
        test_method
    )
    self.assertIs(decorated, test_method)
    self.assertEqual(
        getattr(decorated, op_testing._GPU_GOLDEN_ARCH_ATTR),
        op_testing.GPU_SM90,
    )

  def _write_golden_file(
      self,
      directory: pathlib.Path,
      name: str,
      values: Sequence[float],
      test_case_name: str = "test_fake_op",
  ) -> pathlib.Path:
    """Writes a golden file holding one sample per entry in `values`."""
    golden_data = op_testing.GoldenGpuData()
    for value in values:
      tensor = torch.full((1,), value)
      golden_data.add(
          test_case_name,
          op_testing.OpVariant.BASE,
          torch.float32,
          op_testing.OpInput(FakeSample(f"s{value}", tensor, (), {})),
          op_testing.OpOutput(tensor),
      )
    path = directory / name
    with gzip.open(path, "wb") as f:
      f.write(
          plistlib.dumps(
              golden_data.to_plistlib_pytree(), fmt=plistlib.FMT_BINARY
          )
      )
    return path

  def _load_goldens_from(
      self,
      directory: pathlib.Path,
      test_case_name: str = "test_fake_op",
  ) -> Sequence[float]:
    """Pins and loads the goldens in `directory`; returns one key's values."""
    pin_data = goldens_pin.GoldenPathPinFile(
        version=1,
        gcs_bucket="torchtpu-goldens",
        goldens={
            name: goldens_pin.GoldenPath(
                sha256=hashlib.sha256(
                    (directory / name).read_bytes()
                ).hexdigest(),
                gcs_path=f"ops_test/test/{name}",
            )
            for name in (path.name for path in directory.glob("*.gz"))
        },
    )
    golden_data = op_testing.GoldenGpuData()
    with (
        flagsaver.flagsaver(golden_data_base_dir=str(directory)),
        mock.patch.object(op_testing, "_GOLDEN_GPU_DATA", golden_data),
        mock.patch.object(
            tests_test_utils.goldens_pin,
            "parse_goldens_pin",
            return_value=pin_data,
        ),
    ):
      op_testing._load_golden_files()
    samples = golden_data.get_samples(
        test_case_name, op_testing.OpVariant.BASE, torch.float32
    )
    return [op_output.output_value.item() for _, op_output in samples]

  def test_load_golden_files_unions_both_architectures(self):
    with tempfile.TemporaryDirectory() as tmp:
      directory = pathlib.Path(tmp)
      self._write_golden_file(directory, "ops_test_gpu_golden0.gz", [1.0])
      self._write_golden_file(directory, "ops_test_gpu_golden1.gz", [2.0])
      self._write_golden_file(
          directory,
          "ops_test_gpu_golden_hopper0.gz",
          [3.0],
          test_case_name="test_fake_op_hopper",
      )

      self.assertEqual(self._load_goldens_from(directory), [1.0, 2.0])
      self.assertEqual(
          self._load_goldens_from(directory, "test_fake_op_hopper"), [3.0]
      )

  def test_load_golden_files_accepts_empty_hopper_file(self):
    with tempfile.TemporaryDirectory() as tmp:
      directory = pathlib.Path(tmp)
      self._write_golden_file(directory, "ops_test_gpu_golden0.gz", [1.0])
      self._write_golden_file(directory, "ops_test_gpu_golden_hopper0.gz", [])

      self.assertEqual(self._load_goldens_from(directory), [1.0])

  def test_load_golden_files_ignores_other_prefixes(self):
    with tempfile.TemporaryDirectory() as tmp:
      directory = pathlib.Path(tmp)
      self._write_golden_file(directory, "ops_test_gpu_golden0.gz", [1.0])
      self._write_golden_file(
          directory,
          "ops_test_gpu_golden_hopper0.gz",
          [2.0],
          test_case_name="test_fake_op_hopper",
      )
      self._write_golden_file(directory, "ops_test_gpu_golden_grad0.gz", [3.0])
      self._write_golden_file(
          directory, "ops_test_gpu_golden_grad_hopper0.gz", [4.0]
      )
      self._write_golden_file(
          directory, "ops_test_gpu_golden_compiled0.gz", [5.0]
      )

      self.assertEqual(self._load_goldens_from(directory), [1.0])
      self.assertEqual(
          self._load_goldens_from(directory, "test_fake_op_hopper"), [2.0]
      )

  def test_load_golden_files_rejects_cross_arch_shared_key(self):
    with tempfile.TemporaryDirectory() as tmp:
      directory = pathlib.Path(tmp)
      self._write_golden_file(directory, "ops_test_gpu_golden0.gz", [1.0])
      self._write_golden_file(
          directory, "ops_test_gpu_golden_hopper0.gz", [2.0]
      )

      with self.assertRaisesRegex(  # ASSERT_RAISES_OK=Testing duplicate golden keys error.
          ValueError,
          "ops_test_gpu_golden0.gz and ops_test_gpu_golden_hopper0.gz both"
          " hold",
      ):
        self._load_goldens_from(directory)

  def test_monolithic_fallback_when_per_test_case_dir_absent(self):
    with tempfile.TemporaryDirectory() as tmp:
      directory = pathlib.Path(tmp)
      self._write_golden_file(directory, "ops_test_gpu_golden0.gz", [42.0])

      self.assertEqual(self._load_goldens_from(directory), [42.0])
      self.assertEmpty(
          self._load_goldens_from(directory, test_case_name="test_nonexistent")
      )

  def test_lazy_load_from_per_test_case_dir(self):
    with tempfile.TemporaryDirectory() as tmp:
      directory = pathlib.Path(tmp)
      per_test_case_dir = directory / "ops_test_gpu_golden_per_test_case"
      per_test_case_dir.mkdir()

      golden_writer = op_testing.GoldenGpuData()
      tensor = torch.full((1,), 84.0)
      golden_writer.add(
          "test_fake_lazy_op",
          op_testing.OpVariant.BASE,
          torch.float32,
          op_testing.OpInput(FakeSample("s84", tensor, (), {})),
          op_testing.OpOutput(tensor),
      )
      all_trees = golden_writer.to_plistlib_pytree()
      op_tree = all_trees["test_fake_lazy_op"]

      with open(per_test_case_dir / "test_fake_lazy_op.plist", "wb") as f:
        plistlib.dump(op_tree, f, fmt=plistlib.FMT_BINARY)

      golden_data = op_testing.GoldenGpuData()
      with (
          flagsaver.flagsaver(golden_data_base_dir=str(directory)),
          mock.patch.object(op_testing, "_GOLDEN_GPU_DATA", golden_data),
      ):
        samples = golden_data.get_samples(
            "test_fake_lazy_op", op_testing.OpVariant.BASE, torch.float32
        )
        self.assertLen(samples, 1)
        self.assertEqual(samples[0][1].output_value.item(), 84.0)

        empty_samples = golden_data.get_samples(
            "test_nonexistent_op", op_testing.OpVariant.BASE, torch.float32
        )
        self.assertEmpty(empty_samples)

  @torch._dynamo.config.patch(fail_on_recompile_limit_hit=True)  # pylint: disable=protected-access
  def test_dynamo_cache_limit_disabled_across_subtests(self):
    cnt = torch._dynamo.testing.CompileCounter()  # pylint: disable=protected-access
    compiled_fn = torch.compile(lambda x: x + 1, backend=cnt, dynamic=False)

    num_subtests = _DEFAULT_DYNAMO_RECOMPILE_LIMIT + 4
    # With fail_on_recompile_limit_hit=True, we are checking that calling
    # compiled_fn across more subtests than the default recompile limit does
    # not raise an exception.
    for i in range(1, num_subtests + 1):
      with self.subTest(i=i):
        compiled_fn(torch.ones(i))

    self.assertEqual(cnt.frame_count, num_subtests)


@dataclasses.dataclass(frozen=True)
class _RecordedDoTestOpCall:
  """Arguments captured from a single `TestOps.do_test_op` invocation."""

  test_name: str
  op_name: str
  variant_test_name: str | None
  check_grad: bool

  @property
  def full_key(self) -> str:
    if self.variant_test_name:
      return f"{self.op_name}.{self.variant_test_name}"
    return self.op_name


class DynamismDenyListsTest(seed_test_utils.RepeatableTest):
  """Tests for the dynamism op deny lists in dynamism_test_utils."""

  _DENY_LISTS = {
      "DYNAMISM_DENY_LISTS": dynamism_test_utils.DYNAMISM_DENY_LISTS,
      "DYNAMISM_NOT_APPLICABLE_OPS": (
          dynamism_test_utils.DYNAMISM_NOT_APPLICABLE_OPS
      ),
  }

  def _collect_test_ops_calls(self) -> list[_RecordedDoTestOpCall]:
    calls: list[_RecordedDoTestOpCall] = []
    current_test_name = ""

    def record(
        test_self: op_testing.OpInfoTestBase,
        op_name: str,
        *,
        check_grad: bool = True,
        variant_test_name: str | None = None,
        **kwargs: Any,
    ) -> None:
      del test_self, kwargs
      calls.append(
          _RecordedDoTestOpCall(
              test_name=current_test_name,
              op_name=op_name,
              variant_test_name=variant_test_name,
              check_grad=check_grad,
          )
      )

    test_instance = ops_test.TestOps("test_abs")
    with mock.patch.object(ops_test.TestOps, "do_test_op", record):
      for attr_name in sorted(dir(ops_test.TestOps)):
        if not attr_name.startswith("test_"):
          continue
        method = getattr(test_instance, attr_name)
        if not callable(method):
          continue
        before = len(calls)
        current_test_name = attr_name
        method()
        self.assertGreater(
            len(calls),
            before,
            f"TestOps.{attr_name} did not invoke do_test_op().",
        )
    return calls

  @staticmethod
  def _variant_key_to_base_op_name() -> dict[str, str]:
    base_names = {op.name for op in op_testing._KNOWN_OPS}
    mapping: dict[str, str] = {}
    for op in op_testing._KNOWN_OPS:
      if op.variant_test_name:
        variant_key = dynamism_test_utils.op_deny_list_key(op)
        if variant_key not in base_names:
          mapping[variant_key] = op.name
    return mapping

  def test_entries_are_known_ops(self):
    # verify_op_supports_dynamism() matches OpInfo.name or
    # OpInfo.name.variant_test_name, so a misspelled or stale entry would
    # silently stop skipping its op.
    known_op_keys = {op.name for op in op_testing._KNOWN_OPS} | {
        dynamism_test_utils.op_deny_list_key(op)
        for op in op_testing._KNOWN_OPS
        if op.variant_test_name
    }
    for list_name, deny_lists in self._DENY_LISTS.items():
      for reason, ops in deny_lists.items():
        with self.subTest(deny_list=list_name, reason=reason):
          self.assertEmpty(sorted(set(ops) - known_op_keys))

  def test_entries_have_valid_modes_and_failure_kinds(self):
    valid_kinds = set(dynamism_test_utils.DynamismFailureKind)
    for reason, ops in dynamism_test_utils.DYNAMISM_DENY_LISTS.items():
      for op_key, entry in ops.items():
        with self.subTest(
            deny_list="DYNAMISM_DENY_LISTS", reason=reason, op=op_key
        ):
          mode_kinds = dynamism_test_utils.entry_mode_kinds(entry)
          self.assertNotEmpty(mode_kinds)
          self.assertContainsSubset(
              mode_kinds.keys(), dynamism_test_utils.ALL_MODES
          )
          self.assertContainsSubset(mode_kinds.values(), valid_kinds)

    for reason, ops in dynamism_test_utils.DYNAMISM_NOT_APPLICABLE_OPS.items():
      for op_key, modes in ops.items():
        with self.subTest(
            deny_list="DYNAMISM_NOT_APPLICABLE_OPS", reason=reason, op=op_key
        ):
          self.assertIsInstance(modes, (set, frozenset))
          self.assertNotEmpty(modes)
          self.assertContainsSubset(modes, dynamism_test_utils.ALL_MODES)

  def test_deny_lists_are_disjoint(self):
    variant_to_base = self._variant_key_to_base_op_name()
    seen: dict[tuple[str, dynamism_test_utils.DynamismMode], str] = {}
    duplicates: list[str] = []
    for list_name, deny_lists in self._DENY_LISTS.items():
      for reason, ops in deny_lists.items():
        location = f"{list_name}[{reason!r}]"
        for op_key, entry in ops.items():
          for mode in dynamism_test_utils.entry_mode_kinds(entry):
            key = (op_key, mode)
            if key in seen:
              duplicates.append(
                  f"{op_key} ({mode.value}) in both {seen[key]} and {location}"
              )
            else:
              seen[key] = location
    for (op_key, mode), location in seen.items():
      base_op = variant_to_base.get(op_key)
      if base_op is not None and (base_op, mode) in seen:
        duplicates.append(
            f"variant {op_key} ({mode.value}) in {location} overlaps base"
            f" {base_op} in {seen[(base_op, mode)]}"
        )
    self.assertEmpty(duplicates)

  def test_entries_have_live_test_ops_coverage(self):
    calls = self._collect_test_ops_calls()
    tested_keys = {call.op_name for call in calls} | {
        call.full_key for call in calls
    }
    tested_full_keys = {call.full_key for call in calls}

    for list_name, deny_lists in self._DENY_LISTS.items():
      for reason, ops in deny_lists.items():
        with self.subTest(deny_list=list_name, reason=reason):
          self.assertEmpty(
              sorted(set(ops) - tested_keys),
              "Entries without a TestOps method have no live dynamism test.",
          )

    with self.subTest(deny_list="SYMBOLIC_SHAPES_KNOWN_FAILURES"):
      self.assertEmpty(
          sorted(
              set(ops_test.SYMBOLIC_SHAPES_KNOWN_FAILURES) - tested_full_keys
          ),
          "SYMBOLIC_SHAPES_KNOWN_FAILURES keys must match a TestOps call.",
      )

  def test_no_compile_grad_entries_on_check_grad_false_ops(self):
    calls = self._collect_test_ops_calls()
    grad_checked_keys = {call.op_name for call in calls if call.check_grad} | {
        call.full_key for call in calls if call.check_grad
    }
    dead_entries = [
        op_key
        for ops in dynamism_test_utils.DYNAMISM_DENY_LISTS.values()
        for op_key, entry in ops.items()
        if dynamism_test_utils.COMPILE_GRAD
        in dynamism_test_utils.entry_mode_kinds(entry)
        and op_key not in grad_checked_keys
    ]
    self.assertEmpty(
        sorted(dead_entries),
        "COMPILE_GRAD entries on ops with check_grad=False never execute.",
    )

  def test_no_compiled_entries_masked_by_compiled_deny_list(self):
    variant_to_base = self._variant_key_to_base_op_name()
    compiled_modes = {
        dynamism_test_utils.COMPILE,
        dynamism_test_utils.COMPILE_GRAD,
    }
    masked_entries = [
        (op_key, mode.value)
        for ops in dynamism_test_utils.DYNAMISM_DENY_LISTS.values()
        for op_key, entry in ops.items()
        if variant_to_base.get(op_key, op_key) in op_testing._COMPILED_DENY_LIST
        for mode in dynamism_test_utils.entry_mode_kinds(entry)
        if mode in compiled_modes
    ]
    self.assertEmpty(
        sorted(masked_entries),
        "Ops in op_testing._COMPILED_DENY_LIST already skip all compiled"
        " variants, so COMPILE/COMPILE_GRAD entries in DYNAMISM_DENY_LISTS are"
        " masked.",
    )

  def test_get_dynamism_exclusion_kinds(self):
    add_op = next(op for op in op_testing._KNOWN_OPS if op.name == "add")
    sub_op = next(op for op in op_testing._KNOWN_OPS if op.name == "sub")
    with (
        mock.patch.dict(
            dynamism_test_utils.DYNAMISM_DENY_LISTS,
            {
                "mixed": {
                    "add": {
                        dynamism_test_utils.EAGER: dynamism_test_utils.CRASH,
                        dynamism_test_utils.COMPILE: (
                            dynamism_test_utils.UNOPTIMIZED_ASSERT_CRASH
                        ),
                        dynamism_test_utils.COMPILE_GRAD: (
                            dynamism_test_utils.EXCEPTION
                        ),
                    },
                },
                "default exception": {
                    "sub": {dynamism_test_utils.EAGER},
                },
            },
            clear=True,
        ),
        mock.patch.dict(
            dynamism_test_utils.DYNAMISM_NOT_APPLICABLE_OPS,
            {"by design": {"sub": {dynamism_test_utils.COMPILE}}},
            clear=True,
        ),
    ):
      self.assertEqual(
          dynamism_test_utils.get_dynamism_exclusion(
              add_op, dynamism_test_utils.EAGER
          ),
          dynamism_test_utils.DynamismExclusion(
              reason="mixed", failure_kind=dynamism_test_utils.CRASH
          ),
      )
      self.assertEqual(
          dynamism_test_utils.get_dynamism_exclusion(
              add_op, dynamism_test_utils.COMPILE
          ),
          dynamism_test_utils.DynamismExclusion(
              reason="mixed",
              failure_kind=dynamism_test_utils.UNOPTIMIZED_ASSERT_CRASH,
          ),
      )
      self.assertEqual(
          dynamism_test_utils.get_dynamism_exclusion(
              add_op, dynamism_test_utils.COMPILE_GRAD
          ),
          dynamism_test_utils.DynamismExclusion(
              reason="mixed", failure_kind=dynamism_test_utils.EXCEPTION
          ),
      )
      self.assertEqual(
          dynamism_test_utils.get_dynamism_exclusion(
              sub_op, dynamism_test_utils.EAGER
          ),
          dynamism_test_utils.DynamismExclusion(
              reason="default exception",
              failure_kind=dynamism_test_utils.EXCEPTION,
          ),
      )
      not_applicable = dynamism_test_utils.get_dynamism_exclusion(
          sub_op, dynamism_test_utils.COMPILE
      )
      self.assertIsNotNone(not_applicable)
      self.assertTrue(not_applicable.is_not_applicable)
      self.assertIsNone(not_applicable.failure_kind)
      self.assertIsNone(
          dynamism_test_utils.get_dynamism_exclusion(
              sub_op, dynamism_test_utils.COMPILE_GRAD
          )
      )

  def test_verify_op_supports_dynamism_variant_keys(self):
    pad_reflect = next(
        op
        for op in op_testing._KNOWN_OPS
        if op.name == "nn.functional.pad" and op.variant_test_name == "reflect"
    )
    pad_replicate = next(
        op
        for op in op_testing._KNOWN_OPS
        if op.name == "nn.functional.pad"
        and op.variant_test_name == "replicate"
    )
    with mock.patch.dict(
        dynamism_test_utils.DYNAMISM_DENY_LISTS,
        {
            "reflect only": {
                "nn.functional.pad.reflect": {
                    dynamism_test_utils.EAGER,
                    dynamism_test_utils.COMPILE,
                },
            },
            "all pad in compile": {
                "nn.functional.pad": {dynamism_test_utils.COMPILE},
            },
        },
        clear=True,
    ):
      reflect_eager = dynamism_test_utils.verify_op_supports_dynamism(
          pad_reflect, mode=dynamism_test_utils.EAGER
      )
      self.assertIsNotNone(reflect_eager)
      self.assertIn("nn.functional.pad.reflect", reflect_eager)
      self.assertIn("reflect only", reflect_eager)
      self.assertIsNone(
          dynamism_test_utils.verify_op_supports_dynamism(
              pad_replicate, mode=dynamism_test_utils.EAGER
          )
      )
      # Variant-specific entry takes precedence over base entry.
      reflect_compile = dynamism_test_utils.verify_op_supports_dynamism(
          pad_reflect, mode=dynamism_test_utils.COMPILE
      )
      self.assertIsNotNone(reflect_compile)
      self.assertIn("reflect only", reflect_compile)
      replicate_compile = dynamism_test_utils.verify_op_supports_dynamism(
          pad_replicate, mode=dynamism_test_utils.COMPILE
      )
      self.assertIsNotNone(replicate_compile)
      self.assertIn("all pad in compile", replicate_compile)


class DynamismXfailTest(seed_test_utils.RepeatableTest):
  """Tests in-process xfail and exclusion behavior for --check_dynamism."""

  def _make_golden_data(
      self, shape: tuple[int, ...] = (3, 4)
  ) -> op_testing.GoldenGpuData:
    golden_data = op_testing.GoldenGpuData()
    tensor = torch.ones(shape, dtype=torch.float32)
    golden_data.add(
        "test_abs",
        op_testing.OpVariant.BASE,
        torch.float32,
        op_testing.OpInput(FakeSample("s0", tensor, (), {})),
        op_testing.OpOutput(tensor),
    )
    return golden_data

  def _run_abs_dynamism_test(
      self,
      *,
      subtest_side_effect: Any = None,
      deny_lists: Mapping[str, Mapping[str, Any]] | None = None,
      not_applicable: Mapping[str, Mapping[str, Any]] | None = None,
      sample_shape: tuple[int, ...] = (3, 4),
      flag_overrides: Mapping[str, Any] | None = None,
      optimized_build: bool = True,
  ) -> tuple[unittest.TestResult, mock.MagicMock]:
    class AbsDynamismTest(op_testing.OpInfoTestBase):

      def setUp(self) -> None:
        super().setUp()
        self.synchronize_tensors_in_tear_down = False

      def test_abs(self) -> None:
        self.do_test_op(
            "abs",
            exclude_inplace_dtypes=(torch.float32,),
            check_out_variant=False,
        )

    golden_data = self._make_golden_data(sample_shape)
    flags_dict: dict[str, Any] = {
        "check_dynamism": True,
        "test_mode": op_testing.TestMode.TORCH_TPU_VS_GPU,
    }
    if flag_overrides:
      flags_dict.update(flag_overrides)

    result = unittest.TestResult()
    with (
        flagsaver.flagsaver(**flags_dict),
        mock.patch.object(op_testing, "_GOLDEN_GPU_DATA", golden_data),
        mock.patch.object(
            op_testing, "_dtypes_to_test_for_op", return_value=(torch.float32,)
        ),
        mock.patch.object(
            op_testing.env, "TORCH_TPU_IS_OPTIMIZED_BUILD", optimized_build
        ),
        mock.patch.dict(
            dynamism_test_utils.DYNAMISM_DENY_LISTS,
            deny_lists or {},
            clear=True,
        ),
        mock.patch.dict(
            dynamism_test_utils.DYNAMISM_NOT_APPLICABLE_OPS,
            not_applicable or {},
            clear=True,
        ),
        mock.patch.object(
            AbsDynamismTest, "_sub_test", side_effect=subtest_side_effect
        ) as sub_test_mock,
    ):
      AbsDynamismTest("test_abs").run(result)
    return result, sub_test_mock

  @parameterized.named_parameters(
      ("runtime_error", RuntimeError("boom")),
      ("assertion_error", AssertionError("numerical mismatch")),
  )
  def test_exception_entry_passes_when_subtest_fails(self, exc):
    with mock.patch.object(
        op_testing.tt_testing, "reset_eager_state"
    ) as reset_mock:
      result, sub_test_mock = self._run_abs_dynamism_test(
          subtest_side_effect=exc,
          deny_lists={"known bug": {"abs": {dynamism_test_utils.EAGER}}},
      )
    self.assertTrue(result.wasSuccessful(), msg=result.failures + result.errors)
    self.assertEmpty(result.skipped)
    self.assertGreaterEqual(sub_test_mock.call_count, 1)
    # setUp() calls reset_eager_state once, plus once per failing subtest.
    self.assertGreaterEqual(reset_mock.call_count, 1 + sub_test_mock.call_count)

  def test_exception_entry_fails_loudly_when_all_subtests_pass(self):
    result, sub_test_mock = self._run_abs_dynamism_test(
        subtest_side_effect=None,
        deny_lists={"fixed bug": {"abs": {dynamism_test_utils.EAGER}}},
    )
    self.assertGreaterEqual(sub_test_mock.call_count, 1)
    self.assertLen(result.failures, 1)
    self.assertIn("Remove it from DYNAMISM_DENY_LISTS", result.failures[0][1])

  def test_exception_entry_fails_loudly_when_zero_dynamism_subtests_run(self):
    # A (1,) sample has no dims >= 2, so 0 dynamism subtests execute.
    result, sub_test_mock = self._run_abs_dynamism_test(
        sample_shape=(1,),
        deny_lists={"dead entry": {"abs": {dynamism_test_utils.EAGER}}},
    )
    sub_test_mock.assert_not_called()
    self.assertLen(result.failures, 1)
    self.assertIn("0 dynamism subtests ran", result.failures[0][1])

  def test_exception_entry_skips_do_not_satisfy_expected_failure(self):
    result, _ = self._run_abs_dynamism_test(
        subtest_side_effect=unittest.case.SkipTest("skipped subtest"),
        deny_lists={"skipped bug": {"abs": {dynamism_test_utils.EAGER}}},
    )
    self.assertNotEmpty(result.skipped)
    self.assertLen(result.failures, 1)
    self.assertIn("0 dynamism subtests ran", result.failures[0][1])

  @parameterized.named_parameters(
      ("dtypes_subset", {"dtypes": ["float32"]}),
      ("plan_filter", {"dynamism_plan_filter": "t0d0"}),
  )
  def test_exception_entry_filtered_run_does_not_false_fail(
      self, flag_overrides
  ):
    result, sub_test_mock = self._run_abs_dynamism_test(
        subtest_side_effect=None,
        deny_lists={"other sample fails": {"abs": {dynamism_test_utils.EAGER}}},
        flag_overrides=flag_overrides,
    )
    self.assertTrue(result.wasSuccessful(), msg=result.failures + result.errors)
    self.assertGreaterEqual(sub_test_mock.call_count, 1)

  def test_crash_entry_skips_in_process(self):
    result, sub_test_mock = self._run_abs_dynamism_test(
        deny_lists={
            "aborts": {
                "abs": {dynamism_test_utils.EAGER: dynamism_test_utils.CRASH}
            }
        },
    )
    sub_test_mock.assert_not_called()
    self.assertLen(result.skipped, 1)
    self.assertTrue(result.wasSuccessful())

  @parameterized.named_parameters(
      ("unoptimized_skips", False, None, True, 0),
      ("optimized_passes_when_op_passes", True, None, False, 2),
      (
          "optimized_catches_when_op_fails",
          True,
          RuntimeError("opt err"),
          False,
          2,
      ),
  )
  def test_unoptimized_assert_crash_policy(
      self, optimized_build, side_effect, expect_skipped, expected_calls
  ):
    result, sub_test_mock = self._run_abs_dynamism_test(
        subtest_side_effect=side_effect,
        deny_lists={
            "mlir assert": {
                "abs": {
                    dynamism_test_utils.EAGER: (
                        dynamism_test_utils.UNOPTIMIZED_ASSERT_CRASH
                    )
                }
            }
        },
        optimized_build=optimized_build,
    )
    self.assertTrue(result.wasSuccessful(), msg=result.failures + result.errors)
    self.assertLen(result.skipped, 1 if expect_skipped else 0)
    self.assertEqual(sub_test_mock.call_count, expected_calls)

  def test_ignore_dynamism_deny_list_flag(self):
    # EXCEPTION entry fails normally instead of being swallowed as xfail.
    exc_result, exc_calls = self._run_abs_dynamism_test(
        subtest_side_effect=RuntimeError("boom"),
        deny_lists={"known bug": {"abs": {dynamism_test_utils.EAGER}}},
        flag_overrides={"ignore_dynamism_deny_list": True},
    )
    self.assertGreaterEqual(exc_calls.call_count, 1)
    self.assertFalse(exc_result.wasSuccessful())

    # CRASH entry executes instead of skipping.
    crash_result, crash_calls = self._run_abs_dynamism_test(
        subtest_side_effect=None,
        deny_lists={
            "aborts": {
                "abs": {dynamism_test_utils.EAGER: dynamism_test_utils.CRASH}
            }
        },
        flag_overrides={"ignore_dynamism_deny_list": True},
    )
    self.assertGreaterEqual(crash_calls.call_count, 1)
    self.assertEmpty(crash_result.skipped)
    self.assertTrue(crash_result.wasSuccessful())

    # DYNAMISM_NOT_APPLICABLE_OPS still skips even with the flag set.
    na_result, na_calls = self._run_abs_dynamism_test(
        not_applicable={"by design": {"abs": {dynamism_test_utils.EAGER}}},
        flag_overrides={"ignore_dynamism_deny_list": True},
    )
    na_calls.assert_not_called()
    self.assertLen(na_result.skipped, 1)
    self.assertTrue(na_result.wasSuccessful())


class DynamismCrashCheckTest(seed_test_utils.RepeatableTest):
  """Tests --check_dynamism_crashes parent/subprocess gating and evaluation."""

  def _run_abs_crash_check(
      self,
      *,
      deny_lists: Mapping[str, Mapping[str, Any]] | None = None,
      not_applicable: Mapping[str, Mapping[str, Any]] | None = None,
      subprocess_result: (
          op_testing._DynamismCrashSubprocessResult | None  # pylint: disable=protected-access
      ) = None,
      optimized_build: bool = True,
      tpu_initialized: bool = False,
      flag_overrides: Mapping[str, Any] | None = None,
  ) -> tuple[unittest.TestResult, mock.MagicMock]:
    class AbsCrashTest(op_testing.OpInfoTestBase):

      def test_abs(self) -> None:
        self.do_test_op(
            "abs",
            exclude_inplace_dtypes=(torch.float32,),
            check_out_variant=False,
        )

    if subprocess_result is None:
      subprocess_result = op_testing._DynamismCrashSubprocessResult(  # pylint: disable=protected-access
          exitcode=-6, outcome=None, detail=None, output="Check failed"
      )
    flags_dict: dict[str, Any] = {
        "check_dynamism": True,
        "check_dynamism_crashes": True,
        "test_mode": op_testing.TestMode.TORCH_TPU_VS_GPU,
    }
    if flag_overrides:
      flags_dict.update(flag_overrides)

    result = unittest.TestResult()
    with (
        flagsaver.flagsaver(**flags_dict),
        mock.patch.object(
            op_testing, "_dtypes_to_test_for_op", return_value=(torch.float32,)
        ),
        mock.patch.object(
            op_testing.env, "TORCH_TPU_IS_OPTIMIZED_BUILD", optimized_build
        ),
        mock.patch.object(
            op_testing._device_ops_backend,  # pylint: disable=protected-access
            "_is_initialized",
            return_value=tpu_initialized,
        ),
        mock.patch.dict(
            dynamism_test_utils.DYNAMISM_DENY_LISTS,
            deny_lists or {},
            clear=True,
        ),
        mock.patch.dict(
            dynamism_test_utils.DYNAMISM_NOT_APPLICABLE_OPS,
            not_applicable or {},
            clear=True,
        ),
        mock.patch.object(
            op_testing,
            "_run_dynamism_crash_subprocess",
            return_value=subprocess_result,
        ) as spawn_mock,
    ):
      AbsCrashTest("test_abs").run(result)
    return result, spawn_mock

  @parameterized.named_parameters(
      ("no_entry", None, None),
      (
          "exception_entry",
          {"known": {"abs": {dynamism_test_utils.EAGER}}},
          None,
      ),
      (
          "not_applicable",
          None,
          {"by design": {"abs": {dynamism_test_utils.EAGER}}},
      ),
  )
  def test_non_crashing_entries_skip_without_spawning_or_tpu_setup(
      self, deny_lists, not_applicable
  ):
    absl_tc = absltest.TestCase  # ABSLTEST_OK=Simulate OSS absl-py super() MRO.
    with (
        mock.patch.object(
            absl_tc, "setUp", lambda tc: super(absl_tc, tc).setUp()
        ),
        mock.patch.object(
            absl_tc, "tearDown", lambda tc: super(absl_tc, tc).tearDown()
        ),
        mock.patch.object(
            op_testing.tt_testing, "reset_eager_state"
        ) as reset_mock,
        mock.patch.object(torch, "manual_seed") as seed_mock,
    ):
      result, spawn_mock = self._run_abs_crash_check(
          deny_lists=deny_lists, not_applicable=not_applicable
      )
    self.assertTrue(result.wasSuccessful(), msg=result.failures + result.errors)
    self.assertLen(result.skipped, 1)
    spawn_mock.assert_not_called()
    reset_mock.assert_not_called()
    seed_mock.assert_not_called()

  @parameterized.named_parameters(
      ("optimized_skips", True, 0, True),
      ("unoptimized_spawns_and_passes", False, 1, False),
  )
  def test_unoptimized_assert_crash_build_gating(
      self, optimized_build, expected_spawns, expect_skipped
  ):
    result, spawn_mock = self._run_abs_crash_check(
        deny_lists={
            "mlir assert": {
                "abs": {
                    dynamism_test_utils.EAGER: (
                        dynamism_test_utils.UNOPTIMIZED_ASSERT_CRASH
                    )
                }
            }
        },
        optimized_build=optimized_build,
    )
    self.assertTrue(result.wasSuccessful(), msg=result.failures + result.errors)
    self.assertEqual(spawn_mock.call_count, expected_spawns)
    self.assertLen(result.skipped, 1 if expect_skipped else 0)

  @parameterized.named_parameters(
      ("sigabrt", -6, None),
      ("sigsegv", -11, None),
      ("nonzero_exit", 1, None),
      ("late_signal_after_outcome", -6, "passed"),
      ("runtime_poisoned", 0, "runtime_poisoned"),
  )
  def test_crash_entry_passes_on_subprocess_crash(self, exitcode, outcome):
    result, spawn_mock = self._run_abs_crash_check(
        deny_lists={
            "aborts": {
                "abs": {dynamism_test_utils.EAGER: dynamism_test_utils.CRASH}
            }
        },
        subprocess_result=op_testing._DynamismCrashSubprocessResult(  # pylint: disable=protected-access
            exitcode=exitcode,
            outcome=outcome,
            detail=None,
            output="Fatal error",
        ),
    )
    self.assertTrue(result.wasSuccessful(), msg=result.failures + result.errors)
    self.assertEmpty(result.skipped)
    spawn_mock.assert_called_once()
    child_argv = spawn_mock.call_args.args[2]
    self.assertIn("--check_dynamism=True", child_argv)
    self.assertIn("--check_dynamism_crashes=False", child_argv)
    self.assertIn("--_run_crashing_dynamism_ops=True", child_argv)
    self.assertIn("--ignore_dynamism_deny_list=True", child_argv)

  @parameterized.named_parameters(
      (
          "passed",
          0,
          "passed",
          None,
          "Remove it from DYNAMISM_DENY_LISTS",
      ),
      (
          "python_exception",
          0,
          "failed",
          "RuntimeError: boom",
          "Demote it to DynamismFailureKind.EXCEPTION",
      ),
      (
          "skipped",
          0,
          "skipped",
          "0 dynamism subtests",
          "Re-evaluate this entry",
      ),
      (
          "missing_outcome",
          0,
          None,
          None,
          "without reporting a unittest outcome",
      ),
  )
  def test_crash_entry_fails_loudly_on_non_crash_outcomes(
      self, exitcode, outcome, detail, expected_msg
  ):
    result, spawn_mock = self._run_abs_crash_check(
        deny_lists={
            "aborts": {
                "abs": {dynamism_test_utils.EAGER: dynamism_test_utils.CRASH}
            }
        },
        subprocess_result=op_testing._DynamismCrashSubprocessResult(  # pylint: disable=protected-access
            exitcode=exitcode,
            outcome=outcome,
            detail=detail,
            output="child log tail",
        ),
    )
    spawn_mock.assert_called_once()
    self.assertLen(result.failures, 1)
    self.assertIn(expected_msg, result.failures[0][1])

  def test_fails_if_parent_already_initialized_tpu(self):
    result, spawn_mock = self._run_abs_crash_check(
        deny_lists={
            "aborts": {
                "abs": {dynamism_test_utils.EAGER: dynamism_test_utils.CRASH}
            }
        },
        tpu_initialized=True,
    )
    spawn_mock.assert_not_called()
    self.assertLen(result.failures, 1)
    self.assertIn(
        "initialized the TPU runtime before spawning", result.failures[0][1]
    )

  @parameterized.named_parameters(
      (
          "without_check_dynamism",
          {"check_dynamism": False, "check_dynamism_crashes": True},
      ),
      (
          "with_symbolic_shapes",
          {
              "check_dynamism": True,
              "check_dynamism_crashes": True,
              "check_symbolic_shapes": True,
          },
      ),
      (
          "non_vs_gpu_mode",
          {
              "check_dynamism": True,
              "check_dynamism_crashes": True,
              "test_mode": op_testing.TestMode.PERF,
          },
      ),
  )
  def test_set_up_test_module_rejects_invalid_crash_flags(self, flag_overrides):
    with flagsaver.flagsaver(**flag_overrides):
      with self.assertRaisesRegex(  # ASSERT_RAISES_OK=Tests harness flags.
          ValueError, "--check_dynamism_crashes requires"
      ):
        op_testing.set_up_test_module()

  def test_set_up_test_module_skips_golden_load_in_parent_mode(self):
    with (
        flagsaver.flagsaver(
            check_dynamism=True,
            check_dynamism_crashes=True,
            test_mode=op_testing.TestMode.TORCH_TPU_VS_GPU,
        ),
        mock.patch.object(op_testing, "_load_golden_files") as load_mock,
    ):
      op_testing.set_up_test_module()
    load_mock.assert_not_called()

  def _invoke_crash_worker(
      self,
      test_cls,
      method_name: str,
      *,
      tpu_initialized: bool = False,
      reset_eager_side_effect=None,
  ):
    """Runs `_dynamism_crash_worker` directly with a real Pipe and temp file."""
    receiver_conn, sender_conn = multiprocessing.Pipe(duplex=False)
    output_file = self.create_tempfile()
    saved_stdout_fd = os.dup(1)
    saved_stderr_fd = os.dup(2)
    saved_argv = list(sys.argv)
    try:
      with (
          flagsaver.flagsaver(),
          mock.patch.object(op_testing, "set_up_test_module"),
          mock.patch.object(
              op_testing._device_ops_backend,  # pylint: disable=protected-access
              "_is_initialized",
              return_value=tpu_initialized,
          ),
          mock.patch.object(
              op_testing.tt_testing,
              "reset_eager_state",
              side_effect=reset_eager_side_effect,
          ),
      ):
        op_testing._dynamism_crash_worker(  # pylint: disable=protected-access
            test_cls,
            method_name,
            ["op_testing_test"],
            output_file.full_path,
            sender_conn,
        )
    finally:
      os.dup2(saved_stdout_fd, 1)
      os.dup2(saved_stderr_fd, 2)
      os.close(saved_stdout_fd)
      os.close(saved_stderr_fd)
      sys.argv = saved_argv
    with receiver_conn:
      outcome, detail = receiver_conn.recv()
    return outcome, detail, output_file.read_text()

  def test_crash_worker_outcomes_and_output_capture(self):
    class _FakeWorkerTest(absltest.TestCase):  # ABSLTEST_OK=Local test double.

      def test_pass_with_output(self):
        self._dynamism_subtests_run = 1
        os.write(1, b"raw stdout bytes\n")
        os.write(2, b"raw stderr bytes\n")

      def test_mixed_sample_skip_and_pass(self):
        with self.subTest("skipped_sample"):
          self.skipTest("unsupported dtype torch.complex64")
        self._dynamism_subtests_run = 1

      def test_skip_all_with_reason(self):
        self._dynamism_subtests_run = 0
        self.skipTest("all samples skipped")

      def test_skip_zero_subtests(self):
        self._dynamism_subtests_run = 0

      def test_failure(self):
        self._dynamism_subtests_run = 1
        raise RuntimeError("boom")

    outcome, detail, output = self._invoke_crash_worker(
        _FakeWorkerTest, "test_pass_with_output"
    )
    self.assertEqual(outcome, op_testing._CRASH_OUTCOME_PASSED)  # pylint: disable=protected-access
    self.assertIsNone(detail)
    self.assertIn("raw stdout bytes", output)
    self.assertIn("raw stderr bytes", output)

    outcome, detail, _ = self._invoke_crash_worker(
        _FakeWorkerTest, "test_mixed_sample_skip_and_pass"
    )
    self.assertEqual(outcome, op_testing._CRASH_OUTCOME_PASSED)  # pylint: disable=protected-access
    self.assertIsNone(detail)

    outcome, detail, _ = self._invoke_crash_worker(
        _FakeWorkerTest, "test_skip_all_with_reason"
    )
    self.assertEqual(outcome, op_testing._CRASH_OUTCOME_SKIPPED)  # pylint: disable=protected-access
    self.assertEqual(detail, "all samples skipped")

    outcome, detail, _ = self._invoke_crash_worker(
        _FakeWorkerTest, "test_skip_zero_subtests"
    )
    self.assertEqual(outcome, op_testing._CRASH_OUTCOME_SKIPPED)  # pylint: disable=protected-access
    self.assertIn("0 dynamism subtests", detail)

    outcome, detail, _ = self._invoke_crash_worker(
        _FakeWorkerTest, "test_failure", tpu_initialized=False
    )
    self.assertEqual(outcome, op_testing._CRASH_OUTCOME_FAILED)  # pylint: disable=protected-access
    self.assertIn("RuntimeError: boom", detail)

    outcome, detail, _ = self._invoke_crash_worker(
        _FakeWorkerTest,
        "test_failure",
        tpu_initialized=True,
        reset_eager_side_effect=RuntimeError("TPU halted"),
    )
    self.assertEqual(outcome, op_testing._CRASH_OUTCOME_RUNTIME_POISONED)  # pylint: disable=protected-access
    self.assertIn("RuntimeError: boom", detail)


class CoupledDimDiscoveryTest(seed_test_utils.RepeatableTest):

  def test_collect_tensors_orders_kwargs_and_excludes_out(self):
    t_in = torch.zeros(2, 3)
    t_arg = torch.zeros(3, 4)
    t_alpha = torch.zeros(1)
    t_beta = torch.zeros(2, 4)
    t_out = torch.zeros(2, 4)
    op_args = dynamism_test_utils._OpArgs(  # pylint: disable=protected-access
        input_value=t_in,
        args=(t_arg, "non_tensor", 5),
        kwargs={"out": t_out, "beta": t_beta, "alpha": t_alpha, "scale": 2.0},
    )
    collected = dynamism_test_utils._collect_tensors(op_args)  # pylint: disable=protected-access
    self.assertLen(collected, 4)
    self.assertIs(collected[0], t_in)
    self.assertIs(collected[1], t_arg)
    self.assertIs(collected[2], t_alpha)
    self.assertIs(collected[3], t_beta)

  def test_run_fake_op_allocates_distinct_symbols_and_keeps_out_static(self):
    lhs = torch.ones(4, 4)
    rhs = torch.ones(4, 4)
    out = torch.empty(4, 4)
    op_args = dynamism_test_utils._OpArgs(  # pylint: disable=protected-access
        input_value=lhs,
        args=(rhs,),
        kwargs={"out": out},
    )
    t0d0 = dynamism_test_utils.TensorDim(0, 0)
    t1d1 = dynamism_test_utils.TensorDim(1, 1)
    _, initial_syms = dynamism_test_utils._run_fake_op(  # pylint: disable=protected-access
        torch.add,
        op_args,
        mode=dynamism_test_utils.DynamismMode.COMPILE,
        dynamic_dims={t0d0, t1d1},
    )
    self.assertSameElements(initial_syms.keys(), [t0d0, t1d1])
    self.assertNotEqual(initial_syms[t0d0], initial_syms[t1d1])

  def test_discover_coupled_dims_matmul(self):
    mm_op = op_testing._get_op("mm")
    dyn_info = dynamism_test_utils.DynamicOpInfo.get(mm_op)
    lhs = torch.ones(3, 4)
    rhs = torch.ones(4, 5)
    candidates = list(dyn_info.get_candidate_dims(lhs, (rhs,)))
    op_args = dynamism_test_utils._OpArgs(lhs, (rhs,), {})  # pylint: disable=protected-access
    tensors = dynamism_test_utils._collect_tensors(op_args)  # pylint: disable=protected-access

    discovered = dyn_info._discover_coupled_dims(  # pylint: disable=protected-access
        candidates,
        tensors,
        op_args,
        mm_op,
        dynamism_test_utils.DynamismMode.COMPILE,
    )
    t0d0 = dynamism_test_utils.TensorDim(0, 0)
    t0d1 = dynamism_test_utils.TensorDim(0, 1)
    t1d0 = dynamism_test_utils.TensorDim(1, 0)
    self.assertDictEqual(
        discovered,
        {
            t0d0: (t0d0,),
            t0d1: (t0d1, t1d0),
        },
    )

  def test_discover_coupled_dims_baddbmm(self):
    baddbmm_op = op_testing._get_op("baddbmm")
    dyn_info = dynamism_test_utils.DynamicOpInfo.get(baddbmm_op)
    self_t = torch.ones(2, 3, 5)
    batch1 = torch.ones(2, 3, 4)
    batch2 = torch.ones(2, 4, 5)
    candidates = list(dyn_info.get_candidate_dims(self_t, (batch1, batch2)))
    op_args = dynamism_test_utils._OpArgs(self_t, (batch1, batch2), {})  # pylint: disable=protected-access
    tensors = dynamism_test_utils._collect_tensors(op_args)  # pylint: disable=protected-access

    discovered = dyn_info._discover_coupled_dims(  # pylint: disable=protected-access
        candidates,
        tensors,
        op_args,
        baddbmm_op,
        dynamism_test_utils.DynamismMode.COMPILE,
    )
    t0d0 = dynamism_test_utils.TensorDim(0, 0)
    t0d1 = dynamism_test_utils.TensorDim(0, 1)
    t0d2 = dynamism_test_utils.TensorDim(0, 2)
    t1d0 = dynamism_test_utils.TensorDim(1, 0)
    t1d1 = dynamism_test_utils.TensorDim(1, 1)
    t2d0 = dynamism_test_utils.TensorDim(2, 0)
    t2d2 = dynamism_test_utils.TensorDim(2, 2)
    self.assertDictEqual(
        discovered,
        {
            t0d0: (t0d0, t1d0, t2d0),
            t0d1: (t0d1, t1d1),
            t0d2: (t0d2, t2d2),
        },
    )

  def test_discover_coupled_dims_keyword_tensor_args(self):
    add_op = op_testing._get_op("add")
    dyn_info = dynamism_test_utils.DynamicOpInfo.get(add_op)
    lhs = torch.ones(3, 4)
    rhs = torch.ones(3, 4)
    candidates = list(dyn_info.get_candidate_dims(lhs, ()))
    op_args = dynamism_test_utils._OpArgs(lhs, (), {"other": rhs})  # pylint: disable=protected-access
    tensors = dynamism_test_utils._collect_tensors(op_args)  # pylint: disable=protected-access

    discovered = dyn_info._discover_coupled_dims(  # pylint: disable=protected-access
        candidates,
        tensors,
        op_args,
        torch.add,
        dynamism_test_utils.DynamismMode.COMPILE,
    )
    t0d0 = dynamism_test_utils.TensorDim(0, 0)
    t0d1 = dynamism_test_utils.TensorDim(0, 1)
    t1d0 = dynamism_test_utils.TensorDim(1, 0)
    t1d1 = dynamism_test_utils.TensorDim(1, 1)
    self.assertDictEqual(
        discovered,
        {
            t0d0: (t0d0, t1d0),
            t0d1: (t0d1, t1d1),
        },
    )

  def test_discover_coupled_dims_respects_static_tensor_indices(self):
    conv2d_op = op_testing._get_op("nn.functional.conv2d")
    dyn_info = dynamism_test_utils.DynamicOpInfo.get(conv2d_op)
    # Batch size (4) matches out_channels (4) on weight, but weight must stay
    # static via ConvolutionDynamicOpInfo._static_tensor_indices.
    inp = torch.ones(4, 3, 6, 6)
    weight = torch.ones(4, 3, 3, 3)
    candidates = list(dyn_info.get_candidate_dims(inp, (weight,)))
    op_args = dynamism_test_utils._OpArgs(inp, (weight,), {})  # pylint: disable=protected-access
    tensors = dynamism_test_utils._collect_tensors(op_args)  # pylint: disable=protected-access

    discovered = dyn_info._discover_coupled_dims(  # pylint: disable=protected-access
        candidates,
        tensors,
        op_args,
        torch.nn.functional.conv2d,
        dynamism_test_utils.DynamismMode.COMPILE,
    )
    t0d0 = dynamism_test_utils.TensorDim(0, 0)
    t0d2 = dynamism_test_utils.TensorDim(0, 2)
    t0d3 = dynamism_test_utils.TensorDim(0, 3)
    self.assertDictEqual(
        discovered,
        {
            t0d0: (t0d0,),
            t0d2: (t0d2,),
            t0d3: (t0d3,),
        },
    )

  def test_validate_coupled_plans_filters_specialized_dims(self):
    add_op = op_testing._get_op("add")
    dyn_info = dynamism_test_utils.DynamicOpInfo(add_op)
    inp = torch.ones(3, 4)
    candidates = list(dyn_info.get_candidate_dims(inp, ()))
    op_args = dynamism_test_utils._OpArgs(inp, (), {})  # pylint: disable=protected-access
    tensors = dynamism_test_utils._collect_tensors(op_args)  # pylint: disable=protected-access

    def specialize_dim0(x):
      if x.shape[0] != 3:
        raise ValueError("dim 0 must be 3")
      return x + 1

    discovered = dyn_info._discover_coupled_dims(  # pylint: disable=protected-access
        candidates,
        tensors,
        op_args,
        specialize_dim0,
        dynamism_test_utils.DynamismMode.COMPILE,
    )
    plans = dyn_info._validate_coupled_plans(  # pylint: disable=protected-access
        candidates,
        discovered,
        op_args,
        specialize_dim0,
        dynamism_test_utils.DynamismMode.COMPILE,
    )
    t0d1 = dynamism_test_utils.TensorDim(0, 1)
    self.assertEqual(
        plans,
        (
            dynamism_test_utils.DynamicMarkingPlan(
                label="t0d1",
                tensor_dims=(t0d1,),
                coupled_dims=(t0d1,),
            ),
        ),
    )

  def test_validate_coupled_plans_eager_vs_compile_modes(self):
    mm_op = op_testing._get_op("mm")
    dyn_info = dynamism_test_utils.DynamicOpInfo(mm_op)
    lhs = torch.ones(3, 4)
    rhs = torch.ones(4, 5)
    candidates = list(dyn_info.get_candidate_dims(lhs, (rhs,)))
    op_args = dynamism_test_utils._OpArgs(lhs, (rhs,), {})  # pylint: disable=protected-access
    tensors = dynamism_test_utils._collect_tensors(op_args)  # pylint: disable=protected-access

    t0d0 = dynamism_test_utils.TensorDim(0, 0)
    t0d1 = dynamism_test_utils.TensorDim(0, 1)
    t1d0 = dynamism_test_utils.TensorDim(1, 0)

    proposed_eager = dyn_info._discover_coupled_dims(  # pylint: disable=protected-access
        candidates,
        tensors,
        op_args,
        torch.mm,
        dynamism_test_utils.DynamismMode.EAGER,
    )
    plans_eager = dyn_info._validate_coupled_plans(  # pylint: disable=protected-access
        candidates,
        proposed_eager,
        op_args,
        torch.mm,
        dynamism_test_utils.DynamismMode.EAGER,
    )
    self.assertEqual(
        plans_eager,
        (
            dynamism_test_utils.DynamicMarkingPlan(
                label="t0d0",
                tensor_dims=(t0d0,),
                coupled_dims=(t0d0,),
            ),
            dynamism_test_utils.DynamicMarkingPlan(
                label="t0d1",
                tensor_dims=(t0d1,),
                coupled_dims=(t0d1, t1d0),
            ),
        ),
    )

    proposed_compile = dyn_info._discover_coupled_dims(  # pylint: disable=protected-access
        candidates,
        tensors,
        op_args,
        torch.mm,
        dynamism_test_utils.DynamismMode.COMPILE,
    )
    plans_compile = dyn_info._validate_coupled_plans(  # pylint: disable=protected-access
        candidates,
        proposed_compile,
        op_args,
        torch.mm,
        dynamism_test_utils.DynamismMode.COMPILE,
    )
    self.assertEqual(
        plans_compile,
        (
            dynamism_test_utils.DynamicMarkingPlan(
                label="t0d0",
                tensor_dims=(t0d0,),
                coupled_dims=(t0d0,),
            ),
            dynamism_test_utils.DynamicMarkingPlan(
                label="t0d1",
                tensor_dims=(t0d1,),
                coupled_dims=(t0d1, t1d0),
            ),
            dynamism_test_utils.DynamicMarkingPlan(
                label="t0d0_t0d1",
                tensor_dims=(t0d0, t0d1),
                coupled_dims=(t0d0, t0d1, t1d0),
            ),
        ),
    )

  def test_is_valid_coupled_plan_upper_bound_guards(self):
    abs_op = op_testing._get_op("abs")
    dyn_info = dynamism_test_utils.DynamicOpInfo(abs_op)
    inp = torch.ones(4, 5)
    op_args = dynamism_test_utils._OpArgs(inp, (), {})  # pylint: disable=protected-access
    t0d0 = dynamism_test_utils.TensorDim(0, 0)

    def guard_mod3_eq_1(x):
      if x.shape[0] % 3 != 1:
        raise ValueError("dim 0 mod 3 must be 1")
      return x + 1

    # Both bucket (128 % 3 == 2) and fallback (8 % 3 == 2) upper bounds
    # violate Mod(s0, 3) == 1, so COMPILE validation must reject.
    self.assertFalse(
        dyn_info._is_valid_coupled_plan(  # pylint: disable=protected-access
            guard_mod3_eq_1,
            op_args,
            (t0d0,),
            dynamism_test_utils.DynamismMode.COMPILE,
        )
    )

  def test_is_valid_coupled_plan_compile_grad_checks_backward(self):
    abs_op = op_testing._get_op("abs")
    dyn_info = dynamism_test_utils.DynamicOpInfo(abs_op)
    inp = torch.ones(3, 4, dtype=torch.float32)
    op_args = dynamism_test_utils._OpArgs(inp, (), {})  # pylint: disable=protected-access
    t0d0 = dynamism_test_utils.TensorDim(0, 0)

    class _SpecializeBackwardFn(torch.autograd.Function):

      @staticmethod
      def forward(ctx, x):
        del ctx
        return x.clone()

      @staticmethod
      def backward(ctx, grad_output):
        del ctx
        if grad_output.shape[0] != 3:
          raise RuntimeError("backward requires static dim 0 == 3")
        return grad_output

    op_fn = _SpecializeBackwardFn.apply
    self.assertTrue(
        dyn_info._is_valid_coupled_plan(  # pylint: disable=protected-access
            op_fn,
            op_args,
            (t0d0,),
            dynamism_test_utils.DynamismMode.COMPILE,
        )
    )
    self.assertFalse(
        dyn_info._is_valid_coupled_plan(  # pylint: disable=protected-access
            op_fn,
            op_args,
            (t0d0,),
            dynamism_test_utils.DynamismMode.COMPILE_GRAD,
        )
    )

  def test_generate_exhaustive_marking_plans(self):
    baddbmm_op = op_testing._get_op("baddbmm")
    self_t = torch.ones(2, 3, 5)
    batch1 = torch.ones(2, 3, 4)
    batch2 = torch.ones(2, 4, 5)
    plans = dynamism_test_utils.generate_exhaustive_input_marking_plans(
        baddbmm_op,
        self_t,
        (batch1, batch2),
        mode=dynamism_test_utils.DynamismMode.COMPILE,
        kwargs={},
    )
    t0d0 = dynamism_test_utils.TensorDim(0, 0)
    t0d1 = dynamism_test_utils.TensorDim(0, 1)
    t0d2 = dynamism_test_utils.TensorDim(0, 2)
    t1d0 = dynamism_test_utils.TensorDim(1, 0)
    t1d1 = dynamism_test_utils.TensorDim(1, 1)
    t2d0 = dynamism_test_utils.TensorDim(2, 0)
    t2d2 = dynamism_test_utils.TensorDim(2, 2)
    self.assertEqual(
        plans,
        (
            dynamism_test_utils.DynamicMarkingPlan(
                label="t0d0",
                tensor_dims=(t0d0,),
                coupled_dims=(t0d0, t1d0, t2d0),
            ),
            dynamism_test_utils.DynamicMarkingPlan(
                label="t0d1",
                tensor_dims=(t0d1,),
                coupled_dims=(t0d1, t1d1),
            ),
            dynamism_test_utils.DynamicMarkingPlan(
                label="t0d2",
                tensor_dims=(t0d2,),
                coupled_dims=(t0d2, t2d2),
            ),
            dynamism_test_utils.DynamicMarkingPlan(
                label="t0d0_t0d1_t0d2",
                tensor_dims=(t0d0, t0d1, t0d2),
                coupled_dims=(t0d0, t0d1, t0d2, t1d0, t1d1, t2d0, t2d2),
            ),
        ),
    )

  @parameterized.parameters(
      dynamism_test_utils.DynamismMode.EAGER,
      dynamism_test_utils.DynamismMode.COMPILE,
  )
  def test_foreach_plan_couples_matching_list_elements(self, mode):
    foreach_add_op = op_testing._get_op("_foreach_add")
    self_list = [torch.ones(4, 3), torch.ones(5, 1), torch.ones(1, 6)]
    # Element 2 broadcasts from a lower rank.
    other_list = [torch.ones(4, 3), torch.ones(5, 1), torch.ones(6)]
    plans = dynamism_test_utils.generate_exhaustive_input_marking_plans(
        foreach_add_op, self_list, (other_list,), mode=mode
    )
    td = dynamism_test_utils.TensorDim
    self.assertEqual(
        plans,
        (
            dynamism_test_utils.DynamicMarkingPlan(
                label="t0d0_t1d0_t2d1",
                tensor_dims=(td(0, 0), td(1, 0), td(2, 1)),
                coupled_dims=(
                    td(0, 0),
                    td(1, 0),
                    td(2, 1),
                    td(3, 0),
                    td(4, 0),
                    td(5, 0),
                ),
            ),
        ),
    )

  def test_foreach_plan_couples_all_tensor_list_args(self):
    foreach_addcmul_op = op_testing._get_op("_foreach_addcmul")
    self_list = [torch.ones(4, 4), torch.ones(3, 3)]
    tensor1 = [torch.ones(4, 4), torch.ones(3, 3)]
    tensor2 = [torch.ones(4, 4), torch.ones(3, 3)]
    # The `scalars` tensor (one value per element) is a single tensor rather
    # than a tensor list, so it stays static.
    plans = dynamism_test_utils.generate_exhaustive_input_marking_plans(
        foreach_addcmul_op,
        self_list,
        (tensor1, tensor2, torch.ones(2)),
        mode=dynamism_test_utils.DynamismMode.EAGER,
    )
    td = dynamism_test_utils.TensorDim
    self.assertLen(plans, 1)
    self.assertEqual(
        plans[0].coupled_dims,
        (td(0, 0), td(1, 0), td(2, 0), td(3, 0), td(4, 0), td(5, 0)),
    )


if __name__ == "__main__":
  absltest.main()
