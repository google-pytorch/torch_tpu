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

import gzip
import hashlib
import os
import pathlib
import plistlib
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
from tests import goldens_pin
from tests import op_testing
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
      kwargs: dict[str, Any],
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
      goldens = {}
      for basename in (
          f"{prefix}0.gz",
          f"{prefix}{op_testing._GPU_GOLDEN_HOPPER_SUFFIX}0.gz",
      ):
        dummy_golden = target_tests_dir / basename
        with gzip.open(dummy_golden, "wb") as f:
          plistlib.dump({}, f, fmt=plistlib.FMT_BINARY)
        goldens[basename] = goldens_pin.GoldenPath(
            sha256=hashlib.sha256(dummy_golden.read_bytes()).hexdigest(),
            gcs_path=f"ops_test/test/{basename}",
        )

      mock_pin_data = goldens_pin.GoldenPathPinFile(
          version=1, gcs_bucket="torchtpu-goldens", goldens=goldens
      )
      mock_golden_data = mock.create_autospec(
          op_testing.GoldenGpuData, instance=True
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
          mock.patch.object(op_testing, "_GOLDEN_GPU_DATA", mock_golden_data),
          mock.patch.object(
              tests_test_utils.goldens_pin,
              "parse_goldens_pin",
              return_value=mock_pin_data,
          ),
      ):
        op_testing._load_golden_files()
        mock_golden_data.clear.assert_called_once()
        mock_golden_data.merge_plistlib_pytree.assert_has_calls(
            [mock.call({}), mock.call({})]
        )

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
  ) -> tuple[list[str], unittest.TestResult]:
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
      values: list[float],
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
  ) -> list[float]:
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


if __name__ == "__main__":
  absltest.main()
