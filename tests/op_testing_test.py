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
import os
import pathlib
import plistlib
import tempfile
from typing import Any
from unittest import mock

from absl.testing import absltest
from absl.testing import flagsaver
from absl.testing import parameterized
import torch
from torch.testing._internal import common_methods_invocations
from torch_tpu._internal.utils import test_utils
from tests import op_testing


op_db = common_methods_invocations.op_db


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
    test_utils.assert_close(restored, dense)

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
      empty_file = tmp_path / "site-packages" / "tests" / "op_testing.py"
      empty_file.parent.mkdir(parents=True)
      empty_file.touch()

      test_srcdir = tmp_path / "runfiles"
      target_tests_dir = test_srcdir / workspace_name / "tests"
      target_tests_dir.mkdir(parents=True)

      prefix = op_testing._golden_file_prefix()
      dummy_golden = target_tests_dir / f"{prefix}0.gz"
      with gzip.open(dummy_golden, "wb") as f:
        plistlib.dump({}, f, fmt=plistlib.FMT_BINARY)

      mock_golden_data = mock.MagicMock(spec=op_testing.GoldenGpuData)
      with (
          flagsaver.flagsaver(golden_data_base_dir=""),
          mock.patch.dict(os.environ, {"TEST_SRCDIR": str(test_srcdir)}),
          mock.patch.object(op_testing, "__file__", str(empty_file)),
          mock.patch.object(
              pathlib.Path, "cwd", return_value=tmp_path / "empty_cwd"
          ),
          mock.patch.object(op_testing, "_GOLDEN_GPU_DATA", mock_golden_data),
      ):
        op_testing._load_golden_files()
        mock_golden_data.clear.assert_called_once()
        mock_golden_data.merge_plistlib_pytree.assert_called_once_with({})

  def test_load_golden_files_candidate_dirs_error_message(self):
    with tempfile.TemporaryDirectory() as tmpdir:
      tmp_path = pathlib.Path(tmpdir)
      empty_file = tmp_path / "site-packages" / "tests" / "op_testing.py"
      empty_file.parent.mkdir(parents=True)
      empty_file.touch()

      test_srcdir = tmp_path / "runfiles"

      with (
          flagsaver.flagsaver(golden_data_base_dir=""),
          mock.patch.dict(os.environ, {"TEST_SRCDIR": str(test_srcdir)}),
          mock.patch.object(op_testing, "__file__", str(empty_file)),
          mock.patch.object(
              pathlib.Path, "cwd", return_value=tmp_path / "empty_cwd"
          ),
      ):
        with self.assertRaises(  # ASSERT_RAISES_OK=Missing golden files test.
            ValueError
        ) as cm:
          op_testing._load_golden_files()
        error_msg = str(cm.exception)
        self.assertIn(str(test_srcdir / "_main" / "tests"), error_msg)
        self.assertIn(str(test_srcdir / "torch_tpu" / "tests"), error_msg)
        self.assertNotIn("google3", error_msg)


class GoldenGpuDataLazyDecodeTest(
    absltest.TestCase  # ABSLTEST_OK=No RNG; tests golden-file decoding.
):
  """Tests that golden samples decode on first read, not at merge time."""

  def _make_sample(
      self, name: str
  ) -> tuple[op_testing.OpInput, op_testing.OpOutput]:
    return (
        op_testing.OpInput(FakeSample(name, torch.zeros(1), (), {})),
        op_testing.OpOutput(torch.zeros(1)),
    )

  def _encoded(
      self, samples: list[tuple[op_testing.OpInput, op_testing.OpOutput]]
  ) -> list[Any]:
    return [
        (i.to_plistlib_pytree(), o.to_plistlib_pytree()) for i, o in samples
    ]

  def _assert_samples_equal(self, actual, expected) -> None:
    self.assertEqual(self._encoded(list(actual)), self._encoded(expected))

  def _merged(self, *sample_groups) -> op_testing.GoldenGpuData:
    """Returns a fresh instance with each group merged as its own file."""
    merged = op_testing.GoldenGpuData()
    for group in sample_groups:
      source = op_testing.GoldenGpuData()
      for test_case_name, variant, dtype, sample in group:
        source.add(test_case_name, variant, dtype, *sample)
      merged.merge_plistlib_pytree(source.to_plistlib_pytree())
    return merged

  def _patch_decoder(self):
    return mock.patch.object(
        op_testing.OpInput,
        "from_plistlib_pytree",
        wraps=op_testing.OpInput.from_plistlib_pytree,
    )

  def test_merged_samples_read_back_per_key(self):
    first, second, third = (self._make_sample(f"s{i}") for i in range(3))
    data = self._merged([
        ("case_a", op_testing.OpVariant.BASE, torch.float32, first),
        ("case_a", op_testing.OpVariant.BASE, torch.float32, second),
        ("case_b", op_testing.OpVariant.INPLACE, torch.int32, third),
    ])

    self._assert_samples_equal(
        data.get_samples("case_a", op_testing.OpVariant.BASE, torch.float32),
        [first, second],
    )
    self._assert_samples_equal(
        data.get_samples("case_b", op_testing.OpVariant.INPLACE, torch.int32),
        [third],
    )

  def test_reading_one_key_leaves_the_others_encoded(self):
    wanted, unwanted = self._make_sample("s0"), self._make_sample("s1")
    data = self._merged([
        ("case_a", op_testing.OpVariant.BASE, torch.float32, wanted),
        ("case_a", op_testing.OpVariant.BASE, torch.int32, unwanted),
    ])

    with self._patch_decoder() as decode:
      self._assert_samples_equal(
          data.get_samples("case_a", op_testing.OpVariant.BASE, torch.float32),
          [wanted],
      )
      self.assertEqual(decode.call_count, 1)

      self._assert_samples_equal(
          data.get_samples("case_a", op_testing.OpVariant.BASE, torch.int32),
          [unwanted],
      )
      self.assertEqual(decode.call_count, 2)

  def test_samples_concatenate_in_merge_order(self):
    first, second = self._make_sample("s0"), self._make_sample("s1")
    data = self._merged(
        [("case_a", op_testing.OpVariant.BASE, torch.float32, first)],
        [("case_a", op_testing.OpVariant.BASE, torch.float32, second)],
    )

    self._assert_samples_equal(
        data.get_samples("case_a", op_testing.OpVariant.BASE, torch.float32),
        [first, second],
    )

  def test_merging_after_a_read_appends_to_what_was_decoded(self):
    first, second = self._make_sample("s0"), self._make_sample("s1")
    data = self._merged(
        [("case_a", op_testing.OpVariant.BASE, torch.float32, first)]
    )
    self._assert_samples_equal(
        data.get_samples("case_a", op_testing.OpVariant.BASE, torch.float32),
        [first],
    )

    later = op_testing.GoldenGpuData()
    later.add("case_a", op_testing.OpVariant.BASE, torch.float32, *second)
    data.merge_plistlib_pytree(later.to_plistlib_pytree())

    self._assert_samples_equal(
        data.get_samples("case_a", op_testing.OpVariant.BASE, torch.float32),
        [first, second],
    )

  def test_reading_twice_decodes_once(self):
    sample = self._make_sample("s0")
    data = self._merged(
        [("case_a", op_testing.OpVariant.BASE, torch.float32, sample)]
    )

    with self._patch_decoder() as decode:
      first_read = data.get_samples(
          "case_a", op_testing.OpVariant.BASE, torch.float32
      )
      second_read = data.get_samples(
          "case_a", op_testing.OpVariant.BASE, torch.float32
      )

    self.assertIs(first_read, second_read)
    self._assert_samples_equal(first_read, [sample])
    self.assertEqual(decode.call_count, 1)

  def test_reading_an_absent_key_returns_nothing(self):
    data = self._merged([(
        "case_a",
        op_testing.OpVariant.BASE,
        torch.float32,
        self._make_sample("s0"),
    )])

    self.assertEmpty(
        data.get_samples("case_z", op_testing.OpVariant.BASE, torch.float32)
    )
    self.assertEmpty(
        data.get_samples("case_a", op_testing.OpVariant.OUT, torch.float32)
    )
    self.assertEmpty(
        data.get_samples("case_a", op_testing.OpVariant.BASE, torch.int32)
    )

  def test_clear_drops_samples_that_were_never_read(self):
    data = self._merged([(
        "case_a",
        op_testing.OpVariant.BASE,
        torch.float32,
        self._make_sample("s0"),
    )])

    data.clear()

    self.assertEmpty(
        data.get_samples("case_a", op_testing.OpVariant.BASE, torch.float32)
    )

  def test_encoding_includes_samples_that_were_never_read(self):
    source = op_testing.GoldenGpuData()
    source.add(
        "case_a",
        op_testing.OpVariant.BASE,
        torch.float32,
        *self._make_sample("s0"),
    )
    encoded = source.to_plistlib_pytree()

    data = op_testing.GoldenGpuData()
    data.merge_plistlib_pytree(encoded)

    self.assertEqual(data.to_plistlib_pytree(), encoded)

  def test_add_lands_after_samples_that_were_merged_earlier(self):
    merged, added = self._make_sample("merged"), self._make_sample("added")
    data = self._merged(
        [("case_a", op_testing.OpVariant.BASE, torch.float32, merged)]
    )

    data.add("case_a", op_testing.OpVariant.BASE, torch.float32, *added)

    self._assert_samples_equal(
        data.get_samples("case_a", op_testing.OpVariant.BASE, torch.float32),
        [merged, added],
    )


if __name__ == "__main__":
  absltest.main()
