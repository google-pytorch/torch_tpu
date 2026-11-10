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

"""Unit tests for op_honey_trap."""

import json
import os
from unittest import mock
from absl.testing import absltest
import torch
from torch_tpu._internal.utils import test_utils
from tests import op_honey_trap
from tests import seed_test_utils


class OpHoneyTrapTest(seed_test_utils.RepeatableTest):

  def test_record_signatures(self):
    trap = op_honey_trap.HoneyTrapDispatchMode(
        mode="record", op_hint="add", target_device="cpu"
    )
    with trap:
      x = torch.tensor([1.0, 2.0], dtype=torch.float32)
      y = torch.tensor([3.0, 4.0], dtype=torch.float32)
      _ = torch.add(x, y)
      # Duplicate op to verify deduplication
      _ = torch.add(x, y)

    sigs = trap.get_deduped_signatures()
    self.assertNotEmpty(sigs)
    add_sigs = [s for s in sigs if "add" in s["op"]]
    self.assertLen(add_sigs, 1)
    self.assertEqual(add_sigs[0]["out_dtypes"], ["torch.float32"])

  def test_pause_honey_trap(self):
    trap = op_honey_trap.HoneyTrapDispatchMode(
        mode="record", op_hint="add", target_device="cpu"
    )
    with trap:
      with op_honey_trap.pause_honey_trap():
        x = torch.tensor([1.0, 2.0])
        y = torch.tensor([3.0, 4.0])
        _ = torch.add(x, y)

    sigs = trap.get_deduped_signatures()
    self.assertEmpty(sigs)

  def test_inject_condition_raise(self):
    spec = {
        "signatures": [{"op": "aten.add.Tensor"}],
        "rule": "CONDITION_RAISE",
    }
    trap = op_honey_trap.HoneyTrapDispatchMode(
        mode="inject",
        spec_json=json.dumps(spec),
        op_hint="add",
        target_device="cpu",
    )
    with trap:
      x = torch.tensor([1.0, 2.0])
      y = torch.tensor([3.0, 4.0])
      with self.assertRaisesRegex(  # ASSERT_RAISES_OK=harness_test
          RuntimeError, "HoneyTrap CONDITION_RAISE"
      ):
        _ = torch.add(x, y)

  def test_inject_zero_or_sentinel(self):
    spec = {
        "signatures": [{"op": "aten.add.Tensor"}],
        "rule": "ZERO_OR_SENTINEL",
    }
    trap = op_honey_trap.HoneyTrapDispatchMode(
        mode="inject",
        spec_json=json.dumps(spec),
        op_hint="add",
        target_device="cpu",
    )
    with trap:
      x = torch.tensor([10.0, 20.0])
      y = torch.tensor([30.0, 40.0])
      res = torch.add(x, y)
      test_utils.assert_close(res, torch.zeros_like(res))

  def test_inject_scale_or_offset(self):
    spec = {
        "signatures": [{"op": "aten.add.Tensor"}],
        "rule": "SCALE_OR_OFFSET",
    }
    trap = op_honey_trap.HoneyTrapDispatchMode(
        mode="inject",
        spec_json=json.dumps(spec),
        op_hint="add",
        target_device="cpu",
    )
    with trap:
      x = torch.tensor([10.0, 20.0])
      y = torch.tensor([30.0, 40.0])
      res = torch.add(x, y)
      # Normal output is [40.0, 60.0], scaled by 0.5 -> [20.0, 30.0]
      test_utils.assert_close(res, torch.tensor([20.0, 30.0]))

  def test_inject_tight_drift(self):
    spec = {
        "signatures": [{"op": "aten.add.Tensor"}],
        "rule": "TIGHT_DRIFT",
        "drift": 0.05,
    }
    trap = op_honey_trap.HoneyTrapDispatchMode(
        mode="inject",
        spec_json=json.dumps(spec),
        op_hint="add",
        target_device="cpu",
    )
    with trap:
      x = torch.tensor([1.0, 2.0], dtype=torch.float32)
      y = torch.tensor([3.0, 4.0], dtype=torch.float32)
      res = torch.add(x, y)
      test_utils.assert_close(
          res, torch.tensor([4.05, 6.05], dtype=torch.float32)
      )

  def test_inject_broken_writeback(self):
    spec = {
        "signatures": [{"op": "aten.add.out"}],
        "rule": "BROKEN_WRITEBACK",
    }
    trap = op_honey_trap.HoneyTrapDispatchMode(
        mode="inject",
        spec_json=json.dumps(spec),
        op_hint="add",
        target_device="cpu",
    )
    with trap:
      x = torch.tensor([1.0, 2.0], dtype=torch.float32)
      y = torch.tensor([3.0, 4.0], dtype=torch.float32)
      out = torch.empty(2, dtype=torch.float32)
      res = torch.add(x, y, out=out)
      test_utils.assert_close(out, torch.full_like(out, -999.0))
      test_utils.assert_close(res, torch.full_like(res, -999.0))

    # Non-aliased return with writeable buffer (_native_batch_norm_legit
    # updating running_mean/running_var) preserves clean output while
    # corrupting writeable buffers.
    bn_inp = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
    expected_bn, _, _ = torch.ops.aten._native_batch_norm_legit.default(
        bn_inp, None, None, torch.zeros(2), torch.ones(2), True, 0.1, 1e-5
    )
    bn_spec = {
        "signatures": [{"op": "aten._native_batch_norm_legit.default"}],
        "rule": "BROKEN_WRITEBACK",
    }
    bn_trap = op_honey_trap.HoneyTrapDispatchMode(
        mode="inject",
        spec_json=json.dumps(bn_spec),
        op_hint="_native_batch_norm_legit",
        target_device="cpu",
    )
    with bn_trap:
      running_mean = torch.zeros(2, dtype=torch.float32)
      running_var = torch.ones(2, dtype=torch.float32)
      actual_bn, _, _ = torch.ops.aten._native_batch_norm_legit.default(
          bn_inp, None, None, running_mean, running_var, True, 0.1, 1e-5
      )
      test_utils.assert_close(actual_bn, expected_bn)
      test_utils.assert_close(
          running_mean, torch.full_like(running_mean, -999.0)
      )
      test_utils.assert_close(running_var, torch.full_like(running_var, -999.0))

    with trap:

      # Non-matching op (aten.add.Tensor, not aten.add.out) should not be
      # corrupted
      x2 = torch.tensor([1.0, 2.0], dtype=torch.float32)
      y2 = torch.tensor([3.0, 4.0], dtype=torch.float32)
      res2 = torch.add(x2, y2)
      test_utils.assert_close(
          res2, torch.tensor([4.0, 6.0], dtype=torch.float32)
      )

  def test_inject_skip_out_resize(self):
    spec = {
        "signatures": [{"op": "aten.add.out"}],
        "rule": "SKIP_OUT_RESIZE",
    }
    trap = op_honey_trap.HoneyTrapDispatchMode(
        mode="inject",
        spec_json=json.dumps(spec),
        op_hint="add",
        target_device="cpu",
    )
    with trap:
      x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
      y = torch.tensor([[5.0, 6.0], [7.0, 8.0]])
      out = torch.zeros((1, 1), dtype=torch.float32)
      _ = torch.add(x, y, out=out)
      self.assertEqual(out.shape, (1, 1))

  def test_unmatched_signature_passes_cleanly(self):
    spec = {
        "signatures": [{"op": "aten.sub.Tensor"}],
        "rule": "CONDITION_RAISE",
    }
    trap = op_honey_trap.HoneyTrapDispatchMode(
        mode="inject",
        spec_json=json.dumps(spec),
        op_hint="add",
        target_device="cpu",
    )
    with trap:
      x = torch.tensor([1.0, 2.0])
      y = torch.tensor([3.0, 4.0])
      res = torch.add(x, y)
      test_utils.assert_close(res, torch.tensor([4.0, 6.0]))

  def test_void_returning_mutable_op(self):
    spec = {
        "signatures": [{"op": "aten._foreach_add_.List"}],
        "rule": "ZERO_OR_SENTINEL",
    }
    trap = op_honey_trap.HoneyTrapDispatchMode(
        mode="inject",
        spec_json=json.dumps(spec),
        op_hint="_foreach_add_",
        target_device="cpu",
    )
    with trap:
      x = torch.tensor([10.0, 20.0])
      y = torch.tensor([1.0, 2.0])
      res = torch._foreach_add_([x], [y])
      self.assertEqual(len(res), 1)
      test_utils.assert_close(res[0], torch.zeros_like(x))
      test_utils.assert_close(x, torch.zeros_like(x))

    spec2 = {
        "signatures": [{"op": "aten._foreach_add_.List"}],
        "rule": "SCALE_OR_OFFSET",
    }
    trap2 = op_honey_trap.HoneyTrapDispatchMode(
        mode="inject",
        spec_json=json.dumps(spec2),
        op_hint="_foreach_add_",
        target_device="cpu",
    )
    with trap2:
      x2 = torch.tensor([10.0, 20.0])
      y2 = torch.tensor([1.0, 2.0])
      res2 = torch._foreach_add_([x2], [y2])
      self.assertEqual(len(res2), 1)
      test_utils.assert_close(res2[0], torch.tensor([5.5, 11.0]))
      test_utils.assert_close(x2, torch.tensor([5.5, 11.0]))

    # Directly test void-returning op support (result is None)
    z = torch.tensor([5.0, 6.0])
    void_res = op_honey_trap._apply_value_corruption(
        None,
        op_honey_trap._corrupt_zero_or_sentinel,
        [{"tensor": z}],
    )
    self.assertIsNone(void_res)
    test_utils.assert_close(z, torch.zeros_like(z))

  def test_multi_output_tuple_op(self):
    spec = {
        "signatures": [{"op": "aten.topk.default"}],
        "rule": "ZERO_OR_SENTINEL",
        "output_index": 0,
    }
    trap = op_honey_trap.HoneyTrapDispatchMode(
        mode="inject",
        spec_json=json.dumps(spec),
        op_hint="topk",
        target_device="cpu",
    )
    with trap:
      x = torch.tensor([10.0, 50.0, 20.0])
      vals, inds = torch.topk(x, k=2)
      test_utils.assert_close(vals, torch.zeros_like(vals))
      test_utils.assert_close(inds, torch.tensor([1, 2]))

    spec_qr = {
        "signatures": [{"op": "aten.linalg_qr.default"}],
        "rule": "SCALE_OR_OFFSET",
    }
    trap_qr = op_honey_trap.HoneyTrapDispatchMode(
        mode="inject",
        spec_json=json.dumps(spec_qr),
        op_hint="linalg_qr",
        target_device="cpu",
    )
    with trap_qr:
      a = torch.eye(3, dtype=torch.float32)
      q, r = torch.linalg.qr(a)
      test_utils.assert_close(q, torch.eye(3) * 0.5)
      test_utils.assert_close(r, torch.eye(3) * 0.5)

  def test_non_finite_and_unsigned_tensors(self):
    t_nan_inf = torch.tensor([float("nan"), float("inf"), 10.0])
    corrupted_scale = op_honey_trap._corrupt_scale_or_offset(t_nan_inf)
    test_utils.assert_close(corrupted_scale, torch.tensor([0.0, 0.0, 5.0]))

    corrupted_drift = op_honey_trap._corrupt_tight_drift(t_nan_inf, drift=0.1)
    test_utils.assert_close(corrupted_drift, torch.tensor([0.0, 0.0, 10.1]))

    t_u8 = torch.tensor([0, 5], dtype=torch.uint8)
    corrupted_u8 = op_honey_trap._corrupt_zero_or_sentinel(t_u8)
    test_utils.assert_close(corrupted_u8, torch.zeros_like(t_u8))

    t_u16 = torch.tensor([0, 0], dtype=torch.uint16)
    corrupted_u16 = op_honey_trap._corrupt_zero_or_sentinel(t_u16)
    test_utils.assert_close(corrupted_u16, torch.full_like(t_u16, 99))

  def test_signature_matches(self):
    target = {
        "op": "aten.add.Tensor",
        "read_dtypes": ["torch.float32"],
        "dtype_kwargs": {"dtype": "None"},
    }
    call = {
        "op": "aten.add.Tensor",
        "read_dtypes": ["torch.float32"],
        "dtype_kwargs": {"dtype": "None"},
        "out_dtypes": ["torch.float32"],
    }
    self.assertTrue(op_honey_trap.signature_matches(target, call))
    self.assertFalse(
        op_honey_trap.signature_matches({"op": "aten.sub.Tensor"}, call)
    )

  def test_comma_separated_op_hint(self):
    trap = op_honey_trap.HoneyTrapDispatchMode(
        mode="record", op_hint="add,sum", target_device="cpu"
    )
    with trap:
      x = torch.tensor([1.0, 2.0])
      _ = torch.sum(x)
    sigs = trap.get_deduped_signatures()
    self.assertTrue(any("sum" in s["op"] for s in sigs))

  def test_invalid_spec_json_raises(self):
    with self.assertRaises(ValueError):  # ASSERT_RAISES_OK=harness_test
      _ = op_honey_trap.HoneyTrapDispatchMode(
          mode="inject", spec_json="{invalid_json"
      )

  def test_create_honey_trap(self):
    with mock.patch.dict(os.environ, {"TT_HONEY_TRAP_MODE": "record"}):
      trap = op_honey_trap.create_honey_trap(test_id="test_abc")
      self.assertIsNotNone(trap)
      self.assertEqual(trap._mode, "record")

    with mock.patch.dict(os.environ, {"TT_HONEY_TRAP_MODE": ""}):
      self.assertIsNone(op_honey_trap.create_honey_trap())


if __name__ == "__main__":
  absltest.main()
