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

"""Unit tests for MLIR tensor shape and signature formatting utilities."""

from absl.testing import absltest
from tests import seed_test_utils
import torch
import torch_tpu  # pylint: disable=unused-import
from examples.benchmarks.shape_utils import format_shape_signature
from examples.benchmarks.shape_utils import format_tensor
from examples.benchmarks.shape_utils import format_tensor_spec
from examples.benchmarks.shape_utils import shorten_dtype_name


class ShapeUtilsTest(seed_test_utils.RepeatableTest):

  def test_shorten_dtype_name(self):
    self.assertEqual(shorten_dtype_name(torch.float32), "f32")
    self.assertEqual(shorten_dtype_name(torch.bfloat16), "bf16")
    self.assertEqual(shorten_dtype_name(torch.int64), "i64")
    self.assertEqual(shorten_dtype_name(torch.uint32), "ui32")
    self.assertEqual(shorten_dtype_name(torch.bool), "i1")
    self.assertEqual(shorten_dtype_name(torch.complex64), "complex<f32>")
    self.assertEqual(shorten_dtype_name(torch.complex128), "complex<f64>")
    self.assertEqual(shorten_dtype_name("torch.float16"), "f16")
    self.assertEqual(shorten_dtype_name("double"), "f64")

  def test_format_tensor_spec(self):
    self.assertEqual(
        format_tensor_spec([2048, 2048], torch.bfloat16), "<2048x2048xbf16>"
    )
    self.assertEqual(format_tensor_spec([100], torch.float32), "<100xf32>")
    self.assertEqual(format_tensor_spec([], torch.float32), "<f32>")
    self.assertEqual(format_tensor_spec([2, 3, 4], torch.int64), "<2x3x4xi64>")
    self.assertEqual(
        format_tensor_spec(["b", "s", 128], torch.float32), "<BxSx128xf32>"
    )

  def test_format_tensor(self):
    t2d = torch.empty((2048, 2048), dtype=torch.bfloat16)
    self.assertEqual(format_tensor(t2d), "<2048x2048xbf16>")

    t1d = torch.empty((100,), dtype=torch.float32)
    self.assertEqual(format_tensor(t1d), "<100xf32>")

    t0d = torch.empty((), dtype=torch.float32)
    self.assertEqual(format_tensor(t0d), "<f32>")

    tc = torch.empty((2, 4), dtype=torch.complex64)
    self.assertEqual(format_tensor(tc), "<2x4xcomplex<f32>>")

  def test_format_shape_signature_single_tensor(self):
    inputs_str = "([T([2048, 2048], torch.bfloat16)], {})"
    self.assertEqual(format_shape_signature(inputs_str), "<2048x2048xbf16>")

  def test_format_shape_signature_multiple_tensors(self):
    inputs_str = (
        "([T([4096, 2048], torch.bfloat16), T([2048, 2048], torch.bfloat16)],"
        " {})"
    )
    self.assertEqual(
        format_shape_signature(inputs_str),
        "<4096x2048xbf16>, <2048x2048xbf16>",
    )

  def test_format_shape_signature_mixed_args_and_kwargs(self):
    inputs_str = "([T([2, 3], torch.float32), 1.0], {'dim': 0})"
    self.assertEqual(
        format_shape_signature(inputs_str),
        "<2x3xf32>, 1.0, dim=0",
    )

  def test_format_shape_signature_extended_dtypes(self):
    test_cases = [
        ("([T([2, 4], torch.complex64)], {})", "<2x4xcomplex<f32>>"),
        ("([T([8, 16], torch.complex128)], {})", "<8x16xcomplex<f64>>"),
        ("([T([4], torch.complex32)], {})", "<4xcomplex<f16>>"),
        ("([T([32], torch.uint16)], {})", "<32xui16>"),
        ("([T([32], torch.uint32)], {})", "<32xui32>"),
        ("([T([32], torch.uint64)], {})", "<32xui64>"),
        ("([T([16], torch.float8_e4m3fn)], {})", "<16xf8E4M3FN>"),
        ("([T([16], torch.float8_e5m2)], {})", "<16xf8E5M2>"),
        ("([T([64], torch.qint8)], {})", "<64xqint8>"),
        ("([T([64], torch.quint8)], {})", "<64xquint8>"),
        ("([T([64], torch.qint32)], {})", "<64xqint32>"),
    ]
    for inputs_str, expected in test_cases:
      self.assertEqual(format_shape_signature(inputs_str), expected)

  def test_format_shape_signature_dynamic_shapes(self):
    inputs_str = "([T([b, s], torch.float32), T([s, d], torch.bfloat16)], {})"
    self.assertEqual(
        format_shape_signature(inputs_str),
        "<BxSxf32>, <SxDxbf16>",
    )

  def test_format_shape_signature_fallback_on_invalid(self):
    invalid_str = "invalid syntax"
    self.assertEqual(format_shape_signature(invalid_str), invalid_str)


if __name__ == "__main__":
  absltest.main()
