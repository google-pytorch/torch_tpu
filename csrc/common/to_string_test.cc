/*
 * Copyright 2026 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "csrc/common/to_string.h"

#include <optional>
#include <string>
#include <vector>

#include "ATen/SparseTensorImpl.h"
#include "ATen/core/ATen_fwd.h"
#include "ATen/core/TensorBody.h"
#include "ATen/ops/_nested_tensor_from_tensor_list.h"
#include "ATen/ops/arange.h"
#include "ATen/ops/empty.h"
#include "ATen/ops/ones.h"
#include "ATen/ops/sparse_coo_tensor.h"
#include "ATen/ops/sparse_csr_tensor.h"
#include "ATen/ops/zeros.h"
#include "absl/strings/str_cat.h"
#include "c10/core/ScalarType.h"
#include "c10/core/SymInt.h"
#include "csrc/common/fake_symbol.h"
#include "csrc/common/macro_utils.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "torch/headeronly/core/ScalarType.h"

namespace torch_tpu {
namespace {

using testing::HasSubstr;

TEST(ToStringTensorTest, UndefinedTensor) {
  EXPECT_EQ(ToString(at::Tensor()), "undefined");
  EXPECT_EQ(ToString(at::Tensor(), "x"), "x: undefined");
}

TEST(ToStringTensorTest, StaticShape) {
  const std::string result = ToString(at::empty({2, 3}), "x");

  EXPECT_EQ(result,
            "x: shape=[2, 3], strides=[3, 1], numel=6, dtype=float32, "
            "device=cpu, is_contiguous=1, is_cpu=1, is_cuda=0, is_meta=0, "
            "has_storage=1, storage_offset=0");
}

TEST(ToStringTensorTest, SymbolicShapeIsPrintedWithoutSpecializing) {
#if defined(__has_feature) && __has_feature(hwaddress_sanitizer)
  GTEST_SKIP() << "c10::SymInt pointer packing is incompatible with HWASAN.";
#endif
  int num_specializations = 0;
  const at::Tensor tensor = MakeSymbolicTensor(&num_specializations);
  ASSERT_TRUE(tensor.unsafeGetTensorImpl()->has_symbolic_sizes_strides());

  const std::string result = ToString(tensor);

  // Contiguity of a symbolic tensor would be a guarded expression.
  EXPECT_EQ(result,
            "shape=[2, s0], strides=[s0, 1], numel=2*s0, dtype=float32, "
            "device=cpu, is_cpu=1, is_cuda=0, is_meta=0, has_storage=1, "
            "storage_offset=0");
  EXPECT_EQ(num_specializations, 0);
}

TEST(ToStringTensorTest, SparseCooTensor) {
  // Three uncoalesced entries at (0, 0) of a 2x2 matrix with 4-element values.
  const at::Tensor tensor = at::sparse_coo_tensor(at::zeros({2, 3}, at::kLong),
                                                  at::ones({3, 4}), {2, 2, 4});

  EXPECT_EQ(ToString(tensor),
            "shape=[2, 2, 4], layout=Sparse, numel=16, dtype=float32, "
            "device=cpu, sparse_dim=2, dense_dim=1, nnz=3, is_cpu=1, "
            "is_cuda=0, is_meta=0, has_storage=0");
}

TEST(ToStringTensorTest, SparseCooTensorWithSymbolicNnzDoesNotSpecialize) {
#if defined(__has_feature) && __has_feature(hwaddress_sanitizer)
  GTEST_SKIP() << "c10::SymInt pointer packing is incompatible with HWASAN.";
#endif
  int num_specializations = 0;
  const at::Tensor tensor = at::sparse_coo_tensor(at::zeros({2, 3}, at::kLong),
                                                  at::ones({3}), {2, 2});
  // Mirrors a FakeTensor whose number of specified elements is dynamic.
  // at::Tensor::_values() returns an alias, so modify the impl's own values.
  const auto* sparse_impl =
      dynamic_cast<const at::SparseTensorImpl*>(tensor.unsafeGetTensorImpl());
  ASSERT_NE(sparse_impl, nullptr);
  sparse_impl->values().unsafeGetTensorImpl()->set_sizes_and_strides(
      std::vector<c10::SymInt>{MakeSymbol(3, "u0", &num_specializations)},
      std::vector<c10::SymInt>{c10::SymInt(1)});

  EXPECT_THAT(ToString(tensor), HasSubstr(", nnz=u0, "));
  EXPECT_EQ(num_specializations, 0);
}

TEST(ToStringTensorTest, SparseCsrTensor) {
  // A 2x2 matrix with one specified element per row, both in column 0.
  const at::Tensor tensor =
      at::sparse_csr_tensor(at::arange(3, at::kLong), at::zeros({2}, at::kLong),
                            at::ones({2}), {2, 2}, at::TensorOptions());

  // Sparse compressed tensors have no strides.
  EXPECT_EQ(ToString(tensor),
            "shape=[2, 2], layout=SparseCsr, numel=4, dtype=float32, "
            "device=cpu, sparse_dim=2, dense_dim=0, nnz=2, is_cpu=1, "
            "is_cuda=0, is_meta=0, has_storage=0");
}

TEST(ToStringTensorTest, NestedTensor) {
  const at::Tensor tensor =
      at::_nested_tensor_from_tensor_list({at::ones({2, 3}), at::ones({4, 3})});

  // Nested tensors have neither sizes nor strides.
  EXPECT_EQ(ToString(tensor),
            "is_nested=1, numel=18, dtype=float32, device=cpu, "
            "is_cpu=1, is_cuda=0, is_meta=0, has_storage=1, storage_offset=0");
}

TEST(ToStringTensorTest, ContainersOfTensorsUseTheTensorOverload) {
  const at::Tensor tensor = at::empty({2});
  const std::string expected = ToString(tensor);

  EXPECT_EQ(ToString(std::optional<at::Tensor>(tensor)),
            absl::StrCat("<", expected, ">"));
  EXPECT_EQ(ToString(std::vector<at::Tensor>{tensor, at::Tensor()}),
            absl::StrCat("[", expected, ", undefined]"));
  EXPECT_EQ(ToString(at::TensorList({tensor})),
            absl::StrCat("[", expected, "]"));
  EXPECT_EQ(ToString(at::ITensorListRef({tensor})),
            absl::StrCat("[", expected, "]"));
}

TEST(ToStringScalarTypeTest, ScalarTypes) {
  EXPECT_EQ(ToString(at::ScalarType::Float), "float32");
  EXPECT_EQ(ToString(at::ScalarType::BFloat16), "bfloat16");
  EXPECT_EQ(ToString(at::ScalarType::Float8_e8m0fnu), "float8_e8m0fnu");
#if TT_HAS_FLOAT8_E5M3FNU
  EXPECT_EQ(ToString(at::ScalarType::Float8_e5m3fnu), "float8_e5m3fnu");
#endif
}

}  // namespace
}  // namespace torch_tpu
