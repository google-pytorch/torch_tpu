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

#include "csrc/common/fake_tensor_utils.h"

#include <memory>
#include <optional>
#include <utility>

#include "ATen/core/TensorBody.h"
#include "ATen/ops/empty.h"
#include "ATen/ops/ones.h"
#include "absl/cleanup/cleanup.h"
#include "c10/core/Allocator.h"
#include "c10/core/Device.h"
#include "c10/core/DispatchKey.h"
#include "c10/core/DispatchKeySet.h"
#include "c10/core/ScalarType.h"
#include "c10/core/Storage.h"
#include "c10/core/StorageImpl.h"
#include "c10/core/TensorImpl.h"
#include "c10/core/impl/LocalDispatchKeySet.h"
#include "c10/core/impl/TorchDispatchModeTLS.h"
#include "c10/util/Exception.h"
#include "c10/util/intrusive_ptr.h"
#include "gtest/gtest.h"
#include "torch/headeronly/core/DeviceType.h"

namespace torch_tpu {
namespace {

// Constructs a synthetic `DispatchKey::Python` wrapper tensor matching the
// storage layout created by `THPVariable_make_wrapper_subclass` (null data
// pointer with `storage_device`).
at::Tensor MakeSyntheticPythonSubclassTensor(c10::Device storage_device,
                                             bool throw_on_mutable_data_ptr) {
  c10::Storage storage(c10::Storage::use_byte_size_t{}, /*size_bytes=*/24,
                       at::DataPtr{nullptr, storage_device},
                       /*allocator=*/c10::GetAllocator(c10::kMeta),
                       /*resizable=*/true);
  if (throw_on_mutable_data_ptr) {
    storage.unsafeGetStorageImpl()->set_throw_on_mutable_data_ptr();
  }
  at::Tensor tensor = at::detail::make_tensor<c10::TensorImpl>(
      std::move(storage), c10::DispatchKeySet(c10::DispatchKey::CPU),
      c10::scalarTypeToTypeMeta(c10::ScalarType::Float));
  tensor.unsafeGetTensorImpl()->set_python_dispatch(true);
  tensor.unsafeGetTensorImpl()->set_sizes_contiguous({2, 3});
  return tensor;
}

TEST(IsFakeOrFunctorchTensorTest, RealTensor) {
  EXPECT_FALSE(IsFakeOrFunctorchTensor(at::ones({2, 3})));
}

TEST(IsFakeOrFunctorchTensorTest, MetaTensor) {
  EXPECT_TRUE(IsFakeOrFunctorchTensor(at::empty({2, 3}, at::kMeta)));
}

TEST(IsFakeOrFunctorchTensorTest, UndefinedTensor) {
  EXPECT_TRUE(IsFakeOrFunctorchTensor(at::Tensor()));
}

TEST(IsFakeOrFunctorchTensorTest, ZeroElementTensor) {
  // Real tensors without any bytes have a null data pointer.
  EXPECT_TRUE(IsFakeOrFunctorchTensor(at::empty({0})));
}

TEST(IsFakeOrFunctorchTensorTest, DataPtrAccessThrows) {
  // A storage whose data pointer access throws used to make this throw. Such
  // storages (e.g. of real tensors whose memory was reclaimed by CUDA graph
  // trees) still hold real data, so they are not reported as fake.
  const at::Tensor tensor = at::ones({2, 3});
  tensor.unsafeGetTensorImpl()
      ->unsafe_storage()
      .unsafeGetStorageImpl()
      ->release_data_and_set_meta_custom_data_ptr_error_msg_(std::nullopt);
  ASSERT_THROW(tensor.storage().data_ptr(), c10::Error);

  EXPECT_FALSE(IsFakeOrFunctorchTensor(tensor));
}

TEST(IsFakeOrFunctorchTensorTest, FunctorchActive) {
  // Create the tensor first: running ops with the functorch keys included but
  // no functorch interpreter on the stack fails.
  const at::Tensor tensor = at::ones({2, 3});
  EXPECT_FALSE(IsFunctorchActive());
  {
    const c10::impl::IncludeDispatchKeyGuard guard(
        c10::DispatchKey::FuncTorchDynamicLayerFrontMode);
    EXPECT_TRUE(IsFunctorchActive());
    EXPECT_TRUE(IsFakeOrFunctorchTensor(tensor));
  }
  {
    const c10::impl::IncludeDispatchKeyGuard guard(
        c10::DispatchKey::FuncTorchDynamicLayerBackMode);
    EXPECT_TRUE(IsFunctorchActive());
    EXPECT_TRUE(IsFakeOrFunctorchTensor(tensor));
  }
}

TEST(IsPythonWrapperSubclassTest, UndefinedAndPlainAndMetaTensors) {
  EXPECT_FALSE(IsMetaOrFakeOrFunctionalTensor(at::Tensor()));
  EXPECT_FALSE(IsPythonWrapperSubclass(at::Tensor()));

  const at::Tensor cpu_tensor = at::ones({2, 3});
  EXPECT_FALSE(IsMetaOrFakeOrFunctionalTensor(cpu_tensor));
  EXPECT_FALSE(IsPythonWrapperSubclass(cpu_tensor));

  const at::Tensor meta_tensor = at::empty({2, 3}, at::kMeta);
  EXPECT_TRUE(IsMetaOrFakeOrFunctionalTensor(meta_tensor));
  EXPECT_FALSE(IsPythonWrapperSubclass(meta_tensor));
}

TEST(IsPythonWrapperSubclassTest, FakeTensorWithMetaStorage) {
  const at::Tensor fake_like = MakeSyntheticPythonSubclassTensor(
      c10::Device(c10::DeviceType::Meta), /*throw_on_mutable_data_ptr=*/true);
  EXPECT_TRUE(IsMetaOrFakeOrFunctionalTensor(fake_like));
  EXPECT_FALSE(IsPythonWrapperSubclass(fake_like));
}

TEST(IsPythonWrapperSubclassTest, FunctionalTensorVsUserWrapperSubclass) {
  // Mirrors AOTAutograd `FunctionalTensor` (`PrivateUse1` storage +
  // `throw_on_mutable_data_ptr == true`) vs a user/distributed wrapper
  // subclass such as `DTensor` (`PrivateUse1` storage +
  // `throw_on_mutable_data_ptr == false`).
  const at::Tensor functional_like = MakeSyntheticPythonSubclassTensor(
      c10::Device(c10::DeviceType::PrivateUse1, 0),
      /*throw_on_mutable_data_ptr=*/true);
  const at::Tensor dtensor_like = MakeSyntheticPythonSubclassTensor(
      c10::Device(c10::DeviceType::PrivateUse1, 0),
      /*throw_on_mutable_data_ptr=*/false);

  // Outside `FunctionalTensorMode`, both are treated as Python wrapper
  // subclasses.
  EXPECT_FALSE(IsMetaOrFakeOrFunctionalTensor(functional_like));
  EXPECT_TRUE(IsPythonWrapperSubclass(functional_like));
  EXPECT_FALSE(IsMetaOrFakeOrFunctionalTensor(dtensor_like));
  EXPECT_TRUE(IsPythonWrapperSubclass(dtensor_like));

  // Inside `FunctionalTensorMode` (`TorchDispatchModeKey::FUNCTIONAL` active),
  // `functional_like` is recognized as an AOTAutograd `FunctionalTensor`, while
  // `dtensor_like` (e.g. re-wrapped by `aot_dispatch_subclass`) remains a
  // Python wrapper subclass.
  c10::impl::TorchDispatchModeTLS::set_mode(
      std::shared_ptr<c10::impl::PyObject_TorchDispatchMode>(nullptr),
      c10::impl::TorchDispatchModeKey::FUNCTIONAL);
  absl::Cleanup restore_mode = [] {
    c10::impl::TorchDispatchModeTLS::unset_mode(
        c10::impl::TorchDispatchModeKey::FUNCTIONAL);
  };

  EXPECT_TRUE(IsMetaOrFakeOrFunctionalTensor(functional_like));
  EXPECT_FALSE(IsPythonWrapperSubclass(functional_like));
  EXPECT_FALSE(IsMetaOrFakeOrFunctionalTensor(dtensor_like));
  EXPECT_TRUE(IsPythonWrapperSubclass(dtensor_like));
}

}  // namespace
}  // namespace torch_tpu
