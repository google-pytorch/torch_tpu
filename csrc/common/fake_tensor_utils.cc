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

#include "ATen/core/TensorBody.h"
#include "c10/core/DispatchKey.h"
#include "c10/core/DispatchKeySet.h"
#include "c10/core/StorageImpl.h"
#include "c10/core/TensorImpl.h"
#include "c10/core/impl/LocalDispatchKeySet.h"
#include "c10/core/impl/TorchDispatchModeTLS.h"
#include "c10/util/Exception.h"
#include "csrc/common/const_tensor.h"
#include "torch/headeronly/core/DeviceType.h"

namespace torch_tpu {
namespace {

template <typename StorageImpl>
bool HasThrowOnMutableDataPtr(StorageImpl& storage) {
  if constexpr (requires { storage.throw_on_mutable_data_ptr(); }) {
    return storage.throw_on_mutable_data_ptr();
  } else {
    // Prior to PyTorch 2.16, `StorageImpl::throw_on_mutable_data_ptr_` had no
    // public getter. Distinguish `throw_on_mutable_data_ptr_` from
    // `throw_on_immutable_data_ptr_` by checking that `data()` succeeds while
    // `mutable_data()` raises `c10::Error`. Only reached on
    // `DispatchKey::Python` tensors with non-Meta storage while `FUNCTIONAL`
    // mode is active.
    try {
      storage.data();
    } catch (const c10::Error&) {
      return false;
    }
    try {
      storage.mutable_data();
      return false;
    } catch (const c10::Error&) {
      return true;
    }
  }
}

}  // namespace

bool IsFunctorchActive() {
  const c10::DispatchKeySet include_set =
      c10::impl::tls_local_dispatch_key_set().included_;
  return include_set.has(c10::DispatchKey::FuncTorchDynamicLayerFrontMode) ||
         include_set.has(c10::DispatchKey::FuncTorchDynamicLayerBackMode);
}

bool IsFakeOrFunctorchTensor(const ConstTensor& tensor) {
  return IsFunctorchActive() || !tensor.has_storage_data();
}

bool IsMetaOrFakeOrFunctionalTensor(const ConstTensor& tensor) {
  if (!tensor.defined()) {
    return false;
  }
  if (tensor.is_meta()) {
    return true;
  }
  if (!tensor.key_set().has(c10::DispatchKey::Python) ||
      !tensor.has_storage()) {
    return false;
  }
  const at::Tensor raw = tensor.UnsafeToTensor();
  c10::StorageImpl* storage =
      raw.unsafeGetTensorImpl()->unsafe_storage().unsafeGetStorageImpl();
  if (storage->device_type() == c10::DeviceType::Meta) {
    return true;
  }
  return c10::impl::TorchDispatchModeTLS::get_mode(
             c10::impl::TorchDispatchModeKey::FUNCTIONAL)
             .has_value() &&
         HasThrowOnMutableDataPtr(*storage);
}

bool IsPythonWrapperSubclass(const ConstTensor& tensor) {
  return tensor.defined() && tensor.key_set().has(c10::DispatchKey::Python) &&
         !IsMetaOrFakeOrFunctionalTensor(tensor);
}

}  // namespace torch_tpu
