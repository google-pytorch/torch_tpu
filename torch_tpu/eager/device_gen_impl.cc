/*
 * Copyright 2025 Google LLC
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

#include "torch_tpu/eager/device_gen_impl.h"

#include <cstdint>
#include <deque>
#include <optional>
#include <utility>
#include <vector>

#include "absl/log/absl_check.h"
#include "absl/log/absl_log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Types.h"
#include "ATen/core/ATen_fwd.h"
#include "ATen/core/Generator.h"
#include "ATen/ops/full.h"
#include "c10/core/Device.h"
#include "c10/core/DispatchKey.h"
#include "c10/core/DispatchKeySet.h"
#include "c10/core/GeneratorImpl.h"
#include "c10/core/ScalarType.h"
#include "c10/core/TensorImpl.h"
#include "c10/core/impl/DeviceGuardImplInterface.h"
#include "c10/core/impl/LocalDispatchKeySet.h"
#include "c10/util/ArrayRef.h"
#include "c10/util/CallOnce.h"
#include "c10/util/intrusive_ptr.h"
#include "torch/headeronly/core/DeviceType.h"
#include "torch/headeronly/core/Layout.h"
#include "torch_tpu/common/cache_key.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/eager/device_buffer.h"
#include "torch_tpu/eager/op_dispatcher.h"
#include "torch_tpu/eager/tensor_to_buffer.h"
#include "torch_tpu/ops/macros/kernel.h"
#include "torch_tpu/ops/op_builder_utils.h"
#include "torch_tpu/ops/op_names.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"

namespace torch_tpu {

namespace {

absl::StatusOr<DeviceBufferRef> UpdateDeviceRngState(at::Tensor rng_state,
                                                     uint64_t value,
                                                     int64_t position) {
  ABSL_VLOG(3) << "[UpdateDeviceRngState] rng_state: "
               << ", value: " << value << ", position: " << position;
  ABSL_CHECK(position == 0 || position == 1)  // CRASH_OK
      << "Position must be 0 or 1, got " << position;

  auto set_seed_builder =
      [value, position](mlir::MlirOp input) -> absl::StatusOr<mlir::MlirOp> {
    auto& builder = input.getBuilder();
    auto& op_builder = builder.getOpBuilder();

    mlir::Type ui64_type = op_builder.getIntegerType(64, /*isSigned=*/false);
    mlir::RankedTensorType tensor_ui64_type =
        mlir::RankedTensorType::get({1}, ui64_type);
    mlir::DenseElementsAttr value_attr = mlir::DenseElementsAttr::get(
        tensor_ui64_type, llvm::ArrayRef<uint64_t>({value}));
    mlir::MlirOp value_op = mlir::stablehlo::Constant(builder, value_attr);

    mlir::MlirOp start_indices =
        MakeScalarConstant(builder, position, op_builder.getI64Type());
    return mlir::stablehlo::DynamicUpdateSlice(input, value_op, start_indices);
  };

  TT_ASSIGN_OR_RETURN(auto params, TT_MAKE_OP_PARAM_CACHE_KEYS(value));
  return DispatchOp<1>(
      std::move(set_seed_builder), {rng_state},
      // Override the op name as this is a subroutine rather than a top-level
      // op.
      {.op_name = position == 0 ? OpName::kRngSetSeed : OpName::kRngSetOffset,
       .out_dtype = mlir::ElementType::UI64,
       .out_dims = {2},
       .op_param_cache_keys = std::move(params)});
}

absl::StatusOr<DeviceBufferRef> UpdateDeviceRngSeed(at::Tensor rng_state,
                                                    uint64_t value) {
  return UpdateDeviceRngState(rng_state, value, /*position=*/0);
}

absl::StatusOr<DeviceBufferRef> UpdateDeviceRngOffset(at::Tensor rng_state,
                                                      uint64_t value) {
  return UpdateDeviceRngState(rng_state, value, /*position=*/1);
}

at::Tensor CreateDeviceRngStateTensor(c10::Device device) {
  // Exclude the Python dispatch key to prevent FakeTensorMode and other
  // Python-level dispatch modes from intercepting internal eager tensor
  // allocations and operations in this scope.
  c10::impl::ExcludeDispatchKeyGuard guard(c10::DispatchKey::Python);
  return at::full(/*size=*/{2}, /*fill_value=*/0, /*dtype_opt=*/at::kUInt64,
                  /*layout_opt=*/std::nullopt, /*device_opt=*/device,
                  /*pin_memory_opt=*/std::nullopt);
}

// A singleton that holds one generator per device. The generators are lazily
// initialized.
class DeviceGenerators {
 public:
  // This class is neither copyable nor movable.
  DeviceGenerators(const DeviceGenerators&) = delete;
  DeviceGenerators& operator=(const DeviceGenerators&) = delete;
  DeviceGenerators(DeviceGenerators&&) = delete;
  DeviceGenerators& operator=(DeviceGenerators&&) = delete;

  static DeviceGenerators& GetDefaultInstance();

  // Returns the default generator for the given device index. If idx is -1,
  // returns the default generator for the current device.
  at::Generator& GetDefaultGenerator(c10::DeviceIndex idx = -1);

  // Creates a new generator for the given device index. If idx is -1, creates a
  // new generator for the current device.
  at::Generator CreateGenerator(c10::DeviceIndex idx = -1) const;

  int64_t num_devices() const { return num_devices_; }

 private:
  DeviceGenerators();
  std::vector<at::Generator> generators_;
  std::deque<c10::once_flag> generator_init_flags_;
  int64_t num_devices_;
};

}  // namespace

DeviceGenerators& DeviceGenerators::GetDefaultInstance() {
  // We cannot use absl::NoDestructor here because the constructor is private.
  static auto* const kInstance = new DeviceGenerators();
  return *kInstance;
}

DeviceGenerators::DeviceGenerators() {
  const auto* guard =
      c10::impl::getDeviceGuardImpl(c10::DeviceType::PrivateUse1);
  TT_CHECK_THROW(guard != nullptr, error::kFailedPrecondition)
      << "TPU device guard is not registered.";
  num_devices_ = static_cast<int32_t>(guard->deviceCount());
  generators_.resize(num_devices_);
  generator_init_flags_.resize(num_devices_);
}

at::Generator& DeviceGenerators::GetDefaultGenerator(at::DeviceIndex idx) {
  ABSL_VLOG(1) << "[DeviceGenerators::GetDefaultGenerator] idx: "
               << static_cast<int32_t>(idx);
  if (idx == -1) {
    const auto* guard =
        c10::impl::getDeviceGuardImpl(c10::DeviceType::PrivateUse1);
    TT_CHECK_THROW(guard != nullptr, error::kFailedPrecondition)
        << "PrivateUse1 device guard is not registered.";
    idx = guard->getDevice().index();
  } else {
    TT_CHECK_THROW(idx >= 0 && idx < num_devices_, error::kFailedPrecondition)
        << "The device_index is invalid, expected an index between 0 and"
        << num_devices_ - 1 << " got " << static_cast<int32_t>(idx);
  }
  c10::call_once(generator_init_flags_[idx], [&] {
    generators_[idx] = at::make_generator<DeviceGeneratorImpl>(idx);
    generators_[idx].seed();
  });
  return generators_[idx];
}

at::Generator DeviceGenerators::CreateGenerator(c10::DeviceIndex idx) const {
  ABSL_VLOG(1) << "[CreateGenerator] idx: " << static_cast<int32_t>(idx);
  if (idx == -1) {
    const auto* guard =
        c10::impl::getDeviceGuardImpl(c10::DeviceType::PrivateUse1);
    TT_CHECK_THROW(guard != nullptr, error::kFailedPrecondition)
        << "PrivateUse1 device guard is not registered.";
    idx = guard->getDevice().index();
  }
  TT_CHECK_THROW(idx >= 0 && idx < num_devices_, error::kFailedPrecondition)
      << "The device_index is invalid, expected an index between 0 and"
      << num_devices_ - 1 << " got " << static_cast<int32_t>(idx);
  return at::make_generator<DeviceGeneratorImpl>(idx);
}

/*static*/ c10::DeviceType DeviceGeneratorImpl::device_type() {
  return c10::DeviceType::PrivateUse1;
}

at::Tensor DeviceGeneratorImpl::DeviceStateTensor() const {
  return state_->device_state_tensor;
}

absl::Status DeviceGeneratorImpl::SetDeviceStateTensor(
    at::Tensor device_state_tensor) {
  TT_RETURN_IF_ERROR(CheckDeviceStateTensor(device_state_tensor));
  state_->device_state_tensor = std::move(device_state_tensor);
  return absl::OkStatus();
}

DeviceGeneratorImpl::DeviceGeneratorImpl(c10::DeviceIndex device_index)
    : c10::GeneratorImpl(
          c10::Device(c10::DeviceType::PrivateUse1, device_index),
          c10::DispatchKeySet(c10::DispatchKey::PrivateUse1)) {
  state_ = c10::make_intrusive<DeviceGeneratorState>(
      CreateDeviceRngStateTensor(this->device()));
}

DeviceGeneratorImpl::DeviceGeneratorImpl(c10::DeviceIndex device_index,
                                         at::Tensor rng_state)
    : DeviceGeneratorImpl(device_index) {
  if (this->CheckDeviceStateTensor(rng_state).ok()) {
    state_->device_state_tensor = std::move(rng_state);
  } else {
    // Already initialized in delegating constructor.
    set_state(*rng_state.to(at::kCPU).view(at::kByte).unsafeGetTensorImpl());
  }
}

DeviceGeneratorImpl::DeviceGeneratorImpl(
    c10::DeviceIndex device_index,
    c10::intrusive_ptr<DeviceGeneratorState> state)
    : c10::GeneratorImpl(
          c10::Device(c10::DeviceType::PrivateUse1, device_index),
          c10::DispatchKeySet(c10::DispatchKey::PrivateUse1)),
      state_(std::move(state)) {}

void DeviceGeneratorImpl::set_current_seed(uint64_t seed) {
  // set_current_seed() is invoked by PyTorch and behaves like an op.
  TT_KERNEL(OpName::kRngSetSeed, _,
            (IgnoreInCacheKey(seed, "delegates to UpdateRngSeed()")), {
              TT_ASSIGN_OR_THROW(
                  auto rng_state_buffer,
                  UpdateDeviceRngSeed(state_->device_state_tensor, seed));
              auto new_rng_state = MakeTensor(std::move(rng_state_buffer));
              TT_ASSIGN_OR_THROW(auto rng_state_buffer2,
                                 UpdateDeviceRngOffset(new_rng_state, 0));
              state_->device_state_tensor =
                  MakeTensor(std::move(rng_state_buffer2));
            });
}

void DeviceGeneratorImpl::set_offset(uint64_t offset) {
  // set_offset() is invoked by PyTorch and behaves like an op.
  TT_KERNEL(OpName::kRngSetOffset, _,
            (IgnoreInCacheKey(offset, "delegates to UpdateRngOffset()")), {
              TT_ASSIGN_OR_THROW(
                  auto rng_state_buffer,
                  UpdateDeviceRngOffset(state_->device_state_tensor, offset));
              auto new_rng_state = MakeTensor(std::move(rng_state_buffer));
              state_->device_state_tensor = new_rng_state;
            });
}

uint64_t DeviceGeneratorImpl::get_offset() const {
  return state_->device_state_tensor[1].item<int64_t>();
}

uint64_t DeviceGeneratorImpl::current_seed() const {
  return state_->device_state_tensor[0].item<int64_t>();
}

uint64_t DeviceGeneratorImpl::seed() {
  // seed() is invoked by PyTorch and behaves like an op.
  TT_KERNEL(OpName::kRngSeed, _, (), {
    auto random = c10::detail::getNonDeterministicRandom(false);
    this->set_current_seed(random);
    return random;
  });
}

c10::intrusive_ptr<c10::TensorImpl> DeviceGeneratorImpl::get_state() const {
  // Gets the current internal state of DeviceGeneratorImpl. The internal
  // state is returned as a CPU byte tensor.
  ABSL_VLOG(1) << "[get_state]";

  // Exclude the Python dispatch key to prevent FakeTensorMode and other
  // Python-level dispatch modes from intercepting internal eager tensor
  // allocations and operations in this scope.
  c10::impl::ExcludeDispatchKeyGuard guard(c10::DispatchKey::Python);

  return state_->device_state_tensor.view(at::kByte)
      .to(at::kCPU)
      .getIntrusivePtr();
}

// Sets the internal state of DeviceGeneratorImpl. The new internal state
// must be a strided CPU byte tensor and have appropriate size.
void DeviceGeneratorImpl::set_state(const c10::TensorImpl& new_state_impl) {
  ABSL_VLOG(1) << "[DeviceGeneratorImpl::set_state]" << new_state_impl.dtype();

  // Exclude the Python dispatch key to prevent FakeTensorMode and other
  // Python-level dispatch modes from intercepting internal eager tensor
  // allocations and operations in this scope.
  c10::impl::ExcludeDispatchKeyGuard guard(c10::DispatchKey::Python);

  auto impl_ptr = c10::intrusive_ptr<c10::TensorImpl>::reclaim_copy(
      const_cast<c10::TensorImpl*>(&new_state_impl));
  at::Tensor new_state = at::Tensor(std::move(impl_ptr));

  TT_CHECK_THROW(new_state.device().type() == at::kCPU &&
                     new_state.dtype() == at::kByte &&
                     new_state.layout() == at::kStrided,
                 error::kFailedPrecondition)
      << "expect rng state to be a torch.ByteTensor";
  TT_CHECK_THROW(new_state.is_contiguous(), error::kFailedPrecondition)
      << "expect rng state to be contiguous";
  TT_CHECK_THROW(new_state.sizes() == c10::IntArrayRef({16}),
                 error::kFailedPrecondition)
      << "expect rng state to be shape (16,), got " << new_state.sizes();
  state_->device_state_tensor = new_state.view(at::kUInt64).to(device());
}

void DeviceGeneratorImpl::graphsafe_set_state(
    const c10::intrusive_ptr<c10::GeneratorImpl>& state) {
  auto* other = dynamic_cast<DeviceGeneratorImpl*>(state.get());
  TT_CHECK_THROW(other != nullptr, error::kInvalidArgument)
      << "Expected DeviceGeneratorImpl in graphsafe_set_state";
  ABSL_CHECK(other != nullptr);  // CRASH_OK=satisfying ClangTidy
  state_ = other->state_;
}

c10::intrusive_ptr<c10::GeneratorImpl>
DeviceGeneratorImpl::graphsafe_get_state() const {
  return c10::make_intrusive<DeviceGeneratorImpl>(device().index(), state_);
}

DeviceGeneratorImpl* DeviceGeneratorImpl::clone_impl() const {
  // Exclude the Python dispatch key to prevent FakeTensorMode and other
  // Python-level dispatch modes from intercepting internal eager tensor
  // allocations and operations in this scope.
  c10::impl::ExcludeDispatchKeyGuard guard(c10::DispatchKey::Python);

  return new DeviceGeneratorImpl(device().index(), state_->clone());
}

at::Generator& GetDefaultDeviceGenerator(c10::DeviceIndex idx) {
  return DeviceGenerators::GetDefaultInstance().GetDefaultGenerator(idx);
}

absl::Status DeviceGeneratorImpl::CheckDeviceStateTensor(
    const at::Tensor& rng_state) const {
  TT_RET_CHECK(rng_state.device() == device(), error::kFailedPrecondition)
      << "expected rng_state to be on device " << device() << ", got "
      << rng_state.device();
  TT_RET_CHECK(rng_state.dtype() == at::kUInt64, error::kFailedPrecondition)
      << "expected rng_state to be of dtype UInt64, got " << rng_state.dtype();
  TT_RET_CHECK(rng_state.dim() == 1, error::kFailedPrecondition)
      << "expected rng_state to be a 1D tensor, got " << rng_state.dim();
  TT_RET_CHECK(rng_state.size(0) == 2, error::kFailedPrecondition)
      << "expected rng_state size 2, got " << rng_state.size(0);
  return absl::OkStatus();
}

absl::StatusOr<at::Tensor> DeviceGeneratorImpl::GetAndAdvanceDeviceStateTensor(
    int64_t num_elements, int64_t bit_width) {
  auto rng_input_state = state_->device_state_tensor;
  if (num_elements <= 0 || bit_width <= 0) {
    return rng_input_state;
  }

  // Snapshot the original buffer so that we can return it after updating the
  // state.
  TT_ASSIGN_OR_RETURN(DeviceBufferRef original_buf,
                      GetBufferFromAtTensor(rng_input_state));

  // Dispatch the state update.
  TT_ASSIGN_OR_RETURN(auto state_param_keys,
                      TT_MAKE_OP_PARAM_CACHE_KEYS(num_elements, bit_width));
  auto state_op_builder = [num_elements,
                           bit_width](mlir::MlirOp rng_input_state) {
    return BuildRngStateUpdateShlo(rng_input_state, num_elements, bit_width);
  };
  TT_ASSIGN_OR_RETURN(
      auto rng_output_state_buf,
      (DispatchOp<1>(std::move(state_op_builder), {rng_input_state},
                     // Override the op name as this is a subroutine rather than
                     // a top-level op.
                     {.op_name = OpName::kRngStateUpdate,
                      .out_dtype = mlir::ElementType::UI64,
                      .out_dims = {2},
                      .op_param_cache_keys = std::move(state_param_keys)})));

  // Give back the updated state to the generator.
  auto rng_output_state = MakeTensor(std::move(rng_output_state_buf));
  TT_RETURN_IF_ERROR(SetDeviceStateTensor(rng_output_state));

  // Return a tensor pointing to the original buffer.
  return MakeTensor(std::move(original_buf));
}

at::Generator MakeDeviceGenerator(c10::DeviceIndex idx) {
  return DeviceGenerators::GetDefaultInstance().CreateGenerator(idx);
}

void SetManualSeed(uint64_t seed, c10::DeviceIndex idx) {
  auto gen = DeviceGenerators::GetDefaultInstance().GetDefaultGenerator(idx);
  gen.set_current_seed(seed);
}

void SetManualSeedAll(uint64_t seed) {
  const int64_t num_devices =
      DeviceGenerators::GetDefaultInstance().num_devices();
  for (int i = 0; i < num_devices; ++i) {
    SetManualSeed(seed, i);
  }
}

}  // namespace torch_tpu
