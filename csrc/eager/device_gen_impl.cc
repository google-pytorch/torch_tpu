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

#include "csrc/eager/device_gen_impl.h"

#include <cstdint>
#include <deque>
#include <string>
#include <utility>
#include <vector>

#include "ATen/core/ATen_fwd.h"
#include "ATen/core/Generator.h"
#include "absl/base/no_destructor.h"
#include "absl/log/absl_check.h"
#include "absl/log/absl_log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
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
#include "csrc/common/cache_key.h"
#include "csrc/common/context_states.h"
#include "csrc/common/env_vars.h"
#include "csrc/common/error_utils.h"
#include "csrc/eager/device_buffer.h"
#include "csrc/eager/eager_mode.h"
#include "csrc/eager/materialize.h"
#include "csrc/eager/op_dispatcher.h"
#include "csrc/eager/structured_log_buffer.h"
#include "csrc/eager/tensor_to_buffer.h"
#include "csrc/ops/macros/kernel.h"
#include "csrc/ops/op_builder_utils.h"
#include "csrc/ops/op_names.h"
#include "csrc/pjrt/pjrt_state.h"
#include "csrc/pjrt/pjrt_utils.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Types.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"
#include "torch/headeronly/core/DeviceType.h"
#include "torch/headeronly/core/Layout.h"

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

  TT_ASSIGN_OR_RETURN(auto params,
                      TT_MAKE_OP_PARAM_CACHE_KEYS(value, position));
  return DispatchOp<1>(std::move(set_seed_builder), {rng_state},
                       // Override the op name as this is a subroutine rather
                       // than a top-level op.
                       {.op_name = OpName::kRngSetStateComponent,
                        .out_dtype = mlir::ElementType::UI64,
                        .out_dims = {2},
                        .op_param_cache_keys = std::move(params)});
}

absl::StatusOr<DeviceBufferRef> UpdateDeviceRngOffset(at::Tensor rng_state,
                                                      uint64_t value) {
  return UpdateDeviceRngState(rng_state, value, /*position=*/1);
}

// Allocates and initializes the 1D uint64 RNG state tensor ({seed, offset})
// eagerly on the device using host-to-device memory copy.
//
// This function must be used for generator state initialization and host-driven
// seed/offset updates (like manual seeding).
//
// Standard alternatives like `MakeEmptyTensor` or `at::full` cannot be used
// here because:
// 1. They are deferred and require an active `ScopedPythonContextCapturer`
//    to track the operation. Seeding from Python utility scripts (e.g., in test
//    setup) does not run inside an operator context, so no capturer is alive,
//    causing deferred allocations to crash.
// 2. They dispatch through PyTorch, which violates operator isolation and
//    causes `CompositeOpCheck` failures when lazy generator initialization is
//    triggered from inside a native operator (like `normal_`).
absl::StatusOr<at::Tensor> CreateDeviceRngStateTensor(c10::Device device,
                                                      uint64_t seed,
                                                      uint64_t offset) {
  const uint64_t host_data[2] = {seed, offset};
  TT_ASSIGN_OR_RETURN(
      auto rng_state_buffer,
      TpuMallocAndMemcpyHtoD(host_data, mlir::ElementType::UI64, {2}));
  return MakeTensor(std::move(rng_state_buffer), device.index());
}

// A modifiable string that can be used to force the next InitDefaultGenerator()
// call to fail. This is non-sticky: only the next call is affected. Used for
// testing.
[[nodiscard]] std::string& GetInjectedInitDefaultGeneratorFailure() {
  static absl::NoDestructor<std::string> msg;
  return *msg;
}
}  // namespace

absl::Status InitDefaultGenerator(DeviceGeneratorImpl* gen_impl,
                                  uint64_t seed) {
  auto& failure_msg = GetInjectedInitDefaultGeneratorFailure();
  if (!failure_msg.empty()) {
    std::string err = std::move(failure_msg);
    failure_msg.clear();
    return TT_ERROR(error::kInternal) << err;
  }
  return gen_impl->state_->WriteStateToDevice(seed, 0);
}

// A singleton that holds one generator per device. The generators are lazily
// initialized.
// TODO: make the methods return Status instead of throwing on error.
class DeviceGenerators {
 public:
  // This class is neither copyable nor movable.
  DeviceGenerators(const DeviceGenerators&) = delete;
  DeviceGenerators& operator=(const DeviceGenerators&) = delete;
  DeviceGenerators(DeviceGenerators&&) = delete;
  DeviceGenerators& operator=(DeviceGenerators&&) = delete;

  [[nodiscard]] static DeviceGenerators& GetDefaultInstance();

  // Returns the default generator for the given device index. If idx is -1,
  // returns the default generator for the current device.
  [[nodiscard]] at::Generator& GetDefaultGenerator(c10::DeviceIndex idx = -1);

  // Creates a new generator for the given device index. If idx is -1, creates a
  // new generator for the current device.
  [[nodiscard]] at::Generator CreateGenerator(c10::DeviceIndex idx = -1) const;

  [[nodiscard]] int num_devices() const { return num_devices_; }

 private:
  friend void PyResetDefaultDeviceGeneratorsForTesting();

  explicit DeviceGenerators(int num_devices);

  // Resets all generators to uninitialized state.
  void ResetGenerators() {
    generators_.clear();
    generators_.resize(num_devices_);
    generator_init_flags_.clear();
    generator_init_flags_.resize(num_devices_);
  }

  std::vector<absl::StatusOr<at::Generator>> generators_;
  std::deque<c10::once_flag> generator_init_flags_;
  const int num_devices_ = -1;
};

absl::Status DeviceGeneratorState::MaybeMaterializeDeviceStateTensor(
    bool force_materialization) {
  const auto eager_mode = GetEagerMode();
  if (eager_mode == EagerMode::kInternalCompileFxGraph) {
    // In FX graph compilation mode, fully bypass materialization, including
    // updating the state counter, and ignore force_materialization as no
    // materialization is possible.
    return absl::OkStatus();
  }

  // Materialize if:
  // - force_materialization is true, or
  // - the materialization threshold is reached and we're not in
  //   kInternalDeferAll mode.
  // TODO(b/551978063): this should also materialize in kDeferNever or
  // kDeferNeverAndLaunchBlocking modes, but this breaks excess precision
  // functionality currently.
  const bool should_materialize =
      force_materialization ||
      (++materialize_state_counter_ >= kMaterializationThreshold &&
       eager_mode != EagerMode::kInternalDeferAll);

  if (!should_materialize) {
    return absl::OkStatus();
  }

  // Do a materialization, and block on it if we're in launch blocking mode.
  materialize_state_counter_ = 0;
  TT_ASSIGN_OR_RETURN(DeviceBufferRef buf, GetBuffer(device_state_tensor_));
  TT_RETURN_IF_ERROR(Materialize(buf, MaterializationReason::kExplicitSync));
  if (eager_mode == EagerMode::kDeferNeverAndLaunchBlocking) {
    TT_RETURN_IF_ERROR(buf.Synchronize());
  }
  return absl::OkStatus();
}

absl::Status DeviceGeneratorState::SetDeviceStateTensor(
    at::Tensor device_state_tensor) {
  device_state_tensor_ = std::move(device_state_tensor);
  TT_RETURN_IF_ERROR(MaybeMaterializeDeviceStateTensor());
  return absl::OkStatus();
}

absl::Status DeviceGeneratorState::WriteStateToDevice(uint64_t seed,
                                                      uint64_t offset) {
  TT_ASSIGN_OR_RETURN(
      auto new_rng_state,
      CreateDeviceRngStateTensor(device_state_tensor_.device(), seed, offset));
  return SetDeviceStateTensor(new_rng_state);
}

c10::intrusive_ptr<DeviceGeneratorState> DeviceGeneratorState::clone() const {
  auto new_state =
      c10::make_intrusive<DeviceGeneratorState>(device_state_tensor_.clone());
  TT_THROW_IF_ERROR(new_state->MaybeMaterializeDeviceStateTensor(
      /*force_materialization=*/true));
  return new_state;
}

DeviceGenerators& DeviceGenerators::GetDefaultInstance() {
  // We cannot use absl::NoDestructor here because the constructor is private.
  static auto* const kInstance = []() -> DeviceGenerators* {
    const auto* const guard =
        c10::impl::getDeviceGuardImpl(c10::DeviceType::PrivateUse1);
    ABSL_CHECK(guard != nullptr)  // CRASH_OK
        << "TPU device guard is not registered. This is a TorchTPU bug.";
    return new DeviceGenerators(guard->deviceCount());
  }();
  return *kInstance;
}

DeviceGenerators::DeviceGenerators(int num_devices)
    : num_devices_(num_devices) {
  ResetGenerators();
}

at::Generator& DeviceGenerators::GetDefaultGenerator(at::DeviceIndex idx) {
  ABSL_VLOG(1) << "[DeviceGenerators::GetDefaultGenerator] idx: "
               << static_cast<int>(idx);
  if (idx == -1) {
    const auto* guard =
        c10::impl::getDeviceGuardImpl(c10::DeviceType::PrivateUse1);
    TT_CHECK_THROW(guard != nullptr, error::kFailedPrecondition)
        << "PrivateUse1 device guard is not registered.";
    idx = guard->getDevice().index();
  } else {
    TT_CHECK_THROW(idx >= 0 && idx < generators_.size() &&
                       idx < generator_init_flags_.size(),
                   error::kFailedPrecondition)
        << "The device_index is invalid, expected an index between 0 and"
        << generators_.size() - 1 << " got " << static_cast<int>(idx);
  }

  auto& generator = generators_[idx];
  c10::call_once(generator_init_flags_[idx], [&] {
    generator = at::make_generator<DeviceGeneratorImpl>(idx);
    auto* gen_impl = generator->get<DeviceGeneratorImpl>();
    auto random = c10::detail::getNonDeterministicRandom(false);
    auto status = InitDefaultGenerator(gen_impl, random);
    if (!status.ok()) {
      generator = status;
    }
  });

  // IMPORTANT: if the generator failed to initialize before, we must fail now
  // instead of returning an uninitialized generator.
  TT_THROW_IF_ERROR(generator.status());
  return *generator;
}

at::Generator DeviceGenerators::CreateGenerator(c10::DeviceIndex idx) const {
  ABSL_VLOG(1) << "[CreateGenerator] idx: " << static_cast<int>(idx);
  if (idx == -1) {
    const auto* guard =
        c10::impl::getDeviceGuardImpl(c10::DeviceType::PrivateUse1);
    TT_CHECK_THROW(guard != nullptr, error::kFailedPrecondition)
        << "PrivateUse1 device guard is not registered.";
    idx = guard->getDevice().index();
  }
  TT_CHECK_THROW(idx >= 0 && idx < num_devices_, error::kFailedPrecondition)
      << "The device_index is invalid, expected an index between 0 and"
      << num_devices_ - 1 << " got " << static_cast<int>(idx);
  return at::make_generator<DeviceGeneratorImpl>(idx);
}

/*static*/ c10::DeviceType DeviceGeneratorImpl::device_type() {
  return c10::DeviceType::PrivateUse1;
}

at::Tensor DeviceGeneratorImpl::DeviceStateTensor() const {
  return state_->DeviceStateTensor();
}

absl::Status DeviceGeneratorImpl::SetDeviceStateTensor(
    at::Tensor device_state_tensor) {
  TT_RETURN_IF_ERROR(ValidateDeviceStateTensor(device_state_tensor));
  return state_->SetDeviceStateTensor(std::move(device_state_tensor));
}

DeviceGeneratorImpl::DeviceGeneratorImpl(c10::DeviceIndex device_index)
    : c10::GeneratorImpl(
          c10::Device(c10::DeviceType::PrivateUse1, device_index),
          c10::DispatchKeySet(c10::DispatchKey::PrivateUse1)) {
  TT_ASSIGN_OR_THROW(auto tensor,
                     CreateDeviceRngStateTensor(this->device(), 0, 0));
  state_ = c10::make_intrusive<DeviceGeneratorState>(std::move(tensor));
}

DeviceGeneratorImpl::DeviceGeneratorImpl(c10::DeviceIndex device_index,
                                         at::Tensor rng_state)
    : DeviceGeneratorImpl(device_index) {
  if (ValidateDeviceStateTensor(rng_state).ok()) {
    state_ = c10::make_intrusive<DeviceGeneratorState>(std::move(rng_state));
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
            (IgnoreInCacheKey(seed, "delegates to WriteStateToDevice")),
            { TT_THROW_IF_ERROR(state_->WriteStateToDevice(seed, 0)); });
}

void DeviceGeneratorImpl::set_offset(uint64_t offset) {
  // set_offset() is invoked by PyTorch and behaves like an op.
  TT_KERNEL(OpName::kRngSetOffset, _,
            (IgnoreInCacheKey(offset, "delegates to UpdateRngOffset()")), {
              TT_ASSIGN_OR_THROW(
                  auto rng_state_buffer,
                  UpdateDeviceRngOffset(state_->DeviceStateTensor(), offset));
              auto new_rng_state = MakeTensor(std::move(rng_state_buffer),
                                              this->device().index());
              TT_THROW_IF_ERROR(SetDeviceStateTensor(new_rng_state));
            });
}

uint64_t DeviceGeneratorImpl::get_offset() const {
  return state_->DeviceStateTensor()[1].item<int64_t>();
}

uint64_t DeviceGeneratorImpl::current_seed() const {
  return state_->DeviceStateTensor()[0].item<int64_t>();
}

uint64_t DeviceGeneratorImpl::seed() {
  // seed() is invoked by PyTorch and behaves like an op.
  TT_KERNEL(OpName::kRngSeed, _, (), {
    auto random = c10::detail::getNonDeterministicRandom(false);
    TT_THROW_IF_ERROR(state_->WriteStateToDevice(random, 0));
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

  return state_->DeviceStateTensor()
      .view(at::kByte)
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
  TT_THROW_IF_ERROR(
      SetDeviceStateTensor(new_state.view(at::kUInt64).to(device())));
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
  // Fail early with an actionable error (e.g., VFIO collision) rather than
  // crashing later on an uninitialized device dereference.
  TT_THROW_IF_ERROR(PjrtBackend::GetInstance().EnsureInitialized());
  return DeviceGenerators::GetDefaultInstance().GetDefaultGenerator(idx);
}

absl::Status DeviceGeneratorImpl::ValidateDeviceStateTensor(
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

absl::Status DeviceGeneratorImpl::AdvanceDeviceStateTensor(int64_t num_elements,
                                                           int64_t bit_width) {
  ABSL_CHECK_GE(num_elements, 0)  // CRASH_OK
      << "num_elements must be non-negative.";
  ABSL_CHECK_GE(bit_width, 0) << "bit_width must be non-negative.";  // CRASH_OK

  auto rng_input_state = state_->DeviceStateTensor();
  if (num_elements == 0 || bit_width == 0) {
    return absl::OkStatus();
  }

  // Snapshot the original buffer so that we can return it after updating the
  // state.
  TT_ASSIGN_OR_RETURN(DeviceBufferRef original_buf, GetBuffer(rng_input_state));

  // Dispatch the state update.
  TT_ASSIGN_OR_RETURN(auto state_param_keys,
                      TT_MAKE_OP_PARAM_CACHE_KEYS(num_elements, bit_width));
  auto state_op_builder = [num_elements,
                           bit_width](mlir::MlirOp rng_input_state) {
    return BuildRngStateUpdateShlo(rng_input_state, num_elements, bit_width);
  };
  const OpSplitMode split_mode =
      GetEnvOnce<kTorchTpuInternalSplitRngStateUpdate>().value_or("0") == "1"
          ? OpSplitMode::kSplitBoth
          : OpSplitMode::kNone;

  TT_ASSIGN_OR_RETURN(
      auto rng_output_state_buf,
      (DispatchOp<1>(std::move(state_op_builder), {rng_input_state},
                     // Override the op name as this is a subroutine rather than
                     // a top-level op.
                     {.op_name = OpName::kRngStateUpdate,
                      .out_dtype = mlir::ElementType::UI64,
                      .out_dims = {2},
                      .op_param_cache_keys = std::move(state_param_keys),
                      .split_mode = split_mode})));

  // Give back the updated state to the generator.
  auto rng_output_state =
      MakeTensor(std::move(rng_output_state_buf), this->device().index());
  TT_RETURN_IF_ERROR(SetDeviceStateTensor(rng_output_state));

  return absl::OkStatus();
}

at::Generator MakeDeviceGenerator(c10::DeviceIndex idx) {
  return DeviceGenerators::GetDefaultInstance().CreateGenerator(idx);
}

void SetManualSeed(uint64_t seed, c10::DeviceIndex idx) {
  auto gen = DeviceGenerators::GetDefaultInstance().GetDefaultGenerator(idx);
  gen.set_current_seed(seed);
}

void SetManualSeedAll(uint64_t seed) {
  const int num_devices = DeviceGenerators::GetDefaultInstance().num_devices();
  for (int i = 0; i < num_devices; ++i) {
    SetManualSeed(seed, i);
  }
}

void PySetInitDefaultGeneratorFailureForTesting(std::string failure_message) {
  GetInjectedInitDefaultGeneratorFailure() = std::move(failure_message);
}

void PyResetDefaultDeviceGeneratorsForTesting() {
  DeviceGenerators::GetDefaultInstance().ResetGenerators();
}

}  // namespace torch_tpu
