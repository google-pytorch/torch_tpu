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
#include <string_view>
#include <utility>
#include <vector>

#include "ATen/core/ATen_fwd.h"
#include "ATen/core/Generator.h"
#include "ATen/ops/full.h"
#include "absl/container/flat_hash_map.h"
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
#include "c10/util/Optional.h"
#include "c10/util/intrusive_ptr.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Types.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"
#include "torch/headeronly/core/DeviceType.h"
#include "torch/headeronly/core/Layout.h"
#include "torch_tpu/common/cache_key.h"
#include "torch_tpu/common/context_manager.h"
#include "torch_tpu/common/context_states.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/eager/device_buffer.h"
#include "torch_tpu/eager/materialize.h"
#include "torch_tpu/eager/op_dispatcher.h"
#include "torch_tpu/eager/structured_log_buffer.h"
#include "torch_tpu/eager/tensor_to_buffer.h"
#include "torch_tpu/ops/macros/kernel.h"
#include "torch_tpu/ops/op_builder_utils.h"
#include "torch_tpu/ops/op_names.h"

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

absl::flat_hash_map<c10::DeviceIndex, at::Generator>&
GetExportModeDefaultGenerators() {
  thread_local  // CPP_THREAD_LOCAL_OK=Export overrides are per-thread.
      absl::flat_hash_map<c10::DeviceIndex, at::Generator>* const kInstance =
          new absl::flat_hash_map<c10::DeviceIndex, at::Generator>();  // NOLINT
  return *kInstance;
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

  void SetExportModeDefaultGenerator(c10::DeviceIndex idx,
                                     c10::optional<at::Generator> gen);

  void ClearExportModeDefaultGenerators() {
    GetExportModeDefaultGenerators().clear();
  }

 private:
  DeviceGenerators();
  std::vector<at::Generator> generators_;
  std::deque<c10::once_flag> generator_init_flags_;
  int64_t num_devices_;
};

c10::DeviceIndex NormalizeDeviceIndex(c10::DeviceIndex idx) {
  const int32_t num_devices =
      DeviceGenerators::GetDefaultInstance().num_devices();
  if (idx == -1) {
    const auto* guard =
        c10::impl::getDeviceGuardImpl(c10::DeviceType::PrivateUse1);
    TT_CHECK_THROW(guard != nullptr, error::kFailedPrecondition)
        << "PrivateUse1 device guard is not registered.";
    idx = guard->getDevice().index();
  }
  TT_CHECK_THROW(idx >= 0 && idx < num_devices, error::kFailedPrecondition)
      << "The device_index is invalid, expected an index between 0 and "
      << num_devices - 1 << " got " << static_cast<int32_t>(idx);
  return idx;
}

}  // namespace

absl::Status DeviceGeneratorState::MaybeMaterializeDeviceStateTensor(
    bool force_materialization) {
  if (!force_materialization &&
      ++materialize_state_counter_ < kMaterializationThreshold) {
    return absl::OkStatus();
  }

  materialize_state_counter_ = 0;
  TT_ASSIGN_OR_RETURN(DeviceBufferRef buf, GetBuffer(device_state_tensor_));

  if (buf.is_placeholder() || buf.depends_on_placeholder()) {
    return absl::OkStatus();
  }

  return Materialize(buf, MaterializationReason::kExplicitSync);
}

absl::Status DeviceGeneratorState::SetDeviceStateTensor(
    at::Tensor device_state_tensor) {
  device_state_tensor_ = std::move(device_state_tensor);
  TT_RETURN_IF_ERROR(MaybeMaterializeDeviceStateTensor());
  return absl::OkStatus();
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
  static auto* const kInstance = new DeviceGenerators();
  return *kInstance;
}

void DeviceGenerators::SetExportModeDefaultGenerator(
    const c10::DeviceIndex idx, c10::optional<at::Generator> gen) {
  if (gen.has_value()) {
    at::check_generator<ExportModeDeviceGeneratorImpl>(*gen);
    GetExportModeDefaultGenerators()[idx] = std::move(*gen);
  } else {
    GetExportModeDefaultGenerators().erase(idx);
  }
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
  idx = NormalizeDeviceIndex(idx);

  const ExportMode export_mode =
      GetContextState<ExportModeContextState>().value_or(ExportMode::kDisabled);
  if (export_mode == ExportMode::kEnabled) {
    const auto it = GetExportModeDefaultGenerators().find(idx);
    ABSL_CHECK(  // CRASH_OK=Export mode generator not set is a
                 // bug in TorchTPU.
        it != GetExportModeDefaultGenerators().end())
        << "expected export mode default generator "
           "to be set for device "
        << idx;
    return it->second;
  }

  c10::call_once(generator_init_flags_[idx], [&] {
    generators_[idx] = at::make_generator<EagerModeDeviceGeneratorImpl>(idx);
    generators_[idx].seed();
  });
  return generators_[idx];
}

at::Generator DeviceGenerators::CreateGenerator(c10::DeviceIndex idx) const {
  ABSL_VLOG(1) << "[CreateGenerator] idx: " << static_cast<int32_t>(idx);
  idx = NormalizeDeviceIndex(idx);
  return at::make_generator<EagerModeDeviceGeneratorImpl>(idx);
}

/*static*/ c10::DeviceType DeviceGeneratorImplBase::device_type() {
  return c10::DeviceType::PrivateUse1;
}

DeviceGeneratorImplBase::DeviceGeneratorImplBase(
    const c10::DeviceIndex device_index)
    : c10::GeneratorImpl(
          c10::Device(c10::DeviceType::PrivateUse1, device_index),
          c10::DispatchKeySet(c10::DispatchKey::PrivateUse1)) {}

absl::Status DeviceGeneratorImplBase::CheckDeviceStateTensor(
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

absl::Status DeviceGeneratorImplBase::AdvanceDeviceStateTensor(
    int64_t num_elements, int64_t bit_width) {
  ABSL_CHECK_GE(num_elements, 0)  // CRASH_OK
      << "num_elements must be non-negative.";
  ABSL_CHECK_GE(bit_width, 0) << "bit_width must be non-negative.";  // CRASH_OK

  auto rng_input_state = DeviceStateTensor();
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
  auto rng_output_state =
      MakeTensor(std::move(rng_output_state_buf), this->device().index());
  TT_RETURN_IF_ERROR(CheckDeviceStateTensor(rng_output_state));
  TT_RETURN_IF_ERROR(SetDeviceStateTensor(rng_output_state));

  return absl::OkStatus();
}

// ============================================================================
// EagerModeDeviceGeneratorImpl (non-export mode)
// ============================================================================

EagerModeDeviceGeneratorImpl::EagerModeDeviceGeneratorImpl(
    const c10::DeviceIndex device_index)
    : DeviceGeneratorImplBase(device_index) {
  state_ = c10::make_intrusive<DeviceGeneratorState>(
      CreateDeviceRngStateTensor(this->device()));
}

EagerModeDeviceGeneratorImpl::EagerModeDeviceGeneratorImpl(
    const c10::DeviceIndex device_index,
    c10::intrusive_ptr<DeviceGeneratorState> state)
    : DeviceGeneratorImplBase(device_index), state_(std::move(state)) {}

EagerModeDeviceGeneratorImpl::EagerModeDeviceGeneratorImpl(
    const c10::DeviceIndex device_index, at::Tensor rng_state)
    : DeviceGeneratorImplBase(device_index) {
  if (this->CheckDeviceStateTensor(rng_state).ok()) {
    state_ = c10::make_intrusive<DeviceGeneratorState>(std::move(rng_state));
  } else {
    state_ = c10::make_intrusive<DeviceGeneratorState>(
        CreateDeviceRngStateTensor(this->device()));
    set_state(*rng_state.to(at::kCPU).view(at::kByte).unsafeGetTensorImpl());
  }
}

at::Tensor EagerModeDeviceGeneratorImpl::DeviceStateTensor() const {
  return state_->DeviceStateTensor();
}

absl::Status EagerModeDeviceGeneratorImpl::SetDeviceStateTensor(at::Tensor t) {
  return state_->SetDeviceStateTensor(std::move(t));
}

void EagerModeDeviceGeneratorImpl::set_current_seed(uint64_t seed) {
  // set_current_seed() is invoked by PyTorch and behaves like an op.
  TT_KERNEL(
      OpName::kRngSetSeed, _,
      (IgnoreInCacheKey(seed, "delegates to UpdateRngSeed()")), {
        TT_ASSIGN_OR_THROW(
            auto rng_state_buffer,
            UpdateDeviceRngSeed(state_->DeviceStateTensor(), seed));
        auto new_rng_state =
            MakeTensor(std::move(rng_state_buffer), this->device().index());
        TT_ASSIGN_OR_THROW(auto rng_state_buffer2,
                           UpdateDeviceRngOffset(new_rng_state, 0));
        auto final_rng_state =
            MakeTensor(std::move(rng_state_buffer2), this->device().index());
        TT_THROW_IF_ERROR(CheckDeviceStateTensor(final_rng_state));
        TT_THROW_IF_ERROR(SetDeviceStateTensor(std::move(final_rng_state)));
      });
}

void EagerModeDeviceGeneratorImpl::set_offset(uint64_t offset) {
  // set_offset() is invoked by PyTorch and behaves like an op.
  TT_KERNEL(OpName::kRngSetOffset, _,
            (IgnoreInCacheKey(offset, "delegates to UpdateRngOffset()")), {
              TT_ASSIGN_OR_THROW(
                  auto rng_state_buffer,
                  UpdateDeviceRngOffset(state_->DeviceStateTensor(), offset));
              auto new_rng_state = MakeTensor(std::move(rng_state_buffer),
                                              this->device().index());
              TT_THROW_IF_ERROR(CheckDeviceStateTensor(new_rng_state));
              TT_THROW_IF_ERROR(SetDeviceStateTensor(new_rng_state));
            });
}

uint64_t EagerModeDeviceGeneratorImpl::get_offset() const {
  return state_->DeviceStateTensor()[1].item<int64_t>();
}

uint64_t EagerModeDeviceGeneratorImpl::current_seed() const {
  return state_->DeviceStateTensor()[0].item<int64_t>();
}

uint64_t EagerModeDeviceGeneratorImpl::seed() {
  // seed() is invoked by PyTorch and behaves like an op.
  TT_KERNEL(OpName::kRngSeed, _, (), {
    auto random = c10::detail::getNonDeterministicRandom(false);
    this->set_current_seed(random);
    return random;
  });
}

c10::intrusive_ptr<c10::TensorImpl> EagerModeDeviceGeneratorImpl::get_state()
    const {
  ABSL_VLOG(1) << "[get_state]";
  c10::impl::ExcludeDispatchKeyGuard guard(c10::DispatchKey::Python);
  return state_->DeviceStateTensor()
      .view(at::kByte)
      .to(at::kCPU)
      .getIntrusivePtr();
}

void EagerModeDeviceGeneratorImpl::set_state(
    const c10::TensorImpl& new_state_impl) {
  ABSL_VLOG(1) << "[EagerModeDeviceGeneratorImpl::set_state]"
               << new_state_impl.dtype();
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
  at::Tensor state_tensor = new_state.view(at::kUInt64).to(device());
  TT_THROW_IF_ERROR(CheckDeviceStateTensor(state_tensor));
  TT_THROW_IF_ERROR(SetDeviceStateTensor(std::move(state_tensor)));
}

c10::intrusive_ptr<c10::GeneratorImpl>
EagerModeDeviceGeneratorImpl::graphsafe_get_state() const {
  return c10::make_intrusive<EagerModeDeviceGeneratorImpl>(device().index(),
                                                           state_);
}

void EagerModeDeviceGeneratorImpl::graphsafe_set_state(
    const c10::intrusive_ptr<c10::GeneratorImpl>& state) {
  if (const auto* eager_other =
          dynamic_cast<const EagerModeDeviceGeneratorImpl*>(state.get())) {
    state_ = eager_other->state_;
    return;
  }
  if (const auto* base_other =
          dynamic_cast<const DeviceGeneratorImplBase*>(state.get())) {
    at::Tensor state_tensor = base_other->DeviceStateTensor();
    TT_THROW_IF_ERROR(SetDeviceStateTensor(std::move(state_tensor)));
    return;
  }
  TT_CHECK_THROW(false, error::kInvalidArgument)
      << "Expected DeviceGeneratorImplBase in graphsafe_set_state";
}

DeviceGeneratorImplBase* EagerModeDeviceGeneratorImpl::clone_impl() const {
  c10::impl::ExcludeDispatchKeyGuard guard(c10::DispatchKey::Python);
  return new EagerModeDeviceGeneratorImpl(device().index(), state_->clone());
}

// ============================================================================
// ExportModeDeviceGeneratorImpl
// ============================================================================

ExportModeDeviceGeneratorImpl::ExportModeDeviceGeneratorImpl(
    const at::Generator& default_generator)
    : DeviceGeneratorImplBase(default_generator.device().index()) {
  auto* eager_base = default_generator.get<DeviceGeneratorImplBase>();
  TT_CHECK_THROW(eager_base != nullptr, error::kInvalidArgument)
      << "Expected DeviceGeneratorImplBase in ExportModeDeviceGeneratorImpl "
         "constructor";
  export_state_ = c10::make_intrusive<ExportModeDeviceGeneratorState>(
      eager_base->DeviceStateTensor().clone());
}

ExportModeDeviceGeneratorImpl::ExportModeDeviceGeneratorImpl(
    const c10::DeviceIndex device_index,
    c10::intrusive_ptr<ExportModeDeviceGeneratorState> state)
    : DeviceGeneratorImplBase(device_index), export_state_(std::move(state)) {}

at::Tensor ExportModeDeviceGeneratorImpl::DeviceStateTensor() const {
  return export_state_->DeviceStateTensor();
}

absl::Status ExportModeDeviceGeneratorImpl::SetDeviceStateTensor(
    at::Tensor device_state_tensor) {
  TT_RETURN_IF_ERROR(CheckDeviceStateTensor(device_state_tensor));
  return export_state_->SetDeviceStateTensor(std::move(device_state_tensor));
}

namespace {
[[noreturn]] void CrashExportModeGeneratorUnsupported(std::string_view name) {
  ABSL_CHECK(false)  // CRASH_OK
      << "method " << name
      << " is not supported in ExportModeDeviceGeneratorImpl";
}
}  // namespace

uint64_t ExportModeDeviceGeneratorImpl::get_offset() const {
  CrashExportModeGeneratorUnsupported("get_offset");
}
uint64_t ExportModeDeviceGeneratorImpl::current_seed() const {
  CrashExportModeGeneratorUnsupported("current_seed");
}
uint64_t ExportModeDeviceGeneratorImpl::seed() {
  CrashExportModeGeneratorUnsupported("seed");
}
c10::intrusive_ptr<c10::TensorImpl> ExportModeDeviceGeneratorImpl::get_state()
    const {
  CrashExportModeGeneratorUnsupported("get_state");
}
void ExportModeDeviceGeneratorImpl::set_current_seed(uint64_t seed) {
  CrashExportModeGeneratorUnsupported("set_current_seed");
}
void ExportModeDeviceGeneratorImpl::set_offset(uint64_t offset) {
  CrashExportModeGeneratorUnsupported("set_offset");
}
void ExportModeDeviceGeneratorImpl::set_state(
    const c10::TensorImpl& new_state) {
  CrashExportModeGeneratorUnsupported("set_state");
}

c10::intrusive_ptr<c10::GeneratorImpl>
ExportModeDeviceGeneratorImpl::graphsafe_get_state() const {
  return c10::make_intrusive<ExportModeDeviceGeneratorImpl>(device().index(),
                                                            export_state_);
}

void ExportModeDeviceGeneratorImpl::graphsafe_set_state(
    const c10::intrusive_ptr<c10::GeneratorImpl>& state) {
  if (const auto* exp_other =
          dynamic_cast<const ExportModeDeviceGeneratorImpl*>(state.get())) {
    export_state_ = exp_other->export_state_;
    return;
  }
  if (const auto* base_other =
          dynamic_cast<const DeviceGeneratorImplBase*>(state.get())) {
    at::Tensor state_tensor = base_other->DeviceStateTensor();
    TT_THROW_IF_ERROR(SetDeviceStateTensor(std::move(state_tensor)));
    return;
  }
  TT_CHECK_THROW(false, error::kInvalidArgument)
      << "Expected DeviceGeneratorImplBase in graphsafe_set_state";
}

DeviceGeneratorImplBase* ExportModeDeviceGeneratorImpl::clone_impl() const {
  c10::impl::ExcludeDispatchKeyGuard guard(c10::DispatchKey::Python);
  return new ExportModeDeviceGeneratorImpl(device().index(),
                                           export_state_->clone());
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

void SetExportModeDefaultDeviceGenerator(c10::optional<at::Generator> gen,
                                         c10::DeviceIndex idx) {
  idx = NormalizeDeviceIndex(idx);
  DeviceGenerators::GetDefaultInstance().SetExportModeDefaultGenerator(
      idx, std::move(gen));
}

void ClearExportModeDefaultDeviceGenerators() {
  DeviceGenerators::GetDefaultInstance().ClearExportModeDefaultGenerators();
}

at::Generator& GetDefaultDeviceGenerator(c10::DeviceIndex idx) {
  return DeviceGenerators::GetDefaultInstance().GetDefaultGenerator(idx);
}

}  // namespace torch_tpu
