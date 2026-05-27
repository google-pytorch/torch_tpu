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

#ifndef TORCH_TPU_EAGER_DEVICE_GEN_IMPL_H_
#define TORCH_TPU_EAGER_DEVICE_GEN_IMPL_H_

// RNG Generator Isolation Design:
//
// TorchTPU provides strict RNG generator state isolation between eager
// execution mode and graph export mode (e.g., during torch.compile or
// torch.export tracing). This isolation prevents graph tracing passes (which
// inspect and advance RNG states) from mutating or polluting the global default
// eager RNG state.
//
// Structure:
// - DeviceGeneratorImplBase: Abstract base class defining the unified
//   interface for device RNG generators, exposing virtual state tensor
//   accessors.
// - EagerModeDeviceGeneratorImpl: Concrete subclass used during standard eager
//   execution. Manages the active device RNG state tensor via
//   DeviceGeneratorState.
// - ExportModeDeviceGeneratorImpl: Concrete subclass used exclusively during
//   graph export tracing. Wraps placeholder tensors and rejects unreachable
//   eager accessors.
//
// Isolation Mechanism:
// - A thread-local ExportMode context manager dictates which generator instance
//   is returned by GetDefaultDeviceGenerator().
// - Under export mode, overrides registered via
//   SetExportModeDefaultDeviceGenerator (which are
//   ExportModeDeviceGeneratorImpl instances) are returned.
// - Outside export mode, standard eager EagerModeDeviceGeneratorImpl instances
// are
//   returned.

#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

#include "ATen/core/Generator.h"
#include "ATen/ops/tensor.h"  // IWYU pragma: keep
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "c10/core/Device.h"
#include "c10/core/GeneratorImpl.h"
#include "c10/core/TensorImpl.h"
#include "c10/util/Optional.h"
#include "c10/util/intrusive_ptr.h"
#include "torch/headeronly/core/DeviceType.h"

namespace torch_tpu {

class DeviceBufferRef;

struct RngUsage {
  int64_t num_elements = -1;
  int64_t bit_width = -1;
};

// Forward declaration of DispatchRngOpGeneral with default arguments.
template <typename DispatchFunc>
absl::StatusOr<std::vector<DeviceBufferRef>> DispatchRngOpGeneral(
    c10::optional<at::Generator> generator, DispatchFunc dispatch_func,
    const std::optional<RngUsage>& usage = std::nullopt);

// Holds the device-resident state of the generator (seed and offset) as a
// tensor. This is the TPU equivalent of PyTorch's CUDAGeneratorState,
// encapsulating the current generation state of the device.
class DeviceGeneratorState : public c10::intrusive_ptr_target {
 public:
  static constexpr int kMaterializationThreshold = 4;

  DeviceGeneratorState() = default;
  explicit DeviceGeneratorState(at::Tensor device_state_tensor)
      : device_state_tensor_(std::move(device_state_tensor)) {}

  c10::intrusive_ptr<DeviceGeneratorState> clone() const;

  at::Tensor DeviceStateTensor() const { return device_state_tensor_; }

  // Sets the device-resident RNG state tensor.
  absl::Status SetDeviceStateTensor(at::Tensor device_state_tensor);

 private:
  // Materializes the RNG state tensor if the materialization counter reaches
  // the kMaterializationThreshold. If the tensor is a placeholder or depends on
  // a placeholder, it does not materialize.
  //
  // To prevent the deferred execution graph from growing indefinitely (which
  // causes host-side memory leaks and OOM during compilation), this method
  // throttles the materialization of the state tensor.
  //
  // Every `kMaterializationThreshold` updates, it materializes the state
  // tensor, compiling and executing the deferred updates up to this point.
  absl::Status MaybeMaterializeDeviceStateTensor(
      bool force_materialization = false);

  // 1D uint64 tensor of 2 elements: {seed, offset}
  at::Tensor device_state_tensor_ = at::Tensor();

  // Counter to throttle materialization of RNG state updates.
  int materialize_state_counter_ = 0;
};

// Lightweight RNG state container for export mode.
class ExportModeDeviceGeneratorState : public c10::intrusive_ptr_target {
 public:
  ExportModeDeviceGeneratorState() = default;
  explicit ExportModeDeviceGeneratorState(at::Tensor device_state_tensor)
      : device_state_tensor_(std::move(device_state_tensor)) {}

  c10::intrusive_ptr<ExportModeDeviceGeneratorState> clone() const {
    return c10::make_intrusive<ExportModeDeviceGeneratorState>(
        device_state_tensor_.clone());
  }

  at::Tensor DeviceStateTensor() const { return device_state_tensor_; }
  absl::Status SetDeviceStateTensor(at::Tensor t) {
    device_state_tensor_ = std::move(t);
    return absl::OkStatus();
  }

 private:
  at::Tensor device_state_tensor_ = at::Tensor();
};

// Forward declarations for friends.
at::Tensor PyGetDeviceStateTensor(at::Generator gen);
void PySetDeviceStateTensor(at::Generator gen, at::Tensor rng_state);

// A wrapper on a tensor of two integers, representing the current state (seed
// and offset) of the generator. Given these, we can deterministically generate
// random bits. Kernels that require random bits, will use this tensor as an
// extra argument, and will output the new state of the generator.
//
// Abstract base class defining the common interface and pure virtual state
// tensor accessors for TPU RNG generators across eager and export execution
// modes.
class DeviceGeneratorImplBase : public c10::GeneratorImpl {
 public:
  // This class is neither copyable nor movable.
  DeviceGeneratorImplBase(const DeviceGeneratorImplBase&) = delete;
  DeviceGeneratorImplBase& operator=(const DeviceGeneratorImplBase&) = delete;
  DeviceGeneratorImplBase(DeviceGeneratorImplBase&&) = delete;
  DeviceGeneratorImplBase& operator=(DeviceGeneratorImplBase&&) = delete;

  ~DeviceGeneratorImplBase() override = default;

  // Advances the underlying RNG state tensor by the specified number of random
  // bits consumed by a kernel execution.
  absl::Status AdvanceDeviceStateTensor(int64_t num_elements,
                                        int64_t bit_width);

  // Inherited read accessors.
  uint64_t get_offset() const override = 0;
  uint64_t current_seed() const override = 0;
  c10::intrusive_ptr<c10::TensorImpl> get_state() const override = 0;
  c10::intrusive_ptr<c10::GeneratorImpl> graphsafe_get_state() const override =
      0;

  // Returns the active TPU RNG state tensor {seed, offset}.
  virtual at::Tensor DeviceStateTensor() const = 0;

  static c10::DeviceType device_type();

  absl::Status CheckDeviceStateTensor(const at::Tensor& rng_state) const;

 protected:
  explicit DeviceGeneratorImplBase(c10::DeviceIndex device_index = -1);

 private:
  template <typename DispatchFunc>
  friend absl::StatusOr<std::vector<DeviceBufferRef>> DispatchRngOpGeneral(
      c10::optional<at::Generator> generator, DispatchFunc dispatch_func,
      const std::optional<RngUsage>& usage);

  friend class EagerModeDeviceGeneratorImpl;
  friend class ExportModeDeviceGeneratorImpl;

  friend at::Tensor PyGetDeviceStateTensor(at::Generator gen);
  friend void PySetDeviceStateTensor(at::Generator gen, at::Tensor rng_state);

  DeviceGeneratorImplBase* clone_impl() const override = 0;

  // The following RNG state accessors are private as it's dangerous to modify
  // the state. Only approved friends are allowed to call them.

  uint64_t seed() override = 0;
  void set_current_seed(uint64_t seed) override = 0;
  void set_offset(uint64_t offset) override = 0;
  void set_state(const c10::TensorImpl& new_state) override = 0;
  void graphsafe_set_state(
      const c10::intrusive_ptr<c10::GeneratorImpl>& new_state) override = 0;

  // Overwrites the active TPU RNG state tensor.
  virtual absl::Status SetDeviceStateTensor(at::Tensor t) = 0;
};

// Concrete generator implementation used during normal eager execution.
// Throttles host-device synchronization and manages state via
// DeviceGeneratorState.
class EagerModeDeviceGeneratorImpl : public DeviceGeneratorImplBase {
 public:
  explicit EagerModeDeviceGeneratorImpl(c10::DeviceIndex device_index = -1);
  EagerModeDeviceGeneratorImpl(c10::DeviceIndex device_index,
                               at::Tensor rng_state);
  EagerModeDeviceGeneratorImpl(c10::DeviceIndex device_index,
                               c10::intrusive_ptr<DeviceGeneratorState> state);

  at::Tensor DeviceStateTensor() const override;

  // Implement eager RNG read accessors.
  uint64_t get_offset() const override;
  uint64_t current_seed() const override;
  c10::intrusive_ptr<c10::TensorImpl> get_state() const override;
  c10::intrusive_ptr<c10::GeneratorImpl> graphsafe_get_state() const override;

 private:
  DeviceGeneratorImplBase* clone_impl() const override;

  // The following RNG state accessors are private as it's dangerous to modify
  // the state. Only approved friends are allowed to call them.

  uint64_t seed() override;
  void set_current_seed(uint64_t seed) override;
  void set_offset(uint64_t offset) override;
  void set_state(const c10::TensorImpl& new_state) override;
  void graphsafe_set_state(
      const c10::intrusive_ptr<c10::GeneratorImpl>& new_state) override;

  absl::Status SetDeviceStateTensor(at::Tensor t) override;

  c10::intrusive_ptr<DeviceGeneratorState> state_;
};

// Concrete generator implementation used exclusively during graph export
// tracing. Manages symbolic MLIR placeholder state tensors via
// ExportModeDeviceGeneratorState and triggers fatal check failures if
// unreachable eager accessors are invoked.
class ExportModeDeviceGeneratorImpl : public DeviceGeneratorImplBase {
 public:
  // Constructs an export mode generator by cloning the underlying state tensor
  // of the incoming eager default generator.
  explicit ExportModeDeviceGeneratorImpl(
      const at::Generator& default_generator);

  at::Tensor DeviceStateTensor() const override;

  // Export mode does not support eager read accessors.
  // Implemented to trigger fatal check failures.
  uint64_t get_offset() const override;
  uint64_t current_seed() const override;
  c10::intrusive_ptr<c10::TensorImpl> get_state() const override;
  c10::intrusive_ptr<c10::GeneratorImpl> graphsafe_get_state() const override;

 private:
  friend class c10::intrusive_ptr<ExportModeDeviceGeneratorImpl>;

  ExportModeDeviceGeneratorImpl(
      c10::DeviceIndex device_index,
      c10::intrusive_ptr<ExportModeDeviceGeneratorState> state);

  DeviceGeneratorImplBase* clone_impl() const override;

  // The following RNG state accessors are private as it's dangerous to modify
  // the state. Only approved friends are allowed to call them.

  uint64_t seed() override;
  void set_current_seed(uint64_t seed) override;
  void set_offset(uint64_t offset) override;
  void set_state(const c10::TensorImpl& new_state) override;
  void graphsafe_set_state(
      const c10::intrusive_ptr<c10::GeneratorImpl>& new_state) override;

  absl::Status SetDeviceStateTensor(at::Tensor device_state_tensor) override;

  c10::intrusive_ptr<ExportModeDeviceGeneratorState> export_state_;
};

// Convenience functions.

// Returns the default generator for the given device index. If idx is -1,
// returns the default generator for the current device.
// If ExportMode is enabled, returns the export-mode default generator or
// crashes if not set, otherwise returns the eager default generator.
at::Generator& GetDefaultDeviceGenerator(c10::DeviceIndex idx = -1);

// Sets the export-mode default generator for override.  If idx is -1, sets the
// export-mode default generator for the current device.
// When ExportMode is enabled, GetDefaultDeviceGenerator() will return this
// overridden generator instead of the standard eager generator.
void SetExportModeDefaultDeviceGenerator(c10::optional<at::Generator> gen,
                                         c10::DeviceIndex idx = -1);

// Clears all export-mode default generator overrides, restoring the default
// eager generators for subsequent GetDefaultDeviceGenerator() calls under
// export mode.
void ClearExportModeDefaultDeviceGenerators();

// Creates a new generator for the given device index. If idx is -1, creates a
// new generator for the current device.
at::Generator MakeDeviceGenerator(c10::DeviceIndex idx = -1);

// Sets the seed for the generator of the given device index. If idx is -1,
// sets the seed for the generator of the current device.
void SetManualSeed(uint64_t seed, c10::DeviceIndex idx = -1);

// Sets the seed for all TPU generators.
void SetManualSeedAll(uint64_t seed);

}  // namespace torch_tpu

#endif  // TORCH_TPU_EAGER_DEVICE_GEN_IMPL_H_
