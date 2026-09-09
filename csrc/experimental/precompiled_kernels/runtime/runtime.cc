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

#include "csrc/experimental/precompiled_kernels/runtime/runtime.h"

#include <algorithm>
#include <chrono>  // NOLINT(build/c++11)
#include <cstddef>
#include <cstdint>
#include <future>  // NOLINT(build/c++11)
#include <iterator>
#include <limits>
#include <memory>
#include <optional>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/base/no_destructor.h"
#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/log/absl_log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "csrc/common/compilation.h"
#include "csrc/common/dimension_types.h"
#include "csrc/common/dtype.h"
#include "csrc/common/error_utils.h"
#include "csrc/common/shape.h"
#include "csrc/eager/device_buffer.h"
#include "csrc/eager/traversal.h"
#include "csrc/ops/op_names.h"
#include "csrc/pjrt/pjrt_state.h"
#include "csrc/pjrt/pjrt_utils.h"
#include "tsl/profiler/lib/traceme.h"
#include "xla/future.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/pjrt/pjrt_layout.h"
#include "xla/shape_util.h"

namespace torch_tpu {
using precompiled_kernels::proto::Bundle;
using precompiled_kernels::proto::Dimension;
using precompiled_kernels::proto::Kernel;
using precompiled_kernels::proto::Parameter;
using precompiled_kernels::proto::Parameter_Role_IsValid;
using precompiled_kernels::proto::Tensor;

Catalog::Catalog(const Bundle& bundle) : bundle_(bundle) {
  for (const auto& kernel : bundle.kernels()) {
    entries_[kernel.op()].push_back(&kernel);
  }
}

absl::Span<const Kernel* const> Catalog::Find(std::string_view op) const {
  auto it = entries_.find(op);
  if (it == entries_.end()) return {};
  return it->second;
}

namespace {

// Checks if a dimension matches the expected size or consistent symbol binding.
bool MatchesDimension(const Dimension& dim, int64_t size,
                      absl::flat_hash_map<std::string_view, int64_t>& symbols) {
  if (size <= 0 || size > std::numeric_limits<int32_t>::max()) return false;
  if (dim.has_size()) return dim.size() == size;
  if (dim.has_symbol() && !dim.symbol().empty()) {
    auto [it, inserted] = symbols.emplace(dim.symbol(), size);
    return inserted || it->second == size;
  }
  return false;
}

// Matches a tensor specification against runtime tensor properties.
bool Fits(const Tensor& spec, const DeviceBufferRef& tensor,
          absl::flat_hash_map<std::string_view, int64_t>& symbols) {
  if (spec.dtype() != ConvertTo<xla::PrimitiveType>(tensor.element_type()) ||
      spec.dims_size() != tensor.dimensions().size()) {
    return false;
  }
  for (int i = 0; i < spec.dims_size(); ++i) {
    if (!MatchesDimension(spec.dims(i), tensor.dimensions()[i], symbols)) {
      return false;
    }
  }
  return true;
}

// Checks if XLA compilation is ready (or has finished with error).
bool XlaReady(const CompiledKernel& kernel) {
  const auto ready = [](const auto& future) {
    return !future.valid() || future.wait_for(std::chrono::seconds(0)) ==
                                  std::future_status::ready;
  };
  const auto& adapter = kernel.dynamic_kernel_adapter;
  return ready(kernel.fixed_shape_kernel) &&
         (!adapter || (ready(adapter->preamble) && ready(adapter->postamble)));
}

struct Loaded {
  std::shared_ptr<xla::PjRtClient> client;
  std::unique_ptr<xla::PjRtLoadedExecutable> executable;
  std::vector<std::shared_ptr<const xla::PjRtLayout>> layouts;
};

class ExecutableCache {
 public:
  using Result = absl::StatusOr<std::shared_ptr<Loaded>>;

  struct CacheKey {
    std::shared_ptr<xla::PjRtClient> client;
    const Kernel* kernel;

    template <typename H>
    friend H AbslHashValue(H h, const CacheKey& key) {
      return H::combine(std::move(h), key.client.get(), key.kernel);
    }
    bool operator==(const CacheKey& other) const = default;
  };

  Result GetOrCreate(const Kernel& kernel,
                     std::shared_ptr<xla::PjRtClient> client) {
    absl::MutexLock lock(mutex_);
    const CacheKey key{.client = client, .kernel = &kernel};
    if (auto it = cache_.find(key); it != cache_.end()) return it->second;
    auto result = LoadInternal(kernel, std::move(client));
    return cache_.emplace(key, std::move(result)).first->second;
  }

  void Clear() {
    absl::MutexLock lock(mutex_);
    cache_.clear();
  }

 private:
  static Result LoadInternal(const Kernel& kernel,
                             std::shared_ptr<xla::PjRtClient> client) {
    const int inputs = kernel.inputs_size();
    const int tensors = inputs + kernel.outputs_size();
    for (const auto& p : kernel.parameters()) {
      if (p.tensor() >= tensors ||
          (p.role() == Parameter::INPUT && p.tensor() >= inputs) ||
          (p.role() == Parameter::OUTPUT && p.tensor() < inputs) ||
          !Parameter_Role_IsValid(p.role())) {
        return TT_ERROR(error::kInvalidArgument) << "invalid tensor binding";
      }
    }
    auto loaded = std::make_shared<Loaded>();
    loaded->client = std::move(client);
    auto executable_or = loaded->client->LoadSerializedExecutable(
        kernel.executable(), std::nullopt, xla::LoadOptions());
    if (!executable_or.ok()) {
      ABSL_LOG_EVERY_N_SEC(WARNING, 30)
          << "Failed to load precompiled kernel '" << kernel.op()
          << "': " << executable_or.status();
      return executable_or.status();
    }
    loaded->executable = *std::move(executable_or);
    auto layouts_or = loaded->executable->GetParameterLayouts();
    if (!layouts_or.ok()) {
      ABSL_LOG_EVERY_N_SEC(WARNING, 30)
          << "Failed to get layouts for precompiled kernel '" << kernel.op()
          << "': " << layouts_or.status();
      return layouts_or.status();
    }
    loaded->layouts = *std::move(layouts_or);
    if (loaded->layouts.size() != kernel.parameters_size()) {
      return TT_ERROR(error::kFailedPrecondition)
             << "kernel parameter count changed";
    }
    return loaded;
  }

  absl::Mutex mutex_;
  absl::flat_hash_map<CacheKey, Result> cache_ ABSL_GUARDED_BY(mutex_);
};

ExecutableCache& GetExecutableCache() {
  static absl::NoDestructor<ExecutableCache> cache;
  return *cache;
}

absl::StatusOr<std::shared_ptr<Loaded>> Load(
    const Kernel& kernel, std::shared_ptr<xla::PjRtClient> client) {
  return GetExecutableCache().GetOrCreate(kernel, std::move(client));
}

struct Invocation {
  std::shared_ptr<Loaded> loaded;
  PjRtBufferPointers owned;
  std::vector<xla::PjRtBuffer*> operands;
  xla::PjRtDevice* device;
  xla::ExecuteOptions options;
};

// Validates input tensors, shapes, layouts, and device assignment.
absl::Status ValidateInputs(const Kernel& kernel,
                            std::shared_ptr<Loaded> loaded,
                            absl::Span<xla::PjRtBuffer* const> inputs,
                            absl::Span<const Dimensions> shapes,
                            xla::PjRtDevice* device) {
  const auto devices = loaded->executable->addressable_devices();
  if (absl::c_find(devices, device) == devices.end()) {
    return TT_ERROR(error::kFailedPrecondition)
           << "device outside kernel assignment";
  }
  for (int i = 0; i < inputs.size(); ++i) {
    const auto& shape = inputs[i]->on_device_shape();
    if (inputs[i]->client() != loaded->client.get() ||
        inputs[i]->device() != device || shape.is_dynamic() ||
        shape.element_type() != kernel.inputs(i).dtype() ||
        shape.dimensions() != absl::MakeConstSpan(shapes[i])) {
      return TT_ERROR(error::kFailedPrecondition)
             << "input shape or device mismatch";
    }
  }
  for (int i = 0; i < kernel.parameters_size(); ++i) {
    const auto& p = kernel.parameters(i);
    if (p.role() == Parameter::INPUT &&
        inputs[p.tensor()]->on_device_shape().layout() !=
            loaded->layouts[i]->xla_layout()) {
      return TT_ERROR(error::kFailedPrecondition) << "input layout mismatch";
    }
  }
  return absl::OkStatus();
}

// Allocates a private output buffer or host dimension buffer for a parameter.
absl::StatusOr<std::unique_ptr<xla::PjRtBuffer>> AllocateParameterBuffer(
    const Parameter& p, const Kernel& kernel, xla::PjRtClient& client,
    xla::PjRtMemorySpace* memory, const xla::Layout& layout,
    const Dimensions& dims) {
  if (p.role() == Parameter::OUTPUT) {
    auto shape = xla::ShapeUtil::MakeShape(
        kernel.outputs(p.tensor() - kernel.inputs_size()).dtype(), dims);
    *shape.mutable_layout() = layout;
    return client.CreateUninitializedBuffer(shape, memory);
  }
  const std::vector<int32_t> extents(dims.begin(), dims.end());
  return client.BufferFromHostBuffer(
      extents.data(), xla::S32, {static_cast<int64_t>(dims.size())},
      std::nullopt,
      xla::PjRtClient::HostBufferSemantics::kImmutableOnlyDuringCall, [] {},
      memory, &layout);
}

// Prepares an executable invocation, creating output and dimension buffers.
absl::StatusOr<std::shared_ptr<Invocation>> Prepare(
    const Kernel& kernel, std::shared_ptr<Loaded> loaded,
    absl::Span<xla::PjRtBuffer* const> inputs,
    absl::Span<const Dimensions> shapes) {
  auto call = std::make_shared<Invocation>();
  call->loaded = std::move(loaded);
  auto& client = *call->loaded->client;
  call->device = inputs.empty() ? client.addressable_devices().front()
                                : inputs[0]->device();
  TT_RETURN_IF_ERROR(
      ValidateInputs(kernel, call->loaded, inputs, shapes, call->device));
  TT_ASSIGN_OR_RETURN(auto* memory, call->device->default_memory_space());
  call->options.strict_shape_checking = false;
  for (int i = 0; i < kernel.parameters_size(); ++i) {
    const auto& p = kernel.parameters(i);
    const auto& layout = call->loaded->layouts[i]->xla_layout();
    const auto& dims = shapes[p.tensor()];
    if (p.role() != Parameter::OUTPUT) {
      call->options.non_donatable_input_indices.insert(i);
    }
    if (p.role() == Parameter::INPUT) {
      call->operands.push_back(inputs[p.tensor()]);
    } else {
      TT_ASSIGN_OR_RETURN(
          auto buffer,
          AllocateParameterBuffer(p, kernel, client, memory, layout, dims));
      call->operands.push_back(buffer.get());
      call->owned.push_back(std::move(buffer));
    }
  }
  return call;
}

absl::StatusOr<PjRtBufferPointers> Submit(std::shared_ptr<Invocation> call,
                                          const Kernel& kernel,
                                          absl::Span<const Dimensions> shapes) {
  tsl::profiler::TraceMe trace("PrecompiledExecute");
  std::optional<xla::Future<>> done;
  auto results = call->loaded->executable->ExecuteSharded(
      call->operands, call->device, call->options, done, /*fill_future=*/true);
  if (done) done->OnReady([call](absl::Status) {});
  if (!results.ok()) {
    return TT_ERROR(error::kInternal)
           << "Precompiled kernel execution failed for op " << kernel.op()
           << ": " << results.status().message();
  }
  if (!done) {
    for (auto& result : *results) {
      result->GetReadyFuture().OnReady([call](absl::Status) {});
    }
  }
  if (results->size() != kernel.outputs_size()) {
    return TT_ERROR(error::kInternal)
           << "precompiled output count mismatch for op " << kernel.op();
  }
  for (int i = 0; i < results->size(); ++i) {
    const auto& shape = (*results)[i]->on_device_shape();
    if (shape.is_dynamic() ||
        shape.element_type() != kernel.outputs(i).dtype() ||
        shape.dimensions() !=
            absl::MakeConstSpan(shapes[kernel.inputs_size() + i])) {
      return TT_ERROR(error::kInternal)
             << "precompiled output shape mismatch for op " << kernel.op();
    }
  }
  return results;
}

// Checks if a candidate kernel matches operation inputs and outputs.
bool MatchesCandidate(const Kernel& kernel, const DeferredOp& op,
                      absl::Span<const DeviceBufferRef> outputs) {
  if (kernel.inputs_size() != op.inputs().size() ||
      kernel.outputs_size() != outputs.size()) {
    return false;
  }
  absl::flat_hash_map<std::string_view, int64_t> symbols;
  for (int i = 0; i < op.inputs().size(); ++i) {
    if (!Fits(kernel.inputs(i), op.inputs()[i], symbols)) return false;
  }
  for (int i = 0; i < outputs.size(); ++i) {
    if (!Fits(kernel.outputs(i), outputs[i], symbols)) return false;
  }
  return true;
}

// Verifies hardware kind and platform version against the bundle.
bool PlatformMatches(const Bundle& bundle, const xla::PjRtClient& client,
                     const xla::PjRtDevice& device) {
  if (bundle.device_kind() != device.device_kind() ||
      bundle.platform_version() != client.platform_version()) {
    ABSL_LOG_EVERY_N_SEC(WARNING, 30)
        << "Precompiled bundle platform mismatch: bundle targets "
        << bundle.device_kind() << " (" << bundle.platform_version()
        << "), but runtime has " << device.device_kind() << " ("
        << client.platform_version() << ")";
    return false;
  }
  return true;
}

}  // namespace

void ClearPrecompiledKernelCache() { GetExecutableCache().Clear(); }

Execution MatchPrecompiledKernel(const Traversal& traversal,
                                 const Catalog& catalog) {
  // New attribute-sensitive conventions need an explicit matcher here.
  if (catalog.bundle().format_version() != 1 ||
      traversal.execution_order().size() != 1 || traversal.IsBoundedDynamic() ||
      traversal.core_pinning_mode() != CorePinningMode::kUnpinned ||
      !IsSimpleNodeTraversal(traversal)) {
    return {};
  }
  auto op = traversal.execution_order()[0]->deferred_op();
  if (!op || !op->donated_indices().empty()) return {};
  std::vector<const Kernel*> candidates;
  for (const auto* kernel : catalog.Find(ToString(op->op_name()))) {
    if (MatchesCandidate(*kernel, *op, traversal.outputs())) {
      candidates.push_back(kernel);
    }
  }
  if (candidates.empty()) return {};
  std::vector<size_t> indices;
  std::vector<Dimensions> shapes;
  const auto arguments = traversal.arguments();
  for (const auto& input : op->inputs()) {
    auto it = absl::c_find(arguments, input);
    if (it == arguments.end()) return {};
    indices.push_back(std::distance(arguments.begin(), it));
    shapes.emplace_back(input.dimensions().begin(), input.dimensions().end());
  }
  for (const auto& output : traversal.outputs()) {
    shapes.emplace_back(output.dimensions().begin(), output.dimensions().end());
  }
  return [bundle = &catalog.bundle(), candidates = std::move(candidates),
          indices = std::move(indices), shapes = std::move(shapes)](
             const CompiledKernel& xla_kernel,
             absl::Span<const DeviceBufferRef> arguments)
             -> absl::StatusOr<std::optional<PjRtBufferPointers>> {
    if (XlaReady(xla_kernel)) return std::nullopt;
    std::vector<xla::PjRtBuffer*> inputs;
    inputs.reserve(indices.size());
    for (auto index : indices) {
      TT_ASSIGN_OR_RETURN(auto* buffer, arguments[index].AwaitBuffer());
      inputs.push_back(buffer);
    }
    if (XlaReady(xla_kernel)) return std::nullopt;
    auto client_or = PjrtBackend::GetInstance().GetSharedClient();
    if (!client_or.ok() || (*client_or)->addressable_devices().empty()) {
      return std::nullopt;
    }
    auto client = *std::move(client_or);
    auto* device = inputs.empty() ? client->addressable_devices().front()
                                  : inputs[0]->device();
    if (!PlatformMatches(*bundle, *client, *device)) return std::nullopt;
    for (const auto* kernel : candidates) {
      if (XlaReady(xla_kernel)) return std::nullopt;
      auto loaded = Load(*kernel, client);
      if (!loaded.ok()) continue;
      auto call = Prepare(*kernel, std::move(*loaded), inputs, shapes);
      if (!call.ok()) continue;
      if (XlaReady(xla_kernel)) return std::nullopt;
      // Never decline once a kernel has been submitted.
      TT_ASSIGN_OR_RETURN(auto results,
                          Submit(std::move(*call), *kernel, shapes));
      return std::optional<PjRtBufferPointers>(std::move(results));
    }
    return std::nullopt;
  };
}

}  // namespace torch_tpu
