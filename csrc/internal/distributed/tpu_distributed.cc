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

#include <chrono>  // NOLINT - needed for pybind11 integration
#include <cstddef>
#include <cstdint>
#include <string>
#include <variant>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/log/absl_check.h"
#include "absl/log/check.h"
#include "absl/strings/ascii.h"
#include "c10/util/intrusive_ptr.h"
#include "csrc/common/error_utils.h"
#include "csrc/common/pybind_error_utils.h"
#include "csrc/distributed/process_group_tpu.h"
#include "csrc/distributed/utils.h"
#include "csrc/pjrt/pjrt_state.h"
#include "pybind11/chrono.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "torch/csrc/distributed/c10d/Backend.hpp"
#include "torch/csrc/distributed/c10d/Store.hpp"
#include "torch/csrc/utils/pybind.h"  // IWYU pragma: keep or collective_ops_test fails
#include "xla/pjrt/pjrt_client.h"

namespace torch_tpu {

//
// These are only intended to be called from Python.
//
namespace {

// Direct constructor for independent TPU process groups (such as
// custom-store process groups and `TorchCommsTPU`).
c10::intrusive_ptr<c10d::Backend> MakeProcessGroup(
    const c10::intrusive_ptr<c10d::Store>& store, int rank, int size,
    std::chrono::duration<float>& /*timeout*/) {
  return c10::make_intrusive<ProcessGroupTpu>(
      store, rank, size, ProcessGroupTpu::GroupKind::kIndependent);
}

// Registered as the TPU c10d backend creator via
// `torch.distributed.Backend.register_backend(..., extended_api=True)`.
// Classifies default WORLD, SPMD world subgroups, and independent process
// groups (such as `new_group(..., use_local_synchronization=True)`).
c10::intrusive_ptr<c10d::Backend> MakeC10dProcessGroup(
    pybind11::handle dist_backend_opts, pybind11::handle /*pg_options*/) {
  PyBindModule dist = PyBindModule::Import("torch.distributed");
  const bool is_default_world = !dist.attr("is_initialized")().cast<bool>();
  const bool has_global_ranks =
      pybind11::len(dist_backend_opts.attr("global_ranks_in_group")) > 0;
  const std::string group_id =
      dist_backend_opts.attr("group_id").cast<std::string>();
  // Standard SPMD `new_group`, `new_subgroups`, and `DeviceMesh` assign
  // sequential decimal IDs (`str(_world.group_count)`), whereas
  // `new_group(..., use_local_synchronization=True)` assigns a 40-char
  // SHA-1 hex digest.
  constexpr size_t kSha1HexDigestLen = 40;
  const bool is_sequential_group_id =
      !group_id.empty() && group_id.size() < kSha1HexDigestLen &&
      absl::c_all_of(group_id, absl::ascii_isdigit);

  ProcessGroupTpu::GroupKind kind = ProcessGroupTpu::GroupKind::kIndependent;
  if (is_default_world) {
    kind = ProcessGroupTpu::GroupKind::kDefaultWorld;
  } else if (has_global_ranks && is_sequential_group_id) {
    kind = ProcessGroupTpu::GroupKind::kWorldSubgroup;
  }

  c10::intrusive_ptr<c10d::Store> store =
      dist_backend_opts.attr("store").cast<c10::intrusive_ptr<c10d::Store>>();
  const int rank = dist_backend_opts.attr("group_rank").cast<int>();
  const int size = dist_backend_opts.attr("group_size").cast<int>();
  return c10::make_intrusive<ProcessGroupTpu>(store, rank, size, kind);
}

std::vector<int64_t> GetRankToDeviceId(  // INT_VEC_OK
    c10::intrusive_ptr<c10d::Backend> backend) {
  auto* pg_tpu = dynamic_cast<ProcessGroupTpu*>(backend.get());
  if (pg_tpu != nullptr) {
    return pg_tpu->rank_to_device_id();
  }
  return {};
}

// Global device id of the (single) addressable device this process is using.
int GetGlobalDeviceId() {
  TT_ASSIGN_OR_THROW(int device_id,
                     PjrtBackend::GetInstance().GetGlobalDeviceId(),
                     _ << "failed to get global device ID");
  return device_id;
}

std::vector<int> GetAllGlobalDeviceIds() {
  TT_ASSIGN_OR_THROW(const xla::PjRtClient* const pjrt_client,
                     PjrtBackend::GetInstance().GetClient());

  const auto& devices = pjrt_client->devices();
  std::vector<int> global_device_ids;

  for (const auto& device : devices) {
    ABSL_CHECK(device != nullptr) << "Got a nullptr PjRtDevice.";  // CRASH_OK
    global_device_ids.push_back(device->global_device_id().value());
  }
  return global_device_ids;
}

std::string DeviceDebugString() {
  TT_ASSIGN_OR_THROW(const xla::PjRtDevice* const pjrtdev,
                     PjrtBackend::GetInstance().GetDevice());
  return std::string(pjrtdev->DebugString());
}

// Returns physical grid coordinates (e.g. [x, y, z]) of the local PJRT device
// as reported by the PJRT runtime "coords" attribute. Returns an empty vector
// if coordinates are absent or unavailable.
std::vector<int64_t> GetLocalDeviceCoords() {  // INT_VEC_OK
  TT_ASSIGN_OR_THROW(const xla::PjRtDevice* const pjrtdev,
                     PjrtBackend::GetInstance().GetDevice());
  const auto& attrs = pjrtdev->Attributes();
  auto it = attrs.find("coords");
  if (it != attrs.end()) {
    if (const auto* v =
            std::get_if<std::vector<int64_t>>(&it->second)) {  // INT_VEC_OK
      return *v;
    }
  }
  return {};
}

}  // namespace

TT_PYBIND11_MODULE(tpu_distributed, m) {
  // Custom distributed backend initialization happens from the Python side,
  // so we need to expose this init function.
  m.def("create_process_group", &MakeProcessGroup);
  m.def("create_process_group", &MakeC10dProcessGroup);
  m.def("rank_to_device_id", &GetRankToDeviceId);

  // We expose these as helper functions on the Python side
  m.def("global_device_id", &GetGlobalDeviceId);
  m.def("all_global_device_ids", &GetAllGlobalDeviceIds);
  m.def("device_debug_string", &DeviceDebugString);
  m.def("local_device_coords", &GetLocalDeviceCoords);
  PyBindClass<PjRtDeviceAttributes>(m, "PjRtDeviceAttributes")
      .def_readonly("id", &PjRtDeviceAttributes::id)
      .def_readonly("coords", &PjRtDeviceAttributes::coords)
      .def_readonly("core_on_chip", &PjRtDeviceAttributes::core_on_chip)
      .def_readonly("process_index", &PjRtDeviceAttributes::process_index)
      .def_readonly("slice_index", &PjRtDeviceAttributes::slice_index)
      .def_readonly("device_kind", &PjRtDeviceAttributes::device_kind);
  m.def("global_device_attributes", &GetGlobalDeviceAttributes);

  // Expose this functions to allow entering and exiting the SPMD safe region
  // from python
  m.def("enter_spmd_safe_region", &EnterSpmdSafeRegion);
  m.def("exit_spmd_safe_region", &ExitSpmdSafeRegion);
}

}  // namespace torch_tpu
