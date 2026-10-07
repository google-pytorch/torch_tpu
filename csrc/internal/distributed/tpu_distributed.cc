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
#include <cstdint>
#include <string>
#include <variant>
#include <vector>

#include "absl/log/absl_check.h"
#include "absl/log/check.h"
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
#include "torch/csrc/utils/pybind.h"  // IWYU pragma: keep or multi_tpu_test fails
#include "xla/pjrt/pjrt_client.h"

namespace torch_tpu {

//
// These are only intended to be called from Python.
//
namespace {

// This function will be registered as the TPU distributed backend init
// function (see `torch.distributed.Backend.register_backend`) and called by
// pytorch when the user calls `torch.distributed.init_process_group()`.
c10::intrusive_ptr<c10d::Backend> MakeProcessGroup(
    c10::intrusive_ptr<c10d::Store> store, int rank, int size,
    std::chrono::duration<float>& timeout) {
  return c10::make_intrusive<ProcessGroupTpu>(store, rank, size);
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
