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

// Load the AOT compiler, collect kernel families, and write one bundle.

#include <fstream>
#include <iostream>
#include <memory>
#include <optional>
#include <string>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/status/status.h"
#include "csrc/common/env_vars.h"
#include "csrc/common/error_utils.h"
#include "csrc/common/flags.h"
#include "csrc/experimental/precompiled_kernels/aot/elementwise/compile.h"
#include "csrc/experimental/precompiled_kernels/bundle.pb.h"
#include "xla/pjrt/c_api_client/pjrt_c_api_client.h"
#include "xla/pjrt/pjrt_api.h"
#include "xla/pjrt/pjrt_compiler.h"

ABSL_FLAG(std::string, torch_tpu_internal_output, "",
          "Where to write the Bundle.");
ABSL_FLAG(std::string, torch_tpu_internal_dump, "",
          "If set, a directory to write each kernel's Mosaic text into.");
ABSL_FLAG(std::string, torch_tpu_internal_topology, "tpu7x:2x2",
          "Chip topology as <chip>:<mesh>; the mesh must cover a whole host.");

namespace torch_tpu {
using precompiled_kernels::proto::Bundle;

absl::Status Generate() {
  const std::optional<std::string>& library =
      torch_tpu::GetEnvOnce<kTpuLibraryPathEnvVar>();
  if (!library.has_value() || library->empty()) {
    return TT_ERROR(error::kFailedPrecondition)
           << "TPU_LIBRARY_PATH must name the libtpu.so to compile with";
  }
  if (auto loaded = pjrt::LoadPjrtPlugin("tpu", *library);
      !loaded.ok() && !absl::IsAlreadyExists(loaded.status())) {
    return loaded.status();
  }
  TT_ASSIGN_OR_RETURN(bool initialized, pjrt::IsPjrtPluginInitialized("tpu"));
  if (!initialized) TT_RETURN_IF_ERROR(pjrt::InitializePjrtPlugin("tpu"));
  TT_ASSIGN_OR_RETURN(std::unique_ptr<xla::PjRtCompiler> compiler,
                      xla::GetCApiCompiler("tpu"));

  Bundle bundle;
  bundle.set_format_version(1);
  const std::string& topology_flag =
      torch_tpu::GetFlagOnce<std::string, &FLAGS_torch_tpu_internal_topology>();
  TT_ASSIGN_OR_RETURN(auto topology,
                      xla::GetCApiTopology("tpu", topology_flag, {}));
  const auto devices = topology->DeviceDescriptions();
  if (devices.empty()) {
    return TT_ERROR(error::kInternal) << "topology describes no device";
  }
  bundle.set_device_kind(std::string(devices[0]->device_kind()));
  bundle.set_platform_version(std::string(topology->platform_version()));
  const std::string& dump_flag =
      torch_tpu::GetFlagOnce<std::string, &FLAGS_torch_tpu_internal_dump>();
  TT_RETURN_IF_ERROR(AddKernels(*compiler, *topology, dump_flag, bundle));
  const std::string& output_flag =
      torch_tpu::GetFlagOnce<std::string, &FLAGS_torch_tpu_internal_output>();
  std::ofstream out(output_flag, std::ios::binary);
  if (!out || !bundle.SerializeToOstream(&out)) {
    return TT_ERROR(error::kInternal) << "failed to write the bundle";
  }
  out.close();
  if (!out) {
    return TT_ERROR(error::kInternal) << "bundle write failed";
  }
  return absl::OkStatus();
}

}  // namespace torch_tpu

int main(int argc, char** argv) {
  absl::ParseCommandLine(argc, argv);
  if (absl::Status status = torch_tpu::Generate(); !status.ok()) {
    std::cerr << status << "\n";
    return 1;
  }
  return 0;
}
