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

#include "torch_tpu/ops/experimental/send_recv_kernels.h"

#include <cstdint>
#include <vector>

#include "ATen/core/ATen_fwd.h"
#include "ATen/core/ivalue.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "c10/util/intrusive_ptr.h"
#include "torch/csrc/distributed/c10d/Backend.hpp"
#include "torch/csrc/distributed/c10d/GroupRegistry.hpp"
#include "torch/csrc/distributed/c10d/ProcessGroup.hpp"
#include "torch/csrc/distributed/c10d/Work.hpp"
#include "torch/headeronly/core/DeviceType.h"
#include "torch_tpu/common/cache_key.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/distributed/process_group_tpu.h"
#include "torch_tpu/ops/macros/kernel.h"
#include "torch_tpu/ops/op_names.h"

namespace torch_tpu {

namespace {

absl::StatusOr<ProcessGroupTpu*> GetProcessGroupTpu() {
  constexpr char kDefaultProcessGroupName[] = "0";

  c10::intrusive_ptr<c10d::ProcessGroup> pg =
      c10d::resolve_process_group(kDefaultProcessGroupName);
  TT_RET_CHECK(pg != nullptr, error::kInternal)
      << "failed to resolve default process group";
  c10::intrusive_ptr<c10d::Backend> backend =
      pg->getBackend(c10::DeviceType::PrivateUse1);
  TT_RET_CHECK(backend != nullptr, error::kInternal)
      << "failed to get backend for tpu device";

  auto process_group_tpu = dynamic_cast<ProcessGroupTpu*>(backend.get());
  TT_RET_CHECK(process_group_tpu != nullptr, error::kInternal)
      << "failed to cast c10d::Backend to ProcessGroupTpu";
  return process_group_tpu;
}

}  // namespace

c10::IValue TorchTpuExperimentalSend(at::ITensorListRef tensors, int64_t dst,
                                     int64_t tag) {
  TT_KERNEL(OpName::kDistributedExperimentalSend, _,
            (tensors, IgnoreInCacheKey(dst, "no op being dispatched"),
             IgnoreInCacheKey(tag, "no op being dispatched")),
            {
              TT_ASSIGN_OR_THROW(auto pg, GetProcessGroupTpu());
              std::vector<at::Tensor> tensors_vec(tensors.begin(),
                                                  tensors.end());
              c10::intrusive_ptr<c10d::Work> work =
                  pg->experimental_send(tensors_vec, dst, tag);
              return c10::IValue(work);
            });
}

c10::IValue TorchTpuExperimentalRecv(at::ITensorListRef tensors, int64_t src,
                                     int64_t tag) {
  TT_KERNEL(OpName::kDistributedExperimentalRecv, _,
            (tensors, IgnoreInCacheKey(src, "no op being dispatched"),
             IgnoreInCacheKey(tag, "no op being dispatched")),
            {
              TT_ASSIGN_OR_THROW(auto pg, GetProcessGroupTpu());
              std::vector<at::Tensor> tensors_vec(tensors.begin(),
                                                  tensors.end());
              c10::intrusive_ptr<c10d::Work> work =
                  pg->experimental_recv(tensors_vec, src, tag);
              return c10::IValue(work);
            });
}

}  // namespace torch_tpu
