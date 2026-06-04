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
#include "torch_tpu/_internal/sync/sync.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "ATen/core/TensorBody.h"
#include "absl/base/nullability.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/flags/declare.h"
#include "absl/log/absl_log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Support/DebugStringHelper.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/common/flags.h"
#include "torch_tpu/eager/device_buffer.h"
#include "torch_tpu/eager/events_queue.h"
#include "torch_tpu/eager/materialize.h"
#include "torch_tpu/eager/structured_log_buffer.h"
#include "torch_tpu/eager/tensor_to_buffer.h"
#include "torch_tpu/eager/traversal.h"
#include "torch_tpu/experimental/eager/materialize_new.h"
#include "xla/hlo/translate/register.h"

ABSL_DECLARE_FLAG(bool, torch_tpu_internal_enable_new_materialization);

namespace torch_tpu {

using BufferRefToVarMap = absl::flat_hash_map<DeviceBufferRef, std::string>;

absl::Status SynchronizeTensors(absl::Span<const at::Tensor> tensors) {
  // Materialize the tensors and get their DeviceBufferRefs.
  // MaterializeAndReturn() will materialize each base tensor and each view.
  TT_ASSIGN_OR_RETURN(
      std::vector<DeviceBufferRef> buffer_refs,
      MaterializeAndReturn(tensors, MaterializationReason::kExplicitSync));

  // Wait for all of the PjRtBuffers to be ready.
  for (const DeviceBufferRef& buffer_ref : buffer_refs) {
    TT_RETURN_IF_ERROR(buffer_ref.Synchronize());
  }

  return absl::OkStatus();
}

absl::Status SynchronizeAll(const WaitOnExecution wait) {
  const std::vector<SharedDeviceBufferList> needs_sync =
      GetAllLiveUnsyncedDataPtrs();
  if (needs_sync.empty()) {
    return absl::OkStatus();
  }

  TT_RETURN_IF_ERROR(
      Materialize(needs_sync, MaterializationReason::kExplicitSync));

  if (GetFlagOnce<bool,
                  &FLAGS_torch_tpu_internal_enable_new_materialization>()) {
    // Ensure that all dispatched graphs have been materialized.
    TT_RETURN_IF_ERROR(BlockOnPendingMaterializations());
  }

  if (wait == WaitOnExecution::kYes) {
    for (const auto& node : needs_sync) {
      TT_RETURN_IF_ERROR(node->Synchronize());
    }
  }

  return absl::OkStatus();
}

absl::StatusOr<bool> IsMaterializing(const at::Tensor& tensor) {
  // Since view tensors are ephemeral and re-materialized every time, there's
  // no point in checking whether the view is materialized.
  // Instead, we check the base buffer.
  TT_ASSIGN_OR_RETURN(auto base_buffer_ref, GetBaseBuffer(tensor));
  return base_buffer_ref.is_materializing();
}

absl::StatusOr<bool> IsMaterialized(const at::Tensor& tensor) {
  // Since view tensors are ephemeral and re-materialized every time, there's
  // no point in checking whether the view is ready.
  // Instead, we check the base buffer.
  TT_ASSIGN_OR_RETURN(auto buffer_ref, GetBaseBuffer(tensor));
  return buffer_ref.is_materialized();
}

namespace {

absl::StatusOr<absl_nonnull std::unique_ptr<Traversal>> GetTraversal(
    absl::Span<const DeviceBufferRef> buffer_refs) {
  TT_RET_CHECK(!buffer_refs.empty(), error::kInvalidArgument)
      << "tensors must not be empty";
  absl::flat_hash_set<const DeviceBufferList*> nodes_to_traverse_set;
  std::vector<SharedDeviceBufferList> nodes_to_traverse;
  for (const DeviceBufferRef& buffer_ref : buffer_refs) {
    TT_RET_CHECK(!buffer_ref.is_placeholder(), error::kInternal)
        << "[GetTraversal] was called on a placeholder tensor. This should "
           "never happen.";
    if (nodes_to_traverse_set.insert(buffer_ref.device_buffer_list().get())
            .second) {
      nodes_to_traverse.push_back(buffer_ref.device_buffer_list());
    }
  }

  TT_ASSIGN_OR_RETURN(auto traversal, Traversal::Create(nodes_to_traverse));
  ABSL_VLOG(2) << "[GetTraversal] Traversal created";
  return traversal;
}

}  // namespace

absl::StatusOr<std::string> GetComputationGraphviz(
    absl::Span<const DeviceBufferRef> buffer_refs,
    const BufferRefToVarMap& buffer_ref_to_var) {
  TT_ASSIGN_OR_RETURN(auto traversal, GetTraversal(buffer_refs));
  TT_ASSIGN_OR_RETURN(std::string graphviz_string,
                      GetGraphviz(*traversal, buffer_ref_to_var));
  return graphviz_string;
}

absl::StatusOr<std::string> GetComputationMlir(
    absl::Span<const DeviceBufferRef> buffer_refs) {
  TT_ASSIGN_OR_RETURN(auto traversal, GetTraversal(buffer_refs));
  mlir::DialectRegistry registry;
  xla::RegisterMlirToHloDependentDialects(registry);
  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();
  TT_ASSIGN_OR_RETURN(auto module, traversal->BuildMlirModule(context));
  return mlir::debugString(module.get());
}

}  // namespace torch_tpu
