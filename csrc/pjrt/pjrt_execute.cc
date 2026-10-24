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

#include "csrc/pjrt/pjrt_execute.h"

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/container/inlined_vector.h"
#include "absl/log/absl_check.h"
#include "absl/log/absl_log.h"
#include "absl/log/absl_vlog_is_on.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/escaping.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "csrc/common/compilation.h"
#include "csrc/common/dimension_types.h"
#include "csrc/common/dtype.h"
#include "csrc/common/env_vars.h"
#include "csrc/common/error_utils.h"
#include "csrc/common/shape.h"
#include "csrc/pjrt/pjrt_utils.h"
#include "tsl/profiler/lib/traceme.h"
#include "xla/future.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/layout.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/service/collective_ops_utils.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/xla_data.pb.h"

namespace torch_tpu {
namespace {

xla::Shape MakeXlaShape(const Shape& shape, const xla::Layout& layout) {
  const xla::PrimitiveType xla_type =
      ConvertTo<xla::PrimitiveType>(shape.dtype());
  xla::Shape xla_shape =
      xla::ShapeUtil::MakeShape(xla_type, shape.dimensions());
  *xla_shape.mutable_layout() = layout;
  return xla_shape;
}

// Creates a 1D S32[rank] PjRtBuffer containing the concrete runtime dimension
// sizes of `dimensions` as int32_t values, suitable for passing as a disjoint
// runtime shape metadata parameter (operand_to_metadata) to unbounded-dynamic
// executables.
absl::StatusOr<std::unique_ptr<xla::PjRtBuffer>>
CreateRuntimeShapeMetadataBuffer(
    absl::Span<const int64_t> dimensions, const xla::Layout& layout,
    xla::PjRtClient* absl_nonnull client,
    xla::PjRtMemorySpace* absl_nonnull memory_space) {
  absl::InlinedVector<int32_t, 8> s32_dims;
  s32_dims.reserve(dimensions.size());
  for (const int64_t d : dimensions) {
    if (static_cast<int32_t>(d) != d) {
      return TT_ERROR(error::kInvalidArgument)
             << "Dimension " << d << " exceeds the maximum value for int32_t";
    }
    s32_dims.push_back(d);
  }

  const int64_t rank = s32_dims.size();
  return AdaptXlaError(
      client->BufferFromHostBuffer(
          s32_dims.data(), xla::S32, /*dims=*/{rank},
          /*byte_strides=*/std::nullopt,
          xla::PjRtClient::HostBufferSemantics::kImmutableOnlyDuringCall,
          /*on_done_with_host_buffer=*/nullptr, memory_space, &layout),
      /* context= */ "failed to create runtime shape metadata S32 buffer");
}

// Holds the full physical argument pointer list and any newly allocated
// ephemeral device buffers (pre-allocated donated dynamic output buffers and
// disjoint S32[rank] runtime shape metadata buffers) whose lifetimes must span
// device execution.
struct DynamicExecutionArguments {
  std::vector<xla::PjRtBuffer* absl_nullable> full_args;
  std::vector<std::unique_ptr<xla::PjRtBuffer>> ephemeral_buffers;
};

// Routes logical argument buffers, pre-allocated donated dynamic output
// buffers, and disjoint S32[rank] runtime shape metadata buffers into their
// physical PjRt parameter positions according to `meta`. A donated output
// buffer is allocated in the shape its `physical_shape_fn` hook returns, if
// set, and each metadata buffer reports the on-device shape of the payload
// routed to the parameter it describes.
absl::StatusOr<DynamicExecutionArguments> PrepareDynamicExecutionArguments(
    const UnboundedDynamicExecutionMetadata& meta,
    absl::Span<xla::PjRtBuffer* const> argument_buffers,
    absl::Span<const Shape> logical_output_shapes) {
  TT_ASSIGN_OR_RETURN(xla::PjRtClient* const client, GetPjRtClient());
  TT_ASSIGN_OR_RETURN(xla::PjRtDevice* const device, GetPjRtDevice());
  TT_ASSIGN_OR_RETURN(xla::PjRtMemorySpace * memory_space,
                      GetDefaultMemorySpace(device));

  DynamicExecutionArguments result;
  result.full_args.assign(meta.pjrt_param_shapes.size(), nullptr);

  // Meta will be checked that it is self-consistent on kernel load. We
  // therefore do not need to check it here, but we should add a DCHECK when the
  // check is implemented.
  TT_RET_CHECK(  // ERROR_COV_INFEASIBLE=enforced by high level contract.
      meta.logical_input_to_pjrt_param.size() == argument_buffers.size(),
      error::kInternal)
      << "Mismatch between number of logical inputs in "
         "UnboundedDynamicExecutionMetadata and number of argument "
         "buffers provided.";

  // 1. Route logical input payload buffers.
  for (size_t i = 0; i < argument_buffers.size(); ++i) {
    const int64_t param_idx = meta.logical_input_to_pjrt_param[i];
    result.full_args[param_idx] = argument_buffers[i];
  }

  // 2. Pre-allocate and route donated dynamic output buffers.
  for (const auto& donated_spec : meta.donated_output_params) {
    const Shape& out_shape =
        logical_output_shapes[donated_spec.logical_output_idx];
    const xla::Layout& layout =
        meta.pjrt_param_shapes[donated_spec.pjrt_param_idx].layout();

    xla::Shape xla_out_shape;
    if (donated_spec.physical_shape_fn) {
      TT_ASSIGN_OR_RETURN(xla_out_shape,
                          (*donated_spec.physical_shape_fn)(out_shape, layout));
    } else {
      xla_out_shape = MakeXlaShape(out_shape, layout);
    }
    TT_ASSIGN_OR_RETURN(
        std::unique_ptr<xla::PjRtBuffer> donated_buf,
        AdaptXlaError(
            client->CreateUninitializedBuffer(xla_out_shape, memory_space),
            /* context= */ "failed to allocate donated dynamic output buffer"));
    result.full_args[donated_spec.pjrt_param_idx] = donated_buf.get();
    result.ephemeral_buffers.push_back(std::move(donated_buf));
  }

  // 3. Create and route disjoint S32[rank] runtime shape metadata buffers.
  // Steps 1 and 2 have routed every payload, so each buffer already carries the
  // physical shape the executable will see.
  for (int i = 0; i < meta.metadata_params.size(); ++i) {
    const auto& binding = meta.metadata_params[i];
    xla::PjRtBuffer* const payload = result.full_args[binding.buffer_pjrt_idx];
    TT_RET_CHECK(  // ERROR_COV_INFEASIBLE=enforced by high level contract.
        payload != nullptr, error::kInternal)
        << "Metadata parameter at idx " << i << " describes PjRt parameter "
        << binding.buffer_pjrt_idx << ", which no payload was routed to.";
    const absl::Span<const int64_t> dims =
        payload->on_device_shape().dimensions();

    const xla::Layout& layout =
        meta.pjrt_param_shapes[binding.shape_pjrt_idx].layout();
    TT_ASSIGN_OR_RETURN(
        std::unique_ptr<xla::PjRtBuffer> meta_buf,
        CreateRuntimeShapeMetadataBuffer(dims, layout, client, memory_space));
    result.full_args[binding.shape_pjrt_idx] = meta_buf.get();
    result.ephemeral_buffers.push_back(std::move(meta_buf));
  }

  return result;
}

bool HloModuleContainsCollective(const xla::HloModule* hlo_module) {
  for (const xla::HloComputation* computation : hlo_module->computations()) {
    for (const xla::HloInstruction* instruction : computation->instructions()) {
      if (xla::IsCollective(instruction)) {
        return true;
      }
    }
  }
  return false;
}

// Returns an error if an argument's on-device layout isn't the one the
// executable was compiled for. For static parameters and arguments, PJRT only
// checks an argument's size, so it would silently misread a same-size argument
// with another layout.
absl::Status ValidateArgumentLayouts(
    const LoadedExecutableWithMetadata& executable,
    absl::Span<xla::PjRtBuffer* const> argument_buffers) {
  const std::vector<std::optional<xla::Shape>>& parameter_shapes =
      executable.parameter_device_shapes();
  for (size_t i = 0; i < argument_buffers.size() && i < parameter_shapes.size();
       ++i) {
    // PJRT handles dynamic parameters and arguments itself.
    if (argument_buffers[i] == nullptr || !parameter_shapes[i].has_value() ||
        !argument_buffers[i]->on_device_shape().is_static()) {
      continue;
    }
    const xla::Shape& expected = *parameter_shapes[i];
    const xla::Shape& actual = argument_buffers[i]->on_device_shape();
    // An argument with other dimensions, e.g. a one-element array passed to a
    // scalar parameter, is reinterpreted on purpose. PJRT checks its size.
    if (actual.dimensions() != expected.dimensions()) {
      continue;
    }
    // The dimension order and tiles determine where each element is stored.
    if (actual.layout().minor_to_major() !=
            expected.layout().minor_to_major() ||
        actual.layout().tiles() != expected.layout().tiles()) {
      return TT_ERROR(error::kInvalidArgument)
             << "expected argument " << i << " to have on-device layout "
             << expected.layout().ToString()
             << ", which the program was compiled for, got "
             << actual.layout().ToString()
             << " (a tensor's layout can change after compilation, e.g. "
                "through copy_())";
    }
  }
  return absl::OkStatus();
}

}  // namespace

absl::StatusOr<PjRtBufferPointers> Execute(
    const SharedLoadedExecutableWithMetadata& executable,
    std::vector<xla::PjRtBuffer* absl_nullable> argument_buffers,
    absl::Span<const Shape> logical_output_shapes,
    ArgumentLayoutValidation layout_validation) {
  tsl::profiler::TraceMe trace("Execute");
  xla::ExecuteOptions execute_options{.strict_shape_checking = true};

  // Ephemeral buffers created for dynamic execution (pre-allocated donated
  // dynamic output buffers and disjoint S32[rank] runtime shape metadata
  // buffers) must outlive device execution.
  std::vector<std::unique_ptr<xla::PjRtBuffer>> ephemeral_buffers;

  const std::optional<UnboundedDynamicExecutionMetadata>&
      unbounded_dynamic_meta =
          executable->unbounded_dynamic_execution_metadata();
  if (unbounded_dynamic_meta) {
    TT_ASSIGN_OR_RETURN(
        DynamicExecutionArguments dynamic_args,
        PrepareDynamicExecutionArguments(
            *unbounded_dynamic_meta, argument_buffers, logical_output_shapes));
    argument_buffers = std::move(dynamic_args.full_args);
    ephemeral_buffers = std::move(dynamic_args.ephemeral_buffers);
  }
  if (layout_validation == ArgumentLayoutValidation::kValidate) {
    TT_RETURN_IF_ERROR(ValidateArgumentLayouts(*executable, argument_buffers));
  }

  if (ABSL_VLOG_IS_ON(8)) {
    TT_ASSIGN_OR_RETURN(const auto hlo_modules,
                        executable->GetLoadedExecutable()->GetHloModules());
    ABSL_CHECK_EQ(hlo_modules.size(), 1);  // CRASH_OK
    const auto& hlo_module = hlo_modules[0];

    if (HloModuleContainsCollective(hlo_module.get())) {
      TT_ASSIGN_OR_RETURN(
          const std::string fingerprint,
          executable->GetLoadedExecutable()->FingerprintExecutable());
      const std::optional<int> rank = GetIntegerEnvOnce<int, kRankEnvVar>();
      ABSL_VLOG(8) << "Executable with collectives on rank "
                   << (rank.has_value() ? absl::StrCat(*rank)
                                        : "<unknown_rank>")
                   << " has PjRT executable fingerprint: "
                   << absl::BytesToHexString(fingerprint)
                   << " and HLO module fingerprint: "
                   << hlo_module->GetFingerprint128();
    }
  }

  std::optional<std::vector<xla::Future<>>> returned_futures;
  if (!ephemeral_buffers.empty()) {
    returned_futures.emplace();
  }

  TT_ASSIGN_OR_RETURN(
      std::vector<std::vector<std::unique_ptr<xla::PjRtBuffer>>>
          results_per_device,
      AdaptXlaError(executable->GetLoadedExecutable()->Execute(
                        absl::MakeConstSpan(&argument_buffers, 1),
                        execute_options, returned_futures),
                    /* context= */ "failed to run executable"));

  if (!ephemeral_buffers.empty()) {
    TT_RET_CHECK(  // ERROR_COV_INFEASIBLE=XLA execution contracts guarantee
                   // non-empty results on successful execution
        returned_futures.has_value() && !returned_futures->empty(),
        error::kInternal)
        << "returned_futures is empty";
    (*returned_futures)[0].OnReady(
        [ephemeral_buffers = std::move(ephemeral_buffers)](absl::Status) {});
  }

  TT_RET_CHECK(  // ERROR_COV_INFEASIBLE=XLA execution contracts guarantee
                 // non-empty results on successful execution.
      !results_per_device.empty(), error::kInternal)
      << "xla execution did not return any results";

  PjRtBufferPointers result_pointers = std::move(results_per_device[0]);

  if (unbounded_dynamic_meta && unbounded_dynamic_meta->output_transform) {
    TT_ASSIGN_OR_RETURN(result_pointers,
                        (*unbounded_dynamic_meta->output_transform)(
                            std::move(result_pointers), logical_output_shapes));
  }

  return result_pointers;
}

}  // namespace torch_tpu
