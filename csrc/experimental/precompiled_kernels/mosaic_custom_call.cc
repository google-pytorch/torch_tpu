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

#include "csrc/experimental/precompiled_kernels/mosaic_custom_call.h"

#include <cstdint>
#include <memory>
#include <string_view>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/log/absl_check.h"
#include "absl/status/statusor.h"
#include "absl/strings/escaping.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/types/span.h"
#include "csrc/common/error_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_input_output_alias_config.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/ir/hlo_schedule.h"
#include "xla/layout.h"
#include "xla/service/hlo_module_config.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/xla_data.pb.h"

namespace torch_tpu {
namespace {

// Replaces every host-memory operand in `operands` with an asynchronous copy
// to device memory (`CopyStart` + `CopyDone`).
void CopyHostMemoryOperandsToDevice(xla::HloComputation::Builder& builder,
                                    absl::Span<xla::HloInstruction*> operands) {
  for (xla::HloInstruction*& operand : operands) {
    const xla::Shape& host_shape = operand->shape();
    if (!host_shape.has_layout() ||
        host_shape.layout().memory_space() != xla::Layout::kHostMemorySpace) {
      continue;
    }
    xla::Shape device_shape = host_shape;
    device_shape.mutable_layout()->set_memory_space(
        xla::Layout::kDefaultMemorySpace);
    // CopyStart returns (destination, source, context).
    xla::HloInstruction* const copy_start =
        builder.AddInstruction(xla::HloInstruction::CreateCopyStart(
            xla::ShapeUtil::MakeTupleShape(
                {device_shape, host_shape,
                 /*context=*/xla::ShapeUtil::MakeShape(xla::U32, {})}),
            operand));
    operand = builder.AddInstruction(xla::HloInstruction::CreateUnary(
        device_shape, xla::HloOpcode::kCopyDone, copy_start));
  }
}

}  // namespace

absl::StatusOr<std::unique_ptr<xla::HloModule>> BuildMosaicCustomCall(
    std::string_view serialized_kernel, absl::Span<const xla::Shape> parameters,
    const xla::Shape& result, absl::Span<const int64_t> operand_to_metadata,
    int aliased_parameter, OperandsTransform operands_transform) {
  TT_RET_CHECK(  // ERROR_COV_INFEASIBLE=Covered by mosaic_custom_call_test.
      aliased_parameter < static_cast<int>(parameters.size()),
      error::kInvalidArgument)
      << "aliased parameter " << aliased_parameter << " is out of range for "
      << parameters.size() << " parameters";
  xla::HloComputation::Builder builder("main");
  std::vector<xla::HloInstruction*> operands;
  operands.reserve(parameters.size());
  for (const xla::Shape& shape : parameters) {
    const int p = operands.size();
    operands.push_back(builder.AddInstruction(
        xla::HloInstruction::CreateParameter(p, shape, absl::StrCat("p", p))));
  }
  const xla::HloInstruction* aliased =
      aliased_parameter >= 0 ? operands[aliased_parameter] : nullptr;
  CopyHostMemoryOperandsToDevice(builder, absl::MakeSpan(operands));
  if (operands_transform != nullptr) {
    operands_transform(builder, absl::MakeSpan(operands));
  }
  if (aliased != nullptr) {
    ABSL_CHECK_EQ(  // CRASH_OK=cannot rewrite aliased parameter
        operands[aliased_parameter], aliased);
  }
  auto* call = builder.AddInstruction(xla::HloInstruction::CreateCustomCall(
      result, operands, "tpu_custom_call"));

  // Backend config as a string, for OSS compatibility.
  call->set_raw_backend_config_string(absl::StrCat(
      R"({"device_type":"DEVICE_TYPE_TENSORCORE","custom_call_config":{"body":")",
      absl::Base64Escape(serialized_kernel),
      R"(","serialization_format":1,"needs_layout_passes":true,)"
      R"("needs_hlo_passes":false,"operand_to_metadata":[)",
      absl::StrJoin(operand_to_metadata, ","), "]}}"));
  if (aliased_parameter >= 0) {
    call->set_output_to_operand_aliasing({{{}, {aliased_parameter, {}}}});
  }
  xla::HloModuleConfig config;
  config.set_use_spmd_partitioning(false);
  config.set_use_shardy_partitioner(false);
  config.set_replica_count(1);
  config.set_num_partitions(1);
  auto hlo = std::make_unique<xla::HloModule>("dynamic_kernel", config);

  auto* entry = hlo->AddEntryComputationWithLayouts(builder.Build());
  // Backend-only compilation requires an explicit schedule. Start every copy to
  // the device before waiting for the first so the copies overlap. Every
  // CopyStart reads only a Parameter, so stable partitioning preserves
  // def-before-use order.
  std::vector<xla::HloInstruction*> sequence =
      entry->MakeInstructionPostOrder();
  absl::c_stable_partition(
      sequence, [](const xla::HloInstruction* const instruction) {
        return instruction->opcode() == xla::HloOpcode::kParameter ||
               instruction->opcode() == xla::HloOpcode::kCopyStart;
      });
  xla::HloSchedule schedule(hlo.get());
  schedule.set_sequence(entry, sequence);
  TT_RETURN_IF_ERROR(WrapXlaError(hlo->set_schedule(schedule),
                                  "failed to schedule dynamic kernel"));
  if (aliased_parameter >= 0) {
    TT_RETURN_IF_ERROR(
        WrapXlaError(hlo->input_output_alias_config().SetUpAlias(
                         {}, aliased_parameter, {},
                         xla::HloInputOutputAliasConfig::kMustAlias),
                     "failed to alias private output"));
  }
  return hlo;
}

}  // namespace torch_tpu
