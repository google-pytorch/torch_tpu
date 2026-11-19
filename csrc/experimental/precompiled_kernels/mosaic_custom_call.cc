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
#include <utility>
#include <vector>

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
#include "xla/hlo/ir/hlo_schedule.h"
#include "xla/service/hlo_module_config.h"
#include "xla/shape.h"
#include "xla/shape_util.h"

namespace torch_tpu {
namespace {

// A result that must alias operand `parameter`: the whole result if `output`
// is {}, or element i of a tuple result if it is {i}.
struct ResultAlias {
  xla::ShapeIndex output;
  int64_t parameter;
};

absl::StatusOr<std::unique_ptr<xla::HloModule>> BuildCustomCall(
    std::string_view serialized_kernel, absl::Span<const xla::Shape> parameters,
    const xla::Shape& result, absl::Span<const int64_t> operand_to_metadata,
    absl::Span<const ResultAlias> aliases) {
  xla::HloComputation::Builder builder("main");
  std::vector<xla::HloInstruction*> operands;
  for (const auto& shape : parameters) {
    int p = operands.size();
    operands.push_back(builder.AddInstruction(
        xla::HloInstruction::CreateParameter(p, shape, absl::StrCat("p", p))));
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
  std::vector<
      std::pair<xla::ShapeIndex,  // STD_PAIR_OK=dictated by XLA.
                std::pair<int64_t, xla::ShapeIndex>>>  // STD_PAIR_OK=ditto.
      aliasing;
  aliasing.reserve(aliases.size());
  for (const ResultAlias& alias : aliases) {
    aliasing.push_back({alias.output, {alias.parameter, {}}});
  }
  if (!aliasing.empty()) {
    call->set_output_to_operand_aliasing(std::move(aliasing));
  }
  xla::HloModuleConfig config;
  config.set_use_spmd_partitioning(false);
  config.set_use_shardy_partitioner(false);
  config.set_replica_count(1);
  config.set_num_partitions(1);
  auto hlo = std::make_unique<xla::HloModule>("dynamic_kernel", config);

  auto* entry = hlo->AddEntryComputationWithLayouts(builder.Build());
  // Backend-only compilation requires an explicit schedule.
  xla::HloSchedule schedule(hlo.get());
  schedule.set_sequence(entry, entry->MakeInstructionPostOrder());
  TT_RETURN_IF_ERROR(AdaptXlaError(hlo->set_schedule(schedule),
                                   "failed to schedule dynamic kernel"));
  for (const ResultAlias& alias : aliases) {
    TT_RETURN_IF_ERROR(
        AdaptXlaError(hlo->input_output_alias_config().SetUpAlias(
                          alias.output, alias.parameter, {},
                          xla::HloInputOutputAliasConfig::kMustAlias),
                      "failed to alias private output"));
  }
  return hlo;
}

}  // namespace

absl::StatusOr<std::unique_ptr<xla::HloModule>> BuildMosaicCustomCall(
    std::string_view serialized_kernel, absl::Span<const xla::Shape> parameters,
    const xla::Shape& result, absl::Span<const int64_t> operand_to_metadata,
    int aliased_parameter) {
  std::vector<ResultAlias> aliases;
  if (aliased_parameter >= 0) {
    aliases.push_back({.output = {}, .parameter = aliased_parameter});
  }
  return BuildCustomCall(serialized_kernel, parameters, result,
                         operand_to_metadata, aliases);
}

absl::StatusOr<std::unique_ptr<xla::HloModule>> BuildMosaicTupleCustomCall(
    std::string_view serialized_kernel, absl::Span<const xla::Shape> parameters,
    const xla::Shape& result, absl::Span<const int64_t> operand_to_metadata,
    absl::Span<const int64_t> aliased_parameters) {
  std::vector<ResultAlias> aliases;
  for (int64_t i = 0; i < aliased_parameters.size(); ++i) {
    if (aliased_parameters[i] >= 0) {
      aliases.push_back({.output = {i}, .parameter = aliased_parameters[i]});
    }
  }
  return BuildCustomCall(serialized_kernel, parameters, result,
                         operand_to_metadata, aliases);
}

}  // namespace torch_tpu
