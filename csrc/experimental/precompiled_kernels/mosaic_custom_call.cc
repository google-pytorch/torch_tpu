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
#include <string>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/strings/escaping.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/types/span.h"
#include "csrc/common/error_utils.h"
#include "csrc/internal/mosaic/op_builders.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Bytecode/BytecodeWriter.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_input_output_alias_config.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_schedule.h"
#include "xla/service/hlo_module_config.h"
#include "xla/shape.h"

namespace torch_tpu {
namespace {

absl::StatusOr<std::string> SerializeMosaic(mlir::ModuleOp module) {
  // Use TorchTPU's pinned Mosaic transport version, not upstream's latest.
  if (mlir::failed(SerializeMosaicKernel(module))) {
    return TT_ERROR(error::kInternal) << "Mosaic serialization failed";
  }
  std::string bytes;
  llvm::raw_string_ostream stream(bytes);
  mlir::BytecodeWriterConfig config;
  // Match JAX's Mosaic bytecode encoding; independent of the serde version.
  config.setDesiredBytecodeVersion(0);
  if (mlir::failed(mlir::writeBytecodeToFile(module, stream, config))) {
    return TT_ERROR(error::kInternal) << "Mosaic bytecode emission failed";
  }
  return bytes;
}

}  // namespace

absl::StatusOr<std::unique_ptr<xla::HloModule>> BuildMosaicCustomCall(
    mlir::ModuleOp kernel, absl::Span<const xla::Shape> parameters,
    const xla::Shape& result, absl::Span<const int64_t> operand_to_metadata,
    int aliased_parameter) {
  TT_ASSIGN_OR_RETURN(auto body, SerializeMosaic(kernel));
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
      absl::Base64Escape(body),
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
  // Backend-only compilation requires an explicit schedule.
  xla::HloSchedule schedule(hlo.get());
  schedule.set_sequence(entry, entry->MakeInstructionPostOrder());
  TT_RETURN_IF_ERROR(AdaptXlaError(hlo->set_schedule(schedule),
                                   "failed to schedule dynamic kernel"));
  if (aliased_parameter >= 0) {
    TT_RETURN_IF_ERROR(
        AdaptXlaError(hlo->input_output_alias_config().SetUpAlias(
                          {}, aliased_parameter, {},
                          xla::HloInputOutputAliasConfig::kMustAlias),
                      "failed to alias private output"));
  }
  return hlo;
}

}  // namespace torch_tpu
