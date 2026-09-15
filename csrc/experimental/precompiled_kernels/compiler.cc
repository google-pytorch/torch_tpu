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

#include "csrc/experimental/precompiled_kernels/compiler.h"

#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/strings/escaping.h"
#include "absl/strings/str_cat.h"
#include "csrc/common/error_utils.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Bytecode/BytecodeWriter.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LLVM.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_input_output_alias_config.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_schedule.h"
#include "xla/mosaic/dialect/tpu/transforms/serde.h"
#include "xla/service/hlo_module_config.h"
#include "xla/shape.h"

namespace torch_tpu {

absl::StatusOr<std::string> SerializeMosaic(mlir::ModuleOp module) {
  if (mlir::failed(mlir::verify(module))) {
    return TT_ERROR(error::kInternal)
           << "dynamic Mosaic kernel does not verify";
  }
  module.getContext()->allowUnregisteredDialects(true);
  mlir::PassManager passes(module.getContext());
  const mlir::tpu::MosaicSerdePassOptions options = {
      .serialize = true,
      .target_version = mlir::tpu::MosaicSerdePass::kVersion,
  };
  passes.addPass(mlir::tpu::createMosaicSerdePass(options));
  if (mlir::failed(passes.run(module))) {
    return TT_ERROR(error::kInternal) << "failed to run Mosaic serialization";
  }
  std::string bytes;
  llvm::raw_string_ostream stream(bytes);
  if (mlir::failed(mlir::writeBytecodeToFile(module, stream))) {
    return TT_ERROR(error::kInternal) << "failed to emit Mosaic bytecode";
  }
  return bytes;
}

namespace {

// Parameter slots of the generated custom call, in order.
enum Parameter : int {
  kInput = 0,
  kOutput = 1,
  kInputExtents = 2,
  kOutputExtents = 3,
};

// Preserve dtype, rank and tiling, but never compile concrete extents into a
// payload shape. Metadata travels separately, with no payload prefix.
void MakeExtentsUnbounded(xla::Shape& shape) {
  shape.mutable_layout()->set_dynamic_shape_metadata_prefix_bytes(0);
  for (int64_t d = 0; d < static_cast<int64_t>(shape.dimensions().size());
       ++d) {
    shape.set_dynamic_dimension(d, true);
    shape.set_dimensions(d, xla::Shape::kUnboundedSize);
  }
}

// Payloads read their extents from the two metadata operands. The -1 entries
// mark those operands as static S32[rank] vectors needing none of their own.
std::string MosaicBackendConfig(std::string_view body) {
  return absl::StrCat(
      R"({"device_type":"DEVICE_TYPE_TENSORCORE","custom_call_config":{"body":")",
      absl::Base64Escape(body),
      R"(","serialization_format":1,"needs_layout_passes":true,)"
      R"("needs_hlo_passes":false,"operand_to_metadata":[2,3,-1,-1]}})");
}

}  // namespace

absl::StatusOr<std::unique_ptr<xla::HloModule>> BuildDynamicKernelHlo(
    std::string_view body, std::array<xla::Shape, 4> parameters) {
  // One extent per runtime dimension. A mismatch otherwise surfaces much later
  // as an unexplained custom call failure on device.
  TT_RET_CHECK(parameters[kInputExtents].dimensions(0) ==
                   static_cast<int64_t>(parameters[kInput].dimensions().size()),
               error::kInvalidArgument)
      << "input extents must have one element per input dimension";
  TT_RET_CHECK(
      parameters[kOutputExtents].dimensions(0) ==
          static_cast<int64_t>(parameters[kOutput].dimensions().size()),
      error::kInvalidArgument)
      << "output extents must have one element per output dimension";
  MakeExtentsUnbounded(parameters[kInput]);
  MakeExtentsUnbounded(parameters[kOutput]);

  xla::HloComputation::Builder builder("main");
  std::vector<xla::HloInstruction*> operands;
  for (const xla::Shape& shape : parameters) {
    const int p = operands.size();
    operands.push_back(builder.AddInstruction(
        xla::HloInstruction::CreateParameter(p, shape, absl::StrCat("p", p))));
  }
  xla::HloInstruction* call =
      builder.AddInstruction(xla::HloInstruction::CreateCustomCall(
          parameters[kOutput], operands, "tpu_custom_call"));
  call->set_raw_backend_config_string(MosaicBackendConfig(body));
  // Both the custom call and module promise to write the donated destination,
  // rather than allocate a second result buffer. The result, at index {}, is
  // all of operand kOutput, also at index {}.
  call->set_output_to_operand_aliasing(
      {{/*output_index=*/{}, {kOutput, /*operand_index=*/{}}}});
  xla::HloModuleConfig config;
  config.set_replica_count(1);
  config.set_num_partitions(1);
  auto hlo = std::make_unique<xla::HloModule>("dynamic_kernel", config);
  xla::HloComputation* entry =
      hlo->AddEntryComputationWithLayouts(builder.Build());
  xla::HloSchedule schedule(hlo.get());
  schedule.set_sequence(entry, entry->MakeInstructionPostOrder());
  TT_RETURN_IF_ERROR(AdaptXlaError(hlo->set_schedule(schedule),
                                   "failed to schedule dynamic kernel"));
  TT_RETURN_IF_ERROR(AdaptXlaError(
      hlo->input_output_alias_config().SetUpAlias(
          {}, kOutput, {}, xla::HloInputOutputAliasConfig::kMustAlias),
      "failed to alias private output"));
  return hlo;
}

}  // namespace torch_tpu
