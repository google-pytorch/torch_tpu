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

// Serialize Mosaic, wrap its dynamic tensor parameters in HLO, and compile.

#include "csrc/experimental/precompiled_kernels/aot/elementwise/compile.h"

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/log/absl_log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/escaping.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "csrc/common/error_utils.h"
#include "csrc/experimental/precompiled_kernels/aot/elementwise/kernels.h"
#include "csrc/experimental/precompiled_kernels/bundle.pb.h"
#include "csrc/ops/op_names.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_os_ostream.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Bytecode/BytecodeWriter.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Pass/PassManager.h"
#include "xla/hlo/builder/xla_computation.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_input_output_alias_config.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_schedule.h"
#include "xla/layout.h"
#include "xla/mosaic/dialect/tpu/tpu_dialect.h"
#include "xla/mosaic/dialect/tpu/transforms/serde.h"
#include "xla/pjrt/pjrt_compiler.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/pjrt/pjrt_layout.h"
#include "xla/service/computation_placer.h"
#include "xla/shape.h"
#include "xla/shape_layout.h"
#include "xla/shape_util.h"
#include "xla/xla_data.pb.h"

namespace torch_tpu {
using precompiled_kernels::proto::Bundle;
using precompiled_kernels::proto::Dimension;
using precompiled_kernels::proto::Kernel;
using precompiled_kernels::proto::Parameter;
using precompiled_kernels::proto::Tensor;
namespace {
namespace tpu = mlir::tpu;

// Verifies the kernel and lowers it to the versioned Mosaic transport form
// the compiler accepts, as MLIR bytecode.
absl::StatusOr<std::string> Serialize(mlir::ModuleOp module) {
  if (mlir::failed(mlir::verify(module))) {
    return TT_ERROR(error::kInternal) << "generated kernel does not verify";
  }
  // Serde emits unregistered stable_mosaic ops; allow them only from here on
  // so that verification above stays strict.
  module.getContext()->allowUnregisteredDialects(true);
  mlir::PassManager passes(module.getContext());
  passes.addPass(
      std::make_unique<tpu::MosaicSerdePass>(tpu::MosaicSerdePassOptions{
          .serialize = true,
          .target_version = tpu::MosaicSerdePass::kVersion}));
  if (mlir::failed(passes.run(module))) {
    return TT_ERROR(error::kInternal) << "Mosaic serialization failed";
  }
  std::string bytes;
  llvm::raw_string_ostream stream(bytes);
  if (mlir::failed(mlir::writeBytecodeToFile(module, stream))) {
    return TT_ERROR(error::kInternal) << "bytecode emission failed";
  }
  return bytes;
}

// The elementwise signature and bindings used by both HLO and the runtime.
Kernel Describe(const ElementwiseOp& op, xla::PrimitiveType dtype, int rank) {
  Tensor tensor;
  tensor.set_dtype(dtype);
  for (int d = 0; d < rank; ++d)
    tensor.add_dims()->set_symbol(absl::StrCat("d", d));
  Kernel entry;
  entry.set_op(std::string(ToString(op.name)));
  for (int i = 0; i < op.arity; ++i) *entry.add_inputs() = tensor;
  *entry.add_outputs() = tensor;
  auto bind = [&](Parameter::Role role, int index) {
    Parameter& parameter = *entry.add_parameters();
    parameter.set_role(role);
    parameter.set_tensor(index);
  };
  for (int i = 0; i < op.arity; ++i) bind(Parameter::INPUT, i);
  bind(Parameter::OUTPUT, op.arity);
  bind(Parameter::RUNTIME_DIMENSIONS, 0);
  return entry;
}

const Tensor& TensorOf(const Kernel& kernel, int index) {
  return index < kernel.inputs_size()
             ? kernel.inputs(index)
             : kernel.outputs(index - kernel.inputs_size());
}

// The RUNTIME_DIMENSIONS parameter a payload parameter reads its extents from:
// the one whose tensor has the same symbolic extents. -1 if none (a static
// operand).
int MetadataFor(const Kernel& kernel, int parameter) {
  const Tensor& tensor =
      TensorOf(kernel, kernel.parameters(parameter).tensor());
  for (int q = 0; q < kernel.parameters_size(); ++q) {
    if (kernel.parameters(q).role() == Parameter::RUNTIME_DIMENSIONS &&
        std::equal(
            tensor.dims().begin(), tensor.dims().end(),
            TensorOf(kernel, kernel.parameters(q).tensor()).dims().begin(),
            TensorOf(kernel, kernel.parameters(q).tensor()).dims().end(),
            [](const Dimension& a, const Dimension& b) {
              return a.extent_case() == b.extent_case() &&
                     a.symbol() == b.symbol() && a.size() == b.size();
            })) {
      return q;
    }
  }
  return -1;
}

// Construct a payload shape and layout for the tensor.
absl::StatusOr<xla::Shape> PayloadShape(
    const Tensor& tensor, const xla::PjRtTopologyDescription& topology) {
  const int rank = tensor.dims_size();
  llvm::SmallVector<int64_t, 4> probe(rank, 8);
  if (rank > 0) probe.back() = 128;
  TT_ASSIGN_OR_RETURN(xla::Layout layout,
                      topology.GetDefaultLayout(tensor.dtype(), probe));
  layout.set_dynamic_shape_metadata_prefix_bytes(0);
  llvm::SmallVector<int64_t, 4> unbounded_shape(rank,
                                                xla::Shape::kUnboundedSize);
  xla::Shape shape = xla::ShapeUtil::MakeShape(tensor.dtype(), unbounded_shape);
  for (int i = 0; i < rank; ++i) {
    if (tensor.dims(i).has_size())
      shape.set_dimensions(i, tensor.dims(i).size());
    else
      shape.set_dynamic_dimension(i, true);
  }
  *shape.mutable_layout() = layout;
  return shape;
}

// Construct an extents shape and layout for dimension metadata.
absl::StatusOr<xla::Shape> ExtentsShape(
    const Tensor& tensor, const xla::PjRtTopologyDescription& topology) {
  const int64_t rank = tensor.dims_size();
  TT_ASSIGN_OR_RETURN(xla::Layout layout,
                      topology.GetDefaultLayout(xla::S32, {rank}));
  xla::Shape shape = xla::ShapeUtil::MakeShape(xla::S32, {rank});
  *shape.mutable_layout() = layout;
  return shape;
}

// Wraps the serialized Mosaic body in an HLO module of one custom call whose
// parameters, result, metadata routing and aliasing follow the kernel's
// bindings. Payloads use the chip's default tiling with no shape prefix;
// Mosaic receives the non-extents parameters in order, then the results.
absl::StatusOr<std::unique_ptr<xla::HloModule>> BuildHlo(
    const Kernel& kernel, const std::string& body,
    const xla::PjRtTopologyDescription& topology) {
  xla::HloComputation::Builder builder("main");
  std::vector<xla::HloInstruction*> operands;
  std::vector<int> metadata;
  for (int p = 0; p < kernel.parameters_size(); ++p) {
    const Parameter& parameter = kernel.parameters(p);
    const Tensor& tensor = TensorOf(kernel, parameter.tensor());
    const bool dims = parameter.role() == Parameter::RUNTIME_DIMENSIONS;
    TT_ASSIGN_OR_RETURN(xla::Shape shape, dims
                                              ? ExtentsShape(tensor, topology)
                                              : PayloadShape(tensor, topology));
    operands.push_back(
        builder.AddInstruction(xla::HloInstruction::CreateParameter(
            p, shape,
            absl::StrCat(Parameter::Role_Name(parameter.role()), p))));
    const int metadata_parameter =
        dims || !shape.is_dynamic() ? -1 : MetadataFor(kernel, p);
    if (shape.is_dynamic() && metadata_parameter < 0) {
      return TT_ERROR(error::kInvalidArgument)
             << "dynamic tensor has no RUNTIME_DIMENSIONS binding";
    }
    metadata.push_back(metadata_parameter);
  }
  std::vector<xla::Shape> results;
  for (const Tensor& output : kernel.outputs()) {
    TT_ASSIGN_OR_RETURN(results.emplace_back(), PayloadShape(output, topology));
  }
  const xla::Shape result = results.size() == 1
                                ? results[0]
                                : xla::ShapeUtil::MakeTupleShape(results);
  auto* call = builder.AddInstruction(xla::HloInstruction::CreateCustomCall(
      result, operands, "tpu_custom_call"));
  call->set_raw_backend_config_string(absl::StrCat(
      R"({"device_type":"DEVICE_TYPE_TENSORCORE","custom_call_config":{"body":")",
      absl::Base64Escape(body),
      R"(","serialization_format":1,"needs_layout_passes":true,)"
      R"("needs_hlo_passes":false,"operand_to_metadata":[)",
      absl::StrJoin(metadata, ","), "]}}"));

  xla::HloModuleConfig config;
  config.set_replica_count(1);
  config.set_num_partitions(1);
  auto module = std::make_unique<xla::HloModule>("precompiled", config);
  using TargetIndex = std::pair<int64_t, xla::ShapeIndex>;  // STD_PAIR_OK=XLA
  using OutputAliasing =
      std::pair<xla::ShapeIndex, TargetIndex>;  // STD_PAIR_OK=XLA
  std::vector<OutputAliasing> aliasing;
  for (int p = 0; p < kernel.parameters_size(); ++p) {
    const Parameter& parameter = kernel.parameters(p);
    if (parameter.role() != Parameter::OUTPUT) continue;
    const xla::ShapeIndex index =
        results.size() == 1
            ? xla::ShapeIndex{}
            : xla::ShapeIndex{parameter.tensor() - kernel.inputs_size()};
    aliasing.push_back({index, {p, {}}});
  }
  call->set_output_to_operand_aliasing(aliasing);
  xla::HloComputation* entry =
      module->AddEntryComputationWithLayouts(builder.Build());
  xla::HloSchedule schedule(module.get());
  schedule.set_sequence(entry, entry->MakeInstructionPostOrder());
  TT_RETURN_IF_ERROR(module->set_schedule(schedule));
  for (const auto& [index, parameter] : aliasing) {
    TT_RETURN_IF_ERROR(module->input_output_alias_config().SetUpAlias(
        index, parameter.first, {},
        xla::HloInputOutputAliasConfig::kMustAlias));
  }
  return module;
}

// Compile options that keep the module's parameter and result layouts.
xla::CompileOptions Options(const xla::HloModule& module) {
  xla::CompileOptions options;
  auto& build = options.executable_build_options;
  build.set_num_replicas(1);
  build.set_num_partitions(1);
  build.set_run_backend_only(true);
  xla::DeviceAssignment assignment(1, 1);
  assignment(0, 0) = 0;
  build.set_device_assignment(assignment);
  options.argument_layouts.emplace();
  for (const xla::ShapeLayout& layout :
       module.entry_computation_layout().parameter_layouts()) {
    options.argument_layouts->push_back(layout.shape());
  }
  build.set_result_layout(module.entry_computation_layout().result_shape());
  return options;
}

// Verify that the compiled executable preserves the signature and output-alias
// contract.
absl::Status VerifyContract(const Kernel& entry,
                            const xla::PjRtExecutable& executable) {
  TT_ASSIGN_OR_RETURN(auto modules, executable.GetHloModules());
  TT_ASSIGN_OR_RETURN(auto parameters, executable.GetParameterLayouts());
  TT_ASSIGN_OR_RETURN(auto results, executable.GetOutputLayouts());
  if (modules.size() != 1 || parameters.size() != entry.parameters_size() ||
      results.size() != entry.outputs_size()) {
    return TT_ERROR(error::kInternal)
           << "compiler changed bundle execution contract signature";
  }
  for (int p = 0; p < entry.parameters_size(); ++p) {
    const auto& parameter = entry.parameters(p);
    if (parameter.role() != Parameter::OUTPUT) continue;
    const int output = parameter.tensor() - entry.inputs_size();
    const xla::ShapeIndex index =
        entry.outputs_size() == 1 ? xla::ShapeIndex{} : xla::ShapeIndex{output};
    auto alias =
        modules[0]->input_output_alias_config().GetAliasedParameter(index);
    if (!alias || !alias->must_alias() || alias->parameter_number != p ||
        !alias->parameter_index.empty() ||
        results[output]->xla_layout() != parameters[p]->xla_layout()) {
      return TT_ERROR(error::kInternal)
             << "compiler changed private-output contract";
    }
  }
  return absl::OkStatus();
}

// Compile one operation/dtype/rank variant.
absl::StatusOr<Kernel> CompileVariant(
    xla::PjRtCompiler& compiler, const xla::PjRtTopologyDescription& topology,
    const ElementwiseOp& op, xla::PrimitiveType dtype, int rank,
    const std::string& dump_dir) {
  Kernel entry = Describe(op, dtype, rank);
  mlir::MLIRContext context;
  auto kernel = BuildKernel(context, op, dtype, rank);
  if (!dump_dir.empty()) {
    std::ofstream file(absl::StrCat(dump_dir, "/", ToString(op.name), ".",
                                    xla::PrimitiveType_Name(dtype), ".r", rank,
                                    ".mlir"));
    llvm::raw_os_ostream stream(file);
    kernel->print(stream);
  }
  TT_ASSIGN_OR_RETURN(std::string body, Serialize(*kernel));
  TT_ASSIGN_OR_RETURN(auto hlo, BuildHlo(entry, body, topology));
  TT_ASSIGN_OR_RETURN(
      auto executable,
      compiler.Compile(Options(*hlo), xla::XlaComputation(hlo->ToProto()),
                       topology, /*client=*/nullptr));
  TT_RETURN_IF_ERROR(VerifyContract(entry, *executable));
  TT_ASSIGN_OR_RETURN(std::string bytes, executable->SerializeExecutable());
  entry.set_executable(std::move(bytes));
  return entry;
}

}  // namespace

absl::Status AddKernels(xla::PjRtCompiler& compiler,
                        const xla::PjRtTopologyDescription& topology,
                        const std::string& dump_dir, Bundle& bundle) {
  for (const ElementwiseOp& op : Operations()) {
    for (const xla::PrimitiveType dtype : {xla::F32, xla::BF16}) {
      for (const int rank : {2, 3, 4}) {
        TT_ASSIGN_OR_RETURN(
            Kernel entry,
            CompileVariant(compiler, topology, op, dtype, rank, dump_dir));
        ABSL_LOG(INFO) << bundle.device_kind() << ": " << entry.op() << " "
                       << xla::PrimitiveType_Name(dtype) << " rank " << rank
                       << ", " << entry.executable().size() << " bytes";
        *bundle.add_kernels() = std::move(entry);
      }
    }
  }
  return absl::OkStatus();
}

}  // namespace torch_tpu
