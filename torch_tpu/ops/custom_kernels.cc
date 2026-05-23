// Copyright 2025 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "torch_tpu/ops/custom_kernels.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/base/no_destructor.h"
#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/absl_check.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/WalkResult.h"
#include "mlir/Transforms/Passes.h"
#include "shardy/dialect/sdy/ir/dialect.h"
#include "shardy/dialect/sdy/transforms/common/propagation_options.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/FuncBuilder.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/transforms/Passes.h"
#include "torch_tpu/common/dtype.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/common/fingerprint_utils.h"
#include "torch_tpu/ops/op_builder_utils.h"
#include "torch_tpu/pjrt/pjrt_state.h"
#include "xla/hlo/analysis/alias_info.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/pass/hlo_pass_pipeline.h"
#include "xla/hlo/transforms/simplifiers/flatten_call_graph.h"
#include "xla/hlo/transforms/simplifiers/hlo_dce.h"
#include "xla/hlo/translate/stablehlo.h"
#include "xla/mlir/utils/error_util.h"
#include "xla/mlir_hlo/mhlo/IR/hlo_ops.h"
#include "xla/pjrt/mlir_to_hlo.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_compiler.h"
#include "xla/service/call_inliner.h"
#include "xla/service/spmd/shardy/sdy_round_trip/pipelines.h"
#include "xla/service/spmd/shardy/shardy_xla_pass.h"
#include "xla/service/spmd/spmd_partitioner.h"

namespace torch_tpu {

namespace {

// Returns a unique name for a custom kernel, based on the name, kwargs,
// and input shapes/dtypes.
// This is just the given name, concatenated with a hash of the kwargs and
// input shapes, like "foo_0x1234567890".
std::string GetUniqueCustomKernelName(
    std::string_view name, std::string_view kwargs,
    absl::Span<const absl::Span<const int64_t>> input_dims,
    absl::Span<const mlir::ElementType> input_dtypes) {
  return absl::StrCat(
      name, "_0x", absl::Hex(FingerprintCat(kwargs, input_dtypes, input_dims)));
}

absl::Status ValidateAndEraseShapeAssertions(mlir::ModuleOp module) {
  mlir::BaseScopedDiagnosticHandler diag_handler(module.getContext());
  mlir::PassManager pm(module.getContext());
  pm.addNestedPass<mlir::func::FuncOp>(
      mlir::stablehlo::createStablehloCheckShapeAssertionsPass());
  TT_RET_CHECK(  // ERROR_COV_INFEASIBLE=MLIR pass error not reachable by user
                 // input.
      mlir::succeeded(pm.run(module)), error::kInternal)
      << "failed to validate shape assertions, with MLIR error: "
      << diag_handler.ConsumeStatus().message();

  return absl::OkStatus();
}

// Given a custom MLIR kernel, move it to the destination module and
// return the moved function. Custom kernel must be refined and have a single
// function.
absl::StatusOr<mlir::func::FuncOp> MoveCustomMlirKernel(
    mlir::ModuleOp custom_kernel, mlir::ModuleOp dst_module,
    std::optional<std::string_view> kernel_id_str = std::nullopt) {
  ABSL_CHECK(  // CRASH_OK=Requires custom kernel to share the same context.
      custom_kernel.getContext() == dst_module.getContext())
      << "custom kernel and destination module must have the same context";

  mlir::MLIRContext* context = custom_kernel.getContext();
  mlir::IRRewriter module_rewriter(context);

  mlir::SymbolTableCollection symbol_tables;
  mlir::SymbolTable& dst_st = symbol_tables.getSymbolTable(dst_module);
  mlir::SymbolTable& src_st = symbol_tables.getSymbolTable(custom_kernel);

  // Rename all functions in the module to be unique.
  auto funcs = custom_kernel.getOps<mlir::func::FuncOp>();
  const size_t num_funcs = absl::c_distance(funcs);
  // After running the inliner pass, the kernel must only have 1 function.
  ABSL_CHECK(  // CRASH_OK=Infra only handles importing a single function.
      num_funcs == 1)
      << "custom kernel must have exactly one function after specialization, "
         "got "
      << num_funcs;

  mlir::func::FuncOp src_main = *funcs.begin();
  if (kernel_id_str.has_value()) {
    // Rename the function to the given kernel id string.
    // A kernel id string is expected to be something like
    // "foo_0x1234567890", where the suffix is a hash of the kernel's kwargs and
    // input shapes/dtypes.
    ABSL_CHECK(  // CRASH_OK=Rename should never fail, unrecoverable if it does.
        mlir::succeeded(src_st.rename(src_main, *kernel_id_str)))
        << "failed to rename function " << src_main.getName().str();
  } else {
    // Rename the function to something representative, but also unique.
    mlir::StringRef custom_kernel_name =
        custom_kernel.getName().value_or("kernel");
    ABSL_CHECK(  // CRASH_OK=Rename should never fail, unrecoverable if it does.
        mlir::succeeded(src_st.rename(src_main, custom_kernel_name)))
        << "failed to rename function " << src_main.getName().str();
    ABSL_CHECK(  // CRASH_OK=Rename should never fail, unrecoverable if it does.
        mlir::succeeded(src_st.renameToUnique(src_main, &dst_st)))
        << "failed to rename function " << src_main.getName().str();
  }

  // Now clone the unique functions into the destination module.
  module_rewriter.setInsertionPointToStart(dst_module.getBody());
  mlir::func::FuncOp cloned_entry_fn;
  for (auto func : funcs) {
    auto cloned_func = module_rewriter.clone(*func);
    if (func == src_main) {
      cloned_entry_fn = mlir::cast<mlir::func::FuncOp>(cloned_func);
    }
  }

  // Mark imported function as private, it is no longer the entry function.
  cloned_entry_fn.setVisibility(mlir::SymbolTable::Visibility::Private);

  ABSL_CHECK(  // CRASH_OK=Unreachable given the above logic.
      cloned_entry_fn)
      << "no entry function found in cloned kernel module";

  return cloned_entry_fn;
}

// Given a custom MLIR kernel, possibly with dynamic shapes, specialize it
// into a fully statically shaped kernel using the concrete shapes of the input
// arguments.
absl::Status SpecializeCustomMlirKernel(
    mlir::ModuleOp custom_kernel, mlir::ArrayRef<mlir::Type> input_types) {
  // Capture any MLIR Diagnostics to rethrow as an absl::Status.
  mlir::BaseScopedDiagnosticHandler diag_handler(custom_kernel.getContext());

  // Add a pipeline to specialize the dynamically shaped kernel into a
  // statically shaped kernel for the given argument shapes.
  mlir::PassManager pm(custom_kernel.getContext());
  pm.addPass(mlir::createInlinerPass());
  mlir::stablehlo::createStablehloRemoveDynamismPipeline(pm, input_types);
  TT_RET_CHECK(  // ERROR_COV_INFEASIBLE=MLIR pass error not reachable by user
                 // input.
      mlir::succeeded(pm.run(custom_kernel)), error::kInternal)
      << "failed to specialize custom kernel, with MLIR error: "
      << diag_handler.ConsumeStatus().message();

  return absl::OkStatus();
}

void RemoveCopyOps(mlir::ModuleOp module) {
  module.walk([&](mlir::mhlo::CopyOp copy_op) {
    copy_op.replaceAllUsesWith(copy_op.getOperand());
  });
}

void RemoveSpmdParametersShardings(mlir::ModuleOp module) {
  module->removeAttr("mhlo.spmd_parameters_shardings");
}

absl::Status AddTpuPartitioningPasses(xla::HloModule* module,
                                      const xla::AliasInfo* alias_info,
                                      xla::HloPassPipeline* pipeline) {
  if (!module->config().use_spmd_partitioning()) {
    return absl::OkStatus();
  }
  TT_RET_CHECK(pipeline != nullptr, error::kInternal)
      << "Expected a parent pipeline to be passed as input";

  pipeline->AddPass<xla::HloDCE>();
  // TODO(elliotenglish): Move the internals of this function to the StableHLO
  // prologue passes. This will require determining how to map the HLO parts to
  // StableHLO.
  pipeline->AddPass<xla::sdy::ShardyXLA>(
      /*runSdyShardingPropagation=*/true,
      mlir::sdy::PropagationOptions{
          .enableInsertExplicitCollectives = true
          /*shardy_options.enable_explicit_collectives()*/,
          .removeAllGatherReduceScatterForCMV1 = false,
          .avoidReshardsOnNamedComputations = false
          /*shardy_options.dedup_functions_fully()*/,
      },
      false /*shardy_options.enable_native_non_flat_support()*/);

  xla::spmd::SPMDCollectiveOpsCreator collective_creator =
      xla::spmd::GetDefaultCollectiveOpsCreator(
          module->config().num_partitions(), module->config().replica_count());

  pipeline->AddPass<xla::spmd::SpmdPartitioner>(
      module->config().num_partitions(), module->config().replica_count(),
      xla::spmd::SpmdPartitionerOptions{
          .conv_halo_exchange_always_on_lhs =
              false /*comp_env->xla_jf_spmd_conv_halo_exchange_always_on_lhs()*/
          ,
          .report_instruction_count =
              false /*comp_env->xla_jf_spmd_report_instruction_count()*/,
          .threshold_for_windowed_einsum_mib =
              false /*threshold_for_windowed_einsum_mib*/,
          .unroll_windowed_einsum =
              false /*comp_env->xla_tpu_spmd_unroll_windowed_einsum()*/,
          .bidirectional_windowed_einsum =
              false /*(!comp_env->xla_tpu_spmd_unroll_windowed_einsum() &&
                comp_env->xla_tpu_spmd_bidirectional_windowed_einsum())*/
          ,
          .allow_module_signature_change =
              true /*allow_module_signature_change*/,
          // We use ScheduleAwareAllGatherCSE to remove redundant all-gather.
          .cache_all_gather = false,
          .choose_faster_windowed_einsum_over_mem =
              false /*comp_env->xla_tpu_choose_faster_windowed_einsum_over_mem()*/
          ,
          .skip_checking_windowed_einsum_users =
              false /*comp_env->xla_tpu_skip_checking_windowed_einsum_users()*/,
          .enable_windowed_einsum_for_all_gather =
              false /*enable_cmv1_all_gather*/,
          .enable_windowed_einsum_for_reduce_scatter =
              false /*enable_cmv1_reduce_scatter*/,
          .need_resolve_conflicts = false /*need_resolve_conflicts*/},
      collective_creator);

  pipeline->AddPass<xla::FlattenCallGraph>();
  pipeline->AddPass<xla::CallInliner>();
  pipeline->AddPass<xla::HloDCE>();
  return absl::OkStatus();
}

absl::Status PartitionGlobalToLocal(
    mlir::MLIRContext* context, mlir::ModuleOp module,
    const xla::PjRtTopologyDescription* client_topology) {
  //////////////////////////////////////////////////////////////////////////////
  // Perform Shardy StableHLO/MLIR passes.

  mlir::PassManager shardy_pm(context);
  xla::sdy::addSdyRoundTripExportPipeline(shardy_pm);
  TT_RET_CHECK(mlir::succeeded(shardy_pm.run(module)), error::kInternal)
      << "Running Shardy MLIR passes failed.";

  //////////////////////////////////////////////////////////////////////////////
  // Convert StableHLO to HLO.
  TT_ASSIGN_OR_RETURN(std::unique_ptr<xla::HloModule> hlo_module,
                      xla::ConvertStablehloToHlo(module));

  hlo_module->mutable_config().set_use_spmd_partitioning(true);
  hlo_module->mutable_config().set_use_shardy_partitioner(true);
  TT_ASSIGN_OR_RETURN(int chip_count, client_topology->ChipCount());
  hlo_module->mutable_config().set_num_partitions(chip_count);

  //////////////////////////////////////////////////////////////////////////////
  // Partition HLO module.
  xla::HloPassPipeline pass("tpu-partition-assignment-pipeline");
  xla::AliasInfo alias_info_;
  TT_RETURN_IF_ERROR(
      AddTpuPartitioningPasses(hlo_module.get(), &alias_info_, &pass));

  TT_RETURN_IF_ERROR(pass.Run(hlo_module.get()).status());

  //////////////////////////////////////////////////////////////////////////////
  // Convert HLO back to StableHLO.
  TT_ASSIGN_OR_RETURN(auto new_module_owned,
                      xla::ConvertHloToStablehlo(*context, hlo_module.get()));
  mlir::ModuleOp new_module = new_module_owned.release();

  // TODO(elliotenglish): After partitioning, we may have multiple functions at
  // the top level. We need to inline them back into the main function since
  // this is expected when we move the module into the main TorchTPU graph.
  // HLO's DCE and Inliner passes are unable to do this.
  mlir::PassManager inline_pm(context);
  inline_pm.addPass(mlir::createInlinerPass());
  TT_RET_CHECK(mlir::succeeded(inline_pm.run(new_module)), error::kInternal)
      << "Running inline passes failed.";

  // Replace the contents of the original module with the new module.
  module.getBody()->clear();
  module.getBody()->getOperations().splice(
      module.getBody()->end(), new_module.getBody()->getOperations());
  module->setAttrs(new_module->getAttrDictionary());

  new_module.erase();

  RemoveCopyOps(module);
  RemoveSpmdParametersShardings(module);

  return absl::OkStatus();
}

// Returns true if the module contains any sdy.manual_computation operations.
bool ContainsSdyManualComputation(mlir::ModuleOp module) {
  bool has_manual_computation = false;
  module.walk([&](mlir::Operation* op) {
    if (mlir::isa<mlir::sdy::ManualComputationOp>(op)) {
      has_manual_computation = true;
      return mlir::WalkResult::interrupt();  // Stop walking
    }
    return mlir::WalkResult::advance();
  });
  return has_manual_computation;
}

absl::StatusOr<mlir::func::FuncOp> InjectCustomKernel(
    mlir::MlirBuilder& builder, mlir::ModuleOp custom_kernel,
    mlir::ArrayRef<mlir::Type> input_types,
    std::optional<std::string_view> kernel_id_str = std::nullopt) {
  // TODO(elliotenglish): Currently symbolic shape export is unsupported when we
  // have a distributed kernel. The specialization code also doesn't understand
  // these constructs. Fix this when we implement distributed framework kernels.
  if (!ContainsSdyManualComputation(custom_kernel)) {
    // Specialize the kernel to local shapes.
    TT_RETURN_IF_ERROR(SpecializeCustomMlirKernel(custom_kernel, input_types));
    TT_RETURN_IF_ERROR(ValidateAndEraseShapeAssertions(custom_kernel));
  }

  // Partition global view kernels to local view.
  xla::PjRtClient* client = PjrtBackend::GetInstance().GetClient();
  TT_RET_CHECK(client != nullptr, error::kInternal)
      << "PjRt client is not initialized.";
  TT_ASSIGN_OR_RETURN(const xla::PjRtTopologyDescription* client_topology,
                      client->GetTopologyDescription());
  TT_RETURN_IF_ERROR(PartitionGlobalToLocal(custom_kernel.getContext(),
                                            custom_kernel, client_topology));

  // Move the main function into the builder's current module.
  mlir::ModuleOp module_op = GetModuleOp(builder);
  return MoveCustomMlirKernel(custom_kernel, module_op, kernel_id_str);
}

// An ephemeral, non-owning CustomKernelId.
struct CustomKernelIdRef {
  // The name of the custom kernel (user-provided).
  std::string_view name;
  // A string representation of the keyword arguments to the custom kernel.
  // This can be used to differentiate between calls to the same kernel with the
  // same shapes, but other distinct properties (e.g. tiling settings).
  std::string_view kwargs;
};

// The identifier for a custom kernel, without the specialized shapes.
struct CustomKernelId {
  // A transparent hasher for CustomKernelId and
  // CustomKernelIdRef, so that we can look up kernels without copies.
  struct Hash {
    using is_transparent = void;

    size_t operator()(const CustomKernelIdRef& id) const {
      return FingerprintCat(id.name, id.kwargs);
    }
    size_t operator()(const CustomKernelId& id) const {
      return FingerprintCat(std::string_view(id.name),
                            std::string_view(id.kwargs));
    }
  };

  // A transparent equality functor for CustomKernelId and
  // CustomKernelIdRef.
  struct Eq {
    using is_transparent = void;

    template <typename T1, typename T2>
    bool operator()(const T1& lhs, const T2& rhs) const {
      return lhs.name == rhs.name && lhs.kwargs == rhs.kwargs;
    }
  };

  // The name of the custom kernel (user-provided).
  std::string name;
  // A string representation of the keyword arguments to the custom kernel.
  // This can be used to differentiate between calls to the same kernel with the
  // same shapes, but other distinct properties (e.g. tiling settings).
  std::string kwargs;
};

// The global singleton to hold all custom and specialized kernels.
class CustomKernelRegistry {
 public:
  CustomKernelRegistry() = default;

  // This class is neither copyable nor movable.
  CustomKernelRegistry(const CustomKernelRegistry&) = delete;
  CustomKernelRegistry& operator=(const CustomKernelRegistry&) = delete;
  CustomKernelRegistry(CustomKernelRegistry&&) = delete;
  CustomKernelRegistry& operator=(CustomKernelRegistry&&) = delete;

  static CustomKernelRegistry& GetInstance() {
    static absl::NoDestructor<CustomKernelRegistry> registry;
    return *registry;
  }

  // Registers a custom kernel. This can be any kernel that has been
  // serialized into an MLIR module string. The most common patterns will be
  // either importing from a pre-exported kernel library, or importing from a
  // user-generated kernel authoring framework (e.g. Pallas) in the same
  // process as torch_tpu.
  //
  // Args:
  //  - name: The name of the custom kernel (user-provided).
  //  - kwargs: a string representation of the keyword arguments to the custom
  //    kernel. This is used to differentiate between otherwise-identical
  //    custom kernels; for example, if the same kernel is custom with
  //    different tiling behavior on the same input/output shapes.
  //  - mlir_module_string: The serialized representation of the MLIR
  //    for an mlir::ModuleOp. This must be a single MLIR module, with a @main
  //    function, which may have dynamic shapes (e.g. tensor<?xf32>).
  //
  // Returns true if the kernel was newly inserted, or false if it already
  // existed.
  bool RegisterCustomKernel(std::string_view name, std::string_view kwargs,
                            std::string_view mlir_module_string) {
    absl::MutexLock lock(mutex_);
    if (registry_.contains(CustomKernelId{.name = std::string(name),
                                          .kwargs = std::string(kwargs)})) {
      return false;
    }
    registry_[CustomKernelId{.name = std::string(name),
                             .kwargs = std::string(kwargs)}] =
        std::string(mlir_module_string);
    return true;
  }

  bool LookupCustomKernel(std::string_view name, std::string_view kwargs) {
    absl::MutexLock lock(mutex_);
    return registry_.contains(CustomKernelId{.name = std::string(name),
                                             .kwargs = std::string(kwargs)});
  }

  // Loads a registered custom kernel's MLIR into the given builder, and
  // returns the loaded function to be called (with mlir::func::Call). Returns a
  // NotFound error if the kernel has not been imported with this name and
  // kwargs.
  absl::StatusOr<mlir::func::FuncOp> LoadCustomKernel(
      mlir::MlirBuilder& builder, std::string_view name,
      // TODO: `kwargs` parameter was renamed to `kernel_key` in cl/878060892.
      // This change should be applied to the entire file.
      std::string_view kernel_key,
      absl::Span<const absl::Span<const int64_t>> input_dims,
      absl::Span<const mlir::ElementType> input_dtypes) {
    // Lock the registry while we retrieve the MLIR module string.
    std::string mlir_module_string;
    {
      absl::MutexLock lock(mutex_);
      auto it =
          registry_.find(CustomKernelIdRef{.name = name, .kwargs = kernel_key});
      TT_RET_CHECK(it != registry_.end(), error::kNotFound)
          << "unknown custom kernel; call torch_tpu._internal.pallas."
             "tpu_torch_pallas.register_custom_kernel"
             "(\""
          << name << "\", \"" << kernel_key
          << "\", ...)"
             " to register the kernel before calling it";
      mlir_module_string = it->second;
    }

    // Deserialize the MLIR module from the string.
    TT_ASSIGN_OR_RETURN(
        mlir::OwningOpRef<mlir::ModuleOp> deserialized_module,
        xla::ParseMlirModuleString(mlir_module_string, builder.getContext()));

    // Get the name we want to give the custom kernel.
    std::string kernel_id_str =
        GetUniqueCustomKernelName(name, kernel_key, input_dims, input_dtypes);

    // Build mlir::Types for each input shape.
    mlir::SmallVector<mlir::Type> input_types;
    input_types.reserve(input_dims.size());
    for (int i = 0; i < input_dims.size(); ++i) {
      input_types.push_back(mlir::makeTensorType(
          builder.getContext(), input_dims[i], input_dtypes[i]));
    }

    // Specialize the custom kernel and inject it into the builder.
    return InjectCustomKernel(builder, *deserialized_module, input_types,
                              kernel_id_str);
  }

 private:
  // A map from CustomKernelId to the serialized representation of the
  // MLIR for an mlir::ModuleOp. Can be transparently accessed by
  // CustomKernelIdRef as well.
  absl::flat_hash_map<CustomKernelId, std::string, CustomKernelId::Hash,
                      CustomKernelId::Eq>
      registry_ ABSL_GUARDED_BY(mutex_);
  absl::Mutex mutex_;  // Protects registry_.
};

}  // namespace

bool RegisterCustomKernel(std::string_view name, std::string_view kwargs,
                          std::string_view mlir_module_string) {
  return CustomKernelRegistry::GetInstance().RegisterCustomKernel(
      name, kwargs, mlir_module_string);
}

bool LookupCustomKernel(std::string_view name, std::string_view kwargs) {
  return CustomKernelRegistry::GetInstance().LookupCustomKernel(name, kwargs);
}

absl::StatusOr<DynamicMlirOpResults> CallCustomKernel(
    mlir::MlirBuilder& builder, absl::Span<const mlir::MlirOp> inputs,
    std::string_view name, std::string_view kwargs) {
  // Get the unique name for the custom kernel, based on the name, kwargs,
  // and input shapes.
  std::vector<absl::Span<const int64_t>> input_dims;
  std::vector<mlir::ElementType> input_dtypes;
  input_dims.reserve(inputs.size());
  input_dtypes.reserve(inputs.size());
  for (const auto& input : inputs) {
    mlir::RankedTensorType input_type = GetTensorTypeOrDie(input);
    TT_ASSIGN_OR_RETURN(auto input_dtype, ConvertTo<mlir::ElementType>(
                                              input_type.getElementType()));
    input_dims.push_back(input_type.getShape());
    input_dtypes.push_back(input_dtype);
  }
  std::string unique_name =
      GetUniqueCustomKernelName(name, kwargs, input_dims, input_dtypes);

  // Try to call a previously-loaded kernel with this name.
  mlir::ModuleOp module_op = GetModuleOp(builder);
  if (auto func = module_op.lookupSymbol<mlir::func::FuncOp>(unique_name)) {
    return mlir::func::Call(builder, func, inputs);
  }

  // Could not find a previously-loaded kernel with this name.
  // Load and specialize it here.
  TT_ASSIGN_OR_RETURN(mlir::func::FuncOp func,
                      CustomKernelRegistry::GetInstance().LoadCustomKernel(
                          builder, name, kwargs, input_dims, input_dtypes));
  return mlir::func::Call(builder, func, inputs);
}

absl::StatusOr<DynamicMlirOpResults> BuildSpecializedMlirKernel(
    mlir::MlirBuilder& builder, mlir::ModuleOp custom_kernel,
    absl::Span<mlir::MlirOp> inputs) {
  // Apply specialization and partitioning passes to kernel and then add it to
  // builder.
  // Get the argument types from inputs
  mlir::SmallVector<mlir::Type> input_types = llvm::to_vector(
      llvm::map_range(inputs, [](mlir::MlirOp op) { return op.getType(); }));

  TT_ASSIGN_OR_RETURN(mlir::func::FuncOp main_func,
                      InjectCustomKernel(builder, custom_kernel, input_types));

  return mlir::func::Call(builder, main_func, inputs);
}

}  // namespace torch_tpu
