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

#include "csrc/internal/mosaic/kiln/core/kernel.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>

#include "absl/functional/function_ref.h"
#include "absl/log/absl_check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "csrc/common/error_utils.h"
#include "csrc/internal/mosaic/kiln/core/context.h"
#include "csrc/internal/mosaic/kiln/core/dtype.h"
#include "csrc/internal/mosaic/kiln/core/index_arg.h"
#include "csrc/internal/mosaic/kiln/core/index_map.h"
#include "csrc/internal/mosaic/kiln/core/ref.h"
#include "csrc/internal/mosaic/kiln/core/scalar.h"
#include "csrc/internal/mosaic/kiln/core/specs.h"
#include "csrc/internal/mosaic/op_builders.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/LogicalResult.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/Verifier.h"

namespace torch_tpu::kiln {

using torch_tpu::is_status_or_ref;

namespace {

struct KernelScope {
  KernelScope(mlir::func::FuncOp fn, const KernelSpec& s)
      : builder(mlir::ImplicitLocOpBuilder::atBlockBegin(fn.getLoc(),
                                                         fn.addEntryBlock())),
        spec(s),
        builder_scope(&builder),
        prev(tls_scope) {
    tls_scope = this;
    int arg = 0;
    mlir::Block* entry = builder.getBlock();
    const int num_args = entry->getNumArguments();
    auto bind = [&]<typename T>(llvm::SmallVectorImpl<T>& out, size_t count) {
      for (size_t i = 0; i < count && arg < num_args; ++i) {
        out.push_back(internal::wrap<T>(entry->getArgument(arg++)));
      }
    };
    bind(program_ids, spec.grid.grid.size());
    bind(scalar_prefetch, spec.scalar_prefetch.size());
    bind(inputs, spec.in_specs.size());
    bind(outputs, spec.out_specs.size());
    bind(scratch, spec.scratch.size());
  }
  ~KernelScope() { tls_scope = prev; }

  mlir::ImplicitLocOpBuilder builder;
  const KernelSpec& spec;
  internal::BuilderScope builder_scope;
  llvm::SmallVector<Scalar, 4> program_ids;
  llvm::SmallVector<Ref, 2> scalar_prefetch;
  llvm::SmallVector<Ref, 8> inputs;
  llvm::SmallVector<Ref, 4> outputs;
  llvm::SmallVector<Ref, 4> scratch;
  KernelScope* prev;

  static thread_local KernelScope* tls_scope;  // CPP_THREAD_LOCAL_OK=C++ scope
};

thread_local KernelScope*  // CPP_THREAD_LOCAL_OK=C++ scope
    KernelScope::tls_scope = nullptr;

KernelScope& CurrentScope() {
  ABSL_CHECK(KernelScope::tls_scope != nullptr)  // CRASH_OK
      << "KILN operations may only be called inside BuildKernel()";
  return *KernelScope::tls_scope;
}

mlir::MemRefType BlockType(mlir::MLIRContext* ctx,
                           llvm::ArrayRef<int64_t> shape, DType dtype,
                           MemorySpace space) {
  mlir::Type elem_type = internal::ToMlirType(dtype, ctx);
  return space == MemorySpace::kVmem
             ? mlir::torch_tpu::GetVmemMemRefType(ctx, shape, elem_type)
             : mlir::torch_tpu::GetSmemMemRefType(ctx, shape, elem_type);
}

mlir::torch_tpu::DimensionSemantics ToMlirDimensionSemantics(
    DimensionSemantics s) {
  switch (s) {
    case DimensionSemantics::kParallel:
      return mlir::torch_tpu::DimensionSemantics::kParallel;
    case DimensionSemantics::kArbitrary:
      return mlir::torch_tpu::DimensionSemantics::kArbitrary;
  }
}

std::string PrintModule(mlir::ModuleOp module) {
  std::string out;
  llvm::raw_string_ostream os(out);
  module->print(os, mlir::OpPrintingFlags().useLocalScope());
  return out;
}

// Emits the private function backing an `IndexMap::IndexFn`.
absl::StatusOr<mlir::FlatSymbolRefAttr> EmitIndexMapFunction(
    mlir::ImplicitLocOpBuilder& b, mlir::func::FuncOp kernel_fn,
    const KernelSpec& spec, llvm::ArrayRef<mlir::Type> prefetch_types,
    const IndexMap& map, int64_t block_rank, int64_t index) {
  const std::string name = absl::StrCat(spec.grid.name, "_index_map_", index);
  mlir::Type i32 = b.getI32Type();
  llvm::SmallVector<mlir::Type> arg_types(spec.grid.grid.size(), i32);
  arg_types.append(prefetch_types.begin(), prefetch_types.end());
  llvm::SmallVector<mlir::Type> result_types(block_rank, i32);

  mlir::OpBuilder::InsertionGuard guard(b);
  b.setInsertionPoint(kernel_fn);
  auto fn = mlir::func::FuncOp::create(
      b, name, b.getFunctionType(arg_types, result_types));
  fn.setPrivate();
  KernelScope scope(fn, spec);
  IndexIndices indices = map.Invoke();
  TT_RET_CHECK(indices.size() == block_rank, torch_tpu::error::kInvalidArgument)
      << "index map must return one index per block dimension";
  llvm::SmallVector<mlir::Value> results;
  for (const IndexArg& s : indices) {
    results.push_back(internal::unwrap(s.ToScalar().ToI32()));
  }
  mlir::func::ReturnOp::create(scope.builder, results);

  return mlir::FlatSymbolRefAttr::get(b.getContext(), name);
}

absl::Status SetGridAttributes(mlir::ImplicitLocOpBuilder& b,
                               mlir::func::FuncOp fn, const KernelSpec& spec,
                               llvm::ArrayRef<mlir::Type> prefetch_types) {
  const int64_t grid_rank = spec.grid.grid.size();

  mlir::MLIRContext* context = b.getContext();
  fn->setAttr("iteration_bounds", b.getDenseI64ArrayAttr(spec.grid.grid));
  llvm::SmallVector<mlir::Attribute> semantics;
  for (int i = 0; i < grid_rank; ++i) {
    DimensionSemantics sem = spec.grid.dimension_semantics.empty()
                                 ? DimensionSemantics::kParallel
                                 : spec.grid.dimension_semantics[i];
    semantics.push_back(mlir::torch_tpu::GetDimensionSemanticsAttr(
        context, ToMlirDimensionSemantics(sem)));
  }
  fn->setAttr("dimension_semantics", b.getArrayAttr(semantics));

  int64_t index_map_idx = 0;
  llvm::SmallVector<mlir::Attribute> window_params;
  for (const BlockSpec& s :
       llvm::concat<const BlockSpec>(spec.in_specs, spec.out_specs)) {
    const int64_t block_rank = s.block_shape.size();
    if (s.index_map.has_mapping()) {
      TT_ASSIGN_OR_RETURN(
          mlir::FlatSymbolRefAttr index_map_ref,
          EmitIndexMapFunction(b, fn, spec, prefetch_types, s.index_map,
                               block_rank, index_map_idx++));
      window_params.push_back(b.getDictionaryAttr({
          b.getNamedAttr("transform_indices", index_map_ref),
          b.getNamedAttr("window_bounds",
                         b.getDenseI64ArrayAttr(s.block_shape)),
      }));
    } else {
      llvm::SmallVector<mlir::AffineExpr> results(block_rank,
                                                  b.getAffineConstantExpr(0));
      window_params.push_back(b.getDictionaryAttr({b.getNamedAttr(
          "transform_indices", mlir::AffineMapAttr::get(mlir::AffineMap::get(
                                   grid_rank, 0, results, context)))}));
    }
  }
  fn->setAttr("window_params", b.getArrayAttr(window_params));

  return absl::OkStatus();
}

template <typename RangeT>
const auto& GetAt(const RangeT& range, int64_t idx, std::string_view name) {
  ABSL_CHECK(idx >= 0 && idx < range.size())  // CRASH_OK
      << name << " index " << idx << " is out of bounds (size " << range.size()
      << ")";
  return range[idx];
}

}  // namespace

struct Kernel::Impl {
  std::unique_ptr<mlir::MLIRContext> context;
  mlir::OwningOpRef<mlir::ModuleOp> module;
};

Kernel::Kernel(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
Kernel::Kernel(Kernel&&) noexcept = default;
Kernel& Kernel::operator=(Kernel&&) noexcept = default;
Kernel::~Kernel() = default;

std::string Kernel::ToString() const { return PrintModule(*impl_->module); }

absl::StatusOr<std::string> Kernel::Serialize() && {
  if (llvm::failed(
          mlir::torch_tpu::SerializeMosaicKernel(impl_->module.get()))) {
    return TT_ERROR(::torch_tpu::error::kInternal)
           << "failed to serialize mosaic kernel";
  }
  return ToString();
}

absl::Status Kernel::Verify() const {
  if (llvm::failed(mlir::verify(impl_->module.get()))) {
    return TT_ERROR(::torch_tpu::error::kInternal)
           << "KILN kernel failed verification:\n"
           << ToString();
  }
  return absl::OkStatus();
}

Scalar ProgramId(int axis) {
  return GetAt(CurrentScope().program_ids, axis, "ProgramId");
}
int64_t NumPrograms(int axis) {
  return GetAt(CurrentScope().spec.grid.grid, axis, "NumPrograms");
}
int GridRank() { return CurrentScope().spec.grid.grid.size(); }

Ref ScalarPrefetch(int i) {
  return GetAt(CurrentScope().scalar_prefetch, i, "ScalarPrefetch");
}
Ref In(int i) { return GetAt(CurrentScope().inputs, i, "In"); }
Ref Out(int i) { return GetAt(CurrentScope().outputs, i, "Out"); }
Ref Scratch(int i) { return GetAt(CurrentScope().scratch, i, "Scratch"); }

absl::StatusOr<Kernel> BuildKernel(const KernelSpec& spec,
                                   absl::FunctionRef<void()> body) {
  const int64_t grid_rank = spec.grid.grid.size();
  if (!spec.grid.dimension_semantics.empty() &&
      spec.grid.dimension_semantics.size() != grid_rank) {
    return TT_ERROR(::torch_tpu::error::kInvalidArgument)
           << "dimension_semantics must have one entry per grid axis";
  }

  auto impl = std::make_unique<Kernel::Impl>();
  impl->context = mlir::torch_tpu::CreateMlirContextWithDialects();
  mlir::MLIRContext* context = impl->context.get();

  mlir::Location loc = mlir::UnknownLoc::get(context);
  impl->module = mlir::ModuleOp::create(loc);
  mlir::ModuleOp module = *impl->module;
  mlir::ImplicitLocOpBuilder b(loc, context);
  b.setInsertionPointToEnd(module.getBody());

  // Argument list follows the Mosaic calling convention.
  mlir::Type i32 = b.getI32Type();
  llvm::SmallVector<mlir::Type> prefetch_types;
  for (const ScalarPrefetchSpec& p : spec.scalar_prefetch) {
    prefetch_types.push_back(mlir::torch_tpu::GetSmemMemRefType(
        context, p.shape, internal::ToMlirType(p.dtype, context)));
  }
  llvm::SmallVector<mlir::Type> arg_types(grid_rank, i32);
  arg_types.append(prefetch_types);
  for (const BlockSpec& s :
       llvm::concat<const BlockSpec>(spec.in_specs, spec.out_specs)) {
    arg_types.push_back(
        BlockType(context, s.block_shape, s.dtype, s.memory_space));
  }
  for (const ScratchSpec& s : spec.scratch) {
    arg_types.push_back(BlockType(context, s.shape, s.dtype, s.memory_space));
  }

  auto fn = mlir::func::FuncOp::create(b, spec.grid.name,
                                       b.getFunctionType(arg_types, {}));
  mlir::torch_tpu::SetTcCoreTypeAttr(fn);
  TT_RETURN_IF_ERROR(SetGridAttributes(b, fn, spec, prefetch_types));

  if (!spec.scalar_prefetch.empty()) {
    fn->setAttr("scalar_prefetch",
                b.getI64IntegerAttr(spec.scalar_prefetch.size()));
  }
  fn->setAttr("scratch_operands", b.getI64IntegerAttr(spec.scratch.size()));

  // Populate the body. Constructing `KernelScope` installs the thread-local
  // builder and kernel arguments, and its destructor restores the previous
  // scope.
  {
    KernelScope scope(fn, spec);
    body();
    mlir::func::ReturnOp::create(scope.builder);
  }

  return Kernel(std::move(impl));
}

}  // namespace torch_tpu::kiln
