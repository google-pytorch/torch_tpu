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

#include <string>

#include "csrc/common/status_test_utils.h"
#include "gtest/gtest.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LLVM.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/mosaic/dialect/tpu/tpu_dialect.h"
#include "xla/mosaic/dialect/tpu/transforms/serde.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/xla_data.pb.h"

namespace torch_tpu {
namespace {

TEST(SerializeMosaic, PreservesDynamicKernelAcrossRoundTrip) {
  mlir::MLIRContext context;
  context.loadDialect<mlir::arith::ArithDialect, mlir::func::FuncDialect,
                      mlir::memref::MemRefDialect, mlir::scf::SCFDialect,
                      mlir::tpu::TPUDialect>();
  // A synchronous tiled copy with runtime loop bounds. This tests transport
  // on CPU; compiling and executing the kernel belongs to the next layer.
  auto module = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
    #hbm = #tpu.memory_space<hbm>
    #vmem = #tpu.memory_space<vmem>
    #smem = #tpu.memory_space<semaphore_mem>
    module {
      func.func @main(%src: memref<?x128xf32, #hbm>,
                      %dst: memref<?x128xf32, #hbm>) {
        %dim0 = arith.constant 0 : index
        %zero = arith.constant 0 : i32
        %one = arith.constant 1 : i32
        %step = arith.constant 8 : i32
        %dim = memref.dim %src, %dim0 : memref<?x128xf32, #hbm>
        %rows = arith.index_cast %dim : index to i32
        %tile = memref.alloca() : memref<8x128xf32, #vmem>
        %sem = tpu.sem_alloc : memref<!tpu.semaphore, #smem>
        scf.for %row = %zero to %rows step %step : i32 {
          %in = tpu.memref_slice %src[%row, %zero]
            : memref<?x128xf32, #hbm> -> memref<8x128xf32, #hbm>
          %out = tpu.memref_slice %dst[%row, %zero]
            : memref<?x128xf32, #hbm> -> memref<8x128xf32, #hbm>
          tpu.enqueue_dma source(%in : memref<8x128xf32, #hbm>)
            target(%tile : memref<8x128xf32, #vmem>)
            target_semaphore(%sem : memref<!tpu.semaphore, #smem>)
          tpu.sem_wait %sem, %one : memref<!tpu.semaphore, #smem>
          tpu.enqueue_dma source(%tile : memref<8x128xf32, #vmem>)
            target(%out : memref<8x128xf32, #hbm>)
            target_semaphore(%sem : memref<!tpu.semaphore, #smem>)
          tpu.sem_wait %sem, %one : memref<!tpu.semaphore, #smem>
        }
        return
      }
    }
  )mlir",
                                                        &context);
  ASSERT_TRUE(module);
  mlir::OwningOpRef<mlir::ModuleOp> original = module->clone();
  TT_ASSERT_OK_AND_ASSIGN(const auto bytes, SerializeMosaic(*module));
  auto restored = mlir::parseSourceString<mlir::ModuleOp>(bytes, &context);
  ASSERT_TRUE(restored);
  mlir::PassManager passes(&context);
  passes.addPass(mlir::tpu::createMosaicSerdePass(
      mlir::tpu::MosaicSerdePassOptions{.serialize = false}));
  ASSERT_TRUE(mlir::succeeded(passes.run(*restored)));
  ASSERT_TRUE(mlir::succeeded(mlir::verify(*restored)));
  // Deserialization retains the transport version on the enclosing module.
  (*restored)->removeAttr("stable_mosaic.version");
  EXPECT_TRUE(mlir::OperationEquivalence::isEquivalentTo(
      original->getOperation(), restored->getOperation(),
      mlir::OperationEquivalence::IgnoreLocations));
}

TEST(DynamicKernelHlo, PreservesLayoutsAndBindsRuntimeExtents) {
  auto input = xla::ShapeUtil::MakeShape(xla::F32, {8, 128});
  const auto output = xla::ShapeUtil::MakeShape(xla::S32, {2, 8, 128});
  input.mutable_layout()->add_tiles()->add_dimensions(8).add_dimensions(128);
  input.mutable_layout()->set_dynamic_shape_metadata_prefix_bytes(16);
  TT_ASSERT_OK_AND_ASSIGN(
      const auto module,
      BuildDynamicKernelHlo(
          "mosaic", {input, output, xla::ShapeUtil::MakeShape(xla::S32, {2}),
                     xla::ShapeUtil::MakeShape(xla::S32, {3})}));
  // The builder must strip the caller's prefix from the payload layouts.
  input.mutable_layout()->set_dynamic_shape_metadata_prefix_bytes(0);
  const auto* entry = module->entry_computation();
  ASSERT_EQ(entry->num_parameters(), 4);
  for (const int p : {0, 1}) {
    const auto& shape = entry->parameter_instruction(p)->shape();
    EXPECT_EQ(shape.element_type(), p == 0 ? xla::F32 : xla::S32);
    EXPECT_EQ(shape.dimensions().size(), p == 0 ? 2 : 3);
    EXPECT_EQ(shape.layout(), (p == 0 ? input : output).layout());
    for (int d = 0; d < shape.dimensions().size(); ++d) {
      EXPECT_TRUE(shape.is_dynamic_dimension(d));
      EXPECT_EQ(shape.dimensions(d), xla::Shape::kUnboundedSize);
    }
  }
  EXPECT_EQ(entry->parameter_instruction(2)->shape(),
            xla::ShapeUtil::MakeShape(xla::S32, {2}));
  EXPECT_EQ(entry->parameter_instruction(3)->shape(),
            xla::ShapeUtil::MakeShape(xla::S32, {3}));
  const auto* call = entry->root_instruction();
  EXPECT_EQ(call->opcode(), xla::HloOpcode::kCustomCall);
  EXPECT_EQ(call->custom_call_target(), "tpu_custom_call");
  EXPECT_EQ(call->shape(), entry->parameter_instruction(1)->shape());
  EXPECT_NE(call->raw_backend_config_string().find(
                R"("operand_to_metadata":[2,3,-1,-1])"),
            std::string::npos);
  ASSERT_EQ(call->output_operand_aliasing().size(), 1);
  EXPECT_EQ(call->output_operand_aliasing()[0].second.first, 1);
  const auto alias =
      module->input_output_alias_config().GetAliasedParameter({});
  ASSERT_TRUE(alias);
  EXPECT_TRUE(alias->must_alias());
  EXPECT_EQ(alias->parameter_number, 1);
  EXPECT_TRUE(alias->parameter_index.empty());
}

}  // namespace
}  // namespace torch_tpu
