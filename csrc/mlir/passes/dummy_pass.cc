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

#include "csrc/mlir/passes/dummy_pass.h"

#include <memory>

#include "llvm/ADT/StringRef.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/TypeID.h"

namespace torch_tpu {
namespace {

struct DummyPass
    : public mlir::PassWrapper<DummyPass, mlir::OperationPass<mlir::ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(DummyPass)

  llvm::StringRef getArgument() const override { return "dummy-pass"; }
  llvm::StringRef getDescription() const override { return "Dummy pass."; }

  void runOnOperation() override {
    mlir::ModuleOp module = getOperation();
    mlir::MLIRContext* context = &getContext();
    module->setAttr("torch_tpu.dummy_attr",
                    mlir::StringAttr::get(context, "dummy_value"));
  }
};

}  // namespace

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> createDummyPass() {
  return std::make_unique<DummyPass>();
}

}  // namespace torch_tpu
