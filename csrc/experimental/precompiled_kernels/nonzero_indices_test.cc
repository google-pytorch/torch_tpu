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

#include "csrc/experimental/precompiled_kernels/nonzero_indices.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/flags/flag.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_split.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "csrc/common/cache_key.h"
#include "csrc/common/compilation.h"
#include "csrc/common/compilation_cache.h"
#include "csrc/common/dimension_types.h"
#include "csrc/common/env_vars.h"
#include "csrc/common/error_utils.h"
#include "csrc/common/shape.h"
#include "csrc/common/status_test_utils.h"
#include "csrc/common/unbounded_dynamic_kernel_registry.h"
#include "csrc/ops/nonzero/nonzero.h"
#include "csrc/pjrt/pjrt_execute.h"
#include "csrc/pjrt/pjrt_state.h"
#include "gtest/gtest.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "xla/hlo/builder/xla_computation.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/translate/mhlo_to_hlo/mlir_hlo_to_hlo.h"
#include "xla/layout.h"
#include "xla/literal.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/service/computation_layout.h"
#include "xla/shape.h"
#include "xla/shape_layout.h"
#include "xla/shape_util.h"
#include "xla/xla_data.pb.h"

// The benchmark is skipped unless --baseline names the recorded XLA timings
// for the generation under test. From this directory (use --test_output=all,
// not blaze run or --test_output=streamed, so execution runs on Forge TPU):
//
//   DIR=third_party/py/torch_tpu/csrc/experimental/precompiled_kernels
//   blaze test --test_output=all :nonzero_indices_test_v7 \
//     --test_arg=--baseline=$DIR/testdata/nonzero_xla_baseline_v7.csv
//
// Adding --test_arg=--update re-measures XLA (minutes) and prints a refreshed
// CSV block; copy it over the --baseline file. The test cannot write the file
// itself because runfiles are read-only in the remote test sandbox.
ABSL_FLAG(std::string, baseline, "",
          "XLA reference CSV. Empty: skip the benchmark.");
ABSL_FLAG(bool, update, false,
          "Re-measure XLA and print a CSV to copy over --baseline.");

namespace torch_tpu {
namespace {

// Spread exactly K nonzeros across the input, including its last element.
// Exercise NaN, infinity, negatives and subnormals; only signed zeros are zero.
xla::Literal MakeInput(int64_t rows, int64_t cols, int64_t k) {
  xla::Literal input(xla::ShapeUtil::MakeShape(xla::F32, {rows, cols}));
  const float nonzeros[] = {INFINITY, -2.f, 1e-40f};
  const int64_t size = rows * cols;
  for (int64_t i = 0; i < size; ++i) {
    bool selected = (i + 1) * k / size != i * k / size;
    input.data<float>()[i] = selected ? nonzeros[i % 3] : (i % 2 ? -0.f : 0.f);
  }
  if (k) input.data<float>()[size - 1] = NAN;
  return input;
}

// Compute coordinates independently on the CPU, in logical row-major order.
xla::Literal ReferenceIndices(const xla::Literal& input) {
  int64_t count = 0;
  for (int64_t i = 0; i < input.element_count(); ++i) {
    count += input.data<float>()[i] != 0.f;
  }
  xla::Literal result(xla::ShapeUtil::MakeShape(xla::S64, {count, 2}));
  const int64_t cols = input.shape().dimensions(1);
  for (int64_t i = 0, out = 0; i < input.element_count(); ++i) {
    if (input.data<float>()[i] != 0.f) {
      result.Set<int64_t>({out, 0}, i / cols);
      result.Set<int64_t>({out, 1}, i % cols);
      ++out;
    }
  }
  return result;
}

class NonzeroIndicesTest : public testing::Test {
 protected:
  static void SetUpTestSuite() {
    if (client_ != nullptr) return;

    SetEnv<kTorchTpuInternalUseUnboundedDynamicKernelsEnvVar>("1");

    TT_ASSERT_OK_AND_ASSIGN(client_, PjrtBackend::GetInstance().GetClient());
    ASSERT_FALSE(client_->addressable_devices().empty());
    TT_ASSERT_OK_AND_ASSIGN(
        memory_space_,
        client_->addressable_devices()[0]->default_memory_space());

    UnboundedDynamicKernel nonzero = GetNonzeroUnboundedDynamicKernel();
    const auto start = absl::Now();
    TT_ASSERT_OK_AND_ASSIGN(
        std::unique_ptr<xla::PjRtLoadedExecutable> loaded,
        nonzero.build(*client_, std::make_unique<xla::CompileOptions>()));
    TT_ASSERT_OK_AND_ASSIGN(
        SharedLoadedExecutableWithMetadata executable,
        LoadedExecutableWithMetadata::MakeShared(std::move(loaded),
                                                 /*argument_indices=*/{},
                                                 std::move(nonzero.metadata)));
    executable_ = new SharedLoadedExecutableWithMetadata(std::move(executable));
    compile_ms_ = absl::ToDoubleMilliseconds(absl::Now() - start);
  }

  // Returns the single logical input, the F32[M,N] payload. The S64[K,2]
  // destination and the S32[2] runtime shape metadata operands are synthesized
  // by torch_tpu::Execute from the metadata bindings, not supplied here.
  static absl::StatusOr<std::vector<std::unique_ptr<xla::PjRtBuffer>>>
  PrepareOperands(const xla::Literal& input) {
    const xla::Layout input_layout({1, 0}, {xla::Tile({8, 128})});
    TT_ASSIGN_OR_RETURN(
        auto input_buffer,
        client_->BufferFromHostBuffer(
            input.untyped_data(), xla::F32, input.shape().dimensions(),
            /*byte_strides=*/std::nullopt,
            xla::PjRtClient::HostBufferSemantics::kImmutableOnlyDuringCall,
            /*on_done_with_host_buffer=*/nullptr, memory_space_,
            &input_layout));
    TT_RETURN_IF_ERROR(input_buffer->GetReadyFuture().Await());
    std::vector<std::unique_ptr<xla::PjRtBuffer>> operands;
    operands.push_back(std::move(input_buffer));
    return operands;
  }

  static absl::StatusOr<PjRtBufferPointers> ExecuteKernel(
      const std::vector<std::unique_ptr<xla::PjRtBuffer>>& operands,
      const Shape& logical_output_shape) {
    std::vector<xla::PjRtBuffer*> argument_ptrs;
    argument_ptrs.reserve(operands.size());
    for (const std::unique_ptr<xla::PjRtBuffer>& operand : operands) {
      argument_ptrs.push_back(operand.get());
    }
    return Execute(*executable_, std::move(argument_ptrs),
                   {logical_output_shape});
  }

  void CheckMosaicKernelCorrectness(int64_t rows, int64_t cols, int64_t k) {
    SCOPED_TRACE(testing::Message() << rows << "x" << cols << " K=" << k);
    ASSERT_GT(k, 2);
    auto input = MakeInput(rows, cols, k);
    auto expected = ReferenceIndices(input);
    TT_ASSERT_OK_AND_ASSIGN(const xla::Layout layout,
                            client_->GetDefaultLayout(xla::S64, {k, 2}));
    // The layout the kernel bitcasts its result to. XLA's default layout for
    // S64[K,2] is {0,1:T(2,128)} only when K > 2 (for K <= 2 it is
    // {1,0:T(2,128)}, which the kernel does not support).
    EXPECT_EQ(layout, xla::Layout({0, 1}, {xla::Tile({2, 128})}));
    TT_ASSERT_OK_AND_ASSIGN(auto buffers, PrepareOperands(input));

    const Shape logical_output({k, 2}, mlir::ElementType::I64);
    TT_ASSERT_OK_AND_ASSIGN(const auto outputs,
                            ExecuteKernel(buffers, logical_output));
    ASSERT_EQ(outputs.size(), 1);
    TT_ASSERT_OK_AND_ASSIGN(auto actual, outputs[0]->ToLiteral().Await());
    EXPECT_EQ(*actual, expected);
    TT_ASSERT_OK_AND_ASSIGN(auto unchanged, buffers[0]->ToLiteral().Await());
    if (input.element_count()) {
      EXPECT_EQ(std::memcmp(input.untyped_data(), unchanged->untyped_data(),
                            input.element_count() * sizeof(float)),
                0);
    }
  }

  static inline xla::PjRtClient* client_ = nullptr;
  static inline xla::PjRtMemorySpace* memory_space_ = nullptr;
  static inline SharedLoadedExecutableWithMetadata* executable_ = nullptr;
  static inline double compile_ms_ = 0.0;
};

TEST_F(NonzeroIndicesTest, ReusesExecutableAcrossShapesAndCounts) {
  // {9, 1100} and {17, 3000} span several 1024-column tiles per row, with a
  // masked last tile and a partial last row block.
  const std::pair<int64_t, int64_t> shapes[] = {
      {1, 3},    {3, 1},    {3, 7},     {9, 129},
      {17, 257}, {9, 1100}, {17, 3000}, {1, 16385}};
  for (auto [rows, cols] : shapes) {
    const int64_t size = rows * cols;
    const int64_t ks[] = {3, 127, 128, 129, 1023, 1024, 1025, size};
    for (int64_t k : ks) {
      if (k <= size) CheckMosaicKernelCorrectness(rows, cols, k);
    }
  }
}

TEST_F(NonzeroIndicesTest, UnboundedDynamicCacheServesKernel) {
  CompilationCache& cache = CompilationCache::GetInstance();
  const CompilationCacheKey key = GetNonzeroUnboundedDynamicKernel().key;

  const std::tuple<int64_t, int64_t, int64_t> cases[] = {
      {3, 7, 5}, {17, 257, 44}, {17, 257, 4369}, {1, 16385, 1025}};
  const xla::PjRtLoadedExecutable* registered = nullptr;
  for (const auto [rows, cols, k] : cases) {
    SCOPED_TRACE(testing::Message() << rows << "x" << cols << " K=" << k);
    const Shape input_shape({rows, cols}, mlir::ElementType::F32);
    const Shape output_shape({k, 2}, mlir::ElementType::I64);

    const int64_t hits_before = cache.GetCacheHits();
    const int64_t misses_before = cache.GetCacheMisses();
    TT_ASSERT_OK_AND_ASSIGN(
        CompiledKernel kernel,
        cache.GetOrCompile(
            key, {input_shape}, {output_shape},
            [](mlir::MLIRContext&)
                -> absl::StatusOr<mlir::OwningOpRef<mlir::ModuleOp>> {
              return TT_ERROR(error::kInternal)
                     << "nonzero must be served by the registered kernel";
            },
            xla::CompileOptions()));
    // The first lookup misses and builds the kernel; later ones hit it.
    if (registered == nullptr) {
      EXPECT_EQ(cache.GetCacheHits(), hits_before);
      EXPECT_EQ(cache.GetCacheMisses(), misses_before + 1);
    } else {
      EXPECT_EQ(cache.GetCacheHits(), hits_before + 1);
      EXPECT_EQ(cache.GetCacheMisses(), misses_before);
    }
    TT_ASSERT_OK_AND_ASSIGN(SharedLoadedExecutableWithMetadata executable,
                            kernel.fixed_shape_kernel.get());
    ASSERT_TRUE(executable->unbounded_dynamic_execution_metadata().has_value());
    if (registered == nullptr) {
      registered = executable->GetLoadedExecutable();
    }
    EXPECT_EQ(executable->GetLoadedExecutable(), registered);

    const xla::Literal input = MakeInput(rows, cols, k);
    TT_ASSERT_OK_AND_ASSIGN(const auto operands, PrepareOperands(input));
    TT_ASSERT_OK_AND_ASSIGN(
        const auto outputs,
        Execute(executable, {operands[0].get()}, {output_shape}));
    ASSERT_EQ(outputs.size(), 1);
    TT_ASSERT_OK_AND_ASSIGN(auto actual, outputs[0]->ToLiteral().Await());
    EXPECT_EQ(*actual, ReferenceIndices(input));
  }

  cache.EvictAll();
}

// Logical shapes of a nonzero call, as passed to the compatibility predicate.
struct NonzeroShapesCase {
  std::string_view name;
  std::vector<Shape> inputs;
  std::vector<Shape> outputs;
};

// :kernel_registration, linked into this test, registers the kernel returned by
// GetNonzeroUnboundedDynamicKernel() during static initialization.
TEST(NonzeroRegistrationTest, KernelIsRegisteredUnderNonzeroCacheKey) {
  const UnboundedDynamicKernel nonzero = GetNonzeroUnboundedDynamicKernel();
  const UnboundedDynamicExecutionMetadata* metadata =
      UnboundedDynamicKernelRegistry::GetInstance().GetUnboundedDynamicMetadata(
          nonzero.key);
  ASSERT_NE(metadata, nullptr);
  EXPECT_EQ(metadata->pjrt_param_shapes, nonzero.metadata.pjrt_param_shapes);
}

// The kernel serves any F32[M,N] input with an S64[K,2] output, K > 2.
TEST(NonzeroRegistrationTest, FindsKernelForCompatibleShapes) {
  const UnboundedDynamicKernel nonzero = GetNonzeroUnboundedDynamicKernel();
  const NonzeroShapesCase cases[] = {
      {"square",
       {Shape({4, 4}, mlir::ElementType::F32)},
       {Shape({5, 2}, mlir::ElementType::I64)}},
      {"single row",
       {Shape({1, 16385}, mlir::ElementType::F32)},
       {Shape({1025, 2}, mlir::ElementType::I64)}},
      {"single column",
       {Shape({3, 1}, mlir::ElementType::F32)},
       {Shape({3, 2}, mlir::ElementType::I64)}},
  };
  for (const NonzeroShapesCase& c : cases) {
    SCOPED_TRACE(c.name);
    const UnboundedDynamicKernel* kernel =
        UnboundedDynamicKernelRegistry::GetInstance()
            .FindUnboundedDynamicKernel(nonzero.key.graph_key().shapeless_key(),
                                        c.inputs, c.outputs);
    ASSERT_NE(kernel, nullptr);
    EXPECT_EQ(kernel->key, nonzero.key);
    EXPECT_EQ(kernel->build, nonzero.build);
  }
}

TEST(NonzeroRegistrationTest, RejectsIncompatibleShapes) {
  const ShapelessKey nonzero_key =
      GetNonzeroUnboundedDynamicKernel().key.graph_key().shapeless_key();
  const Shape input({4, 4}, mlir::ElementType::F32);
  const Shape output({5, 2}, mlir::ElementType::I64);
  const NonzeroShapesCase cases[] = {
      {"rank-3 input", {Shape({2, 4, 4}, mlir::ElementType::F32)}, {output}},
      {"rank-1 input", {Shape({16}, mlir::ElementType::F32)}, {output}},
      {"BF16 input", {Shape({4, 4}, mlir::ElementType::BF16)}, {output}},
      {"two inputs", {input, input}, {output}},
      {"no outputs", {input}, {}},
      {"I32 output", {input}, {Shape({5, 2}, mlir::ElementType::I32)}},
      {"output not [K,2]", {input}, {Shape({5, 3}, mlir::ElementType::I64)}},
      // The kernel's output layout requires K > 2.
      {"K == 2", {input}, {Shape({2, 2}, mlir::ElementType::I64)}},
  };
  for (const NonzeroShapesCase& c : cases) {
    SCOPED_TRACE(c.name);
    EXPECT_EQ(UnboundedDynamicKernelRegistry::GetInstance()
                  .FindUnboundedDynamicKernel(nonzero_key, c.inputs, c.outputs),
              nullptr);
  }
}

// Build the real TorchTPU SHLO lowering for live XLA comparisons.
absl::StatusOr<std::unique_ptr<xla::HloModule>> BuildReference(
    const xla::Shape& input, const xla::Shape& output) {
  auto context = MakeMlirContext();
  mlir::OpBuilder b(context.get());
  auto loc = b.getUnknownLoc();
  auto module = mlir::OwningOpRef<mlir::ModuleOp>(mlir::ModuleOp::create(loc));
  b.setInsertionPointToEnd(module->getBody());
  auto function = mlir::func::FuncOp::create(
      b, loc, "main",
      b.getFunctionType(
          {mlir::RankedTensorType::get(input.dimensions(), b.getF32Type())},
          {mlir::RankedTensorType::get(output.dimensions(), b.getI64Type())}));
  auto* entry = function.addEntryBlock();
  b.setInsertionPointToStart(entry);
  mlir::MlirBuilder builder(b, loc);
  mlir::MlirOp argument(builder, entry->getArgument(0));
  TT_ASSIGN_OR_RETURN(auto result,
                      BuildNonzeroShlo(output.dimensions(0), argument));
  mlir::func::ReturnOp::create(b, loc, result.getValue());
  TT_ASSIGN_OR_RETURN(auto hlo, mlir::ConvertMlirHloToHloModule(
                                    *module, {.return_tuple = false}));
  auto* layout = hlo->mutable_config().mutable_entry_computation_layout();
  *layout->mutable_parameter_layout(0) = xla::ShapeLayout(input);
  *layout->mutable_result_layout() = xla::ShapeLayout(output);
  return hlo;
}

// Copy the input to the device with XLA's default layout and return the shape
// the reference module must take. Kept out of CompileShlo so that the measured
// compile time excludes host-to-device transfers.
absl::StatusOr<xla::Shape> TransferShloInput(
    xla::PjRtClient& client, xla::PjRtMemorySpace* memory_space,
    const xla::Literal& input,
    std::vector<std::unique_ptr<xla::PjRtBuffer>>& operands) {
  auto concrete = input.shape();
  TT_ASSIGN_OR_RETURN(*concrete.mutable_layout(),
                      client.GetDefaultLayout(xla::F32, concrete.dimensions()));
  TT_ASSIGN_OR_RETURN(
      auto buffer,
      client.BufferFromHostBuffer(
          input.untyped_data(), xla::F32, concrete.dimensions(),
          /*byte_strides=*/std::nullopt,
          xla::PjRtClient::HostBufferSemantics::kImmutableOnlyDuringCall,
          /*on_done_with_host_buffer=*/nullptr, memory_space,
          &concrete.layout()));
  TT_RETURN_IF_ERROR(buffer->GetReadyFuture().Await());
  operands.clear();
  operands.push_back(std::move(buffer));
  return concrete;
}

// Module building plus executable compilation, the counterpart of what
// SetUpTestSuite times for Mosaic.
absl::StatusOr<std::unique_ptr<xla::PjRtLoadedExecutable>> CompileShlo(
    xla::PjRtClient& client, const xla::Shape& input,
    const xla::Shape& output_shape) {
  TT_ASSIGN_OR_RETURN(auto hlo, BuildReference(input, output_shape));
  xla::CompileOptions options;
  options.argument_layouts = std::vector<xla::Shape>{input};
  options.executable_build_options.set_result_layout(output_shape);
  const xla::XlaComputation computation(hlo->ToProto());
  return client.CompileAndLoad(computation, std::move(options));
}

struct BenchmarkCase {
  int64_t rows;
  int64_t cols;
  int64_t k;
};

constexpr BenchmarkCase kBenchmarkSweep[] = {
    {8, 128, 10},          {8, 128, 1024},      {17, 257, 44},
    {17, 257, 2184},       {17, 257, 4369},     {64, 128, 82},
    {64, 128, 8192},       {256, 512, 1310},    {256, 512, 65536},
    {256, 512, 131072},    {1024, 1024, 10485}, {1024, 1024, 524288},
    {1024, 1024, 1048576}, {2048, 2048, 41943}, {2048, 2048, 2097152},
    {2048, 2048, 4194304},
};

using BaselineMap =
    std::map<std::tuple<int64_t, int64_t, int64_t>, std::pair<double, double>>;

absl::StatusOr<BaselineMap> LoadXlaBaseline(const std::string& path) {
  std::ifstream file(path);
  if (!file.is_open()) {
    return TT_ERROR(error::kInternal)
           << "Failed to open XLA baseline CSV: " << path;
  }
  BaselineMap rows;
  for (std::string line; std::getline(file, line);) {
    if (line.empty() || line[0] == '#') continue;
    int64_t r = 0, c = 0, k = 0;
    double compile_ms = 0.0, exec_us = 0.0;
    std::vector<std::string_view> cols = absl::StrSplit(line, ',');
    if (cols.size() != 5 || !absl::SimpleAtoi(cols[0], &r) ||
        !absl::SimpleAtoi(cols[1], &c) || !absl::SimpleAtoi(cols[2], &k) ||
        !absl::SimpleAtod(cols[3], &compile_ms) ||
        !absl::SimpleAtod(cols[4], &exec_us)) {
      return TT_ERROR(error::kInvalidArgument)
             << "Malformed CSV line: " << line;
    }
    rows[{r, c, k}] = {compile_ms, exec_us};
  }
  return rows;
}

TEST_F(NonzeroIndicesTest, Benchmark) {
  const std::string baseline_path = absl::GetFlag(FLAGS_baseline);
  if (baseline_path.empty()) {
    GTEST_SKIP() << "Pass --baseline=... to run the benchmark.";
  }
  const bool update = absl::GetFlag(FLAGS_update);

  BaselineMap baselines;
  if (!update) {
    TT_ASSERT_OK_AND_ASSIGN(baselines, LoadXlaBaseline(baseline_path));
  }

  std::ostringstream report;
  report << absl::StreamFormat(
      "\nMosaic compile: %.1f ms (once for all shapes/K)\n", compile_ms_);
  report << absl::StreamFormat("%-12s | %8s | %11s | %15s | %16s | %s\n",
                               "Shape (MxN)", "K", "Mosaic (us)",
                               update ? "XLA live (us)" : "XLA cached (us)",
                               "XLA compile (ms)", "Speedup");
  report << std::string(79, '-') << "\n";

  auto time_us = [](auto&& step) -> absl::StatusOr<double> {
    const auto t0 = absl::Now();
    TT_RETURN_IF_ERROR(step());
    const bool fast = absl::Now() - t0 < absl::Milliseconds(10);
    for (int i = 0; i < (fast ? 2 : 0); ++i) {
      TT_RETURN_IF_ERROR(step());
    }
    const int min_iters = fast ? 5 : 2;
    const int max_iters = fast ? 30 : 2;
    int iters = 0;
    const auto start = absl::Now();
    while (
        iters < min_iters ||
        (iters < max_iters && absl::Now() - start < absl::Milliseconds(30))) {
      TT_RETURN_IF_ERROR(step());
      ++iters;
    }
    return absl::ToDoubleMicroseconds(absl::Now() - start) / iters;
  };

  std::ostringstream updated_csv;
  updated_csv << "# rows,cols,k,compile_ms,exec_us\n";

  for (const BenchmarkCase& s : kBenchmarkSweep) {
    ASSERT_NO_FATAL_FAILURE(CheckMosaicKernelCorrectness(s.rows, s.cols, s.k));

    auto input = MakeInput(s.rows, s.cols, s.k);
    auto output_shape = xla::ShapeUtil::MakeShape(xla::S64, {s.k, 2});
    TT_ASSERT_OK_AND_ASSIGN(*output_shape.mutable_layout(),
                            client_->GetDefaultLayout(xla::S64, {s.k, 2}));
    TT_ASSERT_OK_AND_ASSIGN(const auto operands, PrepareOperands(input));

    const Shape logical_output({s.k, 2}, mlir::ElementType::I64);
    TT_ASSERT_OK_AND_ASSIGN(
        const double mosaic_us, time_us([&]() -> absl::Status {
          TT_ASSIGN_OR_RETURN(auto res,
                              ExecuteKernel(operands, logical_output));
          return res[0]->GetReadyFuture().Await();
        }));

    const auto it = baselines.find({s.rows, s.cols, s.k});
    const bool has_xla = update || it != baselines.end();
    double xla_compile_ms = has_xla && !update ? it->second.first : 0.0;
    double xla_exec_us = has_xla && !update ? it->second.second : 0.0;

    if (update) {
      std::vector<std::unique_ptr<xla::PjRtBuffer>> xla_ops;
      TT_ASSERT_OK_AND_ASSIGN(
          const xla::Shape concrete,
          TransferShloInput(*client_, memory_space_, input, xla_ops));
      const auto compile_start = absl::Now();
      TT_ASSERT_OK_AND_ASSIGN(auto xla_exec,
                              CompileShlo(*client_, concrete, output_shape));
      xla_compile_ms = absl::ToDoubleMilliseconds(absl::Now() - compile_start);

      const xla::ExecuteOptions xla_options;
      TT_ASSERT_OK_AND_ASSIGN(
          xla_exec_us, time_us([&]() -> absl::Status {
            TT_ASSIGN_OR_RETURN(
                auto res, xla_exec->ExecuteSharded(
                              {xla_ops[0].get()},
                              client_->addressable_devices()[0], xla_options));
            return res[0]->GetReadyFuture().Await();
          }));
    }

    updated_csv << absl::StreamFormat("%d,%d,%d,%.1f,%.1f\n", s.rows, s.cols,
                                      s.k, xla_compile_ms, xla_exec_us);

    const std::string shape_str = absl::StrFormat("%dx%d", s.rows, s.cols);
    if (has_xla) {
      report << absl::StreamFormat(
          "%-12s | %8d | %11.1f | %15.1f | %16.1f | %6.2fx\n", shape_str, s.k,
          mosaic_us, xla_exec_us, xla_compile_ms, xla_exec_us / mosaic_us);
    } else {
      report << absl::StreamFormat("%-12s | %8d | %11.1f | %15s | %16s | %7s\n",
                                   shape_str, s.k, mosaic_us, "—", "—", "—");
    }
  }

  if (update) {
    report << "\nCopy the block below over " << baseline_path << ":\n\n"
           << updated_csv.str();
  }
  std::cout << report.str() << std::flush;
}

}  // namespace
}  // namespace torch_tpu
