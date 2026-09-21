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
#include "csrc/common/compilation.h"
#include "csrc/common/error_utils.h"
#include "csrc/common/status_test_utils.h"
#include "csrc/experimental/precompiled_kernels/kernel_test_utils.h"
#include "csrc/experimental/precompiled_kernels/mosaic_custom_call.h"
#include "csrc/internal/mosaic/op_builders.h"
#include "csrc/ops/nonzero/nonzero.h"
#include "gtest/gtest.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/translate/mhlo_to_hlo/mlir_hlo_to_hlo.h"
#include "xla/layout.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/service/computation_layout.h"
#include "xla/service/hlo_runner.h"
#include "xla/service/hlo_runner_interface.h"
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
    if (runner_ != nullptr) return;
    TT_ASSERT_OK_AND_ASSIGN(auto owned_client, CreateKernelTestClient());
    client_ = owned_client.get();
    runner_ = new xla::HloRunner(std::move(owned_client));

    auto make_unbounded = [](xla::PrimitiveType type, xla::Tile tile) {
      xla::Shape s = xla::ShapeUtil::MakeShape(
          type, {xla::Shape::kUnboundedSize, xla::Shape::kUnboundedSize},
          {true, true});
      *s.mutable_layout() = xla::Layout({1, 0}, {tile});
      return s;
    };
    xla::Shape input = make_unbounded(xla::F32, xla::Tile({8, 128}));
    xla::Shape output_view = make_unbounded(xla::S32, xla::Tile({1, 128}));
    xla::Shape dims = xla::ShapeUtil::MakeShape(xla::S32, {2});
    *dims.mutable_layout() = xla::Layout({0}, {xla::Tile({128})});
    parameters_ = new std::vector<xla::Shape>{input, output_view, dims, dims};

    const auto start = absl::Now();
    auto context = mlir::torch_tpu::CreateMlirContextWithDialects();
    auto module = BuildNonzeroIndices(*context);
    TT_ASSERT_OK_AND_ASSIGN(
        auto hlo,
        BuildMosaicCustomCall(*module, *parameters_, output_view,
                              {2, 3, -1, -1}, /*aliased_parameter=*/1));
    TT_ASSERT_OK_AND_ASSIGN(
        auto exec,
        runner_->CreateExecutable(std::move(hlo), /*run_hlo_passes=*/false));
    executable_ = exec.release();
    compile_ms_ = absl::ToDoubleMilliseconds(absl::Now() - start);
  }

  static absl::StatusOr<std::unique_ptr<xla::PjRtBuffer>> AllocateIndexOutput(
      const xla::Shape& output) {
    TT_ASSIGN_OR_RETURN(
        auto* memory,
        client_->addressable_devices()[0]->default_memory_space());
    TT_ASSIGN_OR_RETURN(auto buffer,
                        client_->CreateUninitializedBuffer(output, memory));
    TT_ASSIGN_OR_RETURN(auto bytes, buffer->GetOnDeviceSizeInBytes());
    return buffer->Bitcast(
        xla::S32, {static_cast<int64_t>(bytes / (128 * sizeof(int32_t))), 128},
        &(*parameters_)[1].layout());
  }

  static absl::StatusOr<std::vector<std::unique_ptr<xla::PjRtBuffer>>>
  PrepareOperands(const xla::Literal& input, const xla::Shape& output) {
    TT_ASSIGN_OR_RETURN(auto destination, AllocateIndexOutput(output));
    auto input_dims = xla::LiteralUtil::CreateR1<int32_t>(
        {static_cast<int32_t>(input.shape().dimensions(0)),
         static_cast<int32_t>(input.shape().dimensions(1))});
    auto output_dims = xla::LiteralUtil::CreateR1<int32_t>(
        {static_cast<int32_t>(destination->on_device_shape().dimensions(0)),
         128});
    auto concrete = input.shape();
    *concrete.mutable_layout() = (*parameters_)[0].layout();
    TT_ASSIGN_OR_RETURN(auto buffers, runner_->TransferLiteralsToDefaultDevice(
                                          {xla::ShapeLayout(concrete),
                                           xla::ShapeLayout((*parameters_)[2]),
                                           xla::ShapeLayout((*parameters_)[3])},
                                          {&input, &input_dims, &output_dims}));
    for (const auto& buffer : buffers) {
      TT_RETURN_IF_ERROR(buffer->GetReadyFuture().Await());
    }
    buffers.insert(buffers.begin() + 1, std::move(destination));
    return buffers;
  }

  void CheckMosaicKernelCorrectness(int64_t rows, int64_t cols, int64_t k) {
    SCOPED_TRACE(testing::Message() << rows << "x" << cols << " K=" << k);
    ASSERT_GT(k, 2);
    auto input = MakeInput(rows, cols, k);
    auto expected = ReferenceIndices(input);
    auto output_shape = expected.shape();
    TT_ASSERT_OK_AND_ASSIGN(auto layout,
                            client_->GetDefaultLayout(xla::S64, {k, 2}));
    // XLA default layout for S64[K,2] is {0,1:T(2,128)} only when K > 2
    // (for K <= 2 it is {1,0:T(2,128)}, which the kernel does not support).
    EXPECT_EQ(layout, xla::Layout({0, 1}, {xla::Tile({2, 128})}));
    *output_shape.mutable_layout() = layout;
    TT_ASSERT_OK_AND_ASSIGN(auto buffers, PrepareOperands(input, output_shape));

    xla::ExecuteOptions options;
    options.strict_shape_checking = false;
    TT_ASSERT_OK_AND_ASSIGN(auto outputs, runner_->ExecuteWithDeviceBuffers(
                                              executable_, buffers, &options));
    TT_ASSERT_OK_AND_ASSIGN(
        auto native,
        outputs[0]->Bitcast(xla::S64, {k, 2}, &output_shape.layout()));
    TT_ASSERT_OK_AND_ASSIGN(auto actual, native->ToLiteral().Await());
    EXPECT_EQ(*actual, expected);
    TT_ASSERT_OK_AND_ASSIGN(auto unchanged, buffers[0]->ToLiteral().Await());
    if (input.element_count()) {
      EXPECT_EQ(std::memcmp(input.untyped_data(), unchanged->untyped_data(),
                            input.element_count() * sizeof(float)),
                0);
    }
  }

  static inline xla::PjRtClient* client_ = nullptr;
  static inline xla::HloRunner* runner_ = nullptr;
  static inline xla::OpaqueExecutable* executable_ = nullptr;
  static inline std::vector<xla::Shape>* parameters_ = nullptr;
  static inline double compile_ms_ = 0.0;
};

TEST_F(NonzeroIndicesTest, ReusesExecutableAcrossShapesAndCounts) {
  const std::pair<int64_t, int64_t> shapes[] = {
      {1, 3}, {3, 1}, {3, 7}, {9, 129}, {17, 257}, {1, 16385}};
  for (auto [rows, cols] : shapes) {
    const int64_t size = rows * cols;
    const int64_t ks[] = {3, 127, 128, 129, 1023, 1024, 1025, size};
    for (int64_t k : ks) {
      if (k <= size) CheckMosaicKernelCorrectness(rows, cols, k);
    }
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
    xla::HloRunner& runner, xla::PjRtClient& client, const xla::Literal& input,
    std::vector<std::unique_ptr<xla::PjRtBuffer>>& operands) {
  auto concrete = input.shape();
  TT_ASSIGN_OR_RETURN(*concrete.mutable_layout(),
                      client.GetDefaultLayout(xla::F32, concrete.dimensions()));
  TT_ASSIGN_OR_RETURN(operands, runner.TransferLiteralsToDefaultDevice(
                                    {xla::ShapeLayout(concrete)}, {&input}));
  for (const auto& op : operands) {
    TT_RETURN_IF_ERROR(op->GetReadyFuture().Await());
  }
  return concrete;
}

// Module building plus executable compilation, the counterpart of what
// SetUpTestSuite times for Mosaic.
absl::StatusOr<std::unique_ptr<xla::OpaqueExecutable>> CompileShlo(
    xla::HloRunner& runner, const xla::Shape& input,
    const xla::Shape& output_shape) {
  TT_ASSIGN_OR_RETURN(auto hlo, BuildReference(input, output_shape));
  return runner.CreateExecutable(std::move(hlo), /*run_hlo_passes=*/true);
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
    TT_ASSERT_OK_AND_ASSIGN(auto operands,
                            PrepareOperands(input, output_shape));

    xla::ExecuteOptions options;
    options.strict_shape_checking = false;

    TT_ASSERT_OK_AND_ASSIGN(
        const double mosaic_us, time_us([&]() -> absl::Status {
          TT_ASSIGN_OR_RETURN(operands[1], AllocateIndexOutput(output_shape));
          TT_ASSIGN_OR_RETURN(auto res, runner_->ExecuteWithDeviceBuffers(
                                            executable_, operands, &options));
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
          TransferShloInput(*runner_, *client_, input, xla_ops));
      const auto compile_start = absl::Now();
      TT_ASSERT_OK_AND_ASSIGN(auto xla_exec,
                              CompileShlo(*runner_, concrete, output_shape));
      xla_compile_ms = absl::ToDoubleMilliseconds(absl::Now() - compile_start);

      TT_ASSERT_OK_AND_ASSIGN(xla_exec_us, time_us([&]() -> absl::Status {
                                TT_ASSIGN_OR_RETURN(
                                    auto res, runner_->ExecuteWithDeviceBuffers(
                                                  xla_exec.get(), xla_ops,
                                                  /*execute_options=*/nullptr));
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
