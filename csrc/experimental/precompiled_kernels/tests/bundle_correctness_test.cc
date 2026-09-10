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

// Direct PJRT correctness checks, independent of runtime matching and hooks.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/flags/flag.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "csrc/common/dimension_types.h"
#include "csrc/common/status_test_utils.h"
#include "csrc/experimental/precompiled_kernels/bundle.pb.h"
#include "gtest/gtest.h"
#include "xla/literal.h"
#include "xla/pjrt/c_api_client/pjrt_c_api_client.h"
#include "xla/pjrt/pjrt_api.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_layout.h"
#include "xla/shape_util.h"

ABSL_FLAG(std::string, bundle_path, "", "Path to kernels.pb.");

namespace torch_tpu {
namespace {
using precompiled_kernels::proto::Bundle;
using precompiled_kernels::proto::Kernel;
using precompiled_kernels::proto::Parameter;
struct NamedReference {
  std::string_view op;
  int inputs;
  float (*evaluate)(float, float);
};

constexpr NamedReference kReferences[] = {
    {"relu", 1,
     [](float x, float) { return std::isnan(x) ? x : (x > 0 ? x : 0.f); }},
    {"neg.out", 1, [](float x, float) { return -x; }},
    {"abs.out", 1, [](float x, float) { return std::fabs(x); }},
    {"add.out", 2, [](float x, float y) { return x + y; }},
    {"sub.out", 2, [](float x, float y) { return x - y; }},
    {"mul.out", 2, [](float x, float y) { return x * y; }},
    {"maximum.out", 2,
     [](float x, float y) {
       if (std::isnan(x) || std::isnan(y)) return NAN;
       if (x == 0 && y == 0)
         return std::signbit(x) && std::signbit(y) ? -0.f : 0.f;
       return std::max(x, y);
     }},
    {"minimum.out", 2,
     [](float x, float y) {
       if (std::isnan(x) || std::isnan(y)) return NAN;
       if (x == 0 && y == 0)
         return std::signbit(x) || std::signbit(y) ? -0.f : 0.f;
       return std::min(x, y);
     }},
};

const NamedReference* FindReference(std::string_view op) {
  for (const auto& ref : kReferences) {
    if (ref.op == op) return &ref;
  }
  return nullptr;
}

// Distinct inputs; all pairs of special values cover either NaN operand,
// signed zeros, infinities, BF16 rounding and finite multiplication overflow.
float InputValue(int64_t i, int input) {
  constexpr float special[] = {-0.f,   0.f,         -INFINITY, INFINITY, NAN,
                               -3.25f, 1.00390625f, 3.14f,     0x1p120f};
  constexpr int n = sizeof(special) / sizeof(special[0]);
  return i < n * n
             ? special[input == 0 ? i % n : i / n]
             : static_cast<float>((i * 7919 + input * 104729) % 613) / 37.f -
                   8.f;
}

void CheckValues(xla::PjRtBuffer& buffer, const xla::Literal& expected) {
  EXPECT_FALSE(buffer.on_device_shape().is_dynamic());
  ASSERT_TRUE(
      xla::ShapeUtil::Compatible(buffer.on_device_shape(), expected.shape()));
  TT_ASSERT_OK_AND_ASSIGN(auto literal, buffer.ToLiteralSync());
  TT_ASSERT_OK_AND_ASSIGN(auto actual, literal->Convert(xla::F32));
  TT_ASSERT_OK_AND_ASSIGN(auto want, expected.Convert(xla::F32));
  for (int64_t i = 0; i < want.element_count(); ++i) {
    const float x = actual.data<float>()[i], y = want.data<float>()[i];
    ASSERT_TRUE(std::isnan(y)
                    ? std::isnan(x)
                    : x == y && (y != 0 || std::signbit(x) == std::signbit(y)))
        << "element " << i << ": got " << x << ", want " << y;
  }
}

void CheckCase(xla::PjRtClient& client, xla::PjRtLoadedExecutable& executable,
               const Kernel& kernel, const NamedReference& reference,
               const Dimensions& dims) {
  auto* device = client.addressable_devices()[0];
  TT_ASSERT_OK_AND_ASSIGN(auto* memory, device->default_memory_space());
  TT_ASSERT_OK_AND_ASSIGN(auto layouts, executable.GetParameterLayouts());
  ASSERT_EQ(layouts.size(), kernel.parameters_size());
  const auto dtype = kernel.outputs(0).dtype();
  xla::Literal expected(xla::ShapeUtil::MakeShape(xla::F32, dims));
  std::vector<xla::Literal> tensors, rounded_inputs;
  for (int j = 0; j < reference.inputs; ++j) {
    xla::Literal host(expected.shape());
    for (int64_t i = 0; i < host.element_count(); ++i)
      host.data<float>()[i] = InputValue(i, j);
    TT_ASSERT_OK_AND_ASSIGN(auto typed, host.Convert(dtype));
    TT_ASSERT_OK_AND_ASSIGN(auto rounded, typed.Convert(xla::F32));
    tensors.push_back(std::move(typed));
    rounded_inputs.push_back(std::move(rounded));
  }
  // Reference arithmetic uses rounded inputs and rounds the output to dtype.
  for (int64_t i = 0; i < expected.element_count(); ++i)
    expected.data<float>()[i] = reference.evaluate(
        rounded_inputs[0].data<float>()[i],
        reference.inputs == 2 ? rounded_inputs[1].data<float>()[i] : 0.f);
  TT_ASSERT_OK_AND_ASSIGN(auto typed_expected, expected.Convert(dtype));
  tensors.push_back(std::move(typed_expected));
  xla::Literal extents(
      xla::ShapeUtil::MakeShape(xla::S32, {static_cast<int64_t>(dims.size())}));
  std::copy(dims.begin(), dims.end(), extents.data<int32_t>().begin());

  std::vector<std::unique_ptr<xla::PjRtBuffer>> buffers;
  std::vector<xla::PjRtBuffer*> operands;
  xla::ExecuteOptions options;
  options.strict_shape_checking = false;
  for (int p = 0; p < kernel.parameters_size(); ++p) {
    const auto& binding = kernel.parameters(p);
    ASSERT_LT(binding.tensor(), tensors.size());
    const auto& layout = layouts[p]->xla_layout();
    std::unique_ptr<xla::PjRtBuffer> buffer;
    if (binding.role() == Parameter::OUTPUT) {
      ASSERT_EQ(binding.tensor(), reference.inputs);
      auto shape = tensors.back().shape();
      *shape.mutable_layout() = layout;
      TT_ASSERT_OK_AND_ASSIGN(buffer,
                              client.CreateUninitializedBuffer(shape, memory));
    } else {
      ASSERT_TRUE(binding.role() == Parameter::INPUT ||
                  binding.role() == Parameter::RUNTIME_DIMENSIONS);
      if (binding.role() == Parameter::INPUT)
        ASSERT_LT(binding.tensor(), reference.inputs);
      const auto& host = binding.role() == Parameter::RUNTIME_DIMENSIONS
                             ? extents
                             : tensors[binding.tensor()];
      TT_ASSERT_OK_AND_ASSIGN(
          buffer,
          client.BufferFromHostBuffer(
              host.untyped_data(), host.shape().element_type(),
              host.shape().dimensions(), std::nullopt,
              xla::PjRtClient::HostBufferSemantics::kImmutableOnlyDuringCall,
              nullptr, memory, &layout));
      options.non_donatable_input_indices.insert(p);
    }
    operands.push_back(buffer.get());
    buffers.push_back(std::move(buffer));
  }
  std::optional<xla::Future<>> done;
  auto result =
      executable.ExecuteSharded(operands, device, options, done, true);
  // Keep arguments alive through completion, including error completion.
  if (done) ASSERT_TRUE(done->Await().ok());
  TT_ASSERT_OK_AND_ASSIGN(auto outputs, std::move(result));
  for (const auto& output : outputs)
    ASSERT_TRUE(output->GetReadyFuture().Await().ok());
  ASSERT_EQ(outputs.size(), 1);
  CheckValues(*outputs[0], tensors.back());
  for (int p = 0; p < kernel.parameters_size(); ++p) {
    const auto& binding = kernel.parameters(p);
    if (binding.role() != Parameter::INPUT) continue;
    SCOPED_TRACE(absl::StrCat("unchanged input ", binding.tensor()));
    CheckValues(*buffers[p], tensors[binding.tensor()]);
  }
}

TEST(BundleCorrectness, EveryVariantMatchesReference) {
  const char* library = std::getenv("TPU_LIBRARY_PATH");
  ASSERT_NE(library, nullptr);
  auto loaded = pjrt::LoadPjrtPlugin("tpu", library);
  ASSERT_TRUE(loaded.ok() || absl::IsAlreadyExists(loaded.status()))
      << loaded.status();
  TT_ASSERT_OK_AND_ASSIGN(bool initialized,
                          pjrt::IsPjrtPluginInitialized("tpu"));
  if (!initialized) ASSERT_TRUE(pjrt::InitializePjrtPlugin("tpu").ok());
  TT_ASSERT_OK_AND_ASSIGN(auto client, xla::GetCApiClient("tpu"));
  Bundle bundle;
  std::ifstream file(absl::GetFlag(FLAGS_bundle_path), std::ios::binary);
  ASSERT_TRUE(bundle.ParseFromIstream(&file)) << "unreadable --bundle_path";
  ASSERT_EQ(bundle.format_version(), 1);
  ASSERT_FALSE(client->addressable_devices().empty());
  ASSERT_EQ(bundle.device_kind(),
            client->addressable_devices()[0]->device_kind());
  ASSERT_EQ(bundle.platform_version(), client->platform_version());
  ASSERT_GT(bundle.kernels_size(), 0);
  const std::vector<Dimensions> shapes = {
      {8, 128},  {3, 7},         {9, 129},       {33, 257},
      {1, 1},    {16, 1024},     {2, 8, 128},    {3, 9, 129},
      {1, 1, 1}, {2, 3, 8, 128}, {1, 2, 9, 129}, {1, 1, 1, 1}};
  for (const auto& kernel : bundle.kernels()) {
    SCOPED_TRACE(kernel.op());
    const NamedReference* reference = FindReference(kernel.op());
    ASSERT_NE(reference, nullptr) << "no reference for " << kernel.op();
    ASSERT_EQ(kernel.inputs_size(), reference->inputs);
    ASSERT_EQ(kernel.outputs_size(), 1);
    const auto& output = kernel.outputs(0);
    ASSERT_TRUE(output.dtype() == xla::F32 || output.dtype() == xla::BF16);
    ASSERT_GE(output.dims_size(), 2);
    ASSERT_LE(output.dims_size(), 4);
    for (const auto& input : kernel.inputs()) {
      ASSERT_EQ(input.dtype(), output.dtype());
      ASSERT_EQ(input.dims_size(), output.dims_size());
    }
    TT_ASSERT_OK_AND_ASSIGN(
        auto executable,
        client->LoadSerializedExecutable(kernel.executable(), std::nullopt,
                                         xla::LoadOptions()));
    for (const auto& dims : shapes) {
      if (dims.size() != output.dims_size()) continue;
      SCOPED_TRACE(absl::StrCat(xla::PrimitiveType_Name(output.dtype()), " ",
                                absl::StrJoin(dims, "x")));
      CheckCase(*client, *executable, kernel, *reference, dims);
    }
  }
}
}  // namespace
}  // namespace torch_tpu
