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

#include "torch_tpu/_internal/profiler/tpu_profiler_plugin.h"

#include <kineto/ActivityType.h>
#include <kineto/Config.h>
#include <kineto/IActivityProfiler.h>

#include <memory>
#include <set>
#include <string>

#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "gtest/gtest.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/common/utils.h"
#include "tsl/profiler/protobuf/profiler_options.pb.h"

namespace torch_tpu {

absl::Status UpdateProfileOptions(tensorflow::ProfileOptions& opts,
                                  std::string_view custom_config);

namespace {

TEST(TpuProfilerPluginTest, BasicSanity) {
  TpuProfiler profiler;

  // 1. Check name
  EXPECT_EQ(profiler.name(), "tpu_profiler");

  // 2. Check available activities
  const auto& activities = profiler.availableActivities();
  EXPECT_NE(activities.find(libkineto::ActivityType::PRIVATEUSE1_RUNTIME),
            activities.end());

  // 3. Configure session
  libkineto::Config config;
  std::set<libkineto::ActivityType> active_types = {
      libkineto::ActivityType::PRIVATEUSE1_RUNTIME};
  auto session = profiler.configure(active_types, config);
  ASSERT_NE(session, nullptr);

  // 4. Start and Stop
  session->start();
  session->stop();

  // We expect some errors since we are not running on a real TPU
  EXPECT_TRUE(session->errors().empty());
}

TEST(TpuProfilerPluginTest, UpdateProfileOptionsValid) {
  tensorflow::ProfileOptions opts;
  opts.set_device_tracer_level(1);
  opts.set_host_tracer_level(2);
  opts.set_python_tracer_level(0);

  EXPECT_TRUE(UpdateProfileOptions(opts,
                                   "device_tracer_level:3,host_tracer_level:3,"
                                   "python_tracer_level:1")
                  .ok());

  EXPECT_EQ(opts.device_tracer_level(), 3);
  EXPECT_EQ(opts.host_tracer_level(), 3);
  EXPECT_EQ(opts.python_tracer_level(), 1);
}

TEST(TpuProfilerPluginTest, UpdateProfileOptionsWhitespace) {
  tensorflow::ProfileOptions opts;
  opts.set_device_tracer_level(1);
  opts.set_host_tracer_level(2);
  opts.set_python_tracer_level(0);

  EXPECT_TRUE(UpdateProfileOptions(
                  opts,
                  " device_tracer_level : 3 ,  host_tracer_level : 3 , "
                  " python_tracer_level : 1 ")
                  .ok());

  EXPECT_EQ(opts.device_tracer_level(), 3);
  EXPECT_EQ(opts.host_tracer_level(), 3);
  EXPECT_EQ(opts.python_tracer_level(), 1);
}

TEST(TpuProfilerPluginTest, UpdateProfileOptionsInvalidFormat) {
  tensorflow::ProfileOptions opts;

  absl::Status status = UpdateProfileOptions(opts, "invalid_format");
  EXPECT_FALSE(status.ok());
  EXPECT_EQ(status.code(), error::kInvalidArgument);
}

TEST(TpuProfilerPluginTest, UpdateProfileOptionsUnknownOption) {
  tensorflow::ProfileOptions opts;

  absl::Status status = UpdateProfileOptions(opts, "unknown_option:1");
  EXPECT_FALSE(status.ok());
  EXPECT_EQ(status.code(), error::kInvalidArgument);
}

TEST(TpuProfilerPluginTest, UpdateProfileOptionsInvalidValue) {
  tensorflow::ProfileOptions opts;

  absl::Status status = UpdateProfileOptions(opts, "device_tracer_level:abc");
  EXPECT_FALSE(status.ok());
  EXPECT_EQ(status.code(), error::kInvalidArgument);
  EXPECT_EQ(status.message(),
            "expected the value of parameter device_tracer_level to be an "
            "integer, got 'abc'");
}

}  // namespace
}  // namespace torch_tpu
