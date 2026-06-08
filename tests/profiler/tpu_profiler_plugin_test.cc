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

absl::Status UpdateProfileOptions(std::string_view custom_config,
                                  tensorflow::ProfileOptions& opts,
                                  std::string& out_run_dir);

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
  std::string run_dir;
  opts.set_device_tracer_level(1);
  opts.set_host_tracer_level(2);
  opts.set_python_tracer_level(0);

  EXPECT_TRUE(UpdateProfileOptions("device_tracer_level:3,host_tracer_level:3,"
                                   "python_tracer_level:1",
                                   opts, run_dir)
                  .ok());

  EXPECT_EQ(opts.device_tracer_level(), 3);
  EXPECT_EQ(opts.host_tracer_level(), 3);
  EXPECT_EQ(opts.python_tracer_level(), 1);
  EXPECT_TRUE(run_dir.empty());
}

TEST(TpuProfilerPluginTest, UpdateProfileOptionsRunDir) {
  tensorflow::ProfileOptions opts;
  std::string run_dir;

  EXPECT_TRUE(
      UpdateProfileOptions("device_tracer_level:3,run_dir:/home/user/tb_logs,"
                           "host_tracer_level:3",
                           opts, run_dir)
          .ok());

  EXPECT_EQ(opts.device_tracer_level(), 3);
  EXPECT_EQ(opts.host_tracer_level(), 3);
  EXPECT_EQ(run_dir, "/home/user/tb_logs");
}

TEST(TpuProfilerPluginTest, UpdateProfileOptionsWhitespace) {
  tensorflow::ProfileOptions opts;
  std::string run_dir;
  opts.set_device_tracer_level(1);
  opts.set_host_tracer_level(2);
  opts.set_python_tracer_level(0);

  EXPECT_TRUE(UpdateProfileOptions(
                  " device_tracer_level : 3 ,  host_tracer_level : 3 , "
                  " python_tracer_level : 1 ",
                  opts, run_dir)
                  .ok());

  EXPECT_EQ(opts.device_tracer_level(), 3);
  EXPECT_EQ(opts.host_tracer_level(), 3);
  EXPECT_EQ(opts.python_tracer_level(), 1);
  EXPECT_TRUE(run_dir.empty());
}

TEST(TpuProfilerPluginTest, UpdateProfileOptionsInvalidFormat) {
  tensorflow::ProfileOptions opts;
  std::string run_dir;

  absl::Status status = UpdateProfileOptions("invalid_format", opts, run_dir);
  EXPECT_FALSE(status.ok());
  EXPECT_EQ(status.code(), error::kInvalidArgument);
}

TEST(TpuProfilerPluginTest, UpdateProfileOptionsUnknownOption) {
  tensorflow::ProfileOptions opts;
  std::string run_dir;

  absl::Status status = UpdateProfileOptions("unknown_option:1", opts, run_dir);
  EXPECT_FALSE(status.ok());
  EXPECT_EQ(status.code(), error::kInvalidArgument);
}

TEST(TpuProfilerPluginTest, UpdateProfileOptionsInvalidValue) {
  tensorflow::ProfileOptions opts;
  std::string run_dir;

  absl::Status status =
      UpdateProfileOptions("device_tracer_level:abc", opts, run_dir);
  EXPECT_FALSE(status.ok());
  EXPECT_EQ(status.code(), error::kInvalidArgument);
  EXPECT_EQ(status.message(),
            "expected the value of parameter device_tracer_level to be an "
            "integer, got 'abc'");
}

}  // namespace
}  // namespace torch_tpu
