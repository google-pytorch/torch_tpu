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

#include "csrc/internal/profiler/tpu_profiler_plugin.h"

// clang-format off
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include <kineto/ActivityType.h>
#include <kineto/Config.h>
#include <kineto/GenericTraceActivity.h>
#include <kineto/IActivityProfiler.h>
#include "csrc/common/error_utils.h"
#include "tsl/profiler/protobuf/profiler_options.pb.h"
#include "tsl/profiler/protobuf/xplane.pb.h"
// clang-format on

namespace torch_tpu {

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

TEST(TpuProfilerPluginTest, UpdateProfileOptionsRunDirGcs) {
  tensorflow::ProfileOptions opts;
  std::string run_dir;

  EXPECT_TRUE(
      UpdateProfileOptions("run_dir:gs://my-bucket/my-dir", opts, run_dir)
          .ok());

  EXPECT_EQ(run_dir, "gs://my-bucket/my-dir");
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
  EXPECT_TRUE(status.ok());

  const auto& advanced_config = opts.advanced_configuration();
  auto it = advanced_config.find("unknown_option");
  ASSERT_TRUE(it != advanced_config.end());
  EXPECT_EQ(it->second.int64_value(), 1);
}

TEST(TpuProfilerPluginTest, UpdateProfileOptionsAdvancedOptionBasicTypes) {
  tensorflow::ProfileOptions opts;
  std::string run_dir;

  absl::Status status =
      UpdateProfileOptions("bool_opt:true,int_opt:42", opts, run_dir);
  EXPECT_TRUE(status.ok());

  const auto& advanced_config = opts.advanced_configuration();

  auto it_bool = advanced_config.find("bool_opt");
  ASSERT_TRUE(it_bool != advanced_config.end());
  EXPECT_TRUE(it_bool->second.bool_value());

  auto it_int = advanced_config.find("int_opt");
  ASSERT_TRUE(it_int != advanced_config.end());
  EXPECT_EQ(it_int->second.int64_value(), 42);
}

TEST(TpuProfilerPluginTest, UpdateProfileOptionsEscapedQuotes) {
  tensorflow::ProfileOptions opts;
  std::string run_dir;

  // Using R-strings to avoid backslash escape confusion in C++ source
  absl::Status status =
      UpdateProfileOptions(R"(str_single_quote_opt:"val \" ue",)"
                           R"(str_escaped_opt:"a \"nested\" quote")",
                           opts, run_dir);
  EXPECT_TRUE(status.ok());

  const auto& advanced_config = opts.advanced_configuration();

  auto it_escaped = advanced_config.find("str_escaped_opt");
  ASSERT_TRUE(it_escaped != advanced_config.end());
  EXPECT_EQ(it_escaped->second.string_value(), R"(a "nested" quote)");

  auto it_single = advanced_config.find("str_single_quote_opt");
  ASSERT_TRUE(it_single != advanced_config.end());
  EXPECT_EQ(it_single->second.string_value(), R"(val " ue)");
}

TEST(TpuProfilerPluginTest, UpdateProfileOptionsCommasInValue) {
  tensorflow::ProfileOptions opts;
  std::string run_dir;

  absl::Status status = UpdateProfileOptions(
      R"(str_comma_opt:"a \"nested\", comma")", opts, run_dir);
  EXPECT_TRUE(status.ok());

  const auto& advanced_config = opts.advanced_configuration();

  auto it_comma = advanced_config.find("str_comma_opt");
  ASSERT_TRUE(it_comma != advanced_config.end());
  EXPECT_EQ(it_comma->second.string_value(), R"(a "nested", comma)");
}

TEST(TpuProfilerPluginTest, UpdateProfileOptionsColonsInValue) {
  tensorflow::ProfileOptions opts;
  std::string run_dir;

  absl::Status status =
      UpdateProfileOptions("str_colon_opt:\"a:b\"", opts, run_dir);
  EXPECT_TRUE(status.ok());

  const auto& advanced_config = opts.advanced_configuration();

  auto it_colon = advanced_config.find("str_colon_opt");
  ASSERT_TRUE(it_colon != advanced_config.end());
  EXPECT_EQ(it_colon->second.string_value(), "a:b");
}

TEST(TpuProfilerPluginTest, UpdateProfileOptionsBackslashesInValue) {
  tensorflow::ProfileOptions opts;
  std::string run_dir;

  absl::Status status =
      UpdateProfileOptions(R"(str_bs_opt:"some\\path")", opts, run_dir);
  EXPECT_TRUE(status.ok());

  const auto& advanced_config = opts.advanced_configuration();

  auto it_bs = advanced_config.find("str_bs_opt");
  ASSERT_TRUE(it_bs != advanced_config.end());
  EXPECT_EQ(it_bs->second.string_value(), R"(some\path)");
}

TEST(TpuProfilerPluginTest, UpdateProfileOptionsNonAsciiRejected) {
  tensorflow::ProfileOptions opts;
  std::string run_dir;

  // Pass a UTF-8 character (e.g. micro sign 'µ' or similar non-ASCII)
  absl::Status status =
      UpdateProfileOptions("custom_opt:\"value_µ\"", opts, run_dir);
  EXPECT_FALSE(status.ok());
  EXPECT_EQ(status.code(), error::kInvalidArgument);
  EXPECT_NE(status.message().find("contains non-ASCII characters"),
            std::string::npos);
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

TEST(TpuProfilerPluginTest, UpdateProfileOptionsMalformedAdvancedValue) {
  tensorflow::ProfileOptions opts;
  std::string run_dir;

  absl::Status status =
      UpdateProfileOptions("unknown_option:abc", opts, run_dir);
  EXPECT_FALSE(status.ok());
  EXPECT_EQ(status.code(), error::kInvalidArgument);
  EXPECT_EQ(status.message(),
            "expected the advanced option 'unknown_option' to be a "
            "quoted string, boolean (true/false), or integer, got 'abc'");
}

TEST(TpuProfilerPluginTest, UpdateProfileOptionsUnmatchedQuote) {
  tensorflow::ProfileOptions opts;
  std::string run_dir;

  absl::Status status =
      UpdateProfileOptions("unknown_option:\"abc", opts, run_dir);
  EXPECT_FALSE(status.ok());
  EXPECT_EQ(status.code(), error::kInvalidArgument);
  EXPECT_EQ(status.message(),
            "expected the advanced option 'unknown_option' to be a "
            "quoted string, boolean (true/false), or integer, got '\"abc'");
}

TEST(TpuProfilerPluginTest, UpdateProfileOptionsUnmatchedClosingQuote) {
  tensorflow::ProfileOptions opts;
  std::string run_dir;

  absl::Status status =
      UpdateProfileOptions("unknown_option:abc\"", opts, run_dir);
  EXPECT_FALSE(status.ok());
  EXPECT_EQ(status.code(), error::kInvalidArgument);
  EXPECT_EQ(status.message(),
            "expected the advanced option 'unknown_option' to be a "
            "quoted string, boolean (true/false), or integer, got 'abc\"'");
}

TEST(TpuProfilerPluginTest, UpdateProfileOptionsInvalidEscapeSequence) {
  tensorflow::ProfileOptions opts;
  std::string run_dir;

  absl::Status status = UpdateProfileOptions(
      R"(unknown_option:"value with \x invalid escape")", opts, run_dir);
  EXPECT_FALSE(status.ok());
  EXPECT_EQ(status.code(), error::kInvalidArgument);
  EXPECT_EQ(status.message(),
            "Invalid escape sequence: \\x (only \\\\ and \\\" are supported)");
}

TEST(TpuProfilerPluginTest, UpdateProfileOptionsTrailingBackslash) {
  tensorflow::ProfileOptions opts;
  std::string run_dir;

  absl::Status status =
      UpdateProfileOptions(R"(unknown_option:"value\")", opts, run_dir);
  EXPECT_FALSE(status.ok());
  EXPECT_EQ(status.code(), error::kInvalidArgument);
  EXPECT_EQ(status.message(), "Trailing backslash in string");
}

TEST(TpuProfilerPluginTest, UpdateProfileOptionsWorkerRank) {
  tensorflow::ProfileOptions opts;
  std::string run_dir;
  std::optional<std::string> worker_rank;

  // 1. worker_rank present as integer (unquoted)
  EXPECT_TRUE(UpdateProfileOptions("device_tracer_level:3,worker_rank:2,"
                                   "host_tracer_level:3",
                                   opts, run_dir, worker_rank)
                  .ok());
  EXPECT_EQ(opts.device_tracer_level(), 3);
  EXPECT_EQ(opts.host_tracer_level(), 3);
  EXPECT_TRUE(worker_rank.has_value());
  EXPECT_EQ(*worker_rank, "2");

  // 2. worker_rank present as string (quoted)
  EXPECT_TRUE(
      UpdateProfileOptions("device_tracer_level:3,worker_rank:\"worker_A\","
                           "host_tracer_level:3",
                           opts, run_dir, worker_rank)
          .ok());
  EXPECT_EQ(opts.device_tracer_level(), 3);
  EXPECT_EQ(opts.host_tracer_level(), 3);
  EXPECT_TRUE(worker_rank.has_value());
  EXPECT_EQ(*worker_rank, "worker_A");

  // 3. worker_rank missing
  EXPECT_TRUE(
      UpdateProfileOptions("device_tracer_level:1", opts, run_dir, worker_rank)
          .ok());
  EXPECT_FALSE(worker_rank.has_value());

  // 4. worker_rank as raw string (unquoted)
  EXPECT_TRUE(
      UpdateProfileOptions("worker_rank:abc", opts, run_dir, worker_rank).ok());
  EXPECT_TRUE(worker_rank.has_value());
  EXPECT_EQ(*worker_rank, "abc");
}

TEST(TpuProfilerPluginTest, SanitizeStatString) {
  EXPECT_EQ(SanitizeStatString("plain_stat_value"), "plain_stat_value");
  EXPECT_EQ(SanitizeStatString(""), "");

  EXPECT_EQ(SanitizeStatString("\"quoted\""), "'quoted'");
  EXPECT_EQ(SanitizeStatString("foo \"bar\" baz"), "foo 'bar' baz");
  EXPECT_EQ(SanitizeStatString("\"\""), "''");

  EXPECT_EQ(SanitizeStatString("line1\nline2\r\nline3"), "line1line2line3");
  EXPECT_EQ(SanitizeStatString("\n\r\n"), "");

  EXPECT_EQ(
      SanitizeStatString("{\n  \"shape\": [1, 2],\n  \"dtype\": \"f32\"\n}"),
      "{  'shape': [1, 2],  'dtype': 'f32'}");
}

TEST(TpuProfilerPluginTest, AddActivityStringMetadata) {
  libkineto::GenericTraceActivity activity;

  AddActivityStringMetadata(activity, "op_name", "aten::add");
  EXPECT_EQ(activity.getMetadataValue("op_name"), "aten::add");

  AddActivityStringMetadata(activity, "details",
                            "name: \"linear\"\nparams: 1024\r\n");
  EXPECT_EQ(activity.getMetadataValue("details"), "name: 'linear'params: 1024");
}

TEST(TpuProfilerPluginTest, IsDevicePlane) {
  tensorflow::profiler::XPlane plane;

  plane.set_name("/device:TPU:0");
  EXPECT_TRUE(IsDevicePlane(plane));

  plane.set_name("/device:TPU:7");
  EXPECT_TRUE(IsDevicePlane(plane));

  plane.set_name("TPU");
  EXPECT_TRUE(IsDevicePlane(plane));

  // Host and runtime planes are the ones that may be dropped under budget.
  plane.set_name("/host:CPU");
  EXPECT_FALSE(IsDevicePlane(plane));

  plane.set_name("Python");
  EXPECT_FALSE(IsDevicePlane(plane));

  plane.set_name("Task Environment");
  EXPECT_FALSE(IsDevicePlane(plane));

  plane.set_name("");
  EXPECT_FALSE(IsDevicePlane(plane));
}

TEST(TpuProfilerPluginTest, CountTimelineEventsMatchesConversionFilter) {
  tensorflow::profiler::XLine line;

  EXPECT_EQ(CountTimelineEvents(line), 0);

  // Events carrying an offset participate in the timeline and are converted.
  line.add_events()->set_offset_ps(0);
  line.add_events()->set_offset_ps(1000);
  EXPECT_EQ(CountTimelineEvents(line), 2);

  // Stateless/aggregated events have no offset and are skipped by the
  // converter, so they must not be counted as dropped either.
  line.add_events()->set_duration_ps(500);
  EXPECT_EQ(CountTimelineEvents(line), 2);
}

TEST(TpuProfilerPluginTest, CountTimelineEventsAcrossPlaneLines) {
  tensorflow::profiler::XPlane plane;
  EXPECT_EQ(CountTimelineEvents(plane), 0);

  auto* first = plane.add_lines();
  first->add_events()->set_offset_ps(0);
  first->add_events()->set_offset_ps(10);

  auto* second = plane.add_lines();
  second->add_events()->set_offset_ps(20);
  second->add_events()->set_duration_ps(30);  // no offset, not counted

  plane.add_lines();  // empty line

  EXPECT_EQ(CountTimelineEvents(plane), 3);
}

TEST(TpuProfilerPluginTest, OrderPlanesDeviceFirstPutsDevicePlanesAhead) {
  tensorflow::profiler::XSpace xspace;
  // Interleave so a stable partition is distinguishable from the proto order.
  xspace.add_planes()->set_name("/host:CPU");
  xspace.add_planes()->set_name("/device:TPU:0");
  xspace.add_planes()->set_name("/host:python_tracer");
  xspace.add_planes()->set_name("TPU");
  xspace.add_planes()->set_name("/device:TPU:1");

  std::vector<tensorflow::profiler::XPlane*> ordered =
      OrderPlanesDeviceFirst(xspace);

  ASSERT_EQ(ordered.size(), 5);
  // Device planes come first, in their original relative order.
  EXPECT_EQ(ordered[0]->name(), "/device:TPU:0");
  EXPECT_EQ(ordered[1]->name(), "TPU");
  EXPECT_EQ(ordered[2]->name(), "/device:TPU:1");
  // Host planes follow, also order-preserving.
  EXPECT_EQ(ordered[3]->name(), "/host:CPU");
  EXPECT_EQ(ordered[4]->name(), "/host:python_tracer");

  // The returned pointers must alias the planes owned by the XSpace, because
  // processTrace() mutates and clears them through this vector.
  EXPECT_EQ(ordered[0], &xspace.planes(1));
}

TEST(TpuProfilerPluginTest, OrderPlanesDeviceFirstHandlesDegenerateInput) {
  tensorflow::profiler::XSpace empty;
  EXPECT_TRUE(OrderPlanesDeviceFirst(empty).empty());

  tensorflow::profiler::XSpace hosts_only;
  hosts_only.add_planes()->set_name("/host:CPU");
  hosts_only.add_planes()->set_name("/host:other");
  EXPECT_EQ(OrderPlanesDeviceFirst(hosts_only).size(), 2);

  tensorflow::profiler::XSpace devices_only;
  devices_only.add_planes()->set_name("/device:TPU:0");
  EXPECT_EQ(OrderPlanesDeviceFirst(devices_only).size(), 1);
}

// Guards a cross-component log contract: downstream audit tooling detects a
// truncated Kineto export by matching "Hit maximum event limit .* truncating
// trace parsing" against this file's log output. Rewording the message
// silently disables truncation detection downstream, so a truncated capture
// would be audited as complete. Do not relax this test; update the consumers
// first if the wording genuinely must change.
TEST(TpuProfilerPluginTest, TraceTruncationLogPrefixKeepsDetectablePhrase) {
  const std::string prefix = TraceTruncationLogPrefix(1000000, 42);

  EXPECT_TRUE(absl::StrContains(prefix, "Hit maximum event limit"));
  EXPECT_TRUE(absl::StrContains(prefix, "truncating trace parsing"));
  // The detector is a single regex requiring the limit phrase to precede the
  // truncation phrase on one line.
  EXPECT_LT(prefix.find("Hit maximum event limit"),
            prefix.find("truncating trace parsing"));
  EXPECT_EQ(prefix.find('\n'), std::string::npos);
}

TEST(TpuProfilerPluginTest, TraceTruncationLogPrefixReportsCounts) {
  const std::string prefix = TraceTruncationLogPrefix(1000000, 42);

  EXPECT_TRUE(absl::StrContains(prefix, "1000000"));
  EXPECT_TRUE(absl::StrContains(prefix, "42 events dropped"));
}

}  // namespace
}  // namespace torch_tpu
