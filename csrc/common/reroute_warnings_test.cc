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

#include "csrc/common/reroute_warnings.h"

#include <atomic>
#include <concepts>
#include <string>
#include <thread>  // NOLINT(build/c++11)
#include <variant>
#include <vector>

#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "c10/util/Exception.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace torch_tpu {
namespace {

using testing::ElementsAre;
using testing::ElementsAreArray;
using testing::HasSubstr;
using testing::IsEmpty;
using testing::SizeIs;

// Records the c10 warnings raised on the constructing thread while in scope.
class WarningCapture : public c10::WarningHandler {
 public:
  WarningCapture() : guard_(this) {}

  void process(const c10::Warning& warning) override {
    warnings_.push_back(warning);
  }

  [[nodiscard]] const std::vector<c10::Warning>& warnings() const {
    return warnings_;
  }

  [[nodiscard]] std::vector<std::string> messages() const {
    std::vector<std::string> messages;
    messages.reserve(warnings_.size());
    for (const c10::Warning& warning : warnings_) {
      messages.push_back(warning.msg());
    }
    return messages;
  }

 private:
  std::vector<c10::Warning> warnings_;
  c10::WarningUtils::WarningHandlerGuard guard_;
};

// Runs `f` on a new thread that reroutes its warnings, and waits for it.
template <std::invocable F>
void RunOnReroutingThread(F f) {
  std::thread thread([&f] {
    RouteWarningsOnThisThreadToDispatchThread();
    f();
  });
  thread.join();
}

class ReroutedWarningsTest : public testing::Test {
 protected:
  // The buffer is process-wide, so drop whatever earlier tests left in it.
  void SetUp() override {
    WarningCapture discard;
    SurfaceReroutedWarnings();
  }
};

TEST_F(ReroutedWarningsTest, SurfacesWarningFromReroutingThread) {
  WarningCapture capture;

  RunOnReroutingThread([] { TORCH_WARN("__warning__"); });
  EXPECT_THAT(capture.warnings(), IsEmpty());

  SurfaceReroutedWarnings();
  EXPECT_THAT(capture.messages(), ElementsAre("__warning__"));
}

TEST_F(ReroutedWarningsTest, SurfacingEmptiesBuffer) {
  WarningCapture capture;
  RunOnReroutingThread([] { TORCH_WARN("__warning__"); });

  SurfaceReroutedWarnings();
  SurfaceReroutedWarnings();
  EXPECT_THAT(capture.messages(), ElementsAre("__warning__"));
}

TEST_F(ReroutedWarningsTest, SurfacingWithoutWarningsDoesNothing) {
  WarningCapture capture;

  SurfaceReroutedWarnings();
  EXPECT_THAT(capture.warnings(), IsEmpty());
}

TEST_F(ReroutedWarningsTest, DoesNotAffectOtherThreads) {
  WarningCapture capture;
  RunOnReroutingThread([] {});

  TORCH_WARN("__warning__");
  EXPECT_THAT(capture.messages(), ElementsAre("__warning__"));

  SurfaceReroutedWarnings();
  EXPECT_THAT(capture.messages(), ElementsAre("__warning__"));
}

TEST_F(ReroutedWarningsTest, SurfacesInReroutedOrder) {
  WarningCapture capture;
  RunOnReroutingThread([] {
    TORCH_WARN("first");
    TORCH_WARN("second");
  });
  RunOnReroutingThread([] { TORCH_WARN("third"); });

  SurfaceReroutedWarnings();
  EXPECT_THAT(capture.messages(), ElementsAre("first", "second", "third"));
}

TEST_F(ReroutedWarningsTest, PreservesWarningDetails) {
  WarningCapture capture;
  RunOnReroutingThread([] { TORCH_WARN_DEPRECATION("__deprecated__"); });

  SurfaceReroutedWarnings();
  ASSERT_THAT(capture.warnings(), SizeIs(1));

  const c10::Warning& warning = capture.warnings()[0];
  EXPECT_TRUE(std::holds_alternative<c10::DeprecationWarning>(warning.type()));
  EXPECT_EQ(warning.msg(), "__deprecated__");
  EXPECT_FALSE(warning.verbatim());
  EXPECT_THAT(warning.source_location().file,
              HasSubstr("reroute_warnings_test.cc"));
}

TEST(ReroutedWarningsDeathTest, CrashesOnSurfacingInReroutingThread) {
  EXPECT_DEATH(RunOnReroutingThread([] { SurfaceReroutedWarnings(); }),
               "can only be surfaced in dispatch threads");
}

constexpr int kNumThreads = 8;
constexpr int kWarningsPerThread = 16;

// Starts `kNumThreads` threads that reroute their warnings and each raise
// `kWarningsPerThread` warnings. Decrements `num_threads` as each finishes.
std::vector<std::thread> StartWarningThreads(std::atomic<int>& num_threads) {
  num_threads.store(kNumThreads);
  std::vector<std::thread> threads;
  threads.reserve(kNumThreads);

  for (int t = 0; t < kNumThreads; ++t) {
    threads.emplace_back([t, &num_threads] {
      RouteWarningsOnThisThreadToDispatchThread();
      for (int i = 0; i < kWarningsPerThread; ++i) {
        TORCH_WARN("[WARNING] ", i, " in thread ", t);
      }
      num_threads.fetch_sub(1);
    });
  }

  return threads;
}

// Expects `messages` to hold exactly the warnings from `StartWarningThreads`,
// with each thread's warnings in the order it raised them.
void ExpectAllWarningsInPerThreadOrder(
    const std::vector<std::string>& messages) {
  EXPECT_THAT(messages, SizeIs(kNumThreads * kWarningsPerThread));

  for (int t = 0; t < kNumThreads; ++t) {
    const std::string suffix = absl::StrCat(" in thread ", t);
    std::vector<std::string> actual;
    for (const std::string& message : messages) {
      if (absl::EndsWith(message, suffix)) actual.push_back(message);
    }
    std::vector<std::string> expected;
    for (int i = 0; i < kWarningsPerThread; ++i) {
      expected.push_back(absl::StrCat("[WARNING] ", i, suffix));
    }
    EXPECT_THAT(actual, ElementsAreArray(expected)) << "thread " << t;
  }
}

TEST_F(ReroutedWarningsTest, SurfacesWarningsFromConcurrentThreads) {
  WarningCapture capture;
  std::atomic<int> num_threads;
  for (std::thread& thread : StartWarningThreads(num_threads)) {
    thread.join();
  }

  SurfaceReroutedWarnings();
  ExpectAllWarningsInPerThreadOrder(capture.messages());
}

TEST_F(ReroutedWarningsTest, SurfacesWhileThreadsAreRerouting) {
  WarningCapture capture;
  std::atomic<int> num_threads;
  std::vector<std::thread> threads = StartWarningThreads(num_threads);
  while (num_threads.load() > 0) {
    SurfaceReroutedWarnings();
  }
  for (std::thread& thread : threads) {
    thread.join();
  }

  SurfaceReroutedWarnings();
  ExpectAllWarningsInPerThreadOrder(capture.messages());
}

}  // namespace
}  // namespace torch_tpu
