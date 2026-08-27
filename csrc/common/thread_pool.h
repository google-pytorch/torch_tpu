/*
 * Copyright 2025 Google LLC
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

#ifndef TORCH_TPU_THREAD_POOL_H_
#define TORCH_TPU_THREAD_POOL_H_

#include <string>

#include "xla/tsl/platform/threadpool.h"

namespace torch_tpu {

class ThreadPool {
 public:
  // Constructs a pool of `num_threads` threads with the given name. The thread
  // name will be shown as "tf_<name>" in logs. `low_latency_hint` is a hint to
  // the thread pool implementation that lower latency is preferred at the cost
  // of higher CPU usage, e.g. by letting one or more idle threads spin wait.
  ThreadPool(const std::string& name, int num_threads,
             bool low_latency_hint = true);

  // This class is neither copyable nor movable.
  ThreadPool(const ThreadPool&) = delete;
  ThreadPool& operator=(const ThreadPool&) = delete;
  ThreadPool(ThreadPool&&) = delete;
  ThreadPool& operator=(ThreadPool&&) = delete;

  // Schedules f() for execution in the pool of threads. Function object f
  // doesn't need to be copyable.
  template <typename F>
  void Schedule(F&& f) {
    // Wrap f into a shared_ptr to get around that it may not be copyable.
    auto task = std::make_shared<F>(std::forward<F>(f));
    pool_.Schedule([task = std::move(task)]() { (*task)(); });
  }

  int NumThreads() const { return pool_.NumThreads(); }

 private:
  tsl::thread::ThreadPool pool_;
};

}  // namespace torch_tpu

#endif  // TORCH_TPU_THREAD_POOL_H_
