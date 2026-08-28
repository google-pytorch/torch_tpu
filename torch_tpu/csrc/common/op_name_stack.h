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

#ifndef TORCH_TPU_COMMON_OP_NAME_STACK_H_
#define TORCH_TPU_COMMON_OP_NAME_STACK_H_

#include <optional>
#include <stack>

#include "torch_tpu/ops/op_names.h"

namespace torch_tpu {
namespace internal {

// OpNameStack maintains a thread-local stack of OpName enums.
class OpNameStack {
 public:
  // Pushes an OpName onto the current thread's stack.
  static void Push(OpName op_name);

  // Pops the top OpName from the current thread's stack.
  static void Pop();

  // Returns the top OpName from the current thread's stack.
  // Crashes if the stack is empty.
  static OpName Top();

  // Returns the top OpName from the current thread's stack, or std::nullopt if
  // the stack is empty.
  static std::optional<OpName> MaybeTop();

 private:
  static thread_local  // CPP_THREAD_LOCAL_OK=needed only in the dispatching
                       // thread.
      std::stack<OpName>
          stack_;
};

// ScopedOpName is an RAII guard that pushes an OpName onto the thread-local
// stack on construction and pops it on destruction.
class ScopedOpName {
 public:
  explicit ScopedOpName(OpName op_name) { OpNameStack::Push(op_name); }
  ~ScopedOpName() { OpNameStack::Pop(); }

  ScopedOpName(const ScopedOpName&) = delete;
  ScopedOpName& operator=(const ScopedOpName&) = delete;
};

}  // namespace internal
}  // namespace torch_tpu

#endif  // TORCH_TPU_COMMON_OP_NAME_STACK_H_
