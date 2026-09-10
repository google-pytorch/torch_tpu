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

// Dummy C++ file for testing linter.

namespace torch_tpu {

class TorchTpuClass {};

void Foo() { torch_tpu::TorchTpuClass(); }

}  // namespace torch_tpu

void Bar() { torch_tpu::TorchTpuClass(); }

namespace torch_tpu {

void Foo2() { ::torch_tpu::TorchTpuClass(); }

}  // namespace torch_tpu

namespace outer {
namespace torch_tpu {
void Foo3() { ::torch_tpu::TorchTpuClass(); }
}  // namespace torch_tpu
}  // namespace outer

namespace {
void Foo4() { torch_tpu::TorchTpuClass(); }
}  // namespace

namespace foo::bar {
void Foo5() { torch_tpu::TorchTpuClass(); }
}  // namespace foo::bar

namespace torch_tpu::bar {
void Foo5() { torch_tpu::TorchTpuClass(); }
}  // namespace torch_tpu::bar

namespace torch_tpu::internal {
namespace {
void Baz() { torch_tpu::TorchTpuClass(); }
}  // namespace
}  // namespace torch_tpu::internal

namespace torch_tpu {
namespace {
namespace internal {
void Baz2() { torch_tpu::TorchTpuClass(); }
}  // namespace internal
}  // namespace
}  // namespace torch_tpu

namespace torch_tpu {
namespace {
void Baz3() { torch_tpu::TorchTpuClass(); }
}  // namespace
}  // namespace torch_tpu
