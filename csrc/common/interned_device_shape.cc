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

#include "csrc/common/interned_device_shape.h"

#include <cstddef>
#include <utility>
#include <vector>

#include "absl/base/no_destructor.h"
#include "absl/base/nullability.h"
#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_set.h"
#include "absl/container/node_hash_set.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "xla/shape.h"

namespace torch_tpu {

InternedDeviceShapeSpan InternStaticDeviceShapes(
    const absl::Span<const xla::Shape* absl_nullable const> shapes_flat) {
  if (shapes_flat.empty()) {
    return {};
  }
  struct DeviceShapeInternTable {
    absl::Mutex mu;
    // `shapes` requires element pointer stability; `shape_lists` only exposes
    // each `std::vector`'s heap buffer (`data()`), which survives move on
    // rehash.
    absl::node_hash_set<xla::Shape> shapes ABSL_GUARDED_BY(mu);
    absl::flat_hash_set<std::vector<const xla::Shape* absl_nullable>>
        shape_lists ABSL_GUARDED_BY(mu);
  };
  static absl::NoDestructor<DeviceShapeInternTable> table;
  std::vector<const xla::Shape* absl_nullable> ptrs(shapes_flat.size(),
                                                    nullptr);
  absl::MutexLock lock(table->mu);
  for (size_t i = 0; i < shapes_flat.size(); ++i) {
    const xla::Shape* const s = shapes_flat[i];
    if (s != nullptr && s->IsArray() && s->is_static() && s->has_layout()) {
      ptrs[i] = &*table->shapes.insert(*s).first;
    }
  }
  const std::vector<const xla::Shape* absl_nullable>& interned_list =
      *table->shape_lists.insert(std::move(ptrs)).first;
  return InternedDeviceShapeSpan(absl::MakeConstSpan(interned_list));
}

}  // namespace torch_tpu
