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

#ifndef TORCH_TPU_CSRC_COMMON_INTERNED_DEVICE_SHAPE_H_
#define TORCH_TPU_CSRC_COMMON_INTERNED_DEVICE_SHAPE_H_

#include <cstddef>

#include "absl/base/nullability.h"
#include "absl/types/span.h"
#include "xla/shape.h"

namespace torch_tpu {

// View over an immutable list of process-lifetime static on-device `xla::Shape`
// pointers (including layout), with `nullptr` for dynamic, non-array, or
// unknown entries.
//
// Only constructible as empty or via `InternStaticDeviceShapes()`, guaranteeing
// that both the backing pointer array and every non-null pointee remain valid
// for the lifetime of the process without extending executable lifetime or
// reference-counting per buffer.
class InternedDeviceShapeSpan {
 public:
  using SpanType = absl::Span<const xla::Shape* absl_nullable const>;
  using value_type = SpanType::value_type;
  using const_iterator = SpanType::const_iterator;

  constexpr InternedDeviceShapeSpan() = default;

  [[nodiscard]] constexpr bool empty() const { return span_.empty(); }
  [[nodiscard]] constexpr size_t size() const { return span_.size(); }

  constexpr const xla::Shape* absl_nullable operator[](size_t i) const {
    return span_[i];
  }

  [[nodiscard]] constexpr InternedDeviceShapeSpan subspan(
      size_t pos, size_t len = SpanType::npos) const {
    return InternedDeviceShapeSpan(span_.subspan(pos, len));
  }

  constexpr const_iterator begin() const { return span_.begin(); }
  constexpr const_iterator end() const { return span_.end(); }

  [[nodiscard]] constexpr SpanType as_span() const { return span_; }

  friend constexpr bool operator==(InternedDeviceShapeSpan lhs,
                                   InternedDeviceShapeSpan rhs) {
    return lhs.span_ == rhs.span_;
  }
  friend constexpr bool operator!=(InternedDeviceShapeSpan lhs,
                                   InternedDeviceShapeSpan rhs) {
    return lhs.span_ != rhs.span_;
  }

 private:
  friend InternedDeviceShapeSpan InternStaticDeviceShapes(
      absl::Span<const xla::Shape* absl_nullable const> shapes_flat);

  constexpr explicit InternedDeviceShapeSpan(SpanType span) : span_(span) {}

  SpanType span_;
};

static_assert(sizeof(InternedDeviceShapeSpan) ==
              sizeof(InternedDeviceShapeSpan::SpanType));

// Returns process-lifetime copies of `shapes_flat`; equal shapes share one
// pointer. Null, non-array, dynamic, or layout-less entries map to `nullptr`.
//
// Acquires a process-wide mutex and stores entries in a never-freed table
// bounded by distinct compiled signatures (not reclaimed by compilation cache
// eviction); intended for compile/load paths rather than per-tensor runtime
// allocations.
InternedDeviceShapeSpan InternStaticDeviceShapes(
    absl::Span<const xla::Shape* absl_nullable const> shapes_flat);

}  // namespace torch_tpu

#endif  // TORCH_TPU_CSRC_COMMON_INTERNED_DEVICE_SHAPE_H_
