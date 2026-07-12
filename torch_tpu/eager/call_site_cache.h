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

#ifndef TORCH_TPU_EAGER_CALL_SITE_CACHE_H_
#define TORCH_TPU_EAGER_CALL_SITE_CACHE_H_

#include <algorithm>
#include <future>  // NOLINT(build/c++11)
#include <iterator>
#include <memory>
#include <utility>
#include <vector>

#include "ATen/core/Tensor.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/inlined_vector.h"
#include "absl/types/span.h"
#include "torch_tpu/common/cache_key.h"
#include "torch_tpu/common/compilation.h"
#include "torch_tpu/common/dimension_types.h"
#include "torch_tpu/common/shape.h"
#include "torch_tpu/common/utils.h"

namespace torch_tpu {

using CallSiteOutputShapes = absl::InlinedVector<Shape, 2>;

struct CallSiteCacheKey {
  absl::InlinedVector<SmallInt64Vector, 2> input_shapes;
  absl::InlinedVector<c10::ScalarType, 2> input_dtypes;
  OpParamCacheKeys op_param_cache_keys;

  CallSiteCacheKey(absl::InlinedVector<SmallInt64Vector, 2> shapes,
                   absl::InlinedVector<c10::ScalarType, 2> dtypes,
                   OpParamCacheKeys keys)
      : input_shapes(std::move(shapes)),
        input_dtypes(std::move(dtypes)),
        op_param_cache_keys(std::move(keys)) {}

  CallSiteCacheKey(CallSiteCacheKey&&) = default;
  CallSiteCacheKey& operator=(CallSiteCacheKey&&) = default;
  CallSiteCacheKey(const CallSiteCacheKey& other)
      : input_shapes(other.input_shapes),
        input_dtypes(other.input_dtypes),
        op_param_cache_keys(other.op_param_cache_keys.Clone()) {}
  CallSiteCacheKey& operator=(const CallSiteCacheKey& other) {
    if (this != &other) {
      input_shapes = other.input_shapes;
      input_dtypes = other.input_dtypes;
      op_param_cache_keys = other.op_param_cache_keys.Clone();
    }
    return *this;
  }

  bool operator==(const CallSiteCacheKey& other) const {
    if (input_shapes != other.input_shapes ||
        input_dtypes != other.input_dtypes) {
      return false;
    }
    if (op_param_cache_keys.size() != other.op_param_cache_keys.size()) {
      return false;
    }
    return std::equal(op_param_cache_keys.begin(), op_param_cache_keys.end(),
                      other.op_param_cache_keys.begin());
  }

  template <typename H>
  friend H AbslHashValue(H h, const CallSiteCacheKey& key) {
    h = H::combine(std::move(h), key.input_shapes.size());
    for (const auto& shape : key.input_shapes) {
      h = H::combine(std::move(h), absl::MakeConstSpan(shape));
    }
    h = H::combine(std::move(h), absl::MakeConstSpan(key.input_dtypes));
    h = H::combine(std::move(h), key.op_param_cache_keys.size());
    for (const auto& param_key : key.op_param_cache_keys) {
      h = H::combine(std::move(h), param_key);
    }
    return h;
  }
};

struct CallSiteCacheEntry {
  std::weak_ptr<const LoadedExecutableWithMetadata> fixed_shape_kernel;
  CallSiteOutputShapes output_shapes;
};

class CallSiteCache {
 public:
  static constexpr size_t kMaxCacheSize = 64;

  CallSiteCache() = default;
  ~CallSiteCache() = default;

  // Disallow copy and move
  CallSiteCache(const CallSiteCache&) = delete;
  CallSiteCache& operator=(const CallSiteCache&) = delete;
  CallSiteCache(CallSiteCache&&) = delete;
  CallSiteCache& operator=(CallSiteCache&&) = delete;

  bool Lookup(absl::Span<const at::Tensor> inputs,
              const OpParamCacheKeys& param_keys, CompiledKernel& out_kernel,
              CallSiteOutputShapes& out_shapes) {
    absl::InlinedVector<SmallInt64Vector, 2> shapes;
    absl::InlinedVector<c10::ScalarType, 2> dtypes;
    shapes.reserve(inputs.size());
    dtypes.reserve(inputs.size());
    for (const auto& input : inputs) {
      shapes.push_back(CopyIntVector(input.sizes()));
      dtypes.push_back(input.scalar_type());
    }

    CallSiteCacheKey key(std::move(shapes), std::move(dtypes),
                         param_keys.Clone());
    return LookupHelper(key, out_kernel, out_shapes);
  }

  bool Lookup(const at::Tensor& input, CompiledKernel& out_kernel,
              CallSiteOutputShapes& out_shapes) {
    absl::InlinedVector<SmallInt64Vector, 2> shapes = {
        CopyIntVector(input.sizes())};
    absl::InlinedVector<c10::ScalarType, 2> dtypes = {input.scalar_type()};

    CallSiteCacheKey key(std::move(shapes), std::move(dtypes),
                         OpParamCacheKeys::Empty());
    return LookupHelper(key, out_kernel, out_shapes);
  }

  bool Lookup(const at::Tensor& input0, const at::Tensor& input1,
              CompiledKernel& out_kernel, CallSiteOutputShapes& out_shapes) {
    absl::InlinedVector<SmallInt64Vector, 2> shapes = {
        CopyIntVector(input0.sizes()), CopyIntVector(input1.sizes())};
    absl::InlinedVector<c10::ScalarType, 2> dtypes = {input0.scalar_type(),
                                                      input1.scalar_type()};

    CallSiteCacheKey key(std::move(shapes), std::move(dtypes),
                         OpParamCacheKeys::Empty());
    return LookupHelper(key, out_kernel, out_shapes);
  }

  void Update(absl::Span<const at::Tensor> inputs,
              const OpParamCacheKeys& param_keys, CompiledKernel kernel,
              CallSiteOutputShapes output_shapes) {
    absl::InlinedVector<SmallInt64Vector, 2> shapes;
    absl::InlinedVector<c10::ScalarType, 2> dtypes;
    shapes.reserve(inputs.size());
    dtypes.reserve(inputs.size());
    for (const auto& input : inputs) {
      shapes.push_back(CopyIntVector(input.sizes()));
      dtypes.push_back(input.scalar_type());
    }

    CallSiteCacheKey key(std::move(shapes), std::move(dtypes),
                         param_keys.Clone());
    UpdateHelper(std::move(key), std::move(kernel), std::move(output_shapes));
  }

  void Update(std::vector<SmallInt64Vector> shapes,
              std::vector<c10::ScalarType> dtypes,
              const OpParamCacheKeys& param_keys, CompiledKernel kernel,
              CallSiteOutputShapes output_shapes) {
    absl::InlinedVector<SmallInt64Vector, 2> inlined_shapes(
        std::make_move_iterator(shapes.begin()),
        std::make_move_iterator(shapes.end()));
    absl::InlinedVector<c10::ScalarType, 2> inlined_dtypes(
        std::make_move_iterator(dtypes.begin()),
        std::make_move_iterator(dtypes.end()));

    CallSiteCacheKey key(std::move(inlined_shapes), std::move(inlined_dtypes),
                         param_keys.Clone());
    UpdateHelper(std::move(key), std::move(kernel), std::move(output_shapes));
  }

 private:
  bool LookupHelper(const CallSiteCacheKey& key, CompiledKernel& out_kernel,
                    CallSiteOutputShapes& out_shapes) {
    if (ShouldBypass()) {
      return false;
    }
    auto it = entries_.find(key);
    if (it != entries_.end()) {
      auto shared_exec = it->second.fixed_shape_kernel.lock();
      if (shared_exec) {
        std::promise<absl::StatusOr<SharedLoadedExecutableWithMetadata>> p;
        p.set_value(shared_exec);
        out_kernel.fixed_shape_kernel = p.get_future().share();
        out_shapes = it->second.output_shapes;
        UpdateMissTracker(true);
        return true;
      }
      entries_.erase(it);
    }
    UpdateMissTracker(false);
    return false;
  }

  void UpdateHelper(CallSiteCacheKey key, CompiledKernel kernel,
                    CallSiteOutputShapes output_shapes) {
    if (entries_.size() >= kMaxCacheSize) {
      entries_.clear();
    }
    auto exec_or = kernel.fixed_shape_kernel.get();
    if (exec_or.ok()) {
      entries_.emplace(std::move(key),
                       CallSiteCacheEntry{
                           .fixed_shape_kernel = *exec_or,
                           .output_shapes = std::move(output_shapes),
                       });
    }
  }
  bool ShouldBypass() {
    if (bypass_counter_ > 0) {
      bypass_counter_--;
      return true;
    }
    return false;
  }

  void UpdateMissTracker(bool hit) {
    if (hit) {
      consecutive_misses_ = 0;
    } else {
      consecutive_misses_++;
      if (consecutive_misses_ > 32) {
        bypass_counter_ = 1000;
        consecutive_misses_ = 0;
      }
    }
  }

  absl::flat_hash_map<CallSiteCacheKey, CallSiteCacheEntry> entries_;
  size_t consecutive_misses_ = 0;
  size_t bypass_counter_ = 0;
};

}  // namespace torch_tpu

#endif  // TORCH_TPU_EAGER_CALL_SITE_CACHE_H_
