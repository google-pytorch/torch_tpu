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

#ifndef TORCH_TPU_CSRC_COMMON_CONSTEXPR_MAP_H_
#define TORCH_TPU_CSRC_COMMON_CONSTEXPR_MAP_H_

#include <algorithm>
#include <array>
#include <cstddef>
#include <functional>
#include <stdexcept>
#include <type_traits>
#include <utility>

#include "csrc/common/utils.h"

namespace torch_tpu {

// A C++20-compliant flat map container with fixed size N that can be evaluated
// and queried at compile time (as a constexpr).
//
// Keys are sorted at construction time so that lookups (find, contains, at)
// execute in O(log N) comparisons using binary search on a contiguous
// std::array.
//
// This class relies purely on the standard C++20 library and has no
// dependencies on Google-internal code or Abseil.
template <typename Key, typename Value, std::size_t kSize,
          typename Compare = std::less<Key>>
class ConstexprMap {
 private:
  struct Entry {
    Key key;
    Value value;

    friend constexpr bool operator==(const Entry& lhs,
                                     const Entry& rhs) = default;
  };

 public:
  using key_type = Key;
  using value_type = Entry;
  using mapped_type = Value;
  using size_type = std::size_t;
  using difference_type = std::ptrdiff_t;
  using key_compare = Compare;
  using const_iterator = typename std::array<Entry, kSize>::const_iterator;
  using iterator = const_iterator;

  // Constructs an empty map.
  constexpr ConstexprMap()
    requires(kSize == 0)
  = default;

  // Constructor from C-style array of entries.
  constexpr explicit ConstexprMap(const value_type (&data)[kSize],
                                  Compare comp = Compare())
      : data_(std::to_array(data)), comp_(comp) {
    SortAndValidate();
  }

  // Iterators
  [[nodiscard]] constexpr const_iterator begin() const noexcept {
    return data_.begin();
  }
  [[nodiscard]] constexpr const_iterator cbegin() const noexcept {
    return data_.cbegin();
  }

  [[nodiscard]] constexpr const_iterator end() const noexcept {
    return data_.end();
  }
  [[nodiscard]] constexpr const_iterator cend() const noexcept {
    return data_.cend();
  }

  // Capacity
  [[nodiscard]] static constexpr bool empty() noexcept { return kSize == 0; }
  [[nodiscard]] static constexpr size_type size() noexcept { return kSize; }

  // Element access
  template <typename K>
  constexpr const Value& at(const K& key) const {
    const auto it = find(key);
    if (it == end()) {
      throw std::out_of_range(  // throw needed to mimic std::map.
          "ConstexprMap::at: key not found");
    }
    return it->value;
  }

  template <typename K>
  constexpr const Value& operator[](const K& key) const {
    return at(key);
  }

  // Lookup operation.
  template <typename K>
  [[nodiscard]] constexpr const_iterator find(const K& key) const {
    const auto it = lower_bound(key);
    if (it != end() && KeysEquivalent(it->key, key)) {
      return it;
    }
    return end();
  }

  // Returns true if the map contains the given key.
  template <typename K>
  [[nodiscard]] constexpr bool contains(const K& key) const {
    return find(key) != end();
  }

 private:
  // Factory function.
  template <typename K, typename V, std::size_t M, typename C>
  friend constexpr ConstexprMap<K, V, M, C> MakeConstexprMap(
      const typename ConstexprMap<K, V, M, C>::value_type (&entries)[M],
      C comp);

  // Returns an iterator to the first element in the map whose key is not less
  // than the given key.
  template <typename K>
  [[nodiscard]] constexpr const_iterator lower_bound(const K& key) const {
    if constexpr (kSize == 0) {
      return data_.end();
    } else {
      return std::lower_bound(data_.begin(), data_.end(), key,
                              [this](const Entry& item, const K& k) {
                                return KeyLess(item.key, k);
                              });
    }
  }

  // Returns true if k1 is less than k2 according to the key comparator.
  template <typename K1, typename K2>
  constexpr bool KeyLess(const K1& k1, const K2& k2) const {
    if constexpr (std::is_invocable_r_v<bool, const Compare&, const K1&,
                                        const K2&>) {
      return comp_(k1, k2);
    } else if constexpr (std::is_convertible_v<const K1&, const Key&> &&
                         std::is_convertible_v<const K2&, const Key&>) {
      return comp_(static_cast<const Key&>(k1), static_cast<const Key&>(k2));
    } else {
      static_assert(always_false_v<std::pair<K1, K2>>,  // STD_PAIR_OK=generic
                    "Key types not supported by ConstexprMap");
    }
  }

  // Returns true if k1 and k2 are equivalent according to the key comparator.
  template <typename K1, typename K2>
  constexpr bool KeysEquivalent(const K1& k1, const K2& k2) const {
    return !KeyLess(k1, k2) && !KeyLess(k2, k1);
  }

  // Sorts the underlying data and checks for duplicate keys.
  //
  // This function throws if duplicate keys are found, which will cause the
  // program to terminate. This is to mimic the behavior of std::map.
  constexpr void SortAndValidate() {
    if constexpr (kSize > 1) {
      std::sort(data_.begin(), data_.end(),
                [this](const Entry& a, const Entry& b) {
                  return KeyLess(a.key, b.key);
                });
      for (std::size_t i = 1; i < kSize; ++i) {
        if (KeysEquivalent(data_[i - 1].key, data_[i].key)) {
          throw std::invalid_argument(  // throw needed to mimic std::map.
              "Duplicate key in ConstexprMap");
        }
      }
    }
  }

  std::array<Entry, kSize> data_{};
  [[no_unique_address]] Compare comp_{};
};

// Factory function.
template <typename Key, typename Value, std::size_t kSize,
          typename Compare = std::less<Key>>
[[nodiscard]] constexpr ConstexprMap<Key, Value, kSize, Compare>
MakeConstexprMap(
    const typename ConstexprMap<Key, Value, kSize, Compare>::value_type (
        &entries)[kSize],
    Compare comp = Compare()) {
  return ConstexprMap<Key, Value, kSize, Compare>(entries, comp);
}

}  // namespace torch_tpu

#endif  // TORCH_TPU_CSRC_COMMON_CONSTEXPR_MAP_H_
