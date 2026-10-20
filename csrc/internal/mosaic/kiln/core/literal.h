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

#ifndef TORCH_TPU_CSRC_INTERNAL_MOSAIC_KILN_CORE_LITERAL_H_
#define TORCH_TPU_CSRC_INTERNAL_MOSAIC_KILN_CORE_LITERAL_H_

#include <concepts>
#include <cstdint>
#include <variant>

namespace torch_tpu::kiln {

// A numeric literal holding either `int64_t` or `double`. Implicitly
// convertible from all standard C++ integral and floating-point types so every
// overloaded operator and helper supports both ints and floats.
class Literal {
 public:
  template <typename T>
    requires std::integral<T>
  /* implicit */ Literal(T v)  // NOLINT(google-explicit-constructor)
      : value_(static_cast<int64_t>(v)) {}

  template <typename T>
    requires std::floating_point<T>
  /* implicit */ Literal(T v)  // NOLINT(google-explicit-constructor)
      : value_(static_cast<double>(v)) {}

  bool IsInt() const { return std::holds_alternative<int64_t>(value_); }
  bool IsDouble() const { return std::holds_alternative<double>(value_); }

  int64_t AsInt() const {
    return std::visit([](auto v) { return static_cast<int64_t>(v); }, value_);
  }
  double AsDouble() const {
    return std::visit([](auto v) { return static_cast<double>(v); }, value_);
  }

 private:
  std::variant<int64_t, double> value_;
};

// CRTP mixin providing (Derived, Other) and (Other, Derived) binary
// operator overloads. Derived must provide `Derived Coerce(Other) const`.
// TODO(willfroom): move this into internal namespace and separate target.
template <typename Derived, typename Other>
class WithOperandOps {
 public:
  // Arithmetic
  friend Derived operator+(Derived lhs, Other rhs) {
    return lhs + lhs.Coerce(rhs);
  }
  friend Derived operator+(Other lhs, Derived rhs) { return rhs + lhs; }
  friend Derived operator-(Derived lhs, Other rhs) {
    return lhs - lhs.Coerce(rhs);
  }
  friend Derived operator-(Other lhs, Derived rhs) {
    return rhs.Coerce(lhs) - rhs;
  }
  friend Derived operator*(Derived lhs, Other rhs) {
    return lhs * lhs.Coerce(rhs);
  }
  friend Derived operator*(Other lhs, Derived rhs) { return rhs * lhs; }
  friend Derived operator/(Derived lhs, Other rhs) {
    return lhs / lhs.Coerce(rhs);
  }
  friend Derived operator/(Other lhs, Derived rhs) {
    return rhs.Coerce(lhs) / rhs;
  }
  friend Derived operator%(Derived lhs, Other rhs) {
    return lhs % lhs.Coerce(rhs);
  }
  friend Derived operator%(Other lhs, Derived rhs) {
    return rhs.Coerce(lhs) % rhs;
  }

  friend Derived& operator+=(Derived& lhs, Other rhs) {
    return lhs = lhs + rhs;
  }
  friend Derived& operator-=(Derived& lhs, Other rhs) {
    return lhs = lhs - rhs;
  }
  friend Derived& operator*=(Derived& lhs, Other rhs) {
    return lhs = lhs * rhs;
  }
  friend Derived& operator/=(Derived& lhs, Other rhs) {
    return lhs = lhs / rhs;
  }
  friend Derived& operator%=(Derived& lhs, Other rhs) {
    return lhs = lhs % rhs;
  }

  // Bitwise / logical ops
  friend Derived operator&(Derived lhs, Other rhs) {
    return lhs & lhs.Coerce(rhs);
  }
  friend Derived operator&(Other lhs, Derived rhs) { return rhs & lhs; }
  friend Derived operator|(Derived lhs, Other rhs) {
    return lhs | lhs.Coerce(rhs);
  }
  friend Derived operator|(Other lhs, Derived rhs) { return rhs | lhs; }
  friend Derived operator^(Derived lhs, Other rhs) {
    return lhs ^ lhs.Coerce(rhs);
  }
  friend Derived operator^(Other lhs, Derived rhs) { return rhs ^ lhs; }
  friend Derived operator<<(Derived lhs, Other rhs) {
    return lhs << lhs.Coerce(rhs);
  }
  friend Derived operator<<(Other lhs, Derived rhs) {
    return rhs.Coerce(lhs) << rhs;
  }
  friend Derived operator>>(Derived lhs, Other rhs) {
    return lhs >> lhs.Coerce(rhs);
  }
  friend Derived operator>>(Other lhs, Derived rhs) {
    return rhs.Coerce(lhs) >> rhs;
  }

  friend Derived& operator&=(Derived& lhs, Other rhs) {
    return lhs = lhs & rhs;
  }
  friend Derived& operator|=(Derived& lhs, Other rhs) {
    return lhs = lhs | rhs;
  }
  friend Derived& operator^=(Derived& lhs, Other rhs) {
    return lhs = lhs ^ rhs;
  }
  friend Derived& operator<<=(Derived& lhs, Other rhs) {
    return lhs = lhs << rhs;
  }
  friend Derived& operator>>=(Derived& lhs, Other rhs) {
    return lhs = lhs >> rhs;
  }

  // Elementwise min / max / select
  friend Derived Min(Derived lhs, Other rhs) {
    return Min(lhs, lhs.Coerce(rhs));
  }
  friend Derived Min(Other lhs, Derived rhs) {
    return Min(rhs.Coerce(lhs), rhs);
  }
  friend Derived Max(Derived lhs, Other rhs) {
    return Max(lhs, lhs.Coerce(rhs));
  }
  friend Derived Max(Other lhs, Derived rhs) {
    return Max(rhs.Coerce(lhs), rhs);
  }
  friend Derived Select(Derived cond, Derived if_true, Other if_false) {
    return Select(cond, if_true, if_true.Coerce(if_false));
  }
  friend Derived Select(Derived cond, Other if_true, Derived if_false) {
    return Select(cond, if_false.Coerce(if_true), if_false);
  }

  // Comparisons
  friend Derived operator==(Derived lhs, Other rhs) {
    return lhs == lhs.Coerce(rhs);
  }
  friend Derived operator==(Other lhs, Derived rhs) { return rhs == lhs; }
  friend Derived operator!=(Derived lhs, Other rhs) {
    return lhs != lhs.Coerce(rhs);
  }
  friend Derived operator!=(Other lhs, Derived rhs) { return rhs != lhs; }
  friend Derived operator<(Derived lhs, Other rhs) {
    return lhs < lhs.Coerce(rhs);
  }
  friend Derived operator<(Other lhs, Derived rhs) { return rhs > lhs; }
  friend Derived operator<=(Derived lhs, Other rhs) {
    return lhs <= lhs.Coerce(rhs);
  }
  friend Derived operator<=(Other lhs, Derived rhs) { return rhs >= lhs; }
  friend Derived operator>(Derived lhs, Other rhs) {
    return lhs > lhs.Coerce(rhs);
  }
  friend Derived operator>(Other lhs, Derived rhs) { return rhs < lhs; }
  friend Derived operator>=(Derived lhs, Other rhs) {
    return lhs >= lhs.Coerce(rhs);
  }
  friend Derived operator>=(Other lhs, Derived rhs) { return rhs <= lhs; }
};

}  // namespace torch_tpu::kiln

#endif  // TORCH_TPU_CSRC_INTERNAL_MOSAIC_KILN_CORE_LITERAL_H_
