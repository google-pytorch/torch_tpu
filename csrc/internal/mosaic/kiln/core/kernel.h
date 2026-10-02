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

#ifndef TORCH_TPU_CSRC_INTERNAL_MOSAIC_KILN_CORE_KERNEL_H_
#define TORCH_TPU_CSRC_INTERNAL_MOSAIC_KILN_CORE_KERNEL_H_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <utility>

#include "absl/functional/function_ref.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "csrc/internal/mosaic/kiln/core/ref.h"
#include "csrc/internal/mosaic/kiln/core/scalar.h"
#include "csrc/internal/mosaic/kiln/core/specs.h"

namespace torch_tpu::kiln {

// -----------------------------------------------------------------------------
// Grid queries (valid inside `CreateKernel` and `IndexMap::IndexFn`)
// -----------------------------------------------------------------------------

// Current iteration index along `axis` (Pallas `pl.program_id(axis)`).
Scalar ProgramId(int axis);
// Grid bound along `axis` (Pallas `pl.num_programs(axis)`).
int64_t NumPrograms(int axis);
int GridRank();

// Access to kernel ref arguments by position (used by `CreateKernel` to unpack
// arguments into the user callback, and usable inside `IndexMap::IndexFn` for
// `ScalarPrefetch`).
Ref ScalarPrefetch(int i);
Ref In(int i);
Ref Out(int i);
Ref Scratch(int i);

// An opaque handle to a lowered Kiln kernel module.
class Kernel {
 public:
  Kernel(Kernel&&) noexcept;
  Kernel& operator=(Kernel&&) noexcept;
  ~Kernel();

  // Runs Kiln canonicalization passes on the kernel module.
  absl::Status Canonicalize();

  // Pretty-prints the kernel IR (for logging / tests).
  std::string ToString() const;

  // Serializes the kernel for Mosaic custom call execution.
  // This is a destructive operation.
  // Note: This runs `Canonicalize` before serializing.
  absl::StatusOr<std::string> Serialize() &&;

  // Verifies the kernel's internal consistency.
  absl::Status Verify() const;

 private:
  struct Impl;
  explicit Kernel(std::unique_ptr<Impl> impl);
  std::unique_ptr<Impl> impl_;

  friend absl::StatusOr<Kernel> BuildKernel(const KernelSpec& spec,
                                            absl::FunctionRef<void()> body);
};

// Low-level entry point taking an assembled `KernelSpec`. Prefer `CreateKernel`
// below, which unpacks each spec into a `Ref` parameter on the body callback.
absl::StatusOr<Kernel> BuildKernel(const KernelSpec& spec,
                                   absl::FunctionRef<void()> body);

namespace internal {

inline auto RegisterSpec(KernelSpec& k, const ScalarPrefetchSpec& s) {
  int idx = k.scalar_prefetch.size();
  k.scalar_prefetch.push_back(s);
  return [idx] { return ScalarPrefetch(idx); };
}
inline auto RegisterSpec(KernelSpec& k, const InSpec& s) {
  int idx = k.in_specs.size();
  k.in_specs.push_back({s.shape, s.dtype, s.map, s.space});
  return [idx] { return In(idx); };
}
inline auto RegisterSpec(KernelSpec& k, const OutSpec& s) {
  int idx = k.out_specs.size();
  k.out_specs.push_back({s.shape, s.dtype, s.map, s.space});
  return [idx] { return Out(idx); };
}
inline auto RegisterSpec(KernelSpec& k, const ScratchSpec& s) {
  int idx = k.scratch.size();
  k.scratch.push_back(s);
  return [idx] { return Scratch(idx); };
}
template <typename Spec>
auto RegisterSpec(KernelSpec& k, const std::optional<Spec>& s) {
  using Resolver = decltype(RegisterSpec(k, *s));
  std::optional<Resolver> inner;
  if (s.has_value()) {
    inner.emplace(RegisterSpec(k, *s));
  }
  return [inner = std::move(inner)]() -> std::optional<Ref> {
    if (!inner.has_value()) return std::nullopt;
    return (*inner)();
  };
}

template <typename... SpecsAndBody, size_t... Is>
absl::StatusOr<Kernel> CreateKernelImpl(GridSpec grid,
                                        std::index_sequence<Is...>,
                                        SpecsAndBody&&... specs_and_body) {
  auto all_args =
      std::forward_as_tuple(std::forward<SpecsAndBody>(specs_and_body)...);
  auto& body = std::get<sizeof...(Is)>(all_args);

  KernelSpec spec{.grid = std::move(grid)};
  // Order of evaluation in a braced-init-list is left-to-right, so specs are
  // registered in the exact order they were passed.
  std::tuple resolvers{RegisterSpec(spec, std::get<Is>(all_args))...};

  return BuildKernel(spec, [&] { body(std::get<Is>(resolvers)()...); });
}

}  // namespace internal

// Builds a Mosaic kernel from a `GridSpec`, a variadic list of operand specs
// (`ScalarPrefetchSpec`, `InSpec`, `OutSpec`, `ScratchSpec`, or
// `std::optional<...>` of any of those), and a trailing body callback whose
// parameters receive the corresponding `Ref` (or `std::optional<Ref>`) values:
//
//   CreateKernel(grid_spec, in0, in1, opt_mask, out0, scratch0,
//                [&](Ref a, Ref b, std::optional<Ref> mask, Ref c, Ref acc) {
//                  ...
//                });
template <typename... SpecsAndBody>
  requires(sizeof...(SpecsAndBody) >= 1)
absl::StatusOr<Kernel> CreateKernel(GridSpec grid,
                                    SpecsAndBody&&... specs_and_body) {
  return internal::CreateKernelImpl(
      std::move(grid), std::make_index_sequence<sizeof...(SpecsAndBody) - 1>{},
      std::forward<SpecsAndBody>(specs_and_body)...);
}

// Pretty-prints a kernel (for logging / tests).
inline std::string ToString(const Kernel& kernel) { return kernel.ToString(); }

}  // namespace torch_tpu::kiln

#endif  // TORCH_TPU_CSRC_INTERNAL_MOSAIC_KILN_CORE_KERNEL_H_
