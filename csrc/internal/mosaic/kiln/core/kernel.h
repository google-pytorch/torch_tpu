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
#include <type_traits>
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
// `ScalarPrefetch`). Scalar-prefetch operands are always in SMEM; the memory
// space of the others depends on their spec (see `Ref::As`).
SmemRef ScalarPrefetch(int i);
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
// below, which unpacks each spec into a typed ref parameter on the body
// callback.
//
// `body` is invoked synchronously before `BuildKernel` returns and is not
// retained afterwards, so capturing local variables by reference is safe.
absl::StatusOr<Kernel> BuildKernel(const KernelSpec& spec,
                                   absl::FunctionRef<void()> body);

namespace internal {

inline auto RegisterSpec(KernelSpec& k, const ScalarPrefetchSpec& s) {
  int idx = k.scalar_prefetch.size();
  k.scalar_prefetch.push_back(s);
  return [idx] { return ScalarPrefetch(idx); };
}
template <MemorySpace kSpace>
auto RegisterSpec(KernelSpec& k,
                  const BasicBlockSpec<BlockRole::kIn, kSpace>& s) {
  int idx = k.in_specs.size();
  k.in_specs.push_back({s.shape, s.map, kSpace});
  return [idx] { return In(idx).As<TypedRef<kSpace>>(); };
}
template <MemorySpace kSpace>
auto RegisterSpec(KernelSpec& k,
                  const BasicBlockSpec<BlockRole::kOut, kSpace>& s) {
  int idx = k.out_specs.size();
  k.out_specs.push_back({s.shape, s.map, kSpace});
  return [idx] { return Out(idx).As<TypedRef<kSpace>>(); };
}
template <MemorySpace kSpace>
auto RegisterSpec(KernelSpec& k, const BasicScratchSpec<kSpace>& s) {
  int idx = k.scratch.size();
  k.scratch.push_back({s.shape, kSpace});
  return [idx] { return Scratch(idx).As<TypedRef<kSpace>>(); };
}
template <typename Spec>
auto RegisterSpec(KernelSpec& k, const std::optional<Spec>& s) {
  using Resolver = decltype(RegisterSpec(k, *s));
  using Result = std::invoke_result_t<const Resolver&>;
  std::optional<Resolver> inner;
  if (s.has_value()) {
    inner.emplace(RegisterSpec(k, *s));
  }
  return [inner = std::move(inner)]() -> std::optional<Result> {
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

  return BuildKernel(spec, [&] {  // REF_CAPTURE_OK=called synchronously
    body(std::get<Is>(resolvers)()...);
  });
}

}  // namespace internal

// Builds a Mosaic kernel from a `GridSpec`, a variadic list of operand specs
// (`ScalarPrefetchSpec`, `InSpec`, `OutSpec`, `ScratchSpec`, or
// `std::optional<...>` of any of those), and a trailing body callback whose
// parameters receive the corresponding refs, typed by the memory space of their
// spec (`SmemRef` for scalar prefetch, `VmemRef` for default `InSpec` /
// `OutSpec` / `ScratchSpec`, `HbmRef` for `InSpec::Hbm(...)`, ...), or
// `std::optional` of those:
//
//   CreateKernel(grid_spec, in0, in1, opt_mask, out0, scratch0,
//                [](VmemRef a, VmemRef b, std::optional<VmemRef> mask,
//                   VmemRef c, VmemRef acc) {
//                  ...
//                });
//
// The body callback is invoked synchronously before `CreateKernel` returns and
// is not retained afterwards, so capturing local variables by reference is
// safe.
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
