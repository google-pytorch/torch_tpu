# TorchTPU API Design Guidelines

## Introduction

While TorchTPU aims at a great user experience out of the box, power users need
sophisticated control to get the exact behavior and performance they want. To
ensure a predictable and pleasant user experience, TorchTPU APIs need to follow
a common set of guidelines. This document presents such guidelines.

## Principles

This section contains the principles for defining APIs that affect TorchTPU's
behavior. APIs defined in tests, benchmarks, and examples are not governed by
these rules. While not all of our APIs follow these principles today, we shall
follow them in new code and migrate the existing APIs to conform to them.

*   APIs can be either **global** (affecting the entire process) or **local**
    (affecting a region of a particular thread).
*   General preferences
    *   Try not to invent new APIs; **prefer reusing an existing PyTorch API**
        if it can meet our needs well. This minimizes the user migration cost.
    *   Prefer
        [torch.accelerator](https://docs.pytorch.org/docs/2.13/accelerator.html)
        over `torch.tpu`. The former makes it easier for the users to switch
        between different accelerator types.
*   Design choices
    *   For **global** APIs:
        *   Global knobs should always have a Python API, e.g.
            `torch.tpu.allow_excess_precision = True` and
            `torch.set_float32_matmul_precision("highest")`.
            *   Prefer properties to getters/setters.
            *   A user can change the state of a global knob multiple times. In
                such cases, the latest change takes effect.
        *   Optionally, we permit configuring the default value of a global knob
            via an **environment variable**. Add this if we anticipate a need to
            tweak the knob without touching the model code (e.g. a knob used by
            the production operation team).
            *   If there's a
                [well-established PyTorch environment variable](https://docs.pytorch.org/docs/2.13/torch_environment_variables.html)
                for controlling the behavior (e.g. `TORCH_TRACE` or
                `TORCH_SHOW_CPP_STACKTRACES`), we may reuse the same variable
                for TorchTPU to minimize the user migration cost.
            *   An environment variable introduced by TorchTPU should by default
                start with `TORCH_TPU_INTERNAL_`. If we want it to be usable by
                users, it should start with `TORCH_TPU_` without the `INTERNAL_`
                part. If the internal API becomes stable and we are comfortable
                letting users use it, it should be renamed to strip the
                `INTERNAL_` part at that time.
            *   When the environment variable is set and the Python API is
                called, the *latter* takes precedence.
        *   Use **C++ command-line flags** only for Google-internal knobs (e.g.
            it's for Google internal experiments and shouldn't be exposed to
            even power users), as there's no easy way to pass flags to TorchTPU
            in OSS.
            *   All TorchTPU flag names should start with `torch_tpu_internal_`.
        *   Do **NOT** use **Python command-line flags** (`flags.DEFINE_*`).
            They cause client programs that don't parse flags to crash with a
            "reading flag before flag parsing" error.
    *   Rules for **local** APIs:
        *   Local knobs should be implemented as Python **context managers**.
            Each context manager region governs the behavior of all enclosed
            PyTorch code in the current Python thread.
        *   A local knob overrides the corresponding global knob (if any), not
            the other way around. For example, the MLIR tracebacks can be
            controlled via either a global knob (the
            `--torch_tpu_internal_mlir_tracebacks` *internal* flag) or a local
            knob (the `enable_tracebacks` context manager); when both are
            present, the latter takes precedence.
*   Implementation choices
    *   For **global** APIs:
        *   Read a flag only once and memoize it. This ensures consistent
            behavior in case someone tampered with the flag at run time. Such
            read-only behavior aligns with how PyTorch handles knobs like
            `TORCH_SHOW_CPP_STACKTRACES`. To conform to this pattern, use the
            `ReadFlagOnce()` function to read a flag in C++.
        *   Similarly, read an environment variable only once and memoize it.
            Use `GetEnvOnce()` in C++ for this purpose.
        *   When a command-line flag is justified, implement it using the `absl`
            flag library (in both C++ and Python).
    *   For **local** APIs:
        *   It's important to follow TorchTPU Context Manager design pattern to
            implement a context manager (TODO: publish the context manager
            design guide). Otherwise its behavior may be incorrect.
*   Python API paths
    *   A **public path** is a dotted name that reaches an object and has no
        component starting with `_`, e.g.
        `torch.tpu.ops.align_jagged_to_multiple`. **Every public path to a
        TorchTPU object is an API**, even if it exists only because a public
        module imports the object. Adding, removing, or changing a public path
        needs approval from the API owners.
        *   Objects that other libraries own (e.g. `typing.Optional` or `torch`)
            aren't TorchTPU APIs, even when a TorchTPU module imports them.
    *   Each TorchTPU module, class, and function should have **at most one
        public module-level path** that isn't deprecated. Every extra path is
        another API that users may depend on, and each one must go through
        deprecation before we can move or remove the object.
        *   Class members don't count: a method that a subclass inherits is
            reachable through both classes, which is fine.
        *   When we move an API, the old path may stay as a deprecated alias
            during the deprecation period.
        *   This is stricter than PyTorch, where both `torch.nn.Linear` and
            `torch.nn.modules.linear.Linear` are public, and similar to how JAX
            and NumPy keep their implementations in `jax._src` and
            `numpy._core`.
    *   To keep the other paths internal, give them a component that starts with
        `_`. Prefer putting the implementation in a private module (e.g.
        `torch_tpu/_internal/ops/_jagged.py`) and having the public module
        import that module and assign only the public names (e.g.
        `align_jagged_to_multiple = _jagged.align_jagged_to_multiple`), so that
        the implementation's own imports don't become public paths.
    *   **Instance attributes of public classes must start with `_`.** An
        instance attribute such as `self.lr` is a public path too, but the API
        scanner can't discover it, so it would escape API review. To expose
        state, add a property: properties are tracked like other APIs, and one
        without a setter makes the state read-only. `tt_apis_test` enforces
        this. Dataclass fields are exempt.

## Python API Module Structure

While TorchTPU is developed out of the PyTorch source tree today, we want users
to think of it as logically part of PyTorch. Therefore, our public Python APIs
must be exposed under `torch.tpu.*` (as opposed to `torch_tpu`), the same way
CUDA's live under `torch.cuda`. One exception is that if a functionality can be
served via an existing `torch` or
[torch.accelerator](https://docs.pytorch.org/docs/2.13/accelerator.html) API, we
leverage that instead of exposing a new API.

Public TorchTPU Python APIs follow this module structure:

```
torch.tpu        # TorchTPU root module
 |- compiler     # compiler options and control
 |- distributed  # distributed utilities
 |- exceptions   # exceptions that might be raised by TorchTPU
 |- ops          # custom ops
 |- optimizers   # TorchTPU-optimized optimizers
 |- pallas       # utilities for using pallas kernels
 `- profiler     # utilities for profiling TorchTPU programs
```

Adding any new public module requires approval from the API owners.

*   **No public APIs in `torch_tpu` or `torch.backends.tpu`**: The `torch_tpu`
    package has no public APIs; its implementation lives under
    `torch_tpu._internal`. Likewise, we do not expose APIs under
    `torch.backends.tpu`; global knobs such as
    `torch.tpu.allow_excess_precision` live directly on `torch.tpu`.
*   **One canonical path per API**: As described under
    [Principles](#principles), each public TorchTPU API is exposed at only one
    canonical location. Allowing multiple public paths to the same API (for
    example, by accidentally re-exporting an imported symbol in another public
    module) is unnecessary, creates inconsistency, and makes it hard for
    TorchTPU to evolve.
*   **Custom ops in `torch.tpu.ops`**: All custom ops live in `torch.tpu.ops`,
    regardless of whether they are implemented in C++ or Python. Keeping all
    custom ops in one module gives users a consistent interface and lets us
    change an op's implementation between C++ and Python without changing its
    public path.

    *   PyTorch's dispatcher design requires all custom ops implemented in C++
        to be registered in a submodule of `torch.ops` (`torch.ops.tpu`). To
        work around this constraint and expose each C++ custom op `foo` only at
        `torch.tpu.ops.foo`:

        1.  Register the op in C++ with a leading underscore as
            `torch.ops.tpu._foo`, so that its path in `torch.ops.tpu` is
            internal.

        2.  In the `torch.tpu.ops` module, re-export `_foo` as `foo`:

            ```python
            foo = torch.ops.tpu._foo
            ```
