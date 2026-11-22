# Draft issue (not posted): unsandboxed builds of main fail on torch's bundled protobuf headers

**Title:** `--config=no_rbe` builds of main fail to compile protobuf-generated code: torch's bundled `google/protobuf` headers shadow `@com_google_protobuf`

## Summary

Since the rules_python 2.4.x upgrade, any unsandboxed build of main fails while compiling the first target that includes protobuf-generated code, for example `//csrc/ops:xla_utils` (through `xla/xla_data.pb.h`):

```
external/rules_python++pip+torch_tpu_pypi_314_torch_cp314_cp314_manylinux_2_28_x86_64_f8481ea9/site-packages/torch/include/google/protobuf/stubs/port.h:137:32: error: use of undeclared identifier 'PROTOBUF_LONGLONG'
external/.../torch/include/google/protobuf/io/coded_stream.h:674:3: error: unknown type name 'PROTOBUF_NODISCARD'
external/.../torch/include/google/protobuf/io/coded_stream.h:819:21: error: unknown type name 'uint8_t'
```

The same error appears in the repository's own "Build and Test Wheel against Local PyTorch" workflow (every scheduled run on 2026-11-05, e.g. run 42065483896 on 2c17e0797), there through `+torch_local_repo+local_torch/.../google/protobuf/stubs/stringpiece.h:188`.

## Cause

Every torch wheel ships protobuf 3.x headers under `torch/include/google/protobuf`. `define_extra_torch_targets()` in `bazel/pip_parse_targets.bzl` (and `define_torch_local()` in `bazel/torch_local_defs.bzl`) excludes those files from the declared `hdrs` "so they do not shadow @com_google_protobuf in sandboxed/remote builds", but exposes the wheel's include directories with `includes = [...]`. The compiler therefore searches the real `site-packages/torch/include` directory. Excluding undeclared files only helps when the compiler cannot see them, i.e. in a sandbox or on RBE. `--config=no_rbe` pulls in `--config=local`, which sets `--spawn_strategy=standalone`, so the exclusion has no effect and `#include "google/protobuf/io/coded_stream.h"` from protobuf 33.6's generated code resolves to torch's copy.

## Reproduce

On a Linux host with no RBE credentials, at main (a43dc0aba or later):

```
bazel build --config=no_rbe //csrc/ops:xla_utils
```

The same command at 988b22f12 (rules_python 1.8.5 with the repo's patches) succeeds.

## Two possible fixes (branches ready, tested on a v6e-1: TBD)

1. `.bazelrc`: make `no_rbe` disable only remote execution and caching, keeping compile actions sandboxed, with `--test_strategy=standalone` and `--strategy=TestRunner=standalone,local` for the TPU runtime. `--config=local` stays as the explicit unsandboxed debugging config. Smallest change; fixes every `no_rbe` user but not a bare `--config=local` build.
2. `bazel/pip_parse_targets.bzl`: expose each include root through its own `cc_library` with `strip_include_prefix`, so the compiler searches Bazel's virtual include tree, which contains only the declared headers, in every strategy. Robust, but the file carries a note that `includes` was chosen to work around a Bazel issue with `strip_include_prefix` and headers shared between targets, so it needs a layering-check build to confirm.

Happy to send either as a PR; which do you prefer?
