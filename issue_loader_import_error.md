## Summary

When the bundled glue `.so` does not match the installed PyTorch, `import torch_tpu` fails with a bare dynamic-loader error that does not mention either version:

```
ImportError: .../torch_tpu/_internal/env_2_14_0.so: undefined symbol: _ZN3c10...
```

`_versioned_so_loader` picks the newest glue whose version is `<=` the running torch, so this happens whenever torch is newer than every bundled glue, after a `pip install --no-deps`, with a source build against another torch, or after upgrading torch under an installed torch_tpu. I hit it with a from-source cp314 wheel on torch 2.15 (the repo lockfile pins 2.11 / 2.14) and spent a while on `undefined symbol c10::report_overflow` before realising it was a version mismatch.

## Proposed message

`ExtensionProxyLoader.create_module` / `exec_module` re-raise an `ImportError` from the real loader as:

```
ImportError: torch_tpu failed to load its extension module
torch_tpu._internal.env_2_14_0 (built for PyTorch 2.14.0) with PyTorch
2.15.0+cpu installed. torch_tpu wheels only support the PyTorch releases they
were built against; install a matching torch and torch_tpu pair, or build
torch_tpu from source for this PyTorch. Underlying error: ...
```

## Fix

Implemented with a unit test:

https://github.com/joinalahmed/torch_tpu/tree/loader-import-error-context
(diff: https://github.com/google-pytorch/torch_tpu/compare/main...joinalahmed:torch_tpu:loader-import-error-context)

I cannot open the pull request yet: the repository allows users without write access 3 open PRs and mine (#6102, #6105, #6106) fill that quota. I will open it as soon as a slot frees.
