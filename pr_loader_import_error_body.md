## Description

torch_tpu wheels bundle one glue `.so` per supported PyTorch ABI
(`env_2_11_0.so`, ..., `env_2_16_0.so`) and `_versioned_so_loader` picks the
newest glue whose version is `<=` the running torch. When the pick does not
match (torch newer than every glue, a `--no-deps` install, a source build
against another torch, a torch upgraded after torch_tpu), the failure surfaces
from the dynamic loader as

```
ImportError: .../torch_tpu/_internal/env_2_14_0.so: undefined symbol: _ZN3c10...
```

which does not say that the glue was built for PyTorch 2.14 or that 2.15 is
installed. `ExtensionProxyLoader.create_module` and `exec_module` now re-raise
an `ImportError` from the real loader as

```
ImportError: torch_tpu failed to load its extension module
torch_tpu._internal.env_2_14_0 (built for PyTorch 2.14.0) with PyTorch
2.15.0+cpu installed. torch_tpu wheels only support the PyTorch releases they
were built against; install a matching torch and torch_tpu pair, or build
torch_tpu from source for this PyTorch. Underlying error: ...
```

chained with `from` to the original, keeping its `name` and `path`. Other
exception types pass through unchanged. `resolve_suffix` already gives a clear
message when the installed torch predates every glue; this covers the other
direction, where a glue is picked but does not load.

## Test Plan

- New `test_extension_proxy_loader_describes_failed_load` (`create_module`
  and `exec_module` variants) in `tests/_versioned_so_loader_test.py`: asserts
  the message names the module, the glue version, the installed version, quotes
  the underlying error, chains it, restores `spec.name`, and keeps `path`.
- `tests/_versioned_so_loader_test.py` run as an absltest script against the
  patched module on a Cloud TPU VM (Python 3.14, torch 2.11.0+cpu, wheel from
  `main`): OK (all tests). `import torch_tpu` and a TPU op still work with the
  patched loader installed.

Assisted-by: Claude Fable 5.1

🤖 Generated with [Claude Code](https://claude.com/claude-code)
