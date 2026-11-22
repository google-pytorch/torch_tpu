Approach on the branch:

**Where the error is caught.** `ExtensionProxyLoader` in `src/torch_tpu/_versioned_so_loader.py` already wraps the real extension loader. Both `create_module` and `exec_module` now catch `ImportError` from the underlying loader and re-raise through a new `_describe_failure(error)` helper, chaining the original with `raise ... from e` so the dynamic-loader text is preserved as `__cause__` and nothing is lost for debugging.

**What the message contains.** `_describe_failure` derives the glue's PyTorch version from the chosen `.so` name with the existing `_glue_suffix` helper (so `env_2_14_0` reads as "built for PyTorch 2.14.0"), reads the running version through `installed_torch_version()`, and says what to do: install a matching torch and torch_tpu pair, or build from source for this PyTorch. Either version falls back to "an unknown version" rather than failing inside the error path. Nothing about the selection logic changes; only the failure text does.

**Scope.** 29 lines in the loader, and a test in `tests/_versioned_so_loader_test.py` that drives a fake real-loader raising `ImportError` from both `create_module` and `exec_module`, patches `installed_torch_version` to `2.15.0+cpu`, and asserts the message names both versions and keeps the original as `__cause__`.

One judgment call for reviewers: the helper also fires for `ImportError`s that are not version mismatches (a missing `libtpu`, say). The message is worded to still be true in that case ("failed to load its extension module ... Underlying error: ..."), but if you would rather gate it on `undefined symbol` in the original text, that is a two-line change.

I would like to own this issue. Non-collaborators cannot self-assign, so if a maintainer can assign me, I will open the PR the moment a slot frees.
