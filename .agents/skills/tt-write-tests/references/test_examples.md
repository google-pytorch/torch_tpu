# TorchTPU Testing Examples

This reference file contains examples for different types of tests in TorchTPU.

## Examples

### 1. Enabling or adding a test in `tests/ops_test.py`

Most PyTorch `op_db` ops already have a stub in `tests/ops_test.py` tagged with
`@op_testing.unimplemented_on_tpu`. When you implement an op so it is now
testable on TPU, remove `@op_testing.unimplemented_on_tpu` from its existing
stub instead of adding a duplicate method:

```python
# Before (skipped until implemented on TPU):
@op_testing.unimplemented_on_tpu
def test_{op_name}(self):
  self.do_test_op("{op_name}")

# After (enabled on TPU):
def test_{op_name}(self):
  self.do_test_op("{op_name}")
```

If your change also makes additional dtypes, gradients, or `out=` variants
testable, remove the corresponding `exclude_dtypes` / `exclude_inplace_dtypes`
entries or `check_grad=False` / `check_out_variant=False` flags. Only add a new
`test_{op_name}` method if you are adding a brand-new `OpInfo` to `_KNOWN_OPS`.

### 2. Adding a unit test to `tests/ops_unit_test.py`

For edge cases or specific behaviors, add a test to `tests/ops_unit_test.py`:

```python
def test_{op_name}_edge_case(self):
    x = torch.randn(2, 3, device="tpu")
    y = torch.randn(2, 3, device="tpu")

    def test_fn(device):
      return torch.{op_name}(x.to(device), y.to(device))

    self.assert_close_tpu_vs_cpu(test_fn)
```

### 3. Adding an error test to `tests/errors_test.py`

For error tests ensuring quality and parity with GPU, see
[tt-write-error-tests](../../tt-write-error-tests/SKILL.md).

### 4. Adding an error test to `tests/tpu_errors_test.py`

For error tests verifying TPU-only behaviors (decorated with
`@et.why_tpu_only`), see
[tt-write-error-tests](../../tt-write-error-tests/SKILL.md).

### 5. Adding a dynamism test to `tests/dynamism_test.py`

To verify an op supports bounded dynamism, you can add a test to
`tests/dynamism_test.py`:

```python
def test_{op_name}_dynamic_dim(self):
    def dynamic_foo(x, y):
        dynamism.mark_dynamic(x, 0, 2, 20)
        return torch.{op_name}(x, y)

    x = torch.randn(2, 2, device="cpu")
    y = torch.randn(2, 3, device="cpu")
    res = dynamic_foo(x.to(self.device), y.to(self.device))
    with CompilationCounter(self.device) as counter:
        res.cpu()
    self.assertEqual(counter.num_cache_hits(), 0)

    res = dynamic_foo(x.to(self.device), y.to(self.device))
    with CompilationCounter(self.device) as counter:
        res.cpu()
    self.assertEqual(counter.num_cache_hits(), 1)
```
