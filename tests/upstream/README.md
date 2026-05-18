# Upstream PyTorch test runner

Runs **upstream `pytorch/pytorch` tests** unmodified against the torch_tpu
PrivateUse1 backend. The upstream test files are **not** vendored — CI clones
the pytorch source at a pinned SHA and points `PYTORCH_SRC_DIR` at it.

This directory only contains glue:

- `upstream_plugin.py` — pytest plugin that loads torch_tpu, filters collection
  to PrivateUse1/TPU device variants, and applies the skip list.
- `skip_list.py` — `(module, class, method)` → `(skip|xfail, reason)` table.
- `run_upstream.sh` — CI entry point.
- `pytorch_pin.txt` — upstream SHA (or `main`) to test against.

## CI only

This runner is **CI only**. End users of the torch_tpu wheel never need the
pytorch source tree. The dependency on `pytorch/test/` exists solely at
test-execution time on CI runners (and on developer machines reproducing a CI
failure).

## Run locally

```bash
# 1. Install torch_tpu into your env (wheel or editable).
# 2. Point at an existing pytorch checkout (or let the script clone one).
export PYTORCH_SRC_DIR=/path/to/pytorch
./tests/upstream/run_upstream.sh
```

Pass any extra pytest args after the script name, e.g. to narrow to one file:

```bash
./tests/upstream/run_upstream.sh "$PYTORCH_SRC_DIR/test/test_unary_ufuncs.py" -k acos
```

## What "success" means

The workflow step exits 0 if the runner produced a junit XML, regardless of
whether individual tests passed. Test failures live in the junit artifact and
should be triaged into `skip_list.py` over time. A red workflow means the
infrastructure broke (wheel build, install, plugin import, pytorch checkout,
collection error), not "some tests failed".

## Suites — what to run

The set of pytest targets is defined in
[`test_suites.py`](./test_suites.py). Each named suite is a list of
`path::class[::method]` entries relative to `${PYTORCH_SRC_DIR}/test/`.

- Enable a test file → uncomment its line in the suite.
- Disable one → comment it out.
- Add a new suite → add a key to `SUITES` and a corresponding option to the
  `suite` choice input in `.github/workflows/upstream_tests.yml`.

The active suite is picked via the `UPSTREAM_SUITE` env var; CI uses
`workflow_dispatch.inputs.suite` if set, otherwise `DEFAULT_SUITE` from
`test_suites.py`.

Pre-defined suites:

| Suite                 | What it covers                                                  |
|-----------------------|-----------------------------------------------------------------|
| `smoke` (default)     | `test_ops::TestCommon` only — fastest meaningful sweep.         |
| `opinfo_correctness`  | All OpInfo classes in `test_ops`, plus gradient suites.         |
| `numerics`            | Unary/binary ufuncs, reductions, linalg.                        |
| `tensor_ops`          | shape/view/indexing/sort/scatter ops.                           |
| `frontend`            | `test_torch`, `test_autograd`, `test_nn` (often noisy).         |
| `full`                | Kitchen sink — combine on-demand for snapshot runs.             |

## Adding a skip / xfail

Edit `skip_list.py` and add an entry. The method key is the basename **without**
the device suffix that `instantiate_device_type_tests` appends — write
`test_add`, not `test_add_tpu` or `test_add_privateuse1`. See file docstring for
examples.

## Bumping the pytorch pin

Edit `pytorch_pin.txt` to a new commit SHA, push, and triage any new failures
into the skip list. Pinning (rather than tracking `main`) keeps CI reproducible
and bumps deliberate.
