# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Named test suites for the upstream PyTorch runner.

Each entry in SUITES is a pytest target — a path, a `path::class`, or a
`path::class::method`. Paths are relative to ${PYTORCH_SRC_DIR}/test (the
runner prefixes that automatically).

To toggle a test file on/off for a suite, comment / uncomment its line.

To add a new suite, add a key to SUITES. Suite names must be valid choices for
the workflow_dispatch input in .github/workflows/upstream_tests.yml.

Priority ordering follows the dev-discuss braindump
(https://dev-discuss.pytorch.org/t/3357) plus our own reading of which files
offer the highest signal-per-CI-minute for an out-of-tree PrivateUse1 backend.

The `smoke` suite is the default — fast, narrow, every CI run executes it.
The other suites are opt-in via UPSTREAM_SUITE / workflow_dispatch.
"""

DEFAULT_SUITE = "smoke"


SUITES: dict[str, list[str]] = {
    # Fast sanity check — the OpInfo dtype-coverage sweep against every
    # registered op. ~1k items, finishes in ~20 min with the chunked runner.
    # Use this on every PR / scheduled run.
    "smoke": [
        "test_ops.py::TestCommonPRIVATEUSE1",
    ],

    # OpInfo correctness — Mikayla's #1 priority for out-of-tree backends.
    # Auto-scales with every new op or dtype upstream adds.
    "opinfo_correctness": [
        "test_ops.py::TestCommonPRIVATEUSE1",
        # "test_ops.py::TestMathBitsPRIVATEUSE1",          # enable once TestCommon is stable
        # "test_ops.py::TestCompositeCompliancePRIVATEUSE1",
        # "test_ops.py::TestTagsPRIVATEUSE1",
        # "test_ops.py::TestFakeTensorPRIVATEUSE1",
        # "test_ops.py::TestForwardADWithScalarsPRIVATEUSE1",
        # "test_ops_gradients.py::TestBwdGradientsPRIVATEUSE1",
        # "test_ops_fwd_gradients.py::TestFwdGradientsPRIVATEUSE1",
    ],

    # Numerics on the most-used op families.
    "numerics": [
        # "test_unary_ufuncs.py::TestUnaryUfuncsPRIVATEUSE1",
        # "test_binary_ufuncs.py::TestBinaryUfuncsPRIVATEUSE1",
        # "test_reductions.py::TestReductionsPRIVATEUSE1",
        # "test_linalg.py::TestLinalgPRIVATEUSE1",
    ],

    # Tensor manipulation core — indexing, views, shape ops.
    "tensor_ops": [
        # "test_shape_ops.py::TestShapeOpsPRIVATEUSE1",
        # "test_view_ops.py::TestViewOpsPRIVATEUSE1",
        # "test_indexing.py::TestIndexingPRIVATEUSE1",
        # "test_sort_and_select.py::TestSortAndSelectPRIVATEUSE1",
        # "test_scatter_gather_ops.py::TestScatterGatherPRIVATEUSE1",
    ],

    # Frontend — only partially device-generic in v2.10, expect noisy first runs.
    "frontend": [
        # "test_torch.py::TestTorchDeviceTypePRIVATEUSE1",
        # "test_autograd.py::TestAutogradDeviceTypePRIVATEUSE1",
        # "test_nn.py::TestNNDeviceTypePRIVATEUSE1",
    ],

    # The kitchen sink — every file from every other suite. Long-running.
    # Use on-demand via workflow_dispatch when you want a full snapshot.
    "full": [
        "test_ops.py::TestCommonPRIVATEUSE1",
        # "test_ops.py::TestMathBitsPRIVATEUSE1",
        # "test_ops.py::TestCompositeCompliancePRIVATEUSE1",
        # "test_ops_gradients.py::TestBwdGradientsPRIVATEUSE1",
        # "test_ops_fwd_gradients.py::TestFwdGradientsPRIVATEUSE1",
        # "test_unary_ufuncs.py::TestUnaryUfuncsPRIVATEUSE1",
        # "test_binary_ufuncs.py::TestBinaryUfuncsPRIVATEUSE1",
        # "test_reductions.py::TestReductionsPRIVATEUSE1",
        # "test_indexing.py::TestIndexingPRIVATEUSE1",
        # "test_view_ops.py::TestViewOpsPRIVATEUSE1",
    ],
}


def get_suite(name: str) -> list[str]:
    """Return the list of pytest targets for the named suite."""
    if name not in SUITES:
        raise SystemExit(
            f"Unknown suite {name!r}. Known suites: {sorted(SUITES)}"
        )
    return [t for t in SUITES[name] if t and not t.startswith("#")]
