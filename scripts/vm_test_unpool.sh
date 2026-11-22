#!/usr/bin/env bash
# Run ON the VM after the max-unpool2d wheel build: install the wheel into
# ~/env_fix and run every test that covers the new kernels. Output -> stdout.
set -u
cd ~/torch_tpu
PY=~/env_fix/bin/python
filt() { grep -vE '^(I0000|W0000|E0000|num_|peak_|Loading|WARNING|Successfully|Registered|Device|Initializing)|UserWarning|See https|compilation_cache'; }
WHL=$(ls -t bazel-bin/ci/wheel/torch_tpu-*.whl | head -1)
echo "##### wheel: $WHL"
~/env_fix/bin/pip install -q --force-reinstall --no-deps "$WHL" 2>&1 | tail -1
$PY -c 'import torch, torch_tpu; print("torch", torch.__version__, "torch_tpu", torch_tpu.__version__)' 2>&1 | filt

echo "##### repro (#6103)"
$PY ~/max_unpool2d_repro.py 2>&1 | filt | grep -E '^(cpu|tpu) '

echo "##### ops_unit_test -k max_unpool"
PYTHONPATH=. $PY -m pytest -q tests/ops_unit_test.py -k max_unpool -p no:cacheprovider 2>&1 | filt | tail -25

echo "##### errors_test -k max_unpool"
PYTHONPATH=. $PY -m pytest -q tests/errors_test.py -k max_unpool -p no:cacheprovider 2>&1 | filt | tail -15

echo "##### native_functions_test"
# Outside bazel the test has no runfiles; point it at the header shipped in the
# torch wheel.
PYTHONPATH=. $PY - <<'PYEOF' 2>&1 | filt | tail -12
import os, sys, types, torch
sys.argv = ["native_functions_test"]
import tests.native_functions_test as t
hdr = os.path.join(os.path.dirname(torch.__file__), "include/ATen/RegistrationDeclarations.h")
t._HAS_RESOURCES = True
t.runfiles = types.SimpleNamespace(Create=lambda: types.SimpleNamespace(Rlocation=lambda p: hdr))
from absl.testing import absltest
absltest.main(module=t)
PYEOF

echo "##### captum DeepLift conv configs"
CAPTUM_TEST_DEVICE=tpu $PY -m pytest -q ~/test_accelerator.py -k "DeepLift" -p no:cacheprovider 2>&1 | filt | tail -8

echo "##### scalar_buffer_cache_test (unchanged, sanity)"
PYTHONPATH=. $PY -m pytest -q tests/scalar_buffer_cache_test.py -p no:cacheprovider 2>&1 | filt | tail -3
echo "##### UNPOOL TESTS DONE"
