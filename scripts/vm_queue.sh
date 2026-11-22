#!/usr/bin/env bash
# Sequential TPU job queue. Launch detached on the VM: setsid nohup ~/vm_queue.sh > ~/queue.out 2>&1 < /dev/null &
cd ~; export PATH=$HOME/bin:$PATH
filt() { grep -vE '^(I0000|W0000|E0000|num_|peak_|Loading|WARNING|Successfully|Registered|Device|Initializing|Warning)|UserWarning|F\.linear|get_attr_node|^===|csrc/|See https|compilation_cache'; }
run() { local name=$1; shift; echo "##### $name  $(date -u +%H:%M:%S)" | tee -a ~/queue.log; "$@" 2>&1 | filt | tee ~/results_$name.txt | tail -40 >> ~/queue.log; }
PY=~/env/bin/python
# --- PR verification first
run accel_tests        env CAPTUM_TEST_DEVICE=tpu $PY -m pytest -q -ra -p no:cacheprovider test_accelerator.py
run scalar_leak_020    $PY scalar_leak_check.py
# --- verify the fix on the from-source wheel (Python version must match the wheel tag)
while ! grep -q BUILD_EXIT ~/build.log 2>/dev/null; do sleep 30; done
if grep -q BUILD_EXIT=0 ~/build.log; then
  WHL=$(ls ~/torch_tpu/bazel-bin/ci/wheel/*.whl | head -1); TAG=$(basename "$WHL" | grep -oE 'cp3[0-9]+' | head -1); PYV="3.${TAG#cp3}"
  echo "##### wheel $(basename "$WHL") -> python$PYV" | tee -a ~/queue.log
  command -v "python$PYV" >/dev/null || sudo apt-get install -y -qq "python$PYV" "python$PYV-venv" >/dev/null 2>&1
  if command -v "python$PYV" >/dev/null; then
    rm -rf ~/env_fix && "python$PYV" -m venv ~/env_fix && ~/env_fix/bin/pip install -q "$WHL" --extra-index-url https://download.pytorch.org/whl/cpu 2>&1 | tail -2 | tee -a ~/queue.log && ~/env_fix/bin/pip install -q absl-py pytest 2>&1 | tail -1
    run fix_regression_test bash -c "cd ~/torch_tpu && PYTHONPATH=. ~/env_fix/bin/python -m pytest -q tests/scalar_buffer_cache_test.py"
    run scalar_leak_patched ~/env_fix/bin/python ~/scalar_leak_check.py
  else
    echo "python$PYV unavailable; skipping fix verification" | tee -a ~/queue.log ~/results_fix_regression_test.txt
  fi
fi
# --- content
run matrix             $PY captum_tpu_matrix.py -v
run demo               $PY demo_captum_qwen_tpu.py
run accelerator_probe  $PY -c 'import torch, torch_tpu; print("acc:", torch.accelerator.current_accelerator(), "avail:", torch.accelerator.is_available(), "count:", torch.accelerator.device_count()); print("dev:", torch.ones(2, device="tpu").device, "torch.tpu:", hasattr(torch, "tpu"))'
run other_libs         ~/env2/bin/python other_xai_libs_tpu.py
run ig2_highest        env PREC=highest $PY hf_ig_bench2.py
run ig2_high           env PREC=high $PY hf_ig_bench2.py
run ig2_high_daf       env TORCH_TPU_DEFER_AND_FUSE=1 PREC=high $PY hf_ig_bench2.py
run compile_agreement  $PY compile_agreement.py
run profile_strict     $PY profile_ig_modes.py strict
run profile_daf        env TORCH_TPU_DEFER_AND_FUSE=1 $PY profile_ig_modes.py daf
run profile_compile    $PY profile_ig_modes.py compile
echo "##### QUEUE DONE $(date -u +%H:%M:%S)" >> ~/queue.log
