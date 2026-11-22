#!/usr/bin/env bash
cd ~; export PATH=$HOME/bin:$PATH
filt() { grep -vE '^(I0000|W0000|E0000|num_|peak_|Loading|WARNING|Successfully|Registered|Device|Initializing|Warning)|UserWarning|F\.linear|get_attr_node|csrc/|See https|compilation_cache|Error loading program|Error allocating'; }
run() { local name=$1; shift; echo "##### $name  $(date -u +%H:%M:%S)" | tee -a ~/queue2.log; "$@" 2>&1 | filt | tee ~/results_$name.txt | tail -30 >> ~/queue2.log; }
PY=~/env/bin/python
~/env_fix/bin/pip install -q captum 2>&1 | tail -1
run scalar_leak_patched ~/env_fix/bin/python ~/scalar_leak_check.py
run matrix             $PY captum_tpu_matrix.py
run demo               $PY demo_captum_qwen_tpu.py
run compile_nan        $PY compile_nan_check.py
run ig2_compile_err    $PY -c 'import copy,torch;from transformers import AutoModelForCausalLM as M;m=M.from_pretrained("Qwen/Qwen3-0.6B",dtype=torch.float32,device_map="tpu").eval();c=torch.compile(m,backend="tpu");ids=torch.tensor([[1,2,3]],device="tpu")
try: print(c(input_ids=ids, logits_to_keep=1).logits.shape)
except Exception as e: import traceback; traceback.print_exc()'
run profile_strict     $PY profile_ig_modes.py strict
run profile_daf        env TORCH_TPU_DEFER_AND_FUSE=1 $PY profile_ig_modes.py daf
run profile_compile    $PY profile_ig_modes.py compile
echo "##### QUEUE2 DONE $(date -u +%H:%M:%S)" >> ~/queue2.log
