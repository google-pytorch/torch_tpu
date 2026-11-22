#!/usr/bin/env bash
# Run ON a fresh TPU VM (v5litepod-4 or v6e-4) with a data disk attached.
# Mounts the build disk, installs bazelisk + Python envs, clones the fork, and
# builds the torch_tpu wheel from main. Idempotent; safe to re-run.
set -euo pipefail
export PATH=$HOME/bin:$PATH
log() { echo "##### $* $(date -u +%H:%M:%S)" | tee -a ~/setup.log; }

log "mount build disk"
# The data disk is the whole disk with no partitions; the boot disk has them.
DEV=""
for d in $(lsblk -dpno NAME,TYPE | awk '$2=="disk"{print $1}'); do
  if [ "$(lsblk -no NAME "$d" | wc -l)" -eq 1 ]; then DEV="$d"; break; fi
done
sudo mkdir -p /mnt/disks/build
if [ -z "$DEV" ]; then
  log "no data disk attached; using the boot disk for /mnt/disks/build"
elif ! mountpoint -q /mnt/disks/build; then
  sudo blkid "$DEV" >/dev/null 2>&1 || sudo mkfs.ext4 -q -F "$DEV"
  sudo mount -o discard,defaults "$DEV" /mnt/disks/build
fi
sudo chown -R "$USER" /mnt/disks/build; mkdir -p /mnt/disks/build/tmp /mnt/disks/build/bazel
df -h /mnt/disks/build | tail -1 | tee -a ~/setup.log

log "system packages"
sudo apt-get update -qq && sudo apt-get install -y -qq python3.12 python3.12-venv python3.12-dev git build-essential >/dev/null

log "bazelisk"
mkdir -p ~/bin
[ -x ~/bin/bazel ] || { curl -fsSL -o ~/bin/bazel https://github.com/bazelbuild/bazelisk/releases/latest/download/bazelisk-linux-amd64; chmod +x ~/bin/bazel; }
cat > ~/.bazelrc <<'EOF'
startup --output_user_root=/mnt/disks/build/bazel
EOF

log "clone fork"
[ -d ~/torch_tpu ] || git clone -q https://github.com/joinalahmed/torch_tpu.git ~/torch_tpu
cd ~/torch_tpu && git remote get-url upstream >/dev/null 2>&1 || git remote add upstream https://github.com/google-pytorch/torch_tpu.git
git fetch -q --all && git checkout -q main && git reset -q --hard upstream/main

log "python envs"
# env: released wheel for Captum experiments. env_fix: matches the repo lockfile torch for from-source wheels.
[ -d ~/env ] || python3.12 -m venv ~/env
~/env/bin/pip install -q --upgrade pip
~/env/bin/pip install -q torch_tpu "transformers>=5" accelerate captum pytest scikit-learn 2>&1 | grep -iE "error" || true
# The released wheel has no torch_tpu.__version__; read the distribution version instead.
~/env/bin/python -c "import torch, torch_tpu, importlib.metadata as m; print('env: torch', torch.__version__, 'torch_tpu', m.version('torch_tpu'), 'devices', torch.accelerator.device_count())" 2>&1 | tail -1 | tee -a ~/setup.log

log "build wheel from main (background, see ~/build_main.log)"
cd ~/torch_tpu
nohup bash -c 'export PATH=$HOME/bin:$PATH TMPDIR=/mnt/disks/build/tmp; nice -n 10 bazel build --config=wheel_common //ci/wheel:torch_tpu_wheel --config=no_rbe --local_cpu_resources=HOST_CPUS*0.8 > ~/build_main.log 2>&1; echo "BUILD_EXIT=$?" >> ~/build_main.log' >/dev/null 2>&1 &
log "setup done; build running"
