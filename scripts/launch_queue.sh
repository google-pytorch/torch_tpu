#!/usr/bin/env bash
# Run ON the VM. Detaches the TPU queue and the bazel build so they survive ssh session end.
sudo loginctl enable-linger "$USER"
pkill -f 'grep -q BUILD_EXIT' 2>/dev/null; pkill -f 'while ! grep -q BUILD_EXIT' 2>/dev/null
chmod +x ~/vm_queue.sh
cd ~/torch_tpu && git status --short | head -3
setsid nohup bash -c 'cd ~/torch_tpu && export PATH=$HOME/bin:$PATH TMPDIR=/mnt/disks/build/tmp && mkdir -p $TMPDIR && nice -n 19 bazel build --config=wheel_common //ci/wheel:torch_tpu_wheel --config=no_rbe --local_cpu_resources=HOST_CPUS*0.6 > ~/build.log 2>&1; echo "BUILD_EXIT=$?" >> ~/build.log' > /dev/null 2>&1 < /dev/null &
setsid nohup ~/vm_queue.sh > ~/queue.out 2>&1 < /dev/null &
sleep 3; echo "--- detached:"; pgrep -fa 'vm_queue|bazel' | grep -v pgrep | cut -c1-100
