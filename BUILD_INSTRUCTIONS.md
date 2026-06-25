# Building torch_tpu with Multi-Host Fix

## Changes Made
Modified `torch_tpu/distributed/slicebuilder/discovery.cc` to use actual hostnames instead of "localhost" for the SliceBuilder mesh configuration.

## Build Steps

1. Navigate to the torch_tpu directory:
```bash
cd /Users/kenzbensmaine/Documents/Apple_stuff/coreflow/torch_tpu
```

2. Build the wheel (this may take some time):
```bash
bazel build //ci/wheel:torch_tpu_wheel --config=no_rbe
```

3. Create a fresh virtual environment for installation (or use your existing one):
```bash
mkdir -p wheel
cd wheel
python3.12 -m venv venv
source venv/bin/activate
```

4. Install dependencies:
```bash
pip install -r ../requirements/requirements.txt
```

5. Install the newly built wheel:
```bash
pip install ../bazel-bin/ci/wheel/*.whl --force-reinstall
```

## Testing
After rebuilding and installing, test the multi-host setup again. The workers should now advertise their actual hostnames in the SliceBuilder mesh, allowing cross-host communication.

You should see log messages like:
```
Using hostname from TPU_WORKER_HOSTNAMES[0]: coretorch-try-multi-host-job-0-0.coretorch-try-multi-host for rank 0
Worker 0 advertising address: coretorch-try-multi-host-job-0-0.coretorch-try-multi-host:12345
```

Instead of all localhost addresses.
