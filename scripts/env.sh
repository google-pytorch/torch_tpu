# source scripts/env.sh
export CLOUDSDK_PYTHON="$HOME/.local/share/uv/python/cpython-3.12-macos-aarch64-none/bin/python3.12"
export PATH="$HOME/google-cloud-sdk/bin:$PATH"
# Fill these in after `gcloud auth login`:
export TPU_PROJECT="${TPU_PROJECT:-}"
export TPU_ZONE="${TPU_ZONE:-}"            # pick a zone with v6e quota
export TPU_NAME="${TPU_NAME:-xai-v6e}"
export TPU_TYPE="${TPU_TYPE:-v6e-1}"
export TPU_RUNTIME="${TPU_RUNTIME:-v2-alpha-tpuv6e}"
export PATH="$HOME/bin:$PATH"
