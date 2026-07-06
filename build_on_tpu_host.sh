#!/bin/bash
# Build script to run directly on TPU host
set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
print_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
print_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
print_error() { echo -e "${RED}[ERROR]${NC} $1"; }

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Building torch_tpu on TPU host${NC}"
echo -e "${BLUE}========================================${NC}"

# Verify we're on Linux
if [[ "$(uname -s)" != "Linux" ]]; then
    print_error "This script must be run on a Linux TPU host"
    exit 1
fi

print_success "Running on Linux"

# Check and install xxd if needed
if ! command -v xxd &> /dev/null; then
    print_warning "xxd not found, installing..."
    if command -v apt-get &> /dev/null; then
        apt-get update -qq
        apt-get install -y xxd
    elif command -v yum &> /dev/null; then
        yum install -y xxd
    else
        print_error "Cannot install xxd automatically. Please run: apt-get install xxd"
        exit 1
    fi
    print_success "xxd installed"
else
    print_success "xxd already available"
fi

# Check Python 3.12
if ! command -v python3.12 &> /dev/null; then
    print_error "Python 3.12 not found"
    exit 1
fi
print_success "Found Python 3.12"

# Install Bazel if needed
BAZEL_VERSION="8.6.0"
if ! command -v bazel &> /dev/null; then
    print_info "Installing Bazel..."
    TEMP_DIR=$(mktemp -d)
    cd "$TEMP_DIR"

    wget "https://github.com/bazelbuild/bazel/releases/download/${BAZEL_VERSION}/bazel-${BAZEL_VERSION}-installer-linux-x86_64.sh"
    chmod +x "bazel-${BAZEL_VERSION}-installer-linux-x86_64.sh"
    ./bazel-${BAZEL_VERSION}-installer-linux-x86_64.sh --user

    export PATH="$HOME/bin:$PATH"
    echo 'export PATH="$HOME/bin:$PATH"' >> ~/.bashrc

    cd -
    rm -rf "$TEMP_DIR"

    if command -v bazel &> /dev/null; then
        print_success "Bazel installed: $(bazel --version)"
    else
        print_error "Bazel installation failed"
        exit 1
    fi
else
    print_success "Bazel already installed: $(bazel --version)"
fi

# Get the torch_tpu directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Use /data/ for Bazel caches to avoid filling up the root disk
BAZEL_CACHE_DIR="/dev/shm/torch_tpu_wheel"
mkdir -p "$BAZEL_CACHE_DIR"
export BAZEL_OUTPUT_BASE="$BAZEL_CACHE_DIR/output_base"

print_info "Working directory: $PWD"
print_info "Bazel cache: $BAZEL_CACHE_DIR"
print_info "Locating PyTorch installation..."
TORCH_PATH=$(python -c "import torch, os; print(os.path.dirname(torch.__file__))")

if [[ -z "$TORCH_PATH" ]]; then
    print_error "Could not find PyTorch installation"
    exit 1
fi

print_success "Found PyTorch at: $TORCH_PATH"

# Export TORCH_SOURCE environment variable for Bazel.
# The torch_local_repo rule (in bazel/torch_local_repo.bzl) does
# ``cp -sr {TORCH_SOURCE}/* .`` to populate the bazel external repo, and the
# ``local_torch`` BUILD.bazel references libraries at ``torch/lib/libc10.so``
# etc. — i.e. it expects the **parent** of the torch directory (the
# site-packages root) so that ``torch/`` itself becomes a subdirectory in the
# bazel root. Pointing it at the torch directory directly produces
# ``lib/libc10.so`` inside the bazel root and the BUILD's ``torch/lib/...``
# refs miss.
export TORCH_SOURCE="$(dirname "$TORCH_PATH")"
print_info "Using TORCH_SOURCE=$TORCH_SOURCE (parent of torch dir)"

# Clean build
print_info "Cleaning previous builds..."
bazel --output_base="$BAZEL_OUTPUT_BASE" clean 2>/dev/null || true

# Build
print_info "Building wheel against local PyTorch (this will take 15-30 minutes)..."
echo ""

# --define TORCH_SOURCE=local is intentionally NOT passed: it tells the wheel
# rule to skip the pinned torch dep, but it ALSO swaps shims/torch:pytorch's
# python deps to ``[]``, which breaks the SDPA codegen genrule (it does
# ``import torch`` at build time and bazel's hermetic Python sandbox has no
# access to the conda env's torch). Instead we keep the lockfile-driven
# pypi_torch path active and patched requirements.txt to ``torch==2.11.0+cpu``
# so both the codegen and the wheel link against torch 2.11.
# --disk_cache overrides .bazelrc to put cache on the mounted /data/ disk.
if bazel --output_base="$BAZEL_OUTPUT_BASE" build //ci/wheel:torch_tpu_wheel --config=no_rbe --disk_cache="$BAZEL_CACHE_DIR/disk-cache"; then
    print_success "Build completed!"

    # Find and copy wheel
    WHEEL_PATH=$(find bazel-bin/ci/wheel -name "*.whl" -type f | head -1)

    if [[ -n "$WHEEL_PATH" ]]; then
        WHEEL_NAME=$(basename "$WHEEL_PATH")
        mkdir -p dist
        cp "$WHEEL_PATH" "dist/"

        print_success "Wheel: dist/$WHEEL_NAME"
        echo ""
        echo "To install:"
        echo -e "  ${YELLOW}pip install dist/$WHEEL_NAME --force-reinstall${NC}"
        echo ""
    else
        print_error "Wheel not found"
        exit 1
    fi
else
    print_error "Build failed"
    exit 1
fi

# Shut down Bazel server to avoid dangling background processes
# (the server inherits stdout/stderr which prevents kubectl exec from exiting)
bazel --output_base="$BAZEL_OUTPUT_BASE" shutdown 2>/dev/null || true
