#!/bin/bash
set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Building torch_tpu wheel with multi-host fix${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Function to print colored messages
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Detect OS
OS="$(uname -s)"
print_info "Detected OS: $OS"

# Check if running on macOS or Linux
if [[ "$OS" == "Darwin" ]]; then
    print_warning "Building on macOS - this wheel may not work on TPU hosts (which run Linux)"
    print_warning "You should build this on a Linux machine or TPU host"
    read -p "Continue anyway? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
    BAZEL_VERSION="8.6.0"
    BAZEL_INSTALLER="bazel-${BAZEL_VERSION}-installer-darwin-x86_64.sh"
    BAZEL_URL="https://github.com/bazelbuild/bazel/releases/download/${BAZEL_VERSION}/${BAZEL_INSTALLER}"
elif [[ "$OS" == "Linux" ]]; then
    print_success "Running on Linux - correct platform for TPU"
    BAZEL_VERSION="8.6.0"
    BAZEL_INSTALLER="bazel-${BAZEL_VERSION}-installer-linux-x86_64.sh"
    BAZEL_URL="https://github.com/bazelbuild/bazel/releases/download/${BAZEL_VERSION}/${BAZEL_INSTALLER}"
else
    print_error "Unsupported OS: $OS"
    exit 1
fi

# Check for Python 3.12
print_info "Checking for Python 3.12..."
if command -v python3.12 &> /dev/null; then
    PYTHON_VERSION=$(python3.12 --version)
    print_success "Found $PYTHON_VERSION"
else
    print_error "Python 3.12 not found. Please install Python 3.12 first."
    exit 1
fi

# Check for Bazel
print_info "Checking for Bazel..."
if command -v bazel &> /dev/null; then
    BAZEL_CURRENT_VERSION=$(bazel --version | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -1)
    print_success "Found Bazel version $BAZEL_CURRENT_VERSION"

    # Check if version is compatible (major version should match)
    BAZEL_MAJOR=$(echo "$BAZEL_CURRENT_VERSION" | cut -d. -f1)
    REQUIRED_MAJOR=$(echo "$BAZEL_VERSION" | cut -d. -f1)

    if [[ "$BAZEL_MAJOR" != "$REQUIRED_MAJOR" ]]; then
        print_warning "Bazel version mismatch. Required: ${BAZEL_VERSION}.x, Found: $BAZEL_CURRENT_VERSION"
        print_info "Installing recommended Bazel version..."
        INSTALL_BAZEL=true
    else
        print_success "Bazel version is compatible"
        INSTALL_BAZEL=false
    fi
else
    print_warning "Bazel not found. Installing Bazel $BAZEL_VERSION..."
    INSTALL_BAZEL=true
fi

# Install Bazel if needed
if [[ "$INSTALL_BAZEL" == "true" ]]; then
    TEMP_DIR=$(mktemp -d)
    cd "$TEMP_DIR"

    print_info "Downloading Bazel installer..."
    if command -v curl &> /dev/null; then
        curl -L -o "$BAZEL_INSTALLER" "$BAZEL_URL"
    elif command -v wget &> /dev/null; then
        wget -O "$BAZEL_INSTALLER" "$BAZEL_URL"
    else
        print_error "Neither curl nor wget found. Please install one of them."
        exit 1
    fi

    chmod +x "$BAZEL_INSTALLER"

    print_info "Installing Bazel (this may require sudo)..."
    if [[ "$OS" == "Darwin" ]]; then
        # On macOS, install to user directory to avoid sudo
        ./"$BAZEL_INSTALLER" --user
        export PATH="$HOME/bin:$PATH"
        print_info "Added $HOME/bin to PATH"
    else
        # On Linux, try user install first
        if ./"$BAZEL_INSTALLER" --user; then
            export PATH="$HOME/bin:$PATH"
            print_info "Added $HOME/bin to PATH"
        else
            print_warning "User install failed, trying with sudo..."
            sudo ./"$BAZEL_INSTALLER"
        fi
    fi

    cd "$SCRIPT_DIR"
    rm -rf "$TEMP_DIR"

    if command -v bazel &> /dev/null; then
        print_success "Bazel installed successfully: $(bazel --version)"
    else
        print_error "Bazel installation failed"
        exit 1
    fi
fi

# Check for other build dependencies
print_info "Checking for required build tools..."

if ! command -v git &> /dev/null; then
    print_error "git not found. Please install git."
    exit 1
fi

if ! command -v g++ &> /dev/null && ! command -v clang++ &> /dev/null; then
    print_error "C++ compiler not found. Please install g++ or clang."
    exit 1
fi

print_success "All required build tools are present"

# Clean previous builds (optional)
print_info "Cleaning previous build artifacts..."
bazel clean --expunge 2>/dev/null || true

# Show the modification we made
print_info "Verifying multi-host fix in discovery.cc..."
if grep -q "TPU_WORKER_HOSTNAME" "$SCRIPT_DIR/torch_tpu/distributed/slicebuilder/discovery.cc"; then
    print_success "Multi-host fix is present in discovery.cc"
else
    print_error "Multi-host fix not found in discovery.cc"
    print_error "The file may have been reverted. Please re-apply the changes."
    exit 1
fi

# Build the wheel
print_info "Starting build process (this may take 15-30 minutes)..."
echo ""

BUILD_CONFIG="--config=no_rbe"
if [[ "$OS" == "Linux" ]]; then
    # Add additional configs for Linux if needed
    print_info "Using config: $BUILD_CONFIG"
fi

print_info "Running: bazel build //ci/wheel:torch_tpu_wheel $BUILD_CONFIG"
echo ""

if bazel build //ci/wheel:torch_tpu_wheel $BUILD_CONFIG; then
    print_success "Build completed successfully!"
    echo ""

    # Find the wheel file
    WHEEL_PATH=$(find bazel-bin/ci/wheel -name "*.whl" -type f 2>/dev/null | head -1)

    if [[ -n "$WHEEL_PATH" ]]; then
        WHEEL_NAME=$(basename "$WHEEL_PATH")
        print_success "Wheel created: $WHEEL_NAME"
        print_info "Wheel location: $WHEEL_PATH"
        echo ""

        # Copy wheel to a convenient location
        WHEEL_OUTPUT_DIR="$SCRIPT_DIR/dist"
        mkdir -p "$WHEEL_OUTPUT_DIR"
        cp "$WHEEL_PATH" "$WHEEL_OUTPUT_DIR/"
        print_success "Wheel copied to: $WHEEL_OUTPUT_DIR/$WHEEL_NAME"
        echo ""

        # Print installation instructions
        echo -e "${GREEN}========================================${NC}"
        echo -e "${GREEN}Build Complete!${NC}"
        echo -e "${GREEN}========================================${NC}"
        echo ""
        echo "To install the wheel on your TPU hosts:"
        echo ""
        echo "1. Copy the wheel to your TPU hosts:"
        echo -e "   ${YELLOW}scp $WHEEL_OUTPUT_DIR/$WHEEL_NAME your-tpu-host:/tmp/${NC}"
        echo ""
        echo "2. On each TPU host, install the wheel:"
        echo -e "   ${YELLOW}pip install /tmp/$WHEEL_NAME --force-reinstall${NC}"
        echo ""
        echo "3. Restart your training job"
        echo ""

        if [[ "$OS" == "Darwin" ]]; then
            print_warning "IMPORTANT: This wheel was built on macOS"
            print_warning "It may not work on Linux TPU hosts"
            print_warning "Consider building on a Linux machine or directly on a TPU host"
        fi

    else
        print_error "Wheel file not found in bazel-bin/ci/wheel/"
        exit 1
    fi
else
    print_error "Build failed!"
    echo ""
    echo "Common issues:"
    echo "1. Missing dependencies - install with: pip install -r requirements/requirements.txt"
    echo "2. Bazel version mismatch"
    echo "3. Insufficient memory - try closing other applications"
    echo ""
    exit 1
fi
