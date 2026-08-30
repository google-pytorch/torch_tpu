#!/bin/bash
# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# ==============================================================================
# DESCRIPTION:
#   This script is a standalone, manually-executed infrastructure script. It is
#   not executed during the automated CI build process.
#
#   It generates a custom, hermetic CPython 3.12 tarball instrumented with
#   ThreadSanitizer (TSan), along with a standalone TSan shared library. These 2
#   artifacts are uploaded to GCS and downloaded by Bazel during CI tests.
#   1. The tarball is required because standard Python is uninstrumented and
#      will break TSan's memory tracking. Standard Python interpreters are not
#      compiled with TSan. If we execute TSan-instrumented PyTorch extensions
#      inside a standard Python environment, TSan misinterprets Python's
#      internal thread management (like the Global Interpreter Lock (GIL)) and
#      memory allocation, resulting in a flood of false-positive data races that
#      completely mask the real PyTorch bugs.
#   2. The .so file is required because Python imports C++ extensions using
#      dlopen with the RTLD_LOCAL flag. This hides the extension's symbols,
#      causing PyTorch to drop the TSan runtime and fail with undefined symbol:
#      __tsan_func_entry.
#
#   Lineage & Modifications:
#   This script is adapted from rules_ml_toolchain's reference script:
#   https://github.com/google-ml-infra/rules_ml_toolchain/blob/75b375bcf42810358974dd21c8da600d1f81cb01/py/dist/build-python-tsan-x86_64-ubuntu20.sh#L1-L79.
#   However, the original script was designed for Python 3.13/3.14
#   "Free Threading" (no-GIL) builds. Since our project requires Python 3.12,
#   the following critical modifications were made:
#   1. No Free-Threading flags: Python 3.12 does not support `--disable-gil`,
#      `--with-mimalloc`, or the native `--with-thread-sanitizer` configure
#      flags. Instead, we manually pass `-fsanitize=thread` via BASECFLAGS and
#      LDFLAGS to force TSan instrumentation onto the standard Python 3.12 build.
#   2. ABI matching: Instead of using `wget` to download a generic LLVM release
#      from GitHub, this script dynamically triggers Bazel and extracts the
#      exact LLVM 20 toolchain from Bazel's cache. This guarantees ABI
#      compatibility with the compiled PyTorch/XLA C++ extensions.
#   3. Dynamic library discovery: Hardcoded Ubuntu 20.04 library paths
#      (e.g., libssl.so.1.1) were replaced with dynamic resolution to allow
#      execution on newer runner images (like Ubuntu 22.04 with libssl.so.3).
#
# TODO: b/512474130 - Automate this process and/or make it config-based.
# Investigate building a reusable solution as a shared utility across projects.
# USAGE:
#   ONLY modify and run this script in the following scenarios:
#   1. The project changes to a new Python version.
#   2. The LLVM toolchain version is bumped.
#   3. The host OS (Ubuntu) is upgraded. The script bundles system libraries
#      like libssl and libffi, so a major glibc or OpenSSL change in the runner
#      environment might require a fresh tarball.

#   Permissions needed:
#   - The script is executable. Anyone with read access can run it.
#   - The GCS bucket gs://ml-sysroot-testing/python/ is publicly readable.
#     Contact ML Velocity team for write access.
#
#   Steps:
#   1. Run the script in the root of the torch_tpu repository:
#      $ ./ci/configs/build_hermetic_tsan_python.sh
#      Outputs are generated in /tmp/tsan_python_build/:
#        1. cpython-3.12.x-tsan-shared-linux-x86_64-llvm-20-bazel-YYYYMMDD.tar.gz
#        2. libclang_rt.tsan-llvm20-YYYYMMDD.so
#
#   2. Upload the timestamped outputs (in /tmp/tsan_python_build/) to the GCS
#      bucket:
#      $ gcloud storage cp /tmp/tsan_python_build/cpython-3.12.x-tsan-shared-linux-x86_64-llvm-20-bazel-YYYYMMDD.tar.gz gs://ml-sysroot-testing/python/
#      $ gcloud storage cp /tmp/tsan_python_build/libclang_rt.tsan-llvm20-YYYYMMDD.so gs://ml-sysroot-testing/python/
#      Uploaded files can be found at https://console.cloud.google.com/storage/browser/ml-sysroot-testing/python/.
#
#   3. Update hashes in .bazelrc and MODULE.bazel with the output provided at
#      the end of the script.
#
# ==============================================================================

set -euo pipefail

if ! command -v patchelf >/dev/null 2>&1; then
    echo "FATAL: 'patchelf' is required. Please install it (e.g., sudo apt-get install patchelf)."
    exit 1
fi

if [ ! -d "/usr/lib/x86_64-linux-gnu/" ]; then
    echo "FATAL: This script must be run on a Debian/Ubuntu-based host."
    echo "Could not find /usr/lib/x86_64-linux-gnu/, which is required for bundling system libraries (SSL, Crypto, FFI)."
    exit 1
fi

TIMESTAMP=$(date +%Y%m%d)
PY_VERSION="3.12"
PY_REPO_TAG="v${PY_VERSION}.3"
DST_DIR="/tmp/tsan_python_build"
CPYTHON_NAME="cpython-${PY_VERSION}.x-tsan-shared-linux-x86_64-llvm-20-bazel-${TIMESTAMP}"
TSAN_ARTIFACT_NAME="libclang_rt.tsan-llvm20-${TIMESTAMP}.so"

mkdir -p "${DST_DIR}"

echo "1. Forcing Bazel to fetch the LLVM toolchain..."
# Fetch a specific test target rather than //... to avoid parsing the entire
# repository and potentially failing on broken, unrelated targets.
bazel fetch //src/torch_tpu/_internal/utils:hardware_test || true

# Find Bazel's hidden cache directory where external dependencies are stored.
OUTPUT_BASE=$(bazel info output_base)

echo "2. Locating Bazel's internal LLVM toolchain..."
# Dynamically locate the clang binary inside the external cache.
# Filter by 'llvm' to ensure we grab the downloaded rules_ml_toolchain compiler.
LLVM_CLANG=$(find "$OUTPUT_BASE/external" -maxdepth 4 -path "*/bin/clang" | grep -i "llvm" | head -n 1)

if [ -z "$LLVM_CLANG" ]; then
    echo "FATAL: Could not find bin/clang in the Bazel cache at $OUTPUT_BASE/external!"
    exit 1
fi

# The toolchain root is exactly two directories up from bin/clang
LLVM_DIR=$(dirname $(dirname "$LLVM_CLANG"))
echo "--> Found LLVM toolchain at: $LLVM_DIR"

# Dynamically find the TSan shared library inside this specific LLVM folder.
# The exact name varies slightly by architecture (e.g., libclang_rt.tsan-x86_64.so).
TSAN_SO=$(find "$LLVM_DIR" -name "libclang_rt.tsan*.so" -print -quit)
if [ -z "$TSAN_SO" ]; then
    echo "FATAL: Could not find libclang_rt.tsan.so in the LLVM directory!"
    exit 1
fi

# Export standalone .so file for GCS upload and standardize the name.
cp "$TSAN_SO" "${DST_DIR}/${TSAN_ARTIFACT_NAME}"
echo "--> Exported standalone TSan library to: ${DST_DIR}/${TSAN_ARTIFACT_NAME}"

export LD_LIBRARY_PATH="$(dirname "$TSAN_SO"):${LD_LIBRARY_PATH:-}"

echo "3. Cloning and Configuring CPython ${PY_REPO_TAG}..."
cd "${DST_DIR}"
if [ -d "cpython" ]; then
  rm -rf cpython
fi

git clone -b "${PY_REPO_TAG}" --depth 1 https://github.com/python/cpython.git
cd cpython

# Configure CPython to compile with our specific LLVM toolchain.
# -shared-libsan forces it to dynamically link the TSan runtime, allowing the
# LD_PRELOAD wrapper to inject the symbols at execution time.
./configure \
  --enable-shared \
  --prefix="${DST_DIR}/${CPYTHON_NAME}" \
  CC="$LLVM_DIR/bin/clang" \
  CXX="$LLVM_DIR/bin/clang++" \
  LLVM_PROFDATA="$LLVM_DIR/bin/llvm-profdata" \
  BASECFLAGS="-shared-libsan -fsanitize=thread" \
  LDFLAGS="-shared-libsan -fsanitize=thread -Wl,-rpath,'\$\$ORIGIN/../lib:\$\$ORIGIN/../../../lib'"

echo "4. Compiling CPython..."
make -j"$(nproc)"
make install

echo "5. Bundling System & Sanitizer Libraries..."
# Copy the TSan library into the Python lib directory to ensure it is available
# to the interpreter at runtime.
cp "$TSAN_SO" "${DST_DIR}/${CPYTHON_NAME}/lib/"

# Dynamically find and copy required system libraries into the tarball to make
# the Python installation fully portable and hermetic.
SYS_SSL=$(ls /usr/lib/x86_64-linux-gnu/libssl.so.* | head -n 1)
SYS_CRYPTO=$(ls /usr/lib/x86_64-linux-gnu/libcrypto.so.* | head -n 1)
cp "$SYS_SSL" "${DST_DIR}/${CPYTHON_NAME}/lib/"
cp "$SYS_CRYPTO" "${DST_DIR}/${CPYTHON_NAME}/lib/"

# Adjust the RPATH of the SSL library to look for libcrypto in its own directory.
patchelf --set-rpath '$ORIGIN' "${DST_DIR}/${CPYTHON_NAME}/lib/$(basename "$SYS_SSL")"

# Dynamically find and copy the system FFI library (ignoring dpkg artifacts).
SYS_FFI=$(ls /usr/lib/x86_64-linux-gnu/libffi.so.* | grep -v "dpkg" | head -n 1)
cp "$SYS_FFI" "${DST_DIR}/${CPYTHON_NAME}/lib/"

cd "${DST_DIR}/${CPYTHON_NAME}/lib/"
# Create fallback symlinks for older Python extensions (like strict pip wheels)
# that expect specific libffi version numbers.
FFI_BASE=$(basename "$SYS_FFI")
if [ "$FFI_BASE" != "libffi.so.7" ]; then ln -sf "$FFI_BASE" libffi.so.7; fi
if [ "$FFI_BASE" != "libffi.so.8" ]; then ln -sf "$FFI_BASE" libffi.so.8; fi

echo "6. Creating portable CPython archive..."
cd "${DST_DIR}"
tar -czf "${CPYTHON_NAME}.tar.gz" "${CPYTHON_NAME}"

echo "============================================================"
echo " SUCCESS! Artifacts generated in: ${DST_DIR}"
echo "============================================================"
echo "1. Tarball: ${CPYTHON_NAME}.tar.gz"
echo "   SHA256:  $(sha256sum "${CPYTHON_NAME}.tar.gz" | awk '{print $1}')"
echo ""
echo "2. TSan SO: ${TSAN_ARTIFACT_NAME}"
echo "   SHA256:  $(sha256sum "${TSAN_ARTIFACT_NAME}" | awk '{print $1}')"
echo "============================================================"
echo "Upload these to GCS and update MODULE.bazel and .bazelrc with the new hashes."
