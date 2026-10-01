# TorchTPU Python Version Support Policy

**TL;DR:** TorchTPU matches
[PyTorch's Python version support matrix](https://github.com/pytorch/pytorch/blob/main/RELEASE.md#release-compatibility-matrix)
for the minimum version of PyTorch supported by TorchTPU. Currently, TorchTPU
supports **Python 3.11+**.

## Policy

TorchTPU aligns its supported Python versions with upstream PyTorch's support
matrix for our minimum supported PyTorch release:

*   See PyTorch's
    [Release Compatibility Matrix](https://github.com/pytorch/pytorch/blob/main/RELEASE.md#release-compatibility-matrix)
    for the Python versions supported by each PyTorch release.
*   See PyTorch's
    [Python support policy](https://github.com/pytorch/pytorch/blob/main/RELEASE.md#python)
    and
    [RFC-0038](https://github.com/pytorch/rfcs/blob/master/RFC-0038-cpython-support.md)
    for how PyTorch adds and sunsets support for CPython minor versions as they
    reach [End-of-Life (EOL)](https://devguide.python.org/versions/).

## Currently Supported Versions

<!-- LINT.IfChange(python_version_support) -->

TorchTPU currently supports Python 3.11, 3.12, 3.13, and 3.14 (including
free-threaded 3.14t).

<!-- LINT.ThenChange(
    ../../oss/pyproject.toml:requires_python,
    ../../oss/bazel/supported_python_versions.bzl:supported_python_versions,
    ../../oss/README.md:python_version_support,
) -->
