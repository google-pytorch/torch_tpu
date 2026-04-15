load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

# Commit 2025-06-09: 244cec869d12e53378fa0efb610cd4c32a454ec8
http_archive(
    name = "com_google_googletest",
    sha256 = "f253ca1a07262f8efde8328e4b2c68979e40ddfcfc001f70d1d5f612c7de2974",
    strip_prefix = "googletest-28e9d1f26771c6517c3b4be10254887673c94018",
    urls = [
        "https://github.com/google/googletest/archive/28e9d1f26771c6517c3b4be10254887673c940189.zip",
    ],
)

# rules_cc 0.1.2 or higher is necessary for @rules_cc//cc:core_rules target
http_archive(
    name = "rules_cc",
    sha256 = "0d3b4f984c4c2e1acfd1378e0148d35caf2ef1d9eb95b688f8e19ce0c41bdf5b",
    strip_prefix = "rules_cc-0.1.4",
    url = "https://github.com/bazelbuild/rules_cc/releases/download/0.1.4/rules_cc-0.1.4.tar.gz",
)

http_archive(
    name = "rules_license",
    sha256 = "26d4021f6898e23b82ef953078389dd49ac2b5618ac564ade4ef87cced147b38",
    urls = [
        "https://mirror.bazel.build/github.com/bazelbuild/rules_license/releases/download/1.0.0/rules_license-1.0.0.tar.gz",
        "https://github.com/bazelbuild/rules_license/releases/download/1.0.0/rules_license-1.0.0.tar.gz",
    ],
)

# XLA defines rules_ml_toolchain, but does so too late in the ordering of its
# own setup, so we have to manually define it before XLA's setup runs.
# Commit 2026-03-27 : 40efd07eb8e6565e506562f36d7dc43cd83e5b32
http_archive(
    name = "rules_ml_toolchain",
    sha256 = "9bd46bc5e06a56a9335897be630d4b820c678281aa88302518c250921338ad22",
    strip_prefix = "rules_ml_toolchain-40efd07eb8e6565e506562f36d7dc43cd83e5b32",
    urls = [
        "https://github.com/google-ml-infra/rules_ml_toolchain/archive/40efd07eb8e6565e506562f36d7dc43cd83e5b32.tar.gz",
    ],
)

load("//bazel:wheel_deps.bzl", "torch_tpu_deps_repo")
load("//bazel:wheel_version.bzl", "torch_tpu_version_repo")

# torch_tpu_deps_repo(
#     name = "torch_tpu_deps",
#     pyproject_toml = "//:pyproject.toml",
#     requirements_txt = "//requirements:requirements.txt",
# )

# torch_tpu_version_repo(
#     name = "torch_tpu_version",
#     pyproject_toml = "//:pyproject.toml",
# )

# load("//bazel:xla_repo.bzl", "xla_repo")

# xla_repo()

load("@xla//:workspace4.bzl", "xla_workspace4")

xla_workspace4()

load("@xla//:workspace3.bzl", "xla_workspace3")

xla_workspace3()

# Initialize hermetic Python
load("//bazel:rules_python.bzl", "rules_python_repo")

rules_python_repo()

load("@xla//third_party/py:python_init_repositories.bzl", "python_init_repositories")

python_init_repositories(
    default_python_version = "system",
    local_wheel_dist_folder = "dist",
    local_wheel_inclusion_list = ["torch-*"],
    local_wheel_workspaces = ["//:WORKSPACE"],
    requirements = {
        "3.12": "//requirements:requirements.txt",
    },
)

load("@xla//third_party/py:python_init_toolchains.bzl", "python_init_toolchains")

python_init_toolchains()

# load("//bazel:pip_parse.bzl", "torch_tpu_pip_parse")
#
# torch_tpu_pip_parse()
#
# load("@pypi//:requirements.bzl", "install_deps")
#
# install_deps()

load("@xla//:workspace2.bzl", "xla_workspace2")

xla_workspace2()

load("@xla//:workspace1.bzl", "xla_workspace1")

xla_workspace1()

load("@xla//:workspace0.bzl", "xla_workspace0")

xla_workspace0()

load(
    "@rules_ml_toolchain//cc/deps:cc_toolchain_deps.bzl",
    "cc_toolchain_deps",
)

cc_toolchain_deps()

load(
    "@rules_ml_toolchain//gpu/cuda:cuda_json_init_repository.bzl",
    "cuda_json_init_repository",
)

cuda_json_init_repository()

load(
    "@cuda_redist_json//:distributions.bzl",
    "CUDA_REDISTRIBUTIONS",
    "CUDNN_REDISTRIBUTIONS",
)
load(
    "@rules_ml_toolchain//gpu/cuda:cuda_redist_init_repositories.bzl",
    "cuda_redist_init_repositories",
    "cudnn_redist_init_repository",
)

cuda_redist_init_repositories(
    cuda_redistributions = CUDA_REDISTRIBUTIONS,
)

cudnn_redist_init_repository(
    cudnn_redistributions = CUDNN_REDISTRIBUTIONS,
)

load(
    "@rules_ml_toolchain//gpu/cuda:cuda_configure.bzl",
    "cuda_configure",
)

cuda_configure(name = "local_config_cuda")

register_toolchains("@rules_ml_toolchain//cc:linux_x86_64_linux_x86_64")

# --- Local Torch Support ---
# Load the custom repository rule
load("//bazel:torch_local_repo.bzl", "torch_local_repo")

# Define the local torch repository.
# It will check for the TORCH_SOURCE env var.
# If set, it uses that path. If not, it creates a dummy stub to prevent crashes.
torch_local_repo(
    name = "local_torch",
)
