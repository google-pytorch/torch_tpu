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

"""Upstream PyTorch Examples Runner for TorchTPU.

This script executes standalone scripts from the external pytorch/examples
repository (such as regression/main.py, fx/invert.py, fx/module_tracer.py)
against CPU or TPU devices in the TorchTPU CI environment.

Repo: https://github.com/pytorch/examples
"""

import gzip
import os
import pathlib
import runpy
import shlex
import shutil
import sys

from absl import app

# Pre-import torch and torch_tpu to initialize C++ extensions and TPU runtime
# before any upstream script imports numpy or other native libraries.
import torch  # pylint: disable=unused-import  # noqa: F401
from torch.utils import data
import torch_tpu  # pylint: disable=unused-import  # noqa: F401
import torchvision.datasets as vision_datasets


_DATASET_BASE_DIR = "/tmp/gcsfuse/data"
_MNIST_FILES = (
    "train-images-idx3-ubyte",
    "train-labels-idx1-ubyte",
    "t10k-images-idx3-ubyte",
    "t10k-labels-idx1-ubyte",
)


def _stage_mnist_datasets():
  """Stages MNIST raw files from GCS Fuse mount into local TEST_TMPDIR and redirects root.

  Upstream PyTorch examples instantiate datasets.MNIST pointing to arbitrary
  local directories (e.g. '../data' or './data') with download=True. In CI and
  internal sandbox environments, we avoid external network downloads and
  read-only
  filesystem issues by staging pre-existing MNIST files locally and patching
  torchvision's dataset class.

  Raises:
    FileNotFoundError: If the expected dataset mount directory does not exist.
  """

  # 1. Locate source dataset directory (pre-mounted via GCS fuse).
  gcs_raw_dir = os.path.join(_DATASET_BASE_DIR, "MNIST", "raw")
  if not os.path.exists(gcs_raw_dir):
    raise FileNotFoundError(
        f"GCS dataset directory '{gcs_raw_dir}' does not exist. Ensure GCS"
        f" bucket 'torchtpu-shared' is mounted at '{_DATASET_BASE_DIR}'."
    )

  # 2. Prepare local writable directory in TEST_TMPDIR for decompressed MNIST files.
  local_mnist_dir = os.environ.get("TEST_TMPDIR", "/tmp/mnist_data")
  local_raw_dir = os.path.join(local_mnist_dir, "MNIST", "raw")
  os.makedirs(local_raw_dir, exist_ok=True)

  # 3. Decompress `.gz` archives into uncompressed binary files expected by torchvision.
  for fname in _MNIST_FILES:
    target_path = os.path.join(local_raw_dir, fname)
    if not os.path.exists(target_path):
      gz_path = os.path.join(gcs_raw_dir, f"{fname}.gz")
      if not os.path.exists(gz_path):
        raise FileNotFoundError(f"Required MNIST file missing: {gz_path}")
      with gzip.open(gz_path, "rb") as f_in, open(target_path, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)

  # 4. Monkey-patch torchvision MNIST class.
  # We override __init__ on the dataset class so that unmodified upstream scripts
  # (which call MNIST('../data', download=True)) seamlessly use our pre-staged local
  # files without network requests.
  cls = vision_datasets.MNIST
  orig_init = cls.__init__

  # Override __init__ to intercept dataset instantiation:
  #   1. Force root path to our local writable directory (local_mnist_dir).
  #   2. Force download=False to prevent outbound network requests in CI runners.
  def _make_patched_init(orig):
    def _patched_init(self, *args, **kwargs):
      # Disable internet downloads.
      kwargs["download"] = False

      # Force dataset root to our staged directory regardless of what the script requested.
      if args:
        args = (local_mnist_dir,) + args[1:]
      else:
        kwargs["root"] = local_mnist_dir

      return orig(self, *args, **kwargs)

    return _patched_init

  cls.__init__ = _make_patched_init(orig_init)


def _patch_dataloader_no_multiprocessing():
  """Forces DataLoader to use num_workers=0 to prevent /dev/shm shared memory crashes in container CI runners."""
  orig_init = data.DataLoader.__init__

  def _patched_init(self, *args, **kwargs):
    kwargs["num_workers"] = 0
    kwargs["persistent_workers"] = False
    return orig_init(self, *args, **kwargs)

  data.DataLoader.__init__ = _patched_init


def _copy_cora_dataset():
  """Stages Cora dataset from GCS Fuse mount into local './cora' directory.

  Upstream PyTorch GAT example (gat/main.py) looks for './cora/cora.content' and
  './cora/cora.cites' in the current working directory. If missing, it downloads
  cora.tgz from linqs-data.soe.ucsc.edu. In CI and internal sandbox
  environments,
  we avoid external network downloads by staging the Cora dataset from the
  pre-mounted GCS bucket ('torchtpu-shared').

  Raises:
    FileNotFoundError: If the expected dataset files do not exist under the
      GCS mount directory.
  """
  # 1. Locate source Cora dataset directory (pre-mounted via GCS fuse at _DATASET_BASE_DIR/cora).
  gcs_cora_dir = os.path.join(_DATASET_BASE_DIR, "cora")

  gcs_content = os.path.join(gcs_cora_dir, "cora.content")
  gcs_cites = os.path.join(gcs_cora_dir, "cora.cites")
  if not os.path.exists(gcs_content) or not os.path.exists(gcs_cites):
    raise FileNotFoundError(
        f"GCS dataset files '{gcs_content}' or '{gcs_cites}' do not exist."
        " Ensure GCS bucket 'torchtpu-shared' is mounted at"
        f" '{_DATASET_BASE_DIR}'."
    )

  # 2. Prepare local './cora' directory in the current working directory where gat/main.py expects it.
  local_cora_dir = os.path.join(os.getcwd(), "cora")
  os.makedirs(local_cora_dir, exist_ok=True)

  # 3. Copy dataset files locally so gat/main.py detects them and skips network downloads.
  content_path = os.path.join(local_cora_dir, "cora.content")
  cites_path = os.path.join(local_cora_dir, "cora.cites")

  if not os.path.exists(content_path):
    shutil.copyfile(gcs_content, content_path)
  if not os.path.exists(cites_path):
    shutil.copyfile(gcs_cites, cites_path)


def _copy_bsds300_dataset():
  """Stages BSDS300 dataset from GCS Fuse mount into local './dataset/BSDS300/images' directory.

  Upstream PyTorch Super Resolution example (super_resolution/main.py) looks for
  './dataset/BSDS300/images' containing 'train' and 'test' folders. If missing,
  it downloads BSDS300-images.tgz from berkeley.edu. In CI and internal sandbox
  environments, we avoid external network downloads by staging the BSDS300
  dataset
  from the pre-mounted GCS bucket.

  Raises:
    FileNotFoundError: If the expected dataset files do not exist under the
      GCS mount directory.
  """
  # 1. Locate source BSDS300 dataset directory (pre-mounted via GCS fuse at _DATASET_BASE_DIR/BSDS300).
  gcs_bsds_dir = os.path.join(_DATASET_BASE_DIR, "BSDS300")
  if os.path.exists(os.path.join(gcs_bsds_dir, "images")):
    gcs_images_dir = os.path.join(gcs_bsds_dir, "images")
  else:
    raise FileNotFoundError(
        "GCS dataset directory './dataset/BSDS300/images' does not exist."
        " Ensure GCS bucket 'torchtpu-shared' is mounted at"
        f" '{_DATASET_BASE_DIR}'."
    )

  # 2. Prepare local './dataset/BSDS300/images' directory where super_resolution/data.py expects it.
  local_images_dir = os.path.join(os.getcwd(), "dataset", "BSDS300", "images")
  os.makedirs(os.path.dirname(local_images_dir), exist_ok=True)

  # 3. Copy dataset files locally so super_resolution/data.py detects them and skips network downloads.
  if not os.path.exists(local_images_dir):
    shutil.copytree(gcs_images_dir, local_images_dir, dirs_exist_ok=True)


def _copy_resnet18_checkpoints():
  """Stages ResNet18 checkpoint from GCS Fuse mount into Torchvision hub cache.

  Upstream PyTorch ImageNet example (imagenet/main.py) when run with
  --pretrained
  attempts to download 'resnet18-f37072fd.pth' from PyTorch Hub. In CI and
  internal sandbox environments with restricted network access, we stage the
  checkpoint from pre-mounted GCS bucket into Torchvision's hub checkpoint
  cache.
  """
  gcs_ckpt = os.path.join(
      _DATASET_BASE_DIR, "checkpoints", "resnet18-f37072fd.pth"
  )
  if not os.path.exists(gcs_ckpt):
    raise FileNotFoundError(
        f"GCS checkpoint file '{gcs_ckpt}' does not exist. Ensure GCS bucket"
        f" 'torchtpu-shared' is mounted at '{_DATASET_BASE_DIR}'."
    )

  hub_dir = torch.hub.get_dir()
  ckpt_dir = os.path.join(hub_dir, "checkpoints")
  os.makedirs(ckpt_dir, exist_ok=True)

  local_ckpt = os.path.join(ckpt_dir, "resnet18-f37072fd.pth")
  if not os.path.exists(local_ckpt):
    shutil.copyfile(gcs_ckpt, local_ckpt)


def _copy_imagenet_dataset():
  """Stages mini ImageNet dataset from GCS Fuse mount into local './imagenet' directory.

  Upstream PyTorch ImageNet example (imagenet/main.py) looks for
  './imagenet/train' and './imagenet/val' when --dummy is not used. In CI and
  internal sandbox environments with restricted network access, we stage a
  mini synthetic ImageNet dataset from the pre-mounted GCS bucket.
  """
  gcs_imagenet_dir = os.path.join(_DATASET_BASE_DIR, "imagenet_mini")
  if not os.path.exists(gcs_imagenet_dir):
    raise FileNotFoundError(
        f"GCS dataset directory '{gcs_imagenet_dir}' does not exist. Ensure GCS"
        f" bucket 'torchtpu-shared' is mounted at '{_DATASET_BASE_DIR}'."
    )

  local_imagenet_dir = os.path.join(os.getcwd(), "imagenet")
  if not os.path.exists(local_imagenet_dir):
    shutil.copytree(gcs_imagenet_dir, local_imagenet_dir, dirs_exist_ok=True)


def _copy_vgg16_checkpoints():
  """Stages VGG16 checkpoint from GCS Fuse mount into Torchvision hub cache.

  Upstream PyTorch Fast Neural Style example
  (fast_neural_style/neural_style/neural_style.py) uses
  torchvision.models.vgg16,
  which downloads 'vgg16-397923af.pth' from PyTorch Hub. In CI and internal
  sandbox environments with restricted network access, we stage the checkpoint
  from the pre-mounted GCS bucket into Torchvision's hub checkpoint cache.
  """
  gcs_ckpt = os.path.join(
      _DATASET_BASE_DIR, "checkpoints", "vgg16-397923af.pth"
  )
  if not os.path.exists(gcs_ckpt):
    raise FileNotFoundError(
        f"GCS checkpoint file '{gcs_ckpt}' does not exist. Ensure GCS bucket"
        f" 'torchtpu-shared' is mounted at '{_DATASET_BASE_DIR}'."
    )

  hub_dir = torch.hub.get_dir()
  ckpt_dir = os.path.join(hub_dir, "checkpoints")
  os.makedirs(ckpt_dir, exist_ok=True)

  local_ckpt = os.path.join(ckpt_dir, "vgg16-397923af.pth")
  if not os.path.exists(local_ckpt):
    shutil.copyfile(gcs_ckpt, local_ckpt)


def _setup_fast_neural_style():
  """Sets up style images and saved_models directory for Fast Neural Style."""
  # Copy style images from pytorch_examples repo to local working directory if present.
  pytorch_examples_dir = os.environ.get("TORCH_TPU_INTERNAL_TORCH_EXAMPLES_DIR")
  if pytorch_examples_dir:
    style_images_src = os.path.join(
        pytorch_examples_dir,
        "fast_neural_style",
        "images",
        "style-images",
    )
    if os.path.exists(style_images_src):
      local_style_dir = os.path.join(os.getcwd(), "images", "style-images")
      os.makedirs(os.path.dirname(local_style_dir), exist_ok=True)
      if not os.path.exists(local_style_dir):
        shutil.copytree(style_images_src, local_style_dir, dirs_exist_ok=True)

  # Create local './saved_models' directory for saving trained models.
  os.makedirs("saved_models", exist_ok=True)


def _setup_datasets(example: str):
  match example:
    case "gat/main.py":
      _copy_cora_dataset()
    case (
        "mnist/main.py"
        | "mnist_forward_forward/main.py"
        | "siamese_network/main.py"
        | "vae/main.py"
    ):
      # Create local './results' directory for saving checkpoints and models.
      # Required by upstream examples.
      os.makedirs("results", exist_ok=True)
      _stage_mnist_datasets()
    case "super_resolution/main.py":
      _copy_bsds300_dataset()
    case "imagenet/main.py":
      _copy_resnet18_checkpoints()
      _copy_imagenet_dataset()
    case "fast_neural_style/neural_style/neural_style.py":
      _copy_vgg16_checkpoints()
      _setup_fast_neural_style()
    case _:
      pass


def main(argv=None):
  del argv  # Unused.
  if not os.path.exists(_DATASET_BASE_DIR):
    raise FileNotFoundError(
        f"GCS bucket data directory '{_DATASET_BASE_DIR}' does not exist."
        " Ensure GCS bucket 'torchtpu-shared' is mounted at"
        f" '{_DATASET_BASE_DIR}'."
    )

  _patch_dataloader_no_multiprocessing()
  pytorch_examples_dir = os.environ.get("TORCH_TPU_INTERNAL_TORCH_EXAMPLES_DIR")
  if not pytorch_examples_dir:
    raise EnvironmentError("TORCH_TPU_INTERNAL_TORCH_EXAMPLES_DIR is not set.")

  rel_path = os.environ.get(
      "TORCH_TPU_INTERNAL_TORCH_EXAMPLE_PATH", "regression/main.py"
  )
  example_file_path = str(pathlib.Path(pytorch_examples_dir) / rel_path)
  if not os.path.exists(example_file_path):
    raise FileNotFoundError(f"Example file not found: {example_file_path}")

  extra_args = os.environ.get("TORCH_TPU_INTERNAL_TORCH_EXAMPLE_ARGS", "")
  sys.argv = [example_file_path] + (
      shlex.split(extra_args) if extra_args else []
  )

  _setup_datasets(rel_path)

  script_dir = str(pathlib.Path(example_file_path).parent)
  if script_dir not in sys.path:
    sys.path.insert(0, script_dir)
  if pytorch_examples_dir not in sys.path:
    sys.path.insert(0, pytorch_examples_dir)

  runpy.run_path(example_file_path, run_name="__main__")


if __name__ == "__main__":
  app.run(main, argv=[sys.argv[0]])
