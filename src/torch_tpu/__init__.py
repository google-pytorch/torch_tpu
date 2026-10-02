# Copyright 2025 Google LLC
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

"""TorchTPU library."""

import pathlib

from torch_tpu import _tcmalloc_check
from torch_tpu import _versioned_so_loader
from torch_tpu._internal import env_utils as _env_utils
from torch_tpu._internal.utils import hardware as _hardware

# Install the version-dispatch import hook before any torch_tpu._internal C++
# extension is imported, so imports resolve to the glue built for the installed
# PyTorch. This is a no-op for an unversioned build, which has no bundled glue.
_versioned_so_loader.install_hook(pathlib.Path(__file__).parent)

# Read by libtpu: any non-empty value makes it skip the GCE metadata server.
_SKIP_MDS_QUERY_ENV_VAR = "TPU_SKIP_MDS_QUERY"


def _skip_metadata_server_query_without_tpus() -> None:
  """Tells libtpu not to query the GCE metadata server on hosts without TPUs.

  When its TPU environment variables are unset, libtpu looks them up on the
  metadata server and retries each key 30 times, even on HTTP 404. On a host
  without TPUs, anything that initializes PJRT (e.g. autograd querying the TPU
  device guard during a CPU or CUDA `backward()`) then stalls for minutes before
  failing. TPU VMs rely on the metadata server for their topology, so this only
  applies when no TPU chip is on the PCI bus. Mirrors `jax._src.cloud_tpu_init`.
  """
  if _hardware.num_tpu_chips_on_pci() == 0:
    _env_utils.set_str_env(_SKIP_MDS_QUERY_ENV_VAR, "1")


# Must run before libtpu initializes, which happens lazily on first PJRT use.
_skip_metadata_server_query_without_tpus()

try:
  import libtpu  # pylint: disable=g-import-not-at-top # pytype: disable=import-error

  libtpu.configure_library_path()
except ImportError:
  print("libtpu not found.")
  pass

# Warn if tcmalloc is not the allocator of this process. Checked last so the
# probe sees every library `import torch_tpu` pulls in, libtpu included.
_tcmalloc_check.check_tcmalloc()

from torch_tpu import _loader as _  # pylint: disable=g-import-not-at-top  # noqa: F401

# TODO: b/477401982 - Introduce a new flag module to hold all TorchTPU flags.
