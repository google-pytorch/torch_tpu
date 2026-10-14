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

"""OSS RBE worker pool routing and test environment helpers."""

visibility(["//..."])

# TPU generation -> RBE region -> OSS RBE pool, and TPU generation -> test env,
# for single-chip tests.
# TODO: b/567928058 - Add v6e, v7x and multi-chip pools as they come online.
_RBE_TPU_POOLS = {
    "v5": {
        "asia-southeast1": "ci_tpu_v5_asia_se1_rbe",
        "us-central1": "ci_tpu_v5_central1_rbe",
        "us-east1": "ci_tpu_v5_east1_rbe",
        "us-west1": "ci_tpu_v5_full_rbe",
    },
}

_RBE_TPU_ENV = {
    "v5": {
        "TPU_ACCELERATOR_TYPE": "v5e-1",
        "TPU_SKIP_MDS_QUERY": "1",
        "TPU_WORKER_HOSTNAMES": "127.0.0.1",
        "TPU_WORKER_ID": "0",
    },
}

# us-west1 is the default so its action keys, and its remote cache, don't change.
_DEFAULT_RBE_REGION = "us-west1"

def _rbe_region_condition(region):
    if region == _DEFAULT_RBE_REGION:
        return "//conditions:default"
    return "//:rbe_" + region.replace("-", "_")

def _is_multichip_tpu_tag(tag):
    if not tag.startswith("requires-") or ":" not in tag:
        return False
    _, chips = tag.split(":", 1)
    return chips.isdigit() and int(chips) > 1

def select_rbe_shard_count(raw_shard_count, clamped_shard_count):
    """Returns raw_shard_count on RBE and clamped_shard_count on single-VM runners."""
    if clamped_shard_count == raw_shard_count:
        return clamped_shard_count
    return select({
        "//:rbe_test_execution": raw_shard_count,
        "//conditions:default": clamped_shard_count,
    })

def _rbe_tpu_exec_properties(tags, tpu_generation, region):
    """Returns the exec_properties for a single-chip TPU test on RBE in `region`.

    Returns None if there is no pool for the test there. A TPU test with no
    pool gets nothing, so if the RBE filters let one through it fails on the
    CPU pool instead of running on the wrong TPU.
    """
    pool = _RBE_TPU_POOLS.get(tpu_generation, {}).get(region)
    if pool == None:
        return None
    for tag in tags:
        if _is_multichip_tpu_tag(tag):
            return None
    exec_properties = {
        "test.Pool": pool,
        "test.dockerPrivileged": "true",  # The TPU is /dev/vfio.
        # TODO: b/567928058 - Try running tests as a non-root user.
        "test.dockerRunAsRoot": "true",
        "test.dockerShmSize": "4g",
    }
    if "requires-net:external" in tags:
        exec_properties["test.dockerNetwork"] = "standard"
    return exec_properties

def add_rbe_exec_properties(kwargs, tags, tpu_generation):
    """Configures OSS RBE pool and TPU environment for a single-chip TPU test.

    Args:
        kwargs: The rule keyword arguments dict to update in-place.
        tags: The list of tags on the test target.
        tpu_generation: The TPU generation string (e.g. "v5").
    """
    pools_by_region = _RBE_TPU_POOLS.get(tpu_generation, {})
    exec_properties_by_region = {
        region: _rbe_tpu_exec_properties(tags, tpu_generation, region)
        for region in pools_by_region
    }
    if all([p == None for p in exec_properties_by_region.values()]):
        return
    rbe_exec_properties = select({
        _rbe_region_condition(region): exec_properties or {}
        for region, exec_properties in exec_properties_by_region.items()
    })
    existing_exec_properties = kwargs.get("exec_properties")
    kwargs["exec_properties"] = (
        rbe_exec_properties | existing_exec_properties
    ) if existing_exec_properties else rbe_exec_properties
    rbe_env = _RBE_TPU_ENV.get(tpu_generation)
    if rbe_env:
        rbe_env_select = select({
            "//:rbe_test_execution": rbe_env,
            "//conditions:default": {},
        })
        existing_env = kwargs.get("env")
        kwargs["env"] = (rbe_env_select | existing_env) if existing_env else rbe_env_select

rbe_tpu_exec_properties_for_testing = _rbe_tpu_exec_properties
