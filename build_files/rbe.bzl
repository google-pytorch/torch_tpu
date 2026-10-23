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

load("//build_files:tpu_tags.bzl", "required_tpu_chip_count")

visibility(["//..."])

# Every RBE region must be listed: a region missing from a pool's select
# falls through to //conditions:default, i.e. the us-west1 pool.
# TODO: b/567928058 - Have oss/BUILD.bazel load this list so there's one copy.
_RBE_REGIONS = (
    "asia-southeast1",
    "us-central1",
    "us-east1",
    "us-west1",
)

# (TPU generation, chip count) -> RBE region -> OSS RBE pool, and
# (TPU generation, chip count) -> test env.
# TODO: b/567928058 - Add v6e, v7x and 8-chip pools as they come online.
_RBE_TPU_POOLS = {
    ("v5", 1): {
        "asia-southeast1": "ci_tpu_v5_asia_se1_rbe",
        "us-central1": "ci_tpu_v5_central1_rbe",
        "us-east1": "ci_tpu_v5_east1_rbe",
        "us-west1": "ci_tpu_v5_full_rbe",
    },
    ("v5", 4): {
        "us-central1": "ci_tpu_v5_4chip_central1_rbe",
        "us-west1": "ci_tpu_v5_4chip_west1_rbe",
    },
}

def _rbe_tpu_env(accelerator_type):
    return {
        "TPU_ACCELERATOR_TYPE": accelerator_type,
        "TPU_SKIP_MDS_QUERY": "1",
        "TPU_WORKER_HOSTNAMES": "127.0.0.1",
        "TPU_WORKER_ID": "0",
    }

_RBE_TPU_ENV = {
    ("v5", 1): _rbe_tpu_env("v5e-1"),
    ("v5", 4): _rbe_tpu_env("v5e-4"),
}

# Multi-chip tags whose hardware has no RBE worker pool even when
# tpu_generation is "v5" (e.g. v5p-4 stays on the self-hosted runner).
# A denylist, not an allowlist of accelerator tags: an allowlist would also
# reroute single-chip tests. Single-chip v5p tests are tagged
# fails-on-tpu-v5, so they never get tpu_generation "v5".
_NO_RBE_POOL_TAG_PREFIXES = (
    "requires-tpu-v5p:",
    "requires-tpu-v6e:",
    "requires-tpu-v7x:",
)

# us-west1 is the default so its action keys, and its remote cache, don't change.
_DEFAULT_RBE_REGION = "us-west1"

def _rbe_region_condition(region):
    if region == _DEFAULT_RBE_REGION:
        return "//conditions:default"
    return "//:rbe_" + region.replace("-", "_")

def _rbe_tpu_pool_key(tags, tpu_generation):
    if tpu_generation == None:
        return None
    for tag in tags:
        if tag.startswith(_NO_RBE_POOL_TAG_PREFIXES):
            return None
    key = (tpu_generation, required_tpu_chip_count(tags))
    return key if key in _RBE_TPU_POOLS else None

def select_rbe_shard_count(raw_shard_count, clamped_shard_count, tags):
    """Picks the shard_count for a test, which may differ on RBE.

    Single-chip tests use raw_shard_count on RBE. Multi-chip tests keep
    clamped_shard_count everywhere, so high-shard tests don't fill the smaller
    multi-chip RBE worker pools.

    Args:
        raw_shard_count: The shard_count the test asked for.
        clamped_shard_count: The shard_count after `_clamp_shard_count`.
        tags: The list of tags on the test target.

    Returns:
        clamped_shard_count, or a select of raw_shard_count on RBE and
        clamped_shard_count elsewhere.
    """

    # TODO: b/567928058 - Unclamp multi-chip tests on RBE too, capped by the
    # pool's size instead of the self-hosted runner's chip count.
    if clamped_shard_count == raw_shard_count or required_tpu_chip_count(tags) > 1:
        return clamped_shard_count
    return select({
        "//:rbe_test_execution": raw_shard_count,
        "//conditions:default": clamped_shard_count,
    })

def _rbe_tpu_exec_properties(tags, tpu_generation, region):
    """Returns the exec_properties for a TPU test on RBE in `region`.

    Returns None if there is no pool for the test there. A TPU test with no
    pool gets nothing, so if the RBE filters let one through it fails on the
    CPU pool instead of running on the wrong TPU.
    """
    pool_key = _rbe_tpu_pool_key(tags, tpu_generation)
    if pool_key == None:
        return None
    pool = _RBE_TPU_POOLS[pool_key].get(region)
    if pool == None:
        return None
    exec_properties = {
        "test.Pool": pool,
        "test.dockerPrivileged": "true",  # The TPU is /dev/vfio.
        # TODO: b/567928058 - Try running tests as a non-root user.
        "test.dockerRunAsRoot": "true",
    }
    if "requires-net:external" in tags:
        exec_properties["test.dockerNetwork"] = "standard"
    return exec_properties

def add_rbe_exec_properties(kwargs, tags, tpu_generation):
    """Configures OSS RBE pool and TPU environment for a TPU test.

    Args:
        kwargs: The rule keyword arguments dict to update in-place.
        tags: The list of tags on the test target.
        tpu_generation: The TPU generation string (e.g. "v5").
    """
    pool_key = _rbe_tpu_pool_key(tags, tpu_generation)
    if pool_key == None:
        return
    rbe_exec_properties = select({
        _rbe_region_condition(region): (
            _rbe_tpu_exec_properties(tags, tpu_generation, region) or {}
        )
        for region in _RBE_REGIONS
    })
    existing_exec_properties = kwargs.get("exec_properties")
    kwargs["exec_properties"] = (
        rbe_exec_properties | existing_exec_properties
    ) if existing_exec_properties else rbe_exec_properties
    rbe_env = _RBE_TPU_ENV.get(pool_key)
    if rbe_env:
        rbe_env_select = select({
            "//:rbe_test_execution": rbe_env,
            "//conditions:default": {},
        })
        existing_env = kwargs.get("env")
        kwargs["env"] = (rbe_env_select | existing_env) if existing_env else rbe_env_select

rbe_tpu_exec_properties_for_testing = _rbe_tpu_exec_properties
