# TPU v6e memory sizing for Integrated Gradients

Context: Integrated Gradients expands the input to `n_steps` copies in one batched
forward/backward pass, so peak memory scales with `n_steps`. The facts below were
resolved by the `tpu-knowledge-base` agent against its installed source index
(run `20261101T112246Z-ef20b8d5446d`); nothing here was answered from memory and
nothing was run on the TPU VM. Where a source does not state something, that is
said explicitly.

## 1. HBM per TPU v6e (Trillium) chip and HBM bandwidth

- Each v6e chip has 32 GB of HBM.[^v6e] The TPU7x comparison table labels the same
  figure 32 GiB.[^7x]
- HBM bandwidth per chip is 1638 GBps.[^v6e]
- The scaling-book spec table gives the same figures rounded: 32 GiB HBM capacity
  per chip and 1.6e12 bytes/s HBM bandwidth per chip.[^sb]
- Not stated by any source: how much of the 32 GB is free for user tensors. The
  knowledge-base v6e overview notes HBM also holds communication-stack allocations
  and prefetched host data, so the usable budget for the IG batch is less than
  32 GB by an amount the sources do not quantify.[^v6e-page]

## 2. Chips in a v6e-4 and total HBM

- A physical v6e host houses 8 chips. The `ct6e-standard-4t` VM (topology 2x2,
  "v6e-4") is a half-host slice with 4 chips, 180 vCPUs and 720 GB host RAM.[^v6e]
- Total HBM across the slice: 4 chips x 32 GB = 128 GB. This multiplication is
  derived from the per-chip figure; no source states a per-slice HBM total.[^v6e]
- The 128 GB is four separate 32 GB HBMs, one per chip, not a single pool. The
  scaling-book describes v6e as one TensorCore per chip, so a single unsharded IG
  batch must fit in one chip's 32 GB unless the work is sharded across chips.[^sb]

## 3. torch_tpu memory API for the "tpu" device

Yes. The TorchTPU public-API list marks the following as **stable** for TPU (also
available under the `torch.accelerator.memory.` namespace):[^pub]

- `torch.accelerator.memory_allocated`
- `torch.accelerator.memory_reserved`
- `torch.accelerator.max_memory_allocated`
- `torch.accelerator.max_memory_reserved`
- `torch.accelerator.memory_stats`
- `torch.accelerator.reset_peak_memory_stats`
- `torch.accelerator.get_memory_info`
- `torch.accelerator.empty_cache`

Related facts from the same sources:

- There is no `torch.tpu.memory_*` API on the public list.[^pub]
- The only `torch.tpu` capacity items are the static properties
  `torch.tpu.get_device_properties().hbm_bytes_per_chip` and
  `.hbm_bytes_per_device`, both marked experimental.[^pub]
- `get_device_properties.total_memory` is deprecated as of TorchTPU 0.2.1.[^api]
- The `torch.accelerator` APIs require PyTorch 2.11 or newer.[^api]
- Not stated by either torch_tpu page: what "reserved" means for the PJRT
  allocator, the units returned, or whether the figures are per chip or per
  device. Those semantics are covered by PyTorch's own `torch.accelerator`
  documentation, which was not part of this lookup.[^pub][^api]

## Sources

[^v6e]: https://docs.cloud.google.com/tpu/docs/v6e
[^7x]: https://docs.cloud.google.com/tpu/docs/tpu7x
[^sb]: https://jax-ml.github.io/scaling-book/tpus/
[^v6e-page]: Knowledge-base page `data/tpu-knowledge-base/pages/hw/v6e/overview.md` (installed, read-only; cites the v6e doc above, the TPU7x doc above, and https://github.com/openxla/xla/blob/main/docs/sparsecore.md)
[^pub]: https://github.com/google-pytorch/torch_tpu/blob/main/docs/public_apis.md
[^api]: https://github.com/google-pytorch/torch_tpu/blob/main/docs/api_reference.md
