To: tpu-builders-support@google.com
Subject: TPU Builders Program: GPUS_ALL_REGIONS block and quota increase for project gen-lang-client-0320701242 (TorchTPU Content Sprint)

Hello TPU Builders team,

I am a member of the TPU Builders Program and I am participating in the TorchTPU Content Sprint (#TorchTPUSprint, submissions due 25 November 2026). My entry is on explainability as a performance lens for TorchTPU: Captum attribution across execution modes, profiled with XProf. The single-chip work is done and has produced upstream fixes to TorchTPU and Captum; the remaining step is to validate those fixes and the write-up on multi-chip slices before the deadline. I am writing because the project is blocked on two quota items that the Getting Started Guide says to raise with you.

Project

- Project ID: gen-lang-client-0320701242
- Billing is linked and active. The project has been running v6e-1 and v6e-4 on-demand TPU VMs in us-east5-b through the Cloud TPU API over the last two weeks.
- Preferred zones: us-east5-b (fallback us-central1-a). For v5e, us-west4-a or europe-west4-b as listed in the guide would also work.

1. GPUS_ALL_REGIONS is 0

gcloud compute project-info describe reports GPUS_ALL_REGIONS limit 0.0, and the regional PREEMPTIBLE_TPU_V6E / TPU_V6E metrics do not appear in us-east5 or us-central1. Following the guide, I have not tried to request these myself. Could you clear the global restriction and set a baseline v6e allocation (8 chips, preemptible and on-demand) in us-east5 and us-central1, so I can use the FLEX_START path with ct6e-standard-4t instances?

2. v5e serving quota

The Cloud TPU API quotas on the project are:

- tpu-v5s-litepod: 16, tpu-v5s-litepod-preemptible: 16
- tpu-v5s-litepod-serving: 4, tpu-v5s-litepod-serving-preemptible: 4
- tpu-v6e: 16, tpu-v6e-preemptible: 16

A v5litepod-8 create in us-east5-b fails with "Quota limit 'TPUV5sLitepodServingPerProjectPerZoneForTPUAPI' has been exceeded. Limit: 4 in zone us-east5-b." Could tpu-v5s-litepod-serving and tpu-v5s-litepod-serving-preemptible be raised from 4 to 8 chips? A v5litepod-8 is the same configuration as the torch_tpu upstream CI runners, which is why that specific size matters.

Capacity

Separately from quota, on-demand and spot v5litepod-4, v6e-4 and v6e-8 returned "Insufficient capacity" or "no more capacity in the zone" in us-east5-b and us-central1-a for most of 2026-11-01; a queued v6e-4 request eventually landed after several hours. If there is a zone you would recommend for 4 to 8 chip slices in the next two weeks, or a short-term reservation is the better route, I would be glad to follow that.

Use case

The sprint entry validates TorchTPU changes on multi-chip hardware before they go upstream. Open pull requests and issues from this work: google-pytorch/torch_tpu#6102, #6105, #6106 (open PRs), #6103, #6111 to #6114 (issues with fixes on branches), and meta-pytorch/captum#1951, #1952. The single-chip results and the write-up are at https://github.com/joinalahmed/captum-on-torchtpu. The multi-chip runs are needed to cover the v5e test variants that upstream CI uses, to exercise the distributed code paths touched by two of the fixes, and to add a multi-chip attribution comparison to the sprint blog post and the TorchTPU docs case study that is part of the submission.

Expected usage

- Duration: intermittent until the sprint deadline on 25 November 2026, then occasional follow-up on the open PRs, in sessions of 4 to 10 hours. VMs are created per session and deleted afterwards; environments live on a persistent disk.
- Peak concurrent usage: one 8-chip slice (v5e or v6e), occasionally one 4-chip slice alongside it for a few hours.
- Estimated monthly spend at these levels: USD 300 to 600.

Thank you,

Joinal Ahmed
GitHub: joinalahmed
GDE - Google Cloud & AI - TPU, vLLM
