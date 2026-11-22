Follow-up on the v6e-4 (second node, same zone, libtpu 0.0.51 in both envs): the leak reproducer run as four torchrun workers, one per chip, with `dist.init_process_group(backend="tpu_dist")`. Every rank reports the same counts as the single-process run:

```
[rank 0..3] same scalar x5000     : +2 live allocations
[rank 0..3] new scalar each x5000 : +903 live allocations
[rank 0..3] captum IG same n_steps x20   : +0 live allocations
[rank 0..3] captum IG n_steps 10..109    : +280 live allocations
```

Log: https://github.com/joinalahmed/captum-on-torchtpu/blob/main/results/v6e-4/run2/results_fix-scalar-buffer-cache-bound.txt
