# FLUX Performance Optimization Tutorial on Google Cloud TPUs

This tutorial walks users through an example of the kinds of optimizations we
can perform on models like FLUX for TPU.

The lessons we will cover are broadly generalizable to other models and can be
applied to achieve higher performance on TPU.

## Overview

Modern diffusion transformers like FLUX combine dual-stream attention,
multi-axis RoPE, and continuous timestep/prompt conditioning. The reference
implementation suffers from multiple performance bottlenecks on TPUs. In this
tutorial we will focus on:

1.  The advantages of using torch.compile on TPUs.
2.  Elimination of resizing and inefficient layout transformations through views
3.  Removing caching of full matrix and performing contractions directly and
    using single element-wise ops.
4.  Optimization through the use of custom kernels

--------------------------------------------------------------------------------

## Code & Architecture References

This tutorial is directly based on the
[torchtitan implementation of FLUX](https://github.com/google-pytorch/torchtitan/blob/main/torchtitan/models/flux/model/)
and
[black-forest-labs/flux](https://github.com/black-forest-labs/flux/blob/main/src/flux/)
repository.

Sections of code will highlight specific changes from the reference
implementations.

## Running Verification & Performance Profiling on TPU

### Running on TPU:

```bash
bazel test //examples/tutorials/FLUX_optimization:flux_baseline_test
```

### Running Benchmark Binary on a TPU Instance

```bash
bazel run //examples/tutorials/FLUX_optimization:flux_baseline_test -- --num_benchmark_steps=10
```
