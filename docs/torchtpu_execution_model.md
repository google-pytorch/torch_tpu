# TorchTPU execution model

When executing a PyTorch program using TorchTPU, many things happen under the
hood to optimize the execution. Understanding this model can help you understand
how to tune your experience, and take advantage of some of TorchTPU’s unique
strengths.

## Execution pipeline

### Create deferred op

When a PyTorch operation (or just an "op") is executed on the TorchTPU backend,
initially, no work is performed on the TPU itself.

Instead, the logical result of the computation is packaged into a deferred op.
This is a CPU-side memory object which contains all the necessary information to
perform this operation; its inputs, the logical steps to take, and the shape and
type of the outputs, among other things.

The remainder of the pipeline, which replaces this symbolic deferred op with
real data, is called materialization.

### Enqueue op on a stream

TorchTPU supports the `torch.Stream` interface. Whenever a DeferredOp is
created, it is pushed into the current active Stream to queue it for later
operation.

At this point, the calling thread is done; everything past this point happens
asynchronously in the background (except in launch-blocking mode; more on this
later).

### Analyze stream and prepare execution plan

Once there is work to do on a stream, and any cross-stream timing dependencies
are satisfied, at some point a background thread will pick up work from that
stream. This background thread will then jointly analyze the work on the stream,
considering many different factors, including:

*   How many individual executions does this need to be split into?
*   What are the inputs and outputs of each execution block?
*   What level of compilation optimization should be applied to each?
*   Have we seen each computation unit before, or is it new?
*   What core (TensorCore or SparseCore) should this work execute on?

By considering the work to do collectively, the background thread can make
improved execution decisions.

Once the background thread has identified the final outputs of each block, it
will mark these outputs as pending to indicate that they will be executed in the
future, and proceed to the next step.

### Compile (or fetch) executables

For each execution stage, the background thread will check with a compilation
cache to see if an executable has already been compiled with the correct
settings. By default this is a local-memory cache (tier 1), with options to
enable persistent device-local cache (tier 2) or remote caches (tier 3) to
minimize recompilation costs.

On a cache miss, the compilation graph for the unit will be converted into MLIR
(primarily StableHLO, except for user-defined custom kernels) and handed off to
a compilation threadpool. This threadpool will asynchronously work to compile
many compilation units in parallel, and return them to the background worker
thread once ready.

### Start execution

Once the inputs have been collected and the executable has been prepared, an
execution task is enqueued with XLA's runtime,
[PjRt](https://openxla.org/xla/pjrt).

PjRt will return promises for this future output, which is then plugged into the
outputs to indicate that they are materializing.

### (Optional) Wait for completion

To know when work is complete, you can record a `torch.Event` on the stream, and
use the `torch.Event.query` method to inspect it.

Alternatively or in addition, you can block on the stream using any
"synchronize" instruction (`torch.Event.synchronize`,
`torch.Stream.synchronize`, or `torch.accelerator.synchronize` for full-device),
which will pause your Python code until the TPU signals completion.

## Execution modes

TorchTPU's programming pipeline is flexible, and supports a number of different
execution modalities.

### Strict eager mode

Also known as "default eager" or "defer never", this mode is the familiar,
out-of-the-box PyTorch experience. In this mode, every op that is appended to
the queue immediately flags that stream for execution. This results in ops being
processed one-at-a-time by the materialization background worker.

These single-op execution graphs have minimal optimization potential, and so are
compiled with the lower `-O1` compilation level to maximize compile speed.
Because there are many individual compilations, these are parallelized very
effectively over the compilation threadpool, and the level of reuse for single
operations is very high. This results in minimal compilation overhead while
"warming up" the compilation cache.

Note that while this mode executes eagerly and continuously, it still does so
asynchronously; a synchronization is still necessary to wait for the TPU to
finish if you want to observe the results, and any runtime errors may surface
much later than the op which created them.

### Debug eager mode

Also known as "launch blocking", this is enabled by setting the environment
variable `TORCH_TPU_LAUNCH_BLOCKING=1`. This is intended for interactive
debugging only, and not for production use.

This mode is the same as strict eager (op-by-op execution), except that the host
will block and synchronize on every operation. This artificially slows down
execution, but ensures that failures are reported as soon as they occur, to help
pinpoint errors.

### `torch.compile`

For maximum performance, `torch.compile` provides both greater optimization
potential, and reduced runtime overhead.

By compiling a specific function or module with `torch.compile`, TorchTPU's
backend is provided with the full scope of the compilation target as an
[FX graph](https://docs.pytorch.org/docs/stable/fx.html). This gives the maximum
amount of information about the intended execution, along with a guarantee that
it will not change, allowing us to make aggressive optimizations that would be
too risky in eager mode.

As a compiled mode backend, TorchTPU will perform one initial trace (using
"placeholder" tensors, similar to FakeTensor but retaining TPU-specific
metadata), producing a fully-deferred computation graph that replicates the FX
graph. Once this is created, we then provide this StableHLO program to XLA to
compile using the higher `-O2` optimization level. This program is then packaged
into a Python callable and returned to the user for later execution.

When this callable is invoked, we immediately bypass most of the eager execution
steps. No new DeferredOps are created; no further graph analysis is needed; and
the executable is already provided, so no caching is required. Instead, the
executable is immediately pushed to PjRt for execution, minimizing the host-side
cost with a computation graph fully optimized by the XLA compiler.

### Optimized eager mode

Also known as "Defer and Fuse" mode, this is enabled by setting
`TORCH_TPU_DEFER_AND_FUSE=1`. This is a new, hybrid mode, bringing many of the
advantages of `torch.compile` into eager mode.

In this mode, creating a new DeferredOp does not immediately begin execution; we
allow the ops to remain deferred between Python calls, waiting in the Stream
queue.

We will begin materialization when one of two conditions is met:

*   The user performs some action which forces execution. The most common of
    these are starting a device-to-host transfer or an explicit synchronize
    call.
*   We detect a repetition in the dispatched ops, indicating that we are in a
    loop and should compile one iteration of that loop for easy reuse.

This signals a background materialization, similar to default eager mode, except
that the graph being compiled is larger than a single operation. Any tensors
which have already been dropped or overwritten when the materialization signal
was raised will be skipped, usually fusing them into downstream operations, or
eliminating them entirely if the output is discarded without being read. (Note:
dead code elimination is skipped for distributed collective ops to avoid
cross-TPU deadlocks).

Because this results in larger compilation graphs, we use optimization level 2
(`-O2`) to improve the execution time. As we are compiling a smaller number of
larger graphs, there is reduced parallelization benefit from the compilation
threadpool, so you can expect compile times to be longer than strict eager.

While not as optimized as `torch.compile` mode, optimized eager mode shows up to
50% performance improvement over eager default, with no change to model code.

## Pitfalls

While this execution pipeline is optimized for both latency and throughput,
there are cases where performance can suffer.

### All modes: Python-side control flow and data-dependent shapes

In PyTorch, it can be easy to accidentally and silently force a device-to-host
transfer without intending to. An operation like "`if my_tensor.all()`" will
silently construct a boolean-valued tensor, synchronize the stream, and copy
that boolean to the CPU before continuing with the control flow.

In default eager mode, this will not force any new graph breaks, but will force
the stream's entire execution pipeline to drain before proceeding. Even worse,
in optimized eager mode or compiled mode, this forced sync point will force a
graph break in either the eagerly-fused stream or the FX graph, creating an
optimization barrier.

Similar behavior can also occur with PyTorch operations that have data-dependent
output shapes, such as `torch.masked_select`. Because the size of the output
tensor is unknown until the mask input is evaluated, this likewise forces a
fully blocking stream synchronization and device-to-host transfer of the shape
metadata before proceeding.

### Strict eager mode: tiny executables cause microbubbles

Enqueueing an executable with PjRt takes a small amount of time; usually, 50
microseconds or less. In most cases, each executable will take longer than this
(even if only 1 or 2 milliseconds), and will fully hide this latency, leading to
a steady-state with the TPU operating continuously at peak efficiency.

However, if too many executables are enqueued that are below this threshold–such
as when using strict eager mode with small tensor sizes–this can lead to TPU
starvation. When TPU runs faster than the CPU can enqueue more work, the TPU
execution will be interrupted by tiny but frequent "microbubbles" as it
bottlenecks on dispatch speed.

To avoid this, ensure that your models are doing a sufficient amount of work on
each operation to overcome this small launch overhead; or, use optimized eager
mode to automatically fuse executions into compilation blocks on the fly.

### Compiled mode: views as arguments or outputs

When calling a compiled executable, PjRt requires input values to be contiguous,
zero-offset, and of a specific data type. If a Tensor argument is a non-trivial
view or has been bitcast from the original data type, then TorchTPU will make a
temporary copy of the tensor in eager mode (either strict or optimized) to
correctly call the compiled executable.

To prevent this temporary copy, ensure that all inputs to a compiled function
are contiguous. If you plan to call multiple compiled functions in sequence,
ensure that the output of earlier functions are contiguous to prevent eager mode
friction between the optimized functions.

### Optimized eager mode: retaining redundant tensors

In all eager modes, operations will be executed in the order they are created,
and whenever a materialization signal is raised, all currently-live Tensors at
that point will be materialized.

If any of these tensors are unimportant intermediate tensors, this can cause the
defer-and-fuse mechanism to falsely identify them as necessary outputs and
materialize them, potentially reducing the efficiency of the compilation graph.

To improve fusion and minimize memory usage, be sure to clean up tensors as soon
as they are no longer needed, or use in-place operations to replace obsolete
tensors with current values.

## Takeaways

TorchTPU's deferred execution model offers efficient pipelining at a range of
compile time / runtime tradeoffs.

*   **For minimum compile time:** use strict eager mode (the default). This mode
    begins immediate execution and parallelizes compilation to achieve the
    fastest possible time-to-first-result.
*   **For maximum warmed-up throughput and minimal latency:** use
    `torch.compile`. This eliminates almost all eager mode overhead and
    maximizes compiler optimization potential.
*   **For cases where faster runtime is preferred but `torch.compile` is not an
    option:** consider using optimized eager mode.
*   **For debugging:** use launch-blocking mode to force the CPU and TPU to move
    in lockstep.
*   **Avoid forced synchronizations:** hold a minimal set of contiguous tensors,
    and use computationally-dense operations to keep the TPU well-supplied with
    productive work to do.
