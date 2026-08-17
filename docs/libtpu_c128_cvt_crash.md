# libtpu: LOG(FATAL) in XPrecisionRewriter on any convert into complex128

`xprecision_rewriter.cc:2100`'s `Unsupported CVT X{64|128} expansion` check
aborts the whole client process whenever a compiled HLO module contains a
`convert` whose result type is `c128` — it is a `LOG(FATAL)`, not an
`Unimplemented` status, so the embedding process (here: PyTorch running on
TorchTPU) dies with SIGABRT instead of receiving a compile error.

## Impact

Running the upstream PyTorch device-generic test suites (pytorch v2.13.0)
against TorchTPU on TPU v7: **137 of 324 test shard processes aborted on
this one check** (2026-08-21 baseline), discarding every test queued behind
the crash — the op_db suites (`test_ops` 33.8k cases, `test_meta` 41.8k,
`test_unary_ufuncs` 25.6k) report almost nothing. Observed conversion
sources: `f32`, `f64`, `c64`, `s64`, `pred` — any scalar or array shape,
always with `c128` as the destination.

```
F0821 19:54:37 xprecision_rewriter.cc:2100] Unsupported CVT X{64|128} expansion from f32[] to c128[]
F0821 ... xprecision_rewriter.cc:2100] Unsupported CVT X{64|128} expansion from c64[] to c128[]
F0821 ... xprecision_rewriter.cc:2100] Unsupported CVT X{64|128} expansion from f64[2,2] to c128[2,2]
F0821 ... xprecision_rewriter.cc:2100] Unsupported CVT X{64|128} expansion from pred[5,5] to c128[5,5]
```

Related fatals in the same pass observed in the same runs (17 shards):
`xprecision_rewriter.cc` `Check failed: operand->shape().IsTuple()` and
`Check failed: N == hlo->shape().tuple_shapes().size()`.

## Minimal repro

torch 2.13.0+cpu · torch_tpu dev build · libtpu 0.0.41 · TPU v7 (1 chip).

```python
import torch
import torch_tpu  # registers the `tpu` PrivateUse1 backend

x = torch.ones((), device="tpu")   # f32[] on TPU
y = x.to(torch.complex128)         # HLO convert f32[] -> c128[]
print(y.cpu())                     # compile + execute => SIGABRT (exit 134)
```

## Stack (abridged)

```
F xprecision_rewriter.cc:2100] Unsupported CVT X{64|128} expansion from f32[] to c128[]
  xla::jellyfish::(anonymous namespace)::XPrecisionRewriteVisitor::Builder::Cvt()
  xla::jellyfish::(anonymous namespace)::XPrecisionRewriteVisitor::HandleConvert()
  xla::HloInstruction::Accept<>()
  xla::jellyfish::XPrecisionRewriter::RunImpl()
  xla::HloPassPipeline::RunPassesInternal<>()
  xla::jellyfish::DeepseaCompilerBase::RunHloPasses()
  xla::jellyfish::DeepseaCompilerBase::CompileModule()
  xla::jellyfish::TpuJitCompileHloWithOptions()
  xla::TpuCompiler::Compile()
  xla::PjRtCompile()  ->  pjrt::PJRT_Client_Compile()  (via PJRT C API)
  torch_tpu::Compile() / CompilationCache
```

## Ask

1. Implement the CVT-into-c128 expansion, or
2. return `Unimplemented` from the pass instead of `LOG(FATAL)` so PJRT
   clients get a catchable compile error (a Python `RuntimeError` for
   PyTorch users) rather than a process abort.

Until then TorchTPU's upstream-test harness excludes complex128 wholesale
(`tests/torch_test_configs.py`, `_CRASHING_DTYPES`).
