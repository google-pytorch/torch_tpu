"""Which candidate ops actually fail on TPU (main wheel)? Run in env_fix."""
import torch, torch_tpu
tpu = torch.device("tpu")
a = torch.randn(4, 4, device=tpu); b = torch.randn(3, 4, 5, device=tpu); c = torch.randn(3, 5, 6, device=tpu)
n = torch.tensor([[1.0, float("nan")], [2.0, 3.0]], device=tpu)
x = torch.randn(6, device=tpu); y = torch.randn(6, device=tpu); o = torch.empty(6, device=tpu)
cases = {
  "trace": lambda: torch.trace(a),
  "triu_indices": lambda: torch.triu_indices(4, 4, device=tpu),
  "tril_indices": lambda: torch.tril_indices(4, 4, device=tpu),
  "nansum": lambda: torch.nansum(n),
  "nansum.dim": lambda: torch.nansum(n, dim=1),
  "huber_loss": lambda: torch.nn.functional.huber_loss(x, y),
  "smooth_l1_loss": lambda: torch.nn.functional.smooth_l1_loss(x, y),
  "smooth_l1_loss.out": lambda: torch.ops.aten.smooth_l1_loss.out(x, y, 1, 1.0, out=torch.empty((), device=tpu)),
  "std_mean": lambda: torch.std_mean(a),
  "std_mean.dim": lambda: torch.std_mean(a, dim=0),
  "var_mean": lambda: torch.var_mean(a, dim=0),
  "addbmm": lambda: torch.addbmm(torch.zeros(4, 6, device=tpu), b, c),
  "median": lambda: torch.median(x),
  "median.dim": lambda: torch.median(a, dim=1),
  "nanmedian": lambda: torch.nanmedian(x),
  "mode": lambda: torch.mode(a, dim=1),
  "kthvalue": lambda: torch.kthvalue(x, 2),
  "logaddexp": lambda: torch.logaddexp(x, y),
  "logaddexp.out": lambda: torch.logaddexp(x, y, out=o),
  "hypot.out": lambda: torch.hypot(x, y, out=o),
  "copysign.out": lambda: torch.copysign(x, y, out=o),
  "erfc.out": lambda: torch.erfc(x, out=o),
  "sinc.out": lambda: torch.sinc(x, out=o),
  "mish.out": lambda: torch.ops.aten.mish.out(x, out=o),
  "index_select.out": lambda: torch.index_select(a, 0, torch.tensor([0, 2], device=tpu), out=torch.empty(2, 4, device=tpu)),
  "heaviside.out": lambda: torch.heaviside(x, y, out=o),
  "nextafter.out": lambda: torch.nextafter(x, y, out=o),
  "range.out": lambda: torch.range(0, 5, out=o),
  "logspace.out": lambda: torch.logspace(0, 1, 6, out=o),
  "frexp": lambda: torch.frexp(x),
  "renorm": lambda: torch.renorm(a, 2, 0, 1.0),
  "hardshrink.out": lambda: torch.ops.aten.hardshrink.out(x, 0.5, out=o),
  "softshrink.out": lambda: torch.ops.aten.softshrink.out(x, 0.5, out=o),
  "i0.out": lambda: torch.i0(x, out=o),
  "gcd.out": lambda: torch.gcd(torch.tensor([4, 6], device=tpu), torch.tensor([6, 9], device=tpu), out=torch.empty(2, dtype=torch.long, device=tpu)),
  "multi_margin_loss": lambda: torch.nn.functional.multi_margin_loss(a, torch.tensor([0, 1, 2, 3], device=tpu)),
  "multilabel_margin_loss": lambda: torch.nn.functional.multilabel_margin_loss(a, torch.tensor([[0, -1, 0, 0]] * 4, device=tpu)),
  "avg_pool1d": lambda: torch.nn.functional.avg_pool1d(b, 2),
  "constant_pad_nd": lambda: torch.nn.functional.pad(a, (1, 1)),
  "narrow_copy.out": lambda: torch.narrow_copy(a, 0, 0, 2, out=torch.empty(2, 4, device=tpu)),
}
ok, bad = [], []
for name, fn in cases.items():
    try:
        r = fn(); r = r[0] if isinstance(r, tuple) else r; r.cpu()
        ok.append(name)
    except Exception as e:
        msg = str(e).splitlines()[0][:110]
        bad.append((name, msg))
print("OK:", " ".join(ok))
for name, msg in bad: print("FAIL", name, "->", msg)
