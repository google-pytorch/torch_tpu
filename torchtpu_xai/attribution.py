"""Token attribution designed for TPUs.

Every call runs fixed-shape tensor programs: all integrated-gradients steps go through
the model as ONE batch ([steps, T, D]), so a compiled graph is traced once per
(steps, T) and reused, instead of looping in Python with a different graph per step.
"""
from dataclasses import dataclass
from typing import Callable, Optional

import torch


@dataclass
class Attribution:
    scores: torch.Tensor      # [T] per-token attribution (summed over embedding dim)
    target_value: float       # f(input)
    baseline_value: float     # f(baseline)
    convergence_delta: float  # |sum(scores) - (f(x) - f(baseline))|; ~0 when steps suffice


def _default_target(target_id: int) -> Callable[[torch.Tensor], torch.Tensor]:
    # logits [B, T, V] or [B, C] -> [B]
    def f(logits):
        return logits[:, -1, target_id] if logits.dim() == 3 else logits[:, target_id]
    return f


def _embed_forward(model, embeds, attention_mask=None):
    kw = {"inputs_embeds": embeds}
    if attention_mask is not None:
        kw["attention_mask"] = attention_mask
    return model(**kw).logits


def integrated_gradients(
    model,
    input_ids: torch.Tensor,            # [T] single sequence
    target_id: int,
    steps: int = 32,
    baseline_id: int = 0,
    target_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    attention_mask: Optional[torch.Tensor] = None,
) -> Attribution:
    target_fn = target_fn or _default_target(target_id)
    emb = model.get_input_embeddings()
    x = emb(input_ids).detach()                                     # [T, D]
    base = emb(torch.full_like(input_ids, baseline_id)).detach()    # [T, D]

    alphas = torch.linspace(0.0, 1.0, steps + 1, device=x.device, dtype=x.dtype)[1:]  # right Riemann
    path = (base + alphas[:, None, None] * (x - base)).requires_grad_(True)  # [S, T, D]
    mask = attention_mask.expand(steps, -1) if attention_mask is not None else None

    with torch.enable_grad():
        out = target_fn(_embed_forward(model, path, mask))          # [S]
        (grad,) = torch.autograd.grad(out.sum(), path)              # [S, T, D]

    scores = ((x - base) * grad.mean(0)).sum(-1)                    # [T]
    with torch.no_grad():
        fx = target_fn(_embed_forward(model, x[None])).item()
        fb = target_fn(_embed_forward(model, base[None])).item()
    return Attribution(scores.detach(), fx, fb, abs(scores.sum().item() - (fx - fb)))


def grad_x_input(model, input_ids, target_id, target_fn=None) -> torch.Tensor:
    target_fn = target_fn or _default_target(target_id)
    x = model.get_input_embeddings()(input_ids).detach()[None].requires_grad_(True)
    with torch.enable_grad():
        (g,) = torch.autograd.grad(target_fn(_embed_forward(model, x)).sum(), x)
    return (g * x).sum(-1)[0].detach()
