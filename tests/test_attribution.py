import torch
import torch.nn as nn
from types import SimpleNamespace

from torchtpu_xai import integrated_gradients, grad_x_input, to_html


class TinyLM(nn.Module):
    """Minimal stand-in exposing the HF interface xai needs."""
    def __init__(self, v=11, d=8):
        super().__init__()
        torch.manual_seed(0)
        self.emb = nn.Embedding(v, d)
        self.mix = nn.Sequential(nn.Linear(d, d), nn.Tanh())
        self.head = nn.Linear(d, v)

    def get_input_embeddings(self):
        return self.emb

    def forward(self, input_ids=None, inputs_embeds=None, attention_mask=None):
        h = inputs_embeds if inputs_embeds is not None else self.emb(input_ids)
        h = self.mix(h).cumsum(1)  # causal-ish mixing over time
        return SimpleNamespace(logits=self.head(h))


def test_ig_completeness():
    m = TinyLM().eval()
    ids = torch.tensor([3, 5, 7, 2])
    a = integrated_gradients(m, ids, target_id=4, steps=256)
    assert a.scores.shape == (4,)
    assert a.convergence_delta < 5e-3


def test_ig_single_batched_call_matches_shape_only():
    m = TinyLM().eval()
    a = integrated_gradients(m, torch.tensor([1, 2, 3]), target_id=0, steps=8)
    assert torch.isfinite(a.scores).all()


def test_grad_x_input_shape():
    m = TinyLM().eval()
    assert grad_x_input(m, torch.tensor([1, 2, 3]), 0).shape == (3,)


def test_render_escapes():
    assert "&lt;b&gt;" in to_html(["<b>"], [1.0])
