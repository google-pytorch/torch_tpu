"""Logit lens: decode each layer's hidden state through the output head."""
import torch


@torch.no_grad()
def logit_lens(model, input_ids: torch.Tensor, position: int = -1, top_k: int = 1):
    """Returns, per layer, the top_k token ids predicted at `position`.

    Needs a causal LM that supports output_hidden_states and get_output_embeddings().
    Applies the final norm when the base model exposes one (common in HF decoders).
    """
    out = model(input_ids=input_ids[None], output_hidden_states=True)
    head = model.get_output_embeddings()
    norm = getattr(getattr(model, "base_model", model), "norm", None)
    rows = []
    for h in out.hidden_states:
        h = h[:, position]
        if norm is not None:
            h = norm(h)
        rows.append(head(h)[0].topk(top_k).indices.tolist())
    return rows
