import html


def to_html(tokens, scores) -> str:
    """Red = pushes toward the target, blue = pushes away. Works inline in Colab."""
    scores = [float(s) for s in scores]
    m = max((abs(s) for s in scores), default=0.0) or 1.0
    spans = []
    for t, s in zip(tokens, scores):
        a = abs(s) / m
        color = f"rgba(220,50,47,{a:.2f})" if s >= 0 else f"rgba(38,139,210,{a:.2f})"
        spans.append(f'<span title="{s:+.4f}" style="background:{color};padding:2px;margin:1px;'
                     f'border-radius:3px">{html.escape(t)}</span>')
    return "<div style='font-family:monospace;line-height:2'>" + "".join(spans) + "</div>"
