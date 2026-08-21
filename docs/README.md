# TorchTPU Documentation

The documentation site at http://google-pytorch.github.io/torch_tpu/ is built
with [Jupyter Book](https://jupyterbook.org) from the notebooks in
`docs/notebooks/` and deployed by `.github/workflows/docs.yaml` on every push
to `main`. Nothing generated is committed: the HTML is built in CI.

## Authoring model

Every page exists as a [jupytext](https://jupytext.readthedocs.io) pair:

*   `docs/notebooks/<page>.md` — MyST markdown, the **source of truth**. This
    is what you edit and what gets reviewed in PRs.
*   `docs/notebooks/<page>.ipynb` — the paired Jupyter notebook, generated
    from the markdown and committed **with outputs**. This is what the site is
    rendered from, so readers see real TPU outputs. It is marked
    `linguist-generated` in `.gitattributes`, so its diffs are collapsed in
    PRs.

The pairing is configured in `docs/notebooks/jupytext.toml`. CI checks on
every PR that the two files are in sync; it does not (and cannot) execute the
notebooks, because TPU hardware is required.

## Editing a page

1.  Edit `docs/notebooks/<page>.md`.
2.  Regenerate the paired notebook. If the code cells changed, execute it on a
    TPU VM (a v6e GCE instance is easiest) so the committed outputs stay real:

    ```bash
    pip install -r docs/requirements.txt
    jupytext --sync docs/notebooks/<page>.md            # update the .ipynb
    jupyter nbconvert --to notebook --execute --inplace \
        docs/notebooks/<page>.ipynb                     # refresh outputs (TPU VM)
    ```

    For prose-only changes, `jupytext --sync` alone is enough — it updates the
    notebook's text while preserving the existing outputs.
3.  Commit both files.

To add a new page, create the `.md` file (copy an existing header), run the
steps above, and add the page to `docs/_toc.yml` and to the index page
(`docs/notebooks/index.md`).

## Building the site locally

```bash
pip install -r docs/requirements.txt
jupyter-book build docs/
open docs/_build/html/index.html
```

The build renders the committed `.ipynb` files without executing them
(`execute_notebooks: "off"` in `docs/_config.yml`), so it works on any
machine.
