# TorchTPU Documentation

This directory contains the TorchTPU documentation.

## Build locally *(for generating the static HTML files)*

To build the documentation locally, follow these steps:

1.  From the root of the repository, install the documentation dependencies
    listed in `docs/requirements.txt`:

    ```bash
    python -m pip install -r docs/requirements.txt
    ```

2.  From the root of the repo, run the following command:

```bash
mkdocs build
```

3.  The generated HTML files will be located in the `site` directory. Open
    `index.html` in your browser to view the documentation.

## Serve locally *(recommended for local previews)*

To serve the documentation locally, follow these steps:

1.  Install the documentation dependencies from the root of the repository:

```bash
python -m pip install -r docs/requirements.txt
```

2.  From the root of the repository, run the following command:

```bash
mkdocs serve
```

3.  Open your browser and navigate to `http://127.0.0.1:8000/` to view the
    documentation locally.

## Sources for the documentation

The documentation is built from the markdown sources located in the `docs/`
directory. These files serve as the single source of truth for TorchTPU
documentation and tutorials. Jupyter notebooks (`.ipynb`) are generated from
these files through [Jupytext](https://github.com/mwouts/jupytext). Plain
markdown content will be converted into markdown cells, and code blocks will be
converted into executable code cells in the notebooks. After editing one of
these sources, regenerate the notebooks from the root of the repository:

```bash
tools/generate_doc_notebooks.py
```

To run this step on every commit, install the
[pre-commit](https://pre-commit.com) hook once:

```bash
python -m pip install pre-commit
pre-commit install
```

When you commit a change to one of these sources, the hook regenerates its
notebook. If the notebook changed, the commit stops so you can stage the
notebook and commit again.
