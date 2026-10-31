# TorchTPU GitHub work dashboard

An interactive page listing `google-pytorch/torch_tpu` issues and pull requests,
for triaging work as an Oncaller or tracking your own work as an Individual
Contributor. It's published at
`https://google-pytorch.github.io/torch_tpu/developers/dashboard/`.

## Preview changes

The dashboard is static HTML, CSS, and JavaScript with no build step. To preview
changes before they are merged, serve the directory and open
`http://<host>:8765/` in your browser, where `<host>` is the address of the
machine serving it (`localhost` if it's the same machine):

```bash
python3 -m http.server 8765 --directory docs/developers/dashboard
```

## Publish

`.github/workflows/dashboard_pages.yml` publishes the dashboard to GitHub Pages
whenever a change to `docs/developers/dashboard/` reaches `main`. To publish on
demand, run the **Deploy dashboard to GitHub Pages** workflow from the GitHub
Actions tab, or run:

```bash
gh workflow run dashboard_pages.yml --repo google-pytorch/torch_tpu --ref main
```

## Add or change views, filters, and presets

Edit `views.js`. Each filter appears as a drop-down and as a result column; each
preset is a one-click set of filter values for a common task, and a view starts
with its first preset.

## How it works

-   **Search**: The page queries the GitHub Search API from your browser and
    loads up to 200 results per search. Results are reused for 2 minutes; click
    **Refresh** to search again.
-   **Rate limit**: Without a token, GitHub allows 10 searches per minute per IP
    address. Paste an optional read-only token under **GitHub token** to raise
    this to 30 per minute; it's kept only in the open tab's memory.
-   **Priority**: Issue priority comes from the `Priority` organization issue
    field.
