# torch-tpu advice for AI Agents

For BUILD or bzl files:

*   Act as an expert in Bazel, Starlark, rules_python, bazel_skylib, rules_cc
*   Wrap code to 80 columns

When updating Python requirements:

*   Do not modify the locked, resolved `requirements_3_*.txt` files directly.
*   Instead, add the dependency to the appropriate list in `pyproject.toml`.
*   Place core requirements in the main `dependencies` block, and use-case
    specific requirements in the correct `[project.optional-dependencies]` group
    (e.g., `dev`, `test`, etc).
*   Run `./requirements/lock_environments.sh` to update all of the locked
    requirements files automatically.

Instructions related to Bazel repositories:

*   Run `bazel query //external:all` to list all external repositories.
*   Run `bazel query //external:<name>` to list a single one.
*   Add `--output=build` to show the url (and thus version) of the repository
    being used.
*   Only define a repository in WORKSPACE if it doesn't exist, or errors
    indicate a newer version must be used.

When creating a pull request:

*   Build the body from `.github/pull_request_template.md`. Keep its `##
    Description` and `## Test Plan` headings and drop the HTML comment
    placeholders. `gh pr create --body` does not apply the template.
