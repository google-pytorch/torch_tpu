// Copyright 2026 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! Path handling: mapping between repository-relative paths and the canonical
//! "depot" paths that rules are written against, and path-expression globs.
//!
//! Rules are always matched against a canonical, slash-separated path such as
//! `//depot/repo/project/foo/bar.cc`. The caller supplies the canonical path
//! that corresponds to the project root; nothing is inferred from the
//! environment.

use std::path::{Component, Path};

/// Maps project-relative paths to canonical rule paths.
///
/// By default a path `a/b` maps to `<canonical_root>/a/b`. Prefix maps
/// override this for subtrees whose layout differs from the canonical one,
/// e.g. an OSS export where Copybara moved `pkg/` to `src/pkg/`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PathMapper {
    /// Canonical path of the project root, without a trailing slash.
    canonical_root: String,
    /// `(relative prefix, canonical prefix)`, longest relative prefix first.
    prefix_maps: Vec<(String, String)>,
}

impl PathMapper {
    /// Creates a mapper. `canonical_root` is e.g. `//depot/repo/project`.
    pub fn new(canonical_root: &str) -> Self {
        Self { canonical_root: canonical_root.trim_end_matches('/').to_string(), prefix_maps: Vec::new() }
    }

    /// Adds prefix maps `(relative, canonical)`. A relative path equal to or
    /// under `relative` is mapped under `canonical`; the longest match wins.
    pub fn with_prefix_maps<I: IntoIterator<Item = (String, String)>>(mut self, maps: I) -> Self {
        self.prefix_maps.extend(
            maps.into_iter()
                .map(|(r, c)| (r.trim_matches('/').to_string(), c.trim_end_matches('/').to_string())),
        );
        self.prefix_maps.sort_by(|a, b| b.0.len().cmp(&a.0.len()));
        self
    }

    pub fn canonical_root(&self) -> &str {
        &self.canonical_root
    }

    /// Converts a project-relative path (`foo/bar.cc`) to a canonical path.
    ///
    /// Returns `None` if the path is absolute or escapes the root via `..`.
    pub fn to_canonical(&self, relative: &Path) -> Option<String> {
        let mut parts: Vec<&str> = Vec::new();
        for component in relative.components() {
            match component {
                Component::Normal(p) => parts.push(p.to_str()?),
                Component::CurDir => {}
                Component::ParentDir | Component::RootDir | Component::Prefix(_) => return None,
            }
        }
        if parts.is_empty() {
            return None;
        }
        let rel = parts.join("/");
        for (prefix, canonical) in &self.prefix_maps {
            if prefix.is_empty() {
                return Some(format!("{canonical}/{rel}"));
            }
            if rel == *prefix {
                return Some(canonical.clone());
            }
            if let Some(rest) = rel.strip_prefix(prefix.as_str()).and_then(|r| r.strip_prefix('/')) {
                return Some(format!("{canonical}/{rest}"));
            }
        }
        Some(format!("{}/{rel}", self.canonical_root))
    }
}

/// Returns whether `path` is already rooted (starts with `//`, `.../`, or
/// `base_dir`) rather than relative to `base_dir`.
pub fn is_rooted(base_dir: &str, path: &str) -> bool {
    let base = base_dir.trim_end_matches('/');
    path.starts_with("//")
        || path.starts_with(".../")
        || (!base.is_empty()
            && path.strip_prefix(base).is_some_and(|rest| {
                rest.is_empty() || rest.starts_with('/') || rest.starts_with('.')
            }))
}

/// Resolves a rule path against the directory that declares it: rooted paths
/// (starting with `//`, `.../`, or `base_dir`) are kept, relative ones are
/// prefixed.
pub fn resolve(base_dir: &str, path: &str) -> String {
    if is_rooted(base_dir, path) {
        path.to_string()
    } else {
        format!("{}/{}", base_dir.trim_end_matches('/'), path)
    }
}

/// Converts a depot-style path expression to an (unanchored) regex.
///
/// Supported wildcards: `...` matches any sequence of characters including
/// `/`; `*` matches any sequence without `/`. Everything else is literal.
pub fn path_expression_to_regex(expression: &str) -> String {
    let mut out = String::with_capacity(expression.len() * 2);
    let mut rest = expression;
    while !rest.is_empty() {
        if let Some(r) = rest.strip_prefix("...") {
            out.push_str(".*");
            rest = r;
        } else if let Some(r) = rest.strip_prefix('*') {
            out.push_str("[^/]*");
            rest = r;
        } else {
            let c = rest.chars().next().unwrap();
            out.push_str(&regex::escape(c.encode_utf8(&mut [0u8; 4])));
            rest = &rest[c.len_utf8()..];
        }
    }
    out
}

/// Wraps a regex so that it must match the entire input.
pub fn anchor(regex: &str) -> String {
    format!("^(?:{regex})$")
}

#[cfg(test)]
mod tests {
    use super::*;
    use regex::Regex;

    fn expr_matches(expr: &str, path: &str) -> bool {
        Regex::new(&anchor(&path_expression_to_regex(expr))).unwrap().is_match(path)
    }

    #[test]
    fn mapper_prefixes_the_root() {
        let m = PathMapper::new("//depot/repo/proj/");
        assert_eq!(m.canonical_root(), "//depot/repo/proj");
        assert_eq!(m.to_canonical(Path::new("./a/b.cc")).unwrap(), "//depot/repo/proj/a/b.cc");
    }

    #[test]
    fn prefix_maps_undo_export_moves_longest_first() {
        // Mirrors TorchTPU's OSS layout: the repo root is `oss/`,
        // and the Python package lives under `src/torch_tpu/`.
        let g3 = "//depot/repo/tt";
        let m = PathMapper::new(&format!("{g3}/oss")).with_prefix_maps([
            ("src/torch_tpu".to_string(), g3.to_string()),
            ("src/torch_tpu/_internal/special/".to_string(), format!("{g3}/oss/torch_tpu/special")),
            ("/csrc/".to_string(), format!("{g3}/csrc/")),
        ]);
        let c = |p: &str| m.to_canonical(Path::new(p)).unwrap();
        assert_eq!(c("src/torch_tpu/_internal/env.py"), format!("{g3}/_internal/env.py"));
        assert_eq!(c("src/torch_tpu/_internal/special/x.py"), format!("{g3}/oss/torch_tpu/special/x.py"));
        assert_eq!(c("csrc/ops/add.cc"), format!("{g3}/csrc/ops/add.cc"));
        assert_eq!(c("csrc"), format!("{g3}/csrc"));
        assert_eq!(c("ci/tools/run.py"), format!("{g3}/oss/ci/tools/run.py"));
        // Prefixes match whole components only.
        assert_eq!(c("csrcx/a.cc"), format!("{g3}/oss/csrcx/a.cc"));
    }

    #[test]
    fn mapper_rejects_escaping_paths() {
        let m = PathMapper::new("//r");
        assert_eq!(m.to_canonical(Path::new("../x")), None);
        assert_eq!(m.to_canonical(Path::new("/abs/x")), None);
        assert_eq!(m.to_canonical(Path::new("")), None);
    }

    #[test]
    fn mapper_supports_plain_roots() {
        // Outside a monorepo the canonical root can be anything, e.g. empty-ish.
        let m = PathMapper::new("//repo");
        assert_eq!(m.to_canonical(Path::new("src/x.py")).unwrap(), "//repo/src/x.py");
    }

    #[test]
    fn resolve_relative_and_absolute() {
        assert_eq!(resolve("//depot/p", "a/..."), "//depot/p/a/...");
        assert_eq!(resolve("//depot/p/", ".*\\.py"), "//depot/p/.*\\.py");
        assert_eq!(resolve("//depot/p", "//depot/q/x"), "//depot/q/x");
        assert_eq!(resolve("torch_tpu", "torch_tpu/csrc/..."), "torch_tpu/csrc/...");
        assert_eq!(resolve("torch_tpu", "torch_tpu.*/BUILD"), "torch_tpu.*/BUILD");
        assert_eq!(resolve("torch_tpu", ".../*_test.py"), ".../*_test.py");
        assert_eq!(resolve("torch_tpu", "....py"), "torch_tpu/....py");
    }

    #[test]
    fn ellipsis_matches_across_directories() {
        assert!(expr_matches("//depot/p/...", "//depot/p/a/b/c.cc"));
        assert!(expr_matches("//depot/.../METADATA", "//depot/p/x/METADATA"));
        assert!(!expr_matches("//depot/.../METADATA", "//depot/p/x/presubmit.METADATA"));
    }

    #[test]
    fn ellipsis_semantics_match_presubmit_docs() {
        // From presubmit.proto: `.../foo` includes project/foo_dir/foo but not project/foo.
        assert!(expr_matches("//depot/repo/project/.../foo", "//depot/repo/project/foo_dir/foo"));
        assert!(!expr_matches("//depot/repo/project/.../foo", "//depot/repo/project/foo"));
    }

    #[test]
    fn star_does_not_cross_directories() {
        assert!(expr_matches("//depot/.../*_test.py", "//depot/p/a/x_test.py"));
        assert!(expr_matches("//depot/.../*.METADATA", "//depot/p/presubmit.METADATA"));
        assert!(!expr_matches("//depot/p/*.py", "//depot/p/a/x.py"));
        assert!(expr_matches("//depot/p/*.py", "//depot/p/x.py"));
    }

    #[test]
    fn dots_and_regex_chars_are_literal() {
        assert!(expr_matches("//depot/p/copy.bara.sky", "//depot/p/copy.bara.sky"));
        assert!(!expr_matches("//depot/p/copy.bara.sky", "//depot/p/copyXbaraXsky"));
        assert!(expr_matches("//depot/p/a+b(c).txt", "//depot/p/a+b(c).txt"));
    }
}
