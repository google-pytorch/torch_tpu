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

//! Loader for `METADATA` presubmit configuration.
//!
//! Only `presubmit { check_contents { ... } }` entries become rules. Other
//! presubmit checks (description checks, services, lint, ...) are reported as
//! skipped. Field semantics follow `devtools/metadata/presubmit.proto`.

use std::fmt;
use std::path::{Path, PathBuf};

use regex::Regex;

use crate::model::{FileOp, MatchMode, Origin, PathFilter, Rule, RuleSet, Severity, Skipped};
use crate::paths::{anchor, is_rooted, path_expression_to_regex, resolve};
use crate::textproto::{self, Message};

/// An error loading a rule file, with its location when known.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LoadError {
    pub file: PathBuf,
    pub line: Option<usize>,
    pub message: String,
}

impl LoadError {
    pub fn new(file: &Path, line: Option<usize>, message: impl Into<String>) -> Self {
        Self { file: file.to_path_buf(), line, message: message.into() }
    }
}

impl fmt::Display for LoadError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.line {
            Some(line) => write!(f, "{}:{}: {}", self.file.display(), line, self.message),
            None => write!(f, "{}: {}", self.file.display(), self.message),
        }
    }
}

impl std::error::Error for LoadError {}

/// Reads and parses a `METADATA` rule file. Relative paths are resolved
/// against `base_dir`.
pub fn load_file(path: &Path, base_dir: &str) -> Result<RuleSet, LoadError> {
    let text = std::fs::read_to_string(path)
        .map_err(|e| LoadError::new(path, None, format!("cannot read rules file: {e}")))?;
    parse_metadata(&text, path, base_dir)
}

/// Parses METADATA `text`. Relative paths are resolved against `base_dir`.
pub fn parse_metadata(text: &str, file: &Path, base_dir: &str) -> Result<RuleSet, LoadError> {
    let root = textproto::parse(text)
        .map_err(|e| LoadError::new(file, Some(e.line), format!("syntax error at column {}: {}", e.column, e.message)))?;

    let origin = |line| Origin { file: file.to_path_buf(), line };
    let mut set = RuleSet::default();

    for (presubmit, presubmit_line) in root.messages("presubmit") {
        // Compile path filters lazily: blocks without content checks may use
        // patterns we never need to evaluate.
        let mut filter: Option<PathFilter> = None;

        for field in &presubmit.fields {
            let name = field.name.as_str();
            match (&field.value, name) {
                (textproto::Value::Message(check), "check_contents") => {
                    if filter.is_none() {
                        filter = Some(
                            build_filter(presubmit, base_dir)
                                .map_err(|m| LoadError::new(file, Some(presubmit_line), m))?,
                        );
                    }
                    let paths = filter.clone().expect("filter was just built");
                    let block_ops = operations(presubmit, "file_operation")
                        .map_err(|m| LoadError::new(file, Some(presubmit_line), m))?;
                    match build_rule(check, paths, &block_ops, origin(field.line))? {
                        Built::Rule(rule) => set.rules.push(rule),
                        Built::Skipped(skipped) => set.skipped.push(skipped),
                    }
                }
                (textproto::Value::Message(check), kind) if kind.starts_with("check_") => {
                    set.skipped.push(Skipped {
                        kind: kind.to_string(),
                        id: check.message("base").and_then(|b| b.string("id")).map(str::to_string),
                        origin: origin(field.line),
                    });
                }
                (textproto::Value::Str(target), "include_presubmit") => {
                    set.skipped.push(Skipped {
                        kind: "include_presubmit".to_string(),
                        id: Some(target.clone()),
                        origin: origin(field.line),
                    });
                }
                _ => {}
            }
        }
    }
    Ok(set)
}

enum Built {
    Rule(Rule),
    Skipped(Skipped),
}

fn compile(field: &str, pattern: &str) -> Result<Regex, String> {
    Regex::new(pattern).map_err(|e| format!("invalid regex in `{field}` ({pattern:?}): {e}"))
}

fn build_filter(presubmit: &Message, base_dir: &str) -> Result<PathFilter, String> {
    let expression = |field: &str, e: &str| compile(field, &anchor(&path_expression_to_regex(&resolve(base_dir, e))));
    let regexp = |field: &str, r: &str| {
        let full = if is_rooted(base_dir, r) {
            r.to_string()
        } else {
            format!("{}/{}", regex::escape(base_dir.trim_end_matches('/')), r)
        };
        compile(field, &anchor(&full))
    };

    let mut include = Vec::new();
    for e in presubmit.strings("path_expression") {
        include.push(expression("path_expression", e)?);
    }
    for r in presubmit.strings("path_regexp") {
        include.push(regexp("path_regexp", r)?);
    }
    if include.is_empty() {
        include.push(expression("path_expression", "...")?);
    }

    let mut exclude = Vec::new();
    for e in presubmit.strings("path_expression_exclusion") {
        exclude.push(expression("path_expression_exclusion", e)?);
    }
    for r in presubmit.strings("path_regexp_exclusion") {
        exclude.push(regexp("path_regexp_exclusion", r)?);
    }
    Ok(PathFilter { include, exclude })
}

/// Parses the repeated `FileOperation` enum field `name` of `message`.
fn operations(message: &Message, name: &str) -> Result<Vec<FileOp>, String> {
    message
        .idents(name)
        .map(|op| FileOp::parse(op).ok_or_else(|| format!("unknown `{name}` `{op}`")))
        .collect()
}

fn build_rule(check: &Message, paths: PathFilter, block_ops: &[FileOp], origin: Origin) -> Result<Built, LoadError> {
    let base = check.message("base");
    let id = base
        .and_then(|b| b.string("id"))
        .map(str::to_string)
        .unwrap_or_else(|| format!("CheckContents@{}", origin.line));
    let fail = |message: String| LoadError::new(&origin.file, Some(origin.line), format!("{id}: {message}"));
    let flag = |name: &str| check.bool(name).map(|v| v.unwrap_or(false)).map_err(|m| fail(m));

    if flag("check_deleted_only")? {
        return Ok(Built::Skipped(Skipped {
            kind: "check_contents.check_deleted_only".to_string(),
            id: Some(id),
            origin,
        }));
    }

    let optional_regex = |field: &str| -> Result<Option<Regex>, LoadError> {
        check.string(field).map(|p| compile(field, p)).transpose().map_err(|m| fail(m))
    };
    let prohibited = optional_regex("prohibited_regexp")?;
    let required = optional_regex("required_regexp")?;
    let suppression = optional_regex("suppression_regexp")?;
    if prohibited.is_none() && required.is_none() {
        return Err(fail("check_contents needs `prohibited_regexp` or `required_regexp`".into()));
    }

    let delta_only = flag("check_delta_only")?;
    let entire_file = flag("match_entire_file")?;
    let multi_line = flag("match_multi_line")?;
    if delta_only && entire_file {
        return Err(fail("`check_delta_only` is incompatible with `match_entire_file`".into()));
    }
    let mode = if entire_file {
        MatchMode::EntireFile
    } else if delta_only && multi_line {
        MatchMode::ModifiedChunks
    } else {
        // `match_multi_line` is documented to apply only with `check_delta_only`.
        MatchMode::Line
    };

    let severity = match check.ident("failure_status") {
        None => Severity::Error,
        Some(status) => {
            Severity::parse(status).ok_or_else(|| fail(format!("unknown `failure_status` `{status}`")))?
        }
    };

    let message = check
        .string("failure_message")
        .map(str::to_string)
        .unwrap_or_else(|| format!("Check {id} failed."));

    let check_ops = operations(check, "operation").map_err(|m| fail(m))?;
    let operations = if check_ops.is_empty() { block_ops.to_vec() } else { check_ops };

    Ok(Built::Rule(Rule {
        paths,
        prohibited,
        required,
        suppression,
        delta_only,
        mode,
        ignore_large_files: flag("ignore_large_files")? || flag("match_entire_file_ignore_large_files")?,
        severity,
        message,
        disable_tags: base.map(|b| b.strings("disable_tags").map(str::to_string).collect()).unwrap_or_default(),
        operations,
        id,
        origin,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    const BASE: &str = "//depot/repo/proj";

    fn load(text: &str) -> RuleSet {
        parse_metadata(text, Path::new("METADATA"), BASE).unwrap()
    }

    fn load_err(text: &str) -> LoadError {
        parse_metadata(text, Path::new("METADATA"), BASE).unwrap_err()
    }

    #[test]
    fn load_file_reports_missing_file() {
        let err = load_file(Path::new("/nonexistent/METADATA"), "//r").unwrap_err();
        assert!(err.message.contains("cannot read rules file"), "{err}");
    }

    #[test]
    fn parses_a_basic_prohibited_rule() {
        let set = load(
            r#"
            presubmit {
              path_regexp: ".*\\.py"
              path_expression_exclusion: "tests/..."
              check_contents {
                base { id: "CheckEnv" disable_tags: "ROLLBACK_OF" disable_tags: "ENV_OK" }
                failure_status: WARNING
                failure_message: "no " "env"
                prohibited_regexp: "\\bos\\.environ\\b"
                suppression_regexp: "ENV_OK=.+"
                check_delta_only: true
              }
            }"#,
        );
        assert_eq!(set.rules.len(), 1);
        let r = &set.rules[0];
        assert_eq!(r.id, "CheckEnv");
        assert_eq!(r.severity, Severity::Warning);
        assert_eq!(r.message, "no env");
        assert_eq!(r.disable_tags, ["ROLLBACK_OF", "ENV_OK"]);
        assert!(r.delta_only);
        assert_eq!(r.mode, MatchMode::Line);
        assert!(r.prohibited.as_ref().unwrap().is_match("x = os.environ['A']"));
        assert!(r.suppression.as_ref().unwrap().is_match("# ENV_OK=reason"));
        assert!(r.required.is_none());
        assert_eq!(r.origin.line, 5);

        assert!(r.paths.matches("//depot/repo/proj/a/b.py"));
        assert!(!r.paths.matches("//depot/repo/proj/tests/b.py"));
        assert!(!r.paths.matches("//depot/repo/proj/a/b.cc"));
        assert!(!r.paths.matches("//depot/repo/other/a.py"));
    }

    #[test]
    fn default_scope_is_everything_under_base() {
        let set = load(r#"presubmit { check_contents { prohibited_regexp: "x" } }"#);
        let r = &set.rules[0];
        assert!(r.paths.matches("//depot/repo/proj/any/file.txt"));
        assert!(!r.paths.matches("//depot/repo/projx/file.txt"));
        assert_eq!(r.severity, Severity::Error);
        assert_eq!(r.id, "CheckContents@1");
        assert_eq!(r.message, "Check CheckContents@1 failed.");
    }

    #[test]
    fn absolute_paths_are_not_prefixed() {
        let set = load(
            r#"presubmit {
                 path_regexp: "//depot/repo/proj/.*\\.(cc|h)"
                 path_regexp_exclusion: ".*_test\\.cc"
                 path_expression_exclusion: "//depot/.../*.METADATA"
                 check_contents { prohibited_regexp: "x" }
               }"#,
        );
        let p = &set.rules[0].paths;
        assert!(p.matches("//depot/repo/proj/a/b.cc"));
        assert!(!p.matches("//depot/repo/proj/a/b_test.cc"));
        assert!(!p.matches("//depot/repo/proj/a.METADATA"));
    }

    #[test]
    fn path_regexp_is_a_full_match() {
        let set = load(r#"presubmit { path_regexp: "foo" check_contents { prohibited_regexp: "x" } }"#);
        let p = &set.rules[0].paths;
        assert!(p.matches("//depot/repo/proj/foo"));
        assert!(!p.matches("//depot/repo/proj/foo.txt"));
        assert!(!p.matches("//depot/repo/proj/a/foo"));
    }

    #[test]
    fn relative_base_dir_is_escaped_in_regexps() {
        // A literal '.' in the base dir must not act as a wildcard.
        let set = parse_metadata(
            r#"presubmit { path_regexp: "a" check_contents { prohibited_regexp: "x" } }"#,
            Path::new("METADATA"),
            "//r/v1.0",
        )
        .unwrap();
        let p = &set.rules[0].paths;
        assert!(p.matches("//r/v1.0/a"));
        assert!(!p.matches("//r/v1x0/a"));
    }

    #[test]
    fn match_modes() {
        let set = load(
            r#"presubmit {
                 check_contents { base { id: "A" } required_regexp: "x" match_entire_file: true ignore_large_files: true }
                 check_contents { base { id: "B" } prohibited_regexp: "x" check_delta_only: true match_multi_line: true }
                 check_contents { base { id: "C" } prohibited_regexp: "x" match_multi_line: true }
               }"#,
        );
        let modes: Vec<_> = set.rules.iter().map(|r| (r.id.as_str(), r.mode)).collect();
        assert_eq!(
            modes,
            [("A", MatchMode::EntireFile), ("B", MatchMode::ModifiedChunks), ("C", MatchMode::Line)]
        );
        assert!(set.rules[0].ignore_large_files);
        assert!(!set.rules[1].ignore_large_files);
    }

    #[test]
    fn multiple_checks_share_the_block_scope() {
        let set = load(
            r#"presubmit {
                 path_regexp: ".*\\.cc"
                 check_contents { base { id: "A" } prohibited_regexp: "a" }
                 check_contents { base { id: "B" } prohibited_regexp: "b" }
               }"#,
        );
        assert_eq!(set.rules.len(), 2);
        assert!(set.rules.iter().all(|r| r.paths.matches("//depot/repo/proj/x.cc")));
        assert!(set.rules.iter().all(|r| !r.paths.matches("//depot/repo/proj/x.py")));
    }

    #[test]
    fn unsupported_checks_are_reported_as_skipped() {
        let set = load(
            r#"presubmit {
                 check_description { base { id: "Desc" } required_regexp: "x" }
                 check_presubmit_service { base { id: "Svc" } extended_options { [a.b] { c: 1 } } }
                 include_presubmit: "//depot/repo/x/y.METADATA"
                 check_contents { base { id: "Del" } prohibited_regexp: "x" check_deleted_only: true }
               }
               presubmit { check_lint { action: REVIEW } }"#,
        );
        assert!(set.rules.is_empty());
        let kinds: Vec<_> = set.skipped.iter().map(|s| (s.kind.as_str(), s.id.as_deref())).collect();
        assert_eq!(
            kinds,
            [
                ("check_description", Some("Desc")),
                ("check_presubmit_service", Some("Svc")),
                ("include_presubmit", Some("//depot/repo/x/y.METADATA")),
                ("check_contents.check_deleted_only", Some("Del")),
                ("check_lint", None),
            ]
        );
    }

    #[test]
    fn blocks_without_content_checks_do_not_compile_path_regexps() {
        // An RE2-only construct in an unrelated block must not fail loading.
        let set = load(r#"presubmit { path_regexp: "(?<name>x)\\C" check_lint { } }"#);
        assert!(set.rules.is_empty());
    }

    #[test]
    fn errors_point_at_the_offending_check() {
        let err = load_err("presubmit {\n  check_contents {\n    base { id: \"Bad\" }\n    prohibited_regexp: \"(\"\n  }\n}");
        assert_eq!(err.line, Some(2));
        assert!(err.message.starts_with("Bad: invalid regex in `prohibited_regexp`"), "{err}");

        let err = load_err(r#"presubmit { check_contents { failure_message: "m" } }"#);
        assert!(err.message.contains("needs `prohibited_regexp` or `required_regexp`"), "{err}");

        let err = load_err(
            r#"presubmit { check_contents { prohibited_regexp: "x" check_delta_only: true match_entire_file: true } }"#,
        );
        assert!(err.message.contains("incompatible"), "{err}");

        let err = load_err(r#"presubmit { check_contents { prohibited_regexp: "x" failure_status: FATAL } }"#);
        assert!(err.message.contains("unknown `failure_status` `FATAL`"), "{err}");

        let err = load_err("presubmit {\n  path_regexp: \"(\"\n  check_contents { prohibited_regexp: \"x\" }\n}");
        assert_eq!(err.line, Some(1));
        assert!(err.message.contains("path_regexp"), "{err}");

        let err = load_err("presubmit { check_contents { prohibited_regexp: \"x\" }");
        assert!(err.message.starts_with("syntax error"), "{err}");

        let err = load_err(r#"presubmit { check_contents { prohibited_regexp: "x" operation: MOVE } }"#);
        assert!(err.message.contains("unknown `operation` `MOVE`"), "{err}");
    }

    #[test]
    fn operations_come_from_the_check_or_the_block() {
        let set = load(
            r#"
            presubmit {
              file_operation: ADD
              file_operation: BRANCH
              check_contents { base { id: "Block" } prohibited_regexp: "x" }
              check_contents { base { id: "Own" } prohibited_regexp: "x" operation: EDIT }
            }
            presubmit { check_contents { base { id: "All" } prohibited_regexp: "x" } }"#,
        );
        let ops: Vec<_> = set.rules.iter().map(|r| (r.id.as_str(), r.operations.clone())).collect();
        assert_eq!(
            ops,
            [("Block", vec![FileOp::Add, FileOp::Branch]), ("Own", vec![FileOp::Edit]), ("All", vec![])]
        );
        let own = &set.rules[1];
        assert!(own.applies_to(Some(FileOp::Edit)));
        assert!(!own.applies_to(Some(FileOp::Add)));
        assert!(!own.applies_to(None), "no change context");
        assert!(set.rules[2].applies_to(None));
    }
}
