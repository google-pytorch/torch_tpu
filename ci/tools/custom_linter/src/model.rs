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

//! Format-independent rule model shared by all rule loaders and the engine.

use std::fmt;
use std::path::PathBuf;

use regex::Regex;

/// How severe a violation of a rule is.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Severity {
    Notice,
    Warning,
    Error,
}

impl Severity {
    /// Parses a presubmit status name (`ERROR`, `WARNING`, `NOTICE`).
    pub fn parse(name: &str) -> Option<Self> {
        match name {
            "ERROR" => Some(Self::Error),
            "WARNING" => Some(Self::Warning),
            "NOTICE" => Some(Self::Notice),
            _ => None,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Error => "ERROR",
            Self::Warning => "WARNING",
            Self::Notice => "NOTICE",
        }
    }
}

impl fmt::Display for Severity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// The unit of text that regexes are evaluated against.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MatchMode {
    /// Each line separately; suppressions must be on the same line.
    Line,
    /// The whole file at once; suppressions may appear anywhere in the file.
    EntireFile,
    /// Runs of consecutive modified lines, joined into one chunk. Only valid
    /// together with delta-only checking.
    ModifiedChunks,
}

impl MatchMode {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Line => "line",
            Self::EntireFile => "entire-file",
            Self::ModifiedChunks => "modified-chunks",
        }
    }
}

/// Include/exclude filters over canonical paths. Exclusions always win.
#[derive(Debug, Clone)]
pub struct PathFilter {
    /// Anchored regexes; a path is included if any of them matches.
    pub include: Vec<Regex>,
    /// Anchored regexes; a path is excluded if any of them matches.
    pub exclude: Vec<Regex>,
}

impl PathFilter {
    pub fn matches(&self, canonical_path: &str) -> bool {
        self.include.iter().any(|r| r.is_match(canonical_path))
            && !self.exclude.iter().any(|r| r.is_match(canonical_path))
    }
}

/// Where a rule was defined, for diagnostics.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Origin {
    pub file: PathBuf,
    pub line: usize,
}

impl fmt::Display for Origin {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}:{}", self.file.display(), self.line)
    }
}

/// A content rule: a regex that must (or must not) appear in matching files.
#[derive(Debug, Clone)]
pub struct Rule {
    pub id: String,
    pub origin: Origin,
    pub paths: PathFilter,
    /// Pattern that must not appear. At least one of `prohibited` and
    /// `required` is set.
    pub prohibited: Option<Regex>,
    /// Pattern that must appear at least once in the file.
    pub required: Option<Regex>,
    /// Pattern that, when found, suppresses a `prohibited` match.
    pub suppression: Option<Regex>,
    /// Only check lines added or modified by the change under review.
    pub delta_only: bool,
    pub mode: MatchMode,
    /// Skip (instead of failing on) files too large for entire-file matching.
    pub ignore_large_files: bool,
    pub severity: Severity,
    pub message: String,
    /// Change-description tags (`TAG=...`) that disable this rule.
    pub disable_tags: Vec<String>,
    /// File operations the rule applies to; empty means all operations.
    pub operations: Vec<FileOp>,
}

impl Rule {
    /// Whether the rule applies to a file changed by `op`. `None` means there
    /// is no change context (explicit files, `--all`); rules restricted to
    /// specific operations are about changes, so they do not apply.
    pub fn applies_to(&self, op: Option<FileOp>) -> bool {
        self.operations.is_empty() || op.is_some_and(|op| self.operations.contains(&op))
    }
}

/// How a change touched a file (`FileOperation` in `presubmit.proto`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FileOp {
    /// A new file that is not a clean copy of another file.
    Add,
    /// A modification of an existing file.
    Edit,
    Delete,
    /// A clean copy (or rename) of another file, not modified afterwards.
    Branch,
    Integrate,
}

impl FileOp {
    pub fn parse(name: &str) -> Option<Self> {
        match name {
            "ADD" => Some(Self::Add),
            "EDIT" => Some(Self::Edit),
            "DELETE" => Some(Self::Delete),
            "BRANCH" => Some(Self::Branch),
            "INTEGRATE" => Some(Self::Integrate),
            _ => None,
        }
    }
}

/// A check found in a rule file that this tool does not implement.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Skipped {
    /// The kind of check, e.g. `check_description`.
    pub kind: String,
    pub id: Option<String>,
    pub origin: Origin,
}

/// The result of loading one or more rule files.
#[derive(Debug, Clone, Default)]
pub struct RuleSet {
    pub rules: Vec<Rule>,
    /// Entries that were recognized but are not supported, reported so that
    /// users are not misled into thinking they are enforced.
    pub skipped: Vec<Skipped>,
}

impl RuleSet {
    pub fn extend(&mut self, other: RuleSet) {
        for rule in other.rules {
            if !self.rules.iter().any(|r| r.id == rule.id) {
                self.rules.push(rule);
            }
        }
        self.skipped.extend(other.skipped);
    }
}

/// A single rule violation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Violation {
    pub rule_id: String,
    pub severity: Severity,
    /// Project-relative path of the offending file.
    pub path: String,
    /// 1-based line of the match; `None` for whole-file findings such as a
    /// missing required pattern.
    pub line: Option<usize>,
    pub message: String,
}
