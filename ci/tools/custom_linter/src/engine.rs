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

//! Rule evaluation. Files are filtered by path before being read, and are
//! processed in parallel. The engine returns violations; it never prints.

use std::collections::BTreeSet;
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

use crate::model::{FileOp, MatchMode, Rule, Violation};

/// Files larger than this are not matched in entire-file mode.
pub const MAX_ENTIRE_FILE_BYTES: usize = 1 << 20;

/// Bytes inspected to decide whether a file is binary.
const BINARY_SNIFF_BYTES: usize = 8000;

/// A file to check.
#[derive(Debug, Clone)]
pub struct FileTask {
    /// Project-relative path, used in reports.
    pub rel_path: String,
    /// Canonical path, used for rule path matching.
    pub canonical: String,
    /// Location on disk.
    pub abs_path: PathBuf,
    /// Lines added or modified by the change; `None` means every line counts
    /// as changed (explicit files, `--all`, or a newly added file).
    pub changed_lines: Option<BTreeSet<usize>>,
    /// How the change touched the file; `None` without a change context.
    pub op: Option<FileOp>,
}

/// Aggregate results of a run.
#[derive(Debug, Default)]
pub struct Outcome {
    pub violations: Vec<Violation>,
    /// Files that matched at least one rule and were read.
    pub files_checked: usize,
}

/// Checks `tasks` against `rules` using up to `jobs` threads.
///
/// Returns an error listing unreadable files; partial results are discarded
/// because an incomplete run must not look like a clean one.
pub fn run(rules: &[&Rule], tasks: &[FileTask], jobs: usize) -> Result<Outcome, String> {
    let next = AtomicUsize::new(0);
    let results = Mutex::new((Vec::new(), 0usize, Vec::new()));
    let workers = jobs.max(1).min(tasks.len().max(1));

    std::thread::scope(|scope| {
        for _ in 0..workers {
            scope.spawn(|| {
                let mut violations = Vec::new();
                let mut checked = 0usize;
                let mut errors = Vec::new();
                loop {
                    let i = next.fetch_add(1, Ordering::Relaxed);
                    let Some(task) = tasks.get(i) else { break };
                    match check_task(rules, task) {
                        Ok(None) => {}
                        Ok(Some(found)) => {
                            checked += 1;
                            violations.extend(found);
                        }
                        Err(e) => errors.push(e),
                    }
                }
                let mut shared = results.lock().unwrap();
                shared.0.extend(violations);
                shared.1 += checked;
                shared.2.extend(errors);
            });
        }
    });

    let (mut violations, files_checked, mut errors) = results.into_inner().unwrap();
    if !errors.is_empty() {
        errors.sort();
        return Err(errors.join("\n"));
    }
    violations.sort_by(|a, b| (&a.path, a.line, &a.rule_id).cmp(&(&b.path, b.line, &b.rule_id)));
    Ok(Outcome { violations, files_checked })
}

/// Returns `Ok(None)` if no rule applies to the file (it is not read).
fn check_task(rules: &[&Rule], task: &FileTask) -> Result<Option<Vec<Violation>>, String> {
    let applicable: Vec<&Rule> =
        rules.iter().copied().filter(|r| r.applies_to(task.op) && r.paths.matches(&task.canonical)).collect();
    if applicable.is_empty() {
        return Ok(None);
    }
    let bytes = std::fs::read(&task.abs_path).map_err(|e| format!("cannot read {}: {e}", task.abs_path.display()))?;
    if bytes[..bytes.len().min(BINARY_SNIFF_BYTES)].contains(&0) {
        return Ok(Some(Vec::new())); // Content checks only apply to text files.
    }
    let text = String::from_utf8_lossy(&bytes);
    let file = TextFile::new(&text);
    let mut violations = Vec::new();
    for rule in applicable {
        violations.extend(check_file(rule, &file, task.changed_lines.as_ref()).into_iter().map(|(line, message)| {
            Violation {
                rule_id: rule.id.clone(),
                severity: rule.severity,
                path: task.rel_path.clone(),
                line,
                message: message.unwrap_or_else(|| rule.message.clone()),
            }
        }));
    }
    Ok(Some(violations))
}

/// A file's text with precomputed line boundaries.
pub struct TextFile<'a> {
    text: &'a str,
    /// Byte offset where each line starts; line N (1-based) starts at
    /// `line_starts[N - 1]`.
    line_starts: Vec<usize>,
}

impl<'a> TextFile<'a> {
    pub fn new(text: &'a str) -> Self {
        let mut line_starts = vec![0];
        line_starts.extend(text.match_indices('\n').map(|(i, _)| i + 1).filter(|&i| i < text.len()));
        Self { text, line_starts }
    }

    pub fn line_count(&self) -> usize {
        if self.text.is_empty() {
            0
        } else {
            self.line_starts.len()
        }
    }

    /// Text of 1-based line `n`, without the line terminator.
    pub fn line(&self, n: usize) -> &'a str {
        let line = &self.text[self.range(n, n)];
        // `range` excludes the `\n`; the last line may still end with one.
        let line = line.strip_suffix('\n').unwrap_or(line);
        line.strip_suffix('\r').unwrap_or(line)
    }

    /// 1-based line containing byte `offset`.
    pub fn line_of(&self, offset: usize) -> usize {
        self.line_starts.partition_point(|&start| start <= offset)
    }

    /// Byte range covering lines `first..=last`, without the final newline.
    fn range(&self, first: usize, last: usize) -> std::ops::Range<usize> {
        let start = self.line_starts[first - 1];
        let end = self.line_starts.get(last).map_or(self.text.len(), |&next| next - 1);
        start..end.max(start)
    }
}

/// A finding: 1-based line (if any) and an optional message override.
type Finding = (Option<usize>, Option<String>);

/// Evaluates one rule against one file.
pub fn check_file(rule: &Rule, file: &TextFile<'_>, changed: Option<&BTreeSet<usize>>) -> Vec<Finding> {
    let mut findings = Vec::new();
    // When the rule is not delta-only, every line of a touched file is checked.
    let changed = if rule.delta_only { changed } else { None };

    match rule.mode {
        MatchMode::EntireFile => {
            if file.text.len() > MAX_ENTIRE_FILE_BYTES {
                if !rule.ignore_large_files {
                    findings.push((
                        None,
                        Some(format!(
                            "File is larger than {MAX_ENTIRE_FILE_BYTES} bytes and cannot be matched by {}.",
                            rule.id
                        )),
                    ));
                }
                return findings;
            }
            if let Some(prohibited) = &rule.prohibited {
                let suppressed = rule.suppression.as_ref().is_some_and(|s| s.is_match(file.text));
                if !suppressed {
                    findings.extend(prohibited.find_iter(file.text).map(|m| (Some(file.line_of(m.start())), None)));
                }
            }
            if let Some(required) = &rule.required {
                if !required.is_match(file.text) {
                    findings.push((None, None));
                }
            }
        }
        MatchMode::Line => {
            if let Some(prohibited) = &rule.prohibited {
                let mut check = |n: usize| {
                    let line = file.line(n);
                    if prohibited.is_match(line) && !rule.suppression.as_ref().is_some_and(|s| s.is_match(line)) {
                        findings.push((Some(n), None));
                    }
                };
                match changed {
                    Some(lines) => lines.iter().copied().filter(|&n| n >= 1 && n <= file.line_count()).for_each(&mut check),
                    None => (1..=file.line_count()).for_each(&mut check),
                }
            }
            check_required_by_line(rule, file, &mut findings);
        }
        MatchMode::ModifiedChunks => {
            if let Some(prohibited) = &rule.prohibited {
                for (first, last) in chunks(changed, file.line_count()) {
                    let range = file.range(first, last);
                    let chunk = &file.text[range.clone()];
                    if rule.suppression.as_ref().is_some_and(|s| s.is_match(chunk)) {
                        continue;
                    }
                    findings
                        .extend(prohibited.find_iter(chunk).map(|m| (Some(file.line_of(range.start + m.start())), None)));
                }
            }
            check_required_by_line(rule, file, &mut findings);
        }
    }
    findings
}

fn check_required_by_line(rule: &Rule, file: &TextFile<'_>, findings: &mut Vec<Finding>) {
    if let Some(required) = &rule.required {
        if !(1..=file.line_count()).any(|n| required.is_match(file.line(n))) {
            findings.push((None, None));
        }
    }
}

/// Groups line numbers into runs of consecutive lines, clamped to the file.
fn chunks(changed: Option<&BTreeSet<usize>>, line_count: usize) -> Vec<(usize, usize)> {
    if line_count == 0 {
        return Vec::new();
    }
    let Some(lines) = changed else {
        return vec![(1, line_count)];
    };
    let mut out: Vec<(usize, usize)> = Vec::new();
    for &n in lines.iter().filter(|&&n| n >= 1 && n <= line_count) {
        match out.last_mut() {
            Some((_, last)) if *last + 1 == n => *last = n,
            _ => out.push((n, n)),
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{Origin, PathFilter, Severity};
    use regex::Regex;

    fn rule(mode: MatchMode, prohibited: Option<&str>, required: Option<&str>, suppression: Option<&str>) -> Rule {
        Rule {
            id: "R".into(),
            origin: Origin { file: "M".into(), line: 1 },
            paths: PathFilter { include: vec![Regex::new("^//p/.*\\.cc$").unwrap()], exclude: vec![] },
            prohibited: prohibited.map(|p| Regex::new(p).unwrap()),
            required: required.map(|p| Regex::new(p).unwrap()),
            suppression: suppression.map(|p| Regex::new(p).unwrap()),
            delta_only: false,
            mode,
            ignore_large_files: false,
            severity: Severity::Error,
            message: "msg".into(),
            disable_tags: vec![],
            operations: vec![],
        }
    }

    fn lines_of(findings: &[Finding]) -> Vec<Option<usize>> {
        findings.iter().map(|f| f.0).collect()
    }

    fn set(v: &[usize]) -> BTreeSet<usize> {
        v.iter().copied().collect()
    }

    #[test]
    fn text_file_lines() {
        let f = TextFile::new("a\r\nb\n\nc");
        assert_eq!(f.line_count(), 4);
        assert_eq!([f.line(1), f.line(2), f.line(3), f.line(4)], ["a", "b", "", "c"]);
        assert_eq!(f.line_of(0), 1);
        assert_eq!(f.line_of(3), 2);
        assert_eq!(f.line_of(7), 4);

        let f = TextFile::new("x\ny\n");
        assert_eq!(f.line_count(), 2);
        assert_eq!(f.line(2), "y");
        assert_eq!(TextFile::new("").line_count(), 0);
    }

    #[test]
    fn line_mode_prohibited_with_same_line_suppression() {
        let r = rule(MatchMode::Line, Some(r"\bbad\b"), None, Some("OK=.+"));
        let f = TextFile::new("good\nbad\nbad // OK=reason\nbad");
        assert_eq!(lines_of(&check_file(&r, &f, None)), [Some(2), Some(4)]);
    }

    #[test]
    fn delta_only_checks_changed_lines_only() {
        let mut r = rule(MatchMode::Line, Some("bad"), None, None);
        let f = TextFile::new("bad\nbad\nbad");
        r.delta_only = true;
        assert_eq!(lines_of(&check_file(&r, &f, Some(&set(&[2, 99])))), [Some(2)]);
        assert_eq!(lines_of(&check_file(&r, &f, None)), [Some(1), Some(2), Some(3)]);
        // Non-delta rules ignore the changed-line set.
        r.delta_only = false;
        assert_eq!(check_file(&r, &f, Some(&set(&[2]))).len(), 3);
    }

    #[test]
    fn required_in_line_mode_needs_one_matching_line() {
        let r = rule(MatchMode::Line, None, Some("^TT_KERNEL$"), None);
        assert!(check_file(&r, &TextFile::new("x\nTT_KERNEL\n"), None).is_empty());
        assert_eq!(lines_of(&check_file(&r, &TextFile::new("x TT_KERNEL\n"), None)), [None]);
    }

    #[test]
    fn entire_file_mode() {
        let r = rule(MatchMode::EntireFile, None, Some(r"Copyright \d{4}(.|\n)*Apache"), None);
        assert!(check_file(&r, &TextFile::new("// Copyright 2026\n// Apache\n"), None).is_empty());
        assert_eq!(lines_of(&check_file(&r, &TextFile::new("// Apache\n"), None)), [None]);

        let r = rule(MatchMode::EntireFile, Some(r"^(\s|\n)*#include"), None, None);
        assert_eq!(lines_of(&check_file(&r, &TextFile::new("\n\n#include <x>\n"), None)), [Some(1)]);

        // A suppression anywhere in the file suppresses everything.
        let r = rule(MatchMode::EntireFile, Some("bad"), None, Some("NOLINT_FILE"));
        let f = TextFile::new("bad\nx\nNOLINT_FILE\n");
        assert!(check_file(&r, &f, None).is_empty());
    }

    #[test]
    fn entire_file_rejects_large_files_unless_ignored() {
        let big = "a".repeat(MAX_ENTIRE_FILE_BYTES + 1);
        let mut r = rule(MatchMode::EntireFile, Some("zzz"), None, None);
        let found = check_file(&r, &TextFile::new(&big), None);
        assert_eq!(found.len(), 1);
        assert!(found[0].1.as_ref().unwrap().contains("larger than"));
        r.ignore_large_files = true;
        assert!(check_file(&r, &TextFile::new(&big), None).is_empty());
    }

    #[test]
    fn modified_chunks_match_across_lines_within_a_chunk() {
        let mut r = rule(MatchMode::ModifiedChunks, Some(r"(?s)foo\(\s*bar"), None, Some("OK_CHUNK"));
        r.delta_only = true;
        let f = TextFile::new("foo(\nbar)\nx\nfoo(\nbar) OK_CHUNK\nx\nfoo(\nx\nbar)");
        // Lines 1-2 form a chunk with a match; lines 4-5 are suppressed;
        // lines 7 and 9 are separate chunks, so no match spans them.
        assert_eq!(lines_of(&check_file(&r, &f, Some(&set(&[1, 2, 4, 5, 7, 9])))), [Some(1)]);
        // Without a changed set, the whole file is one chunk.
        assert_eq!(check_file(&r, &f, None).len(), 0, "suppression anywhere in the single chunk wins");
    }

    #[test]
    fn chunk_grouping() {
        assert_eq!(chunks(Some(&set(&[1, 2, 4, 5, 6, 9, 50])), 10), [(1, 2), (4, 6), (9, 9)]);
        assert_eq!(chunks(None, 3), [(1, 3)]);
        assert!(chunks(None, 0).is_empty());
    }

    #[test]
    fn run_filters_by_path_before_reading_and_sorts_results() {
        let dir = std::env::var_os("TEST_TMPDIR").map(PathBuf::from).unwrap_or_else(std::env::temp_dir);
        let dir = dir.join(format!("custom_linter_engine_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("b.cc"), "bad\n").unwrap();
        std::fs::write(dir.join("a.cc"), "ok\nbad\n").unwrap();
        std::fs::write(dir.join("bin.cc"), b"bad\0bad").unwrap();

        let r = rule(MatchMode::Line, Some("bad"), None, None);
        let task = |name: &str| FileTask {
            rel_path: name.into(),
            canonical: format!("//p/{name}"),
            abs_path: dir.join(name),
            changed_lines: None,
            op: None,
        };
        let tasks = vec![
            task("b.cc"),
            task("a.cc"),
            task("bin.cc"),
            // Does not match the rule's path filter, so it is never read.
            FileTask { canonical: "//p/x.py".into(), abs_path: dir.join("missing.py"), ..task("x.py") },
        ];
        let outcome = run(&[&r], &tasks, 4).unwrap();
        let found: Vec<_> = outcome.violations.iter().map(|v| (v.path.as_str(), v.line)).collect();
        assert_eq!(found, [("a.cc", Some(2)), ("b.cc", Some(1))]);
        assert_eq!(outcome.files_checked, 3);

        // A matching but unreadable file is an error, not a silent pass.
        let err = run(&[&r], &[task("missing.cc")], 2).unwrap_err();
        assert!(err.contains("cannot read"), "{err}");

        // Operation-restricted rules only apply to files changed that way.
        let mut edit_only = rule(MatchMode::Line, Some("bad"), None, None);
        edit_only.operations = vec![FileOp::Edit];
        let with_op = |op| FileTask { op, ..task("b.cc") };
        assert_eq!(run(&[&edit_only], &[with_op(Some(FileOp::Edit))], 1).unwrap().violations.len(), 1);
        assert!(run(&[&edit_only], &[with_op(Some(FileOp::Add))], 1).unwrap().violations.is_empty());
        assert!(run(&[&edit_only], &[with_op(None)], 1).unwrap().violations.is_empty());
    }
}
