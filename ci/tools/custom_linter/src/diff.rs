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

//! Parser for git-style unified diffs (`diff --git`, as produced by git,
//! `hg diff --git` and `jj diff --git`), reduced to the files touched and the
//! line numbers added or modified in the new version of each file.

use std::collections::BTreeSet;

/// One file touched by a change.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FileDiff {
    /// Path of the new version, relative to the VCS root.
    pub path: String,
    pub deleted: bool,
    /// The file did not exist before (and is not a copy or rename).
    pub added: bool,
    /// The file is a copy or rename of another file.
    pub copied: bool,
    /// 1-based line numbers added or modified in the new version.
    pub added_lines: BTreeSet<usize>,
}

impl FileDiff {
    fn new(path: String) -> Self {
        Self { path, deleted: false, added: false, copied: false, added_lines: BTreeSet::new() }
    }
}

/// Parses a git-style diff. Context lines are ignored, so any `-U` value works,
/// but `-U0` is cheapest.
pub fn parse(diff: &str) -> Result<Vec<FileDiff>, String> {
    let mut files: Vec<FileDiff> = Vec::new();
    let mut in_hunk = false;

    for (index, line) in diff.lines().enumerate() {
        let lineno = index + 1;
        if let Some(header) = line.strip_prefix("diff --git ") {
            let path = header_new_path(header)
                .ok_or_else(|| format!("line {lineno}: cannot parse diff header `{line}`"))?;
            files.push(FileDiff::new(path));
            in_hunk = false;
            continue;
        }
        let Some(current) = files.last_mut() else {
            continue; // Preamble before the first file (e.g. commit metadata).
        };
        if line.starts_with("@@") {
            let start = parse_hunk_new_range(line)
                .ok_or_else(|| format!("line {lineno}: cannot parse hunk header `{line}`"))?;
            current.added_lines.extend(start);
            in_hunk = true;
            continue;
        }
        if in_hunk && (line.starts_with('+') || line.starts_with('-') || line.starts_with(' ') || line.starts_with('\\')) {
            continue;
        }
        in_hunk = false;
        if line == "+++ /dev/null" || line.starts_with("deleted file mode") {
            current.deleted = true;
        } else if line == "--- /dev/null" || line.starts_with("new file mode") {
            current.added = true;
        } else if let Some(rest) = line.strip_prefix("+++ ") {
            if let Some(path) = unquote(rest).strip_prefix("b/") {
                current.path = path.to_string();
            }
        } else if let Some(path) = line.strip_prefix("rename to ").or_else(|| line.strip_prefix("copy to ")) {
            current.path = unquote(path);
            current.copied = true;
        }
    }
    Ok(files)
}

/// Extracts the `b/...` path from `a/<old> b/<new>`.
fn header_new_path(header: &str) -> Option<String> {
    if header.starts_with('"') {
        // Quoted paths: `"a/x y" "b/x y"`.
        let second = header.rfind(" \"")?;
        let new = unquote(&header[second + 1..]);
        return new.strip_prefix("b/").map(str::to_string);
    }
    let split = header.rfind(" b/")?;
    Some(header[split + 3..].to_string())
}

/// Removes C-style quoting that git applies to unusual paths.
fn unquote(path: &str) -> String {
    let Some(inner) = path.strip_prefix('"').and_then(|p| p.strip_suffix('"')) else {
        return path.split('\t').next().unwrap_or(path).to_string();
    };
    let mut out = Vec::new();
    let bytes = inner.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'\\' && i + 1 < bytes.len() {
            i += 1;
            match bytes[i] {
                b'n' => out.push(b'\n'),
                b't' => out.push(b'\t'),
                b'"' => out.push(b'"'),
                b'\\' => out.push(b'\\'),
                d @ b'0'..=b'7' => {
                    let mut value = u32::from(d - b'0');
                    for _ in 0..2 {
                        if let Some(&n @ b'0'..=b'7') = bytes.get(i + 1) {
                            value = value * 8 + u32::from(n - b'0');
                            i += 1;
                        }
                    }
                    out.push(value as u8);
                }
                other => out.push(other),
            }
        } else {
            out.push(bytes[i]);
        }
        i += 1;
    }
    String::from_utf8_lossy(&out).into_owned()
}

/// Returns the new-file line range of a hunk header `@@ -a,b +c,d @@`.
fn parse_hunk_new_range(line: &str) -> Option<std::ops::Range<usize>> {
    let plus = line.split_whitespace().find(|t| t.starts_with('+'))?;
    let spec = &plus[1..];
    let (start, count) = match spec.split_once(',') {
        Some((s, c)) => (s.parse::<usize>().ok()?, c.parse::<usize>().ok()?),
        None => (spec.parse::<usize>().ok()?, 1),
    };
    Some(start..start + count)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lines(v: &[usize]) -> BTreeSet<usize> {
        v.iter().copied().collect()
    }

    #[test]
    fn parses_modified_added_and_deleted_files() {
        let diff = "\
diff --git a/repo/p/BUILD b/repo/p/BUILD
--- a/repo/p/BUILD
+++ b/repo/p/BUILD
@@ -43,0 +44,3 @@ exports_files([\"LICENSE\"])
+a
+b
+c
@@ -50 +53 @@
-old
+new
diff --git a/new.py b/new.py
new file mode 100644
--- /dev/null
+++ b/new.py
@@ -0,0 +1,2 @@
+x
+y
diff --git a/gone.cc b/gone.cc
deleted file mode 100644
--- a/gone.cc
+++ /dev/null
@@ -1,2 +0,0 @@
-a
-b
";
        let files = parse(diff).unwrap();
        assert_eq!(
            files,
            [
                FileDiff { added_lines: lines(&[44, 45, 46, 53]), ..FileDiff::new("repo/p/BUILD".into()) },
                FileDiff { added: true, added_lines: lines(&[1, 2]), ..FileDiff::new("new.py".into()) },
                FileDiff { deleted: true, ..FileDiff::new("gone.cc".into()) },
            ]
        );
    }

    #[test]
    fn pure_deletion_hunks_add_no_lines() {
        let diff = "diff --git a/x b/x\n--- a/x\n+++ b/x\n@@ -3,2 +2,0 @@\n-a\n-b\n";
        assert_eq!(parse(diff).unwrap()[0].added_lines, lines(&[]));
    }

    #[test]
    fn renames_and_copies_use_the_new_path_and_only_changed_lines() {
        let diff = "\
diff --git a/old.py b/new.py
similarity index 90%
rename from old.py
rename to new.py
--- a/old.py
+++ b/new.py
@@ -5 +5 @@
-x
+y
diff --git a/src.cc b/dst.cc
copy from src.cc
copy to dst.cc
";
        let files = parse(diff).unwrap();
        assert_eq!(files[0].path, "new.py");
        assert_eq!(files[0].added_lines, lines(&[5]));
        assert_eq!(files[1].path, "dst.cc");
        assert!(files[1].added_lines.is_empty(), "a clean copy has no new lines");
        assert!(files.iter().all(|f| f.copied && !f.added));
    }

    #[test]
    fn hunk_body_lines_that_look_like_headers_are_ignored() {
        // A removed line whose content is "-- a/x" must not be taken as a header.
        let diff = "diff --git a/x b/x\n--- a/x\n+++ b/x\n@@ -1 +1 @@\n--- a/y\n+++ b/z\n";
        let files = parse(diff).unwrap();
        assert_eq!(files.len(), 1);
        assert_eq!(files[0].path, "x");
    }

    #[test]
    fn mode_changes_and_binary_files_have_no_lines() {
        let diff = "\
diff --git a/run.sh b/run.sh
old mode 100644
new mode 100755
diff --git a/img.png b/img.png
Binary files a/img.png and b/img.png differ
";
        let files = parse(diff).unwrap();
        assert_eq!(files.len(), 2);
        assert!(files.iter().all(|f| f.added_lines.is_empty() && !f.deleted));
    }

    #[test]
    fn paths_with_spaces_and_quotes() {
        let diff = "diff --git a/my dir/f.py b/my dir/f.py\n--- a/my dir/f.py\n+++ b/my dir/f.py\n@@ -1 +1 @@\n+x\n\
                    diff --git \"a/t\\303\\251.py\" \"b/t\\303\\251.py\"\n@@ -0,0 +1 @@\n+x\n";
        let files = parse(diff).unwrap();
        assert_eq!(files[0].path, "my dir/f.py");
        assert_eq!(files[1].path, "té.py");
    }

    #[test]
    fn ignores_preamble_and_reports_bad_hunks() {
        assert!(parse("commit abc\nAuthor: x\n").unwrap().is_empty());
        let err = parse("diff --git a/x b/x\n@@ garbage @@\n").unwrap_err();
        assert!(err.contains("cannot parse hunk header"), "{err}");
    }
}
