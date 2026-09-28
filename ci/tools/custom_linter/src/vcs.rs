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

//! Version-control backends (`git`, `hg`, `jj`).
//!
//! Every backend produces a git-style diff (parsed by [`crate::diff`]) and the
//! description of the change under review. Failures are reported as errors;
//! callers must never fall back to checking everything silently.

use std::fmt;
use std::io::ErrorKind;
use std::path::{Path, PathBuf};
use std::process::Command;

use crate::diff::{self, FileDiff};

/// Ignore user Mercurial configuration (pagers, colors, aliases) for stable output.
const HG_ENV: &[(&str, &str)] = &[("HGPLAIN", "1")];

/// An error talking to the version-control system.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VcsError(pub String);

impl fmt::Display for VcsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for VcsError {}

/// Which version-control backend to use.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VcsKind {
    Git,
    Hg,
    Jj,
}

impl VcsKind {
    pub fn parse(name: &str) -> Option<Self> {
        match name {
            "git" => Some(Self::Git),
            "hg" => Some(Self::Hg),
            "jj" => Some(Self::Jj),
            _ => None,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Git => "git",
            Self::Hg => "hg",
            Self::Jj => "jj",
        }
    }

    /// The marker directory at the repository root.
    fn marker(self) -> &'static str {
        match self {
            Self::Git => ".git",
            Self::Hg => ".hg",
            Self::Jj => ".jj",
        }
    }
}

/// An opened version-control repository.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Repo {
    kind: VcsKind,
    root: PathBuf,
}

impl Repo {
    pub fn name(&self) -> &'static str {
        self.kind.as_str()
    }

    /// Repository root; diff paths are relative to it.
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Revision to diff against when the user gives none.
    pub fn default_base(&self) -> Result<String, VcsError> {
        match self.kind {
            // Uncommitted changes to tracked files.
            VcsKind::Git => Ok("HEAD".to_string()),
            // In Fig, `p4base` (all unsubmitted work); elsewhere, the working
            // copy parent (uncommitted changes only).
            VcsKind::Hg => match run("hg", &["log", "-r", "p4base", "-T", "{node}"], &self.root, HG_ENV) {
                Ok(_) => Ok("p4base".to_string()),
                Err(_) => Ok(".".to_string()),
            },
            // The parent of the working-copy commit, i.e. the changes in `@`.
            VcsKind::Jj => Ok("@-".to_string()),
        }
    }

    /// Git-style diff from `base` to the working copy, limited to `scope`
    /// (a directory inside the repository).
    pub fn diff(&self, base: &str, scope: &Path) -> Result<String, VcsError> {
        match self.kind {
            // Two-dot diff from `base` to the working tree. Unlike `base...`,
            // this needs no merge base, so it works in shallow CI clones, and
            // it includes uncommitted changes.
            VcsKind::Git => run(
                "git",
                &[
                    "diff",
                    "-U0",
                    "--no-color",
                    "--no-ext-diff",
                    "--src-prefix=a/",
                    "--dst-prefix=b/",
                    "-M",
                    base,
                    "--",
                    ".",
                ],
                scope,
                &[],
            ),
            VcsKind::Hg => run("hg", &["diff", "--git", "-U", "0", "-r", base, "."], scope, HG_ENV),
            VcsKind::Jj => run(
                "jj",
                &[
                    "diff",
                    "--git",
                    "--context",
                    "0",
                    "--color",
                    "never",
                    "--no-pager",
                    "--from",
                    base,
                    "--to",
                    "@",
                    ".",
                ],
                scope,
                &[],
            ),
        }
    }

    /// Descriptions of the commits after `base` up to the working copy.
    pub fn description(&self, base: &str) -> Result<String, VcsError> {
        match self.kind {
            VcsKind::Git => run("git", &["log", "--format=%B", &format!("{base}..HEAD")], &self.root, &[]),
            VcsKind::Hg => {
                let revset = format!("({base})::. - ({base})");
                run("hg", &["log", "-r", &revset, "-T", "{desc}\n\n"], &self.root, HG_ENV)
            }
            VcsKind::Jj => {
                let revset = format!("({base})..@");
                run(
                    "jj",
                    &[
                        "log",
                        "--no-graph",
                        "--color",
                        "never",
                        "--no-pager",
                        "-r",
                        &revset,
                        "-T",
                        "description ++ \"\\n\"",
                    ],
                    &self.root,
                    &[],
                )
            }
        }
    }

    /// Files touched since `base` under `scope`, with their changed lines.
    pub fn changed_files(&self, base: &str, scope: &Path) -> Result<Vec<FileDiff>, VcsError> {
        let text = self.diff(base, scope)?;
        diff::parse(&text).map_err(|e| VcsError(format!("cannot parse `{}` diff output: {e}", self.name())))
    }
}

/// Opens the repository of kind `kind` (or auto-detects one) containing `start`.
///
/// Detection walks up from `start`; at each level jj is preferred over git (a
/// colocated jj repository also has `.git`), then git, then hg.
pub fn open(kind: Option<VcsKind>, start: &Path) -> Result<Repo, VcsError> {
    let order = match kind {
        Some(k) => vec![k],
        None => vec![VcsKind::Jj, VcsKind::Git, VcsKind::Hg],
    };
    for dir in start.ancestors() {
        for &k in &order {
            if dir.join(k.marker()).exists() {
                return Ok(Repo { kind: k, root: dir.to_path_buf() });
            }
        }
    }
    Err(VcsError(match kind {
        Some(k) => format!("no {} repository found at or above {}", k.marker(), start.display()),
        None => format!(
            "no git, hg or jj repository found at or above {}; pass --vcs, --all or explicit files",
            start.display()
        ),
    }))
}

/// Runs a VCS command in `cwd` and returns its standard output.
fn run(program: &str, args: &[&str], cwd: &Path, envs: &[(&str, &str)]) -> Result<String, VcsError> {
    let output = Command::new(program)
        .args(args)
        .current_dir(cwd)
        .envs(envs.iter().copied())
        .output()
        .map_err(|e| match e.kind() {
            ErrorKind::NotFound => VcsError(format!("`{program}` is not installed or not on PATH")),
            _ => VcsError(format!("cannot run `{program}`: {e}")),
        })?;
    if !output.status.success() {
        return Err(VcsError(format!(
            "`{program} {}` failed ({}): {}",
            args.join(" "),
            output.status,
            String::from_utf8_lossy(&output.stderr).trim()
        )));
    }
    Ok(String::from_utf8_lossy(&output.stdout).into_owned())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_dir(name: &str) -> PathBuf {
        let base = std::env::var_os("TEST_TMPDIR").map(PathBuf::from).unwrap_or_else(std::env::temp_dir);
        let dir = base.join(format!("custom_linter_vcs_{name}_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn detects_marker_directories_walking_up() {
        let root = temp_dir("detect");
        let nested = root.join("a/b");
        std::fs::create_dir_all(&nested).unwrap();
        std::fs::create_dir(root.join(".hg")).unwrap();
        assert_eq!(open(None, &nested).map(|v| v.name()).unwrap(), "hg");

        // A colocated jj repository also has `.git`; jj wins.
        std::fs::create_dir(root.join(".git")).unwrap();
        assert_eq!(open(None, &nested).map(|v| v.name()).unwrap(), "git");
        std::fs::create_dir(root.join(".jj")).unwrap();
        let vcs = open(None, &nested).unwrap();
        assert_eq!(vcs.name(), "jj");
        assert_eq!(vcs.root(), root.as_path());

        // An explicit kind skips detection order.
        assert_eq!(open(Some(VcsKind::Hg), &nested).map(|v| v.name()).unwrap(), "hg");
    }

    #[test]
    fn missing_repository_is_an_error() {
        let root = temp_dir("none");
        let err = open(Some(VcsKind::Git), &root).err().unwrap();
        assert!(err.0.contains("no .git repository"), "{err}");
    }

    #[test]
    fn missing_binary_is_a_clear_error() {
        let err = run("definitely-not-a-vcs-binary", &[], Path::new("."), &[]).unwrap_err();
        assert!(err.0.contains("is not installed"), "{err}");
    }

    #[test]
    fn parses_kind_names() {
        assert_eq!(VcsKind::parse("git"), Some(VcsKind::Git));
        assert_eq!(VcsKind::parse("hg"), Some(VcsKind::Hg));
        assert_eq!(VcsKind::parse("jj"), Some(VcsKind::Jj));
        assert_eq!(VcsKind::parse("svn"), None);
    }
}
