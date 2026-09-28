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

//! Subcommand orchestration: wires rules, VCS, engine and reports together.
//! Kept separate from `main` so it can be tested without spawning processes.

use std::io::Write;
use std::path::{Component, Path, PathBuf};
use std::time::Instant;

use crate::cli::{Command, Options, USAGE};
use crate::diff::FileDiff;
use crate::engine::{self, FileTask};
use crate::model::{FileOp, RuleSet, Severity};
use crate::paths::PathMapper;
use crate::report::{self, Summary};
use crate::rules;
use crate::tags::Tags;
use crate::vcs;

pub const EXIT_OK: u8 = 0;
pub const EXIT_VIOLATIONS: u8 = 1;
pub const EXIT_ERROR: u8 = 2;

/// Directories never descended into by `--all`.
const SKIPPED_DIRS: &[&str] = &[".git", ".hg", ".jj", ".nox", ".venv", "__pycache__", "target"];

/// Runs the program with `args` (without the program name) and returns the
/// exit code. Reports go to `out`; diagnostics go to `err`.
pub fn main(args: Vec<String>, out: &mut dyn Write, err: &mut dyn Write) -> u8 {
    let command = match crate::cli::parse(args) {
        Ok(c) => c,
        Err(e) => {
            let _ = writeln!(err, "error: {e}\n\n{USAGE}");
            return EXIT_ERROR;
        }
    };
    let result = match command {
        Command::Help => write!(out, "{USAGE}").map(|_| EXIT_OK).map_err(|e| e.to_string()),
        Command::Version => {
            writeln!(out, "custom_linter {}", option_env!("CARGO_PKG_VERSION").unwrap_or("dev")).map(|_| EXIT_OK).map_err(|e| e.to_string())
        }
        Command::ListRules(o) => list_rules(&o, out, err),
        Command::Check(o) => check(&o, out, err),
    };
    result.unwrap_or_else(|e| {
        let _ = writeln!(err, "error: {e}");
        EXIT_ERROR
    })
}

/// Loads all rule files. Relative paths in a rule file are resolved against
/// the canonical directory of that file when it lies inside the project;
/// otherwise against the canonical root.
fn load_rules(o: &Options, project: Option<(&Path, &PathMapper)>) -> Result<RuleSet, String> {
    let mut set = RuleSet::default();
    for path in &o.rules {
        let base_dir = project
            .and_then(|(root, mapper)| {
                let abs = std::fs::canonicalize(path).ok()?;
                let canonical = mapper.to_canonical(abs.strip_prefix(root).ok()?)?;
                canonical.rsplit_once('/').map(|(dir, _)| dir.to_string())
            })
            .unwrap_or_else(|| o.canonical_root.clone());
        set.extend(rules::load_file(path, &base_dir).map_err(|e| e.to_string())?);
    }
    if !o.only_rules.is_empty() {
        if let Some(unknown) = o.only_rules.iter().find(|id| !set.rules.iter().any(|r| &r.id == *id)) {
            return Err(format!("--rule {unknown}: no such rule in the loaded rule files"));
        }
        set.rules.retain(|r| o.only_rules.contains(&r.id));
    }
    Ok(set)
}

fn report_skipped(set: &RuleSet, verbose: bool, err: &mut dyn Write) {
    if verbose {
        for s in &set.skipped {
            let _ = writeln!(err, "skipped: {} {} ({})", s.kind, s.id.as_deref().unwrap_or("-"), s.origin);
        }
    }
}

fn list_rules(o: &Options, out: &mut dyn Write, err: &mut dyn Write) -> Result<u8, String> {
    let set = load_rules(o, None)?;
    for rule in &set.rules {
        writeln!(
            out,
            "{}\t{}\t{}\t{}\t{}",
            rule.id,
            rule.severity,
            rule.mode.as_str(),
            if rule.delta_only { "delta-only" } else { "all-lines" },
            rule.origin,
        )
        .map_err(|e| e.to_string())?;
    }
    report_skipped(&set, o.verbose, err);
    let _ = writeln!(err, "{} rules loaded, {} unsupported entries skipped", set.rules.len(), set.skipped.len());
    Ok(EXIT_OK)
}

fn check(o: &Options, out: &mut dyn Write, err: &mut dyn Write) -> Result<u8, String> {
    let t0 = Instant::now();
    let project_root = std::fs::canonicalize(&o.project_root)
        .map_err(|e| format!("invalid --project-root {}: {e}", o.project_root.display()))?;
    let mapper = PathMapper::new(&o.canonical_root).with_prefix_maps(o.path_maps.clone());
    let set = load_rules(o, Some((&project_root, &mapper)))?;
    report_skipped(&set, o.verbose, err);

    // Collect files and the change description.
    let t_walk_start = Instant::now();
    let (files, vcs_description) = if o.all {
        (walk(&project_root)?, None)
    } else if !o.files.is_empty() {
        (explicit_files(&o.files, &project_root)?, None)
    } else {
        let (files, description) = changed_files(o, &project_root, err)?;
        (files, Some(description))
    };
    let description = match &o.description_file {
        Some(path) => std::fs::read_to_string(path)
            .map_err(|e| format!("cannot read --description-file {}: {e}", path.display()))?,
        None => vcs_description.unwrap_or_default(),
    };

    // Rules disabled by tags in the description are not evaluated.
    let tags = Tags::parse(&description);
    let mut active = Vec::with_capacity(set.rules.len());
    for rule in &set.rules {
        match tags.disabling(rule) {
            Some(tag) if o.verbose => {
                let _ = writeln!(err, "disabled: {} (by {tag}=)", rule.id);
            }
            Some(_) => {}
            None => active.push(rule),
        }
    }

    let tasks: Vec<FileTask> = files
        .into_iter()
        .filter_map(|(rel, changed_lines, op)| {
            let canonical = mapper.to_canonical(Path::new(&rel))?;
            Some(FileTask { abs_path: project_root.join(&rel), rel_path: rel, canonical, changed_lines, op })
        })
        .collect();
    let t_walk = t_walk_start.elapsed();

    let detected_cpus = std::thread::available_parallelism().map_or(1, |n| n.get());
    let jobs = o.jobs.unwrap_or(detected_cpus);
    let t_eval_start = Instant::now();
    let outcome = engine::run(&active, &tasks, jobs)?;
    let t_eval = t_eval_start.elapsed();
    let t_total = t0.elapsed();

    let summary = Summary::new(&outcome.violations, outcome.files_checked);
    write!(out, "{}", report::render(o.format, &outcome.violations, &summary)).map_err(|e| e.to_string())?;
    let _ = writeln!(
        err,
        "custom_linter: {} files considered, {} checked; {} errors, {} warnings, {} notices",
        tasks.len(),
        summary.files_checked,
        summary.errors,
        summary.warnings,
        summary.notices
    );
    let _ = writeln!(
        err,
        "INFO: Checked {} files across {} rules using {}/{} CPU worker(s) in {:.3}s (walk={:.3}s, eval={:.3}s)",
        summary.files_checked,
        active.len(),
        jobs,
        detected_cpus,
        t_total.as_secs_f64(),
        t_walk.as_secs_f64(),
        t_eval.as_secs_f64(),
    );
    Ok(if outcome.violations.iter().any(|v| v.severity == Severity::Error) { EXIT_VIOLATIONS } else { EXIT_OK })
}

/// A project-relative path, its changed lines (`None` = all lines) and how
/// the change touched it (`None` = no change context).
type FileEntry = (String, Option<std::collections::BTreeSet<usize>>, Option<FileOp>);

/// Maps a diff entry to a file operation. A copy or rename with no
/// further changes is a BRANCH; with changes, it counts as a new file (ADD).
fn file_op(d: &FileDiff) -> FileOp {
    match (d.added, d.copied) {
        (true, _) => FileOp::Add,
        (false, true) if d.added_lines.is_empty() => FileOp::Branch,
        (false, true) => FileOp::Add,
        (false, false) => FileOp::Edit,
    }
}

/// Files changed since the base revision, restricted to the project.
fn changed_files(o: &Options, project_root: &Path, err: &mut dyn Write) -> Result<(Vec<FileEntry>, String), String> {
    let repo = vcs::open(o.vcs, project_root).map_err(|e| e.to_string())?;
    let base = match &o.base {
        Some(b) => b.clone(),
        None => repo.default_base().map_err(|e| e.to_string())?,
    };
    if o.verbose {
        let _ = writeln!(err, "vcs: {} at {}, base {base}", repo.name(), repo.root().display());
    }
    let diffs = repo.changed_files(&base, project_root).map_err(|e| e.to_string())?;
    let description = if o.description_file.is_some() {
        String::new() // Not needed; avoid a VCS call that could fail.
    } else {
        repo.description(&base).map_err(|e| e.to_string())?
    };
    let mut files = Vec::new();
    for d in diffs.into_iter().filter(|d| !d.deleted) {
        let Ok(rel) = repo.root().join(&d.path).strip_prefix(project_root).map(Path::to_path_buf) else {
            continue; // Outside the project (the diff is scoped, but be safe).
        };
        let op = file_op(&d);
        files.push((slash_path(&rel), Some(d.added_lines), Some(op)));
    }
    Ok((files, description))
}

/// Explicit files, which must be inside the project; all lines are checked.
fn explicit_files(files: &[PathBuf], project_root: &Path) -> Result<Vec<FileEntry>, String> {
    files
        .iter()
        .map(|f| {
            let abs = std::fs::canonicalize(f).map_err(|e| format!("cannot open {}: {e}", f.display()))?;
            let rel = abs
                .strip_prefix(project_root)
                .map_err(|_| format!("{} is outside --project-root {}", f.display(), project_root.display()))?;
            Ok((slash_path(rel), None, None))
        })
        .collect()
}

/// Every regular file under `root`, excluding VCS metadata and symlinks.
fn walk(root: &Path) -> Result<Vec<FileEntry>, String> {
    let mut out = Vec::new();
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        let entries = std::fs::read_dir(&dir).map_err(|e| format!("cannot list {}: {e}", dir.display()))?;
        for entry in entries {
            let entry = entry.map_err(|e| format!("cannot list {}: {e}", dir.display()))?;
            let file_type = entry.file_type().map_err(|e| format!("cannot stat {}: {e}", entry.path().display()))?;
            let path = entry.path();
            if file_type.is_dir() {
                if !SKIPPED_DIRS.iter().any(|s| entry.file_name() == *s) {
                    stack.push(path);
                }
            } else if file_type.is_file() {
                if let Ok(rel) = path.strip_prefix(root) {
                    out.push((slash_path(rel), None, None));
                }
            }
        }
    }
    out.sort_by(|a, b| a.0.cmp(&b.0));
    Ok(out)
}

/// Joins normal path components with `/`, independent of the platform.
fn slash_path(path: &Path) -> String {
    path.components()
        .filter_map(|c| match c {
            Component::Normal(p) => Some(p.to_string_lossy().into_owned()),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("/")
}
