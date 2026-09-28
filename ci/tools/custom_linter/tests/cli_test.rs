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

//! End-to-end tests of the command-line interface.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

fn binary() -> PathBuf {
    // Blaze passes the path explicitly; Cargo provides it for integration tests.
    PathBuf::from(option_env!("CUSTOM_LINTER_BIN").or(option_env!("CARGO_BIN_EXE_custom_linter")).unwrap())
}

fn sample_metadata() -> PathBuf {
    match option_env!("SAMPLE_METADATA") {
        Some(p) => PathBuf::from(p),
        None => PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("testdata/sample.textproto"),
    }
}

fn run(args: &[&str]) -> Output {
    Command::new(binary()).args(args).output().expect("failed to run custom_linter")
}

#[test]
fn list_rules_prints_supported_rules() {
    let metadata = sample_metadata();
    let out = run(&["list-rules", "--rules", metadata.to_str().unwrap(), "--canonical-root", "//repo"]);
    let stdout = String::from_utf8(out.stdout).unwrap();
    let stderr = String::from_utf8(out.stderr).unwrap();
    assert!(out.status.success(), "stderr: {stderr}");

    let lines: Vec<&str> = stdout.lines().collect();
    assert_eq!(lines.len(), 3, "{stdout}");
    assert!(lines[0].starts_with("CheckNoPrint\tWARNING\tline\tdelta-only\t"), "{stdout}");
    assert!(lines[1].starts_with("CheckLicense\tERROR\tentire-file\tall-lines\t"), "{stdout}");
    assert!(lines[2].starts_with("CheckYear\tERROR\tline\tdelta-only\t"), "{stdout}");
    assert!(stderr.contains("3 rules loaded, 1 unsupported entries skipped"), "{stderr}");
    assert!(!stderr.contains("skipped: check_description"), "{stderr}");
}

#[test]
fn list_rules_verbose_reports_skipped_checks() {
    let metadata = sample_metadata();
    let out = run(&["list-rules", "--verbose", "--rules", metadata.to_str().unwrap(), "--canonical-root", "//repo"]);
    let stderr = String::from_utf8(out.stderr).unwrap();
    assert!(out.status.success(), "stderr: {stderr}");
    assert!(stderr.contains("skipped: check_description CheckDescription"), "{stderr}");
}

#[test]
fn rule_filter_keeps_only_the_named_rules() {
    let metadata = sample_metadata();
    let base = ["list-rules", "--rules", metadata.to_str().unwrap(), "--canonical-root", "//repo"];
    let out = run(&[&base[..], &["--rule", "CheckYear", "--rule=CheckNoPrint"]].concat());
    let stdout = String::from_utf8(out.stdout).unwrap();
    let ids: Vec<&str> = stdout.lines().map(|l| l.split('\t').next().unwrap()).collect();
    assert_eq!(ids, ["CheckNoPrint", "CheckYear"], "{stdout}");

    let out = run(&[&base[..], &["--rule", "CheckTypo"]].concat());
    assert_eq!(out.status.code(), Some(2));
    assert!(String::from_utf8(out.stderr).unwrap().contains("--rule CheckTypo: no such rule"));
}

#[test]
fn load_errors_exit_with_code_2() {
    let out = run(&["list-rules", "--rules", "/nonexistent/METADATA", "--canonical-root", "//repo"]);
    assert_eq!(out.status.code(), Some(2));
    let stderr = String::from_utf8(out.stderr).unwrap();
    assert!(stderr.contains("error: /nonexistent/METADATA: cannot read rules file"), "{stderr}");
}

#[test]
fn usage_errors_exit_with_code_2() {
    let out = run(&["list-rules", "--rules", "x.METADATA"]);
    assert_eq!(out.status.code(), Some(2));
    assert!(String::from_utf8(out.stderr).unwrap().contains("--canonical-root is required"));
}

// ---------------------------------------------------------------------------
// `check`

/// A fresh, empty directory for one test.
fn temp_dir(name: &str) -> PathBuf {
    let base = std::env::var_os("TEST_TMPDIR").map(PathBuf::from).unwrap_or_else(std::env::temp_dir);
    let dir = base.join(format!("custom_linter_cli_{name}_{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    dir
}

fn write(path: &Path, contents: &str) {
    fs::create_dir_all(path.parent().unwrap()).unwrap();
    fs::write(path, contents).unwrap();
}

/// Writes a project that violates both sample rules.
fn sample_project(name: &str) -> PathBuf {
    let dir = temp_dir(name);
    write(&dir.join("a.py"), "import x\nprint(1)\nprint(2)  # PRINT_OK=debug\n");
    write(&dir.join("tests/t.py"), "print(3)\n");
    write(&dir.join("lib.cc"), "int x;\n");
    write(&dir.join("ok.cc"), "// Copyright 2026\n");
    dir
}

/// Runs `check` with the sample rules against `project`, plus `extra` args.
fn check(project: &Path, extra: &[&str]) -> (Option<i32>, String, String) {
    let metadata = sample_metadata();
    let mut args = vec![
        "check",
        "--rules",
        metadata.to_str().unwrap(),
        "--canonical-root",
        "//repo",
        "--project-root",
        project.to_str().unwrap(),
    ];
    args.extend_from_slice(extra);
    let out = run(&args);
    (out.status.code(), String::from_utf8(out.stdout).unwrap(), String::from_utf8(out.stderr).unwrap())
}

#[test]
fn check_explicit_files_checks_all_lines() {
    let dir = sample_project("explicit");
    let a = dir.join("a.py");
    let t = dir.join("tests/t.py");
    let (code, stdout, stderr) = check(&dir, &[a.to_str().unwrap(), t.to_str().unwrap()]);
    // Only a WARNING: exit 0. tests/ is excluded; the suppressed line is not reported.
    assert_eq!(code, Some(0), "{stderr}");
    assert_eq!(stdout, "a.py:2: WARNING [CheckNoPrint] Use logging instead of print().\n");
    assert!(stderr.contains("2 files considered, 1 checked"), "{stderr}");
}

#[test]
fn check_all_reports_errors_as_json_and_exits_1() {
    let dir = sample_project("all");
    let (code, stdout, stderr) = check(&dir, &["--all", "--format=json"]);
    assert_eq!(code, Some(1), "{stderr}");
    assert!(stdout.starts_with("{\"violations\":["), "{stdout}");
    assert!(stdout.contains("\"rule_id\":\"CheckNoPrint\",\"severity\":\"WARNING\",\"path\":\"a.py\",\"line\":2"));
    assert!(stdout.contains("\"rule_id\":\"CheckLicense\",\"severity\":\"ERROR\",\"path\":\"lib.cc\",\"line\":null"));
    assert!(!stdout.contains("ok.cc") && !stdout.contains("tests/t.py"), "{stdout}");
    assert!(stdout.contains("\"summary\":{\"files_checked\":3,\"errors\":1,\"warnings\":1,\"notices\":0}"), "{stdout}");
}

#[test]
fn check_description_tags_disable_rules() {
    let dir = sample_project("tags");
    let description = dir.join("description.txt");
    write(&description, "Revert something.\n\nROLLBACK_OF=123\n");
    let (code, stdout, stderr) =
        check(&dir, &["--all", "--verbose", "--description-file", description.to_str().unwrap()]);
    assert_eq!(code, Some(1), "{stderr}");
    assert!(!stdout.contains("CheckNoPrint"), "{stdout}");
    assert!(stderr.contains("disabled: CheckNoPrint (by ROLLBACK_OF=)"), "{stderr}");
}

#[test]
fn check_rejects_files_outside_the_project() {
    let dir = sample_project("outside");
    let other = temp_dir("outside_other").join("x.py");
    write(&other, "print(1)\n");
    let (code, _, stderr) = check(&dir, &[other.to_str().unwrap()]);
    assert_eq!(code, Some(2));
    assert!(stderr.contains("is outside --project-root"), "{stderr}");
}

#[test]
fn check_without_a_repository_is_an_error() {
    let dir = sample_project("norepo");
    let (code, _, stderr) = check(&dir, &["--vcs", "git"]);
    assert_eq!(code, Some(2));
    assert!(stderr.contains("no .git repository"), "{stderr}");
}

#[test]
fn path_maps_undo_an_exported_layout() {
    // An OSS-style export of //repo: the repo root is //repo/oss, the package
    // moved to src/pkg, tests/ stays at //repo/tests, and the rules file was
    // copied to ci/. Without the maps, nothing would match.
    let dir = temp_dir("exported");
    fs::create_dir_all(dir.join("ci")).unwrap();
    fs::copy(sample_metadata(), dir.join("ci/presubmit.METADATA")).unwrap();
    write(&dir.join("src/pkg/a.py"), "print(1)\n");
    write(&dir.join("tests/t.py"), "print(2)\n");
    let rules = dir.join("ci/presubmit.METADATA");
    let args = |maps: bool| {
        let mut a = vec![
            "check".to_string(),
            "--all".into(),
            "--rules".into(),
            rules.to_str().unwrap().into(),
            "--canonical-root".into(),
            "//repo/oss".into(),
            "--project-root".into(),
            dir.to_str().unwrap().into(),
        ];
        if maps {
            for m in ["src/pkg=//repo", "tests=//repo/tests", "ci/presubmit.METADATA=//repo/presubmit.METADATA"] {
                a.extend(["--path-map".to_string(), m.to_string()]);
            }
        }
        a
    };
    let run_with = |maps| {
        let out = Command::new(binary()).args(args(maps)).output().unwrap();
        (out.status.code(), String::from_utf8(out.stdout).unwrap(), String::from_utf8(out.stderr).unwrap())
    };

    let (code, stdout, stderr) = run_with(true);
    assert_eq!(code, Some(0), "{stderr}");
    assert_eq!(stdout, "src/pkg/a.py:1: WARNING [CheckNoPrint] Use logging instead of print().\n", "{stderr}");

    let (_, stdout, _) = run_with(false);
    assert_eq!(stdout, "", "rules resolve under //repo/oss/ci without the maps");
}

// ---------------------------------------------------------------------------
// git integration

/// Runs git with a fixed identity and no user/system configuration.
fn git(dir: &Path, args: &[&str]) -> Output {
    let out = Command::new("git")
        .args(args)
        .current_dir(dir)
        .env("GIT_CONFIG_NOSYSTEM", "1")
        .env("GIT_CONFIG_GLOBAL", "/dev/null")
        .env("GIT_AUTHOR_NAME", "t")
        .env("GIT_AUTHOR_EMAIL", "t@example.com")
        .env("GIT_COMMITTER_NAME", "t")
        .env("GIT_COMMITTER_EMAIL", "t@example.com")
        .output()
        .expect("failed to run git");
    assert!(out.status.success(), "git {args:?}: {}", String::from_utf8_lossy(&out.stderr));
    out
}

fn have_git() -> bool {
    let found = Command::new("git").arg("--version").output().is_ok_and(|o| o.status.success());
    if !found {
        eprintln!("git not found; skipping git integration test");
    }
    found
}

/// A git repository with the sample project committed; the project lives in
/// a subdirectory to exercise VCS-root vs project-root mapping.
fn git_repo(name: &str) -> (PathBuf, PathBuf) {
    let repo = temp_dir(name);
    let project = repo.join("proj");
    write(&project.join("a.py"), "print(0)\n");
    write(&project.join("lib.cc"), "int x;\n");
    write(&repo.join("outside.py"), "print(0)\n");
    git(&repo, &["init", "-q"]);
    git(&repo, &["add", "-A"]);
    git(&repo, &["commit", "-q", "-m", "initial"]);
    (repo, project)
}

#[test]
fn git_checks_only_changed_lines_of_changed_files() {
    if !have_git() {
        return;
    }
    let (repo, project) = git_repo("git_delta");
    // Uncommitted edit: only line 2 is new. lib.cc is untouched, so its
    // missing license is not reported; outside.py is outside the project.
    write(&project.join("a.py"), "print(0)\nprint(1)\n");
    write(&repo.join("outside.py"), "print(0)\nprint(1)\n");
    let (code, stdout, stderr) = check(&project, &["--verbose"]);
    assert_eq!(code, Some(0), "{stderr}");
    assert_eq!(stdout, "a.py:2: WARNING [CheckNoPrint] Use logging instead of print().\n", "{stderr}");
    assert!(stderr.contains("vcs: git"), "{stderr}");

    // Touching lib.cc makes the (non-delta) license rule check the whole file.
    write(&project.join("lib.cc"), "int x;\nint y;\n");
    let (code, stdout, stderr) = check(&project, &[]);
    assert_eq!(code, Some(1), "{stderr}");
    assert!(stdout.contains("lib.cc: ERROR [CheckLicense]"), "{stdout}");
}

#[test]
fn git_operation_restricted_rules_skip_new_files() {
    if !have_git() {
        return;
    }
    let (repo, project) = git_repo("git_ops");
    // New file: CheckYear (EDIT only) must not fire. Edited file: it must.
    // Untracked files are not part of the change, so stage the new one.
    write(&project.join("new.cc"), "// Copyright 2026\n");
    write(&project.join("lib.cc"), "// Copyright 2027\nint x;\n");
    git(&repo, &["add", "proj/new.cc"]);
    let (code, stdout, stderr) = check(&project, &[]);
    assert_eq!(code, Some(1), "{stderr}");
    assert_eq!(stdout, "lib.cc:1: ERROR [CheckYear] Do not update the copyright year.\n", "{stderr}");
}

#[test]
fn git_commit_description_tags_disable_rules() {
    if !have_git() {
        return;
    }
    let (repo, project) = git_repo("git_tags");
    write(&project.join("a.py"), "print(0)\nprint(1)\n");
    git(&repo, &["commit", "-q", "-am", "Revert.\n\nROLLBACK_OF=1"]);
    let (code, stdout, stderr) = check(&project, &["--base", "HEAD~1"]);
    assert_eq!(code, Some(0), "{stderr}");
    assert_eq!(stdout, "", "{stderr}");
}

#[test]
fn git_works_in_a_shallow_clone() {
    if !have_git() {
        return;
    }
    let (repo, _) = git_repo("git_shallow_src");
    let clone = temp_dir("git_shallow").join("clone");
    let url = format!("file://{}", repo.display());
    git(clone.parent().unwrap(), &["clone", "-q", "--depth=1", &url, clone.to_str().unwrap()]);
    let project = clone.join("proj");
    write(&project.join("a.py"), "print(0)\nprint(1)\n");
    let (code, stdout, stderr) = check(&project, &[]);
    assert_eq!(code, Some(0), "{stderr}");
    assert_eq!(stdout, "a.py:2: WARNING [CheckNoPrint] Use logging instead of print().\n", "{stderr}");
}
