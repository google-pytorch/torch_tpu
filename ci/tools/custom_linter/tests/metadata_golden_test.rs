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

//! Golden test: loads the real TorchTPU `custom_lint_rules.METADATA` (or
//! `ci/custom_lint_rules.textproto` in OSS, plus `presubmit.METADATA` when
//! provided) and checks that every content rule is understood (parses, and its
//! regexes compile with the Rust `regex` engine).

use std::path::PathBuf;

use custom_linter::model::{FileOp, MatchMode, Rule, RuleSet, Severity};
use custom_linter::rules::{load_file, parse_metadata};

const MONOREPO_ROOT: &str = concat!("//depot/google3", "/third_party/py/torch_tpu");
const OSS_ROOT: &str = "torch_tpu";

/// Every `check_contents` id in `custom_lint_rules.METADATA`, in file order.
/// Update this list when adding or removing a shared content check.
const EXPECTED_SHARED_IDS: &[&str] = &[
    "CheckCopybaraTags",
    "CheckPythonEnvAccess",
    "CheckKernelMacro",
    "CheckTolerancesUpdatedByTool",
    "CheckAssertOkAndAssign",
    "CheckC10ToStringUsage",
    "CheckDeprecatedMutexLock",
    "CheckIgnoreInCacheKeyLegacy",
    "CheckIncludeBeforeCopyright",
    "CheckThreadLocalNoDestructor",
    "CheckThreadLocalNoTag",
    "CheckMaybeWrapDim",
    "CheckDisallowedLogMacros",
    "CheckStdEndl",
    "CheckUninitializedTensor",
    "CheckErrorApi",
    "CheckMacroNamePrefix",
    "CheckIntVec",
    "CheckVecMethod",
    "CheckErrorSpace",
    "CheckLeadingColons",
    "CheckUsingDecls",
    "CheckMlirSignedInt",
    "CheckLambdaCapture",
    "CheckAtInferSize",
    "CheckStdFunction",
    "CheckStringView",
    "CheckCheckFunctionName",
    "CheckHashVsFingerprint",
    "CheckNoVoidCast",
    "CheckIgnoreError",
    "CheckCppFlagNames",
    "CheckEnvVarName",
    "CheckGetenv",
    "CheckGetFlag",
    "CheckCrashingMacros",
    "CheckRawOpName",
    "CheckMImpl",
    "CheckStdPair",
    "CheckPyFunctionSuffix",
    "CheckTorchThreadLocalStateGuards",
    "CheckRawPyBindUsage",
    "CheckExplicitPrecisionUsage",
    "CheckNotestOss",
    "CheckNobuildOss",
    "CheckBannedApiDependency",
    "CheckNoSan",
    "CheckNoTapTag",
    "CheckCcTargetUsage",
    "CheckPyLibraryUsage",
    "CheckPyTestUsage",
    "CheckTorchTpuPackageEnd",
    "CheckPythonErrorTestArgs",
    "CheckPythonErrorTestArgsParentheses",
    "CheckPythonErrorTestBackslashes",
    "CheckAssertRaises",
    "CheckRelativeImports",
    "CheckPythonFlags",
    "CheckAbslTestCase",
    "CheckAssertClose",
    "CheckPythonLogging",
    "CheckInitDeviceUsage",
    "CheckCopyrightYearUpdate",
    "CheckToAccumulateTypeUsage",
];

fn shared_rules_path_and_root() -> (PathBuf, &'static str) {
    if let Some(p) = option_env!("CUSTOM_LINT_RULES_METADATA") {
        return (PathBuf::from(p), MONOREPO_ROOT);
    }
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let oss_path = manifest_dir.join("../../custom_lint_rules.textproto");
    if oss_path.exists() {
        return (oss_path, OSS_ROOT);
    }
    (manifest_dir.join("../../../../custom_lint_rules.METADATA"), MONOREPO_ROOT)
}

fn root() -> &'static str {
    shared_rules_path_and_root().1
}

fn load_shared() -> RuleSet {
    let (path, base) = shared_rules_path_and_root();
    load_file(&path, base).unwrap_or_else(|e| panic!("{e}"))
}

fn load() -> RuleSet {
    let mut set = load_shared();
    if let Some(presubmit) = option_env!("PRESUBMIT_METADATA") {
        set.extend(load_file(&PathBuf::from(presubmit), MONOREPO_ROOT).unwrap_or_else(|e| panic!("{e}")));
    }
    set
}

fn rule<'a>(set: &'a RuleSet, id: &str) -> &'a Rule {
    set.rules.iter().find(|r| r.id == id).unwrap_or_else(|| panic!("rule {id} not found"))
}

fn path(relative: &str) -> String {
    format!("{}/{relative}", root())
}

#[test]
fn loads_every_content_check() {
    let shared = load_shared();
    let ids: Vec<&str> = shared.rules.iter().map(|r| r.id.as_str()).collect();
    assert_eq!(ids, EXPECTED_SHARED_IDS);

    if option_env!("PRESUBMIT_METADATA").is_some() {
        let all = load();
        assert!(all.rules.iter().any(|r| r.id == "CheckLicenseHeader"));
    }
}

#[test]
fn every_rule_has_a_message_and_a_pattern() {
    for r in &load().rules {
        assert!(!r.message.trim().is_empty(), "{} has no failure message", r.id);
        assert!(r.prohibited.is_some() || r.required.is_some(), "{} has no pattern", r.id);
    }
}

#[test]
fn non_content_checks_are_reported_as_skipped() {
    let set = load();
    let kinds: Vec<&str> = set.skipped.iter().map(|s| s.kind.as_str()).collect();
    assert!(kinds.contains(&"check_file_operation"), "expected a skipped check_file_operation; got {kinds:?}");
    if option_env!("PRESUBMIT_METADATA").is_some() {
        for kind in ["check_description", "check_presubmit_service", "include_presubmit"] {
            assert!(kinds.contains(&kind), "expected a skipped {kind}; got {kinds:?}");
        }
    }
}

#[test]
fn license_header_rule() {
    if option_env!("PRESUBMIT_METADATA").is_none() {
        return;
    }
    let set = load();
    let r = rule(&set, "CheckLicenseHeader");
    assert_eq!(r.mode, MatchMode::EntireFile);
    assert_eq!(r.severity, Severity::Error);
    assert!(!r.delta_only);
    assert_eq!(r.disable_tags, ["SKIP_LICENSE_HEADER_CHECK"]);
    assert!(r.paths.matches(&path("csrc/ops/add.cc")));
    assert!(r.paths.matches(&path("tests/BUILD")));
    assert!(!r.paths.matches(&path("google/tools/x.py")), "google/ is excluded");
    assert!(!r.paths.matches(&path("docs/index.md")));

    let required = r.required.as_ref().unwrap();
    assert!(required.is_match("# Copyright 2026 Google LLC\n#\n# Licensed under the Apache License, Version 2.0"));
    assert!(!required.is_match("# Copyright Google\n"));
}

#[test]
fn copybara_tags_rule_scope() {
    let set = load();
    let r = rule(&set, "CheckCopybaraTags");
    assert!(r.delta_only);
    assert_eq!(r.mode, MatchMode::Line);
    assert_eq!(r.disable_tags, ["ALLOW_IRREVERSIBLE_TRANSFORM", "ROLLBACK_OF"]);
    assert!(r.paths.matches(&path("csrc/a.cc")));
    assert!(!r.paths.matches(&path("copy.bara.sky")));
    // Built with concat! so that this line does not itself trip the rule.
    assert!(r.prohibited.as_ref().unwrap().is_match(concat!("# copybara", ":strip_begin")));
    assert!(!r.prohibited.as_ref().unwrap().is_match(concat!("# copybara", ":comment")));
}

#[test]
fn copyright_year_rule_only_applies_to_edits() {
    let set = load();
    let r = rule(&set, "CheckCopyrightYearUpdate");
    assert_eq!(r.operations, [FileOp::Edit]);
    assert!(r.applies_to(Some(FileOp::Edit)));
    assert!(!r.applies_to(Some(FileOp::Add)), "new files may carry any year");
}

#[test]
fn python_env_access_rule() {
    let set = load();
    let r = rule(&set, "CheckPythonEnvAccess");
    assert!(r.paths.matches(&path("_internal/env.py")));
    assert!(r.paths.matches(&path("oss/torch_tpu/foo.py")));
    for excluded in [
        "_internal/env_test.py",
        "tests/ops_test.py",
        "examples/benchmarks/x.py",
        "oss/ci/tools/run.py",
        "oss/noxfile.py",
        "google/tools/x.py",
    ] {
        assert!(!r.paths.matches(&path(excluded)), "{excluded} should be excluded");
    }

    let prohibited = r.prohibited.as_ref().unwrap();
    let suppression = r.suppression.as_ref().unwrap();
    assert!(prohibited.is_match("x = os.environ['A']"));
    assert!(prohibited.is_match("os.getenv('A')"));
    assert!(suppression.is_match("os.getenv('A')  # PY_ENVIRON_OK=needed"));
    assert!(suppression.is_match("# read os.environ here"));
    assert!(suppression.is_match("Uses `os.environ` internally."));
    assert!(!suppression.is_match("os.getenv('A')"));
}

#[test]
fn cc_target_rule_matches_build_files() {
    let set = load();
    let r = rule(&set, "CheckCcTargetUsage");
    assert!(r.paths.matches(&path("csrc/ops/BUILD")));
    assert!(!r.paths.matches(&path("csrc/ops/add.cc")));
    assert!(r.prohibited.as_ref().unwrap().is_match("cc_library("));
}

#[test]
fn delta_only_and_entire_file_are_never_combined() {
    for r in &load().rules {
        assert!(!(r.delta_only && r.mode == MatchMode::EntireFile), "{}", r.id);
    }
}

#[test]
fn oss_transformed_rules_match_same_paths() {
    let (shared_path, _) = shared_rules_path_and_root();
    let raw = std::fs::read_to_string(&shared_path).unwrap();
    let oss_raw = raw
        .replace(MONOREPO_ROOT, OSS_ROOT)
        .replace("//depot/.../", ".../");
    let oss_set = parse_metadata(&oss_raw, &shared_path, OSS_ROOT).unwrap();
    let mono_set = load_shared();
    assert_eq!(oss_set.rules.len(), mono_set.rules.len());

    let sample_rel_paths = [
        "csrc/ops/add.cc",
        "csrc/ops/BUILD",
        "tests/BUILD",
        "tests/ops_test.py",
        "_internal/env.py",
        "_internal/env_test.py",
        "examples/benchmarks/x.py",
        "oss/torch_tpu/foo.py",
        "oss/ci/tools/run.py",
        "oss/noxfile.py",
        "google/tools/x.py",
        "docs/index.md",
        "copy.bara.sky",
    ];

    for (oss_rule, mono_rule) in oss_set.rules.iter().zip(mono_set.rules.iter()) {
        assert_eq!(oss_rule.id, mono_rule.id);
        for rel in sample_rel_paths {
            let oss_full = format!("{OSS_ROOT}/{rel}");
            let mono_full = format!("{}/{rel}", root());
            assert_eq!(
                oss_rule.paths.matches(&oss_full),
                mono_rule.paths.matches(&mono_full),
                "path matching mismatch on rule {} for {rel}",
                oss_rule.id
            );
        }
    }
}

