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

//! Command-line parsing. Hand-written (rather than using an argument-parsing
//! crate) to keep the dependency set to `regex` alone; the surface is small.

use std::path::PathBuf;

use crate::report::Format;
use crate::vcs::VcsKind;

pub const USAGE: &str = "\
Usage:
  custom_linter check --rules FILE --canonical-root ROOT [options] [FILES...]
  custom_linter list-rules --rules FILE --canonical-root ROOT [--verbose]
  custom_linter --help | --version

Commands:
  check        Check files against the rules. By default checks the lines
               changed in the working copy relative to --base, as reported by
               the version-control system. With --all, checks every file under
               --project-root. With FILES, checks exactly those files.
  list-rules   Print the rules loaded from the rule files.

Options:
  --rules FILE             Rule file to load. May be repeated.
  --rule ID                Only evaluate the rule with this id. May be
                           repeated. Unknown ids are an error.
  --canonical-root ROOT    Canonical path that rules are written against for
                           the project root, e.g. torch_tpu.
                           Relative paths in a rule file resolve against the
                           file's own canonical directory if it is inside the
                           project, else against ROOT.
  --project-root DIR       Directory that ROOT maps to (default: .).
  --path-map LOCAL=CANON   Map the project-relative prefix LOCAL to the
                           canonical prefix CANON instead of ROOT/LOCAL, for
                           exported layouts (e.g. Copybara moves). May be
                           repeated; the longest matching LOCAL wins.
  --vcs auto|git|hg|jj     Version-control system (default: auto).
  --base REV               Revision to diff against (default: git HEAD,
                           hg p4base or ., jj @-).
  --all                    Check every file under --project-root.
  --description-file FILE  Change description used for disable tags
                           (default: the VCS log since --base).
  --format text|json       Output format (default: text).
  --jobs N                 Worker threads (default: available CPUs).
  --verbose                Report skipped entries and disabled rules.

Exit status: 0 if no ERROR violations, 1 if any, 2 on usage or internal error.
";

/// Parsed command line.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Command {
    Help,
    Version,
    ListRules(Options),
    Check(Options),
}

/// Options shared by the subcommands. Options irrelevant to a subcommand are
/// rejected during parsing.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Options {
    pub rules: Vec<PathBuf>,
    /// Rule ids to keep; empty means all rules.
    pub only_rules: Vec<String>,
    pub canonical_root: String,
    pub project_root: PathBuf,
    /// `(project-relative prefix, canonical prefix)` overrides.
    pub path_maps: Vec<(String, String)>,
    /// `None` means auto-detect.
    pub vcs: Option<VcsKind>,
    pub base: Option<String>,
    pub all: bool,
    pub description_file: Option<PathBuf>,
    pub format: Format,
    pub jobs: Option<usize>,
    pub verbose: bool,
    pub files: Vec<PathBuf>,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            rules: Vec::new(),
            only_rules: Vec::new(),
            canonical_root: String::new(),
            project_root: PathBuf::from("."),
            path_maps: Vec::new(),
            vcs: None,
            base: None,
            all: false,
            description_file: None,
            format: Format::Text,
            jobs: None,
            verbose: false,
            files: Vec::new(),
        }
    }
}

/// Flags accepted by `list-rules`; `check` accepts all flags.
const LIST_RULES_FLAGS: &[&str] = &["rules", "canonical-root", "rule", "verbose"];

/// Parses `args` (without the program name).
pub fn parse<I, S>(args: I) -> Result<Command, String>
where
    I: IntoIterator<Item = S>,
    S: Into<String>,
{
    let mut args = args.into_iter().map(Into::into);
    let subcommand = match args.next().as_deref() {
        None => return Err("missing command".into()),
        Some("-h" | "--help" | "help") => return Ok(Command::Help),
        Some("-V" | "--version") => return Ok(Command::Version),
        Some(name @ ("check" | "list-rules")) => name.to_string(),
        Some(other) => return Err(format!("unknown command `{other}`")),
    };
    let is_check = subcommand == "check";

    let mut o = Options::default();
    let mut only_positional = false;
    while let Some(arg) = args.next() {
        if only_positional || !arg.starts_with("--") || arg == "-" {
            if arg.starts_with('-') && arg != "-" && !only_positional {
                return Err(format!("unknown option `{arg}`"));
            }
            if !is_check {
                return Err(format!("unexpected argument `{arg}`"));
            }
            o.files.push(PathBuf::from(arg));
            continue;
        }
        if arg == "--" {
            only_positional = true;
            continue;
        }
        if arg == "--help" {
            return Ok(Command::Help);
        }
        let (name, inline) = match arg[2..].split_once('=') {
            Some((n, v)) => (n.to_string(), Some(v.to_string())),
            None => (arg[2..].to_string(), None),
        };
        if !is_check && !LIST_RULES_FLAGS.contains(&name.as_str()) {
            return Err(format!("option `--{name}` is not valid for `{subcommand}`"));
        }
        let mut value = || -> Result<String, String> {
            match inline.clone() {
                Some(v) => Ok(v),
                None => args.next().ok_or_else(|| format!("option `--{name}` requires a value")),
            }
        };
        match name.as_str() {
            "rules" => o.rules.push(PathBuf::from(value()?)),
            "rule" => o.only_rules.push(value()?),
            "canonical-root" => o.canonical_root = value()?,
            "project-root" => o.project_root = PathBuf::from(value()?),
            "path-map" => {
                let v = value()?;
                let (local, canonical) = v
                    .split_once('=')
                    .filter(|(_, c)| !c.is_empty())
                    .ok_or_else(|| format!("invalid --path-map `{v}`: expected LOCAL=CANONICAL"))?;
                o.path_maps.push((local.to_string(), canonical.to_string()));
            }
            "vcs" => {
                let v = value()?;
                o.vcs = match v.as_str() {
                    "auto" => None,
                    _ => Some(VcsKind::parse(&v).ok_or_else(|| format!("invalid --vcs `{v}`"))?),
                };
            }
            "base" => o.base = Some(value()?),
            "description-file" => o.description_file = Some(PathBuf::from(value()?)),
            "format" => {
                let v = value()?;
                o.format = Format::parse(&v).ok_or_else(|| format!("invalid --format `{v}`"))?;
            }
            "jobs" => {
                let v = value()?;
                let n: usize = v.parse().ok().filter(|&n| n > 0).ok_or_else(|| format!("invalid --jobs `{v}`"))?;
                o.jobs = Some(n);
            }
            "all" | "verbose" => {
                if inline.is_some() {
                    return Err(format!("option `--{name}` takes no value"));
                }
                if name == "all" {
                    o.all = true;
                } else {
                    o.verbose = true;
                }
            }
            _ => return Err(format!("unknown option `--{name}`")),
        }
    }

    if o.rules.is_empty() {
        return Err("at least one --rules is required".into());
    }
    if o.canonical_root.is_empty() {
        return Err("--canonical-root is required".into());
    }
    if o.all && !o.files.is_empty() {
        return Err("--all cannot be combined with explicit files".into());
    }
    if (o.all || !o.files.is_empty()) && (o.base.is_some() || o.vcs.is_some()) {
        return Err("--base and --vcs only apply when checking changed files".into());
    }
    Ok(if is_check { Command::Check(o) } else { Command::ListRules(o) })
}

#[cfg(test)]
mod tests {
    use super::*;

    const REQUIRED: [&str; 4] = ["--rules", "r.METADATA", "--canonical-root", "//d/p"];

    fn check(extra: &[&str]) -> Result<Command, String> {
        parse(["check"].iter().chain(REQUIRED.iter()).chain(extra.iter()).copied())
    }

    fn options(extra: &[&str]) -> Options {
        match check(extra).unwrap() {
            Command::Check(o) => o,
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn parses_check_defaults() {
        let o = options(&[]);
        assert_eq!(o.rules, [PathBuf::from("r.METADATA")]);
        assert_eq!(o.canonical_root, "//d/p");
        assert_eq!(o.project_root, PathBuf::from("."));
        assert_eq!((o.vcs, o.all, o.format, o.jobs), (None, false, Format::Text, None));
    }

    #[test]
    fn parses_all_value_styles_and_positionals() {
        let o = options(&[
            "--vcs=hg",
            "--base",
            "p4base",
            "--format=json",
            "--jobs",
            "3",
            "--verbose",
            "--rules=more.METADATA",
            "--project-root=/src",
            "--description-file",
            "d.txt",
        ]);
        assert_eq!(o.vcs, Some(VcsKind::Hg));
        assert_eq!(o.base.as_deref(), Some("p4base"));
        assert_eq!((o.format, o.jobs, o.verbose), (Format::Json, Some(3), true));
        assert_eq!(o.rules.len(), 2);
        assert_eq!(o.project_root, PathBuf::from("/src"));
        assert_eq!(o.description_file, Some(PathBuf::from("d.txt")));

        let o = options(&["a.py", "--", "--weird.py"]);
        assert_eq!(o.files, [PathBuf::from("a.py"), PathBuf::from("--weird.py")]);

        let o = options(&["--path-map", "src/pkg=//d/p", "--path-map=ci=//d/p/oss/ci"]);
        assert_eq!(
            o.path_maps,
            [("src/pkg".to_string(), "//d/p".to_string()), ("ci".to_string(), "//d/p/oss/ci".to_string())]
        );
        assert!(check(&["--path-map", "nocanonical"]).unwrap_err().contains("invalid --path-map"));
        assert!(check(&["--path-map", "x="]).unwrap_err().contains("invalid --path-map"));
    }

    #[test]
    fn rejects_bad_usage() {
        for (args, want) in [
            (vec!["check", "--canonical-root", "//x"], "--rules"),
            (vec!["check", "--rules", "r"], "--canonical-root"),
            (vec!["lint"], "unknown command"),
            (vec![], "missing command"),
        ] {
            let err = parse(args).unwrap_err();
            assert!(err.contains(want), "{err}");
        }
        for (extra, want) in [
            (vec!["--bogus"], "unknown option"),
            (vec!["-x"], "unknown option"),
            (vec!["--jobs", "0"], "invalid --jobs"),
            (vec!["--format", "xml"], "invalid --format"),
            (vec!["--vcs", "svn"], "invalid --vcs"),
            (vec!["--all", "a.py"], "cannot be combined"),
            (vec!["--all", "--base", "x"], "only apply"),
            (vec!["--all=1"], "takes no value"),
            (vec!["--base"], "requires a value"),
        ] {
            let err = check(&extra).unwrap_err();
            assert!(err.contains(want), "{extra:?}: {err}");
        }
        let err = parse(["list-rules", "--rules", "r", "--canonical-root", "//x", "--all"]).unwrap_err();
        assert!(err.contains("not valid for `list-rules`"), "{err}");
    }

    #[test]
    fn help_and_version() {
        assert_eq!(parse(["--help"]), Ok(Command::Help));
        assert_eq!(parse(["check", "--help"]), Ok(Command::Help));
        assert_eq!(parse(["--version"]), Ok(Command::Version));
    }
}
