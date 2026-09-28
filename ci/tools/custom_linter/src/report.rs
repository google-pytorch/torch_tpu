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

//! Output formatting for violations. JSON is written by hand to avoid a
//! serialization dependency; the schema is small and stable.

use std::fmt::Write as _;

use crate::model::{Severity, Violation};

/// Output format selected on the command line.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Format {
    Text,
    Json,
}

impl Format {
    pub fn parse(name: &str) -> Option<Self> {
        match name {
            "text" => Some(Self::Text),
            "json" => Some(Self::Json),
            _ => None,
        }
    }
}

/// Counts of violations by severity.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Summary {
    pub files_checked: usize,
    pub errors: usize,
    pub warnings: usize,
    pub notices: usize,
}

impl Summary {
    pub fn new(violations: &[Violation], files_checked: usize) -> Self {
        let count = |s: Severity| violations.iter().filter(|v| v.severity == s).count();
        Self {
            files_checked,
            errors: count(Severity::Error),
            warnings: count(Severity::Warning),
            notices: count(Severity::Notice),
        }
    }
}

/// Renders violations in `format`.
pub fn render(format: Format, violations: &[Violation], summary: &Summary) -> String {
    match format {
        Format::Text => render_text(violations),
        Format::Json => render_json(violations, summary),
    }
}

/// One violation per entry: `path:line: SEVERITY [RuleId] message`, with
/// continuation lines of multi-line messages indented.
fn render_text(violations: &[Violation]) -> String {
    let mut out = String::new();
    for v in violations {
        let location = match v.line {
            Some(line) => format!("{}:{line}", v.path),
            None => v.path.clone(),
        };
        let mut lines = v.message.trim().lines();
        let first = lines.next().unwrap_or("");
        let _ = writeln!(out, "{location}: {} [{}] {first}", v.severity, v.rule_id);
        for rest in lines {
            let _ = writeln!(out, "    {}", rest.trim_end());
        }
    }
    out
}

fn render_json(violations: &[Violation], summary: &Summary) -> String {
    let mut out = String::from("{\"violations\":[");
    for (i, v) in violations.iter().enumerate() {
        if i > 0 {
            out.push(',');
        }
        let line = v.line.map_or_else(|| "null".to_string(), |l| l.to_string());
        let _ = write!(
            out,
            "{{\"rule_id\":{},\"severity\":{},\"path\":{},\"line\":{line},\"message\":{}}}",
            json_string(&v.rule_id),
            json_string(v.severity.as_str()),
            json_string(&v.path),
            json_string(v.message.trim()),
        );
    }
    let _ = writeln!(
        out,
        "],\"summary\":{{\"files_checked\":{},\"errors\":{},\"warnings\":{},\"notices\":{}}}}}",
        summary.files_checked, summary.errors, summary.warnings, summary.notices
    );
    out
}

/// Encodes `s` as a JSON string literal.
fn json_string(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => {
                let _ = write!(out, "\\u{:04x}", c as u32);
            }
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn violation(line: Option<usize>, severity: Severity, message: &str) -> Violation {
        Violation { rule_id: "R".into(), severity, path: "a/b.py".into(), line, message: message.into() }
    }

    #[test]
    fn text_format() {
        let vs = [violation(Some(3), Severity::Error, "Bad.\nSecond line.  "), violation(None, Severity::Warning, "W")];
        assert_eq!(
            render(Format::Text, &vs, &Summary::default()),
            "a/b.py:3: ERROR [R] Bad.\n    Second line.\na/b.py: WARNING [R] W\n"
        );
    }

    #[test]
    fn json_format_escapes_and_summarizes() {
        let vs = [violation(None, Severity::Notice, "say \"hi\"\\\n\u{1}")];
        let summary = Summary::new(&vs, 7);
        assert_eq!(
            render(Format::Json, &vs, &summary),
            "{\"violations\":[{\"rule_id\":\"R\",\"severity\":\"NOTICE\",\"path\":\"a/b.py\",\"line\":null,\
             \"message\":\"say \\\"hi\\\"\\\\\\n\\u0001\"}],\
             \"summary\":{\"files_checked\":7,\"errors\":0,\"warnings\":0,\"notices\":1}}\n"
        );
    }
}
