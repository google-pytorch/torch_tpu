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

//! Change-description tags (`KEY=value` lines), used to disable rules.

use std::collections::HashSet;

use crate::model::Rule;

/// The set of tag keys present in a change description.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Tags {
    keys: HashSet<String>,
}

impl Tags {
    /// Extracts tags from `description`. A tag is a line of the form
    /// `KEY=value` where `KEY` is an upper-case identifier at the start of the
    /// line (leading whitespace allowed). Mentions in prose do not count.
    pub fn parse(description: &str) -> Self {
        let keys = description
            .lines()
            .filter_map(|line| {
                let (key, _) = line.trim_start().split_once('=')?;
                let key = key.trim_end();
                let mut chars = key.chars();
                let first = chars.next()?;
                let valid = first.is_ascii_uppercase()
                    && chars.all(|c| c.is_ascii_uppercase() || c.is_ascii_digit() || c == '_');
                valid.then(|| key.to_string())
            })
            .collect();
        Self { keys }
    }

    pub fn contains(&self, key: &str) -> bool {
        self.keys.contains(key)
    }

    /// Returns the first of `rule`'s disable tags present, if any.
    pub fn disabling<'a>(&self, rule: &'a Rule) -> Option<&'a str> {
        rule.disable_tags.iter().map(String::as_str).find(|t| self.contains(t))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_tag_lines() {
        let tags = Tags::parse(
            "Summary line\n\nBody mentions ALLOW_CC_TARGETS in prose.\n\
             BUG=123\n  ROLLBACK_OF=456\nMARKDOWN = true\nlower=x\nX1_Y=\n",
        );
        assert!(tags.contains("BUG"));
        assert!(tags.contains("ROLLBACK_OF"));
        assert!(tags.contains("MARKDOWN"));
        assert!(tags.contains("X1_Y"));
        assert!(!tags.contains("ALLOW_CC_TARGETS"));
        assert!(!tags.contains("lower"));
    }

    #[test]
    fn does_not_match_prefixes_or_mid_line_mentions() {
        let tags = Tags::parse("Do not set ALLOW_CC_TARGETS=1 here\nALLOW_CC_TARGETS_EXTRA=1\n");
        assert!(!tags.contains("ALLOW_CC_TARGETS"));
        assert!(tags.contains("ALLOW_CC_TARGETS_EXTRA"));
    }

    #[test]
    fn empty_description_has_no_tags() {
        assert_eq!(Tags::parse(""), Tags::default());
    }
}
