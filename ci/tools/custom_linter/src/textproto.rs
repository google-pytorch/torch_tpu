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

//! A lenient, schema-less parser for the protobuf text format.
//!
//! Rule files such as `METADATA` reference many internal proto schemas and
//! extensions. Rather than depending on those schemas, this parser builds a
//! generic tree of fields and lets each rule loader pick the fields it
//! understands. Unknown fields are preserved in the tree and simply ignored by
//! consumers.

use std::fmt;

/// A parse error with a 1-based source position.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParseError {
    pub line: usize,
    pub column: usize,
    pub message: String,
}

impl fmt::Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}:{}: {}", self.line, self.column, self.message)
    }
}

impl std::error::Error for ParseError {}

/// A scalar or nested message value.
#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    /// A (possibly concatenated) string literal, with escapes decoded.
    Str(String),
    /// An unquoted token: identifier, enum value, boolean or number.
    Ident(String),
    Message(Message),
}

/// A single `name: value` entry. Repeated fields appear as multiple entries.
#[derive(Debug, Clone, PartialEq)]
pub struct Field {
    /// Field name. Extensions keep their bracketed form, e.g. `[a.b.c]`.
    pub name: String,
    pub value: Value,
    /// 1-based line where the field name starts.
    pub line: usize,
}

/// An ordered list of fields.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Message {
    pub fields: Vec<Field>,
}

impl Message {
    /// Returns all fields named `name`, in source order.
    pub fn fields_named<'a>(&'a self, name: &'a str) -> impl Iterator<Item = &'a Field> + 'a {
        self.fields.iter().filter(move |f| f.name == name)
    }

    /// Returns all nested messages named `name`.
    pub fn messages<'a>(&'a self, name: &'a str) -> impl Iterator<Item = (&'a Message, usize)> + 'a {
        self.fields_named(name).filter_map(|f| match &f.value {
            Value::Message(m) => Some((m, f.line)),
            _ => None,
        })
    }

    /// Returns the first nested message named `name`.
    pub fn message(&self, name: &str) -> Option<&Message> {
        self.fields.iter().find_map(|f| match &f.value {
            Value::Message(m) if f.name == name => Some(m),
            _ => None,
        })
    }

    /// Returns all string values of the repeated field `name`.
    pub fn strings<'a>(&'a self, name: &'a str) -> impl Iterator<Item = &'a str> + 'a {
        self.fields_named(name).filter_map(|f| match &f.value {
            Value::Str(s) => Some(s.as_str()),
            _ => None,
        })
    }

    /// Returns the last string value of `name` (proto "last one wins").
    pub fn string(&self, name: &str) -> Option<&str> {
        self.fields.iter().rev().find_map(|f| match &f.value {
            Value::Str(s) if f.name == name => Some(s.as_str()),
            _ => None,
        })
    }

    /// Returns the last unquoted token of `name`, e.g. an enum value.
    pub fn ident(&self, name: &str) -> Option<&str> {
        self.fields.iter().rev().find_map(|f| match &f.value {
            Value::Ident(s) if f.name == name => Some(s.as_str()),
            _ => None,
        })
    }

    /// Returns all unquoted tokens of the repeated field `name`.
    pub fn idents<'a>(&'a self, name: &'a str) -> impl Iterator<Item = &'a str> + 'a {
        self.fields_named(name).filter_map(|f| match &f.value {
            Value::Ident(s) => Some(s.as_str()),
            _ => None,
        })
    }

    /// Returns the boolean value of `name`, accepting `true`/`false`/`1`/`0`.
    pub fn bool(&self, name: &str) -> Result<Option<bool>, String> {
        match self.ident(name) {
            None => Ok(None),
            Some("true" | "True" | "t" | "1") => Ok(Some(true)),
            Some("false" | "False" | "f" | "0") => Ok(Some(false)),
            Some(other) => Err(format!("field `{name}`: expected a boolean, got `{other}`")),
        }
    }
}

/// Parses a text-format document into its top-level message.
pub fn parse(input: &str) -> Result<Message, ParseError> {
    let tokens = tokenize(input)?;
    let mut parser = Parser { tokens, pos: 0 };
    let message = parser.parse_fields(None)?;
    Ok(message)
}

#[derive(Debug, Clone, PartialEq)]
enum Tok {
    Ident(String),
    Str(String),
    Punct(char),
}

#[derive(Debug, Clone)]
struct Token {
    tok: Tok,
    line: usize,
    column: usize,
}

fn tokenize(input: &str) -> Result<Vec<Token>, ParseError> {
    let chars: Vec<char> = input.chars().collect();
    let mut tokens = Vec::new();
    let (mut i, mut line, mut col) = (0usize, 1usize, 1usize);

    let err = |line, column, message: String| ParseError { line, column, message };

    while i < chars.len() {
        let c = chars[i];
        match c {
            '\n' => {
                i += 1;
                line += 1;
                col = 1;
            }
            c if c.is_whitespace() => {
                i += 1;
                col += 1;
            }
            '#' => {
                while i < chars.len() && chars[i] != '\n' {
                    i += 1;
                }
            }
            '"' | '\'' => {
                let (start_line, start_col) = (line, col);
                let quote = c;
                i += 1;
                col += 1;
                let mut bytes: Vec<u8> = Vec::new();
                loop {
                    let Some(&ch) = chars.get(i) else {
                        return Err(err(start_line, start_col, "unterminated string".into()));
                    };
                    if ch == '\n' {
                        return Err(err(start_line, start_col, "newline in string".into()));
                    }
                    i += 1;
                    col += 1;
                    if ch == quote {
                        break;
                    }
                    if ch != '\\' {
                        let mut buf = [0u8; 4];
                        bytes.extend_from_slice(ch.encode_utf8(&mut buf).as_bytes());
                        continue;
                    }
                    let Some(&esc) = chars.get(i) else {
                        return Err(err(line, col, "unterminated escape".into()));
                    };
                    i += 1;
                    col += 1;
                    match esc {
                        'n' => bytes.push(b'\n'),
                        't' => bytes.push(b'\t'),
                        'r' => bytes.push(b'\r'),
                        'a' => bytes.push(0x07),
                        'b' => bytes.push(0x08),
                        'f' => bytes.push(0x0c),
                        'v' => bytes.push(0x0b),
                        '\\' | '\'' | '"' | '?' => bytes.push(esc as u8),
                        '0'..='7' => {
                            let mut value = esc.to_digit(8).unwrap();
                            for _ in 0..2 {
                                match chars.get(i).and_then(|c| c.to_digit(8)) {
                                    Some(d) => {
                                        value = value * 8 + d;
                                        i += 1;
                                        col += 1;
                                    }
                                    None => break,
                                }
                            }
                            if value > 0xff {
                                return Err(err(line, col, "octal escape out of range".into()));
                            }
                            bytes.push(value as u8);
                        }
                        'x' => {
                            let mut value = 0u32;
                            let mut digits = 0;
                            while digits < 2 {
                                match chars.get(i).and_then(|c| c.to_digit(16)) {
                                    Some(d) => {
                                        value = value * 16 + d;
                                        i += 1;
                                        col += 1;
                                        digits += 1;
                                    }
                                    None => break,
                                }
                            }
                            if digits == 0 {
                                return Err(err(line, col, "invalid \\x escape".into()));
                            }
                            bytes.push(value as u8);
                        }
                        other => {
                            return Err(err(line, col, format!("unknown escape `\\{other}`")));
                        }
                    }
                }
                let s = String::from_utf8_lossy(&bytes).into_owned();
                // Adjacent string literals are concatenated.
                if let Some(Token { tok: Tok::Str(prev), .. }) = tokens.last_mut() {
                    prev.push_str(&s);
                } else {
                    tokens.push(Token { tok: Tok::Str(s), line: start_line, column: start_col });
                }
            }
            '{' | '}' | '[' | ']' | '<' | '>' | ':' | ',' | ';' | '/' => {
                tokens.push(Token { tok: Tok::Punct(c), line, column: col });
                i += 1;
                col += 1;
            }
            c if c.is_ascii_alphanumeric() || c == '_' || c == '-' || c == '+' || c == '.' => {
                let (start_line, start_col) = (line, col);
                let mut s = String::new();
                while let Some(&ch) = chars.get(i) {
                    if ch.is_ascii_alphanumeric() || matches!(ch, '_' | '-' | '+' | '.') {
                        s.push(ch);
                        i += 1;
                        col += 1;
                    } else {
                        break;
                    }
                }
                tokens.push(Token { tok: Tok::Ident(s), line: start_line, column: start_col });
            }
            other => {
                return Err(err(line, col, format!("unexpected character `{other}`")));
            }
        }
    }
    Ok(tokens)
}

struct Parser {
    tokens: Vec<Token>,
    pos: usize,
}

impl Parser {
    fn peek(&self) -> Option<&Token> {
        self.tokens.get(self.pos)
    }

    fn next(&mut self) -> Option<Token> {
        let t = self.tokens.get(self.pos).cloned();
        self.pos += 1;
        t
    }

    fn error_here(&self, message: impl Into<String>) -> ParseError {
        let (line, column) = match self.peek().or(self.tokens.last()) {
            Some(t) => (t.line, t.column),
            None => (1, 1),
        };
        ParseError { line, column, message: message.into() }
    }

    fn eat_punct(&mut self, c: char) -> bool {
        if matches!(self.peek(), Some(Token { tok: Tok::Punct(p), .. }) if *p == c) {
            self.pos += 1;
            true
        } else {
            false
        }
    }

    /// Parses fields until `close` (or end of input when `close` is None).
    fn parse_fields(&mut self, close: Option<char>) -> Result<Message, ParseError> {
        let mut message = Message::default();
        loop {
            match (self.peek(), close) {
                (None, None) => return Ok(message),
                (None, Some(c)) => return Err(self.error_here(format!("expected `{c}`"))),
                (Some(Token { tok: Tok::Punct(p), .. }), Some(c)) if *p == c => {
                    self.pos += 1;
                    return Ok(message);
                }
                _ => {}
            }
            self.parse_field(&mut message)?;
            // Optional field separators.
            if !self.eat_punct(',') {
                self.eat_punct(';');
            }
        }
    }

    fn parse_field_name(&mut self) -> Result<(String, usize), ParseError> {
        let Some(token) = self.next() else {
            return Err(self.error_here("expected field name"));
        };
        match token.tok {
            Tok::Ident(name) => Ok((name, token.line)),
            Tok::Punct('[') => {
                // Extension or Any type URL: `[foo.bar]` or `[type.googleapis.com/foo.Bar]`.
                let mut name = String::from("[");
                loop {
                    match self.next() {
                        Some(Token { tok: Tok::Ident(s), .. }) => name.push_str(&s),
                        Some(Token { tok: Tok::Punct('/'), .. }) => name.push('/'),
                        Some(Token { tok: Tok::Punct(']'), .. }) => break,
                        _ => return Err(self.error_here("malformed extension name")),
                    }
                }
                name.push(']');
                Ok((name, token.line))
            }
            _ => Err(ParseError {
                line: token.line,
                column: token.column,
                message: "expected field name".into(),
            }),
        }
    }

    fn parse_field(&mut self, message: &mut Message) -> Result<(), ParseError> {
        let (name, line) = self.parse_field_name()?;
        let has_colon = self.eat_punct(':');
        if self.eat_punct('[') {
            // List syntax: `name: [v1, v2]` or `name [{..}, {..}]`.
            if self.eat_punct(']') {
                return Ok(());
            }
            loop {
                let value = self.parse_value(has_colon)?;
                message.fields.push(Field { name: name.clone(), value, line });
                if self.eat_punct(']') {
                    return Ok(());
                }
                if !self.eat_punct(',') {
                    return Err(self.error_here("expected `,` or `]` in list"));
                }
            }
        }
        let value = self.parse_value(has_colon)?;
        message.fields.push(Field { name, value, line });
        Ok(())
    }

    fn parse_value(&mut self, has_colon: bool) -> Result<Value, ParseError> {
        if self.eat_punct('{') {
            return Ok(Value::Message(self.parse_fields(Some('}'))?));
        }
        if self.eat_punct('<') {
            return Ok(Value::Message(self.parse_fields(Some('>'))?));
        }
        if !has_colon {
            return Err(self.error_here("expected `:` before scalar value"));
        }
        match self.next() {
            Some(Token { tok: Tok::Str(s), .. }) => Ok(Value::Str(s)),
            Some(Token { tok: Tok::Ident(s), .. }) => Ok(Value::Ident(s)),
            _ => {
                self.pos -= 1;
                Err(self.error_here("expected a value"))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_scalars_and_nested_messages() {
        let msg = parse(
            r#"
            # comment
            name: "a"  # trailing comment
            count: 3
            enabled: true
            status: ERROR
            child { inner: "x" }
            child: { inner: "y" }
            other < inner: "z" >
            "#,
        )
        .unwrap();
        assert_eq!(msg.string("name"), Some("a"));
        assert_eq!(msg.ident("count"), Some("3"));
        assert_eq!(msg.bool("enabled"), Ok(Some(true)));
        assert_eq!(msg.ident("status"), Some("ERROR"));
        let inners: Vec<_> =
            msg.messages("child").map(|(m, _)| m.string("inner").unwrap()).collect();
        assert_eq!(inners, ["x", "y"]);
        assert_eq!(msg.message("other").unwrap().string("inner"), Some("z"));
    }

    #[test]
    fn concatenates_adjacent_strings_across_lines() {
        let msg = parse("msg:\n  \"foo \"\n  'bar'\n  \"baz\"\n").unwrap();
        assert_eq!(msg.string("msg"), Some("foo barbaz"));
    }

    #[test]
    fn decodes_escapes() {
        let msg = parse(r#"re: "\\bos\\.(a|b)\\b\n\t\"q\" \101\x42 \'""#).unwrap();
        assert_eq!(msg.string("re"), Some("\\bos\\.(a|b)\\b\n\t\"q\" AB '"));
    }

    #[test]
    fn keeps_utf8_in_strings() {
        let msg = parse("s: \"año → ok\"").unwrap();
        assert_eq!(msg.string("s"), Some("año → ok"));
    }

    #[test]
    fn repeated_fields_and_lists() {
        let msg = parse(r#"tag: "a" tag: "b" tag: ["c", "d"] m [{x: 1}, {x: 2}] empty: []"#).unwrap();
        assert_eq!(msg.strings("tag").collect::<Vec<_>>(), ["a", "b", "c", "d"]);
        assert_eq!(msg.messages("m").count(), 2);
        assert_eq!(msg.fields_named("empty").count(), 0);
    }

    #[test]
    fn parses_extensions() {
        let msg = parse(
            r#"extended_options {
                 [devtools.copybara.service.CopybaraOptions.copybara_options] { blocking: false }
               }
               any { [type.googleapis.com/foo.Bar] { a: 1 } }"#,
        )
        .unwrap();
        let ext = msg.message("extended_options").unwrap();
        assert_eq!(ext.fields[0].name, "[devtools.copybara.service.CopybaraOptions.copybara_options]");
        assert_eq!(msg.message("any").unwrap().fields[0].name, "[type.googleapis.com/foo.Bar]");
    }

    #[test]
    fn records_field_lines() {
        let msg = parse("a: 1\n\nb {\n  c: 2\n}\n").unwrap();
        assert_eq!(msg.fields[0].line, 1);
        assert_eq!(msg.fields[1].line, 3);
        assert_eq!(msg.message("b").unwrap().fields[0].line, 4);
    }

    #[test]
    fn separators_are_optional() {
        let msg = parse("a: 1, b: 2; c: 3").unwrap();
        assert_eq!(msg.fields.len(), 3);
    }

    #[test]
    fn reports_errors_with_position() {
        let err = parse("a {\n  b: \"x\"\n").unwrap_err();
        assert!(err.message.contains("expected `}`"), "{err}");

        let err = parse("a: \"unterminated\n").unwrap_err();
        assert_eq!((err.line, err.column), (1, 4));
        assert!(err.message.contains("newline in string"));

        let err = parse("a \"x\"").unwrap_err();
        assert!(err.message.contains("expected `:`"), "{err}");

        let err = parse("a: \"\\q\"").unwrap_err();
        assert!(err.message.contains("unknown escape"), "{err}");
    }

    #[test]
    fn invalid_bool_is_an_error() {
        let msg = parse("flag: maybe").unwrap();
        assert!(msg.bool("flag").is_err());
    }
}
