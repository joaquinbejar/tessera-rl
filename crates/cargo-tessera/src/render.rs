// SPDX-License-Identifier: MIT OR Apache-2.0

//! Template rendering for `cargo tessera new` (#351): `{{key}}`
//! placeholders, and sections kept or dropped by a flag. A section is a
//! line `{{#flag}}` (kept when the flag is set) or `{{^flag}}` (kept when
//! it is not), closed by a line `{{/flag}}`; sections nest.

/// The values and flags a template is rendered with.
#[derive(Debug, Clone, Default)]
pub struct Vars {
    values: Vec<(&'static str, String)>,
    flags: Vec<(&'static str, bool)>,
}

impl Vars {
    /// Adds a `{{key}}` value.
    #[must_use]
    pub fn value(mut self, key: &'static str, value: impl Into<String>) -> Self {
        self.values.push((key, value.into()));
        self
    }

    /// Adds a section flag.
    #[must_use]
    pub fn flag(mut self, key: &'static str, on: bool) -> Self {
        self.flags.push((key, on));
        self
    }

    fn lookup_flag(&self, key: &str) -> Option<bool> {
        self.flags.iter().find(|(k, _)| *k == key).map(|(_, v)| *v)
    }

    fn lookup_value(&self, key: &str) -> Option<&str> {
        self.values
            .iter()
            .find(|(k, _)| *k == key)
            .map(|(_, v)| v.as_str())
    }
}

/// A template that does not render: a bug in the embedded templates.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum RenderError {
    /// A placeholder or section with no value or flag.
    #[error("template {template}, line {line}: unknown key `{key}`")]
    Unknown {
        /// The template.
        template: &'static str,
        /// The line, 1-based.
        line: usize,
        /// The key.
        key: String,
    },
    /// A section closed out of order, or never closed.
    #[error("template {template}, line {line}: unbalanced section `{key}`")]
    Unbalanced {
        /// The template.
        template: &'static str,
        /// The line, 1-based.
        line: usize,
        /// The key.
        key: String,
    },
}

/// A section marker line: `#`, `^` or `/`, and the flag.
fn marker(line: &str) -> Option<(char, &str)> {
    let inner = line.trim().strip_prefix("{{")?.strip_suffix("}}")?;
    let mut chars = inner.chars();
    let kind = chars.next().filter(|c| matches!(c, '#' | '^' | '/'))?;
    Some((kind, chars.as_str()))
}

/// Renders `text` (named `template` in errors) with `vars`.
///
/// # Errors
///
/// [`RenderError`] for an unknown key or an unbalanced section.
pub fn render(template: &'static str, text: &str, vars: &Vars) -> Result<String, RenderError> {
    let mut out = String::with_capacity(text.len());
    // Open sections: the flag and whether its lines are kept.
    let mut open: Vec<(&str, bool)> = Vec::new();
    for (k, line) in text.split_inclusive('\n').enumerate() {
        let number = k.saturating_add(1);
        if let Some((kind, key)) = marker(line) {
            if kind == '/' {
                match open.pop() {
                    Some((top, _)) if top == key => {}
                    _ => {
                        return Err(RenderError::Unbalanced {
                            template,
                            line: number,
                            key: key.to_owned(),
                        });
                    }
                }
            } else {
                let on = vars.lookup_flag(key).ok_or_else(|| RenderError::Unknown {
                    template,
                    line: number,
                    key: key.to_owned(),
                })?;
                open.push((key, if kind == '#' { on } else { !on }));
            }
            continue;
        }
        if open.iter().all(|(_, keep)| *keep) {
            out.push_str(&substitute(template, number, line, vars)?);
        }
    }
    if let Some((key, _)) = open.pop() {
        return Err(RenderError::Unbalanced {
            template,
            line: text.lines().count(),
            key: key.to_owned(),
        });
    }
    Ok(out)
}

/// Replaces every `{{key}}` of one line.
fn substitute(
    template: &'static str,
    number: usize,
    line: &str,
    vars: &Vars,
) -> Result<String, RenderError> {
    let mut out = String::with_capacity(line.len());
    let mut rest = line;
    while let Some(start) = rest.find("{{") {
        let (before, after) = rest.split_at(start);
        out.push_str(before);
        let after = after.get(2..).unwrap_or("");
        let Some(end) = after.find("}}") else {
            return Err(RenderError::Unknown {
                template,
                line: number,
                key: after.trim_end().to_owned(),
            });
        };
        let (key, tail) = after.split_at(end);
        let value = vars.lookup_value(key).ok_or_else(|| RenderError::Unknown {
            template,
            line: number,
            key: key.to_owned(),
        })?;
        out.push_str(value);
        rest = tail.get(2..).unwrap_or("");
    }
    out.push_str(rest);
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::{RenderError, Vars, render};

    fn vars() -> Vars {
        Vars::default()
            .value("name", "grid")
            .flag("snap", true)
            .flag("gym", false)
    }

    #[test]
    fn test_render_values_and_sections_substituted() {
        let text = "a {{name}} b {{name}}\n{{#snap}}\nkept\n{{^gym}}\nnested kept\n{{/gym}}\n{{/snap}}\n{{#gym}}\ndropped\n{{/gym}}\n{{^snap}}\ndropped\n{{/snap}}\nend\n";
        assert_eq!(
            render("t", text, &vars()).unwrap(),
            "a grid b grid\nkept\nnested kept\nend\n"
        );
    }

    #[test]
    fn test_render_unknown_key_refused() {
        assert_eq!(
            render("t", "x {{nope}}\n", &vars()).unwrap_err(),
            RenderError::Unknown {
                template: "t",
                line: 1,
                key: "nope".to_owned(),
            }
        );
        assert!(matches!(
            render("t", "{{#nope}}\n{{/nope}}\n", &vars()),
            Err(RenderError::Unknown { .. })
        ));
        assert!(matches!(
            render("t", "x {{name\n", &vars()),
            Err(RenderError::Unknown { .. })
        ));
    }

    #[test]
    fn test_render_unbalanced_sections_refused() {
        for text in [
            "{{#snap}}\nx\n",
            "{{/snap}}\n",
            "{{#snap}}\n{{#gym}}\n{{/snap}}\n{{/gym}}\n",
        ] {
            assert!(
                matches!(
                    render("t", text, &vars()),
                    Err(RenderError::Unbalanced { .. })
                ),
                "{text:?}"
            );
        }
    }
}
