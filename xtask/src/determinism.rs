// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Airclay LLC

//! Deterministic-source lint for library source and the listening renderer.

use std::fs;
use std::path::{Path, PathBuf};

const TRANSCENDENTALS: &[&str] = &[
    "sin", "cos", "sin_cos", "tan", "asin", "acos", "atan", "atan2", "sinh", "cosh", "tanh",
    "asinh", "acosh", "atanh", "exp", "exp2", "exp_m1", "ln", "ln_1p", "log", "log2", "log10",
    "powf", "sqrt", "cbrt", "hypot",
];

#[derive(Debug)]
struct Diagnostic {
    path: PathBuf,
    line: usize,
    message: &'static str,
    source: String,
}

/// Run the deterministic-source lint and print GitHub Actions compatible
/// diagnostics.
pub(crate) fn check(root: &Path) -> bool {
    let src = root.join("src");
    let mut files = Vec::new();
    if let Err(error) = collect_rs_files(&src, &mut files) {
        let file = display_path(root, &src);
        println!("{file}:1: could not enumerate deterministic source: {error}");
        println!("::error file={file},line=1::could not enumerate deterministic source: {error}");
        return false;
    }
    files.push(root.join("xtask/src/listen.rs"));
    files.sort();

    let mut diagnostics = Vec::new();
    for path in files {
        let Ok(text) = fs::read_to_string(&path) else {
            diagnostics.push(Diagnostic {
                path,
                line: 0,
                message: "could not read Rust source file",
                source: String::new(),
            });
            continue;
        };
        check_file(root, &path, &text, &mut diagnostics);
    }

    if diagnostics.is_empty() {
        println!(
            "No FMA contraction, stray std transcendental, or misplaced test-oracle tag found."
        );
        return true;
    }

    for d in diagnostics {
        let file = display_path(root, &d.path);
        let line = d.line.max(1);
        println!("{}:{}: {}", file, line, d.message);
        if !d.source.trim().is_empty() {
            println!("    {}", d.source.trim());
        }
        println!("::error file={},line={}::{}", file, line, d.message);
    }
    false
}

fn collect_rs_files(dir: &Path, out: &mut Vec<PathBuf>) -> std::io::Result<()> {
    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        if path.is_dir() {
            collect_rs_files(&path, out)?;
        } else if path.extension().is_some_and(|ext| ext == "rs") {
            out.push(path);
        }
    }
    Ok(())
}

fn check_file(root: &Path, path: &Path, text: &str, diagnostics: &mut Vec<Diagnostic>) {
    let rel = normalized_rel(root, path);
    let is_math_wrapper = rel == "src/dsp/math.rs";
    let mut oracle_state = OracleState::default();

    for (line_idx, line) in text.lines().enumerate() {
        let line_no = line_idx + 1;

        if has_fma_or_fast_math(line) {
            diagnostics.push(Diagnostic {
                path: path.to_path_buf(),
                line: line_no,
                message: "FMA contraction / fast-math rejected in deterministic source",
                source: line.to_owned(),
            });
        }

        if !is_math_wrapper && !line.contains("test-oracle:") && has_bare_transcendental(line) {
            diagnostics.push(Diagnostic {
                path: path.to_path_buf(),
                line: line_no,
                message: "bare std transcendental in deterministic source; use bisque::dsp::math",
                source: line.to_owned(),
            });
        }

        oracle_state.observe(path, line_no, line, diagnostics);
    }
}

fn has_fma_or_fast_math(line: &str) -> bool {
    line.contains("mul_add")
        || line.contains("fmuladd")
        || line.contains("fast_math")
        || line.contains("fast-math")
}

fn has_bare_transcendental(line: &str) -> bool {
    let compact: String = line.chars().filter(|c| !c.is_whitespace()).collect();
    TRANSCENDENTALS.iter().any(|name| {
        let method = format!(".{name}(");
        let f32_assoc = format!("f32::{name}(");
        let f64_assoc = format!("f64::{name}(");
        let f32_turbofish = format!("<f32>::{name}(");
        let f64_turbofish = format!("<f64>::{name}(");
        compact.contains(&method)
            || compact.contains(&f32_assoc)
            || compact.contains(&f64_assoc)
            || compact.contains(&f32_turbofish)
            || compact.contains(&f64_turbofish)
    })
}

#[derive(Default)]
struct OracleState {
    pending_cfg_test: bool,
    in_trailing_tests: bool,
}

impl OracleState {
    fn observe(
        &mut self,
        path: &Path,
        line_no: usize,
        line: &str,
        diagnostics: &mut Vec<Diagnostic>,
    ) {
        let trimmed = line.trim_start();

        if trimmed == "#[cfg(test)]" {
            self.pending_cfg_test = true;
            return;
        }

        if self.pending_cfg_test && is_ignorable_between_cfg_and_item(trimmed) {
            return;
        }

        if self.pending_cfg_test {
            self.in_trailing_tests = trimmed.starts_with("mod tests")
                && trimmed["mod tests".len()..].trim_start().starts_with('{');
            self.pending_cfg_test = false;
            return;
        }

        if self.in_trailing_tests && is_top_level_item(line) {
            diagnostics.push(Diagnostic {
                path: path.to_path_buf(),
                line: line_no,
                message: "top-level item after trailing #[cfg(test)] mod tests",
                source: line.to_owned(),
            });
        }

        if line.contains("test-oracle:") && !self.in_trailing_tests {
            diagnostics.push(Diagnostic {
                path: path.to_path_buf(),
                line: line_no,
                message: "test-oracle tag outside trailing #[cfg(test)] mod tests",
                source: line.to_owned(),
            });
        }
    }
}

fn is_ignorable_between_cfg_and_item(trimmed: &str) -> bool {
    trimmed.is_empty() || trimmed.starts_with("#[") || trimmed.starts_with("//")
}

fn is_top_level_item(line: &str) -> bool {
    if line.starts_with(char::is_whitespace) || line.trim().is_empty() || line.starts_with("//") {
        return false;
    }
    let line = line
        .strip_prefix("pub ")
        .or_else(|| {
            let rest = line.strip_prefix("pub(")?;
            let close = rest.find(')')?;
            Some(rest[close + 1..].trim_start())
        })
        .unwrap_or(line);

    [
        "use ", "mod ", "fn ", "impl", "struct ", "enum ", "trait ", "type ", "const ", "static ",
    ]
    .iter()
    .any(|prefix| line.starts_with(prefix))
}

fn display_path(root: &Path, path: &Path) -> String {
    normalized_rel(root, path)
}

fn normalized_rel(root: &Path, path: &Path) -> String {
    path.strip_prefix(root)
        .unwrap_or(path)
        .to_string_lossy()
        .replace('\\', "/")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detects_method_and_associated_transcendentals() {
        assert!(has_bare_transcendental("let y = x.sin();"));
        assert!(has_bare_transcendental("let y = f32::sin(x);"));
        assert!(has_bare_transcendental("let y = <f64>::log10(x);"));
        assert!(has_bare_transcendental("let y = x.powf(exponent);"));
        for method in [
            "sqrt", "hypot", "cbrt", "sin_cos", "exp_m1", "ln_1p", "asinh", "acosh", "atanh",
        ] {
            assert!(
                has_bare_transcendental(&format!("let y = x.{method}(arg);")),
                "{method} must be rejected"
            );
        }
        assert!(!has_bare_transcendental("let y = math::sin(x);"));
    }

    #[test]
    fn missing_source_tree_fails_closed() {
        let root = std::env::temp_dir().join(format!(
            "bisque-missing-determinism-root-{}",
            std::process::id()
        ));
        assert!(!check(&root));
    }

    #[test]
    fn reports_oracle_and_trailing_test_module_violations() {
        let root = Path::new("repo");
        let path = Path::new("repo/src/probe.rs");
        let text = "\
// test-oracle: outside test module
fn production_transcendental(x: f32) -> f32 {
    f32::sin(x)
}

#[cfg(test)]
mod tests {
    fn ok() {
        let _ = f64::cos(0.0); // test-oracle: allowed
    }
}

fn production_after_tests() {}
";
        let mut diagnostics = Vec::new();
        check_file(root, path, text, &mut diagnostics);
        let messages: Vec<_> = diagnostics.iter().map(|d| d.message).collect();

        assert!(messages.contains(&"test-oracle tag outside trailing #[cfg(test)] mod tests"));
        assert!(messages
            .contains(&"bare std transcendental in deterministic source; use bisque::dsp::math"));
        assert!(messages.contains(&"top-level item after trailing #[cfg(test)] mod tests"));
    }
}
