// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Airclay LLC

//! Documentation compilation and structured metadata checks.

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use serde_json::Value;

pub(crate) fn check(root: &Path) -> std::io::Result<()> {
    check_metadata(root)?;

    let target = root.join("target").join("docs-test");
    if target.exists() {
        fs::remove_dir_all(&target)?;
    }

    let result = compile_examples(root, &target);
    let cleanup = fs::remove_dir_all(&target);
    result.and(cleanup)
}

fn compile_examples(root: &Path, target: &Path) -> std::io::Result<()> {
    let build = Command::new("cargo")
        .args(["build", "--locked", "--all-features"])
        .env("CARGO_TARGET_DIR", target)
        .current_dir(root)
        .status()?;
    if !build.success() {
        return Err(std::io::Error::other(
            "all-feature documentation build failed",
        ));
    }

    let deps = target.join("debug").join("deps");
    let rlib = bisque_rlib(&deps)?;
    let readme = Command::new("rustdoc")
        .arg("--test")
        .arg("--edition=2021")
        .arg("-Dwarnings")
        .arg("--extern")
        .arg(format!("bisque={}", rlib.display()))
        .arg("-L")
        .arg(format!("dependency={}", deps.display()))
        .arg(root.join("README.md"))
        .current_dir(root)
        .status()?;
    if !readme.success() {
        return Err(std::io::Error::other("README Rust example tests failed"));
    }

    let book = Command::new("mdbook")
        .arg("test")
        .arg("-L")
        .arg(&deps)
        .arg("docs")
        .current_dir(root)
        .status()?;
    if !book.success() {
        return Err(std::io::Error::other("mdBook snippet tests failed"));
    }
    Ok(())
}

fn bisque_rlib(deps: &Path) -> std::io::Result<PathBuf> {
    let mut matches = fs::read_dir(deps)?
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|path| {
            path.extension()
                .is_some_and(|extension| extension == "rlib")
                && path
                    .file_name()
                    .and_then(|name| name.to_str())
                    .is_some_and(|name| name.starts_with("libbisque-"))
        })
        .collect::<Vec<_>>();
    matches.sort();
    match matches.as_slice() {
        [rlib] => Ok(rlib.clone()),
        _ => Err(std::io::Error::other(format!(
            "expected one bisque rlib in {}, found {}",
            deps.display(),
            matches.len()
        ))),
    }
}

fn check_metadata(root: &Path) -> std::io::Result<()> {
    let output = Command::new("cargo")
        .args(["metadata", "--locked", "--no-deps", "--format-version=1"])
        .current_dir(root)
        .output()?;
    if !output.status.success() {
        return Err(std::io::Error::other("cargo metadata failed"));
    }
    let metadata: Value = serde_json::from_slice(&output.stdout)
        .map_err(|error| std::io::Error::other(format!("invalid cargo metadata: {error}")))?;
    let package = metadata["packages"]
        .as_array()
        .and_then(|packages| packages.iter().find(|package| package["name"] == "bisque"))
        .ok_or_else(|| std::io::Error::other("bisque package missing from cargo metadata"))?;

    let readme = fs::read_to_string(root.join("README.md"))?;
    let api_surface = fs::read_to_string(root.join("docs/src/api-surface.md"))?;
    check_version_and_msrv(&readme, package)?;

    let expected_features = cargo_features(package)?;
    let readme_features = feature_table(&readme)?;
    let api_features = feature_table(&api_surface)?;
    require_equal("README feature table", &readme_features, &expected_features)?;
    require_equal("API feature table", &api_features, &expected_features)?;

    let expected_examples = cargo_examples(package);
    let readme_examples = readme_examples(&readme);
    require_equal(
        "README example commands",
        &readme_examples,
        &expected_examples,
    )
}

fn check_version_and_msrv(readme: &str, package: &Value) -> std::io::Result<()> {
    let version = package["version"]
        .as_str()
        .ok_or_else(|| std::io::Error::other("bisque package version is missing"))?;
    let rust_version = package["rust_version"]
        .as_str()
        .ok_or_else(|| std::io::Error::other("bisque rust-version is missing"))?;
    for expected in [
        format!("bisque = \"{version}\""),
        format!("bisque = {{ version = \"{version}\", features = [\"spectral\"] }}"),
        format!("[![MSRV: {rust_version}]"),
    ] {
        if !readme.contains(&expected) {
            return Err(std::io::Error::other(format!(
                "README is missing Cargo metadata value `{expected}`"
            )));
        }
    }
    Ok(())
}

fn cargo_features(package: &Value) -> std::io::Result<BTreeMap<String, bool>> {
    let features = package["features"]
        .as_object()
        .ok_or_else(|| std::io::Error::other("bisque features are missing"))?;
    let defaults = features
        .get("default")
        .and_then(Value::as_array)
        .ok_or_else(|| std::io::Error::other("bisque default features are missing"))?
        .iter()
        .filter_map(Value::as_str)
        .collect::<BTreeSet<_>>();
    Ok(features
        .keys()
        .filter(|feature| feature.as_str() != "default")
        .map(|feature| (feature.clone(), defaults.contains(feature.as_str())))
        .collect())
}

fn feature_table(markdown: &str) -> std::io::Result<BTreeMap<String, bool>> {
    let mut lines = markdown.lines();
    lines
        .find(|line| line.trim_start().starts_with("| Feature | Default |"))
        .ok_or_else(|| std::io::Error::other("documentation has no feature table"))?;
    let _separator = lines.next();

    let mut features = BTreeMap::new();
    for line in lines.take_while(|line| line.trim_start().starts_with('|')) {
        let cells = line
            .trim()
            .trim_matches('|')
            .split('|')
            .map(str::trim)
            .collect::<Vec<_>>();
        if cells.len() < 2 {
            return Err(std::io::Error::other("malformed feature table row"));
        }
        let feature = cells[0]
            .strip_prefix('`')
            .and_then(|cell| cell.strip_suffix('`'))
            .ok_or_else(|| std::io::Error::other("feature name is not code-formatted"))?;
        let default = match cells[1] {
            "Yes" => true,
            "No" => false,
            value => {
                return Err(std::io::Error::other(format!(
                    "feature `{feature}` has invalid default value `{value}`"
                )))
            }
        };
        if features.insert(feature.to_owned(), default).is_some() {
            return Err(std::io::Error::other(format!(
                "duplicate documented feature `{feature}`"
            )));
        }
    }
    Ok(features)
}

fn cargo_examples(package: &Value) -> BTreeSet<String> {
    package["targets"]
        .as_array()
        .into_iter()
        .flatten()
        .filter(|target| {
            target["kind"]
                .as_array()
                .is_some_and(|kinds| kinds.iter().any(|kind| kind == "example"))
        })
        .filter_map(|target| target["name"].as_str().map(str::to_owned))
        .collect()
}

fn readme_examples(readme: &str) -> BTreeSet<String> {
    readme
        .lines()
        .filter_map(|line| {
            line.trim()
                .strip_prefix("cargo run --example ")
                .and_then(|rest| rest.split_whitespace().next())
                .map(str::to_owned)
        })
        .collect()
}

fn require_equal<T>(label: &str, actual: &T, expected: &T) -> std::io::Result<()>
where
    T: std::fmt::Debug + PartialEq,
{
    if actual == expected {
        Ok(())
    } else {
        Err(std::io::Error::other(format!(
            "{label} differs from Cargo metadata\nactual: {actual:?}\nexpected: {expected:?}"
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn feature_tables_parse_names_and_defaults() {
        let markdown = "| Feature | Default | Purpose |\n| --- | --- | --- |\n\
                        | `filters` | Yes | Filters |\n| `spectral` | No | FFT |\n";
        assert_eq!(
            feature_table(markdown).unwrap(),
            BTreeMap::from([("filters".to_owned(), true), ("spectral".to_owned(), false)])
        );
    }

    #[test]
    fn readme_example_commands_ignore_arguments() {
        let markdown = "cargo run --example offline_chain\n\
                        cargo run --example author_kernel --features test-support\n";
        assert_eq!(
            readme_examples(markdown),
            BTreeSet::from(["author_kernel".to_owned(), "offline_chain".to_owned()])
        );
    }
}
