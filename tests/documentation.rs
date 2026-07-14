// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Airclay LLC

//! Documentation synchronization tests.
//!
//! These checks keep the structured public inventories aligned with the
//! feature-enabled runtime registry.

#![cfg(feature = "test-support")]

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};

use bisque::parameter::Unit;
use bisque::testing::registry::{
    meter_entries, processor_entries, variable_rate_entries, DriveMode, ProcessorAuthoring,
};

/// The documented API surface, baked in at compile time so a moved page
/// breaks the build.
const API_SURFACE: &str = include_str!("../docs/src/api-surface.md");
/// The processor catalog, baked in for runtime and parameter checks.
const PROCESSOR_CATALOG: &str = include_str!("../docs/src/processor-catalog.md");

/// Exclusion rule for `Public item` rows that are not runtime processing
/// units and therefore need no registry entry:
///
/// - suffix rule: `*Settings` (construction), `*Params` (smoothed values),
///   `*Reading` (readouts), `*Coeffs` (readouts), `*Kind` (shape selectors),
/// - `SCREAMING_SNAKE` rows are constants,
/// - rows starting with a lowercase letter are helper functions,
/// - a documented list of non-processor utility types.
fn is_excluded(item: &str) -> bool {
    /// Utility types documented in domain tables that are not host-driven
    /// processors, meters, or rate changers: an oscillator shape selector
    /// and the spectral building blocks (which have their own domain
    /// tests but no `Processor`/`Measurer`/`VariableRate` impl).
    const NON_RUNTIME_TYPES: &[&str] = &["Waveform", "Complex", "Fft", "Stft", "Window"];
    const EXCLUDED_SUFFIXES: &[&str] = &["Settings", "Params", "Reading", "Coeffs", "Kind"];
    if EXCLUDED_SUFFIXES.iter().any(|s| item.ends_with(s)) {
        return true;
    }
    if item.chars().all(|c| !c.is_lowercase()) {
        return true; // SCREAMING_SNAKE constant
    }
    if item.chars().next().is_some_and(char::is_lowercase) {
        return true; // helper function
    }
    NON_RUNTIME_TYPES.contains(&item)
}

/// Whether the Cargo feature named in a domain section is enabled for this
/// test build.
fn feature_enabled(feature: &str) -> bool {
    match feature {
        "filters" => cfg!(feature = "filters"),
        "dynamics" => cfg!(feature = "dynamics"),
        "mastering" => cfg!(feature = "mastering"),
        "analysis" => cfg!(feature = "analysis"),
        "generators" => cfg!(feature = "generators"),
        "time" => cfg!(feature = "time"),
        "repair" => cfg!(feature = "repair"),
        "spectral" => cfg!(feature = "spectral"),
        other => panic!("public documentation names an unknown domain feature `{other}`"),
    }
}

/// Read the feature line that follows a domain heading.
fn catalog_feature(line: &str) -> Option<&str> {
    line.trim()
        .strip_prefix("Feature `")
        .and_then(|feature| feature.strip_suffix('`'))
}

/// Return trimmed cells from a Markdown table row.
fn table_cells(line: &str) -> Vec<&str> {
    line.trim_matches('|').split('|').map(str::trim).collect()
}

/// Remove the code formatting from a single table cell.
fn code_cell(cell: &str) -> Option<&str> {
    cell.strip_prefix('`')
        .and_then(|value| value.strip_suffix('`'))
}

/// Parse the non-excluded `Public item` rows of every feature-enabled
/// domain section (`## bisque::<domain>`).
///
/// Only tables whose header row is `| Public item | Role |` are read, so
/// the `Parameter constants` tables are skipped.
fn documented_processor_names() -> BTreeSet<String> {
    let mut names = BTreeSet::new();
    let mut section_enabled = false;
    let mut awaiting_feature = false;
    let mut in_item_table = false;
    for line in API_SURFACE.lines() {
        let line = line.trim_end();
        if let Some(header) = line.strip_prefix("## ") {
            in_item_table = false;
            if let Some(module) = header
                .trim()
                .strip_prefix("`bisque::")
                .and_then(|m| m.strip_suffix('`'))
            {
                // Domain sections carry a `Feature:` line; `bisque::dsp`
                // and `bisque::testing` do not document processors.
                awaiting_feature = module != "dsp" && module != "testing";
                section_enabled = false;
            } else {
                awaiting_feature = false;
                section_enabled = false;
            }
            continue;
        }
        if awaiting_feature {
            if let Some(feature) = line
                .trim()
                .strip_prefix("Feature: `")
                .and_then(|f| f.strip_suffix('`'))
            {
                section_enabled = feature_enabled(feature);
                awaiting_feature = false;
            }
            continue;
        }
        if !section_enabled {
            continue;
        }
        if line.starts_with("| Public item |") {
            in_item_table = true;
            continue;
        }
        if !line.starts_with('|') {
            in_item_table = false;
            continue;
        }
        if !in_item_table || line.starts_with("| ---") {
            continue;
        }
        let first_cell = line
            .trim_start_matches('|')
            .split('|')
            .next()
            .unwrap_or("")
            .trim();
        let Some(item) = first_cell
            .strip_prefix('`')
            .and_then(|c| c.strip_suffix('`'))
        else {
            continue;
        };
        if !is_excluded(item) {
            names.insert(item.to_owned());
        }
    }
    names
}

/// Every registered name across the processor, meter, and variable-rate
/// registries (deduplicated: one type may register several variants).
fn registered_names() -> BTreeSet<String> {
    let mut names = BTreeSet::new();
    for e in processor_entries() {
        names.insert(e.name.to_owned());
    }
    for e in meter_entries() {
        names.insert(e.name.to_owned());
    }
    for e in variable_rate_entries() {
        names.insert(e.name.to_owned());
    }
    names
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct DocumentedRuntime {
    trait_name: String,
    io: Option<String>,
}

/// Parse runtime contracts from the feature-enabled `Type` tables.
fn catalog_runtime_contracts() -> BTreeMap<String, DocumentedRuntime> {
    let mut contracts = BTreeMap::new();
    let mut awaiting_feature = false;
    let mut section_enabled = false;
    let mut in_type_table = false;
    let mut trait_index = None;
    let mut io_index = None;

    for line in PROCESSOR_CATALOG.lines() {
        let line = line.trim_end();
        if line.starts_with("## ") {
            awaiting_feature = true;
            section_enabled = false;
            in_type_table = false;
            continue;
        }
        if awaiting_feature {
            if let Some(feature) = catalog_feature(line) {
                section_enabled = feature_enabled(feature);
                awaiting_feature = false;
            }
            continue;
        }
        if !section_enabled {
            continue;
        }
        if line.starts_with("| Type |") {
            in_type_table = true;
            let headers = table_cells(line);
            trait_index = headers.iter().position(|cell| *cell == "Trait");
            io_index = headers.iter().position(|cell| *cell == "I/O");
            continue;
        }
        if !line.starts_with('|') {
            in_type_table = false;
            continue;
        }
        if !in_type_table || line.starts_with("| ---") {
            continue;
        }
        let cells = table_cells(line);
        if let Some(item) = cells.first().and_then(|cell| code_cell(cell)) {
            if !is_excluded(item) {
                let trait_index = trait_index
                    .unwrap_or_else(|| panic!("runtime catalog table has no Trait column: {line}"));
                let trait_name = cells
                    .get(trait_index)
                    .and_then(|cell| code_cell(cell))
                    .unwrap_or_else(|| panic!("runtime trait is not code-formatted: {line}"));
                let io = io_index.map(|index| {
                    cells
                        .get(index)
                        .unwrap_or_else(|| panic!("runtime I/O cell is missing: {line}"))
                        .to_string()
                });
                let previous = contracts.insert(
                    item.to_owned(),
                    DocumentedRuntime {
                        trait_name: trait_name.to_owned(),
                        io,
                    },
                );
                assert!(previous.is_none(), "duplicate runtime catalog row `{item}`");
            }
        }
    }
    contracts
}

fn catalog_runtime_names() -> BTreeSet<String> {
    catalog_runtime_contracts().into_keys().collect()
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct ActualRuntime {
    trait_name: &'static str,
    io: Option<&'static str>,
}

fn registered_runtime_contracts() -> BTreeMap<String, ActualRuntime> {
    let mut contracts = BTreeMap::new();
    for entry in processor_entries() {
        let io = match entry.drive {
            DriveMode::Effect => "In-place",
            DriveMode::Source => "Output-only",
            DriveMode::Split => "Split",
        };
        let actual = ActualRuntime {
            trait_name: match entry.authoring {
                ProcessorAuthoring::Kernel => "Kernel",
                ProcessorAuthoring::Direct => "Processor",
            },
            io: Some(io),
        };
        if let Some(previous) = contracts.insert(entry.name.to_owned(), actual.clone()) {
            assert_eq!(
                previous, actual,
                "registry variants differ for {}",
                entry.name
            );
        }
    }
    for entry in meter_entries() {
        assert!(
            contracts
                .insert(
                    entry.name.to_owned(),
                    ActualRuntime {
                        trait_name: "Measurer",
                        io: None,
                    },
                )
                .is_none(),
            "duplicate runtime registry name {}",
            entry.name
        );
    }
    for entry in variable_rate_entries() {
        assert!(
            contracts
                .insert(
                    entry.name.to_owned(),
                    ActualRuntime {
                        trait_name: "VariableRate",
                        io: Some("Pull source"),
                    },
                )
                .is_none(),
            "duplicate runtime registry name {}",
            entry.name
        );
    }
    contracts
}

#[derive(Clone, Debug, PartialEq)]
struct DocumentedParam {
    min: f64,
    max: Option<f64>,
    unit: String,
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct ActualParam {
    min: f64,
    max: f64,
    unit: Unit,
}

fn parse_range(cell: &str) -> (f64, Option<f64>) {
    let range =
        code_cell(cell).unwrap_or_else(|| panic!("parameter range is not code-formatted: {cell}"));
    let (min, max) = range
        .split_once("..=")
        .unwrap_or_else(|| panic!("parameter range is not inclusive: {range}"));
    let min = min
        .parse()
        .unwrap_or_else(|_| panic!("parameter minimum is not numeric: {min}"));
    let max = if let Ok(value) = max.parse() {
        Some(value)
    } else {
        assert_eq!(
            max, "max_delay_ms",
            "unsupported symbolic parameter maximum: {max}"
        );
        None
    };
    (min, max)
}

/// Parse processor parameter names, numeric bounds, and units.
fn catalog_parameters() -> BTreeMap<String, BTreeMap<String, DocumentedParam>> {
    let mut parameters = BTreeMap::<String, BTreeMap<String, DocumentedParam>>::new();
    let mut awaiting_feature = false;
    let mut section_enabled = false;
    let mut in_parameter_table = false;

    for line in PROCESSOR_CATALOG.lines() {
        let line = line.trim_end();
        if line.starts_with("## ") {
            awaiting_feature = true;
            section_enabled = false;
            in_parameter_table = false;
            continue;
        }
        if awaiting_feature {
            if let Some(feature) = catalog_feature(line) {
                section_enabled = feature_enabled(feature);
                awaiting_feature = false;
            }
            continue;
        }
        if !section_enabled {
            continue;
        }
        if line.starts_with("| Constant | Name | Range | Unit |") {
            in_parameter_table = true;
            continue;
        }
        if !line.starts_with('|') {
            in_parameter_table = false;
            continue;
        }
        if !in_parameter_table || line.starts_with("| ---") {
            continue;
        }

        let cells = table_cells(line);
        assert!(
            cells.len() >= 4,
            "processor catalog parameter row has fewer than four cells: {line}"
        );
        let name = code_cell(cells[1])
            .unwrap_or_else(|| panic!("parameter name is not code-formatted: {line}"));
        let (min, max) = parse_range(cells[2]);
        let documented = DocumentedParam {
            min,
            max,
            unit: cells[3].to_owned(),
        };

        for constant in cells[0].split(',') {
            let constant = constant.trim();
            let constant = code_cell(constant)
                .unwrap_or_else(|| panic!("parameter constant is not code-formatted: {line}"));
            let (type_name, _) = constant
                .split_once("::")
                .unwrap_or_else(|| panic!("parameter constant has no owning type: {constant}"));
            let previous = parameters
                .entry(type_name.to_owned())
                .or_default()
                .insert(name.to_owned(), documented.clone());
            assert!(
                previous.is_none(),
                "duplicate catalog parameter `{type_name}::{name}`"
            );
        }
    }
    parameters
}

/// Gather parameter metadata and require duplicate runtime variants to
/// publish the same metadata for their shared type name.
fn registered_parameters() -> BTreeMap<String, BTreeMap<String, ActualParam>> {
    let mut parameters = BTreeMap::new();
    for entry in processor_entries() {
        let processor = (entry.make)();
        let parameter_count = processor.param_info().len();
        let current = processor
            .param_info()
            .iter()
            .map(|info| {
                (
                    info.name.to_owned(),
                    ActualParam {
                        min: info.range.0,
                        max: info.range.1,
                        unit: info.unit,
                    },
                )
            })
            .collect::<BTreeMap<_, _>>();
        assert_eq!(
            current.len(),
            parameter_count,
            "{} exposes duplicate parameter names",
            entry.name
        );
        if current.is_empty() {
            continue;
        }
        if let Some(previous) = parameters.insert(entry.name.to_owned(), current.clone()) {
            assert_eq!(
                previous, current,
                "registry variants for {} expose different parameter metadata",
                entry.name
            );
        }
    }
    parameters
}

fn documented_unit(unit: Unit) -> &'static str {
    match unit {
        Unit::Db => "dB",
        Unit::Hz => "Hz",
        Unit::Ms => "ms",
        Unit::Q => "Q",
        Unit::Linear => "Linear",
        _ => panic!("processor catalog has no spelling for unit {unit:?}"),
    }
}

fn collect_markdown_tree(dir: &Path, files: &mut Vec<PathBuf>) {
    for entry in fs::read_dir(dir).unwrap_or_else(|error| {
        panic!(
            "could not read documentation directory {}: {error}",
            dir.display()
        )
    }) {
        let path = entry.expect("documentation directory entry").path();
        if path.is_dir() {
            collect_markdown_tree(&path, files);
        } else if path.extension().is_some_and(|extension| extension == "md") {
            files.push(path);
        }
    }
}

fn markdown_sources() -> Vec<PathBuf> {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let mut files = fs::read_dir(root)
        .expect("workspace root")
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|path| path.extension().is_some_and(|extension| extension == "md"))
        .collect::<Vec<_>>();
    collect_markdown_tree(&root.join("docs/src"), &mut files);
    collect_markdown_tree(&root.join("benches"), &mut files);
    files.sort();
    files
}

/// Extract inline Markdown links outside fenced code blocks.
fn markdown_links(markdown: &str) -> Vec<(usize, String)> {
    let mut links = Vec::new();
    let mut fenced = false;
    for (index, line) in markdown.lines().enumerate() {
        let trimmed = line.trim_start();
        if trimmed.starts_with("```") || trimmed.starts_with("~~~") {
            fenced = !fenced;
            continue;
        }
        if fenced {
            continue;
        }

        let mut rest = line;
        while let Some(start) = rest.find("](") {
            let after = &rest[start + 2..];
            let Some(end) = after.find(')') else {
                break;
            };
            let target = after[..end]
                .split_whitespace()
                .next()
                .unwrap_or("")
                .trim_matches(['<', '>']);
            if !target.is_empty() {
                links.push((index + 1, target.to_owned()));
            }
            rest = &after[end + 1..];
        }
    }
    links
}

fn local_link_target(target: &str) -> Option<&str> {
    if target.starts_with('#')
        || target.starts_with("http://")
        || target.starts_with("https://")
        || target.starts_with("mailto:")
    {
        return None;
    }
    target
        .split(['#', '?'])
        .next()
        .filter(|path| !path.is_empty())
}

#[test]
fn every_documented_processor_row_has_a_registry_entry_and_vice_versa() {
    let documented = documented_processor_names();
    let registered = registered_names();
    // Parser sanity: with any domain feature enabled, the page must yield
    // rows (an empty parse would make the set comparison vacuous).
    let any_domain = [
        "filters",
        "dynamics",
        "mastering",
        "analysis",
        "generators",
        "time",
        "repair",
        "spectral",
    ]
    .iter()
    .any(|f| feature_enabled(f));
    assert!(
        !any_domain || !documented.is_empty(),
        "no processor rows parsed from docs/src/api-surface.md; the parser or page moved"
    );
    let missing: Vec<&String> = documented.difference(&registered).collect();
    assert!(
        missing.is_empty(),
        "documented in docs/src/api-surface.md but missing from \
         src/testing/registry.rs: {missing:?}"
    );
    let extra: Vec<&String> = registered.difference(&documented).collect();
    assert!(
        extra.is_empty(),
        "registered in src/testing/registry.rs but not documented in \
         docs/src/api-surface.md: {extra:?}"
    );
}

#[test]
fn processor_catalog_runtime_inventory_matches_registry() {
    let documented = catalog_runtime_names();
    let registered = registered_names();
    let missing: Vec<&String> = registered.difference(&documented).collect();
    assert!(
        missing.is_empty(),
        "registered runtime types missing from docs/src/processor-catalog.md: {missing:?}"
    );
    let extra: Vec<&String> = documented.difference(&registered).collect();
    assert!(
        extra.is_empty(),
        "runtime types documented in docs/src/processor-catalog.md but missing from the registry: {extra:?}"
    );
}

#[test]
fn processor_catalog_runtime_contracts_match_registry() {
    let documented = catalog_runtime_contracts();
    let registered = registered_runtime_contracts();
    assert_eq!(
        documented.keys().collect::<BTreeSet<_>>(),
        registered.keys().collect::<BTreeSet<_>>(),
        "processor catalog runtime contract rows must match the registry"
    );

    for (name, actual) in registered {
        let expected = &documented[&name];
        assert_eq!(
            expected.trait_name, actual.trait_name,
            "processor catalog trait differs for {name}"
        );
        assert_eq!(
            expected.io.as_deref(),
            actual.io,
            "processor catalog I/O differs for {name}"
        );
    }
}

#[test]
fn processor_catalog_parameter_metadata_matches_registry() {
    let documented = catalog_parameters();
    let registered = registered_parameters();
    let documented_types = documented.keys().collect::<BTreeSet<_>>();
    let registered_types = registered.keys().collect::<BTreeSet<_>>();
    assert_eq!(
        documented_types, registered_types,
        "processor catalog parameter tables must match parameterized registry types"
    );

    for (type_name, actual_params) in registered {
        let documented_params = &documented[&type_name];
        let documented_names = documented_params.keys().collect::<BTreeSet<_>>();
        let actual_names = actual_params.keys().collect::<BTreeSet<_>>();
        assert_eq!(
            documented_names, actual_names,
            "processor catalog parameter names differ for {type_name}"
        );

        for (name, actual) in actual_params {
            let expected = &documented_params[&name];
            assert_eq!(
                expected.min, actual.min,
                "catalog minimum differs for {type_name}::{name}"
            );
            if let Some(max) = expected.max {
                assert_eq!(
                    max, actual.max,
                    "catalog maximum differs for {type_name}::{name}"
                );
            }
            assert_eq!(
                expected.unit,
                documented_unit(actual.unit),
                "catalog unit differs for {type_name}::{name}"
            );
        }
    }
}

#[test]
fn every_book_page_is_listed_in_summary() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let docs = root.join("docs/src");
    let mut pages = Vec::new();
    collect_markdown_tree(&docs, &mut pages);
    let actual = pages
        .into_iter()
        .filter(|path| path.file_name().is_some_and(|name| name != "SUMMARY.md"))
        .map(|path| {
            path.strip_prefix(&docs)
                .expect("book page is under docs/src")
                .to_string_lossy()
                .replace('\\', "/")
        })
        .collect::<BTreeSet<_>>();
    let summary = fs::read_to_string(docs.join("SUMMARY.md")).expect("docs summary");
    let listed = markdown_links(&summary)
        .into_iter()
        .filter_map(|(_, target)| local_link_target(&target).map(str::to_owned))
        .filter(|target| {
            Path::new(target)
                .extension()
                .is_some_and(|extension| extension.eq_ignore_ascii_case("md"))
        })
        .map(|target| target.replace('\\', "/"))
        .collect::<BTreeSet<_>>();
    assert_eq!(
        listed, actual,
        "docs/src/SUMMARY.md must list every book page"
    );
}

#[test]
fn local_documentation_links_resolve() {
    for source in markdown_sources() {
        let markdown = fs::read_to_string(&source)
            .unwrap_or_else(|error| panic!("could not read {}: {error}", source.display()));
        for (line, target) in markdown_links(&markdown) {
            let Some(target) = local_link_target(&target) else {
                continue;
            };
            let resolved = source
                .parent()
                .expect("documentation file has a parent")
                .join(target);
            assert!(
                resolved.exists(),
                "{}:{line} links to missing local path `{target}`",
                source.display()
            );
        }
    }
}
