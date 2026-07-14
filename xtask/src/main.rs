// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Airclay LLC

//! Workspace task runner. Run with `cargo xtask <task>` (see `.cargo/config.toml`).
//!
//! Tasks:
//!
//! - `gen-snapshots --reason "<text>"` drives every registered snapshot case
//!   and rewrites `testdata/snapshots.manifest` plus the small human-readable
//!   slices. `--reason` is required so a regeneration is always deliberate;
//!   state it in the change description. It is not written to the manifest,
//!   which carries only the format meta and the case hashes.
//! - `check` regenerates the cases and verifies `testdata/` is unchanged, so
//!   `cargo xtask check` on a clean tree stays clean while any case-hash change
//!   still trips it.
//! - `check-determinism` rejects FMA contraction, fast-math, bare std
//!   transcendentals in production source, and misplaced `test-oracle:` tags.
//! - `check-docs` validates README and feature metadata and compile-tests the
//!   README and mdBook Rust examples.
//! - `listen` renders audible comparisons and smoke cases to `target/listen/`
//!   with an `index.md` describing what to listen for in each case. The bench
//!   helps find audible problems. The numeric suite decides pass/fail.
//!
//! The producer and verifier run in the default `dev` profile.

mod determinism;
mod docs;
mod listen;

use std::collections::HashSet;
use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::process::{Command, ExitCode};
use std::{env, fs};

use bisque::parameter::ParamEvent;
use bisque::testing::snapshot_cases as support;
use bisque::testing::{snapshot_hex, Buffers};

/// One driven snapshot row.
///
/// Stores the sort key, manifest line, and output slice source.
struct SnapshotRow {
    id: &'static str,
    signal: &'static str,
    line: String,
    out: Buffers,
}

fn main() -> ExitCode {
    let args: Vec<String> = env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("gen-snapshots") => {
            let reason = match parse_reason(&args[1..]) {
                Ok(reason) => reason,
                Err(msg) => {
                    eprintln!("xtask: {msg}");
                    eprintln!("usage: cargo xtask gen-snapshots --reason \"<why>\"");
                    return ExitCode::FAILURE;
                }
            };
            match gen_snapshots() {
                Ok(()) => {
                    eprintln!("xtask: regenerated snapshots (reason: {reason})");
                    ExitCode::SUCCESS
                }
                Err(e) => {
                    eprintln!("xtask: gen-snapshots failed: {e}");
                    ExitCode::FAILURE
                }
            }
        }
        Some("check") => check(),
        Some("check-docs") => match docs::check(&workspace_root()) {
            Ok(()) => ExitCode::SUCCESS,
            Err(e) => {
                eprintln!("xtask: check-docs failed: {e}");
                ExitCode::FAILURE
            }
        },
        Some("check-determinism") => {
            if determinism::check(&workspace_root()) {
                ExitCode::SUCCESS
            } else {
                ExitCode::FAILURE
            }
        }
        Some("listen") => match listen::listen(&workspace_root()) {
            Ok(()) => ExitCode::SUCCESS,
            Err(e) => {
                eprintln!("xtask: listen failed: {e}");
                ExitCode::FAILURE
            }
        },
        Some(other) => {
            eprintln!(
                "xtask: unknown task `{other}`. Try: gen-snapshots, check, check-docs, \
                 check-determinism, listen."
            );
            ExitCode::FAILURE
        }
        None => {
            eprintln!(
                "usage: cargo xtask <gen-snapshots --reason \"<why>\"|check|check-docs|\
                 check-determinism|listen>"
            );
            ExitCode::FAILURE
        }
    }
}

/// Parse the required `--reason <text>` (or `--reason=<text>`) argument.
fn parse_reason(rest: &[String]) -> Result<String, String> {
    let reason = match rest {
        [flag] if flag.starts_with("--reason=") => flag["--reason=".len()..].to_owned(),
        [flag, value] if flag == "--reason" => value.clone(),
        [] => {
            return Err(
                "gen-snapshots requires --reason \"<why>\" so every regeneration is \
                 deliberate; state the reason in the change description"
                    .to_owned(),
            )
        }
        _ => return Err(format!("unexpected gen-snapshots arguments: {rest:?}")),
    };
    // Keep the status message on one line.
    let reason = reason.replace(['\t', '\n', '\r'], " ").trim().to_owned();
    if reason.is_empty() {
        return Err("--reason must not be empty".to_owned());
    }
    Ok(reason)
}

/// The workspace root. xtask is located at `<root>/xtask`.
fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("xtask is at <root>/xtask")
        .to_path_buf()
}

/// Regenerate the committed snapshot manifest and per-case slices.
///
/// The case list is sorted before writing. Only the format meta and the case
/// hashes are written; there is no regeneration reason or timestamp.
fn gen_snapshots() -> std::io::Result<()> {
    let root = workspace_root();
    if !cfg!(target_os = "linux") {
        eprintln!(
            "xtask: note: CI regenerates snapshots on Linux; this is a {} bootstrap run. \
             Each committed case is expected to match across supported CI operating systems.",
            env::consts::OS
        );
    }

    let lock = fs::read_to_string(root.join("Cargo.lock"))?;
    let libm = read_libm_version(&lock)
        .ok_or_else(|| std::io::Error::other("Cargo.lock does not resolve libm"))?;
    let pin = format!("={libm}");
    let toml = fs::read_to_string(root.join("Cargo.toml"))?;
    if !toml.contains(&format!("libm = \"{pin}\"")) {
        return Err(std::io::Error::other(format!(
            "Cargo.toml must pin the resolved libm version as libm = \"{pin}\""
        )));
    }

    let mut lines: Vec<String> = vec![
        "# bisque snapshot manifest.".to_owned(),
        "# Generated by `cargo xtask gen-snapshots`. DO NOT EDIT BY HAND.".to_owned(),
        "# A row records output expected to be byte-exact across supported platforms.".to_owned(),
        "#".to_owned(),
        "#   meta <key> <value>".to_owned(),
        "#   case <id> <signal> <region> <events> <frames>x<channels> <algo>:<hex>".to_owned(),
        "#".to_owned(),
        "# Hash: FNV-1a-128 over f32-le-planar-v1 bytes.".to_owned(),
        "meta\tmanifest-version\t2".to_owned(),
        "meta\thash\tfnv1a-128".to_owned(),
        "meta\tcanon\tf32-le-planar-v1".to_owned(),
        format!("meta\tlibm\t{pin}"),
    ];

    // Drive each processor and VariableRate case to a row and slice.
    let mut rows: Vec<SnapshotRow> = Vec::new();
    for c in support::snapshot_cases() {
        let out = support::drive_case(&c);
        let line = manifest_row(
            c.id,
            c.signal,
            c.region.tag(),
            &encode_events(c.events),
            &out,
        );
        rows.push(SnapshotRow {
            id: c.id,
            signal: c.signal,
            line,
            out,
        });
    }
    for c in support::vr_snapshot_cases() {
        let out = support::drive_vr_case(&c);
        // A VariableRate has no per-block events and no flush tail.
        let line = manifest_row(c.id, c.signal, "stretch", "-", &out);
        rows.push(SnapshotRow {
            id: c.id,
            signal: c.signal,
            line,
            out,
        });
    }
    rows.sort_by(|a, b| (a.id, a.signal).cmp(&(b.id, b.signal)));
    remove_stale_slices(&root, &rows)?;
    for r in &rows {
        lines.push(r.line.clone());
        write_slice(&root, r.id, r.signal, &r.out)?;
    }

    let mut body = lines.join("\n");
    body.push('\n');
    fs::write(root.join("testdata/snapshots.manifest"), body)?;
    Ok(())
}

/// Regenerate, then assert the working tree under `testdata/` is unchanged.
///
/// A clean tree regenerates to a byte-identical manifest while a case-hash
/// change still shows as a diff.
fn check() -> ExitCode {
    let root = workspace_root();
    if let Err(e) = gen_snapshots() {
        eprintln!("xtask: gen-snapshots failed: {e}");
        return ExitCode::FAILURE;
    }
    // `git status --porcelain` reports tracked and untracked testdata changes.
    match Command::new("git")
        .current_dir(&root)
        .args(["status", "--porcelain", "--", "testdata"])
        .output()
    {
        Ok(out) if out.status.success() && out.stdout.is_empty() => ExitCode::SUCCESS,
        Ok(out) if out.status.success() => {
            eprint!("{}", String::from_utf8_lossy(&out.stdout));
            eprintln!(
                "xtask: snapshots are stale or untracked. run `cargo xtask gen-snapshots \
                 --reason \"<why>\"` and commit testdata/."
            );
            ExitCode::FAILURE
        }
        Ok(out) => {
            eprintln!(
                "xtask: git status failed: {}",
                String::from_utf8_lossy(&out.stderr)
            );
            ExitCode::FAILURE
        }
        Err(e) => {
            eprintln!("xtask: could not run git: {e}");
            ExitCode::FAILURE
        }
    }
}

/// Format one `case` manifest row.
///
/// The output's own dimensions are recorded.
fn manifest_row(id: &str, signal: &str, region: &str, events: &str, out: &Buffers) -> String {
    let channels = out.len();
    let frames = out.first().map_or(0, Vec::len);
    format!(
        "case\t{id}\t{signal}\t{region}\t{events}\t{frames}x{channels}\tfnv1a128:{}",
        snapshot_hex(out)
    )
}

/// Encode a case's events as a compact `paramId@absOffset=value` list (or `-`).
fn encode_events(events: &[ParamEvent]) -> String {
    if events.is_empty() {
        return "-".to_owned();
    }
    events
        .iter()
        .map(|e| format!("{}@{}={}", e.param.0, e.offset, e.value))
        .collect::<Vec<_>>()
        .join(",")
}

/// Write a small human-readable slice.
///
/// The slice is a 16-frame window from the middle of the output, written as
/// decimal values plus raw f32 hex bits.
fn write_slice(root: &Path, id: &str, signal: &str, out: &Buffers) -> std::io::Result<()> {
    let frames = out.first().map_or(0, Vec::len);
    let n = 16.min(frames);
    let start = frames.saturating_sub(n) / 2;
    let mut s = String::new();
    let _ = writeln!(
        s,
        "# snapshot slice: {id} / {signal} ({frames} frames, {} channels)",
        out.len()
    );
    let _ = writeln!(
        s,
        "# middle window [{start}, {}) per channel: <index>\\t<value>\\t<hexbits>",
        start + n
    );
    for (ch, plane) in out.iter().enumerate() {
        let _ = writeln!(s, "ch{ch}");
        for (i, &v) in plane.iter().enumerate().skip(start).take(n) {
            let _ = writeln!(s, "{i}\t{v:+.9e}\t{:08x}", v.to_bits());
        }
    }
    fs::write(
        root.join("testdata/snapshots/slices")
            .join(slice_name(id, signal)),
        s,
    )
}

fn slice_name(id: &str, signal: &str) -> String {
    format!("{id}.{signal}.slice")
}

/// Remove managed slice files that no current snapshot row will rewrite.
fn remove_stale_slices(root: &Path, rows: &[SnapshotRow]) -> std::io::Result<()> {
    let dir = root.join("testdata/snapshots/slices");
    fs::create_dir_all(&dir)?;
    let expected: HashSet<String> = rows
        .iter()
        .map(|row| slice_name(row.id, row.signal))
        .collect();
    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        if path.extension().is_some_and(|ext| ext == "slice")
            && !expected.contains(entry.file_name().to_string_lossy().as_ref())
        {
            fs::remove_file(path)?;
        }
    }
    Ok(())
}

/// Read the resolved `libm` version from `Cargo.lock`.
fn read_libm_version(lock: &str) -> Option<String> {
    let mut lines = lock.lines();
    while let Some(line) = lines.next() {
        if line.trim() == "name = \"libm\"" {
            for next in lines.by_ref() {
                let t = next.trim();
                if let Some(v) = t.strip_prefix("version = \"") {
                    return Some(v.trim_end_matches('"').to_owned());
                }
                if t.starts_with("[[") {
                    break;
                }
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn reason_is_required_and_kept_on_one_line() {
        assert!(parse_reason(&[]).is_err());
        assert!(parse_reason(&["--reason=".to_owned()]).is_err());
        assert_eq!(
            parse_reason(&["--reason".to_owned(), "  coefficient\nchange  ".to_owned()]),
            Ok("coefficient change".to_owned())
        );
        assert_eq!(
            parse_reason(&["--reason=expected output".to_owned()]),
            Ok("expected output".to_owned())
        );
    }

    #[test]
    fn libm_version_is_read_from_its_package_record() {
        let lock = r#"
[[package]]
name = "another"
version = "1.0.0"

[[package]]
name = "libm"
version = "0.2.11"
source = "registry+https://github.com/rust-lang/crates.io-index"
"#;
        assert_eq!(read_libm_version(lock), Some("0.2.11".to_owned()));
        assert_eq!(read_libm_version("[[package]]\nname = \"other\"\n"), None);
    }

    #[test]
    fn stale_snapshot_slices_are_removed_without_touching_other_files() {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system clock after Unix epoch")
            .as_nanos();
        let root = env::temp_dir().join(format!("bisque-xtask-{}-{unique}", std::process::id()));
        let slices = root.join("testdata/snapshots/slices");
        fs::create_dir_all(&slices).expect("create temporary slices directory");
        fs::write(slices.join("current.signal.slice"), "current").expect("write current slice");
        fs::write(slices.join("stale.signal.slice"), "stale").expect("write stale slice");
        fs::write(slices.join(".gitkeep"), "").expect("write unmanaged file");
        let rows = [SnapshotRow {
            id: "current",
            signal: "signal",
            line: String::new(),
            out: Vec::new(),
        }];

        remove_stale_slices(&root, &rows).expect("remove stale slices");

        assert!(slices.join("current.signal.slice").is_file());
        assert!(!slices.join("stale.signal.slice").exists());
        assert!(slices.join(".gitkeep").is_file());
        fs::remove_dir_all(root).expect("remove temporary directory");
    }
}
