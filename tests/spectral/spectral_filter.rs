// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Airclay LLC

//! Contract tests for `SpectralFilter`.
//!
//! Covers full-band passthrough, latency, low-pass and high-pass behavior, flush
//! behavior, block-size invariance, and reset behavior.
#![cfg(feature = "spectral")]
// Standard transform notation: n size, h hop, x signal, p/i/k indices, t time.
#![allow(clippy::many_single_char_names)]

use std::f64::consts::TAU;

use bisque::processor::{AudioBlockMut, DspError, ProcessContext, Processor};
use bisque::spectral::{SpectralFilter, SpectralFilterSettings, Window};
use bisque::testing::{bits_eq, Buffers, Contract};

const FS: f64 = 48_000.0;
const N: usize = 1024;
const H: usize = 512;

/// Component amplitude at `freq` via a one-bin DTFT.
fn amp_at(signal: &[f64], freq: f64) -> f64 {
    let (mut re, mut im) = (0.0, 0.0);
    for (n, &s) in signal.iter().enumerate() {
        let ang = TAU * freq * n as f64 / FS;
        re += s * ang.cos();
        im -= s * ang.sin();
    }
    2.0 * (re * re + im * im).sqrt() / signal.len() as f64
}

/// Drain the latency tail of `proc` into the body output (the `BodyFlush` shape).
fn append_flush(c: &Contract, proc: &mut SpectralFilter, mut out: Buffers) -> Buffers {
    let look = Processor::<f32>::latency(proc);
    let nch = out.len();
    let mut tail = vec![vec![0.0f32; look]; nch];
    let produced = {
        // The host cap is the `look`-frame stage itself.
        let mut planes: Vec<&mut [f32]> = tail.iter_mut().map(Vec::as_mut_slice).collect();
        let mut block = AudioBlockMut::new(&mut planes);
        Processor::<f32>::flush(proc, &mut block)
    };
    let _ = c;
    for (ch, t) in out.iter_mut().zip(&tail) {
        ch.extend_from_slice(&t[..produced.frames]);
    }
    out
}

/// Build, prepare, drive the body at `block`, then append the flush tail.
fn drive_full(make: impl Fn() -> SpectralFilter, input: &[Vec<f32>], block: usize) -> Buffers {
    let c = Contract::default();
    let mut proc = make();
    Processor::<f32>::prepare(&mut proc, c.spec).expect("prepare");
    let body = c.run_split_reusing(&mut proc, input, block);
    append_flush(&c, &mut proc, body)
}

/// A two-channel test signal.
fn stereo(len: usize, f: impl Fn(usize) -> f32) -> Buffers {
    let ch: Vec<f32> = (0..len).map(&f).collect();
    vec![ch.clone(), ch]
}

#[test]
fn passthrough_reconstructs_the_interior() {
    // Keeping every bin delays the input by N across the overlapped interior.
    let len = 16_000;
    let input = stereo(len, |i| {
        let t = i as f64;
        (0.5 * (t * 0.05).sin() + 0.3 * (t * 0.211).cos()) as f32
    });
    let out = drive_full(|| SpectralFilter::low_pass(N, H, 30_000.0), &input, 256);
    assert_eq!(out[0].len(), len + N, "output length is input + latency");
    for p in (N - H)..len {
        assert!(
            (out[0][N + p] - input[0][p]).abs() < 1e-5,
            "reconstruct at p={p}: {} vs {}",
            out[0][N + p],
            input[0][p]
        );
    }
}

#[test]
fn latency_is_exactly_one_window() {
    // An impulse at k reappears at N+k and the first N output samples are silent.
    let len = 4096;
    let k = 1500;
    let input = stereo(len, |i| if i == k { 1.0 } else { 0.0 });
    let out = drive_full(|| SpectralFilter::low_pass(N, H, 30_000.0), &input, 128);

    for (i, &v) in out[0][..N].iter().enumerate() {
        assert!(v.abs() < 1e-6, "latency fill not silent at {i}: {v}");
    }
    let peak = (0..out[0].len())
        .max_by(|&a, &b| out[0][a].abs().partial_cmp(&out[0][b].abs()).unwrap())
        .unwrap();
    assert_eq!(peak, N + k, "impulse must be delayed by exactly N");
}

#[test]
fn low_pass_removes_the_high_tone_and_keeps_the_low() {
    let len = 16_000;
    let (low_f, high_f) = (1_000.0, 8_000.0);
    let input = stereo(len, |i| {
        let t = i as f64;
        (0.5 * (TAU * low_f * t / FS).sin() + 0.5 * (TAU * high_f * t / FS).sin()) as f32
    });
    let out = drive_full(|| SpectralFilter::low_pass(N, H, 4_000.0), &input, 256);
    let interior: Vec<f64> = out[0][2 * N..len].iter().map(|&v| f64::from(v)).collect();
    assert!(amp_at(&interior, high_f) < 0.05, "8 kHz must be removed");
    assert!(amp_at(&interior, low_f) > 0.4, "1 kHz must survive");
}

#[test]
fn high_pass_is_the_mirror() {
    let len = 16_000;
    let (low_f, high_f) = (1_000.0, 8_000.0);
    let input = stereo(len, |i| {
        let t = i as f64;
        (0.5 * (TAU * low_f * t / FS).sin() + 0.5 * (TAU * high_f * t / FS).sin()) as f32
    });
    let out = drive_full(|| SpectralFilter::high_pass(N, H, 4_000.0), &input, 256);
    let interior: Vec<f64> = out[0][2 * N..len].iter().map(|&v| f64::from(v)).collect();
    assert!(amp_at(&interior, low_f) < 0.05, "1 kHz must be removed");
    assert!(amp_at(&interior, high_f) > 0.4, "8 kHz must survive");
}

#[test]
fn flush_conserves_and_reports_done() {
    let len = 8_000;
    let input = stereo(len, |i| (TAU * 1_000.0 * i as f64 / FS).sin() as f32 * 0.5);
    let c = Contract::default();
    let mut proc = SpectralFilter::low_pass(N, H, 30_000.0);
    Processor::<f32>::prepare(&mut proc, c.spec).unwrap();
    let body = c.run_split_reusing(&mut proc, &input, 256);

    let mut tail = vec![vec![0.0f32; N]; 2];
    let produced = {
        let mut planes: Vec<&mut [f32]> = tail.iter_mut().map(Vec::as_mut_slice).collect();
        let mut block = AudioBlockMut::new(&mut planes);
        Processor::<f32>::flush(&mut proc, &mut block)
    };
    assert_eq!(produced.frames, N, "flush drains exactly N frames");
    assert!(produced.done, "flush reports done");
    assert_eq!(
        body[0].len() + produced.frames,
        len + N,
        "total output = L + N"
    );

    // Energy is conserved across the interior.
    let e_in: f64 = input[0][(N - H)..len]
        .iter()
        .map(|&v| f64::from(v).powi(2))
        .sum();
    let mut full = body[0].clone();
    full.extend_from_slice(&tail[0]);
    let e_out: f64 = ((N - H)..len).map(|p| f64::from(full[N + p]).powi(2)).sum();
    assert!(
        (e_out / e_in - 1.0).abs() < 1e-3,
        "energy {e_out} vs {e_in}"
    );
}

/// Flush up to `frames` and return the produced count.
fn flush_frames(proc: &mut SpectralFilter, frames: usize) -> usize {
    let mut stage: Buffers = vec![vec![0.0f32; frames]; 2];
    let mut planes: Vec<&mut [f32]> = stage.iter_mut().map(Vec::as_mut_slice).collect();
    let mut block = AudioBlockMut::new(&mut planes);
    Processor::<f32>::flush(proc, &mut block).frames
}

#[test]
fn processing_new_input_restarts_the_drain() {
    // A drain delivers at most the N-frame tail, and a process, flush,
    // process, flush sequence treats each drain independently, so the second
    // stream's tail drains in full.
    let len = 4_096;
    let input = stereo(len, |i| ((i as f64 * 0.05).sin() * 0.5) as f32);
    let c = Contract::default();
    let mut proc = SpectralFilter::low_pass(N, H, 30_000.0);
    Processor::<f32>::prepare(&mut proc, c.spec).expect("prepare");

    let _ = c.run_split_reusing(&mut proc, &input, 256);
    let first = flush_frames(&mut proc, N);
    assert_eq!(first, N, "the first drain yields the whole tail");
    let empty = flush_frames(&mut proc, N);
    assert_eq!(empty, 0, "an exhausted drain writes nothing");

    let _ = c.run_split_reusing(&mut proc, &input, 256);
    let second = flush_frames(&mut proc, N);
    assert_eq!(second, N, "a new process block must start a fresh drain");
}

#[test]
fn invalid_band_edges_are_rejected() {
    // `high_hz <= low_hz` is false for NaN, so without an explicit check a
    // NaN edge would flow into the bin index casts as a nonsense filter.
    // Infinity stays valid: `high_pass` uses an unbounded top edge on purpose.
    let spec = Contract::default().spec;
    for (lo, hi) in [(f64::NAN, 6_000.0), (100.0, f64::NAN), (f64::NAN, f64::NAN)] {
        let mut p = SpectralFilter::band(N, H, lo, hi);
        assert!(
            matches!(
                Processor::<f32>::prepare(&mut p, spec),
                Err(DspError::InvalidParam(_))
            ),
            "band({lo}, {hi}) must be rejected"
        );
    }
    let mut hp = SpectralFilter::high_pass(N, H, 8_000.0);
    assert!(
        Processor::<f32>::prepare(&mut hp, spec).is_ok(),
        "an infinite high edge stays valid"
    );

    for (lo, hi) in [
        (-200.0, -100.0),
        (f64::NEG_INFINITY, 6_000.0),
        (100.0, f64::NEG_INFINITY),
    ] {
        let mut p = SpectralFilter::band(N, H, lo, hi);
        assert!(matches!(
            Processor::<f32>::prepare(&mut p, spec),
            Err(DspError::InvalidParam(_))
        ));
    }
}

#[test]
fn hop_must_be_within_the_documented_inclusive_range() {
    let spec = Contract::default().spec;
    for hop in [0, N + 1] {
        let mut proc = SpectralFilter::band(N, hop, 100.0, 8_000.0);
        assert!(
            matches!(
                Processor::<f32>::prepare(&mut proc, spec),
                Err(DspError::InvalidParam(_))
            ),
            "hop {hop} must be rejected for size {N}"
        );
    }

    let mut tapered = SpectralFilter::band(N, N, 100.0, 8_000.0);
    assert!(matches!(
        Processor::<f32>::prepare(&mut tapered, spec),
        Err(DspError::InvalidParam(_))
    ));

    let mut boundary = SpectralFilter::with_settings(
        SpectralFilterSettings::new()
            .size(N)
            .hop(N)
            .window(Window::Rectangular)
            .band(100.0, 8_000.0),
    );
    assert!(
        Processor::<f32>::prepare(&mut boundary, spec).is_ok(),
        "hop == size is valid when every window-product phase is non-zero"
    );
}

#[test]
fn rectangular_window_supports_no_overlap_passthrough() {
    let n = 256;
    let len = 2_048;
    let input = stereo(len, |i| ((i as f64 * 0.037).sin() * 0.5) as f32);
    let output = drive_full(
        || {
            SpectralFilter::with_settings(
                SpectralFilterSettings::new()
                    .size(n)
                    .hop(n)
                    .window(Window::Rectangular)
                    .high_hz(30_000.0),
            )
        },
        &input,
        127,
    );
    for p in 0..len {
        assert!(
            (output[0][n + p] - input[0][p]).abs() < 1e-5,
            "no-overlap rectangular reconstruction diverged at {p}"
        );
    }
}

#[test]
fn size_must_be_at_least_two_and_may_be_odd() {
    let spec = Contract::default().spec;
    for size in [0, 1] {
        let mut proc = SpectralFilter::band(size, 1, 100.0, 8_000.0);
        assert!(
            matches!(
                Processor::<f32>::prepare(&mut proc, spec),
                Err(DspError::InvalidParam(_))
            ),
            "size {size} must be rejected"
        );
    }

    let mut boundary = SpectralFilter::band(2, 1, 100.0, 8_000.0);
    assert!(
        Processor::<f32>::prepare(&mut boundary, spec).is_ok(),
        "size 2 is the valid lower boundary"
    );

    let mut odd = SpectralFilter::band(255, 85, 100.0, 8_000.0);
    assert!(
        Processor::<f32>::prepare(&mut odd, spec).is_ok(),
        "odd transform sizes are supported"
    );
}

#[test]
fn odd_size_passthrough_reconstructs_the_interior() {
    let n = 255;
    let hop = 85;
    let len = 4_000;
    let input = stereo(len, |i| ((i as f64 * 0.037).sin() * 0.5) as f32);
    let output = drive_full(|| SpectralFilter::low_pass(n, hop, 30_000.0), &input, 127);
    for p in (n - hop)..len {
        assert!(
            (output[0][n + p] - input[0][p]).abs() < 1e-5,
            "odd-size reconstruction diverged at {p}"
        );
    }
}

#[test]
fn zero_frame_process_does_not_restart_a_partial_drain() {
    let input = stereo(4_096, |i| ((i as f64 * 0.05).sin() * 0.5) as f32);
    let contract = Contract::default();
    let mut proc = SpectralFilter::low_pass(N, H, 30_000.0);
    Processor::<f32>::prepare(&mut proc, contract.spec).expect("prepare");
    let _ = contract.run_split_reusing(&mut proc, &input, 256);

    let first = flush_frames(&mut proc, 100);
    assert_eq!(first, 100);

    let inputs: [&[f32]; 2] = [&[], &[]];
    let mut left: [f32; 0] = [];
    let mut right: [f32; 0] = [];
    let mut outputs: [&mut [f32]; 2] = [&mut left, &mut right];
    let mut ctx = ProcessContext::split(&inputs, &mut outputs, 0);
    Processor::<f32>::process(&mut proc, &mut ctx);

    assert_eq!(
        flush_frames(&mut proc, N),
        N - 100,
        "an empty process block must not restart the drain"
    );
}

#[test]
fn output_is_block_size_invariant() {
    // Internal hop buffering produces byte-identical body+flush output across
    // host block splits.
    let len = 12_000;
    let input = stereo(len, |i| (i as f64 * 0.037).sin() as f32 * 0.6);
    let make = || SpectralFilter::low_pass(N, H, 6_000.0);
    let reference = drive_full(make, &input, 997);
    for &block in &[1usize, 7, 32, 64, 128, 257] {
        let out = drive_full(make, &input, block);
        assert!(bits_eq(&out, &reference), "block {block} diverged");
    }
}

#[test]
fn reset_equivalence_no_state_leak() {
    let len = 6_000;
    let input = stereo(len, |i| (i as f64 * 0.041).sin() as f32 * 0.5);
    let make = || SpectralFilter::low_pass(N, H, 5_000.0);
    let c = Contract::default();
    let fresh = drive_full(make, &input, 64);

    let mut proc = make();
    Processor::<f32>::prepare(&mut proc, c.spec).unwrap();
    let _ = c.run_split_reusing(&mut proc, &input, 50); // dirty the buffers
    Processor::<f32>::reset(&mut proc);
    let body = c.run_split_reusing(&mut proc, &input, 64);
    let after = append_flush(&c, &mut proc, body);

    assert!(
        bits_eq(&after, &fresh),
        "reset must reproduce a fresh filter"
    );
}
