// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Airclay LLC

//! The listening bench renders audible comparisons and smoke cases to
//! `target/listen/`.
//!
//! Listening checks and the numeric suite cover different risks. The
//! contract tests prove explicit properties. This bench renders audible
//! artifacts that are easier to catch by ear. Fixed inputs and settings make
//! review repeatable. Pass/fail stays with the numeric suite, and each audible
//! finding gets a numeric test.
//!
//! Signals use bisque generators and processors except for controls that are
//! explicitly labeled in the index. Ordinary cases use 24-bit PCM. The dither
//! comparison uses 16-bit PCM so quantization behavior remains audible.

use std::collections::HashSet;
use std::fmt::Write as _;
use std::fs;
use std::io::Write as _;
use std::path::Path;

use bisque::analysis::{
    linear_to_dbfs, LoudnessMeter, LoudnessMeterSettings, PeakMeter, TruePeakMeter,
};
use bisque::dsp::math;
use bisque::dynamics::{
    Compressor, CompressorSettings, Expander, ExpanderSettings, Gate, GateSettings,
};
use bisque::filters::{Biquad, BiquadSettings};
use bisque::generators::{PolyBlepOsc, SineOsc, WhiteNoise};
use bisque::mastering::{Dither, Gain, Limiter, LimiterSettings};
use bisque::parameter::{ParamEvent, ParamId};
use bisque::processor::{
    AudioBlock, AudioBlockMut, IoMode, Kernel, KernelProcessor, Measurer, ProcessContext,
    ProcessSpec, Processor, RingSource, Tail, VariableRate,
};
use bisque::spectral::SpectralFilter;
use bisque::testing::registry::{processor_entries, variable_rate_entries, DriveMode};
use bisque::time::{Delay, DelaySettings, TimeStretch, TimeStretchSettings};

const RATE: u32 = 48_000;
const BLOCK: usize = 512;

fn spec() -> ProcessSpec {
    spec_for_channels(1)
}

fn spec_for_channels(channels: usize) -> ProcessSpec {
    ProcessSpec {
        sample_rate: RATE,
        channels,
        max_block: BLOCK,
        max_memory: None,
    }
}

fn seconds(s: f64) -> usize {
    (s * f64::from(RATE)) as usize
}

/// One rendered file plus the line the index prints for it.
#[derive(Clone)]
struct Rendered {
    file: String,
    note: String,
    peak_dbfs: f64,
    frames: usize,
    channels: usize,
    bits_per_sample: u16,
}

/// One bench case with a heading, files, and listening notes.
struct Case {
    title: String,
    what: String,
    listen_for: String,
    files: Vec<Rendered>,
}

/// Samples produced by a bounded drain and whether the processor finished.
struct Drained {
    samples: Vec<f32>,
    done: bool,
}

/// Render the bench into `<root>/target/listen/` and write `index.md`.
pub fn listen(root: &Path) -> std::io::Result<()> {
    let dir = root.join("target/listen");
    if dir.exists() {
        fs::remove_dir_all(&dir)?;
    }
    fs::create_dir_all(&dir)?;

    let mut cases = vec![
        case_smoothing(&dir)?,
        case_filter_sweep(&dir)?,
        case_aliasing(&dir)?,
        case_limiter_transients(&dir)?,
        case_compressor(&dir)?,
        case_stereo_sidechain(&dir)?,
        case_expander_and_gate(&dir)?,
        case_eq_shapes(&dir)?,
        case_delay_tail(&dir)?,
        case_dither_fade(&dir)?,
        case_time_stretch(&dir)?,
        case_spectral(&dir)?,
    ];
    cases.push(case_baselines(&dir)?);
    validate_cases(&dir, &cases)?;

    let mut index = String::new();
    let _ = writeln!(index, "# bisque listening bench\n");
    let _ = writeln!(
        index,
        "Fixed inputs and settings make these renders repeatable listening\n\
         checks. They are not cross-platform snapshot files. If you hear an\n\
         audible issue, write the numeric test that catches it. Pass/fail\n\
         lives in the test suite.\n"
    );
    let _ = writeln!(
        index,
        "Listen at a moderate level first. One case asks you to raise the\n\
         volume near the end of a fade; turn it back down before the next\n\
         file.\n"
    );
    for c in &cases {
        let _ = writeln!(index, "## {}\n", c.title);
        let _ = writeln!(index, "{}\n", c.what);
        let _ = writeln!(index, "### What to listen for\n\n{}\n", c.listen_for);
        for r in &c.files {
            let channels = match r.channels {
                1 => "mono".to_owned(),
                2 => "stereo".to_owned(),
                n => format!("{n} channels"),
            };
            let _ = writeln!(
                index,
                "- [`{0}`]({0}) ({1:.2} s, {channels}, {2}-bit PCM, peak {3:.1} dBFS). {4}",
                r.file,
                r.frames as f64 / f64::from(RATE),
                r.bits_per_sample,
                r.peak_dbfs,
                r.note
            );
        }
        let _ = writeln!(index);
    }
    fs::write(dir.join("index.md"), index)?;
    println!(
        "listen: {} cases rendered to {}",
        cases.len(),
        dir.display()
    );
    Ok(())
}

// ---------------------------------------------------------------------------
// Drive helpers
// ---------------------------------------------------------------------------

/// Run a prepared processor over `buf` in BLOCK chunks.
///
/// Split processors read a copy of the current chunk and write into `buf`.
/// Every declared sidechain receives the same deterministic chunk as its main
/// input. `events_for(sample_pos)` supplies per-block events (offset 0 only).
fn run<P: Processor<f32> + ?Sized>(
    p: &mut P,
    buf: &mut [f32],
    mut events_for: impl FnMut(u64) -> Vec<ParamEvent>,
) {
    let total = buf.len();
    let mut pos = 0usize;
    while pos < total {
        let n = BLOCK.min(total - pos);
        let events = events_for(pos as u64);
        let (_, rest) = buf.split_at_mut(pos);
        let (chunk, _) = rest.split_at_mut(n);
        let input_samples = chunk.to_vec();
        let input_planes: [&[f32]; 1] = [input_samples.as_slice()];
        let sidechains: Vec<_> = (0..p.sidechain_inputs())
            .map(|_| AudioBlock::new(&input_planes))
            .collect();
        let mut planes: [&mut [f32]; 1] = [chunk];
        let mut ctx = match p.io_mode() {
            IoMode::InPlace => ProcessContext::in_place(&mut planes, pos as u64),
            IoMode::OutputOnly => ProcessContext::output_only(&mut planes, pos as u64),
            IoMode::Split => ProcessContext::split(&input_planes, &mut planes, pos as u64),
        }
        .with_events(&events)
        .with_sidechains(&sidechains);
        p.process(&mut ctx);
        pos += n;
    }
}

/// Drain a processor's tail (up to `cap` frames; ends early on `done`).
fn drain<P: Processor<f32> + ?Sized>(p: &mut P, cap: usize) -> Drained {
    let mut out = vec![0.0f32; cap];
    let mut written = 0usize;
    let mut done = false;
    while written < cap {
        let n = BLOCK.min(cap - written);
        let produced = {
            let (_, rest) = out.split_at_mut(written);
            let (chunk, _) = rest.split_at_mut(n);
            let mut planes: [&mut [f32]; 1] = [chunk];
            let mut block = AudioBlockMut::new(&mut planes);
            p.flush(&mut block)
        };
        written += produced.frames;
        done = produced.done;
        if done || produced.frames == 0 {
            break;
        }
    }
    out.truncate(written);
    Drained { samples: out, done }
}

/// Render a prepared source kernel into a fresh buffer.
fn render_source<K: Kernel<f32>>(kernel: K, frames: usize, set: &[(ParamId, f64)]) -> Vec<f32> {
    let mut p = KernelProcessor::new(kernel);
    Processor::<f32>::prepare(&mut p, spec()).expect("source prepare");
    for &(id, v) in set {
        p.set_parameter_immediate(id, v)
            .expect("known parameter id");
    }
    let mut buf = vec![0.0f32; frames];
    run(&mut p, &mut buf, |_| Vec::new());
    buf
}

fn render_kernel<K: Kernel<f32>>(kernel: K, input: &[f32]) -> Vec<f32> {
    let mut p = KernelProcessor::new(kernel);
    Processor::<f32>::prepare(&mut p, spec()).expect("kernel prepare");
    let mut out = input.to_vec();
    run(&mut p, &mut out, |_| Vec::new());
    out
}

/// Run one stereo in-place processor with an independent stereo sidechain bus.
fn run_stereo_sidechain<P: Processor<f32> + ?Sized>(
    p: &mut P,
    main: &mut [Vec<f32>],
    key: &[Vec<f32>],
) {
    debug_assert_eq!(main.len(), 2);
    debug_assert_eq!(key.len(), 2);
    let frames = main[0].len();
    debug_assert!(main.iter().all(|plane| plane.len() == frames));
    debug_assert!(key.iter().all(|plane| plane.len() == frames));
    let mut pos = 0;
    while pos < frames {
        let end = (pos + BLOCK).min(frames);
        let key_planes: Vec<&[f32]> = key.iter().map(|plane| &plane[pos..end]).collect();
        let sidechains = [AudioBlock::new(&key_planes)];
        let mut planes: Vec<&mut [f32]> =
            main.iter_mut().map(|plane| &mut plane[pos..end]).collect();
        let mut ctx =
            ProcessContext::in_place(&mut planes, pos as u64).with_sidechains(&sidechains);
        p.process(&mut ctx);
        pos = end;
    }
}

fn render_variable_rate(
    processor: &mut dyn VariableRate<f32>,
    input: Vec<Vec<f32>>,
) -> std::io::Result<Vec<Vec<f32>>> {
    let channels = input.len();
    if channels == 0 || input.iter().any(|plane| plane.len() != input[0].len()) {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "variable-rate listening input must be rectangular",
        ));
    }
    processor
        .prepare(spec_for_channels(channels))
        .map_err(|error| std::io::Error::other(error.to_string()))?;
    let mut source = RingSource::new(input);
    let mut output = vec![Vec::new(); channels];
    let mut calls = 0usize;
    loop {
        let mut stage = vec![vec![0.0f32; BLOCK]; channels];
        let produced = {
            let mut planes: Vec<&mut [f32]> = stage.iter_mut().map(Vec::as_mut_slice).collect();
            let mut block = AudioBlockMut::new(&mut planes);
            processor.process(&mut source, &mut block)
        };
        if produced.frames > BLOCK {
            return Err(std::io::Error::other(
                "variable-rate processor exceeded output capacity",
            ));
        }
        for channel in 0..channels {
            output[channel].extend_from_slice(&stage[channel][..produced.frames]);
        }
        calls += 1;
        if produced.done {
            break;
        }
        if calls > 100_000 {
            return Err(std::io::Error::other(
                "variable-rate listening render did not terminate",
            ));
        }
    }
    Ok(output)
}

fn integrated_lufs(samples: &[f32]) -> f64 {
    let duration = samples.len() as f64 / f64::from(RATE);
    let mut meter = LoudnessMeter::with_settings(
        LoudnessMeterSettings::with_max_integrated_seconds(duration.max(0.4) + 0.1),
    );
    Measurer::<f32>::prepare(&mut meter, spec()).expect("loudness meter prepare");
    for chunk in samples.chunks(BLOCK) {
        let planes: [&[f32]; 1] = [chunk];
        meter.observe(AudioBlock::new(&planes));
    }
    Measurer::<f32>::read(&meter).integrated_lufs
}

/// Match `candidate` to `reference` as closely as the real-time meter permits.
fn match_loudness(reference: &[f32], candidate: &mut [f32]) -> f64 {
    let reference_lufs = integrated_lufs(reference);
    let candidate_lufs = integrated_lufs(candidate);
    if !reference_lufs.is_finite() || !candidate_lufs.is_finite() {
        return 0.0;
    }
    let requested = reference_lufs - candidate_lufs;
    let headroom = -true_peak_of(candidate) - 0.5;
    let applied = requested.min(headroom).clamp(-120.0, 24.0);
    let mut gain = KernelProcessor::new(Gain::new());
    Processor::<f32>::prepare(&mut gain, spec()).expect("gain prepare");
    gain.set_parameter_immediate(Gain::GAIN_DB, applied)
        .expect("Gain declares GAIN_DB");
    run(&mut gain, candidate, |_| Vec::new());
    applied
}

fn append_silence(samples: &mut Vec<f32>, duration_seconds: f64) {
    samples.resize(samples.len() + seconds(duration_seconds), 0.0);
}

fn render_latency_compensated(
    processor: &mut dyn Processor<f32>,
    input: &[f32],
    tail_cap: usize,
) -> std::io::Result<Vec<f32>> {
    processor
        .prepare(spec())
        .map_err(|error| std::io::Error::other(error.to_string()))?;
    let latency = processor.latency();
    let mut rendered = input.to_vec();
    run(processor, &mut rendered, |_| Vec::new());
    let tail = drain(processor, tail_cap);
    if !tail.done {
        return Err(std::io::Error::other(
            "latency-compensated listening render exceeded its tail cap",
        ));
    }
    rendered.extend_from_slice(&tail.samples);
    let end = latency
        .checked_add(input.len())
        .ok_or_else(|| std::io::Error::other("latency compensation overflow"))?;
    if rendered.len() < end {
        return Err(std::io::Error::other(
            "processor produced too little output for latency compensation",
        ));
    }
    Ok(rendered[latency..end].to_vec())
}

/// Linear fade-in and fade-out over `n` frames, so the test signal edges stay
/// out of the case under test.
fn fade_edges(buf: &mut [f32], n: usize) {
    let n = n.min(buf.len() / 2);
    for i in 0..n {
        let g = i as f32 / n as f32;
        buf[i] *= g;
        let end = buf.len() - 1 - i;
        buf[end] *= g;
    }
}

// ---------------------------------------------------------------------------
// WAV output
// ---------------------------------------------------------------------------

/// Write one mono render as high-resolution PCM without adding another effect.
fn write_wav(dir: &Path, file: &str, samples: &[f32], note: &str) -> std::io::Result<Rendered> {
    write_wav_planar(dir, file, &[samples], 24, note)
}

/// Write one stereo render as high-resolution PCM.
fn write_wav_stereo(
    dir: &Path,
    file: &str,
    samples: &[Vec<f32>],
    note: &str,
) -> std::io::Result<Rendered> {
    let planes: Vec<&[f32]> = samples.iter().map(Vec::as_slice).collect();
    write_wav_planar(dir, file, &planes, 24, note)
}

/// Write one mono render as rounded 16-bit PCM without applying dither.
///
/// Only the controlled dither comparison uses this path.
fn write_wav_raw(dir: &Path, file: &str, samples: &[f32], note: &str) -> std::io::Result<Rendered> {
    write_wav_planar(dir, file, &[samples], 16, note)
}

/// Write interleaved integer PCM from planar samples.
fn write_wav_planar(
    dir: &Path,
    file: &str,
    samples: &[&[f32]],
    bits_per_sample: u16,
    note: &str,
) -> std::io::Result<Rendered> {
    if samples.is_empty() || samples.len() > usize::from(u16::MAX) {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "WAV output requires 1..=u16::MAX channels",
        ));
    }
    let frames = samples[0].len();
    if frames == 0 || samples.iter().any(|plane| plane.len() != frames) {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "WAV output requires non-empty rectangular planes",
        ));
    }
    if samples
        .iter()
        .any(|plane| plane.iter().any(|sample| !sample.is_finite()))
    {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "WAV output contains a non-finite sample",
        ));
    }
    if !matches!(bits_per_sample, 16 | 24) {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "listening WAV output supports 16-bit or 24-bit PCM",
        ));
    }

    let peak_dbfs = peak_of_planar(samples);
    if !peak_dbfs.is_finite() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("{file} is silent"),
        ));
    }
    if peak_dbfs > 0.0 {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("{file} clips at {peak_dbfs:.2} dBFS"),
        ));
    }

    let channels = u16::try_from(samples.len()).expect("channel count checked above");
    let bytes_per_sample = usize::from(bits_per_sample / 8);
    let block_align = usize::from(channels)
        .checked_mul(bytes_per_sample)
        .ok_or_else(|| std::io::Error::other("WAV block alignment overflow"))?;
    let data_len = frames
        .checked_mul(block_align)
        .ok_or_else(|| std::io::Error::other("WAV data length overflow"))?;
    let data_len_u32 = u32::try_from(data_len)
        .map_err(|_| std::io::Error::other("WAV data exceeds the RIFF size limit"))?;
    let riff_len = data_len_u32
        .checked_add(36)
        .ok_or_else(|| std::io::Error::other("WAV data exceeds the RIFF size limit"))?;
    let file_len = data_len
        .checked_add(44)
        .ok_or_else(|| std::io::Error::other("WAV file length overflow"))?;
    let block_align_u16 = u16::try_from(block_align)
        .map_err(|_| std::io::Error::other("WAV block alignment exceeds u16"))?;
    let byte_rate = RATE
        .checked_mul(u32::from(block_align_u16))
        .ok_or_else(|| std::io::Error::other("WAV byte rate overflow"))?;

    let mut bytes = Vec::with_capacity(file_len);
    bytes.extend_from_slice(b"RIFF");
    bytes.extend_from_slice(&riff_len.to_le_bytes());
    bytes.extend_from_slice(b"WAVEfmt ");
    bytes.extend_from_slice(&16u32.to_le_bytes());
    bytes.extend_from_slice(&1u16.to_le_bytes()); // PCM
    bytes.extend_from_slice(&channels.to_le_bytes());
    bytes.extend_from_slice(&RATE.to_le_bytes());
    bytes.extend_from_slice(&byte_rate.to_le_bytes());
    bytes.extend_from_slice(&block_align_u16.to_le_bytes());
    bytes.extend_from_slice(&bits_per_sample.to_le_bytes());
    bytes.extend_from_slice(b"data");
    bytes.extend_from_slice(&data_len_u32.to_le_bytes());
    for frame in 0..frames {
        for plane in samples {
            let x = f64::from(plane[frame]);
            if bits_per_sample == 16 {
                let v = (x * 32_768.0).round().clamp(-32_768.0, 32_767.0) as i16;
                bytes.extend_from_slice(&v.to_le_bytes());
            } else {
                let v = (x * 8_388_608.0).round().clamp(-8_388_608.0, 8_388_607.0) as i32;
                bytes.extend_from_slice(&v.to_le_bytes()[..3]);
            }
        }
    }
    let mut f = fs::File::create(dir.join(file))?;
    f.write_all(&bytes)?;
    Ok(Rendered {
        file: file.to_owned(),
        note: note.to_owned(),
        peak_dbfs,
        frames,
        channels: usize::from(channels),
        bits_per_sample,
    })
}

fn peak_of_planar(samples: &[&[f32]]) -> f64 {
    let mut peak = PeakMeter::new();
    Measurer::<f32>::prepare(&mut peak, spec_for_channels(samples.len())).expect("meter prepare");
    let frames = samples.first().map_or(0, |plane| plane.len());
    for pos in (0..frames).step_by(BLOCK) {
        let end = (pos + BLOCK).min(frames);
        let planes: Vec<&[f32]> = samples.iter().map(|plane| &plane[pos..end]).collect();
        peak.observe(AudioBlock::new(&planes));
    }
    linear_to_dbfs(Measurer::<f32>::read(&peak))
}

fn true_peak_of(samples: &[f32]) -> f64 {
    let mut peak = TruePeakMeter::new();
    Measurer::<f32>::prepare(&mut peak, spec()).expect("true-peak meter prepare");
    for chunk in samples.chunks(BLOCK) {
        let planes: [&[f32]; 1] = [chunk];
        peak.observe(AudioBlock::new(&planes));
    }
    let zeros = vec![0.0; Measurer::<f32>::latency(&peak)];
    let planes: [&[f32]; 1] = [&zeros];
    peak.observe(AudioBlock::new(&planes));
    linear_to_dbfs(Measurer::<f32>::read(&peak))
}

fn validate_cases(dir: &Path, cases: &[Case]) -> std::io::Result<()> {
    let mut titles = HashSet::new();
    let mut files = HashSet::new();
    for case in cases {
        if case.title.trim().is_empty()
            || case.what.trim().is_empty()
            || case.listen_for.trim().is_empty()
        {
            return Err(std::io::Error::other(
                "listening cases require complete notes",
            ));
        }
        if !titles.insert(case.title.as_str()) {
            return Err(std::io::Error::other(format!(
                "duplicate listening case title `{}`",
                case.title
            )));
        }
        if case.files.is_empty() {
            return Err(std::io::Error::other(format!(
                "listening case `{}` has no files",
                case.title
            )));
        }
        for rendered in &case.files {
            if !files.insert(rendered.file.as_str()) {
                return Err(std::io::Error::other(format!(
                    "duplicate listening filename `{}`",
                    rendered.file
                )));
            }
            if rendered.note.trim().is_empty()
                || rendered.frames == 0
                || rendered.channels == 0
                || !rendered.peak_dbfs.is_finite()
                || rendered.peak_dbfs > 0.0
            {
                return Err(std::io::Error::other(format!(
                    "invalid listening metadata for `{}`",
                    rendered.file
                )));
            }
            validate_wav(&dir.join(&rendered.file), rendered)?;
        }
    }

    for entry in processor_entries() {
        let file = format!("baseline_{}.wav", entry.id);
        if !files.contains(file.as_str()) {
            return Err(std::io::Error::other(format!(
                "missing processor baseline `{file}`"
            )));
        }
    }
    for entry in variable_rate_entries() {
        let file = format!("baseline_{}.wav", entry.id);
        if !files.contains(file.as_str()) {
            return Err(std::io::Error::other(format!(
                "missing variable-rate baseline `{file}`"
            )));
        }
    }
    Ok(())
}

fn validate_wav(path: &Path, rendered: &Rendered) -> std::io::Result<()> {
    let bytes = fs::read(path)?;
    let invalid = || {
        std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("invalid listening WAV `{}`", path.display()),
        )
    };
    if bytes.len() < 44
        || &bytes[0..4] != b"RIFF"
        || &bytes[8..16] != b"WAVEfmt "
        || &bytes[36..40] != b"data"
    {
        return Err(invalid());
    }
    let u16_at = |at| u16::from_le_bytes([bytes[at], bytes[at + 1]]);
    let u32_at = |at| u32::from_le_bytes(bytes[at..at + 4].try_into().expect("four bytes"));
    let channels = u16_at(22);
    let bits = u16_at(34);
    let block_align = u16_at(32);
    let data_len = u32_at(40) as usize;
    let expected_align = usize::from(channels) * usize::from(bits / 8);
    let expected_data_len = rendered.frames.checked_mul(expected_align);
    let expected_file_len = data_len.checked_add(44);
    if u32_at(4) as usize != bytes.len() - 8
        || u32_at(16) != 16
        || u16_at(20) != 1
        || usize::from(channels) != rendered.channels
        || u32_at(24) != RATE
        || usize::from(block_align) != expected_align
        || u32_at(28) != RATE * u32::from(block_align)
        || bits != rendered.bits_per_sample
        || expected_data_len != Some(data_len)
        || expected_file_len != Some(bytes.len())
    {
        return Err(invalid());
    }
    Ok(())
}

fn ev0(param: ParamId, value: f64) -> ParamEvent {
    ParamEvent {
        offset: 0,
        param,
        value,
    }
}

// ---------------------------------------------------------------------------
// Curated cases
// ---------------------------------------------------------------------------

/// The tremolo gain schedule shared by both smoothing renders.
fn tremolo_db(pos: u64) -> f64 {
    if (pos / (u64::from(RATE) / 8)) % 2 == 0 {
        -1.5
    } else {
        -20.0
    }
}

/// What the smoother bank buys: stepped gain vs event-smoothed gain.
fn case_smoothing(dir: &Path) -> std::io::Result<Case> {
    let tone = render_source(
        SineOsc::new(),
        seconds(3.0),
        &[(SineOsc::FREQUENCY_HZ, 330.0), (SineOsc::AMPLITUDE, 0.45)],
    );

    // A shows stepped gain built with public API only, by snapping the gain at
    // every block edge instead of sending events.
    let mut stepped = tone.clone();
    let mut gain_a = KernelProcessor::new(Gain::new());
    Processor::<f32>::prepare(&mut gain_a, spec()).expect("prepare");
    {
        let total = stepped.len();
        let mut pos = 0usize;
        while pos < total {
            let n = BLOCK.min(total - pos);
            gain_a
                .set_parameter_immediate(Gain::GAIN_DB, tremolo_db(pos as u64))
                .expect("Gain declares GAIN_DB");
            let (_, rest) = stepped.split_at_mut(pos);
            let (chunk, _) = rest.split_at_mut(n);
            let mut planes: [&mut [f32]; 1] = [chunk];
            let mut ctx = ProcessContext::in_place(&mut planes, pos as u64);
            gain_a.process(&mut ctx);
            pos += n;
        }
    }
    fade_edges(&mut stepped, seconds(0.01));

    // B sends the same schedule as sample-stamped events. The bank ramps them.
    let mut smoothed = tone;
    let mut gain_b = KernelProcessor::new(Gain::new());
    Processor::<f32>::prepare(&mut gain_b, spec()).expect("prepare");
    run(&mut gain_b, &mut smoothed, |pos| {
        vec![ev0(Gain::GAIN_DB, tremolo_db(pos))]
    });
    fade_edges(&mut smoothed, seconds(0.01));

    Ok(Case {
        title: "Parameter smoothing, stepped vs ramped".to_owned(),
        what: "The same 8 Hz gain tremolo on a 330 Hz sine, twice. The stepped \
               render snaps the gain at every block edge. It is a deliberately \
               degraded control built with `set_parameter_immediate` per block. The \
               ramped render sends the same \
               schedule as events, which the smoother bank ramps over 5 ms."
            .to_owned(),
        listen_for: "The stepped file clicks at every level change. The ramped \
                     file should make each transition gradual and much less \
                     abrupt."
            .to_owned(),
        files: vec![
            write_wav(
                dir,
                "smoothing_a_stepped.wav",
                &stepped,
                "gain stepped per block",
            )?,
            write_wav(
                dir,
                "smoothing_b_ramped.wav",
                &smoothed,
                "the same schedule through event smoothing",
            )?,
        ],
    })
}

/// A high-Q filter swept fast: smoothing under a hostile automation load.
fn case_filter_sweep(dir: &Path) -> std::io::Result<Case> {
    let mut saw = render_source(
        PolyBlepOsc::saw(),
        seconds(4.0),
        &[
            (PolyBlepOsc::FREQUENCY_HZ, 110.0),
            (PolyBlepOsc::AMPLITUDE, 0.25),
        ],
    );
    let dry = write_wav(dir, "sweep_a_saw.wav", &saw, "the unfiltered sawtooth")?;

    let mut lp = KernelProcessor::new(Biquad::lowpass());
    Processor::<f32>::prepare(&mut lp, spec()).expect("prepare");
    lp.set_parameter_immediate(Biquad::CUTOFF_HZ, 300.0)
        .expect("Biquad declares CUTOFF_HZ");
    lp.set_parameter_immediate(Biquad::Q, 8.0)
        .expect("Biquad declares Q");
    let total = saw.len() as f64;
    run(&mut lp, &mut saw, |pos| {
        // Up then down, log domain, twice through the range.
        let t = 2.0 * (pos as f64 / total);
        let t = if t > 1.0 { 2.0 - t } else { t };
        let hz = 300.0 * math::pow(8_000.0f64 / 300.0, t);
        vec![ev0(Biquad::CUTOFF_HZ, hz)]
    });
    fade_edges(&mut saw, seconds(0.01));

    Ok(Case {
        title: "High-Q filter sweep".to_owned(),
        what: "A Q = 8 low-pass swept 300 Hz to 8 kHz and back over a sawtooth, \
               driven by one automation event per block. Frequency parameters \
               ramp in the log domain."
            .to_owned(),
        listen_for: "A smooth resonant wah in both directions. Listen for \
                     granular stepping or zipper noise riding the sweep, \
                     especially near the resonance."
            .to_owned(),
        files: vec![
            dry,
            write_wav(dir, "sweep_b_lowpass_q8.wav", &saw, "the swept filter")?,
        ],
    })
}

/// Band-limited synthesis vs a deliberately aliased naive reference.
fn case_aliasing(dir: &Path) -> std::io::Result<Case> {
    let segment_frames = seconds(0.7);
    let mut naive = Vec::new();
    let mut blep = Vec::new();
    for frequency in [200.0, 500.0, 1_000.0, 2_000.0, 3_500.0, 5_000.0] {
        let mut naive_segment = vec![0.0f32; segment_frames];
        let mut phase = 0.0f64;
        for sample in &mut naive_segment {
            *sample = (0.3 * (2.0 * phase - 1.0)) as f32;
            phase += frequency / f64::from(RATE);
            if phase >= 1.0 {
                phase -= 1.0;
            }
        }
        fade_edges(&mut naive_segment, seconds(0.01));
        naive.extend_from_slice(&naive_segment);

        let mut blep_segment = render_source(
            PolyBlepOsc::saw(),
            segment_frames,
            &[
                (PolyBlepOsc::FREQUENCY_HZ, frequency),
                (PolyBlepOsc::AMPLITUDE, 0.3),
            ],
        );
        fade_edges(&mut blep_segment, seconds(0.01));
        blep.extend_from_slice(&blep_segment);
    }

    Ok(Case {
        title: "Aliasing, naive saw and PolyBLEP".to_owned(),
        what: "The same six sawtooth pitches from 200 Hz to 5 kHz, rendered \
               with identical levels, segment timing, phase resets, edge fades, \
               and 24-bit encoding. The naive file is a deliberately aliased \
               textbook phase accumulator. The PolyBLEP file uses bisque's \
               alias-reduced oscillator."
            .to_owned(),
        listen_for: "Inharmonic tones and roughness gathering around the higher \
                     naive pitches. The PolyBLEP pitches should remain cleaner, \
                     while retaining the same fundamental pitch and level."
            .to_owned(),
        files: vec![
            write_wav(
                dir,
                "aliasing_a_naive.wav",
                &naive,
                "deliberately degraded reference, naive aliased sawtooth",
            )?,
            write_wav(dir, "aliasing_b_polyblep.wav", &blep, "PolyBLEP sawtooth")?,
        ],
    })
}

/// Transients into heavy limiting, checking for a clean attack.
fn case_limiter_transients(dir: &Path) -> std::io::Result<Case> {
    let frames = seconds(4.0);
    let mut sig = render_source(
        SineOsc::new(),
        frames,
        &[(SineOsc::FREQUENCY_HZ, 220.0), (SineOsc::AMPLITUDE, 0.12)],
    );
    // Place short loud noise bursts on the quiet bed every 600 ms.
    let mut burst = render_source(
        WhiteNoise::new(),
        seconds(0.04),
        &[(WhiteNoise::AMPLITUDE, 0.85)],
    );
    fade_edges(&mut burst, seconds(0.004));
    let step = seconds(0.6);
    let mut at = seconds(0.3);
    while at + burst.len() < frames {
        for (i, &b) in burst.iter().enumerate() {
            sig[at + i] = (sig[at + i] + b).clamp(-1.0, 1.0);
        }
        at += step;
    }
    fade_edges(&mut sig, seconds(0.01));
    let reference = sig.clone();
    let dry = write_wav(
        dir,
        "limiter_a_dry.wav",
        &sig,
        "quiet bed with loud noise bursts",
    )?;

    let mut limited = sig;
    let mut lim = KernelProcessor::new(Limiter::with_settings(
        LimiterSettings::new().threshold_db(-12.0),
    ));
    Processor::<f32>::prepare(&mut lim, spec()).expect("prepare");
    run(&mut lim, &mut limited, |_| Vec::new());
    let tail = drain(&mut lim, seconds(0.1));
    if !tail.done {
        return Err(std::io::Error::other(
            "limiter listening render exceeded its tail cap",
        ));
    }
    limited.extend_from_slice(&tail.samples);
    let match_db = match_loudness(&reference, &mut limited);

    Ok(Case {
        title: "Limiter on transients".to_owned(),
        what: format!(
            "A quiet 220 Hz bed with hard 40 ms noise bursts, through the \
             limiter set to -12 dBFS with its 4x real-time peak detector and \
             default 0.1 dB safety margin. The processed file receives \
             {match_db:+.1} dB of loudness compensation while retaining 0.5 dB \
             of headroom according to the same real-time meter. Expected behavior \
             is the bed ducking during each burst and swelling back over the 50 ms \
             release. Detection is channel-linked, with one gain for everything."
        ),
        listen_for: "Bursts tamed to a constant level with a clean onset every \
                     time. Listen for ticks or snaps at the instant a burst \
                     begins, or crackle riding the release swell."
            .to_owned(),
        files: vec![
            dry,
            write_wav(
                dir,
                "limiter_b_ceiling.wav",
                &limited,
                &format!("limited at -12 dBFS, tail flushed, {match_db:+.1} dB compensation"),
            )?,
        ],
    })
}

/// Program-style compression with smooth pumping.
fn case_compressor(dir: &Path) -> std::io::Result<Case> {
    let frames = seconds(4.0);
    // An amplitude-pulsed saw approximates a bass line hitting a compressor.
    let mut sig = render_source(
        PolyBlepOsc::saw(),
        frames,
        &[
            (PolyBlepOsc::FREQUENCY_HZ, 82.4), // E2
            (PolyBlepOsc::AMPLITUDE, 0.4),
        ],
    );
    // Pulse the level with the library's own gain smoothing (2 Hz).
    let mut pulse = KernelProcessor::new(Gain::new());
    Processor::<f32>::prepare(&mut pulse, spec()).expect("prepare");
    run(&mut pulse, &mut sig, |pos| {
        let phase = (pos / (u64::from(RATE) / 2)) % 2;
        vec![ev0(Gain::GAIN_DB, if phase == 0 { 0.0 } else { -14.0 })]
    });
    fade_edges(&mut sig, seconds(0.01));
    let reference = sig.clone();
    let dry = write_wav(dir, "compressor_a_dry.wav", &sig, "pulsed bass line")?;

    let mut squashed = sig;
    let mut comp = KernelProcessor::new(Compressor::with_settings(
        CompressorSettings::new()
            .threshold_db(-24.0)
            .ratio(6.0)
            .attack_ms(5.0)
            .release_ms(120.0),
    ));
    Processor::<f32>::prepare(&mut comp, spec()).expect("prepare");
    run(&mut comp, &mut squashed, |_| Vec::new());
    let match_db = match_loudness(&reference, &mut squashed);

    Ok(Case {
        title: "Compressor character".to_owned(),
        what: format!(
            "A pulsed low-E bass line into the feed-forward compressor \
             (-24 dBFS threshold, 6:1, 5 ms attack, 120 ms release). The \
             processed file receives {match_db:+.1} dB of loudness compensation \
             while retaining 0.5 dB of headroom according to bisque's real-time \
             peak meter."
        ),
        listen_for: "Loud pulses pulled down toward the quiet ones, with an \
                     audible but smooth release swell after each transition. \
                     Listen for crackle on the attacks or a stepped, granular \
                     release."
            .to_owned(),
        files: vec![
            dry,
            write_wav(
                dir,
                "compressor_b_squashed.wav",
                &squashed,
                &format!("compressed, {match_db:+.1} dB loudness compensation"),
            )?,
        ],
    })
}

/// Stereo program material ducked by an independent sidechain key.
fn case_stereo_sidechain(dir: &Path) -> std::io::Result<Case> {
    let frames = seconds(4.0);
    let mut main = vec![
        render_source(
            PolyBlepOsc::saw(),
            frames,
            &[
                (PolyBlepOsc::FREQUENCY_HZ, 110.0),
                (PolyBlepOsc::AMPLITUDE, 0.2),
            ],
        ),
        render_source(
            SineOsc::new(),
            frames,
            &[(SineOsc::FREQUENCY_HZ, 330.0), (SineOsc::AMPLITUDE, 0.2)],
        ),
    ];
    for channel in &mut main {
        fade_edges(channel, seconds(0.01));
    }

    let mut key = vec![vec![0.0f32; frames]; 2];
    let mut burst = render_source(
        WhiteNoise::new(),
        seconds(0.08),
        &[(WhiteNoise::AMPLITUDE, 0.8)],
    );
    fade_edges(&mut burst, seconds(0.005));
    let mut at = seconds(0.4);
    let mut key_channel = 0;
    while at + burst.len() < frames {
        key[key_channel][at..at + burst.len()].copy_from_slice(&burst);
        key_channel ^= 1;
        at += seconds(0.6);
    }

    let dry = main.clone();
    let mut ducked = main;
    let mut compressor = KernelProcessor::new(Compressor::with_settings(
        CompressorSettings::new()
            .threshold_db(-30.0)
            .ratio(10.0)
            .attack_ms(2.0)
            .release_ms(160.0)
            .use_sidechain(true),
    ));
    Processor::<f32>::prepare(&mut compressor, spec_for_channels(2)).expect("prepare");
    run_stereo_sidechain(&mut compressor, &mut ducked, &key);

    Ok(Case {
        title: "Stereo sidechain ducking".to_owned(),
        what: "A steady stereo program has a sawtooth on the left and a sine on \
               the right. Bursts in a separate stereo noise key alternate \
               between the left and right channels every 600 ms. The key is \
               included as its own file so the expected ducking times are easy \
               to identify."
            .to_owned(),
        listen_for: "Both program channels dipping together for every key burst, \
                     including bursts on either side of the key bus. The noise \
                     key must not leak into the output. The stereo image should \
                     remain stable, and each recovery should be smooth and free \
                     of clicks."
            .to_owned(),
        files: vec![
            write_wav_stereo(dir, "sidechain_a_dry.wav", &dry, "steady stereo program")?,
            write_wav_stereo(dir, "sidechain_b_key.wav", &key, "sidechain key only")?,
            write_wav_stereo(
                dir,
                "sidechain_c_ducked.wav",
                &ducked,
                "program ducked by the independent key",
            )?,
        ],
    })
}

/// Downward expansion and gating on alternating strong and quiet passages.
fn case_expander_and_gate(dir: &Path) -> std::io::Result<Case> {
    let frames = seconds(4.0);
    let mut signal = render_source(
        SineOsc::new(),
        frames,
        &[(SineOsc::FREQUENCY_HZ, 220.0), (SineOsc::AMPLITUDE, 0.25)],
    );
    let mut pulse = KernelProcessor::new(Gain::new());
    Processor::<f32>::prepare(&mut pulse, spec()).expect("prepare");
    run(&mut pulse, &mut signal, |pos| {
        let quiet = (pos / (u64::from(RATE) / 2)) % 2 == 1;
        vec![ev0(Gain::GAIN_DB, if quiet { -30.0 } else { 0.0 })]
    });
    fade_edges(&mut signal, seconds(0.01));

    let expanded = render_kernel(
        Expander::with_settings(
            ExpanderSettings::new()
                .threshold_db(-30.0)
                .ratio(2.5)
                .attack_ms(5.0)
                .release_ms(100.0),
        ),
        &signal,
    );
    let gated = render_kernel(
        Gate::with_settings(
            GateSettings::new()
                .threshold_db(-30.0)
                .ratio(8.0)
                .range_db(-60.0)
                .attack_ms(2.0)
                .release_ms(120.0),
        ),
        &signal,
    );

    Ok(Case {
        title: "Expander and gate transitions".to_owned(),
        what: "A 220 Hz tone alternates every half second between a strong \
               passage and a passage 30 dB quieter. The expander pushes the \
               quiet passages farther down. The gate closes more decisively."
            .to_owned(),
        listen_for: "Clean openings, smooth decays, and stable tone during the \
                     strong passages. Listen for chatter, clicks, or a rough \
                     staircase as either processor closes."
            .to_owned(),
        files: vec![
            write_wav(
                dir,
                "dynamics_a_dry.wav",
                &signal,
                "alternating input levels",
            )?,
            write_wav(
                dir,
                "dynamics_b_expander.wav",
                &expanded,
                "downward expansion below the threshold",
            )?,
            write_wav(
                dir,
                "dynamics_c_gate.wav",
                &gated,
                "gated below the threshold",
            )?,
        ],
    })
}

/// Non-neutral shelf and peaking settings that make each shape audible.
fn case_eq_shapes(dir: &Path) -> std::io::Result<Case> {
    let frames = seconds(3.0);
    let mut signal = render_source(WhiteNoise::new(), frames, &[(WhiteNoise::AMPLITUDE, 0.07)]);
    let low = render_source(
        SineOsc::new(),
        frames,
        &[(SineOsc::FREQUENCY_HZ, 100.0), (SineOsc::AMPLITUDE, 0.08)],
    );
    let high = render_source(
        SineOsc::new(),
        frames,
        &[(SineOsc::FREQUENCY_HZ, 6_000.0), (SineOsc::AMPLITUDE, 0.08)],
    );
    for ((sample, low), high) in signal.iter_mut().zip(low).zip(high) {
        *sample += low + high;
    }
    fade_edges(&mut signal, seconds(0.01));

    let low_shelf = render_kernel(
        Biquad::with_settings(BiquadSettings::low_shelf().cutoff_hz(250.0).gain_db(9.0)),
        &signal,
    );
    let peaking = render_kernel(
        Biquad::with_settings(
            BiquadSettings::peaking()
                .cutoff_hz(1_200.0)
                .q(1.5)
                .gain_db(9.0),
        ),
        &signal,
    );
    let high_shelf = render_kernel(
        Biquad::with_settings(BiquadSettings::high_shelf().cutoff_hz(4_000.0).gain_db(9.0)),
        &signal,
    );

    Ok(Case {
        title: "Shelf and peaking EQ shapes".to_owned(),
        what: "The same low tone, high tone, and quiet broadband noise pass \
               through a 9 dB low shelf, peaking band, and high shelf. These \
               non-neutral settings exercise the shapes whose registry defaults \
               intentionally use 0 dB gain."
            .to_owned(),
        listen_for: "A clear bass lift, a focused midrange emphasis, and a clear \
                     treble lift in the corresponding files. None should add \
                     crackle, instability, or unrelated tonal artifacts."
            .to_owned(),
        files: vec![
            write_wav(dir, "eq_a_dry.wav", &signal, "unfiltered program")?,
            write_wav(dir, "eq_b_low_shelf.wav", &low_shelf, "9 dB low shelf")?,
            write_wav(dir, "eq_c_peaking.wav", &peaking, "9 dB peaking band")?,
            write_wav(dir, "eq_d_high_shelf.wav", &high_shelf, "9 dB high shelf")?,
        ],
    })
}

/// A feedback delay tail draining to true silence.
fn case_delay_tail(dir: &Path) -> std::io::Result<Case> {
    let mut ping = vec![0.0f32; seconds(2.0)];
    let burst = seconds(0.3);
    let rendered = render_source(
        SineOsc::new(),
        burst,
        &[(SineOsc::FREQUENCY_HZ, 440.0), (SineOsc::AMPLITUDE, 0.5)],
    );
    ping[..burst].copy_from_slice(&rendered);
    fade_edges(&mut ping[..burst], seconds(0.02));
    let dry = write_wav(dir, "delay_a_ping.wav", &ping, "a single dry ping")?;

    let mut echoed = ping;
    let mut delay = KernelProcessor::new(Delay::with_settings(
        DelaySettings::new()
            .delay_ms(330.0)
            .feedback(0.7)
            .mix(0.5)
            .max_delay_ms(400.0),
    ));
    Processor::<f32>::prepare(&mut delay, spec()).expect("prepare");
    run(&mut delay, &mut echoed, |_| Vec::new());
    // Drain the whole tail. `flush` reports done at the decay floor.
    let tail = drain(&mut delay, seconds(12.0));
    if !tail.done {
        return Err(std::io::Error::other(
            "delay listening render exceeded its tail cap",
        ));
    }
    echoed.extend_from_slice(&tail.samples);
    append_silence(&mut echoed, 0.5);

    Ok(Case {
        title: "Delay tail drain".to_owned(),
        what: "One ping into a 330 ms delay with 0.7 feedback. Everything \
               after the input ends comes from `flush`, which drains until \
               the ring decays below the -120 dBFS floor. A half-second of \
               digital silence follows the completed drain."
            .to_owned(),
        listen_for: "Echoes decaying evenly into true silence. Listen for fizz \
                     or crackle deep in the decay, or a tail that cuts off \
                     abruptly."
            .to_owned(),
        files: vec![
            dry,
            write_wav(
                dir,
                "delay_b_tail_drained.wav",
                &echoed,
                "echoes, full tail, and silent post-roll",
            )?,
        ],
    })
}

/// 16-bit truncation vs the library's TPDF dither on a long fade.
fn case_dither_fade(dir: &Path) -> std::io::Result<Case> {
    let frames = seconds(6.0);
    let mut tone = render_source(
        SineOsc::new(),
        frames,
        &[(SineOsc::FREQUENCY_HZ, 660.0), (SineOsc::AMPLITUDE, 0.5)],
    );
    // Fade to -84 dB through the library's own gain automation.
    let mut fader = KernelProcessor::new(Gain::new());
    Processor::<f32>::prepare(&mut fader, spec()).expect("prepare");
    let total = tone.len() as f64;
    run(&mut fader, &mut tone, |pos| {
        vec![ev0(Gain::GAIN_DB, -84.0 * (pos as f64 / total))]
    });
    fade_edges(&mut tone, seconds(0.01));
    let dithered = render_kernel(Dither::new(), &tone);

    Ok(Case {
        title: "Dither on a long fade (raise the volume near the end)".to_owned(),
        what: "A 660 Hz tone fading to -84 dBFS over six seconds, quantized to \
               16 bits twice. One file omits dither and is the deliberately \
               degraded reference quantized by the bench itself. The other \
               passes through the library's seeded TPDF dither. For blind \
               evaluation, have someone else pick the playback order."
            .to_owned(),
        listen_for: "Raise the volume for the final seconds, and lower it \
                     before the next file. The undithered fade turns granular \
                     and buzzy before dropping out; the dithered fade stays a \
                     clean tone sinking into smooth hiss."
            .to_owned(),
        files: vec![
            write_wav_raw(
                dir,
                "dither_a_undithered.wav",
                &tone,
                "deliberately degraded reference, undithered 16-bit quantization",
            )?,
            write_wav_raw(dir, "dither_b_tpdf.wav", &dithered, "TPDF-dithered 16-bit")?,
        ],
    })
}

/// Sustained and percussive material through two time-stretch ratios.
fn case_time_stretch(dir: &Path) -> std::io::Result<Case> {
    let frames = seconds(4.0);
    let mut program = render_source(
        SineOsc::new(),
        frames,
        &[(SineOsc::FREQUENCY_HZ, 220.0), (SineOsc::AMPLITUDE, 0.12)],
    );
    let saw = render_source(
        PolyBlepOsc::saw(),
        frames,
        &[
            (PolyBlepOsc::FREQUENCY_HZ, 330.0),
            (PolyBlepOsc::AMPLITUDE, 0.08),
        ],
    );
    for (sample, saw) in program.iter_mut().zip(saw) {
        *sample += saw;
    }
    let mut transient = render_source(
        WhiteNoise::new(),
        seconds(0.025),
        &[(WhiteNoise::AMPLITUDE, 0.35)],
    );
    fade_edges(&mut transient, seconds(0.003));
    let mut at = seconds(0.25);
    while at + transient.len() < frames {
        for (slot, transient) in program[at..at + transient.len()].iter_mut().zip(&transient) {
            *slot = (*slot + *transient).clamp(-1.0, 1.0);
        }
        at += seconds(0.5);
    }
    fade_edges(&mut program, seconds(0.01));

    let mut shorter_processor =
        TimeStretch::<f32>::with_settings(TimeStretchSettings::new().stretch(0.75));
    let mut longer_processor =
        TimeStretch::<f32>::with_settings(TimeStretchSettings::new().stretch(1.5));
    let mut reference = program.clone();
    let mut shorter =
        render_variable_rate(&mut shorter_processor, vec![program.clone()])?.remove(0);
    let mut longer = render_variable_rate(&mut longer_processor, vec![program])?.remove(0);
    append_silence(&mut reference, 0.25);
    append_silence(&mut shorter, 0.25);
    append_silence(&mut longer, 0.25);

    Ok(Case {
        title: "Time stretching on tones and transients".to_owned(),
        what: "A sustained two-pitch texture with short noise attacks is rendered \
               unchanged, shortened to 0.75 times its duration, and lengthened \
               to 1.5 times its duration. A short silent post-roll follows each \
               file. TimeStretch uses plain overlap-add, so pitch instability, \
               grain-rate coloration, and transient softening away from unity are \
               expected."
            .to_owned(),
        listen_for: "The requested duration and continuous playback, with no \
                     missing chunks, boundary clicks, or sudden level jumps. \
                     Compare the pitch instability and grain-rate coloration on \
                     the sustained tones, and the attack smearing at both ratios."
            .to_owned(),
        files: vec![
            write_wav(
                dir,
                "stretch_a_original.wav",
                &reference,
                "original duration",
            )?,
            write_wav(
                dir,
                "stretch_b_0.75x.wav",
                &shorter,
                "0.75 times the original duration",
            )?,
            write_wav(
                dir,
                "stretch_c_1.5x.wav",
                &longer,
                "1.5 times the original duration",
            )?,
        ],
    })
}

/// Spectral overlap-add passthrough and low-pass filtering.
fn case_spectral(dir: &Path) -> std::io::Result<Case> {
    let frames = seconds(4.0);
    let mut signal = render_source(
        PolyBlepOsc::saw(),
        frames,
        &[
            (PolyBlepOsc::FREQUENCY_HZ, 220.0),
            (PolyBlepOsc::AMPLITUDE, 0.12),
        ],
    );
    let high = render_source(
        SineOsc::new(),
        frames,
        &[(SineOsc::FREQUENCY_HZ, 8_000.0), (SineOsc::AMPLITUDE, 0.1)],
    );
    for (sample, high) in signal.iter_mut().zip(high) {
        *sample += high;
    }
    let mut transient = render_source(
        WhiteNoise::new(),
        seconds(0.02),
        &[(WhiteNoise::AMPLITUDE, 0.2)],
    );
    fade_edges(&mut transient, seconds(0.003));
    let mut at = seconds(0.35);
    while at + transient.len() < frames {
        for (slot, transient) in signal[at..at + transient.len()].iter_mut().zip(&transient) {
            *slot += *transient;
        }
        at += seconds(0.7);
    }
    fade_edges(&mut signal, seconds(0.01));

    let mut passthrough = SpectralFilter::band(1024, 512, 0.0, f64::INFINITY);
    let mut lowpass = SpectralFilter::low_pass(1024, 512, 3_000.0);
    let preroll = 1024;
    let mut padded = vec![0.0; preroll];
    padded.extend_from_slice(&signal);
    let mut dry = signal.clone();
    let mut passed =
        render_latency_compensated(&mut passthrough, &padded, seconds(0.1))?.split_off(preroll);
    let mut filtered =
        render_latency_compensated(&mut lowpass, &padded, seconds(0.1))?.split_off(preroll);
    append_silence(&mut dry, 0.25);
    append_silence(&mut passed, 0.25);
    append_silence(&mut filtered, 0.25);

    Ok(Case {
        title: "Spectral overlap-add".to_owned(),
        what: "A low sawtooth, an 8 kHz tone, and short noise attacks pass through \
               a 1024-frame spectral passthrough and a 3 kHz spectral low-pass. \
               A silent preroll lets overlap-add settle before the comparison. \
               The processed files are latency-compensated and followed by a \
               short silent post-roll."
            .to_owned(),
        listen_for: "The passthrough should retain the dry signal without \
                     flutter, phasing, level modulation, or hop-boundary ticks. \
                     The low-pass should remove the high tone cleanly while the \
                     low sawtooth remains stable."
            .to_owned(),
        files: vec![
            write_wav(dir, "spectral_a_dry.wav", &dry, "unprocessed reference")?,
            write_wav(
                dir,
                "spectral_b_passthrough.wav",
                &passed,
                "latency-compensated spectral passthrough",
            )?,
            write_wav(
                dir,
                "spectral_c_lowpass.wav",
                &filtered,
                "3 kHz spectral low-pass",
            )?,
        ],
    })
}

// ---------------------------------------------------------------------------
// Registry baselines
// ---------------------------------------------------------------------------

/// A short default-settings render per registry entry: an audible smoke test
/// that every enrolled processor produces sane output.
fn case_baselines(dir: &Path) -> std::io::Result<Case> {
    // A musical program signal for effects: a saw-plus-sine mix.
    let mut program = render_source(
        PolyBlepOsc::saw(),
        seconds(2.0),
        &[
            (PolyBlepOsc::FREQUENCY_HZ, 110.0),
            (PolyBlepOsc::AMPLITUDE, 0.22),
        ],
    );
    let sine = render_source(
        SineOsc::new(),
        seconds(2.0),
        &[(SineOsc::FREQUENCY_HZ, 440.0), (SineOsc::AMPLITUDE, 0.12)],
    );
    for (a, b) in program.iter_mut().zip(&sine) {
        *a += b;
    }
    fade_edges(&mut program, seconds(0.01));

    let mut files = Vec::new();
    for entry in processor_entries() {
        let mut p = (entry.make)();
        Processor::<f32>::prepare(&mut *p, spec()).expect("baseline prepare");
        let mut buf = match entry.drive {
            DriveMode::Source => vec![0.0f32; seconds(2.0)],
            _ => program.clone(),
        };
        run(&mut *p, &mut buf, |_| Vec::new());
        let tail_done = if p.tail() == Tail::None {
            None
        } else {
            let tail = drain(&mut *p, seconds(3.0));
            buf.extend_from_slice(&tail.samples);
            Some(tail.done)
        };
        fade_edges(&mut buf, seconds(0.01));
        let file = format!("baseline_{}.wav", entry.id);
        let mut note = match entry.drive {
            DriveMode::Source => "source, default settings".to_owned(),
            _ => "default settings over the program mix".to_owned(),
        };
        match tail_done {
            Some(true) => note.push_str(", tail drained"),
            Some(false) => note.push_str(", tail capped at three seconds"),
            None => {}
        }
        files.push(write_wav(dir, &file, &buf, &note)?);
    }

    for entry in variable_rate_entries() {
        let mut processor = (entry.make)();
        let mut rendered = render_variable_rate(&mut *processor, vec![program.clone()])?.remove(0);
        fade_edges(&mut rendered, seconds(0.01));
        let file = format!("baseline_{}.wav", entry.id);
        files.push(write_wav(
            dir,
            &file,
            &rendered,
            "registered variable-rate processor over the program mix",
        )?);
    }

    let what = String::from(
        "One short smoke render per registered audio processor and variable-rate \
         processor. Effects process a saw-plus-sine program mix, sources render \
         themselves, sidechain processors receive the program mix as their key, \
         and tails drain for up to three seconds. Meters are excluded because \
         they do not produce audio. Neutral defaults may sound unchanged; the \
         focused cases above exercise their audible behavior. New registry \
         entries appear here automatically.",
    );

    Ok(Case {
        title: "Registry baselines".to_owned(),
        what,
        listen_for: "Nonempty, finite, artifact-free output and sensible tail \
                     completion. These are smoke renders, not controlled A/B tests."
            .to_owned(),
        files,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn test_dir(name: &str) -> std::path::PathBuf {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system clock after Unix epoch")
            .as_nanos();
        let dir = std::env::temp_dir().join(format!(
            "bisque-listen-{name}-{}-{unique}",
            std::process::id()
        ));
        fs::create_dir_all(&dir).expect("create temporary listening directory");
        dir
    }

    #[test]
    fn wav_writer_emits_valid_24_bit_stereo_pcm() {
        let dir = test_dir("stereo");
        let left = [0.25, -0.5, 0.125];
        let right = [-0.25, 0.5, -0.125];

        let rendered =
            write_wav_stereo(&dir, "stereo.wav", &[left.to_vec(), right.to_vec()], "test")
                .expect("write stereo WAV");

        assert_eq!(rendered.frames, 3);
        assert_eq!(rendered.channels, 2);
        assert_eq!(rendered.bits_per_sample, 24);
        validate_wav(&dir.join("stereo.wav"), &rendered).expect("validate stereo WAV");
        let bytes = fs::read(dir.join("stereo.wav")).expect("read stereo WAV");
        assert_eq!(u16::from_le_bytes([bytes[22], bytes[23]]), 2);
        assert_eq!(u16::from_le_bytes([bytes[34], bytes[35]]), 24);
        assert_eq!(
            &bytes[44..56],
            &[0x00, 0x00, 0x20, 0x00, 0x00, 0xE0, 0x00, 0x00, 0xC0, 0x00, 0x00, 0x40]
        );

        fs::remove_dir_all(dir).expect("remove temporary listening directory");
    }

    #[test]
    fn wav_writer_rejects_silence_non_finite_samples_and_clipping() {
        let dir = test_dir("invalid");
        let invalid = [
            ("silence.wav", vec![0.0, 0.0]),
            ("nan.wav", vec![0.25, f32::NAN]),
            ("clipping.wav", vec![0.25, 1.01]),
        ];

        for (file, samples) in invalid {
            assert!(
                write_wav(&dir, file, &samples, "test").is_err(),
                "{file} should be rejected"
            );
            assert!(!dir.join(file).exists());
        }

        fs::remove_dir_all(dir).expect("remove temporary listening directory");
    }

    #[test]
    fn case_validation_rejects_duplicate_filenames() {
        let dir = test_dir("duplicates");
        let rendered = write_wav(&dir, "duplicate.wav", &[0.25, -0.25], "test").expect("write WAV");
        let cases = [Case {
            title: "Duplicate test".to_owned(),
            what: "Exercises filename validation.".to_owned(),
            listen_for: "No listening is required.".to_owned(),
            files: vec![rendered.clone(), rendered],
        }];

        let error = validate_cases(&dir, &cases).expect_err("duplicate filename should fail");
        assert!(error.to_string().contains("duplicate listening filename"));

        fs::remove_dir_all(dir).expect("remove temporary listening directory");
    }
}
