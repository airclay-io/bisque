// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Airclay LLC

//! A realistic offline mastering session, start to finish.
//!
//! 1. Compose a short stereo clip with the library's own sources: a bass line
//!    whose notes are sample-stamped frequency events, hats from filtered
//!    noise, and a pad widened by two different delay times.
//! 2. Master it the way an engineer would: DC block, rumble high-pass, glue
//!    compression, then measure integrated loudness and apply the computed
//!    gain toward a -16 LUFS target (measure-then-apply), then true-peak
//!    limit and dither.
//! 3. Drain the chain's latency with `flush`, report the meters, and write
//!    `target/examples_out/` WAVs you can listen to (premaster and master).
//!
//! Contract touchpoints are marked in comments: one `ProcessSpec`, exact
//! configuration via settings, sample-stamped `ParamEvent` offsets,
//! chunked processing with `sample_pos` bookkeeping, `latency()`/`flush`.

use std::io::Write as _;
use std::path::Path;

use bisque::analysis::{linear_to_dbfs, LoudnessMeter, TruePeakMeter};
use bisque::dynamics::{Compressor, CompressorSettings};
use bisque::filters::{Biquad, BiquadSettings};
use bisque::generators::{
    PolyBlepOsc, PolyBlepOscSettings, SineOsc, SineOscSettings, WhiteNoise, WhiteNoiseSettings,
};
use bisque::mastering::{Dither, Gain, GainSettings, Limiter, LimiterSettings};
use bisque::parameter::ParamEvent;
use bisque::processor::{
    AudioBlock, AudioBlockMut, DspError, IoMode, Kernel, KernelProcessor, Measurer, ProcessContext,
    ProcessSpec, Processor,
};
use bisque::time::{Delay, DelaySettings};

const RATE: u32 = 48_000;
const CHUNK: usize = 512;
/// Sixteen eighth notes at 120 BPM: six seconds.
const EIGHTH: usize = 18_000;
const TOTAL: usize = EIGHTH * 16;

fn spec(channels: usize) -> ProcessSpec {
    // Each processor uses one fixed spec. Processor-owned audio state is
    // allocated during `prepare`. This offline host may allocate its own work.
    ProcessSpec {
        sample_rate: RATE,
        channels,
        max_block: CHUNK,
        max_memory: None,
    }
}

/// Drive a prepared in-place or output-only processor, chunk by chunk.
/// `events_for(chunk_start, chunk_len)` returns block-relative events.
fn run(
    p: &mut dyn Processor<f32>,
    planes: &mut [&mut [f32]],
    mut events_for: impl FnMut(usize, usize) -> Vec<ParamEvent>,
) {
    let total = planes[0].len();
    let mut pos = 0usize;
    while pos < total {
        let n = CHUNK.min(total - pos);
        let events = events_for(pos, n);
        // The offline host rebuilds the plane table for each chunk.
        // `sample_pos` keeps the absolute timeline continuous.
        let mut chunk: Vec<&mut [f32]> =
            planes.iter_mut().map(|pl| &mut pl[pos..pos + n]).collect();
        let mut ctx = match p.io_mode() {
            IoMode::InPlace => ProcessContext::in_place(&mut chunk, pos as u64),
            IoMode::OutputOnly => ProcessContext::output_only(&mut chunk, pos as u64),
            IoMode::Split => panic!("run does not support split I/O"),
        }
        .with_events(&events);
        p.process(&mut ctx);
        pos += n;
    }
}

/// Render a mono source kernel configured before prepare.
fn render_mono<K: Kernel<f32>>(kernel: K, frames: usize) -> Vec<f32> {
    let mut p = KernelProcessor::new(kernel);
    p.prepare(spec(1)).expect("prepare");
    let mut buf = vec![0.0f32; frames];
    run(&mut p, &mut [&mut buf], |_, _| Vec::new());
    buf
}

/// Feed a stereo pair to a meter in chunks.
fn observe<M: Measurer<f32>>(m: &mut M, left: &[f32], right: &[f32]) {
    for (l, r) in left.chunks(CHUNK).zip(right.chunks(CHUNK)) {
        m.observe(AudioBlock::new(&[l, r]));
    }
}

/// Integrated LUFS of a stereo pair.
fn integrated_lufs(left: &[f32], right: &[f32]) -> f64 {
    let mut meter = LoudnessMeter::new();
    Measurer::<f32>::prepare(&mut meter, spec(2)).expect("prepare");
    observe(&mut meter, left, right);
    Measurer::<f32>::read(&meter).integrated_lufs
}

// The example is one linear mastering walkthrough; splitting it would hurt it.
#[allow(clippy::too_many_lines)]
fn main() -> Result<(), DspError> {
    // -----------------------------------------------------------------------
    // Compose: every sound below comes from the library's own sources.
    // -----------------------------------------------------------------------

    // Bass: one PolyBLEP saw. The note pattern uses sample-stamped frequency
    // events landing on exact eighth-note boundaries, including nonzero offsets
    // within a chunk. Log-domain smoothing gives a musical glide.
    let notes = [
        55.0, 55.0, 82.4, 55.0, 73.4, 55.0, 110.0, 82.4, //
        55.0, 55.0, 82.4, 55.0, 73.4, 73.4, 61.7, 55.0,
    ];
    let mut bass = vec![0.0f32; TOTAL];
    {
        let mut osc = KernelProcessor::new(PolyBlepOsc::with_settings(
            PolyBlepOscSettings::new()
                .frequency_hz(notes[0])
                .amplitude(0.30),
        ));
        osc.prepare(spec(1))?;
        run(&mut osc, &mut [&mut bass], |pos, len| {
            // Contract touchpoint: events carry exact in-block offsets.
            let mut events = Vec::new();
            for (i, &hz) in notes.iter().enumerate() {
                let at = i * EIGHTH;
                if at >= pos && at < pos + len {
                    events.push(ParamEvent {
                        offset: (at - pos) as u32,
                        param: PolyBlepOsc::FREQUENCY_HZ,
                        value: hz,
                    });
                }
            }
            events
        });
        // Tame the saw's top end so it sits like a bass.
        let mut lp = KernelProcessor::new(Biquad::with_settings(
            BiquadSettings::lowpass().cutoff_hz(500.0),
        ));
        lp.prepare(spec(1))?;
        run(&mut lp, &mut [&mut bass], |_, _| Vec::new());
    }

    // Hats: filtered noise, gated on the offbeats (a compositional envelope
    // applied by the host; the noise and filter are the library's).
    let mut hats = render_mono(
        WhiteNoise::with_settings(WhiteNoiseSettings::new().amplitude(0.20)),
        TOTAL,
    );
    {
        let mut hp = KernelProcessor::new(Biquad::with_settings(
            BiquadSettings::highpass().cutoff_hz(7_000.0),
        ));
        hp.prepare(spec(1))?;
        run(&mut hp, &mut [&mut hats], |_, _| Vec::new());
        let (open, ramp) = (2_400usize, 144usize); // 50 ms hits, 3 ms edges
        for (i, s) in hats.iter_mut().enumerate() {
            let in_eighth = i % EIGHTH;
            let offbeat = (i / EIGHTH) % 2 == 1;
            let g = if !offbeat || in_eighth >= open {
                0.0
            } else if in_eighth < ramp {
                in_eighth as f32 / ramp as f32
            } else if in_eighth >= open - ramp {
                (open - in_eighth) as f32 / ramp as f32
            } else {
                1.0
            };
            *s *= g;
        }
    }

    // Pad: two sines a major third apart, widened by giving each side its own
    // delay time. Different tails per side is what makes it stereo.
    let mut pad_l = render_mono(
        SineOsc::with_settings(SineOscSettings::new().frequency_hz(220.0)),
        TOTAL,
    );
    let mut pad_r = render_mono(
        SineOsc::with_settings(SineOscSettings::new().frequency_hz(277.2)),
        TOTAL,
    );
    for (side, ms) in [(&mut pad_l, 331.0), (&mut pad_r, 473.0)] {
        for s in side.iter_mut() {
            *s *= 0.14;
        }
        let mut delay = KernelProcessor::new(Delay::with_settings(
            DelaySettings::new()
                .delay_ms(ms)
                .feedback(0.35)
                .mix(0.35)
                .max_delay_ms(500.0),
        ));
        delay.prepare(spec(1))?;
        run(&mut delay, &mut [side], |_, _| Vec::new());
    }

    // Mix down to the stereo premaster.
    let mut left = vec![0.0f32; TOTAL];
    let mut right = vec![0.0f32; TOTAL];
    for i in 0..TOTAL {
        let center = bass[i] + hats[i] * 0.6;
        left[i] = center + pad_l[i];
        right[i] = center + pad_r[i];
    }
    for buf in [&mut left, &mut right] {
        let fade = 480;
        for i in 0..fade {
            let g = i as f32 / fade as f32;
            buf[i] *= g;
            let e = TOTAL - 1 - i;
            buf[e] *= g;
        }
    }
    write_wav_stereo("premaster.wav", &left, &right);

    // -----------------------------------------------------------------------
    // Master: clean up, glue, measure-then-apply loudness, limit, dither.
    // -----------------------------------------------------------------------
    let mut chain_latency = 0usize;

    let mut stage = |p: &mut dyn Processor<f32>, l: &mut Vec<f32>, r: &mut Vec<f32>| {
        run(p, &mut [l.as_mut_slice(), r.as_mut_slice()], |_, _| {
            Vec::new()
        });
        chain_latency += p.latency();
    };

    let mut dc = KernelProcessor::new(bisque::repair::DcBlocker::new());
    dc.prepare(spec(2))?;
    stage(&mut dc, &mut left, &mut right);

    let mut hp = KernelProcessor::new(Biquad::with_settings(
        BiquadSettings::highpass().cutoff_hz(30.0),
    ));
    hp.prepare(spec(2))?;
    stage(&mut hp, &mut left, &mut right);

    let mut glue = KernelProcessor::new(Compressor::with_settings(
        CompressorSettings::new()
            .threshold_db(-20.0)
            .ratio(2.0)
            .attack_ms(15.0)
            .release_ms(150.0),
    ));
    glue.prepare(spec(2))?;
    stage(&mut glue, &mut left, &mut right);

    // Measure-then-apply: read integrated loudness, compute the makeup toward
    // the -16 LUFS target, and construct the stage with that fixed value.
    let pre_gain_lufs = integrated_lufs(&left, &right);
    let makeup_db = (-16.0 - pre_gain_lufs).clamp(-24.0, 24.0);
    let mut makeup =
        KernelProcessor::new(Gain::with_settings(GainSettings::new().gain_db(makeup_db)));
    makeup.prepare(spec(2))?;
    stage(&mut makeup, &mut left, &mut right);

    assert_eq!(
        chain_latency, 0,
        "stages before the final limiter must have zero latency"
    );
    let mut ceiling = KernelProcessor::new(Limiter::with_settings(
        LimiterSettings::new().threshold_db(-1.0),
    ));
    ceiling.prepare(spec(2))?;
    run(
        &mut ceiling,
        &mut [left.as_mut_slice(), right.as_mut_slice()],
        |_, _| Vec::new(),
    );
    chain_latency += ceiling.latency();

    // Contract touchpoint: the limiter holds `latency()` frames; drain them
    // so the master is sample-complete.
    let mut tail_l = vec![0.0f32; chain_latency.max(1)];
    let mut tail_r = vec![0.0f32; tail_l.len()];
    let mut drained = 0usize;
    while drained < tail_l.len() {
        let produced = {
            let mut planes: [&mut [f32]; 2] = [&mut tail_l[drained..], &mut tail_r[drained..]];
            let mut block = AudioBlockMut::new(&mut planes);
            ceiling.flush(&mut block)
        };
        drained += produced.frames;
        if produced.done || produced.frames == 0 {
            break;
        }
    }
    left.extend_from_slice(&tail_l[..drained]);
    right.extend_from_slice(&tail_r[..drained]);
    assert!(
        drained >= chain_latency,
        "the drain must complete the delayed input body"
    );
    compensate_latency(&mut left, chain_latency);
    compensate_latency(&mut right, chain_latency);

    let mut quantize = KernelProcessor::new(Dither::new());
    quantize.prepare(spec(2))?;
    run(
        &mut quantize,
        &mut [left.as_mut_slice(), right.as_mut_slice()],
        |_, _| Vec::new(),
    );

    write_wav_stereo("master.wav", &left, &right);

    // Report through the library's own meters.
    let after_lufs = integrated_lufs(&left, &right);
    let mut tp = TruePeakMeter::new();
    Measurer::<f32>::prepare(&mut tp, spec(2))?;
    observe(&mut tp, &left, &right);
    println!("pre-gain loudness:  {pre_gain_lufs:.1} LUFS");
    println!("makeup applied:     {makeup_db:+.1} dB (target -16 LUFS)");
    println!("master loudness:    {after_lufs:.1} LUFS");
    println!(
        "master true peak:   {:.2} dBTP (ceiling -1.0)",
        linear_to_dbfs(Measurer::<f32>::read(&tp))
    );
    println!("chain latency:      {chain_latency} frames, {drained} frames compensated");
    println!("wrote target/examples_out/premaster.wav and master.wav");
    Ok(())
}

/// Remove leading processing latency after the delayed body has been drained.
fn compensate_latency(samples: &mut Vec<f32>, latency: usize) {
    assert!(latency <= samples.len(), "latency exceeds rendered audio");
    let output_frames = samples.len() - latency;
    samples.copy_within(latency.., 0);
    samples.truncate(output_frames);
}

/// Interleave and write a 16-bit stereo WAV under `target/examples_out/`.
fn write_wav_stereo(name: &str, left: &[f32], right: &[f32]) {
    let dir = Path::new("target/examples_out");
    std::fs::create_dir_all(dir).expect("create output dir");
    let n = left.len() as u32;
    let mut bytes = Vec::with_capacity(44 + left.len() * 4);
    bytes.extend_from_slice(b"RIFF");
    bytes.extend_from_slice(&(36 + n * 4).to_le_bytes());
    bytes.extend_from_slice(b"WAVEfmt ");
    bytes.extend_from_slice(&16u32.to_le_bytes());
    bytes.extend_from_slice(&1u16.to_le_bytes()); // PCM
    bytes.extend_from_slice(&2u16.to_le_bytes()); // stereo
    bytes.extend_from_slice(&RATE.to_le_bytes());
    bytes.extend_from_slice(&(RATE * 4).to_le_bytes());
    bytes.extend_from_slice(&4u16.to_le_bytes());
    bytes.extend_from_slice(&16u16.to_le_bytes());
    bytes.extend_from_slice(b"data");
    bytes.extend_from_slice(&(n * 4).to_le_bytes());
    for (&l, &r) in left.iter().zip(right) {
        for x in [l, r] {
            let v = (f64::from(x) * 32768.0).round().clamp(-32768.0, 32767.0) as i16;
            bytes.extend_from_slice(&v.to_le_bytes());
        }
    }
    let mut f = std::fs::File::create(dir.join(name)).expect("create wav");
    f.write_all(&bytes).expect("write wav");
}
