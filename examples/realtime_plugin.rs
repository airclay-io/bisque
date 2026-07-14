// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Airclay LLC

//! A realistic realtime host: sidechain ducking, the way a broadcast or
//! podcast chain does it. A music bus is compressed with the voice bus as the
//! key, so the music dips whenever the voice speaks and swells back between
//! phrases.
//!
//! Realtime contract touchpoints demonstrated:
//!
//! - the chain is erased to `Box<dyn Processor<f32> + Send>`, the host seam;
//! - the callback loop uses a DIFFERENT block size on almost every call
//!   (hosts do this; output is bit-identical under any split by contract);
//! - plane tables are rebuilt on the stack per callback, and nothing in the
//!   callback loop allocates after `prepare`;
//! - the sidechain key is mono and broadcasts across the stereo main bus;
//! - automation arrives as sample-stamped `ParamEvent`s on the absolute
//!   timeline (a fade-in on the music during the first second).
//!
//! The result lands in `target/examples_out/` so you can hear the duck.

use std::io::Write as _;
use std::path::Path;

use bisque::analysis::{linear_to_dbfs, PeakMeter};
use bisque::dynamics::{Compressor, CompressorSettings};
use bisque::filters::{Biquad, BiquadSettings};
use bisque::generators::{
    PolyBlepOsc, PolyBlepOscSettings, SineOsc, SineOscSettings, WhiteNoise, WhiteNoiseSettings,
};
use bisque::mastering::{Gain, GainSettings};
use bisque::parameter::ParamEvent;
use bisque::processor::{
    AudioBlock, DspError, IoMode, Kernel, KernelProcessor, Measurer, ProcessContext, ProcessSpec,
    Processor,
};

const RATE: u32 = 48_000;
const MAX_BLOCK: usize = 512;
const TOTAL: usize = 48_000 * 6;

fn spec(channels: usize) -> ProcessSpec {
    ProcessSpec {
        sample_rate: RATE,
        channels,
        max_block: MAX_BLOCK,
        max_memory: None,
    }
}

/// Render a mono source configured before prepare.
fn render_mono<K: Kernel<f32>>(kernel: K, frames: usize) -> Vec<f32> {
    let mut p = KernelProcessor::new(kernel);
    p.prepare(spec(1)).expect("prepare");
    let mut buf = vec![0.0f32; frames];
    run_mono(&mut p, &mut buf);
    buf
}

/// Drive a prepared mono in-place or output-only processor over a whole buffer.
fn run_mono(p: &mut dyn Processor<f32>, buf: &mut [f32]) {
    let total = buf.len();
    let mut pos = 0usize;
    while pos < total {
        let n = MAX_BLOCK.min(total - pos);
        let mut planes: [&mut [f32]; 1] = [&mut buf[pos..pos + n]];
        let mut ctx = match p.io_mode() {
            IoMode::InPlace => ProcessContext::in_place(&mut planes, pos as u64),
            IoMode::OutputOnly => ProcessContext::output_only(&mut planes, pos as u64),
            IoMode::Split => panic!("run_mono does not support split I/O"),
        };
        p.process(&mut ctx);
        pos += n;
    }
}

// The example is one linear callback-host walkthrough; splitting would hurt it.
#[allow(clippy::too_many_lines)]
fn main() -> Result<(), DspError> {
    // -----------------------------------------------------------------------
    // Program material (setup phase).
    // -----------------------------------------------------------------------

    // Music bus: a bass saw plus a pad tone, slightly wide.
    let mut music_l = render_mono(
        PolyBlepOsc::with_settings(
            PolyBlepOscSettings::new()
                .frequency_hz(110.0)
                .amplitude(0.22),
        ),
        TOTAL,
    );
    let pad = render_mono(
        SineOsc::with_settings(SineOscSettings::new().frequency_hz(330.0).amplitude(0.10)),
        TOTAL,
    );
    let mut music_r = music_l.clone();
    for i in 0..TOTAL {
        music_l[i] += pad[i];
        music_r[i] += pad[i] * 0.6;
    }

    // Voice bus (the sidechain key): band-passed noise gated into phrases of
    // 0.9 s speech and 0.9 s pause with 5 ms edges.
    let mut voice = render_mono(
        WhiteNoise::with_settings(WhiteNoiseSettings::new().amplitude(0.5)),
        TOTAL,
    );
    for kernel in [
        Biquad::with_settings(BiquadSettings::highpass().cutoff_hz(300.0)),
        Biquad::with_settings(BiquadSettings::lowpass().cutoff_hz(3_000.0)),
    ] {
        let mut f = KernelProcessor::new(kernel);
        f.prepare(spec(1))?;
        run_mono(&mut f, &mut voice);
    }
    let (phrase, ramp) = (43_200usize, 240usize);
    for (i, s) in voice.iter_mut().enumerate() {
        let t = i % (phrase * 2);
        let g = if t >= phrase {
            0.0
        } else if t < ramp {
            t as f32 / ramp as f32
        } else if t >= phrase - ramp {
            (phrase - t) as f32 / ramp as f32
        } else {
            1.0
        };
        *s *= g;
    }

    // -----------------------------------------------------------------------
    // The plugin chain, erased to the host seam.
    // -----------------------------------------------------------------------
    let ducker = Compressor::with_settings(
        CompressorSettings::new()
            .threshold_db(-32.0)
            .ratio(8.0)
            .attack_ms(8.0)
            .release_ms(300.0)
            .use_sidechain(true),
    );
    let mut chain: Vec<Box<dyn Processor<f32> + Send>> = vec![
        Box::new(KernelProcessor::new(Gain::with_settings(
            GainSettings::new().gain_db(-40.0),
        ))),
        Box::new(KernelProcessor::new(ducker)),
    ];
    let mut latency = 0usize;
    for p in &mut chain {
        p.prepare(spec(2))?;
        latency += p.latency();
    }
    println!("chain latency reported to the host: {latency} frames");

    // Fade-in automation: -40 dB stepping to 0 dB across the first second,
    // written on the absolute timeline.
    let fade_curve: [(usize, f64); 9] =
        core::array::from_fn(|k| (k * 6_000, -40.0 + 5.0 * k as f64));

    // -----------------------------------------------------------------------
    // The callback loop. A real host owns this thread; nothing below
    // allocates: plane tables and the event scratch live on the stack.
    // -----------------------------------------------------------------------
    let block_sizes = [512usize, 128, 64, 333, 256, 512, 97, 480];
    let mut event_scratch: [ParamEvent; 9] = [ParamEvent {
        offset: 0,
        param: Gain::GAIN_DB,
        value: 0.0,
    }; 9];

    let mut pos = 0usize;
    let mut calls = 0usize;
    while pos < TOTAL {
        // Contract touchpoint. Hosts may change their block size at any time.
        // The rendered bytes stay independent of the split.
        let n = block_sizes[calls % block_sizes.len()].min(TOTAL - pos);
        calls += 1;

        // Collect the automation points that land inside this block, stamped
        // at their exact in-block offsets.
        let mut n_events = 0usize;
        for &(at, db) in &fade_curve {
            if at >= pos && at < pos + n {
                event_scratch[n_events] = ParamEvent {
                    offset: (at - pos) as u32,
                    param: Gain::GAIN_DB,
                    value: db,
                };
                n_events += 1;
            }
        }

        for (stage, p) in chain.iter_mut().enumerate() {
            // Contract touchpoint: stack plane tables, rebuilt per callback.
            let mut planes: [&mut [f32]; 2] =
                [&mut music_l[pos..pos + n], &mut music_r[pos..pos + n]];
            let key: [&[f32]; 1] = [&voice[pos..pos + n]];
            let sidechain = [AudioBlock::new(&key)];
            // Contract touchpoint: the block's absolute stream position is
            // part of context construction, so a streaming host cannot forget
            // its cursor.
            let mut ctx = ProcessContext::in_place(&mut planes, pos as u64)
                // Contract touchpoint: the mono voice key broadcasts across
                // the stereo main bus for detection.
                .with_sidechains(if p.sidechain_inputs() > 0 {
                    &sidechain
                } else {
                    &[]
                })
                // Only the fade stage consumes the gain events.
                .with_events(if stage == 0 {
                    &event_scratch[..n_events]
                } else {
                    &[]
                });
            p.process(&mut ctx);
        }
        pos += n;
    }
    println!("processed {TOTAL} frames in {calls} callbacks of varying sizes");

    // -----------------------------------------------------------------------
    // Hear it and measure it: the broadcast mix, and the duck depth.
    // -----------------------------------------------------------------------
    let mut out_l = vec![0.0f32; TOTAL];
    let mut out_r = vec![0.0f32; TOTAL];
    for i in 0..TOTAL {
        out_l[i] = music_l[i] + voice[i] * 0.9;
        out_r[i] = music_r[i] + voice[i] * 0.9;
    }
    write_wav_stereo("ducked_broadcast.wav", &out_l, &out_r);

    let seg_peak = |buf: &[f32], from: usize, to: usize| {
        let mut m = PeakMeter::new();
        Measurer::<f32>::prepare(&mut m, spec(1)).expect("prepare");
        for chunk in buf[from..to].chunks(MAX_BLOCK) {
            m.observe(AudioBlock::new(&[chunk]));
        }
        linear_to_dbfs(Measurer::<f32>::read(&m))
    };
    // The second phrase speaks over [1.8 s, 2.7 s); a pause follows it.
    let speaking = seg_peak(&music_l, 48_000 * 2, 48_000 * 5 / 2);
    let pause = seg_peak(&music_l, 48_000 * 3, 48_000 * 7 / 2);
    println!("music during speech:   {speaking:.1} dBFS");
    println!("music between phrases: {pause:.1} dBFS");
    println!("duck depth:            {:.1} dB", pause - speaking);
    println!("wrote target/examples_out/ducked_broadcast.wav");
    Ok(())
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
