// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Airclay LLC

//! Shared snapshot-case registry.
//!
//! `xtask gen-snapshots` and the `snapshots::` tests use this registry. Generic
//! hashing, byte-layout, and signal helpers live in `bisque::testing`.

#![allow(dead_code)]

use super::{loud_stereo, sweep_stereo, tone_stereo, Buffers, Contract};
use crate::filters::Biquad;
use crate::mastering::{Dither, DitherSettings, Gain, Limiter, Scale};
use crate::parameter::ParamEvent;
use crate::processor::KernelProcessor;
use crate::processor::VariableRate;
use crate::processor::{AudioBlockMut, Processor, Tail};
use crate::repair::{DcBlocker, DcOffset};
use crate::time::{TimeStretch, TimeStretchSettings};

/// Block size used when driving snapshot cases.
pub const SNAPSHOT_BLOCK: usize = 128;

/// Seed used by the dither snapshot.
const DITHER_SEED: u64 = 0xDEAD_BEEF_CAFE_F00D;

/// Which observable surface a snapshot hashes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Region {
    /// The process body only.
    Body,
    /// The process body followed by the drained flush tail.
    BodyFlush,
}

impl Region {
    #[must_use]
    pub fn tag(self) -> &'static str {
        match self {
            Region::Body => "body",
            Region::BodyFlush => "body+flush",
        }
    }
}

/// A processor snapshot case with fixed signal, events, and observable region.
#[derive(Clone, Copy, Debug)]
pub struct Case {
    pub id: &'static str,
    pub signal: &'static str,
    pub region: Region,
    pub events: &'static [ParamEvent],
    pub frames: usize,
    pub make: fn() -> Box<dyn Processor<f32>>,
    pub signal_fn: fn(usize) -> Buffers,
}

/// Return all committed processor snapshot cases.
///
/// Each case is expected to be byte-exact across supported platforms.
#[must_use]
#[allow(clippy::too_many_lines)] // A flat literal registry: one Case per processor.
pub fn snapshot_cases() -> Vec<Case> {
    vec![
        Case {
            id: "biquad-lp",
            signal: "sweep-stereo",
            region: Region::Body,
            events: &[
                ParamEvent {
                    offset: 0,
                    param: Biquad::CUTOFF_HZ,
                    value: 700.0,
                },
                ParamEvent {
                    offset: 400,
                    param: Biquad::CUTOFF_HZ,
                    value: 3000.0,
                },
            ],
            frames: 1000,
            make: || Box::new(KernelProcessor::new(Biquad::lowpass())) as Box<dyn Processor<f32>>,
            signal_fn: sweep_stereo,
        },
        Case {
            id: "biquad-hp",
            signal: "sweep-stereo",
            region: Region::Body,
            events: &[],
            frames: 1000,
            make: || Box::new(KernelProcessor::new(Biquad::highpass())) as Box<dyn Processor<f32>>,
            signal_fn: sweep_stereo,
        },
        Case {
            id: "dc-blocker",
            signal: "tone-stereo",
            region: Region::Body,
            events: &[],
            frames: 1500,
            make: || Box::new(KernelProcessor::new(DcBlocker::new())) as Box<dyn Processor<f32>>,
            signal_fn: tone_stereo,
        },
        Case {
            id: "gain-neg6db",
            signal: "tone-stereo",
            region: Region::Body,
            events: &[ParamEvent {
                offset: 0,
                param: Gain::GAIN_DB,
                value: -6.0,
            }],
            frames: 1500,
            make: || Box::new(KernelProcessor::new(Gain::new())) as Box<dyn Processor<f32>>,
            signal_fn: tone_stereo,
        },
        Case {
            id: "dither16",
            signal: "tone-stereo",
            region: Region::Body,
            events: &[],
            frames: 1500,
            make: || {
                // 16-bit at the pinned seed: the committed hash depends on both.
                Box::new(KernelProcessor::new(Dither::with_settings(
                    DitherSettings::new().bits(16).seed(DITHER_SEED),
                ))) as Box<dyn Processor<f32>>
            },
            signal_fn: tone_stereo,
        },
        Case {
            id: "limiter-default",
            signal: "loud-stereo",
            region: Region::BodyFlush,
            events: &[
                ParamEvent {
                    offset: 0,
                    param: Limiter::THRESHOLD_DB,
                    value: -2.0,
                },
                ParamEvent {
                    offset: 500,
                    param: Limiter::THRESHOLD_DB,
                    value: -1.0,
                },
            ],
            frames: 1000,
            make: || Box::new(KernelProcessor::new(Limiter::new())) as Box<dyn Processor<f32>>,
            signal_fn: loud_stereo,
        },
        Case {
            id: "scale-neg6db",
            signal: "tone-stereo",
            region: Region::Body,
            events: &[],
            frames: 1500,
            make: || {
                Box::new(KernelProcessor::new(Scale::from_db(-6.0))) as Box<dyn Processor<f32>>
            },
            signal_fn: tone_stereo,
        },
        Case {
            id: "dc-offset",
            signal: "tone-stereo",
            region: Region::Body,
            events: &[],
            frames: 1500,
            make: || {
                Box::new(KernelProcessor::new(DcOffset::per_channel(vec![0.1, -0.1])))
                    as Box<dyn Processor<f32>>
            },
            signal_fn: tone_stereo,
        },
    ]
}

/// Look up a snapshot case by id.
#[must_use]
pub fn snapshot_case(id: &str) -> Case {
    snapshot_cases()
        .into_iter()
        .find(|c| c.id == id)
        .unwrap_or_else(|| panic!("unknown snapshot case `{id}`"))
}

/// A `VariableRate` snapshot case with fixed settings and signal.
///
/// This is separate from [`Case`] because a `VariableRate` pulls from a `Source`
/// and produces an output frame count that may differ from the input.
#[derive(Clone, Copy, Debug)]
pub struct VrCase {
    pub id: &'static str,
    pub signal: &'static str,
    pub frames: usize,
    pub make: fn() -> Box<dyn VariableRate<f32>>,
    pub signal_fn: fn(usize) -> Buffers,
}

/// Return all committed `VariableRate` snapshot cases.
#[must_use]
pub fn vr_snapshot_cases() -> Vec<VrCase> {
    vec![VrCase {
        id: "timestretch-2x",
        signal: "tone-stereo",
        frames: 1500,
        make: || {
            Box::new(TimeStretch::<f32>::with_settings(
                TimeStretchSettings::new().stretch(2.0),
            )) as Box<dyn VariableRate<f32>>
        },
        signal_fn: tone_stereo,
    }]
}

/// Drive a `VariableRate` snapshot case through the manifest and test path.
#[must_use]
pub fn drive_vr_case(c: &VrCase) -> Buffers {
    let cont = Contract::default();
    let signal = (c.signal_fn)(c.frames);
    let mut v = (c.make)();
    v.prepare(cont.spec).expect("prepare");
    cont.stretch_reusing(&mut *v, &signal, SNAPSHOT_BLOCK, usize::MAX)
}

/// Drive a processor snapshot case through the manifest and test path.
///
/// `Body` cases run only the signal body. `BodyFlush` cases append the drained
/// flush tail to the body output.
pub fn drive_case(c: &Case) -> Buffers {
    let cont = Contract::default();
    let signal = (c.signal_fn)(c.frames);
    let mut proc = (c.make)();
    proc.prepare(cont.spec).expect("prepare");
    let mut out = cont.run_reusing(&mut *proc, &signal, c.events, SNAPSHOT_BLOCK);
    if matches!(c.region, Region::BodyFlush) {
        let mut remaining = match proc.tail() {
            Tail::None => 0,
            Tail::Frames(frames) => frames,
            Tail::Infinite => panic!("BodyFlush snapshot cases require a finite tail"),
        };
        let mut done = remaining == 0;
        while remaining > 0 {
            let mut stage: Buffers = vec![vec![0.0f32; remaining]; out.len()];
            let produced = {
                let mut planes: Vec<&mut [f32]> = stage.iter_mut().map(Vec::as_mut_slice).collect();
                let mut block = AudioBlockMut::new(&mut planes);
                proc.flush(&mut block)
            };
            assert!(
                produced.frames <= remaining,
                "flush exceeded its output capacity"
            );
            for (channel, staged) in out.iter_mut().zip(&stage) {
                channel.extend_from_slice(&staged[..produced.frames]);
            }
            remaining -= produced.frames;
            done = produced.done;
            if done {
                break;
            }
            assert!(produced.frames > 0, "flush made no progress before done");
        }
        assert!(done, "flush exceeded its declared finite tail");
    }
    out
}
