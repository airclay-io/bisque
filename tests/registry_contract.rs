// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Airclay LLC

//! Registry-driven contract tests.
//!
//! `bisque::testing::registry` enumerates every built-in processor, meter, and
//! rate changer. This suite drives each entry through the shared lifecycle
//! contracts (prepare, metadata consistency, block-size invariance, reset
//! equivalence, and footprint stability).

#![cfg(feature = "test-support")]

use bisque::parameter::ParamEvent;
use bisque::processor::{AudioBlock, AudioBlockMut, DspError, ProcessSpec, Processor, Tail};
use bisque::testing::registry::{
    meter_entries, processor_entries, variable_rate_entries, BoxedVariableRate, DriveMode,
    MeterEntry, ProcessorEntry,
};
use bisque::testing::{bits_eq, sine, Buffers, Contract};

/// Frame count used by the invariance and reset checks. Kept within
/// `Contract::default().spec.max_block` so the whole-block reference is legal,
/// and long enough to cross several control-rate boundaries and (for the
/// spectral entry) several STFT hops.
const FRAMES: usize = 3000;

/// A deterministic event schedule derived from an entry's declared parameters.
///
/// For each declared parameter the schedule sets a quarter-range value, a
/// three-quarter-range value, and the default, at staggered offsets. This
/// exercises latching and smoothing for every parameter without per-entry
/// hand-written data.
fn schedule_for(proc: &dyn Processor<f32>, frames: usize) -> Vec<ParamEvent> {
    let mut events = Vec::new();
    for (k, info) in proc.param_info().iter().enumerate() {
        let span = info.range.1 - info.range.0;
        let stagger = (k as u32) * 17;
        let at = |base: u32| (base + stagger) % (frames as u32);
        events.push(ParamEvent {
            offset: at(0),
            param: info.id,
            value: info.range.0 + 0.25 * span,
        });
        events.push(ParamEvent {
            offset: at(500),
            param: info.id,
            value: info.range.0 + 0.75 * span,
        });
        events.push(ParamEvent {
            offset: at(1100),
            param: info.id,
            value: info.default,
        });
    }
    events.sort_by_key(|e| e.offset);
    events
}

/// The schedule for a fresh instance of `entry`.
fn entry_schedule(entry: &ProcessorEntry, frames: usize) -> Vec<ParamEvent> {
    schedule_for(&*(entry.make)(), frames)
}

/// A deterministic sidechain key signal, distinct from the main signal.
fn key_signal(channels: usize, frames: usize) -> Buffers {
    sine(channels, frames)
        .into_iter()
        .map(|ch| ch.into_iter().rev().map(|s| s * 0.8).collect())
        .collect()
}

/// Lifecycle and invariance contracts for every registry entry.
mod contract {
    use super::*;

    #[test]
    fn every_entry_prepares_and_declares_consistent_metadata() {
        let c = Contract::default();
        for entry in processor_entries() {
            let mut proc = (entry.make)();
            proc.prepare(c.spec).unwrap_or_else(|e| {
                panic!("{}: prepare failed on the standard spec: {e:?}", entry.id)
            });
            assert_eq!(
                proc.io_mode(),
                entry.drive.io_mode(),
                "{}: declared drive mode disagrees with io_mode()",
                entry.id
            );
            assert_eq!(
                proc.sidechain_inputs(),
                entry.sidechain_inputs,
                "{}: declared sidechain bus count disagrees with sidechain_inputs()",
                entry.id
            );
            for (index, info) in proc.param_info().iter().enumerate() {
                assert_eq!(
                    info.id.0 as usize, index,
                    "{}: param ids must be sequential from 0",
                    entry.id
                );
            }
        }
    }

    #[test]
    fn every_entry_tail_covers_its_latency() {
        let c = Contract::default();
        for entry in processor_entries() {
            let mut proc = (entry.make)();
            proc.prepare(c.spec).expect("prepare");
            let latency = proc.latency();
            match proc.tail() {
                Tail::None => assert_eq!(
                    latency, 0,
                    "{}: nonzero latency requires a flushable tail",
                    entry.id
                ),
                Tail::Frames(frames) => assert!(
                    frames >= latency,
                    "{}: finite tail {frames} must cover latency {latency}",
                    entry.id
                ),
                Tail::Infinite => {}
            }
        }
    }

    #[test]
    fn every_entry_footprint_is_stable_across_prepares() {
        // Exact byte counts stay per-processor; the shared contract is that a
        // second prepare on the same spec reports the same footprint, and that
        // declared parameters imply reported (smoother-bank) state.
        let c = Contract::default();
        for entry in processor_entries() {
            let mut proc = (entry.make)();
            proc.prepare(c.spec).expect("prepare");
            let first = proc.memory_footprint();
            proc.prepare(c.spec).expect("second prepare");
            let second = proc.memory_footprint();
            assert_eq!(
                first, second,
                "{}: memory_footprint must be stable across prepares",
                entry.id
            );
            if !proc.param_info().is_empty() {
                assert!(
                    first > 0,
                    "{}: declared parameters imply prepare-allocated state, so \
                     memory_footprint() must be > 0",
                    entry.id
                );
            }
        }
    }

    #[test]
    fn every_entry_is_block_size_invariant() {
        let c = Contract::default();
        for entry in processor_entries() {
            let events = entry_schedule(&entry, FRAMES);
            let make = || (entry.make)();
            if entry.sidechain_inputs > 0 {
                let input = sine(c.spec.channels, FRAMES);
                let key = key_signal(c.spec.channels, FRAMES);
                let reference = c.run_with_sidechain(make, &input, &key, &events, FRAMES);
                for &block in &c.block_sizes {
                    let out = c.run_with_sidechain(make, &input, &key, &events, block);
                    assert!(
                        bits_eq(&out, &reference),
                        "{}: sidechain block size {block} diverged from the whole-block reference",
                        entry.id
                    );
                }
            } else {
                match entry.drive {
                    DriveMode::Effect => {
                        let input = sine(c.spec.channels, FRAMES);
                        c.assert_block_size_invariant(make, &input, &events);
                    }
                    DriveMode::Source => {
                        c.assert_generator_block_size_invariant(make, FRAMES, &events);
                    }
                    DriveMode::Split => {
                        let input = sine(c.spec.channels, FRAMES);
                        let reference = c.run_split(make, &input, FRAMES);
                        for &block in &c.block_sizes {
                            let out = c.run_split(make, &input, block);
                            assert!(
                                bits_eq(&out, &reference),
                                "{}: split block size {block} diverged from the whole-block reference",
                                entry.id
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn every_entry_reset_is_equivalent_to_fresh() {
        let c = Contract::default();
        for entry in processor_entries() {
            let events = entry_schedule(&entry, FRAMES);
            let make = || (entry.make)();
            if entry.sidechain_inputs > 0 {
                let input = sine(c.spec.channels, FRAMES);
                let key = key_signal(c.spec.channels, FRAMES);
                let fresh = c.run_with_sidechain(make, &input, &key, &events, 64);
                let mut proc = (entry.make)();
                proc.prepare(c.spec).expect("prepare");
                let _ = c.run_with_sidechain_reusing(&mut *proc, &input, &key, &events, 50);
                proc.reset();
                let after = c.run_with_sidechain_reusing(&mut *proc, &input, &key, &events, 64);
                assert!(
                    bits_eq(&after, &fresh),
                    "{}: reset must reproduce a fresh instance bit for bit",
                    entry.id
                );
            } else {
                match entry.drive {
                    DriveMode::Effect => {
                        let input = sine(c.spec.channels, FRAMES);
                        c.assert_reset_equivalence(make, &input, &events);
                    }
                    DriveMode::Source => {
                        let fresh = c.generate(make, FRAMES, &events, 64);
                        let mut proc = (entry.make)();
                        proc.prepare(c.spec).expect("prepare");
                        let _ = c.generate_reusing(&mut *proc, FRAMES, &events, 50);
                        proc.reset();
                        let after = c.generate_reusing(&mut *proc, FRAMES, &events, 64);
                        assert!(
                            bits_eq(&after, &fresh),
                            "{}: reset must reproduce a fresh source bit for bit",
                            entry.id
                        );
                    }
                    DriveMode::Split => {
                        let input = sine(c.spec.channels, FRAMES);
                        let fresh = c.run_split(make, &input, 64);
                        let mut proc = (entry.make)();
                        proc.prepare(c.spec).expect("prepare");
                        let _ = c.run_split_reusing(&mut *proc, &input, 50);
                        proc.reset();
                        let after = c.run_split_reusing(&mut *proc, &input, 64);
                        assert!(
                            bits_eq(&after, &fresh),
                            "{}: reset must reproduce a fresh instance bit for bit",
                            entry.id
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn every_small_tail_entry_restarts_its_drain_on_new_input() {
        // A drain delivers at most the processor's remaining declared tail,
        // and new input (or reset) starts a fresh drain. Entries with a small
        // declared tail are drained to exhaustion, which observably pins the
        // drained-frame counter and its reset on `process`/`reset`. This
        // catches retained counters that would skip the next stream's tail.
        // Large-bound recursive tails share the same counter logic and end
        // their drains through the decay early-exit; their domain tests pin
        // that path.
        const MAX_BOUND: usize = 8192;
        let c = Contract::default();
        let drain = |proc: &mut Box<dyn Processor<f32> + Send>| -> usize {
            let mut total = 0usize;
            loop {
                let mut stage: Buffers = vec![vec![0.0f32; 256]; c.spec.channels];
                let produced = {
                    let mut planes: Vec<&mut [f32]> =
                        stage.iter_mut().map(Vec::as_mut_slice).collect();
                    let mut out = AudioBlockMut::new(&mut planes);
                    proc.flush(&mut out)
                };
                total += produced.frames;
                if produced.frames == 0 || produced.done {
                    break;
                }
            }
            total
        };
        let input = sine(c.spec.channels, FRAMES);
        for entry in processor_entries() {
            if entry.sidechain_inputs > 0 {
                continue; // no sidechain entry declares a tail
            }
            let mut proc = (entry.make)();
            proc.prepare(c.spec).expect("prepare");
            let Tail::Frames(bound) = proc.tail() else {
                continue;
            };
            if bound == 0 || bound > MAX_BOUND {
                continue; // exhaustion must be cheap to observe
            }
            let drive = |proc: &mut Box<dyn Processor<f32> + Send>| match entry.drive {
                DriveMode::Effect => {
                    let _ = c.run_reusing(proc, &input, &[], 64);
                }
                DriveMode::Source => {
                    let _ = c.generate_reusing(proc, FRAMES, &[], 64);
                }
                DriveMode::Split => {
                    let _ = c.run_split_reusing(proc, &input, 64);
                }
            };
            drive(&mut proc);
            assert_eq!(
                drain(&mut proc),
                bound,
                "{}: the first drain delivers the declared tail",
                entry.id
            );
            assert_eq!(
                drain(&mut proc),
                0,
                "{}: an exhausted drain writes nothing",
                entry.id
            );
            drive(&mut proc);
            assert_eq!(
                drain(&mut proc),
                bound,
                "{}: new input must start a fresh drain",
                entry.id
            );
            // Reset also starts a fresh drain (of silence). This catches
            // drained-frame counters that would consume the post-reset tail.
            proc.reset();
            assert_eq!(
                drain(&mut proc),
                bound,
                "{}: reset must start a fresh drain",
                entry.id
            );
        }
    }

    #[test]
    fn every_observed_done_state_is_terminal() {
        let c = Contract::default();
        let input = vec![vec![0.0f32; 1]; c.spec.channels];
        for entry in processor_entries() {
            if entry.sidechain_inputs > 0 {
                continue;
            }
            let mut proc = (entry.make)();
            proc.prepare(c.spec).expect("prepare");
            if matches!(proc.tail(), Tail::None) {
                continue;
            }
            match entry.drive {
                DriveMode::Effect => {
                    let _ = c.run_reusing(&mut proc, &input, &[], 1);
                }
                DriveMode::Source => {
                    let _ = c.generate_reusing(&mut proc, 1, &[], 1);
                }
                DriveMode::Split => {
                    let _ = c.run_split_reusing(&mut proc, &input, 1);
                }
            }

            let mut stage = vec![vec![0.0f32; 256]; c.spec.channels];
            let first = {
                let mut planes: Vec<&mut [f32]> = stage.iter_mut().map(Vec::as_mut_slice).collect();
                proc.flush(&mut AudioBlockMut::new(&mut planes))
            };
            if !first.done {
                continue;
            }

            let second = {
                let mut planes: Vec<&mut [f32]> = stage.iter_mut().map(Vec::as_mut_slice).collect();
                proc.flush(&mut AudioBlockMut::new(&mut planes))
            };
            assert_eq!(
                second.frames, 0,
                "{}: a flush after done must write no frames",
                entry.id
            );
            assert!(
                second.done,
                "{}: a flush after done must remain terminal",
                entry.id
            );
        }
    }

    /// The signal every meter check observes. Long enough for the loudness
    /// meter's 400 ms momentary window to fire at 48 kHz.
    fn meter_signal() -> Buffers {
        sine(2, 24_000)
    }

    /// Prepare a fresh instance of `entry`, observe `signal` in `block`-frame
    /// chunks, and return the reading's canonical rendering.
    fn observe_reading(entry: &MeterEntry, signal: &Buffers, block: usize) -> String {
        let c = Contract::default();
        let mut meter = (entry.make)();
        meter.prepare(c.spec).expect("prepare");
        let frames = signal.first().map_or(0, Vec::len);
        let mut pos = 0;
        while pos < frames {
            let blk = block.min(frames - pos);
            let planes: Vec<&[f32]> = signal.iter().map(|ch| &ch[pos..pos + blk]).collect();
            meter.observe(bisque::processor::AudioBlock::new(&planes));
            pos += blk;
        }
        meter.reading_debug()
    }

    #[test]
    fn every_meter_prepares_and_reports_stable_footprint() {
        let c = Contract::default();
        for entry in meter_entries() {
            let mut meter = (entry.make)();
            meter.prepare(c.spec).unwrap_or_else(|e| {
                panic!("{}: prepare failed on the standard spec: {e:?}", entry.id)
            });
            let first = meter.memory_footprint();
            meter.prepare(c.spec).expect("second prepare");
            assert_eq!(
                first,
                meter.memory_footprint(),
                "{}: memory_footprint must be stable across prepares",
                entry.id
            );
        }
    }

    #[cfg(debug_assertions)]
    #[test]
    fn every_meter_reports_focused_geometry_failures() {
        use std::panic::{catch_unwind, AssertUnwindSafe};

        let c = Contract::default();
        for entry in meter_entries() {
            let mut unprepared = (entry.make)();
            let signal = vec![vec![0.0f32; 1]; c.spec.channels];
            let result = catch_unwind(AssertUnwindSafe(|| {
                let planes: Vec<&[f32]> = signal.iter().map(Vec::as_slice).collect();
                unprepared.observe(AudioBlock::new(&planes));
            }));
            assert!(
                result.is_err(),
                "{} must reject observe before prepare",
                entry.id
            );

            let mut wrong_channels = (entry.make)();
            wrong_channels.prepare(c.spec).expect("prepare");
            let signal = vec![vec![0.0f32; 1]; c.spec.channels + 1];
            let result = catch_unwind(AssertUnwindSafe(|| {
                let planes: Vec<&[f32]> = signal.iter().map(Vec::as_slice).collect();
                wrong_channels.observe(AudioBlock::new(&planes));
            }));
            assert!(
                result.is_err(),
                "{} must reject a channel count mismatch",
                entry.id
            );

            let mut oversized = (entry.make)();
            oversized.prepare(c.spec).expect("prepare");
            let signal = vec![vec![0.0f32; c.spec.max_block + 1]; c.spec.channels];
            let result = catch_unwind(AssertUnwindSafe(|| {
                let planes: Vec<&[f32]> = signal.iter().map(Vec::as_slice).collect();
                oversized.observe(AudioBlock::new(&planes));
            }));
            assert!(
                result.is_err(),
                "{} must reject a block larger than max_block",
                entry.id
            );
        }
    }

    #[test]
    fn every_meter_reading_is_observe_invariant_across_block_sizes() {
        let c = Contract::default();
        let signal = meter_signal();
        for entry in meter_entries() {
            // Blocks are capped at `max_block`, so the reference observes the
            // signal in the largest legal chunks.
            let whole = observe_reading(&entry, &signal, c.spec.max_block);
            for &block in &c.block_sizes {
                let split = observe_reading(&entry, &signal, block);
                assert_eq!(
                    split, whole,
                    "{}: reading at block size {block} diverged from the whole-signal reading",
                    entry.id
                );
            }
        }
    }

    #[test]
    fn every_meter_reset_is_equivalent_to_fresh() {
        let c = Contract::default();
        let signal = meter_signal();
        let dirty = vec![vec![1.5f32; signal[0].len()]; signal.len()];
        for entry in meter_entries() {
            let fresh = observe_reading(&entry, &signal, 64);
            let mut meter = (entry.make)();
            meter.prepare(c.spec).expect("prepare");
            let frames = dirty[0].len();
            // Advance internal state (in legal max_block chunks) before reset.
            let mut pos = 0;
            while pos < frames {
                let blk = c.spec.max_block.min(frames - pos);
                let planes: Vec<&[f32]> = dirty.iter().map(|ch| &ch[pos..pos + blk]).collect();
                meter.observe(bisque::processor::AudioBlock::new(&planes));
                pos += blk;
            }
            meter.reset();
            let mut pos = 0;
            while pos < frames {
                let blk = 64.min(frames - pos);
                let planes: Vec<&[f32]> = signal.iter().map(|ch| &ch[pos..pos + blk]).collect();
                meter.observe(bisque::processor::AudioBlock::new(&planes));
                pos += blk;
            }
            assert_eq!(
                meter.reading_debug(),
                fresh,
                "{}: reset must reproduce a fresh instance's reading",
                entry.id
            );
        }
    }

    #[test]
    fn every_meter_reprepare_is_equivalent_to_fresh() {
        // `prepare` establishes the post-prepare state regardless of history.
        // A meter reused across streams (prepare, observe, prepare again) reads
        // exactly like a fresh instance, with no prior accumulation in the new
        // stream.
        let c = Contract::default();
        let signal = meter_signal();
        for entry in meter_entries() {
            let fresh = observe_reading(&entry, &signal, 64);
            let mut meter = (entry.make)();
            meter.prepare(c.spec).expect("prepare");
            // Dirty the accumulated state with one legal block, LOUDER than
            // the measured signal: a max-tracking meter whose re-prepare
            // misses the re-baseline then visibly carries the prior peak.
            let loud: Vec<Vec<f32>> = signal
                .iter()
                .map(|ch| ch[..c.spec.max_block].iter().map(|s| s * 2.0).collect())
                .collect();
            let planes: Vec<&[f32]> = loud.iter().map(Vec::as_slice).collect();
            meter.observe(bisque::processor::AudioBlock::new(&planes));
            meter.prepare(c.spec).expect("second prepare");
            let frames = signal[0].len();
            let mut pos = 0;
            while pos < frames {
                let blk = 64.min(frames - pos);
                let planes: Vec<&[f32]> = signal.iter().map(|ch| &ch[pos..pos + blk]).collect();
                meter.observe(bisque::processor::AudioBlock::new(&planes));
                pos += blk;
            }
            assert_eq!(
                meter.reading_debug(),
                fresh,
                "{}: a re-prepared meter must read like a fresh instance",
                entry.id
            );
        }
    }

    #[test]
    fn every_entry_memory_budget_is_fit_or_fail() {
        // `ProcessSpec::max_memory` is enforced for every family. The reported
        // footprint fits exactly; one byte less is `OverBudget`. Failed prepares
        // leave processors unprepared.
        let c = Contract::default();
        let with_cap = |cap| ProcessSpec {
            max_memory: Some(cap),
            ..c.spec
        };
        for entry in processor_entries() {
            let mut probe = (entry.make)();
            probe.prepare(c.spec).expect("prepare");
            let footprint = probe.memory_footprint();
            let mut fits = (entry.make)();
            fits.prepare(with_cap(footprint)).unwrap_or_else(|e| {
                panic!("{}: must fit a cap of its own footprint: {e:?}", entry.id)
            });
            if footprint > 0 {
                let mut over = (entry.make)();
                assert!(
                    matches!(
                        over.prepare(with_cap(footprint - 1)),
                        Err(DspError::OverBudget { .. })
                    ),
                    "{}: one byte under the footprint must be OverBudget",
                    entry.id
                );
            }
        }
        for entry in meter_entries() {
            let mut probe = (entry.make)();
            probe.prepare(c.spec).expect("prepare");
            let footprint = probe.memory_footprint();
            let mut fits = (entry.make)();
            fits.prepare(with_cap(footprint)).unwrap_or_else(|e| {
                panic!("{}: must fit a cap of its own footprint: {e:?}", entry.id)
            });
            if footprint > 0 {
                let mut over = (entry.make)();
                assert!(
                    matches!(
                        over.prepare(with_cap(footprint - 1)),
                        Err(DspError::OverBudget { .. })
                    ),
                    "{}: one byte under the footprint must be OverBudget",
                    entry.id
                );
            }
        }
        for entry in variable_rate_entries() {
            let mut probe = (entry.make)();
            probe.prepare(c.spec).expect("prepare");
            let footprint = probe.memory_footprint();
            let mut fits = (entry.make)();
            fits.prepare(with_cap(footprint)).unwrap_or_else(|e| {
                panic!("{}: must fit a cap of its own footprint: {e:?}", entry.id)
            });
            if footprint > 0 {
                let mut over = (entry.make)();
                assert!(
                    matches!(
                        over.prepare(with_cap(footprint - 1)),
                        Err(DspError::OverBudget { .. })
                    ),
                    "{}: one byte under the footprint must be OverBudget",
                    entry.id
                );
            }
        }
    }

    #[test]
    fn every_variable_rate_entry_prepares_and_is_stretch_invariant() {
        let c = Contract::default();
        let input = sine(c.spec.channels, 1500);
        for entry in variable_rate_entries() {
            let mut v = (entry.make)();
            v.prepare(c.spec).unwrap_or_else(|e| {
                panic!("{}: prepare failed on the standard spec: {e:?}", entry.id)
            });
            let first = v.memory_footprint();
            v.prepare(c.spec).expect("second prepare");
            assert_eq!(
                first,
                v.memory_footprint(),
                "{}: memory_footprint must be stable across prepares",
                entry.id
            );
            c.assert_stretch_block_size_invariant(|| BoxedVariableRate((entry.make)()), &input);
        }
    }
}
