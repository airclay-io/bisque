// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Airclay LLC

//! No-allocation tests for the audio path.
//!
//! A counting global allocator verifies that `process`, `observe`, `flush`,
//! and variable-rate processing do not allocate after `prepare`.
//!
//! Per-processor `process`/`flush`/`observe` coverage iterates
//! `bisque::testing::registry`, so a new registry entry is covered here
//! automatically. Spectral entries run in `tests/spectral_no_alloc.rs`.

#![cfg(all(
    feature = "analysis",
    feature = "dynamics",
    feature = "filters",
    feature = "generators",
    feature = "mastering",
    feature = "repair",
    feature = "test-support",
    feature = "time"
))]
#![allow(unsafe_code)]

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

use bisque::host::PreparedProcessor;
use bisque::mastering::Gain;
use bisque::parameter::ParamEvent;
use bisque::processor::{
    AudioBlock, AudioBlockMut, ProcessContext, ProcessSpec, Processor, RingSource, Tail,
};
use bisque::testing::registry;

/// Ring of the most recent allocation sizes for over-count reports. Sizes
/// usually identify the caller.
const DIAG_SLOTS: usize = 32;

thread_local! {
    /// Per-thread allocation count. The no-allocation property is a property
    /// of the thread driving the audio path: libtest's main thread performs
    /// its own bookkeeping (HashMap inserts in `test::run_tests`)
    /// concurrently with the measured region, which a process-global counter
    /// falsely attributes to the audio path on slow runners.
    static ALLOCS: Cell<usize> = const { Cell::new(0) };
    /// Per-thread ring of recent allocation sizes for allocation reports.
    static DIAG_SIZES: [Cell<usize>; DIAG_SLOTS] =
        const { [const { Cell::new(0) }; DIAG_SLOTS] };
    /// While set, a counted allocation on this thread prints a backtrace to
    /// stderr, visible in the test's captured output.
    static ARMED: Cell<bool> = const { Cell::new(false) };
    /// Backtrace budget that keeps runaway paths from writing megabytes.
    static DIAG_BACKTRACES: Cell<usize> = const { Cell::new(4) };
    /// Recursion guard: allocations made by the diagnostic itself (backtrace
    /// capture, formatting, stderr) are forwarded uncounted.
    static IN_DIAG: Cell<bool> = const { Cell::new(false) };
}

/// This thread's allocation count.
fn allocs() -> usize {
    ALLOCS.with(Cell::get)
}

/// The sizes recorded on this thread for allocations `before..after`.
fn diag_sizes(before: usize, after: usize) -> Vec<usize> {
    DIAG_SIZES.with(|sizes| {
        (before..after)
            .map(|n| sizes[n % DIAG_SLOTS].get())
            .collect()
    })
}

/// Arm the backtrace diagnostic for this thread's counted regions.
fn arm(on: bool) {
    ARMED.with(|flag| flag.set(on));
}

fn report_armed_alloc(n: usize, size: usize) {
    let budgeted = ARMED.with(Cell::get)
        && DIAG_BACKTRACES.with(|left| {
            let l = left.get();
            left.set(l.saturating_sub(1));
            l > 0
        });
    if budgeted {
        IN_DIAG.with(|flag| {
            if !flag.replace(true) {
                let bt = std::backtrace::Backtrace::force_capture();
                eprintln!("armed allocation #{n} of {size} bytes at:\n{bt}");
                flag.set(false);
            }
        });
    }
}

struct Counting;

// SAFETY: forwards every call to the system allocator unchanged, only bumping
// a per-thread counter on allocation. The forwarded pointers/layouts are
// exactly those the caller supplied, so the System allocator's invariants are
// upheld. `try_with` skips counting during thread teardown, when the
// thread-locals are gone; teardown allocations are not audio-path work.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if IN_DIAG.try_with(Cell::get).unwrap_or(true) {
            return System.alloc(layout);
        }
        let n = ALLOCS.with(|c| {
            let n = c.get();
            c.set(n + 1);
            n
        });
        DIAG_SIZES.with(|sizes| sizes[n % DIAG_SLOTS].set(layout.size()));
        report_armed_alloc(n, layout.size());
        System.alloc(layout)
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        System.dealloc(ptr, layout);
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        if IN_DIAG.try_with(Cell::get).unwrap_or(true) {
            return System.realloc(ptr, layout, new_size);
        }
        let n = ALLOCS.with(|c| {
            let n = c.get();
            c.set(n + 1);
            n
        });
        DIAG_SIZES.with(|sizes| sizes[n % DIAG_SLOTS].set(new_size));
        report_armed_alloc(n, new_size);
        System.realloc(ptr, layout, new_size)
    }
}

#[global_allocator]
static GLOBAL: Counting = Counting;

// Nested in `no_alloc` so the CI `-- no_alloc::` filter selects this test.
mod no_alloc {
    use super::{
        allocs, arm, diag_sizes, registry, AudioBlock, AudioBlockMut, Gain, ParamEvent,
        PreparedProcessor, ProcessContext, ProcessSpec, Processor, RingSource, Tail,
    };

    /// A deterministic event schedule from an entry's declared parameters.
    ///
    /// Built before the counter is armed. Setting quarter-range,
    /// three-quarter-range, and default values exercises latching, smoothing,
    /// and per-sub-block recomputation (for example biquad coefficients) for
    /// every declared parameter.
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
                offset: at(400),
                param: info.id,
                value: info.range.0 + 0.75 * span,
            });
            events.push(ParamEvent {
                offset: at(700),
                param: info.id,
                value: info.default,
            });
        }
        events.sort_by_key(|e| e.offset);
        events
    }

    /// Run `process` repeatedly with the counter armed and assert zero
    /// allocations, reporting the registry id with any allocation.
    fn assert_process_is_alloc_free(
        id: &str,
        proc: &mut dyn Processor<f32>,
        ctx: &mut ProcessContext<'_, '_, f32>,
    ) {
        let before = allocs();
        arm(true);
        for _ in 0..64 {
            proc.process(ctx);
        }
        arm(false);
        let after = allocs();
        assert_eq!(
            after - before,
            0,
            "{id}: process allocated {} time(s) (sizes {:?}); the audio path must be alloc-free",
            after - before,
            diag_sizes(before, after)
        );
    }

    // The counter is per-thread, so every armed region stays in this one
    // test function: one test, one thread, one uninterrupted measurement.
    #[test]
    #[allow(clippy::too_many_lines)]
    fn process_does_not_allocate() {
        let spec = ProcessSpec {
            sample_rate: 48_000,
            channels: 2,
            max_block: 8192,
            max_memory: None,
        };
        let frames = 1000;
        let channels = spec.channels;

        // --- Registry-driven coverage -----------------------------------
        // Every non-spectral processor entry processes with the counter armed.
        // prepare() may allocate; processors, buffers, events, and contexts are
        // built before each entry's armed region.
        for entry in registry::processor_entries()
            .iter()
            .filter(|e| e.feature != "spectral")
        {
            let mut proc = (entry.make)();
            proc.prepare(spec).unwrap_or_else(|e| {
                panic!("{}: prepare failed on the standard spec: {e:?}", entry.id)
            });
            let events = schedule_for(&*proc, frames);

            // KernelProcessor above unity so limiter/dynamics gain paths engage;
            // sources overwrite a zeroed buffer.
            let mut buf: Vec<Vec<f32>> = match entry.drive {
                registry::DriveMode::Source => vec![vec![0.0f32; frames]; channels],
                _ => (0..channels)
                    .map(|ch| {
                        (0..frames)
                            .map(|i| (i as f32 * (0.013 + 0.006 * ch as f32)).sin() * 1.6)
                            .collect()
                    })
                    .collect(),
            };
            // A key signal for sidechain entries, always built (cheap) so the
            // borrow lives long enough.
            let sc_buf: Vec<Vec<f32>> = (0..channels)
                .map(|ch| {
                    (0..frames)
                        .map(|i| (i as f32 * (0.021 + 0.004 * ch as f32)).cos() * 0.8)
                        .collect()
                })
                .collect();
            let sc_planes: Vec<&[f32]> = sc_buf.iter().map(Vec::as_slice).collect();
            let sc_blocks = [AudioBlock::new(&sc_planes)];
            let sidechain: &[AudioBlock<'_, '_, f32>] = if entry.sidechain_inputs > 0 {
                &sc_blocks
            } else {
                &[]
            };

            match entry.drive {
                registry::DriveMode::Split => {
                    let in_planes: Vec<&[f32]> = buf.iter().map(Vec::as_slice).collect();
                    let mut out_buf: Vec<Vec<f32>> = vec![vec![0.0f32; frames]; channels];
                    let mut out_planes: Vec<&mut [f32]> =
                        out_buf.iter_mut().map(Vec::as_mut_slice).collect();
                    let mut ctx = ProcessContext::split(&in_planes, &mut out_planes, 0)
                        .with_sidechains(sidechain)
                        .with_events(&events);
                    assert_process_is_alloc_free(entry.id, &mut *proc, &mut ctx);
                }
                registry::DriveMode::Effect => {
                    let mut planes: Vec<&mut [f32]> =
                        buf.iter_mut().map(Vec::as_mut_slice).collect();
                    let mut ctx = ProcessContext::in_place(&mut planes, 0)
                        .with_sidechains(sidechain)
                        .with_events(&events);
                    assert_process_is_alloc_free(entry.id, &mut *proc, &mut ctx);
                }
                registry::DriveMode::Source => {
                    let mut planes: Vec<&mut [f32]> =
                        buf.iter_mut().map(Vec::as_mut_slice).collect();
                    let mut ctx = ProcessContext::output_only(&mut planes, 0)
                        .with_sidechains(sidechain)
                        .with_events(&events);
                    assert_process_is_alloc_free(entry.id, &mut *proc, &mut ctx);
                }
            }

            if !matches!(proc.tail(), Tail::None) {
                let mut flush_buf = vec![vec![0.0f32; frames]; channels];
                let mut flush_planes: Vec<&mut [f32]> =
                    flush_buf.iter_mut().map(Vec::as_mut_slice).collect();
                let mut flush_block = AudioBlockMut::new(&mut flush_planes);
                let before = allocs();
                arm(true);
                for _ in 0..64 {
                    let produced = proc.flush(&mut flush_block);
                    if produced.frames == 0 || produced.done {
                        break;
                    }
                }
                arm(false);
                let after = allocs();
                assert_eq!(
                    after - before,
                    0,
                    "{}: flush allocated {} time(s) (sizes {:?}); the audio path must be \
                     alloc-free",
                    entry.id,
                    after - before,
                    diag_sizes(before, after)
                );
            }
        }

        // Every meter entry observes with the counter armed.
        let meter_buf: Vec<Vec<f32>> = (0..channels)
            .map(|ch| {
                (0..frames)
                    .map(|i| (i as f32 * (0.03 + 0.01 * ch as f32)).sin() * 1.2)
                    .collect()
            })
            .collect();
        let meter_planes: Vec<&[f32]> = meter_buf.iter().map(Vec::as_slice).collect();
        for entry in registry::meter_entries() {
            let mut meter = (entry.make)();
            meter.prepare(spec).unwrap_or_else(|e| {
                panic!("{}: prepare failed on the standard spec: {e:?}", entry.id)
            });
            let before = allocs();
            for _ in 0..64 {
                meter.observe(AudioBlock::new(&meter_planes));
            }
            let delta = allocs() - before;
            assert_eq!(
                delta, 0,
                "{}: observe allocated {delta} time(s); the audio path must be alloc-free",
                entry.id
            );
        }

        // Every variable-rate entry pulls from a RingSource with the counter
        // armed; reset and rewind exercise the pull/refill path each iteration.
        let vr_in: Vec<Vec<f32>> = (0..channels)
            .map(|ch| {
                (0..4096)
                    .map(|i| (i as f32 * (0.02 + 0.004 * ch as f32)).sin() * 0.5)
                    .collect()
            })
            .collect();
        for entry in registry::variable_rate_entries() {
            let mut v = (entry.make)();
            v.prepare(spec).unwrap_or_else(|e| {
                panic!("{}: prepare failed on the standard spec: {e:?}", entry.id)
            });
            let mut src = RingSource::new(vr_in.clone());
            let mut out: Vec<Vec<f32>> = vec![vec![0.0f32; 256]; channels];
            let mut planes: Vec<&mut [f32]> = out.iter_mut().map(Vec::as_mut_slice).collect();
            let mut blk = AudioBlockMut::new(&mut planes);
            let before = allocs();
            for _ in 0..64 {
                v.reset();
                src.rewind();
                let _ = v.process(&mut src, &mut blk);
            }
            let delta = allocs() - before;
            assert_eq!(
                delta, 0,
                "{}: variable-rate processing allocated {delta} time(s); the audio path must \
                 be alloc-free",
                entry.id
            );
        }

        // PreparedProcessor adds timeline ownership and geometry validation but
        // no post-prepare allocation.
        let mut prepared = PreparedProcessor::prepare_kernel(Gain::new(), spec).unwrap();
        let mut prepared_buf = vec![vec![0.25f32; 64]; channels];
        let before = allocs();
        for _ in 0..64 {
            let (left, right) = prepared_buf.split_at_mut(1);
            let mut prepared_planes = [left[0].as_mut_slice(), right[0].as_mut_slice()];
            prepared.process_in_place(&mut prepared_planes, &[]);
        }
        let delta = allocs() - before;
        assert_eq!(
            delta, 0,
            "PreparedProcessor allocated {delta} time(s); the audio path must be alloc-free"
        );
    }
}
