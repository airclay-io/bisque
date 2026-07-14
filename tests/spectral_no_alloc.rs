// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Airclay LLC

//! No-allocation tests for spectral audio paths.
//!
//! A counting global allocator verifies that `process` and `flush` do not
//! allocate after `prepare`. Per-processor `process` and `flush` coverage iterates the
//! spectral entries of `bisque::testing::registry`, so a new spectral entry is
//! covered here automatically.

#![cfg(all(feature = "spectral", feature = "test-support"))]
#![allow(unsafe_code)]

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

use bisque::processor::{AudioBlockMut, ProcessContext, ProcessSpec, Processor, Tail};
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

mod no_alloc {
    use super::{
        allocs, arm, diag_sizes, registry, AudioBlockMut, ProcessContext, ProcessSpec, Processor,
        Tail,
    };

    // The counter is per-thread, so every armed region stays in this one
    // test function: one test, one thread, one uninterrupted measurement.
    #[test]
    #[allow(clippy::too_many_lines)]
    fn process_and_flush_do_not_allocate() {
        let n = 1024;
        let spec = ProcessSpec {
            sample_rate: 48_000,
            channels: 2,
            max_block: 8192,
            max_memory: None,
        };
        let frames = n; // a frame fires within each block
        let channels = spec.channels;

        // --- Registry-driven coverage -----------------------------------
        // Every spectral processor entry processes with the counter armed.
        // prepare() may allocate; buffers and contexts are built per entry
        // before its armed region.
        for entry in registry::processor_entries()
            .iter()
            .filter(|e| e.feature == "spectral")
        {
            let mut proc = (entry.make)();
            proc.prepare(spec).unwrap_or_else(|e| {
                panic!("{}: prepare failed on the standard spec: {e:?}", entry.id)
            });
            let in_buf: Vec<Vec<f32>> = (0..channels)
                .map(|ch| {
                    (0..frames)
                        .map(|i| (i as f32 * (0.02 + 0.007 * ch as f32)).sin() * 0.5)
                        .collect()
                })
                .collect();
            let mut out_buf: Vec<Vec<f32>> = vec![vec![0.0f32; frames]; channels];
            let mut in_place = in_buf.clone();
            let before;
            match entry.drive {
                registry::DriveMode::Split => {
                    let in_planes: Vec<&[f32]> = in_buf.iter().map(Vec::as_slice).collect();
                    let mut out_planes: Vec<&mut [f32]> =
                        out_buf.iter_mut().map(Vec::as_mut_slice).collect();
                    let mut ctx = ProcessContext::split(&in_planes, &mut out_planes, 0);
                    before = allocs();
                    arm(true);
                    for _ in 0..64 {
                        proc.process(&mut ctx);
                    }
                    arm(false);
                }
                registry::DriveMode::Effect => {
                    let mut planes: Vec<&mut [f32]> =
                        in_place.iter_mut().map(Vec::as_mut_slice).collect();
                    let mut ctx = ProcessContext::in_place(&mut planes, 0);
                    before = allocs();
                    arm(true);
                    for _ in 0..64 {
                        proc.process(&mut ctx);
                    }
                    arm(false);
                }
                registry::DriveMode::Source => {
                    let mut planes: Vec<&mut [f32]> =
                        in_place.iter_mut().map(Vec::as_mut_slice).collect();
                    let mut ctx = ProcessContext::output_only(&mut planes, 0);
                    before = allocs();
                    arm(true);
                    for _ in 0..64 {
                        proc.process(&mut ctx);
                    }
                    arm(false);
                }
            }
            let after = allocs();
            assert_eq!(
                after - before,
                0,
                "{}: process allocated {} time(s) (sizes {:?}); the audio path (incl. the \
                 FFT) must be alloc-free",
                entry.id,
                after - before,
                diag_sizes(before, after)
            );

            if !matches!(proc.tail(), Tail::None) {
                let mut flush_buf = vec![vec![0.0f32; n]; channels];
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
                    "{}: flush allocated {} time(s) (sizes {:?}); the spectral audio path \
                     must be alloc-free",
                    entry.id,
                    after - before,
                    diag_sizes(before, after)
                );
            }
        }
    }
}
