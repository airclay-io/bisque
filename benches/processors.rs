// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Airclay LLC

#![allow(missing_docs)]

//! Representative CPU benchmarks. These record trends; CI does not impose
//! machine-dependent pass/fail thresholds.

use bisque::analysis::{LoudnessMeter, TruePeakMeter};
use bisque::dynamics::Compressor;
use bisque::filters::Biquad;
use bisque::mastering::{Gain, Limiter};
use bisque::parameter::ParamEvent;
use bisque::processor::{
    AudioBlock, AudioBlockMut, Kernel, KernelProcessor, Measurer, ProcessContext, ProcessSpec,
    Processor, Produced, Source, VariableRate,
};
use bisque::time::{Delay, TimeStretch, TimeStretchSettings};
use criterion::{
    black_box, criterion_group, criterion_main, BatchSize, Bencher, BenchmarkId, Criterion,
    Throughput,
};

const BLOCK_SIZES: [usize; 4] = [1, 32, 64, 512];
const CHANNELS: usize = 2;

fn spec(max_block: usize) -> ProcessSpec {
    ProcessSpec {
        sample_rate: 48_000,
        channels: CHANNELS,
        max_block,
        max_memory: None,
    }
}

fn signal(frames: usize, level: f32) -> Vec<Vec<f32>> {
    vec![vec![level; frames]; CHANNELS]
}

fn stereo_planes(block: &[Vec<f32>]) -> [&[f32]; CHANNELS] {
    let (left, right) = block.split_at(1);
    [left[0].as_slice(), right[0].as_slice()]
}

fn stereo_planes_mut(block: &mut [Vec<f32>]) -> [&mut [f32]; CHANNELS] {
    let (left, right) = block.split_at_mut(1);
    [left[0].as_mut_slice(), right[0].as_mut_slice()]
}

fn prepared_kernel<K: Kernel<f32>>(kernel: K, max_block: usize) -> KernelProcessor<K> {
    let mut processor = Kernel::<f32>::into_processor(kernel);
    Processor::<f32>::prepare(&mut processor, spec(max_block)).expect("benchmark prepare");
    processor
}

fn process_in_place<P: Processor<f32>>(
    processor: &mut P,
    block: &mut [Vec<f32>],
    events: &[ParamEvent],
    sample_pos: u64,
) {
    let mut planes = stereo_planes_mut(block);
    let mut ctx = ProcessContext::in_place(&mut planes, sample_pos).with_events(events);
    processor.process(&mut ctx);
}

fn process_with_sidechain<P: Processor<f32>>(
    processor: &mut P,
    block: &mut [Vec<f32>],
    sidechain: &[Vec<f32>],
    sample_pos: u64,
) {
    let sidechain_planes = stereo_planes(sidechain);
    let sidechains = [AudioBlock::new(&sidechain_planes)];
    let mut planes = stereo_planes_mut(block);
    let mut ctx = ProcessContext::in_place(&mut planes, sample_pos).with_sidechains(&sidechains);
    processor.process(&mut ctx);
}

fn benchmark_in_place_with_events<'events, P, E>(
    b: &mut Bencher<'_>,
    processor: &mut P,
    frames: usize,
    level: f32,
    mut events_for_block: E,
) where
    P: Processor<f32>,
    E: FnMut() -> &'events [ParamEvent],
{
    let mut sample_pos = 0u64;
    b.iter_batched_ref(
        || signal(frames, level),
        |block| {
            process_in_place(processor, block, events_for_block(), sample_pos);
            sample_pos += frames as u64;
            black_box(block.as_slice());
        },
        BatchSize::LargeInput,
    );
}

fn benchmark_in_place<P: Processor<f32>>(
    b: &mut Bencher<'_>,
    processor: &mut P,
    frames: usize,
    level: f32,
    events: &[ParamEvent],
) {
    benchmark_in_place_with_events(b, processor, frames, level, || events);
}

fn benchmark_with_sidechain<P: Processor<f32>>(
    b: &mut Bencher<'_>,
    processor: &mut P,
    frames: usize,
    level: f32,
    sidechain: &[Vec<f32>],
) {
    let mut sample_pos = 0u64;
    b.iter_batched_ref(
        || signal(frames, level),
        |block| {
            process_with_sidechain(processor, block, sidechain, sample_pos);
            sample_pos += frames as u64;
            black_box(block.as_slice());
        },
        BatchSize::LargeInput,
    );
}

#[allow(clippy::too_many_lines)]
fn processor_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("processors");
    for frames in BLOCK_SIZES {
        group.throughput(Throughput::Elements((frames * CHANNELS) as u64));

        group.bench_with_input(BenchmarkId::new("gain_static", frames), &frames, |b, &n| {
            let mut p = prepared_kernel(Gain::new(), n);
            benchmark_in_place(b, &mut p, n, 0.25, &[]);
        });

        group.bench_with_input(
            BenchmarkId::new("biquad_static", frames),
            &frames,
            |b, &n| {
                let mut p = prepared_kernel(Biquad::lowpass(), n);
                benchmark_in_place(b, &mut p, n, 0.25, &[]);
            },
        );

        group.bench_with_input(
            BenchmarkId::new("biquad_automated", frames),
            &frames,
            |b, &n| {
                let mut p = prepared_kernel(Biquad::lowpass(), n);
                let low_events = [ParamEvent {
                    offset: 0,
                    param: Biquad::CUTOFF_HZ,
                    value: 200.0,
                }];
                let high_events = [ParamEvent {
                    offset: 0,
                    param: Biquad::CUTOFF_HZ,
                    value: 12_000.0,
                }];
                let mut select_high = false;
                benchmark_in_place_with_events(b, &mut p, n, 0.25, || {
                    select_high = !select_high;
                    if select_high {
                        &high_events
                    } else {
                        &low_events
                    }
                });
            },
        );

        group.bench_with_input(BenchmarkId::new("compressor", frames), &frames, |b, &n| {
            let mut p = prepared_kernel(Compressor::new(), n);
            benchmark_in_place(b, &mut p, n, 0.5, &[]);
        });

        group.bench_with_input(
            BenchmarkId::new("compressor_sidechain", frames),
            &frames,
            |b, &n| {
                let mut p = prepared_kernel(Compressor::with_sidechain(), n);
                let sidechain = signal(n, 1.0);
                benchmark_with_sidechain(b, &mut p, n, 0.25, &sidechain);
            },
        );

        group.bench_with_input(BenchmarkId::new("limiter", frames), &frames, |b, &n| {
            let mut p = prepared_kernel(Limiter::new(), n);
            benchmark_in_place(b, &mut p, n, 1.2, &[]);
        });

        group.bench_with_input(BenchmarkId::new("delay", frames), &frames, |b, &n| {
            let mut p = prepared_kernel(Delay::new(), n);
            benchmark_in_place(b, &mut p, n, 0.25, &[]);
        });

        group.bench_with_input(
            BenchmarkId::new("gain_dense_events", frames),
            &frames,
            |b, &n| {
                let mut p = prepared_kernel(Gain::new(), n);
                let events: Vec<ParamEvent> = (0..n)
                    .map(|offset| ParamEvent {
                        offset: offset as u32,
                        param: Gain::GAIN_DB,
                        value: if offset % 2 == 0 { -12.0 } else { 6.0 },
                    })
                    .collect();
                benchmark_in_place(b, &mut p, n, 0.25, &events);
            },
        );
    }
    group.finish();
}

fn meter_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("meters");
    for frames in BLOCK_SIZES {
        group.throughput(Throughput::Elements((frames * CHANNELS) as u64));
        let block = signal(frames, 0.25);
        let planes = stereo_planes(&block);

        group.bench_with_input(BenchmarkId::new("true_peak", frames), &frames, |b, &n| {
            let mut meter = TruePeakMeter::new();
            Measurer::<f32>::prepare(&mut meter, spec(n)).expect("benchmark prepare");
            b.iter(|| {
                meter.observe(AudioBlock::new(&planes));
                black_box(&meter);
            });
        });

        group.bench_with_input(BenchmarkId::new("loudness", frames), &frames, |b, &n| {
            let mut meter = LoudnessMeter::new();
            Measurer::<f32>::prepare(&mut meter, spec(n)).expect("benchmark prepare");
            b.iter(|| {
                meter.observe(AudioBlock::new(&planes));
                black_box(&meter);
            });
        });
    }
    group.finish();
}

#[derive(Debug)]
struct ConstantSource {
    channels: usize,
}

impl Source<f32> for ConstantSource {
    fn channels(&self) -> usize {
        self.channels
    }

    fn pull(&mut self, out: &mut AudioBlockMut<'_, '_, f32>) -> Produced {
        for channel in 0..out.channels() {
            out.channel_mut(channel).fill(0.25);
        }
        Produced {
            frames: out.frames(),
            done: false,
        }
    }
}

fn variable_rate_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("variable_rate");
    for frames in BLOCK_SIZES {
        group.throughput(Throughput::Elements((frames * CHANNELS) as u64));
        group.bench_with_input(
            BenchmarkId::new("time_stretch", frames),
            &frames,
            |b, &n| {
                let mut stretch =
                    TimeStretch::with_settings(TimeStretchSettings::new().stretch(1.25));
                VariableRate::<f32>::prepare(&mut stretch, spec(n)).expect("benchmark prepare");
                let mut source = ConstantSource { channels: CHANNELS };
                let mut block = signal(n, 0.0);
                let produced = {
                    let mut planes = stereo_planes_mut(&mut block);
                    let mut out = AudioBlockMut::new(&mut planes);
                    stretch.process(&mut source, &mut out)
                };
                assert_eq!(
                    produced.frames, n,
                    "endless benchmark source must fill output"
                );
                assert!(
                    !produced.done,
                    "endless benchmark source must remain active"
                );
                b.iter(|| {
                    let mut planes = stereo_planes_mut(&mut block);
                    let mut out = AudioBlockMut::new(&mut planes);
                    let _ = black_box(stretch.process(&mut source, &mut out));
                    black_box(block.as_slice());
                });
            },
        );
    }
    group.finish();
}

#[cfg(feature = "spectral")]
fn spectral_benchmarks(c: &mut Criterion) {
    use bisque::spectral::SpectralFilter;

    let mut group = c.benchmark_group("spectral");
    for frames in BLOCK_SIZES {
        group.throughput(Throughput::Elements((frames * CHANNELS) as u64));
        group.bench_with_input(BenchmarkId::new("filter", frames), &frames, |b, &n| {
            let mut filter = SpectralFilter::low_pass(1024, 512, 8_000.0);
            Processor::<f32>::prepare(&mut filter, spec(n)).expect("benchmark prepare");
            let input = signal(n, 0.25);
            let input_planes = stereo_planes(&input);
            let mut output = signal(n, 0.0);
            let mut pos = 0u64;
            b.iter(|| {
                let mut output_planes = stereo_planes_mut(&mut output);
                let mut ctx = ProcessContext::split(&input_planes, &mut output_planes, pos);
                filter.process(&mut ctx);
                pos += n as u64;
                black_box(&output);
            });
        });
    }
    group.finish();
}

#[cfg(feature = "spectral")]
criterion_group!(
    benches,
    processor_benchmarks,
    meter_benchmarks,
    variable_rate_benchmarks,
    spectral_benchmarks
);

#[cfg(not(feature = "spectral"))]
criterion_group!(
    benches,
    processor_benchmarks,
    meter_benchmarks,
    variable_rate_benchmarks
);

criterion_main!(benches);
