// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Airclay LLC

//! Property-based invariance tests.
//!
//! Registered processors are compared across random block partitions and
//! event schedules. Registered variable-rate processors are compared across
//! random source-pull and output-capacity patterns.
//!
//! Each test builds its own runner from a fixed `ChaCha` seed and disables
//! failure persistence. Runs are reproducible and do not write
//! `proptest-regressions` files.

#![cfg(feature = "test-support")]

/// Nested in `contract` so the CI `-- contract::` filter selects these tests.
mod contract {
    use bisque::parameter::ParamEvent;
    use bisque::processor::{
        AudioBlock, AudioBlockMut, ProcessContext, Processor, Produced, Source,
    };
    use bisque::testing::registry::{self, DriveMode, ProcessorEntry, VariableRateEntry};
    use bisque::testing::{bits_eq, sine, Buffers, Contract};
    use proptest::prelude::*;
    use proptest::test_runner::{Config, RngAlgorithm, TestCaseError, TestRng, TestRunner};

    /// Long enough to cross the default spectral window and many control cells.
    const TOTAL: usize = 1537;

    /// Fixed RNG seed. The suite derives no randomness from ambient entropy.
    const SEED: [u8; 32] = *b"bisque-property-contract-seed-01";

    fn runner() -> TestRunner {
        let config = Config {
            // Registry breadth supplies coverage, so a modest per-entry count
            // keeps the all-features suite practical.
            cases: 16,
            failure_persistence: None,
            ..Config::default()
        };
        TestRunner::new_with_rng(config, TestRng::from_seed(RngAlgorithm::ChaCha, &SEED))
    }

    fn chunks_strategy() -> impl Strategy<Value = Vec<usize>> {
        proptest::collection::vec(1usize..=257, 1..=24)
    }

    fn raw_events_strategy() -> impl Strategy<Value = Vec<(u32, usize, f64)>> {
        proptest::collection::vec((0..TOTAL as u32, 0usize..8, 0.0f64..=1.0), 0..=12)
    }

    fn schedule(proc: &dyn Processor<f32>, raw: &[(u32, usize, f64)]) -> Vec<ParamEvent> {
        let infos = proc.param_info();
        if infos.is_empty() {
            return Vec::new();
        }
        let mut events: Vec<ParamEvent> = raw
            .iter()
            .map(|&(offset, index, t)| {
                let info = &infos[index % infos.len()];
                ParamEvent {
                    offset,
                    param: info.id,
                    value: info.range.0 + t * (info.range.1 - info.range.0),
                }
            })
            .collect();
        events.sort_by_key(|event| event.offset);
        events
    }

    fn events_in(events: &[ParamEvent], lo: usize, hi: usize) -> Vec<ParamEvent> {
        events
            .iter()
            .filter(|event| {
                let offset = event.offset as usize;
                offset >= lo && offset < hi
            })
            .map(|event| ParamEvent {
                offset: event.offset - lo as u32,
                param: event.param,
                value: event.value,
            })
            .collect()
    }

    fn key_signal(channels: usize, frames: usize) -> Buffers {
        sine(channels, frames)
            .into_iter()
            .map(|channel| {
                channel
                    .into_iter()
                    .rev()
                    .map(|sample| sample * 0.8)
                    .collect()
            })
            .collect()
    }

    fn run_processor_chunked(
        entry: &ProcessorEntry,
        proc: &mut dyn Processor<f32>,
        input: &Buffers,
        key: &Buffers,
        events: &[ParamEvent],
        chunks: &[usize],
    ) -> Buffers {
        let frames = input.first().map_or(0, Vec::len);
        let channels = input.len();
        let mut output = match entry.drive {
            DriveMode::Effect => input.clone(),
            DriveMode::Source | DriveMode::Split => vec![vec![0.0f32; frames]; channels],
        };
        let mut pos = 0usize;
        let mut chunk_index = 0usize;

        while pos < frames {
            let block = chunks[chunk_index % chunks.len()].min(frames - pos);
            chunk_index += 1;
            let (lo, hi) = (pos, pos + block);
            let block_events = events_in(events, lo, hi);
            let key_planes: Vec<&[f32]> = key.iter().map(|channel| &channel[lo..hi]).collect();
            let sidechains: Vec<AudioBlock<'_, '_, f32>> = (0..entry.sidechain_inputs)
                .map(|_| AudioBlock::new(&key_planes))
                .collect();

            match entry.drive {
                DriveMode::Effect => {
                    let mut planes: Vec<&mut [f32]> = output
                        .iter_mut()
                        .map(|channel| &mut channel[lo..hi])
                        .collect();
                    let mut ctx = ProcessContext::in_place(&mut planes, pos as u64)
                        .with_sidechains(&sidechains)
                        .with_events(&block_events);
                    proc.process(&mut ctx);
                }
                DriveMode::Source => {
                    let mut planes: Vec<&mut [f32]> = output
                        .iter_mut()
                        .map(|channel| &mut channel[lo..hi])
                        .collect();
                    let mut ctx = ProcessContext::output_only(&mut planes, pos as u64)
                        .with_sidechains(&sidechains)
                        .with_events(&block_events);
                    proc.process(&mut ctx);
                }
                DriveMode::Split => {
                    let input_planes: Vec<&[f32]> =
                        input.iter().map(|channel| &channel[lo..hi]).collect();
                    let mut output_planes: Vec<&mut [f32]> = output
                        .iter_mut()
                        .map(|channel| &mut channel[lo..hi])
                        .collect();
                    let mut ctx =
                        ProcessContext::split(&input_planes, &mut output_planes, pos as u64)
                            .with_sidechains(&sidechains)
                            .with_events(&block_events);
                    proc.process(&mut ctx);
                }
            }
            pos = hi;
        }
        output
    }

    fn processor_partition_matches_whole(
        entry: &ProcessorEntry,
        chunks: &[usize],
        raw: &[(u32, usize, f64)],
    ) -> Result<(), TestCaseError> {
        let contract = Contract::default();
        let input = sine(contract.spec.channels, TOTAL);
        let key = key_signal(contract.spec.channels, TOTAL);
        let probe = (entry.make)();
        let events = schedule(&*probe, raw);

        let mut reference_proc = (entry.make)();
        reference_proc
            .prepare(contract.spec)
            .expect("prepare reference");
        let reference =
            run_processor_chunked(entry, &mut *reference_proc, &input, &key, &events, &[TOTAL]);

        let mut chunked_proc = (entry.make)();
        chunked_proc
            .prepare(contract.spec)
            .expect("prepare chunked");
        let chunked =
            run_processor_chunked(entry, &mut *chunked_proc, &input, &key, &events, chunks);

        prop_assert!(
            bits_eq(&chunked, &reference),
            "{} diverged for chunks {chunks:?} and {} events",
            entry.id,
            events.len()
        );
        Ok(())
    }

    fn check_processor(entry: &ProcessorEntry) {
        runner()
            .run(
                &(chunks_strategy(), raw_events_strategy()),
                |(chunks, raw)| processor_partition_matches_whole(entry, &chunks, &raw),
            )
            .unwrap_or_else(|error| panic!("{} property failed: {error}", entry.id));
    }

    #[test]
    fn every_registered_processor_is_invariant_to_random_partitions() {
        for entry in registry::processor_entries() {
            check_processor(&entry);
        }
    }

    struct PatternSource {
        data: Buffers,
        read: usize,
        chunks: Vec<usize>,
        chunk_index: usize,
    }

    impl PatternSource {
        fn new(data: Buffers, chunks: &[usize]) -> Self {
            Self {
                data,
                read: 0,
                chunks: chunks.to_vec(),
                chunk_index: 0,
            }
        }

        fn remaining(&self) -> usize {
            self.data.first().map_or(0, Vec::len) - self.read
        }
    }

    impl Source<f32> for PatternSource {
        fn channels(&self) -> usize {
            self.data.len()
        }

        fn pull(&mut self, out: &mut AudioBlockMut<'_, '_, f32>) -> Produced {
            let cap = self.chunks[self.chunk_index % self.chunks.len()];
            self.chunk_index += 1;
            let frames = out.frames().min(self.remaining()).min(cap);
            for channel in 0..self.data.len() {
                out.channel_mut(channel)[..frames]
                    .copy_from_slice(&self.data[channel][self.read..self.read + frames]);
            }
            self.read += frames;
            Produced {
                frames,
                done: self.remaining() == 0,
            }
        }
    }

    fn run_variable_rate(
        entry: &VariableRateEntry,
        input: &Buffers,
        output_chunks: &[usize],
        source_chunks: &[usize],
    ) -> Buffers {
        let contract = Contract::default();
        let mut processor = (entry.make)();
        processor
            .prepare(contract.spec)
            .expect("prepare variable-rate");
        let mut source = PatternSource::new(input.clone(), source_chunks);
        let mut output = vec![Vec::new(); contract.spec.channels];
        for output_index in 0..100_000 {
            let capacity = output_chunks[output_index % output_chunks.len()];
            let mut stage = vec![vec![0.0f32; capacity]; contract.spec.channels];
            let produced = {
                let mut planes: Vec<&mut [f32]> = stage.iter_mut().map(Vec::as_mut_slice).collect();
                let mut block = AudioBlockMut::new(&mut planes);
                processor.process(&mut source, &mut block)
            };
            for (channel, staged) in output.iter_mut().zip(&stage) {
                channel.extend_from_slice(&staged[..produced.frames]);
            }
            if produced.done {
                return output;
            }
        }
        panic!("{} did not finish under bounded chunk patterns", entry.id);
    }

    fn variable_rate_patterns_match_reference(
        entry: &VariableRateEntry,
        output_chunks: &[usize],
        source_chunks: &[usize],
    ) -> Result<(), TestCaseError> {
        let contract = Contract::default();
        let input = sine(contract.spec.channels, TOTAL);
        let reference_capacity = 2 * TOTAL + 4096;
        let reference = run_variable_rate(entry, &input, &[reference_capacity], &[TOTAL]);
        let patterned = run_variable_rate(entry, &input, output_chunks, source_chunks);
        prop_assert!(
            bits_eq(&patterned, &reference),
            "{} diverged for output chunks {output_chunks:?} and source chunks {source_chunks:?}",
            entry.id
        );
        Ok(())
    }

    fn check_variable_rate(entry: &VariableRateEntry) {
        runner()
            .run(
                &(chunks_strategy(), chunks_strategy()),
                |(output_chunks, source_chunks)| {
                    variable_rate_patterns_match_reference(entry, &output_chunks, &source_chunks)
                },
            )
            .unwrap_or_else(|error| panic!("{} property failed: {error}", entry.id));
    }

    #[test]
    fn every_variable_rate_entry_is_invariant_to_random_pull_and_output_patterns() {
        for entry in registry::variable_rate_entries() {
            check_variable_rate(&entry);
        }
    }
}
