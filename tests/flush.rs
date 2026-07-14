// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Airclay LLC

//! Flush-contract tests for tail draining.
//!
//! Pins the per-call `flush` semantics: a call writes at most `out.frames()`
//! frames, any total cap on a drain belongs to the host, a `Tail::Infinite`
//! drain (via the shared `InfiniteTailKernel` test double) never reports
//! `done` and must be capped by the host, and a finite drain ends with
//! `done == true` after exactly its declared tail.

#![cfg(feature = "test-support")]

mod contract {
    use bisque::processor::KernelProcessor;
    use bisque::processor::{
        AudioBlockMut, ProcessContext, ProcessSpec, Processor, Produced, Tail,
    };
    use bisque::testing::{Buffers, Contract, InfiniteTailKernel};

    /// Flush a prepared processor into a `frames`-frame stereo stage.
    fn flush_stage(proc: &mut impl Processor<f32>, frames: usize) -> (Buffers, Produced) {
        let mut stage: Buffers = vec![vec![0.0f32; frames]; 2];
        let produced = {
            let mut planes: Vec<&mut [f32]> = stage.iter_mut().map(Vec::as_mut_slice).collect();
            let mut out = AudioBlockMut::new(&mut planes);
            proc.flush(&mut out)
        };
        (stage, produced)
    }

    #[test]
    fn a_host_caps_an_infinite_drain_by_bounding_requested_frames() {
        // The infinite-tail host pattern: the processor never reports `done`,
        // so the host bounds the total frames it requests. Every requested
        // frame is filled, and the tail stays live at the cap.
        let c = Contract::default();
        let mut proc = KernelProcessor::new(InfiniteTailKernel::new());
        Processor::<f32>::prepare(&mut proc, c.spec).expect("prepare");
        assert_eq!(
            Processor::<f32>::tail(&proc),
            Tail::Infinite,
            "the double declares Infinite"
        );

        // Seed the recursive state with a non-zero signal.
        let input: Buffers = vec![vec![0.5f32; 256]; 2];
        let _ = c.run_reusing(&mut proc, &input, &[], 64);

        let cap = 1000usize;
        let mut total = 0usize;
        let mut first_frame = 0.0f32;
        while total < cap {
            let want = 300.min(cap - total);
            let (stage, produced) = flush_stage(&mut proc, want);
            assert!(
                !produced.done,
                "an infinite tail reports done == false on every call"
            );
            assert_eq!(
                produced.frames, want,
                "an infinite tail fills every requested frame"
            );
            if total == 0 {
                first_frame = stage[0][0];
            }
            total += produced.frames;
        }
        assert_eq!(total, cap, "the host's cap bounds the whole drain");
        assert!(
            first_frame != 0.0,
            "the drained tail carries the decaying state, not silence"
        );
    }

    #[cfg(feature = "mastering")]
    #[test]
    fn a_finite_drain_ends_done_after_its_declared_tail() {
        // The finite-tail host pattern: keep requesting frames until `done`.
        // The limiter's lookahead drains completely across undersized stages.
        use bisque::mastering::Limiter;
        use bisque::testing::sine;

        let c = Contract::default();
        let mut proc = KernelProcessor::new(Limiter::new());
        Processor::<f32>::prepare(&mut proc, c.spec).expect("prepare");
        let look = Processor::<f32>::latency(&proc);
        assert_eq!(
            Processor::<f32>::tail(&proc),
            Tail::Frames(look),
            "the tail is the lookahead"
        );
        let _ = c.run_reusing(&mut proc, &sine(2, 600), &[], 64);

        let mut total = 0usize;
        loop {
            let (_, produced) = flush_stage(&mut proc, 50);
            total += produced.frames;
            if produced.done {
                break;
            }
            assert!(total <= look, "a finite tail must not over-produce");
        }
        assert_eq!(
            total, look,
            "a finite drain yields exactly the declared tail"
        );
    }

    #[cfg(feature = "mastering")]
    #[test]
    fn same_length_compensation_handles_input_shorter_than_latency() {
        use bisque::mastering::Limiter;

        let mut contract = Contract::default();
        contract.spec.channels = 1;
        let mut proc = KernelProcessor::new(Limiter::new());
        proc.prepare(contract.spec).unwrap();
        let latency = proc.latency();
        let input: Buffers = vec![vec![0.1f32; 8]];
        assert!(input[0].len() < latency);

        let body = contract.run_reusing(&mut proc, &input, &[], 8);
        let mut timeline = body[0].clone();
        loop {
            let mut stage = [vec![0.0f32; 17]];
            let produced = {
                let mut planes = [stage[0].as_mut_slice()];
                let mut out = AudioBlockMut::new(&mut planes);
                proc.flush(&mut out)
            };
            timeline.extend_from_slice(&stage[0][..produced.frames]);
            if produced.done {
                break;
            }
        }

        let compensated = &timeline[latency..latency + input[0].len()];
        assert_eq!(compensated, input[0].as_slice());
    }

    #[cfg(feature = "filters")]
    mod chain {
        use super::*;
        use bisque::filters::MovingAverage;

        fn mono_spec(max_block: usize) -> ProcessSpec {
            ProcessSpec {
                sample_rate: 48_000,
                channels: 1,
                max_block,
                max_memory: None,
            }
        }

        fn process_split(
            proc: &mut impl Processor<f32>,
            input: &[f32],
            sample_pos: u64,
        ) -> Vec<f32> {
            let input_planes = [input];
            let mut output = vec![0.0f32; input.len()];
            let mut output_planes = [output.as_mut_slice()];
            let mut ctx = ProcessContext::split(&input_planes, &mut output_planes, sample_pos);
            proc.process(&mut ctx);
            output
        }

        fn flush_mono(proc: &mut impl Processor<f32>, chunk: usize) -> Vec<f32> {
            let mut drained = Vec::new();
            loop {
                let mut stage = vec![0.0f32; chunk];
                let produced = {
                    let mut planes = [stage.as_mut_slice()];
                    let mut out = AudioBlockMut::new(&mut planes);
                    proc.flush(&mut out)
                };
                drained.extend_from_slice(&stage[..produced.frames]);
                if produced.done {
                    return drained;
                }
            }
        }

        #[test]
        fn moving_average_orders_delayed_body_before_append_only_tail() {
            let mut proc = KernelProcessor::new(MovingAverage::new(5));
            proc.prepare(mono_spec(64)).unwrap();
            assert_eq!(proc.latency(), 2);
            assert_eq!(proc.tail(), Tail::Frames(4));
            let _ = process_split(&mut proc, &[1.0, 0.0, 0.0], 0);
            let tail = flush_mono(&mut proc, 1);
            assert_eq!(tail.len(), 4);
            assert!(tail[..proc.latency()].iter().any(|sample| *sample != 0.0));
            assert_eq!(tail[proc.latency()..], [0.0, 0.0]);
        }

        #[test]
        fn upstream_flush_runs_through_downstream_before_downstream_flush() {
            let input = [1.0f32, 0.0, 0.0];
            let spec = mono_spec(32);
            let mut upstream = KernelProcessor::new(MovingAverage::new(5));
            let mut downstream = KernelProcessor::new(MovingAverage::new(5));
            upstream.prepare(spec).unwrap();
            downstream.prepare(spec).unwrap();
            let aggregate_latency = upstream.latency() + downstream.latency();

            let first = process_split(&mut upstream, &input, 0);
            let mut timeline = process_split(&mut downstream, &first, 0);
            let upstream_tail = flush_mono(&mut upstream, 2);
            timeline.extend(process_split(
                &mut downstream,
                &upstream_tail,
                input.len() as u64,
            ));
            timeline.extend(flush_mono(&mut downstream, 2));

            let mut padded = input.to_vec();
            padded.resize(input.len() + 8, 0.0);
            let mut ref_upstream = KernelProcessor::new(MovingAverage::new(5));
            let mut ref_downstream = KernelProcessor::new(MovingAverage::new(5));
            ref_upstream.prepare(spec).unwrap();
            ref_downstream.prepare(spec).unwrap();
            let reference_first = process_split(&mut ref_upstream, &padded, 0);
            let reference = process_split(&mut ref_downstream, &reference_first, 0);

            assert_eq!(timeline, reference);
            assert_eq!(aggregate_latency, 4);
            assert_eq!(
                timeline[aggregate_latency..aggregate_latency + input.len()].len(),
                input.len()
            );
        }
    }

    #[cfg(feature = "spectral")]
    #[test]
    fn spectral_tail_equals_latency_and_drains_to_done() {
        use bisque::spectral::SpectralFilter;

        let mut proc = SpectralFilter::low_pass(64, 32, 8_000.0);
        let spec = ProcessSpec {
            sample_rate: 48_000,
            channels: 2,
            max_block: 128,
            max_memory: None,
        };
        Processor::<f32>::prepare(&mut proc, spec).unwrap();
        let latency = Processor::<f32>::latency(&proc);
        assert_eq!(Processor::<f32>::tail(&proc), Tail::Frames(latency));
        let input = [vec![0.0f32; 32], vec![0.0f32; 32]];
        let input_planes: Vec<&[f32]> = input.iter().map(Vec::as_slice).collect();
        let mut output = [vec![0.0f32; 32], vec![0.0f32; 32]];
        let mut output_planes: Vec<&mut [f32]> = output.iter_mut().map(Vec::as_mut_slice).collect();
        let mut ctx = ProcessContext::split(&input_planes, &mut output_planes, 0);
        Processor::<f32>::process(&mut proc, &mut ctx);
        let (_, produced) = flush_stage(&mut proc, latency);
        assert!(produced.done);
        assert_eq!(produced.frames, latency);
    }
}
