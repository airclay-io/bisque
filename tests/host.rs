// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Airclay LLC

//! Contract tests for the optional single-processor host helper.

#![cfg(all(feature = "filters", feature = "mastering"))]

use bisque::filters::{Biquad, MovingAverage};
#[cfg(feature = "generators")]
use bisque::generators::SineOsc;
use bisque::host::PreparedProcessor;
use bisque::mastering::{Gain, GainSettings, Limiter, LimiterSettings};
use bisque::parameter::{ParamEvent, ParamId, ParamInfo, ParamSetError, Unit};
use bisque::processor::{
    AudioBlockMut, IoMode, KernelProcessor, ProcessContext, ProcessSpec, Processor, Tail,
};

const SENTINEL_PARAMS: [ParamInfo; 1] = [ParamInfo::new(
    ParamId(42),
    "sentinel",
    (-2.0, 3.0),
    0.25,
    Unit::Linear,
)];

#[derive(Debug)]
struct SentinelProcessor;

impl Processor<f32> for SentinelProcessor {
    fn prepare(&mut self, _spec: ProcessSpec) -> Result<(), bisque::processor::DspError> {
        Ok(())
    }

    fn reset(&mut self) {}

    fn latency(&self) -> usize {
        7
    }

    fn tail(&self) -> Tail {
        Tail::Frames(11)
    }

    fn io_mode(&self) -> IoMode {
        IoMode::Split
    }

    fn sidechain_inputs(&self) -> usize {
        2
    }

    fn param_info(&self) -> &[ParamInfo] {
        &SENTINEL_PARAMS
    }

    fn memory_footprint(&self) -> usize {
        4_096
    }

    fn process(&mut self, _ctx: &mut ProcessContext<'_, '_, f32>) {}
}

fn spec(channels: usize) -> ProcessSpec {
    ProcessSpec {
        sample_rate: 48_000,
        channels,
        max_block: 64,
        max_memory: None,
    }
}

#[test]
fn prepared_metadata_delegates_exact_values() {
    let prepared = PreparedProcessor::prepare(SentinelProcessor, spec(1)).expect("prepare");
    assert_eq!(prepared.latency(), 7);
    assert_eq!(prepared.tail(), Tail::Frames(11));
    assert_eq!(prepared.io_mode(), IoMode::Split);
    assert_eq!(prepared.sidechain_inputs(), 2);
    assert_eq!(prepared.memory_footprint(), 4_096);
    assert_eq!(prepared.param_info().len(), 1);
    assert_eq!(prepared.param_info()[0].id, ParamId(42));
    assert_eq!(prepared.param_info()[0].range, (-2.0, 3.0));
}

#[test]
fn prepared_matches_direct_and_tracks_irregular_blocks() {
    let settings = GainSettings::new().gain_db(-6.0);
    let mut direct = KernelProcessor::new(Gain::with_settings(settings));
    direct.prepare(spec(2)).unwrap();
    let mut prepared =
        PreparedProcessor::prepare_kernel(Gain::with_settings(settings), spec(2)).expect("prepare");

    let mut direct_audio = vec![vec![0.25f32; 93]; 2];
    let mut prepared_audio = direct_audio.clone();
    let mut pos = 0u64;
    for &(start, end) in &[(0, 7), (7, 40), (40, 93)] {
        {
            let mut planes: Vec<&mut [f32]> = direct_audio
                .iter_mut()
                .map(|channel| &mut channel[start..end])
                .collect();
            let mut ctx = ProcessContext::in_place(&mut planes, pos);
            direct.process(&mut ctx);
        }
        {
            let mut planes: Vec<&mut [f32]> = prepared_audio
                .iter_mut()
                .map(|channel| &mut channel[start..end])
                .collect();
            prepared.process_in_place(&mut planes, &[]);
        }
        pos += (end - start) as u64;
        assert_eq!(prepared.sample_pos(), pos);
    }
    assert_eq!(prepared_audio, direct_audio);
    assert_eq!(prepared.spec(), &spec(2));
}

#[test]
fn prepared_events_match_the_raw_processor_bit_for_bit() {
    let mut direct = KernelProcessor::new(Gain::new());
    direct.prepare(spec(1)).unwrap();
    let mut prepared = PreparedProcessor::prepare_kernel(Gain::new(), spec(1)).unwrap();
    let events = [
        ParamEvent {
            offset: 0,
            param: Gain::GAIN_DB,
            value: -12.0,
        },
        ParamEvent {
            offset: 15,
            param: Gain::GAIN_DB,
            value: 3.0,
        },
        ParamEvent {
            offset: 40,
            param: Gain::GAIN_DB,
            value: -6.0,
        },
    ];
    let mut direct_audio = [vec![0.5f32; 64]];
    let mut prepared_audio = direct_audio.clone();
    {
        let mut planes = [direct_audio[0].as_mut_slice()];
        let mut ctx = ProcessContext::in_place(&mut planes, 0).with_events(&events);
        direct.process(&mut ctx);
    }
    {
        let mut planes = [prepared_audio[0].as_mut_slice()];
        prepared.process_in_place(&mut planes, &events);
    }
    assert_eq!(prepared_audio, direct_audio);
    assert_eq!(prepared.sample_pos(), 64);
}

#[test]
fn prepared_in_place_reuses_one_plane_table_across_calls() {
    let mut prepared = PreparedProcessor::prepare_kernel(Gain::new(), spec(1)).unwrap();
    let mut audio = [vec![0.5f32; 8]];
    let mut planes = [audio[0].as_mut_slice()];

    prepared.process_in_place(&mut planes, &[]);
    prepared.process_in_place(&mut planes, &[]);

    assert_eq!(prepared.sample_pos(), 16);
}

#[test]
fn explicit_f64_path_restart_seek_and_immediate_write_work() {
    let mut prepared =
        PreparedProcessor::<KernelProcessor<Gain, f64>, f64>::prepare_kernel_with_sample_type(
            Gain::new(),
            spec(1),
        )
        .expect("prepare f64");
    prepared
        .set_parameter_immediate(Gain::GAIN_DB, -12.0)
        .unwrap();
    let mut audio = [vec![1.0f64; 8]];
    let mut planes: Vec<&mut [f64]> = audio.iter_mut().map(Vec::as_mut_slice).collect();
    prepared.process_in_place(&mut planes, &[]);
    assert_eq!(prepared.sample_pos(), 8);
    assert!(audio[0][0] < 1.0);

    prepared.restart();
    assert_eq!(prepared.sample_pos(), 0);
    prepared.seek(12_345);
    assert_eq!(prepared.sample_pos(), 12_345);
}

#[test]
fn split_io_and_boxed_processors_preserve_the_raw_contract() {
    let mut split = PreparedProcessor::prepare_kernel(MovingAverage::new(3), spec(1)).unwrap();
    let input = [vec![1.0f32; 16]];
    let mut output = [vec![0.0f32; 16]];
    let input_planes: Vec<&[f32]> = input.iter().map(Vec::as_slice).collect();
    let mut output_planes: Vec<&mut [f32]> = output.iter_mut().map(Vec::as_mut_slice).collect();
    split.process_split(&input_planes, &mut output_planes, &[]);
    assert_eq!(split.sample_pos(), 16);

    let boxed: Box<dyn Processor<f32> + Send> = Box::new(KernelProcessor::new(
        Gain::with_settings(GainSettings::new().gain_db(-3.0)),
    ));
    let mut boxed = PreparedProcessor::prepare(boxed, spec(1)).unwrap();
    boxed
        .set_parameter_immediate(Gain::GAIN_DB, -12.0)
        .expect("boxed delegation must reach the inner processor");
    let mut audio = [vec![1.0f32; 4]];
    let mut planes: Vec<&mut [f32]> = audio.iter_mut().map(Vec::as_mut_slice).collect();
    boxed.process_in_place(&mut planes, &[]);
    assert!(audio[0][0] < 0.3);

    let boxed64: Box<dyn Processor<f64> + Send> =
        Box::new(KernelProcessor::<Gain, f64>::with_sample_type(Gain::new()));
    let mut boxed64 = PreparedProcessor::prepare(boxed64, spec(1)).unwrap();
    let mut audio64 = [vec![0.5f64; 4]];
    let mut planes64: Vec<&mut [f64]> = audio64.iter_mut().map(Vec::as_mut_slice).collect();
    boxed64.process_in_place(&mut planes64, &[]);
    assert_eq!(audio64[0], vec![0.5; 4]);
}

#[test]
fn boxed_process_and_flush_match_concrete_bit_for_bit() {
    let settings = LimiterSettings::new().lookahead_ms(1.0);
    let mut concrete = KernelProcessor::new(Limiter::with_settings(settings));
    let mut boxed: Box<dyn Processor<f32> + Send> =
        Box::new(KernelProcessor::new(Limiter::with_settings(settings)));
    concrete.prepare(spec(1)).unwrap();
    boxed.prepare(spec(1)).unwrap();
    assert_eq!(concrete.latency(), boxed.latency());
    assert_eq!(concrete.tail(), boxed.tail());
    assert_eq!(concrete.io_mode(), boxed.io_mode());
    assert_eq!(concrete.sidechain_inputs(), boxed.sidechain_inputs());
    assert_eq!(concrete.memory_footprint(), boxed.memory_footprint());
    assert_eq!(concrete.param_info().len(), boxed.param_info().len());
    for (concrete_info, boxed_info) in concrete.param_info().iter().zip(boxed.param_info()) {
        assert_eq!(concrete_info.id, boxed_info.id);
        assert_eq!(concrete_info.range, boxed_info.range);
        assert_eq!(concrete_info.default, boxed_info.default);
        assert_eq!(concrete_info.unit, boxed_info.unit);
        assert_eq!(concrete_info.value_scale, boxed_info.value_scale);
        assert_eq!(concrete_info.smoothing, boxed_info.smoothing);
        assert_eq!(concrete_info.smoothing_ms, boxed_info.smoothing_ms);
    }

    assert!(matches!(
        concrete.set_parameter_immediate(ParamId(99), 0.0),
        Err(ParamSetError::UnknownParam(ParamId(99)))
    ));
    assert!(matches!(
        boxed.set_parameter_immediate(ParamId(99), 0.0),
        Err(ParamSetError::UnknownParam(ParamId(99)))
    ));
    assert!(matches!(
        concrete.set_parameter_immediate(Limiter::THRESHOLD_DB, f64::NAN),
        Err(ParamSetError::NonFiniteValue { param, value })
            if param == Limiter::THRESHOLD_DB && value.is_nan()
    ));
    assert!(matches!(
        boxed.set_parameter_immediate(Limiter::THRESHOLD_DB, f64::NAN),
        Err(ParamSetError::NonFiniteValue { param, value })
            if param == Limiter::THRESHOLD_DB && value.is_nan()
    ));

    let mut concrete_audio = [vec![0.25f32; 32]];
    let mut boxed_audio = concrete_audio.clone();
    {
        let mut planes = [concrete_audio[0].as_mut_slice()];
        let mut ctx = ProcessContext::in_place(&mut planes, 0);
        concrete.process(&mut ctx);
    }
    {
        let mut planes = [boxed_audio[0].as_mut_slice()];
        let mut ctx = ProcessContext::in_place(&mut planes, 0);
        boxed.process(&mut ctx);
    }
    assert_eq!(concrete_audio, boxed_audio);

    let mut concrete_tail = [vec![0.0f32; 64]];
    let mut boxed_tail = [vec![0.0f32; 64]];
    let concrete_produced = {
        let mut planes = [concrete_tail[0].as_mut_slice()];
        concrete.flush(&mut AudioBlockMut::new(&mut planes))
    };
    let boxed_produced = {
        let mut planes = [boxed_tail[0].as_mut_slice()];
        boxed.flush(&mut AudioBlockMut::new(&mut planes))
    };
    assert_eq!(concrete_produced, boxed_produced);
    assert_eq!(concrete_tail, boxed_tail);
}

#[test]
fn flush_delegates_without_advancing_the_input_timeline() {
    let settings = LimiterSettings::new().lookahead_ms(1.0);
    let mut prepared = PreparedProcessor::prepare_kernel(Limiter::with_settings(settings), spec(1))
        .expect("prepare limiter");
    let mut audio = [vec![0.5f32; 8]];
    let mut planes: Vec<&mut [f32]> = audio.iter_mut().map(Vec::as_mut_slice).collect();
    prepared.process_in_place(&mut planes, &[]);
    let pos = prepared.sample_pos();

    let mut tail = [vec![0.0f32; 128]];
    let mut tail_planes: Vec<&mut [f32]> = tail.iter_mut().map(Vec::as_mut_slice).collect();
    let mut out = AudioBlockMut::new(&mut tail_planes);
    let produced = prepared.flush(&mut out);
    assert!(produced.frames > 0);
    assert!(produced.done);
    assert_eq!(prepared.sample_pos(), pos);
}

#[test]
fn early_done_flush_is_terminal() {
    let mut prepared = PreparedProcessor::prepare_kernel(Biquad::lowpass(), spec(1)).unwrap();
    let mut silence = [vec![0.0f32; 1]];
    let mut input_planes = [silence[0].as_mut_slice()];
    prepared.process_in_place(&mut input_planes, &[]);

    for _ in 0..2 {
        let mut stage = [vec![0.0f32; 16]];
        let mut planes = [stage[0].as_mut_slice()];
        let produced = prepared.flush(&mut AudioBlockMut::new(&mut planes));
        assert_eq!(produced.frames, 0);
        assert!(produced.done);
    }
}

#[test]
fn already_done_flush_remains_terminal() {
    let mut prepared = PreparedProcessor::prepare_kernel(Gain::new(), spec(1)).unwrap();
    for _ in 0..2 {
        let mut stage = [vec![0.0f32; 8]];
        let mut planes = [stage[0].as_mut_slice()];
        let produced = prepared.flush(&mut AudioBlockMut::new(&mut planes));
        assert_eq!(produced.frames, 0);
        assert!(produced.done);
    }
}

#[cfg(feature = "test-support")]
#[test]
fn infinite_flush_remains_host_capped() {
    use bisque::testing::InfiniteTailKernel;

    let mut prepared =
        PreparedProcessor::prepare_kernel(InfiniteTailKernel::new(), spec(1)).unwrap();
    let mut input = [vec![0.5f32; 8]];
    let mut input_planes = [input[0].as_mut_slice()];
    prepared.process_in_place(&mut input_planes, &[]);

    let mut stage = [vec![0.0f32; 16]];
    let mut planes = [stage[0].as_mut_slice()];
    let produced = prepared.flush(&mut AudioBlockMut::new(&mut planes));
    assert_eq!(produced.frames, 16);
    assert!(!produced.done);
}

#[test]
fn zero_frame_processing_does_not_advance_time() {
    let mut prepared = PreparedProcessor::prepare_kernel(Gain::new(), spec(1)).unwrap();
    let mut audio = [Vec::<f32>::new()];
    let mut planes: Vec<&mut [f32]> = audio.iter_mut().map(Vec::as_mut_slice).collect();
    prepared.process_in_place(&mut planes, &[] as &[ParamEvent]);
    assert_eq!(prepared.sample_pos(), 0);
}

#[cfg(feature = "generators")]
#[test]
fn output_only_processing_advances_time_and_generates_audio() {
    let mut prepared = PreparedProcessor::prepare_kernel(SineOsc::new(), spec(1)).unwrap();
    assert_eq!(prepared.io_mode(), IoMode::OutputOnly);
    let mut audio = [vec![0.0f32; 64]];
    let mut planes = [audio[0].as_mut_slice()];
    prepared.process_output_only(&mut planes, &[]);
    assert_eq!(prepared.sample_pos(), 64);
    assert!(audio[0].iter().any(|sample| *sample != 0.0));
}

#[cfg(debug_assertions)]
#[test]
#[should_panic(expected = "flush output channel count")]
fn flush_rejects_wrong_channel_geometry() {
    let mut prepared = PreparedProcessor::prepare_kernel(Gain::new(), spec(2)).unwrap();
    let mut tail = [vec![0.0f32; 256]];
    let mut planes: Vec<&mut [f32]> = tail.iter_mut().map(Vec::as_mut_slice).collect();
    let mut out = AudioBlockMut::new(&mut planes);
    let _ = prepared.flush(&mut out);
}

#[cfg(debug_assertions)]
#[test]
#[should_panic(expected = "main channel count")]
fn process_rejects_wrong_channel_geometry() {
    let mut prepared = PreparedProcessor::prepare_kernel(Gain::new(), spec(2)).unwrap();
    let mut audio = [vec![0.0f32; 8]];
    let mut planes = [audio[0].as_mut_slice()];
    prepared.process_in_place(&mut planes, &[]);
}

#[test]
fn kernel_settings_remain_available_through_immutable_readout() {
    let prepared = PreparedProcessor::prepare_kernel(
        Gain::with_settings(GainSettings::new().gain_db(-9.0)),
        spec(1),
    )
    .unwrap();
    assert_eq!(
        Processor::<f32>::param_info(prepared.processor())[0].default,
        -9.0
    );
}

#[test]
fn restart_restores_configured_defaults_after_an_immediate_write() {
    let mut prepared = PreparedProcessor::prepare_kernel(
        Gain::with_settings(GainSettings::new().gain_db(-6.0)),
        spec(1),
    )
    .unwrap();
    prepared
        .set_parameter_immediate(Gain::GAIN_DB, -24.0)
        .unwrap();
    prepared.restart();

    let mut audio = [vec![1.0f32; 1]];
    let mut planes = [audio[0].as_mut_slice()];
    prepared.process_in_place(&mut planes, &[]);
    assert_eq!(audio[0][0], bisque::dsp::db_to_linear(-6.0) as f32);
}

#[test]
fn fixed_settings_and_equivalent_immediate_restore_agree() {
    let mut configured = PreparedProcessor::prepare_kernel(
        Gain::with_settings(GainSettings::new().gain_db(-6.0)),
        spec(1),
    )
    .unwrap();
    let mut restored = PreparedProcessor::prepare_kernel(Gain::new(), spec(1)).unwrap();
    restored
        .set_parameter_immediate(Gain::GAIN_DB, -6.0)
        .unwrap();

    let mut configured_audio = [vec![0.5f32; 64]];
    let mut restored_audio = configured_audio.clone();
    {
        let mut planes = [configured_audio[0].as_mut_slice()];
        configured.process_in_place(&mut planes, &[]);
    }
    {
        let mut planes = [restored_audio[0].as_mut_slice()];
        restored.process_in_place(&mut planes, &[]);
    }
    assert_eq!(configured_audio, restored_audio);
}

#[cfg(feature = "dynamics")]
#[test]
fn canonical_process_preserves_sidechain_routing() {
    use bisque::dynamics::{Compressor, CompressorSettings};
    use bisque::processor::{AudioBlock, Io};

    let compressor = Compressor::with_settings(
        CompressorSettings::new()
            .threshold_db(-30.0)
            .ratio(8.0)
            .use_sidechain(true),
    );
    let mut prepared = PreparedProcessor::prepare_kernel(compressor, spec(2)).unwrap();
    let mut main = [vec![0.25f32; 64], vec![0.25f32; 64]];
    let key = [vec![1.0f32; 64]];
    let key_planes: Vec<&[f32]> = key.iter().map(Vec::as_slice).collect();
    let sidechains = [AudioBlock::new(&key_planes)];
    let mut main_planes: Vec<&mut [f32]> = main.iter_mut().map(Vec::as_mut_slice).collect();
    prepared.process(
        Io::InPlace(AudioBlockMut::new(&mut main_planes)),
        &sidechains,
        &[],
    );
    assert!(main[0].iter().any(|sample| *sample < 0.25));
}
