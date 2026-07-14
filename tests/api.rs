// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Airclay LLC

//! Public import smoke tests for the one-crate API.

#[test]
fn contract_types_have_canonical_module_paths() {
    use bisque::parameter::{
        NoParams, ParamEvent, ParamId, ParamInfo, ParamSetError, ParamValueError, Params,
        Smoothing, Unit, ValueScale,
    };
    use bisque::processor::{
        AudioBlock, AudioBlockMut, DspError, Io, IoMode, Kernel, Measurer, ProcessContext,
        ProcessSpec, Processor, Produced, RingSource, Sample, Source, SubBlock, Tail, VariableRate,
    };

    fn _sample<T: Sample>() {}
    fn _kernel<T: Kernel<f32>>() {}
    fn _measurer<T: Measurer<f32>>() {}
    fn _params<T: Params>() {}
    fn _processor<T: Processor<f32>>() {}
    fn _source<T: Source<f32>>() {}
    fn _variable_rate<T: VariableRate<f32>>() {}

    let _ = core::mem::size_of::<AudioBlock<'_, '_, f32>>();
    let _ = core::mem::size_of::<AudioBlockMut<'_, '_, f32>>();
    let _ = core::mem::size_of::<DspError>();
    let _ = core::mem::size_of::<Io<'_, '_, f32>>();
    let _ = core::mem::size_of::<IoMode>();
    let _ = core::mem::size_of::<NoParams>();
    let _ = core::mem::size_of::<ParamEvent>();
    let _ = core::mem::size_of::<ParamId>();
    let _ = core::mem::size_of::<ParamInfo>();
    let _ = core::mem::size_of::<ParamSetError>();
    let _ = core::mem::size_of::<ParamValueError>();
    let _ = core::mem::size_of::<ProcessContext<'_, '_, f32>>();
    let _ = core::mem::size_of::<ProcessSpec>();
    let _ = core::mem::size_of::<Produced>();
    let _ = core::mem::size_of::<RingSource<f32>>();
    let _ = core::mem::size_of::<Smoothing>();
    let _ = core::mem::size_of::<SubBlock<'_, '_, '_, f32>>();
    let _ = core::mem::size_of::<Tail>();
    let _ = core::mem::size_of::<Unit>();
    let _ = core::mem::size_of::<ValueScale>();
}

#[test]
fn downstream_static_param_tables_use_only_const_builders() {
    use bisque::parameter::{ParamId, ParamInfo, Smoothing, Unit, ValueScale};

    const PARAMS: [ParamInfo; 1] =
        [
            ParamInfo::new(ParamId(0), "frequency", (20.0, 20_000.0), 440.0, Unit::Hz)
                .with_smoothing(Smoothing::OnePole)
                .with_smoothing_ms(12.0)
                .with_value_scale(ValueScale::Linear),
        ];

    assert_eq!(PARAMS[0].id, ParamId(0));
    assert_eq!(PARAMS[0].value_scale, ValueScale::Linear);
}

#[test]
fn params_macro_is_available_at_the_crate_root() {
    bisque::params! {
        /// Smoothed parameter values for the smoke test.
        pub struct SmokeParams {
            /// A test value.
            pub value => VALUE,
        }
    }
    assert_eq!(SmokeParams::VALUE, bisque::parameter::ParamId(0));
}

#[test]
fn prepared_processor_lives_under_host() {
    use bisque::host::PreparedProcessor;
    use bisque::processor::KernelProcessor;

    struct DummyKernel;
    let _ = core::mem::size_of::<PreparedProcessor<KernelProcessor<DummyKernel>>>();
}

#[test]
fn dsp_helpers_live_under_dsp() {
    use bisque::dsp::oversample::PolyphaseUpsampler;
    use bisque::dsp::{db_to_linear, linear_to_db_floor, math, SmootherBank};
    // The kernel wrapper and processing spec are part of the processor contract.
    use bisque::processor::{KernelProcessor, ProcessSpec};

    struct DummyKernel;

    let _ = math::sin(0.0);
    let _ = db_to_linear(0.0);
    let _ = linear_to_db_floor(1.0, -120.0);
    let _ = core::mem::size_of::<KernelProcessor<DummyKernel>>();
    let _ = core::mem::size_of::<PolyphaseUpsampler>();
    let _ = core::mem::size_of::<SmootherBank>();

    let spec = ProcessSpec {
        sample_rate: 48_000,
        channels: 2,
        max_block: 512,
        max_memory: None,
    };
    let _ = SmootherBank::try_new(&[], &spec).expect("empty smoother bank");

    let mut oversampler = PolyphaseUpsampler::new(4, 12);
    let _ = oversampler.latency();
    let _ = oversampler.tail_frames();
    let _ = oversampler.drain_peak();
}

#[cfg(feature = "test-support")]
#[test]
fn testing_registry_is_supported_test_surface() {
    use bisque::testing::registry::{
        meter_entries, processor_entries, variable_rate_entries, BoxedVariableRate, DriveMode,
        MeterEntry, ProcessorAuthoring, ProcessorEntry, VariableRateEntry,
    };

    let _: fn() -> Vec<ProcessorEntry> = processor_entries;
    let _: fn() -> Vec<MeterEntry> = meter_entries;
    let _: fn() -> Vec<VariableRateEntry> = variable_rate_entries;
    let _ = core::mem::size_of::<BoxedVariableRate>();
    let _ = core::mem::size_of::<DriveMode>();
    let _ = core::mem::size_of::<ProcessorAuthoring>();
}

#[cfg(feature = "analysis")]
#[test]
fn analysis_imports_are_at_the_domain_root() {
    use bisque::analysis::{
        ClipMeter, ClipMeterSettings, LoudnessMeter, LoudnessMeterSettings, LoudnessReading,
        MeanMeter, PeakMeter, RmsMeter, TruePeakMeter, WindowedRmsMeter, WindowedRmsMeterSettings,
        DEFAULT_MAX_INTEGRATED_SECONDS,
    };

    let _ = DEFAULT_MAX_INTEGRATED_SECONDS;
    let _ = core::mem::size_of::<ClipMeter>();
    let _ = core::mem::size_of::<ClipMeterSettings>();
    let _ = core::mem::size_of::<LoudnessMeter>();
    let _ = core::mem::size_of::<LoudnessMeterSettings>();
    let _ = core::mem::size_of::<LoudnessReading>();
    let _ = core::mem::size_of::<MeanMeter>();
    let _ = core::mem::size_of::<PeakMeter>();
    let _ = core::mem::size_of::<RmsMeter>();
    let _ = core::mem::size_of::<TruePeakMeter>();
    let _ = core::mem::size_of::<WindowedRmsMeter>();
    let _ = core::mem::size_of::<WindowedRmsMeterSettings>();
}

#[cfg(feature = "dynamics")]
#[test]
fn dynamics_imports_are_at_the_domain_root() {
    use bisque::dynamics::{
        Compressor, CompressorParams, CompressorSettings, Expander, ExpanderParams,
        ExpanderSettings, Gate, GateParams, GateSettings,
    };

    let _ = core::mem::size_of::<Compressor>();
    let _ = core::mem::size_of::<CompressorParams>();
    let _ = core::mem::size_of::<CompressorSettings>();
    let _ = core::mem::size_of::<Expander>();
    let _ = core::mem::size_of::<ExpanderParams>();
    let _ = core::mem::size_of::<ExpanderSettings>();
    let _ = core::mem::size_of::<Gate>();
    let _ = core::mem::size_of::<GateParams>();
    let _ = core::mem::size_of::<GateSettings>();
    let _ = Compressor::THRESHOLD_DB;
    let _ = Compressor::RATIO;
    let _ = Compressor::MAKEUP_DB;
    let _ = Expander::THRESHOLD_DB;
    let _ = Expander::RATIO;
    let _ = Expander::with_sidechain();
    let _ = Gate::THRESHOLD_DB;
    let _ = Gate::RATIO;
    let _ = Gate::RANGE_DB;
    let _ = Gate::with_sidechain();
}

#[cfg(feature = "filters")]
#[test]
fn filters_imports_are_at_the_domain_root() {
    use bisque::filters::{
        Biquad, BiquadCoeffs, BiquadKind, BiquadParams, BiquadSettings, MovingAverage,
    };

    let _ = core::mem::size_of::<Biquad>();
    let _ = core::mem::size_of::<BiquadCoeffs>();
    let _ = core::mem::size_of::<BiquadKind>();
    let _ = core::mem::size_of::<BiquadParams>();
    let _ = core::mem::size_of::<BiquadSettings>();
    let _ = core::mem::size_of::<MovingAverage>();
    let _ = Biquad::CUTOFF_HZ;
    let _ = Biquad::Q;
    let _ = Biquad::GAIN_DB;
    let _ = BiquadCoeffs::try_rbj(BiquadKind::Lowpass, 48_000.0, 1_000.0, 0.707, 0.0);
    let _ = MovingAverage::new(16).group_delay_frames();
}

#[cfg(feature = "generators")]
#[test]
fn generators_imports_are_at_the_domain_root() {
    use bisque::generators::{
        PolyBlepOsc, PolyBlepOscParams, PolyBlepOscSettings, SineOsc, SineOscParams,
        SineOscSettings, Waveform, WhiteNoise, WhiteNoiseParams, WhiteNoiseSettings,
    };

    let _ = core::mem::size_of::<PolyBlepOsc>();
    let _ = core::mem::size_of::<PolyBlepOscParams>();
    let _ = core::mem::size_of::<PolyBlepOscSettings>();
    let _ = core::mem::size_of::<SineOsc>();
    let _ = core::mem::size_of::<SineOscParams>();
    let _ = core::mem::size_of::<SineOscSettings>();
    let _ = core::mem::size_of::<Waveform>();
    let _ = core::mem::size_of::<WhiteNoise>();
    let _ = core::mem::size_of::<WhiteNoiseParams>();
    let _ = core::mem::size_of::<WhiteNoiseSettings>();
    let _ = SineOsc::FREQUENCY_HZ;
    let _ = SineOsc::AMPLITUDE;
    let _ = PolyBlepOsc::FREQUENCY_HZ;
    let _ = PolyBlepOsc::AMPLITUDE;
    let _ = WhiteNoise::AMPLITUDE;
}

#[cfg(feature = "mastering")]
#[test]
fn mastering_imports_are_at_the_domain_root() {
    use bisque::mastering::{
        Dither, DitherSettings, Gain, GainParams, GainSettings, Limiter, LimiterParams,
        LimiterSettings, Scale,
    };

    let _ = core::mem::size_of::<Dither>();
    let _ = core::mem::size_of::<DitherSettings>();
    let _ = core::mem::size_of::<Gain>();
    let _ = core::mem::size_of::<GainParams>();
    let _ = core::mem::size_of::<GainSettings>();
    let _ = core::mem::size_of::<Scale>();
    let _ = core::mem::size_of::<Limiter>();
    let _ = core::mem::size_of::<LimiterParams>();
    let _ = core::mem::size_of::<LimiterSettings>();
    let _ = Gain::GAIN_DB;
    let _ = Limiter::THRESHOLD_DB;
}

#[cfg(feature = "repair")]
#[test]
fn repair_imports_are_at_the_domain_root() {
    use bisque::repair::{DcBlocker, DcBlockerParams, DcBlockerSettings, DcOffset};

    let _ = core::mem::size_of::<DcBlocker>();
    let _ = core::mem::size_of::<DcBlockerParams>();
    let _ = core::mem::size_of::<DcBlockerSettings>();
    let _ = core::mem::size_of::<DcOffset>();
    let _ = DcBlocker::CUTOFF_HZ;
    let _ = DcOffset::broadcast(0.1);
    let _ = DcOffset::per_channel(vec![0.1, -0.1]);
    let _ = DcOffset::per_channel_from_slice(&[0.1, -0.1]);
}

/// Every built-in processor erases to the host seam
/// `Box<dyn Processor<f32> + Send>`.
#[cfg(all(feature = "mastering", feature = "time", feature = "generators"))]
#[test]
fn built_in_processors_erase_to_the_send_processor_seam() {
    use bisque::generators::PolyBlepOsc;
    use bisque::mastering::{Dither, Gain};
    use bisque::processor::KernelProcessor;
    use bisque::processor::Processor;
    use bisque::time::Delay;

    fn assert_send(_p: Box<dyn Processor<f32> + Send>) {}
    assert_send(Box::new(KernelProcessor::new(Gain::new())));
    assert_send(Box::new(KernelProcessor::new(Dither::new())));
    assert_send(Box::new(KernelProcessor::new(Delay::new())));
    assert_send(Box::new(KernelProcessor::new(PolyBlepOsc::new())));
}

#[cfg(feature = "mastering")]
#[test]
fn kernel_processor_can_be_typed_for_f64() {
    use bisque::mastering::Gain;
    use bisque::processor::{Kernel, KernelProcessor, ProcessContext, ProcessSpec, Processor};

    let spec = ProcessSpec {
        sample_rate: 48_000,
        channels: 1,
        max_block: 2,
        max_memory: None,
    };

    let mut gain: KernelProcessor<Gain, f64> = KernelProcessor::with_sample_type(Gain::new());
    gain.prepare(spec).expect("prepare f64 processor");
    gain.set_parameter_immediate(Gain::GAIN_DB, -6.0)
        .expect("known parameter id");

    let mut samples = [1.0_f64, -1.0];
    {
        let mut planes: [&mut [f64]; 1] = [&mut samples];
        let mut ctx = ProcessContext::in_place(&mut planes, 0);
        gain.process(&mut ctx);
    }
    assert!(
        samples[0] > 0.49 && samples[0] < 0.51,
        "f64 gain output {}",
        samples[0]
    );
    assert!(
        samples[1] < -0.49 && samples[1] > -0.51,
        "f64 gain output {}",
        samples[1]
    );

    let mut via_trait = <Gain as Kernel<f64>>::into_processor(Gain::new());
    via_trait.prepare(spec).expect("prepare f64 via trait");
}

#[cfg(feature = "spectral")]
#[test]
fn spectral_imports_are_at_the_domain_root() {
    use bisque::spectral::{Complex, Fft, SpectralFilter, SpectralFilterSettings, Stft, Window};

    let _ = core::mem::size_of::<Complex<f32>>();
    let _ = core::mem::size_of::<Fft>();
    let _ = core::mem::size_of::<SpectralFilter>();
    let _ = core::mem::size_of::<SpectralFilterSettings>();
    let _ = core::mem::size_of::<Stft>();
    let _ = core::mem::size_of::<Window>();
    let filter = SpectralFilter::low_pass(255, 85, 8_000.0);
    assert_eq!(filter.size(), 255);
    assert_eq!(filter.hop(), 85);

    let configured = SpectralFilter::with_settings(
        SpectralFilterSettings::new()
            .size(256)
            .hop(256)
            .window(Window::Rectangular)
            .band(200.0, 8_000.0),
    );
    assert_eq!(configured.size(), 256);
    assert_eq!(configured.hop(), 256);
}

#[cfg(feature = "time")]
#[test]
fn time_imports_are_at_the_domain_root() {
    use bisque::time::{Delay, DelayParams, DelaySettings, TimeStretch, TimeStretchSettings};

    let _ = core::mem::size_of::<Delay>();
    let _ = core::mem::size_of::<DelayParams>();
    let _ = core::mem::size_of::<DelaySettings>();
    let _ = core::mem::size_of::<TimeStretch>();
    let _ = core::mem::size_of::<TimeStretchSettings>();
    let _ = Delay::DELAY_MS;
    let _ = Delay::FEEDBACK;
    let _ = Delay::MIX;
}
