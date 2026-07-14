// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Airclay LLC

//! Author a same-rate effect with typed parameters and run contract support.

use bisque::parameter::{ParamId, ParamInfo, Unit};
use bisque::processor::{DspError, Kernel, KernelProcessor, ProcessSpec, Sample, SubBlock};
use bisque::testing::Contract;

bisque::params! {
    /// Smoothed values passed to [`Trim::render`].
    pub struct TrimParams {
        /// Linear gain.
        pub gain => GAIN,
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
#[non_exhaustive]
struct TrimSettings {
    gain: f64,
}

impl TrimSettings {
    const fn new() -> Self {
        Self { gain: 1.0 }
    }

    const fn gain(mut self, gain: f64) -> Self {
        self.gain = gain;
        self
    }
}

impl Default for TrimSettings {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Clone, Debug)]
struct Trim {
    params: [ParamInfo; 1],
}

impl Trim {
    const GAIN: ParamId = TrimParams::GAIN;

    fn new() -> Self {
        Self::with_settings(TrimSettings::new())
    }

    fn with_settings(settings: TrimSettings) -> Self {
        Self {
            params: [ParamInfo::new(
                Self::GAIN,
                "gain",
                (0.0, 2.0),
                settings.gain,
                Unit::Linear,
            )],
        }
    }
}

impl<T: Sample> Kernel<T> for Trim {
    type Params = TrimParams;

    fn prepare(&mut self, _spec: ProcessSpec) -> Result<(), DspError> {
        Ok(())
    }

    fn reset(&mut self) {}

    fn param_info(&self) -> &[ParamInfo] {
        &self.params
    }

    fn render(&mut self, io: &mut SubBlock<'_, '_, '_, T>, params: &TrimParams) {
        for channel in 0..io.channels() {
            for sample in io.channel_mut(channel) {
                *sample = T::from_f64(sample.to_f64() * params.gain);
            }
        }
    }
}

fn main() {
    let input = vec![vec![0.25f32; 257]; 2];
    let contract = Contract::default();

    let unity = contract.run(|| KernelProcessor::new(Trim::new()), &input, &[], 64);
    assert_eq!(unity, input, "the default trim must be transparent");

    let half = contract.run(
        || KernelProcessor::new(Trim::with_settings(TrimSettings::new().gain(0.5))),
        &input,
        &[],
        64,
    );
    assert!(
        half.iter().flatten().all(|&sample| sample == 0.125),
        "the configured trim must apply its declared gain"
    );

    contract.assert_block_size_invariant(
        || KernelProcessor::new(Trim::with_settings(TrimSettings::new().gain(0.5))),
        &input,
        &[],
    );
    println!("Trim passed its output check and the shared block-size contract");
}
