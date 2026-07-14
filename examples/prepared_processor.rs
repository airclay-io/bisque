// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Airclay LLC

//! Configure, prepare, and run one processor through the optional host helper.

use bisque::host::PreparedProcessor;
use bisque::mastering::{Gain, GainSettings};
use bisque::processor::ProcessSpec;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let spec = ProcessSpec {
        sample_rate: 48_000,
        channels: 2,
        max_block: 256,
        max_memory: None,
    };
    let gain = Gain::with_settings(GainSettings::new().gain_db(-6.0));
    let mut gain = PreparedProcessor::prepare_kernel(gain, spec)?;

    let mut first_left = [0.25f32; 128];
    let mut first_right = [0.25f32; 128];
    gain.process_in_place(&mut [&mut first_left, &mut first_right], &[]);

    let mut second_left = [0.25f32; 64];
    let mut second_right = [0.25f32; 64];
    gain.process_in_place(&mut [&mut second_left, &mut second_right], &[]);

    println!(
        "processed {} frames; latency={} tail={:?}",
        gain.sample_pos(),
        gain.latency(),
        gain.tail()
    );
    Ok(())
}
