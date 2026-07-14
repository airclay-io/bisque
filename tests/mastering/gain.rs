// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Airclay LLC

//! Contract tests for `Gain`.
//!
//! Covers gain identity, settled gain factor, parameter clamping, unknown
//! parameters, block-size invariance, and reset behavior.

use bisque::mastering::{Gain, GainSettings};
use bisque::parameter::ParamId;
use bisque::processor::{Kernel, KernelProcessor};
use bisque::testing::{bits_eq, ev, sine, Contract};

const GAIN: ParamId = Gain::GAIN_DB;

/// A fresh, unprepared gain processor.
fn gain() -> KernelProcessor<Gain> {
    KernelProcessor::new(Gain::new())
}

/// Audio behavior checks.
mod audio {
    use super::*;

    #[test]
    fn zero_db_is_identity() {
        // 0 dB is unity, so output is bit-identical to input.
        let input = sine(2, 777);
        let out = Contract::default().run(gain, &input, &[], 64);
        assert!(bits_eq(&out, &input), "0 dB must be a bit-exact no-op");
    }

    #[test]
    fn settled_gain_matches_independent_factor() {
        // After the ramp settles, output equals input * 10^(-6/20).
        let frames = 2000;
        let input = sine(2, frames);
        let out = Contract::default().run(gain, &input, &[ev(0, GAIN, -6.0)], 64);
        let expected = 10f64.powf(-6.0 / 20.0); // independent of the impl's exp() path
        for ch in 0..input.len() {
            for i in 1000..frames {
                let want = (f64::from(input[ch][i]) * expected) as f32;
                let got = out[ch][i];
                let tol = 1e-5 * want.abs().max(1e-6);
                assert!(
                    (got - want).abs() <= tol,
                    "ch{ch}[{i}]: got {got}, want ~{want}"
                );
            }
        }
    }

    #[test]
    fn out_of_range_event_is_clamped() {
        // +100 dB is above the +24 dB ceiling and clamps to +24 dB.
        let frames = 2000;
        let input = sine(2, frames);
        let out = Contract::default().run(gain, &input, &[ev(0, GAIN, 100.0)], 64);
        let expected = 10f64.powf(24.0 / 20.0);
        for ch in 0..input.len() {
            for i in 1500..frames {
                let want = (f64::from(input[ch][i]) * expected) as f32;
                let tol = 1e-4 * want.abs().max(1e-6);
                assert!((out[ch][i] - want).abs() <= tol, "clamp to +24 dB");
            }
        }
    }

    #[test]
    fn unknown_param_event_is_a_no_op() {
        // An event for an undeclared parameter is ignored.
        let input = sine(2, 500);
        let out = Contract::default().run(gain, &input, &[ev(0, ParamId(99), -24.0)], 64);
        assert!(bits_eq(&out, &input), "unknown param id must be a no-op");
    }

    #[test]
    fn swept_gain_over_full_scale_sine_is_click_free() {
        // Sweep gain from -96 dB to +24 dB over a steady full-scale sine and
        // assert that no sample-to-sample jump exceeds the analytic worst case
        // of the declared smoothing.
        //
        // Bound derivation. The output is out[n] = g[n] * x[n] with g held
        // constant within each 32-frame control-rate sub-block, so
        //
        //   |out[n] - out[n-1]| <= g_max * |x[n] - x[n-1]|
        //                        + |g[n] - g[n-1]| * |x[n-1]|.
        //
        // Signal term: x is a full-scale (A = 1) sine at f = 200 Hz, so
        // |x[n] - x[n-1]| <= A * w with w = 2*pi*f/fs, and g <= g_max =
        // 10^(24/20).
        //
        // Gain-step term: each 2 dB target change ramps linearly in dB over
        // 5 ms at the 32-frame control rate. There are 7.5 smoothing steps,
        // so each control update moves by 2 / 7.5 dB and
        // |g[n] - g[n-1]| <= g_max * (1 - 10^(-(2/7.5)/20)).
        //
        // Both terms can coincide at a boundary frame, so the worst case is
        // their sum; the assertion allows a 2x margin.
        let fs = 48_000.0;
        let f = 200.0;
        let amp = 1.0f64;
        let w = 2.0 * std::f64::consts::PI * f / fs;
        let db_step = 2.0;
        let smoothing_steps = (5.0e-3 * fs) / 32.0;
        let ramp_step_db = db_step / smoothing_steps;
        let g_max = 10f64.powf(24.0 / 20.0);
        let signal_term = g_max * amp * w;
        let gain_step_term = g_max * (1.0 - 10f64.powf(-ramp_step_db / 20.0)) * amp;
        let bound = 2.0 * (signal_term + gain_step_term);

        // Start at -96 dB, then allow each 2 dB target change eight control
        // updates to settle before the next one.
        let target_steps = (120.0 / db_step) as u32;
        let updates_per_target = smoothing_steps.ceil() as u32;
        let events: Vec<_> = (0..=target_steps)
            .map(|k| {
                ev(
                    32 * updates_per_target * k,
                    GAIN,
                    -96.0 + db_step * f64::from(k),
                )
            })
            .collect();
        let frames = 32 * updates_per_target as usize * (target_steps as usize + 1) + 128;
        let input: Vec<Vec<f32>> = vec![
            (0..frames)
                .map(|i| bisque::dsp::math::sin(i as f64 * w) as f32)
                .collect();
            2
        ];
        let out = Contract::default().run(
            || KernelProcessor::new(Gain::with_settings(GainSettings::new().gain_db(-96.0))),
            &input,
            &events,
            128,
        );

        let mut max_jump = 0.0f64;
        for ch in &out {
            for pair in ch.windows(2) {
                max_jump = max_jump.max((f64::from(pair[1]) - f64::from(pair[0])).abs());
            }
        }
        assert!(
            max_jump <= bound,
            "swept gain clicked: max sample-to-sample jump {max_jump} exceeds the analytic \
             worst case with margin ({bound})"
        );
        // Sanity: the sweep actually reached high gain (peak well above 0 dBFS).
        let peak = out
            .iter()
            .flatten()
            .fold(0.0f32, |acc, &s| acc.max(s.abs()));
        assert!(
            peak > 10.0,
            "sweep peak {peak} did not reach the high-gain range"
        );
    }
}

#[test]
fn block_size_invariance_is_bit_exact() {
    // Events include a duplicate offset and a control-rate grid boundary.
    let input = sine(2, 1000);
    let events = [
        ev(0, GAIN, -6.0),
        ev(0, GAIN, -3.0),
        ev(64, GAIN, -12.0),
        ev(300, GAIN, 0.0),
        ev(513, GAIN, -18.0),
    ];
    Contract::default().assert_block_size_invariant(gain, &input, &events);
}

#[test]
fn reset_equivalence_no_state_leak() {
    // After reset, the same input reproduces a fresh instance's output exactly.
    let input = sine(2, 800);
    let events = [ev(0, GAIN, -9.0), ev(200, GAIN, -3.0)];
    Contract::default().assert_reset_equivalence(gain, &input, &events);
}

#[test]
fn settings_define_metadata_and_the_first_frame() {
    let settings = GainSettings::new().gain_db(-6.0);
    let kernel = Gain::with_settings(settings);
    assert_eq!(Kernel::<f32>::param_info(&kernel)[0].default, -6.0);

    let input = vec![vec![1.0f32; 8]; 2];
    let out = Contract::default().run(
        || KernelProcessor::new(Gain::with_settings(settings)),
        &input,
        &[],
        8,
    );
    let expected = bisque::dsp::db_to_linear(-6.0) as f32;
    assert_eq!(out[0][0], expected);
}

#[test]
fn default_constructor_matches_default_settings() {
    let direct = Gain::new();
    let configured = Gain::with_settings(GainSettings::default());
    assert_eq!(
        Kernel::<f32>::param_info(&direct)[0].default,
        Kernel::<f32>::param_info(&configured)[0].default
    );
}
