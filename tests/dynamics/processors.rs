// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Airclay LLC

//! Contract tests for dynamics processors.
//!
//! Covers compressor, expander, and gate curves, transparency regions, sidechain
//! routing, block-size invariance, reset behavior, validation, and declared
//! metadata.

use std::f64::consts::LN_10;

use bisque::dynamics::{
    Compressor, CompressorSettings, Expander, ExpanderSettings, Gate, GateSettings,
};
use bisque::processor::KernelProcessor;
use bisque::processor::{DspError, Kernel, ProcessSpec, Processor};
use bisque::testing::{bits_eq, ev, Buffers, Contract};

const THRESHOLD: bisque::parameter::ParamId = Compressor::THRESHOLD_DB;
const RATIO: bisque::parameter::ParamId = Compressor::RATIO;

/// A fresh main-detecting compressor at the defaults (-20 dB, 4:1, no makeup).
fn comp() -> KernelProcessor<Compressor> {
    KernelProcessor::new(Compressor::new())
}

/// A fresh expander at the defaults (-40 dB, 2:1).
fn expander() -> KernelProcessor<Expander> {
    KernelProcessor::new(Expander::new())
}

/// A fresh gate at the defaults (-40 dB, -60 dB range).
fn gate() -> KernelProcessor<Gate> {
    KernelProcessor::new(Gate::new())
}

/// Convert dB to linear amplitude.
fn db_to_lin(db: f64) -> f64 {
    (db * (LN_10 / 20.0)).exp()
}

/// An equal-amplitude stereo tone (both channels the same amplitude `amp`).
fn tone(frames: usize, amp: f32) -> Buffers {
    let ch: Vec<f32> = (0..frames).map(|i| amp * (i as f32 * 0.05).sin()).collect();
    vec![ch.clone(), ch]
}

/// The peak (max |.|) over the settled second half of channel 0.
fn settled_peak(out: &[Vec<f32>]) -> f64 {
    let n = out[0].len();
    out[0][n / 2..]
        .iter()
        .fold(0.0f64, |m, &x| m.max(f64::from(x).abs()))
}

mod audio {
    use super::*;

    #[test]
    fn static_curve_compresses_to_the_ratio() {
        // Near-instant attack and long release isolate the static compression
        // curve. A -10 dBFS sine into a -20 dBFS, 4:1 compressor produces a
        // -17.5 dBFS settled peak.
        let amp = db_to_lin(-10.0) as f32;
        let peak_hold = || {
            KernelProcessor::new(Compressor::with_settings(
                CompressorSettings::new()
                    .threshold_db(-20.0)
                    .ratio(4.0)
                    .attack_ms(0.001)
                    .release_ms(10_000.0)
                    .makeup_db(0.0)
                    .use_sidechain(false),
            ))
        };
        let out = Contract::default().run(peak_hold, &tone(8000, amp), &[], 64);
        let peak = settled_peak(&out);
        let expected = db_to_lin(-17.5);
        assert!(
            (peak - expected).abs() < 0.005,
            "settled peak {peak} should match the static curve {expected}"
        );
    }

    #[test]
    fn zero_times_apply_detector_changes_immediately() {
        let amp = db_to_lin(-10.0) as f32;
        let quiet = db_to_lin(-40.0) as f32;
        let make = || {
            KernelProcessor::new(Compressor::with_settings(
                CompressorSettings::new()
                    .threshold_db(-20.0)
                    .ratio(4.0)
                    .attack_ms(0.0)
                    .release_ms(0.0),
            ))
        };
        let input = vec![vec![amp, quiet], vec![amp, quiet]];
        let out = Contract::default().run(make, &input, &[], 2);
        let expected = db_to_lin(-17.5) as f32;
        assert!(
            (out[0][0] - expected).abs() < 1e-6,
            "zero attack must apply the settled curve immediately"
        );
        assert_eq!(
            out[0][1], quiet,
            "zero release must stop gain reduction immediately"
        );
    }

    #[test]
    fn below_threshold_is_transparent() {
        // Below the threshold with no makeup, gain is 1.0.
        let input = tone(2000, db_to_lin(-30.0) as f32);
        let out = Contract::default().run(comp, &input, &[], 64);
        assert!(
            bits_eq(&out, &input),
            "below threshold must pass through unchanged"
        );
    }

    #[test]
    fn sidechain_ducks_the_main() {
        // A loud sidechain drives gain reduction on a quiet main signal.
        let main = tone(8000, db_to_lin(-20.0) as f32);
        let side = tone(8000, db_to_lin(-6.0) as f32);
        let ducked = settled_peak(&Contract::default().run_with_sidechain(
            || KernelProcessor::new(Compressor::with_sidechain()),
            &main,
            &side,
            &[],
            64,
        ));
        // The sidechain is 14 dB over threshold, producing 10.5 dB of reduction.
        let expected = db_to_lin(-20.0) * db_to_lin(-10.5);
        assert!(
            (ducked - expected).abs() < 0.005,
            "sidechain duck {ducked} should match {expected}"
        );
        // Self-detection at the threshold produces much less reduction.
        let self_detected = settled_peak(&Contract::default().run(comp, &main, &[], 64));
        assert!(
            ducked < 0.6 * self_detected,
            "sidechain ({ducked}) must duck below self-detection ({self_detected})"
        );
    }

    #[test]
    fn non_finite_main_and_sidechain_samples_are_silence() {
        let c = Contract::default();
        let mut bad_main = tone(512, db_to_lin(-6.0) as f32);
        let mut clean_main = bad_main.clone();
        bad_main[0][64] = f32::NAN;
        bad_main[1][257] = f32::INFINITY;
        clean_main[0][64] = 0.0;
        clean_main[1][257] = 0.0;
        let bad_out = c.run(comp, &bad_main, &[], 64);
        let clean_out = c.run(comp, &clean_main, &[], 64);
        assert!(bits_eq(&bad_out, &clean_out));

        let main = tone(512, db_to_lin(-20.0) as f32);
        let mut bad_side = tone(512, db_to_lin(-6.0) as f32);
        let mut clean_side = bad_side.clone();
        bad_side[0][96] = f32::NAN;
        bad_side[1][96] = f32::INFINITY;
        clean_side[0][96] = 0.0;
        clean_side[1][96] = 0.0;
        let make = || KernelProcessor::new(Compressor::with_sidechain());
        let bad_out = c.run_with_sidechain(make, &main, &bad_side, &[], 64);
        let clean_out = c.run_with_sidechain(make, &main, &clean_side, &[], 64);
        assert!(bits_eq(&bad_out, &clean_out));
    }
}

#[test]
fn block_size_invariance_is_bit_exact() {
    // Active compression with threshold and ratio events is split-invariant.
    let input = tone(2000, db_to_lin(-6.0) as f32);
    let events = [ev(0, THRESHOLD, -18.0), ev(500, RATIO, 8.0)];
    Contract::default().assert_block_size_invariant(comp, &input, &events);
}

#[test]
fn sidechain_path_is_block_size_invariant() {
    let main = tone(2000, db_to_lin(-15.0) as f32);
    let side = tone(2000, db_to_lin(-6.0) as f32);
    let make = || KernelProcessor::new(Compressor::with_sidechain());
    let c = Contract::default();
    let reference = c.run_with_sidechain(make, &main, &side, &[], 2000);
    for &block in &[1usize, 7, 64, 128] {
        let out = c.run_with_sidechain(make, &main, &side, &[], block);
        assert!(
            bits_eq(&out, &reference),
            "sidechain diverged at block {block}"
        );
    }
}

#[test]
fn reset_equivalence_no_state_leak() {
    let input = tone(2000, db_to_lin(-6.0) as f32);
    let events = [ev(0, THRESHOLD, -24.0)];
    Contract::default().assert_reset_equivalence(comp, &input, &events);
}

mod validation {
    use super::*;

    fn assert_invalid<K: Kernel<f32>>(kernel: K) {
        let mut processor = KernelProcessor::new(kernel);
        assert!(matches!(
            Processor::<f32>::prepare(&mut processor, Contract::default().spec),
            Err(DspError::InvalidParam(_))
        ));
    }

    #[test]
    fn zero_sample_rate_is_rejected() {
        let mut p = KernelProcessor::new(Compressor::new());
        let bad = ProcessSpec {
            sample_rate: 0,
            ..Contract::default().spec
        };
        assert!(matches!(
            Processor::<f32>::prepare(&mut p, bad),
            Err(DspError::UnsupportedSpec(_))
        ));
    }

    #[test]
    fn invalid_attack_and_release_times_are_rejected() {
        for bad in [-1.0, f64::NAN, f64::INFINITY] {
            assert_invalid(Compressor::with_settings(
                CompressorSettings::new().attack_ms(bad),
            ));
            assert_invalid(Compressor::with_settings(
                CompressorSettings::new().release_ms(bad),
            ));
            assert_invalid(Expander::with_settings(
                ExpanderSettings::new().attack_ms(bad),
            ));
            assert_invalid(Expander::with_settings(
                ExpanderSettings::new().release_ms(bad),
            ));
            assert_invalid(Gate::with_settings(GateSettings::new().attack_ms(bad)));
            assert_invalid(Gate::with_settings(GateSettings::new().release_ms(bad)));
        }
    }

    #[test]
    fn invalid_startup_controls_are_rejected() {
        for bad in [1.0, f64::NAN] {
            assert_invalid(Compressor::with_settings(
                CompressorSettings::new().threshold_db(bad),
            ));
            assert_invalid(Expander::with_settings(
                ExpanderSettings::new().threshold_db(bad),
            ));
            assert_invalid(Gate::with_settings(GateSettings::new().threshold_db(bad)));
        }
        for bad in [0.0, f64::NAN] {
            assert_invalid(Compressor::with_settings(
                CompressorSettings::new().ratio(bad),
            ));
            assert_invalid(Expander::with_settings(ExpanderSettings::new().ratio(bad)));
            assert_invalid(Gate::with_settings(GateSettings::new().ratio(bad)));
        }
        for bad in [-1.0, f64::NAN] {
            assert_invalid(Compressor::with_settings(
                CompressorSettings::new().makeup_db(bad),
            ));
        }
        for bad in [1.0, f64::NAN] {
            assert_invalid(Gate::with_settings(GateSettings::new().range_db(bad)));
        }
    }
}

// --- Expander ---

mod expander_audio {
    use super::*;

    #[test]
    fn static_curve_expands_below_threshold() {
        // A -30 dBFS sine into a -20 dBFS, 2:1 expander settles at -40 dBFS.
        let amp = db_to_lin(-30.0) as f32;
        let make = || {
            KernelProcessor::new(Expander::with_settings(
                ExpanderSettings::new()
                    .threshold_db(-20.0)
                    .ratio(2.0)
                    .attack_ms(0.001)
                    .release_ms(10_000.0)
                    .use_sidechain(false),
            ))
        };
        let out = Contract::default().run(make, &tone(8000, amp), &[], 64);
        let peak = settled_peak(&out);
        let expected = db_to_lin(-40.0);
        assert!(
            (peak - expected).abs() < 0.001,
            "expander settled peak {peak} should match {expected}"
        );
    }

    #[test]
    fn passes_above_threshold() {
        // Above the default threshold, gain settles to 1.0.
        let amp = db_to_lin(-10.0) as f32;
        let out = Contract::default().run(expander, &tone(4000, amp), &[], 64);
        assert!(
            (settled_peak(&out) - f64::from(amp)).abs() < 1e-4,
            "above threshold the expander is transparent"
        );
    }

    #[test]
    fn a_loud_sidechain_opens_the_expander() {
        let amp = db_to_lin(-60.0) as f32;
        let main = tone(8_000, amp);
        let side = tone(8_000, db_to_lin(-10.0) as f32);
        let c = Contract::default();
        let keyed = settled_peak(&c.run_with_sidechain(
            || KernelProcessor::new(Expander::with_sidechain()),
            &main,
            &side,
            &[],
            64,
        ));
        let self_detected = settled_peak(&c.run(expander, &main, &[], 64));
        assert!((keyed - f64::from(amp)).abs() < 1e-5);
        assert!(keyed > self_detected * 5.0);
    }
}

#[test]
fn expander_block_size_invariant() {
    let input = tone(2000, db_to_lin(-30.0) as f32);
    Contract::default().assert_block_size_invariant(expander, &input, &[]);
}

// --- Gate ---

mod gate_audio {
    use super::*;

    #[test]
    fn opens_above_threshold() {
        // Above the threshold, the gate is open and settled output equals input.
        let amp = db_to_lin(-10.0) as f32;
        let out = Contract::default().run(gate, &tone(4000, amp), &[], 64);
        assert!(
            (settled_peak(&out) - f64::from(amp)).abs() < 1e-4,
            "an open gate is transparent"
        );
    }

    #[test]
    fn closes_below_threshold() {
        // Below the threshold, the gate attenuates the signal.
        let amp = db_to_lin(-55.0) as f32;
        let out = Contract::default().run(gate, &tone(8000, amp), &[], 64);
        let peak = settled_peak(&out);
        assert!(
            peak < f64::from(amp) * db_to_lin(-20.0),
            "a closed gate heavily attenuates (peak {peak} for input {amp})"
        );
    }

    #[test]
    fn a_loud_sidechain_opens_the_gate() {
        let amp = db_to_lin(-60.0) as f32;
        let main = tone(8_000, amp);
        let side = tone(8_000, db_to_lin(-10.0) as f32);
        let c = Contract::default();
        let keyed = settled_peak(&c.run_with_sidechain(
            || KernelProcessor::new(Gate::with_sidechain()),
            &main,
            &side,
            &[],
            64,
        ));
        let self_detected = settled_peak(&c.run(gate, &main, &[], 64));
        assert!((keyed - f64::from(amp)).abs() < 1e-5);
        assert!(keyed > self_detected * 100.0);
    }
}

#[test]
fn gate_block_size_invariant() {
    let input = tone(2000, db_to_lin(-50.0) as f32);
    Contract::default().assert_block_size_invariant(gate, &input, &[]);
}

#[test]
fn expander_and_gate_reset_equivalence() {
    let input = tone(2000, db_to_lin(-30.0) as f32);
    Contract::default().assert_reset_equivalence(expander, &input, &[]);
    Contract::default().assert_reset_equivalence(gate, &input, &[]);
}

mod gate_audio_static {
    use super::*;

    #[test]
    fn static_curve_attenuates_to_the_ratio() {
        // Near-instant attack and long release isolate the static gate curve.
        // With the range floor below the expected output, a -30 dBFS sine into a
        // -20 dBFS, 2:1 gate settles at -40 dBFS.
        let amp = db_to_lin(-30.0) as f32;
        let make = || {
            KernelProcessor::new(Gate::with_settings(
                GateSettings::new()
                    .threshold_db(-20.0)
                    .ratio(2.0)
                    .range_db(-120.0)
                    .attack_ms(0.001)
                    .release_ms(10_000.0)
                    .use_sidechain(false),
            ))
        };
        let out = Contract::default().run(make, &tone(8000, amp), &[], 64);
        let peak = settled_peak(&out);
        let expected = db_to_lin(-40.0);
        assert!(
            (peak - expected).abs() < 0.001,
            "gate settled peak {peak} should match the static curve {expected}"
        );
    }
}

// Declared metadata checks.

mod contract {
    use super::*;

    /// Assert that the kernel allocates nothing in `prepare`: the envelope and
    /// the per-run gain scratch are inline fixed-size state, so the kernel's
    /// own footprint is zero (the `KernelProcessor` wrapper adds the smoother bank).
    fn assert_footprint_exact<K: Kernel<f32>>(mut k: K) {
        let spec = Contract::default().spec;
        Kernel::<f32>::prepare(&mut k, spec).expect("prepare");
        assert_eq!(
            Kernel::<f32>::memory_footprint(&k),
            0,
            "dynamics kernels hold only scalar and inline fixed-size state"
        );
    }

    #[test]
    fn footprints_are_zero_heap_bytes() {
        assert_footprint_exact(Compressor::new());
        assert_footprint_exact(Expander::new());
        assert_footprint_exact(Gate::new());
    }

    #[test]
    fn sidechain_inputs_track_the_detection_source() {
        // Sidechain-keyed processors declare one key bus. Self-detecting
        // processors declare none.
        assert_eq!(
            Kernel::<f32>::sidechain_inputs(&Compressor::with_sidechain()),
            1
        );
        assert_eq!(Kernel::<f32>::sidechain_inputs(&Compressor::new()), 0);

        let exp_keyed = Expander::with_settings(ExpanderSettings::new().use_sidechain(true));
        assert_eq!(Kernel::<f32>::sidechain_inputs(&exp_keyed), 1);
        assert_eq!(Kernel::<f32>::sidechain_inputs(&Expander::new()), 0);
        assert_eq!(
            Kernel::<f32>::sidechain_inputs(&Expander::with_sidechain()),
            1
        );

        let gate_keyed = Gate::with_settings(GateSettings::new().use_sidechain(true));
        assert_eq!(Kernel::<f32>::sidechain_inputs(&gate_keyed), 1);
        assert_eq!(Kernel::<f32>::sidechain_inputs(&Gate::new()), 0);
        assert_eq!(Kernel::<f32>::sidechain_inputs(&Gate::with_sidechain()), 1);
    }

    #[test]
    fn declared_param_ranges_keep_their_sign() {
        // Threshold and range bounds are negative dB ranges.
        let exp = Expander::new();
        assert_eq!(
            Kernel::<f32>::param_info(&exp)[0].range,
            (-80.0, 0.0),
            "expander threshold range"
        );

        let gate = Gate::new();
        let params = Kernel::<f32>::param_info(&gate);
        assert_eq!(params[0].range, (-80.0, 0.0), "gate threshold range");
        assert_eq!(params[2].range, (-120.0, 0.0), "gate range control");
    }
}
