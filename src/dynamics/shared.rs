// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Airclay LLC

//! Shared dynamics helpers.

use std::f64::consts::LN_10;

use crate::dsp::driver::MAX_RUN_FRAMES;
use crate::dsp::math;
use crate::dsp::sanitize::{finite_or_zero, flush_denormal};
use crate::parameter::{ParamId, ParamInfo, Unit};
use crate::processor::{DspError, ProcessSpec, Sample, SubBlock};

/// dB to linear scale factor.
const LN10_OVER_20: f64 = LN_10 / 20.0;

/// Envelope floor used before logarithmic expansion curves.
pub(super) const ENV_FLOOR: f64 = 1e-10;

/// The shared ratio parameter id used by [`ratio_param`]. Dynamics processors
/// declare threshold at id 0 and the ratio at id 1.
pub(super) const CONTROL_A: ParamId = ParamId(1); // ratio

/// Convert dBFS to linear amplitude.
pub(super) fn db_to_lin(db: f64) -> f64 {
    math::exp(db * LN10_OVER_20)
}

/// Shared dynamics engine with linked peak detection, attack/release smoothing
/// on the detected level, optional sidechain input, and linked gain application.
#[derive(Debug, Clone)]
pub(super) struct DynamicsCore {
    attack_ms: f64,
    release_ms: f64,
    pub(super) use_sidechain: bool,
    fs: f64,
    atk: f64, // attack one-pole coefficient
    rel: f64, // release one-pole coefficient
    env: f64, // linked detection envelope (linear)
    // Per-frame gain for the current run. A run is capped at one control-rate
    // cell, so this is inline and fixed-size.
    gain_scratch: [f64; MAX_RUN_FRAMES],
}

impl DynamicsCore {
    pub(super) fn new(attack_ms: f64, release_ms: f64, use_sidechain: bool) -> Self {
        Self {
            attack_ms,
            release_ms,
            use_sidechain,
            fs: 0.0,
            atk: 0.0,
            rel: 0.0,
            env: 0.0,
            gain_scratch: [0.0; MAX_RUN_FRAMES],
        }
    }

    pub(super) fn prepare(&mut self, spec: ProcessSpec) -> Result<(), DspError> {
        if spec.sample_rate == 0 {
            return Err(DspError::UnsupportedSpec("sample rate must be non-zero"));
        }
        if !self.attack_ms.is_finite() || self.attack_ms < 0.0 {
            return Err(DspError::InvalidParam(
                "dynamics attack_ms must be finite and non-negative",
            ));
        }
        if !self.release_ms.is_finite() || self.release_ms < 0.0 {
            return Err(DspError::InvalidParam(
                "dynamics release_ms must be finite and non-negative",
            ));
        }
        self.fs = f64::from(spec.sample_rate);
        self.atk = one_pole_coeff(self.attack_ms, self.fs);
        self.rel = one_pole_coeff(self.release_ms, self.fs);
        self.env = 0.0;
        Ok(())
    }

    pub(super) fn reset(&mut self) {
        self.env = 0.0;
    }

    // No `footprint` helper: all state is scalar or inline fixed-size and
    // `prepare` allocates nothing, so the kernels keep the trait's default
    // zero `memory_footprint`.

    /// Detect level, compute gain with `gain_fn`, and apply it to the main signal.
    pub(super) fn render<T: Sample, F: Fn(f64) -> f64>(
        &mut self,
        io: &mut SubBlock<'_, '_, '_, T>,
        gain_fn: F,
    ) {
        let (atk, rel) = (self.atk, self.rel);
        let nch = io.channels();
        let run = io.frames();
        debug_assert!(run <= MAX_RUN_FRAMES, "run length is within one CR cell");
        let sidechain_channels = if self.use_sidechain && io.sidechain_buses() > 0 {
            io.sidechain_channels(0)
        } else {
            0
        };
        debug_assert!(
            !self.use_sidechain || sidechain_channels > 0,
            "sidechain-enabled dynamics require sidechain bus 0"
        );
        let use_sidechain = self.use_sidechain && sidechain_channels > 0;
        let det_ch = if use_sidechain {
            sidechain_channels
        } else {
            nch
        };

        // Detect, follow, and compute gain per frame.
        for i in 0..run {
            let mut level = 0.0f64;
            for ch in 0..det_ch {
                let s = if use_sidechain {
                    io.sidechain(0, ch)[i]
                } else {
                    io.input(ch)[i]
                };
                level = level.max(finite_or_zero(s.to_f64()).abs());
            }
            // Attack when the level is rising, release when falling. The
            // envelope is recursive state. Flush it so a long silence decays
            // to exactly zero before denormal magnitudes linger.
            let coeff = if level > self.env { atk } else { rel };
            self.env = flush_denormal(self.env + (level - self.env) * coeff);
            self.gain_scratch[i] = gain_fn(self.env);
        }

        // Apply the linked gain to the main signal.
        let gain = &self.gain_scratch;
        for ch in 0..nch {
            for (i, slot) in io.channel_mut(ch).iter_mut().enumerate() {
                *slot = T::from_f64(finite_or_zero(slot.to_f64()) * gain[i]);
            }
        }
    }
}

/// One-pole coefficient for a `ms` time constant at sample rate `fs`.
fn one_pole_coeff(ms: f64, fs: f64) -> f64 {
    if ms == 0.0 {
        1.0
    } else {
        let samples = ms * 1e-3 * fs;
        1.0 - math::exp(-1.0 / samples)
    }
}

/// A `ParamInfo` for a dB-domain control.
pub(super) fn db_param(
    id: ParamId,
    name: &'static str,
    range: (f64, f64),
    default: f64,
) -> ParamInfo {
    ParamInfo::new(id, name, range, default, Unit::Db)
}

/// A `ParamInfo` for the ratio control.
pub(super) fn ratio_param(default: f64) -> ParamInfo {
    ParamInfo::new(CONTROL_A, "ratio", (1.0, 20.0), default, Unit::Linear)
}

#[cfg(test)]
mod tests {
    use super::{one_pole_coeff, DynamicsCore};

    /// `one_pole_coeff` equals `1 - e^(-1/N)` with `N` in samples.
    #[test]
    fn one_pole_coeff_matches_the_analytical_value() {
        // 10 ms at 48 kHz is 480 samples.
        let (ms, fs): (f64, f64) = (10.0, 48_000.0);
        let samples = ms * 1e-3 * fs;
        let expected = 1.0 - (-1.0 / samples).exp(); // test-oracle: independent reference
        let got = one_pole_coeff(ms, fs);
        assert!(
            (got - expected).abs() < 1e-12,
            "one_pole_coeff({ms}, {fs}) = {got}, expected {expected}"
        );
        // A 480-sample pole has a coefficient near 0.00208.
        assert!(
            (0.0020..0.0022).contains(&got),
            "coefficient {got} is not the expected ~0.00208"
        );
    }

    /// Zero is immediate and positive sub-sample time constants retain their
    /// analytical value.
    #[test]
    fn one_pole_coeff_handles_zero_and_sub_sample_times() {
        assert_eq!(one_pole_coeff(0.0, 48_000.0), 1.0);

        // Half a sample is 2 in the recurrence exponent.
        let half_sample_ms = 0.5 * 1_000.0 / 48_000.0;
        let expected = 1.0 - (-2.0f64).exp(); // test-oracle: independent reference
        let got = one_pole_coeff(half_sample_ms, 48_000.0);
        assert!(
            (got - expected).abs() < 1e-12,
            "sub-sample time constant must retain its value: {got} vs {expected}"
        );
    }

    #[test]
    fn envelope_flushes_to_exact_zero_after_long_silence() {
        // The envelope is recursive state. Once the release decays it below
        // the sanitizer floor, it flushes to exactly zero on the realtime path.
        use crate::processor::{AudioBlockMut, Io, ProcessSpec, SubBlock};

        let mut core = DynamicsCore::new(0.0, 0.0, false); // one-sample times
        core.prepare(ProcessSpec {
            sample_rate: 48_000,
            channels: 1,
            max_block: 32,
            max_memory: None,
        })
        .expect("prepare");

        let drive = |value: f32, core: &mut DynamicsCore| {
            let mut buf = [value; 32];
            let mut planes: [&mut [f32]; 1] = [&mut buf];
            let block = AudioBlockMut::new(&mut planes);
            let mut io = Io::InPlace(block);
            let mut sub = SubBlock {
                io: &mut io,
                sc: &[],
                start: 0,
                len: 32,
            };
            core.render(&mut sub, |_| 1.0);
        };

        drive(1.0, &mut core);
        assert!(core.env > 0.5, "the attack charged the envelope");
        // One-sample release decays by ~0.368 per frame; a few 32-frame
        // blocks of silence cross the 1e-30 floor, after which the state
        // must be exactly zero, not a denormal survivor.
        for _ in 0..40 {
            drive(0.0, &mut core);
        }
        assert_eq!(
            core.env, 0.0,
            "a silent decay must flush to exact zero below the floor"
        );
    }
}
