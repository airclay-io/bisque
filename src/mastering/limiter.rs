// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Airclay LLC

//! Lookahead peak limiting.

use std::f64::consts::{LN_10, TAU};

use crate::dsp::driver::MAX_RUN_FRAMES;
use crate::dsp::math;
use crate::dsp::memory::MemoryLayout;
use crate::dsp::oversample::PolyphaseUpsampler;
use crate::dsp::sanitize::{finite_or_zero, flush_denormal};
use crate::parameter::{ParamId, ParamInfo, Unit};
use crate::processor::{
    AudioBlockMut, DspError, IoMode, Kernel, ProcessSpec, Produced, Sample, SubBlock, Tail,
};

// ---------------------------------------------------------------------------
// Lookahead limiter
// ---------------------------------------------------------------------------

/// True-peak detector oversampling factor and taps per phase.
const TP_FACTOR: usize = 4;
const TP_TAPS: usize = 12;
const TP_TAIL_FRAMES: usize = TP_TAPS - 1;

/// Construction settings for [`Limiter`].
#[derive(Clone, Copy, Debug, PartialEq)]
#[non_exhaustive]
pub struct LimiterSettings {
    /// Limiter ceiling in dBFS.
    pub threshold_db: f64,
    /// Safety margin below `threshold_db` used by the true-peak detector.
    ///
    /// A positive margin lowers the internal gain target by this many dB. Set to
    /// `0.0` for no additional margin.
    pub true_peak_margin_db: f64,
    /// Lookahead time in milliseconds. This determines latency and tail length.
    ///
    /// The rounded lookahead must be at least 11 frames so the delay contains
    /// every sample that can contribute to the true-peak detector response.
    pub lookahead_ms: f64,
    /// Release time in milliseconds. Zero applies release immediately.
    pub release_ms: f64,
}

impl Default for LimiterSettings {
    fn default() -> Self {
        Self {
            threshold_db: -1.0,
            true_peak_margin_db: 0.1,
            lookahead_ms: 1.5,
            release_ms: 50.0,
        }
    }
}

impl LimiterSettings {
    /// Default limiter settings.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the limiter ceiling in dBFS.
    #[must_use]
    pub fn threshold_db(mut self, threshold_db: f64) -> Self {
        self.threshold_db = threshold_db;
        self
    }

    /// Set the true-peak safety margin in dB.
    #[must_use]
    pub fn true_peak_margin_db(mut self, true_peak_margin_db: f64) -> Self {
        self.true_peak_margin_db = true_peak_margin_db;
        self
    }

    /// Set the lookahead time in milliseconds.
    ///
    /// The rounded lookahead must be at least 11 frames.
    #[must_use]
    pub fn lookahead_ms(mut self, lookahead_ms: f64) -> Self {
        self.lookahead_ms = lookahead_ms;
        self
    }

    /// Set the release time in milliseconds. Zero applies release immediately.
    #[must_use]
    pub fn release_ms(mut self, release_ms: f64) -> Self {
        self.release_ms = release_ms;
        self
    }
}

crate::params! {
    /// Smoothed parameter values for [`Limiter`].
    pub struct LimiterParams {
        /// Limiter ceiling in dBFS.
        pub threshold_db => THRESHOLD_DB,
    }
}

/// A lookahead true-peak limiter.
///
/// The signal is delayed by the lookahead. For each frame, required gain is
/// `threshold / peak` when the linked peak is above threshold, otherwise `1.0`.
/// The applied gain is the minimum required gain over the lookahead window
/// (tracked by a monotonic-wedge sliding minimum), shaped by a raised-cosine
/// (Hann) attack smoother, then smoothed upward on release. The attack uses the
/// lookahead remaining after the true-peak detector tail, so every detector
/// report takes full effect before its contributing samples leave the delay.
///
/// Detection uses the maximum of sample peak and oversampled true peak. The
/// internal gain target is `threshold_db - true_peak_margin_db`. Latency and
/// tail length are equal to the lookahead. `flush` drains the delay line while
/// the true-peak detector keeps running over silent input, using the gain
/// target last seen by `render`: an inter-sample peak within the detector's
/// FIR group delay of end-of-input is only reported during the drain, while
/// the samples that produced it are still in the delay line.
///
/// The oversampler's small FIR group delay is not compensated relative to the
/// sample-domain peak detector. Preparation requires enough lookahead to retain
/// its complete response. Gain is applied at sample rate. The raised-cosine
/// attack removes the gain-step discontinuities that cause most reconstructed
/// true-peak overshoot, but difficult transients can still reconstruct above
/// the detector target. This real-time detector is not an offline conformance
/// meter. The default settings include a `0.1` dB margin. Increase it when a
/// downstream limit requires more protection.
#[derive(Debug, Clone)]
pub struct Limiter {
    params: [ParamInfo; 1],
    true_peak_margin_db: f64,
    lookahead_ms: f64,
    release_ms: f64,
    fs: f64,
    look: usize,
    release_coeff: f64,
    delays: Vec<Vec<f64>>, // [channel][look] audio delay lines
    // Sliding minimum over the required-gain stream: an ascending-minima wedge
    // (monotonic deque) in fixed ring storage of capacity look + 1.
    wedge_idx: Vec<u64>, // [look + 1] frame index per wedge entry
    wedge_val: Vec<f64>, // [look + 1] required-gain value per wedge entry
    wedge_head: usize,   // ring position of the front (the current minimum)
    wedge_len: usize,    // live wedge entries
    frame: u64,          // frames pushed through `step_env` since prepare/reset
    // Raised-cosine attack smoother over the windowed minimum.
    hann: Vec<f64>,     // normalized Hann weights, sum = 1.0
    wm_hist: Vec<f64>,  // windowed-minimum history ring
    wm_pos: usize,      // next write position in `wm_hist`
    wm_nonunity: usize, // `wm_hist` entries that are not exactly 1.0
    env: f64,           // current applied gain
    delay_pos: usize,
    // Per-frame applied gain for the current run. A run is capped at one
    // control-rate cell, so this is inline and fixed-size.
    env_scratch: [f64; MAX_RUN_FRAMES],
    tp_os: Vec<PolyphaseUpsampler>, // per-channel true-peak oversampler
    flush_thresh: f64,              // linear gain target last seen by render
    flushed: usize,                 // tail frames drained since the last process/reset
}

impl Limiter {
    /// Threshold parameter in dBFS.
    pub const THRESHOLD_DB: ParamId = LimiterParams::THRESHOLD_DB;

    /// A limiter configured from `settings`. The lookahead fixes the latency.
    #[must_use]
    pub fn with_settings(settings: LimiterSettings) -> Self {
        Self {
            params: [ParamInfo::new(
                Self::THRESHOLD_DB,
                "threshold",
                (-30.0, 0.0),
                settings.threshold_db,
                Unit::Db,
            )],
            true_peak_margin_db: settings.true_peak_margin_db,
            lookahead_ms: settings.lookahead_ms,
            release_ms: settings.release_ms,
            fs: 0.0,
            look: 0,
            release_coeff: 0.0,
            delays: Vec::new(),
            wedge_idx: Vec::new(),
            wedge_val: Vec::new(),
            wedge_head: 0,
            wedge_len: 0,
            frame: 0,
            hann: Vec::new(),
            wm_hist: Vec::new(),
            wm_pos: 0,
            wm_nonunity: 0,
            env: 1.0,
            delay_pos: 0,
            env_scratch: [0.0; MAX_RUN_FRAMES],
            tp_os: Vec::new(),
            flush_thresh: 1.0,
            flushed: 0,
        }
    }

    /// Restore the flush threshold cache to the `param_info` default and clear
    /// the drained-frame counter. Called from `prepare` and `reset`.
    fn reset_flush_state(&mut self) {
        self.flushed = 0;
        let p = &self.params[0];
        let db = finite_or_zero(p.default).clamp(p.range.0, p.range.1);
        self.flush_thresh = math::exp((db - self.true_peak_margin_db) * (LN_10 / 20.0));
    }

    /// A limiter with -1 dBFS ceiling, 0.1 dB true-peak margin, 1.5 ms
    /// lookahead, and 50 ms release.
    #[must_use]
    pub fn new() -> Self {
        Self::with_settings(LimiterSettings::default())
    }

    /// Push one frame's linked required gain `rg` through the lookahead window
    /// minimum, the raised-cosine attack smoother, and the release smoother.
    ///
    /// Returns the gain to apply to the delayed output.
    ///
    /// A detector report can arrive at most `TP_TAIL_FRAMES` after any input
    /// that contributed to it. The attack history is no longer than the delay
    /// remaining after that tail. The sliding minimum therefore fills the
    /// complete history before the contributing input plays. Because the Hann
    /// weights are non-negative and sum to one, the applied gain is no greater
    /// than the reported required gain when that input leaves the delay.
    fn step_env(&mut self, rg: f64) -> f64 {
        let rg = finite_or_zero(rg).clamp(0.0, 1.0);
        let cap = self.wedge_val.len();
        let n = self.frame;
        self.frame += 1;
        // Ascending-minima wedge: drop back entries dominated by `rg` (they are
        // >= rg and expire sooner), drop the front once it leaves the window,
        // then append. The front is the minimum over the last look + 1 frames.
        while self.wedge_len > 0 {
            let back = (self.wedge_head + self.wedge_len - 1) % cap;
            if self.wedge_val[back] >= rg {
                self.wedge_len -= 1;
            } else {
                break;
            }
        }
        while self.wedge_len > 0 && n - self.wedge_idx[self.wedge_head] > self.look as u64 {
            self.wedge_head = if self.wedge_head + 1 == cap {
                0
            } else {
                self.wedge_head + 1
            };
            self.wedge_len -= 1;
        }
        let back = (self.wedge_head + self.wedge_len) % cap;
        self.wedge_idx[back] = n;
        self.wedge_val[back] = rg;
        self.wedge_len += 1;
        let wm = self.wedge_val[self.wedge_head];
        // Raised-cosine attack smoothing: convolve the windowed-minimum history
        // with the normalized Hann weights.
        let l = self.wm_hist.len();
        let old = self.wm_hist[self.wm_pos];
        if old != 1.0 {
            self.wm_nonunity -= 1;
        }
        if wm != 1.0 {
            self.wm_nonunity += 1;
        }
        self.wm_hist[self.wm_pos] = wm;
        self.wm_pos = if self.wm_pos + 1 == l {
            0
        } else {
            self.wm_pos + 1
        };
        let fir = if self.wm_nonunity == 0 {
            // Exact-transparency fast path: every history entry is exactly 1.0,
            // so the convolution is the Hann weight sum. Emitting 1.0 directly
            // keeps below-threshold passthrough bit-exact, which the float
            // normalization of the weight sum would not.
            1.0
        } else {
            let mut acc = 0.0;
            let mut p = if self.wm_pos == 0 {
                l - 1
            } else {
                self.wm_pos - 1
            };
            for &w in &self.hann {
                acc += w * self.wm_hist[p];
                p = if p == 0 { l - 1 } else { p - 1 };
            }
            acc
        };
        self.env = finite_or_zero(self.env).clamp(0.0, 1.0);
        if fir <= self.env {
            self.env = fir; // raised-cosine attack
        } else {
            self.env += (fir - self.env) * self.release_coeff; // release
        }
        self.env = finite_or_zero(self.env).clamp(0.0, 1.0);
        self.env
    }
}

impl Default for Limiter {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Sample> Kernel<T> for Limiter {
    type Params = LimiterParams;

    fn prepare(&mut self, spec: ProcessSpec) -> Result<(), DspError> {
        if spec.sample_rate == 0 {
            return Err(DspError::UnsupportedSpec("sample rate must be non-zero"));
        }
        if !self.true_peak_margin_db.is_finite() || self.true_peak_margin_db < 0.0 {
            return Err(DspError::InvalidParam(
                "true_peak_margin_db must be finite and non-negative",
            ));
        }
        if !self.lookahead_ms.is_finite() || self.lookahead_ms <= 0.0 {
            return Err(DspError::InvalidParam(
                "limiter lookahead_ms must be finite and positive",
            ));
        }
        if !self.release_ms.is_finite() || self.release_ms < 0.0 {
            return Err(DspError::InvalidParam(
                "limiter release_ms must be finite and non-negative",
            ));
        }
        self.fs = f64::from(spec.sample_rate);
        self.look = (self.lookahead_ms * 1e-3 * self.fs).round().max(1.0) as usize;
        if self.look < TP_TAIL_FRAMES {
            return Err(DspError::InvalidParam(
                "limiter lookahead_ms must round to at least 11 frames",
            ));
        }
        let wedge_len = self.look.checked_add(1).ok_or(DspError::InvalidParam(
            "limiter lookahead_ms produces an unaddressable delay line",
        ))?;
        let attack_len = (self.look - TP_TAIL_FRAMES).max(1);
        let os_elements = TP_FACTOR
            .checked_mul(TP_TAPS)
            .and_then(|n| n.checked_add(TP_TAPS))
            .ok_or(DspError::UnsupportedSpec(
                "true-peak oversampler layout exceeds addressable memory",
            ))?;
        MemoryLayout::new()
            .repeated_array::<f64>(spec.channels, self.look) // delay lines
            .array::<u64>(wedge_len)
            .array::<f64>(wedge_len)
            .array::<f64>(attack_len) // Hann window
            .array::<f64>(attack_len) // weighted-minimum history
            .repeated_array::<f64>(spec.channels, os_elements)
            .preflight(spec.max_memory)?;
        let rel = self.release_ms * 1e-3 * self.fs;
        self.release_coeff = if rel == 0.0 {
            1.0
        } else {
            1.0 - math::exp(-1.0 / rel)
        };
        self.delays = vec![vec![0.0; self.look]; spec.channels];
        // Wedge storage: at most look + 1 live entries (one per window frame).
        self.wedge_idx = vec![0; wedge_len];
        self.wedge_val = vec![0.0; wedge_len];
        self.wedge_head = 0;
        self.wedge_len = 0;
        self.frame = 0;
        // Interior Hann samples, strictly positive, normalized to sum to 1.0.
        let l = attack_len;
        let mut hann = vec![0.0; l];
        let mut sum = 0.0;
        for (j, w) in hann.iter_mut().enumerate() {
            *w = 0.5 - 0.5 * math::cos(TAU * (j as f64 + 1.0) / (l as f64 + 1.0));
            sum += *w;
        }
        for w in &mut hann {
            *w /= sum;
        }
        self.hann = hann;
        self.wm_hist = vec![1.0; l];
        self.wm_pos = 0;
        self.wm_nonunity = 0;
        self.env = 1.0;
        self.delay_pos = 0;
        self.tp_os = (0..spec.channels)
            .map(|_| PolyphaseUpsampler::new(TP_FACTOR, TP_TAPS))
            .collect();
        self.reset_flush_state();
        Ok(())
    }

    fn reset(&mut self) {
        for d in &mut self.delays {
            d.fill(0.0);
        }
        self.wedge_head = 0;
        self.wedge_len = 0;
        self.frame = 0;
        self.wm_hist.fill(1.0); // unity gain history
        self.wm_pos = 0;
        self.wm_nonunity = 0;
        for os in &mut self.tp_os {
            os.reset();
        }
        self.env = 1.0;
        self.delay_pos = 0;
        self.reset_flush_state();
    }

    fn latency(&self) -> usize {
        self.look
    }

    fn tail(&self) -> Tail {
        Tail::Frames(self.look)
    }

    fn io_mode(&self) -> IoMode {
        IoMode::InPlace
    }

    fn memory_footprint(&self) -> usize {
        let f = std::mem::size_of::<f64>();
        let u = std::mem::size_of::<u64>();
        self.delays.iter().map(|d| d.len() * f).sum::<usize>()
            + self.wedge_idx.len() * u
            + self.wedge_val.len() * f
            + self.hann.len() * f
            + self.wm_hist.len() * f
            + self
                .tp_os
                .iter()
                .map(PolyphaseUpsampler::footprint)
                .sum::<usize>()
    }

    fn param_info(&self) -> &[ParamInfo] {
        &self.params
    }

    fn render(&mut self, io: &mut SubBlock<'_, '_, '_, T>, params: &LimiterParams) {
        // New input starts a new drain: the drained-frame counter resets.
        self.flushed = 0;
        let thresh_db = params.threshold_db;
        let target_db = thresh_db - self.true_peak_margin_db;
        let thresh = math::exp(target_db * (LN_10 / 20.0));
        // Cache the last rendered gain target so flush can keep detecting.
        self.flush_thresh = thresh;
        let nch = io.channels();
        let run = io.frames();
        debug_assert!(run <= MAX_RUN_FRAMES, "run length is within one CR cell");
        // Compute the applied gain per frame from the linked true peak.
        for i in 0..run {
            let mut peak = 0.0f64;
            for ch in 0..nch {
                let x = finite_or_zero(io.input(ch)[i].to_f64());
                let tp = self.tp_os[ch].peak_abs(x);
                peak = peak.max(x.abs()).max(tp);
            }
            // Below the target the ratio exceeds one and clamps to unity, so
            // there is no branch (and no boundary case) at peak == thresh.
            let rg = (thresh / peak).min(1.0);
            self.env_scratch[i] = self.step_env(rg);
        }
        // Output the delayed signal times the gain.
        let look = self.look;
        let start = self.delay_pos;
        let env = &self.env_scratch;
        for ch in 0..nch {
            let dline = &mut self.delays[ch];
            let buf = io.channel_mut(ch);
            let mut p = start;
            for (i, slot) in buf.iter_mut().enumerate() {
                let x = finite_or_zero(slot.to_f64());
                let d = finite_or_zero(dline[p]);
                dline[p] = x;
                p = if p + 1 == look { 0 } else { p + 1 };
                *slot = T::from_f64(flush_denormal(d * env[i]));
            }
        }
        self.delay_pos = (start + run) % look;
    }

    fn flush(&mut self, out: &mut AudioBlockMut<'_, '_, T>) -> Produced {
        let look = self.look;
        let nch = out.channels();
        // The tail is the samples still in the delay line.
        let want = out.frames().min(look - self.flushed);
        let thresh = self.flush_thresh;
        for i in 0..want {
            // The detector keeps running over silent input: an inter-sample
            // peak within the FIR group delay of end-of-input is only
            // reported here, while the samples that produced it are still in
            // the delay line. The lookahead window contains the report before
            // those samples play, so the same detector target and gain rule
            // carry over to the drained samples.
            let mut peak = 0.0f64;
            for os in &mut self.tp_os {
                peak = peak.max(os.peak_abs(0.0));
            }
            // Below the target the ratio exceeds one and clamps to unity, so
            // there is no branch (and no boundary case) at peak == thresh.
            let rg = (thresh / peak).min(1.0);
            let e = self.step_env(rg);
            let p = self.delay_pos;
            for ch in 0..nch {
                let d = finite_or_zero(self.delays[ch][p]);
                self.delays[ch][p] = 0.0;
                out.channel_mut(ch)[i] = T::from_f64(flush_denormal(d * e));
            }
            self.delay_pos = if p + 1 == look { 0 } else { p + 1 };
        }
        self.flushed += want;
        Produced {
            frames: want,
            done: self.flushed >= look,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn settings_builders_preserve_each_requested_value() {
        let settings = LimiterSettings::new()
            .threshold_db(-6.0)
            .true_peak_margin_db(0.5)
            .lookahead_ms(2.0)
            .release_ms(80.0);
        assert_eq!(settings.threshold_db, -6.0);
        assert_eq!(settings.true_peak_margin_db, 0.5);
        assert_eq!(settings.lookahead_ms, 2.0);
        assert_eq!(settings.release_ms, 80.0);
    }

    #[test]
    fn replacing_the_last_reduced_history_value_restores_unity_count() {
        let mut limiter = Limiter::new();
        limiter.look = 0;
        limiter.wedge_idx = vec![0];
        limiter.wedge_val = vec![1.0];
        limiter.wedge_head = 0;
        limiter.wedge_len = 0;
        limiter.hann = vec![1.0];
        limiter.wm_hist = vec![0.5];
        limiter.wm_pos = 0;
        limiter.wm_nonunity = 1;
        limiter.env = 1.0;
        limiter.release_coeff = 1.0;

        assert_eq!(limiter.step_env(1.0), 1.0);
        assert_eq!(limiter.wm_nonunity, 0);
        assert_eq!(limiter.wm_hist, vec![1.0]);
    }

    #[test]
    fn invalid_lookahead_reports_the_focused_validation_error() {
        let spec = ProcessSpec {
            sample_rate: 48_000,
            channels: 2,
            max_block: 512,
            max_memory: None,
        };
        for bad in [0.0, f64::NAN] {
            let mut limiter = Limiter::with_settings(LimiterSettings::new().lookahead_ms(bad));
            assert!(matches!(
                Kernel::<f32>::prepare(&mut limiter, spec),
                Err(DspError::InvalidParam(
                    "limiter lookahead_ms must be finite and positive"
                ))
            ));
        }
    }

    /// `memory_footprint` equals the byte count derived from the allocation layout.
    #[test]
    fn footprint_is_the_exact_layout_byte_count() {
        let f = std::mem::size_of::<f64>();
        let u = std::mem::size_of::<u64>();
        // 1.5 ms lookahead at 48 kHz is 72 delay samples per channel.
        let look = 72usize;
        let attack = look - TP_TAIL_FRAMES;
        // Each true-peak oversampler holds coefficients plus a delay line.
        let per_os = (TP_FACTOR * TP_TAPS + TP_TAPS) * f;
        for nch in [1usize, 2, 3] {
            let spec = ProcessSpec {
                sample_rate: 48_000,
                channels: nch,
                max_block: 8192,
                max_memory: None,
            };
            let mut k = Limiter::new();
            Kernel::<f32>::prepare(&mut k, spec).expect("prepare");
            // The layout depends on the lookahead.
            assert_eq!(k.look, look, "lookahead samples");

            let ch = nch;
            // Delay lines; wedge index and value rings (look + 1 entries each);
            // Hann weights and windowed-minimum history (attack entries each);
            // and oversamplers. The per-run envelope scratch is inline
            // fixed-size state, not a prepare-time allocation.
            let expected = ch * look * f
                + (look + 1) * u
                + (look + 1) * f
                + attack * f
                + attack * f
                + ch * per_os;
            assert_eq!(
                Kernel::<f32>::memory_footprint(&k),
                expected,
                "footprint for {nch} channels must equal the layout byte count"
            );
        }
    }

    #[test]
    fn huge_lookahead_is_rejected_by_budget_before_allocation() {
        let mut limiter = Limiter::with_settings(LimiterSettings::new().lookahead_ms(1.0e12));
        let result = Kernel::<f32>::prepare(
            &mut limiter,
            ProcessSpec {
                sample_rate: 48_000,
                channels: 2,
                max_block: 512,
                max_memory: Some(1024),
            },
        );
        assert!(matches!(result, Err(DspError::OverBudget { .. })));
    }

    #[test]
    fn release_coefficient_preserves_zero_and_sub_sample_times() {
        let spec = ProcessSpec {
            sample_rate: 48_000,
            channels: 2,
            max_block: 512,
            max_memory: None,
        };

        let mut immediate = Limiter::with_settings(LimiterSettings::new().release_ms(0.0));
        Kernel::<f32>::prepare(&mut immediate, spec).expect("zero release");
        assert_eq!(immediate.release_coeff, 1.0);

        let half_sample_ms = 0.5 * 1000.0 / f64::from(spec.sample_rate);
        let mut sub_sample =
            Limiter::with_settings(LimiterSettings::new().release_ms(half_sample_ms));
        Kernel::<f32>::prepare(&mut sub_sample, spec).expect("half-sample release");
        let expected = 1.0 - math::exp(-2.0);
        assert_eq!(sub_sample.release_coeff, expected);
    }
}
