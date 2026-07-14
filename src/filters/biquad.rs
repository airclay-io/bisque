// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Airclay LLC

//! RBJ biquad filters and frequency-response readouts.
//!
//! `Biquad` implements low-pass, high-pass, low-shelf, high-shelf, and peaking
//! filter shapes. Every shape declares cutoff, Q, and gain parameters; low-pass
//! and high-pass ignore the gain.
//!
//! The implementation uses Direct Form I with `f64` state. Recursive state below
//! a denormal floor is flushed to zero.

use std::f64::consts::PI;

use crate::dsp::math;
use crate::dsp::memory::MemoryLayout;
use crate::dsp::sanitize::{finite_or_zero, flush_denormal};
use crate::parameter::{ParamId, ParamInfo, Unit};
use crate::processor::{
    AudioBlockMut, DspError, IoMode, Kernel, ProcessSpec, Produced, Sample, SubBlock, Tail,
};

/// Tail decay floor. The drain is treated as ended once every state value is
/// below this magnitude.
const TAIL_FLOOR: f64 = 1e-6;

/// Headroom multiplier for the declared tail bound: resonant state can exceed
/// full scale and a near-real pole pair rings with a large envelope constant,
/// so the bound budgets decay from `TAIL_HEADROOM` down to [`TAIL_FLOOR`].
const TAIL_HEADROOM: f64 = 1e3;

crate::params! {
    /// Smoothed parameter values for [`Biquad`].
    pub struct BiquadParams {
        /// Cutoff or center frequency in Hz.
        pub cutoff_hz => CUTOFF_HZ,
        /// Quality factor. Controls resonance or transition shape, including
        /// the shelf transition.
        pub q => Q,
        /// Shelf or peak gain in dB. Ignored by low-pass and high-pass shapes.
        pub gain_db => GAIN_DB,
    }
}

/// RBJ cookbook response shape.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum BiquadKind {
    /// 12 dB/octave low-pass.
    Lowpass,
    /// 12 dB/octave high-pass.
    Highpass,
    /// Low-frequency shelf. Boosts or cuts below the corner by `gain` dB. Q
    /// controls the transition shape and possible resonance.
    LowShelf,
    /// High-frequency shelf. Boosts or cuts above the corner by `gain` dB. Q
    /// controls the transition shape and possible resonance.
    HighShelf,
    /// Peaking filter. Boosts or cuts around the center with bandwidth set by `Q`.
    Peaking,
}

/// Construction settings for [`Biquad`].
#[derive(Clone, Copy, Debug, PartialEq)]
#[non_exhaustive]
pub struct BiquadSettings {
    /// Filter response shape.
    pub kind: BiquadKind,
    /// Initial cutoff or center frequency in Hz.
    pub cutoff_hz: f64,
    /// Initial quality factor or shelf transition control.
    pub q: f64,
    /// Initial shelf or peak gain in dB.
    pub gain_db: f64,
}

impl BiquadSettings {
    /// Default low-pass settings: 1 kHz, Q = 1/sqrt(2), and 0 dB gain.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            kind: BiquadKind::Lowpass,
            cutoff_hz: 1_000.0,
            q: std::f64::consts::FRAC_1_SQRT_2,
            gain_db: 0.0,
        }
    }

    /// Low-pass settings.
    #[must_use]
    pub const fn lowpass() -> Self {
        Self::new()
    }

    /// High-pass settings.
    #[must_use]
    pub const fn highpass() -> Self {
        Self::new().kind(BiquadKind::Highpass)
    }

    /// Low-shelf settings.
    #[must_use]
    pub const fn low_shelf() -> Self {
        Self::new().kind(BiquadKind::LowShelf)
    }

    /// High-shelf settings.
    #[must_use]
    pub const fn high_shelf() -> Self {
        Self::new().kind(BiquadKind::HighShelf)
    }

    /// Peaking-filter settings.
    #[must_use]
    pub const fn peaking() -> Self {
        Self::new().kind(BiquadKind::Peaking)
    }

    /// Set the response shape.
    #[must_use]
    pub const fn kind(mut self, kind: BiquadKind) -> Self {
        self.kind = kind;
        self
    }

    /// Set the initial cutoff or center frequency in Hz.
    #[must_use]
    pub const fn cutoff_hz(mut self, cutoff_hz: f64) -> Self {
        self.cutoff_hz = cutoff_hz;
        self
    }

    /// Set the initial quality factor or shelf transition control.
    #[must_use]
    pub const fn q(mut self, q: f64) -> Self {
        self.q = q;
        self
    }

    /// Set the initial shelf or peak gain in dB.
    #[must_use]
    pub const fn gain_db(mut self, gain_db: f64) -> Self {
        self.gain_db = gain_db;
        self
    }
}

impl Default for BiquadSettings {
    fn default() -> Self {
        Self::new()
    }
}

/// Normalized biquad coefficients (`a0 == 1`) and response readouts.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BiquadCoeffs {
    /// Feed-forward coefficient for `x[n]`.
    pub b0: f64,
    /// Feed-forward coefficient for `x[n-1]`.
    pub b1: f64,
    /// Feed-forward coefficient for `x[n-2]`.
    pub b2: f64,
    /// Feedback coefficient for `y[n-1]`.
    pub a1: f64,
    /// Feedback coefficient for `y[n-2]`.
    pub a2: f64,
}

impl BiquadCoeffs {
    /// Checked RBJ cookbook coefficients for `kind` at cutoff or center `f0`
    /// Hz.
    ///
    /// `q` is the quality factor. `gain_db` is the shelf or peak gain and is
    /// ignored by low-pass and high-pass filters. `fs` is the sample rate in Hz.
    ///
    /// # Errors
    /// Returns [`DspError::InvalidParam`] unless every value is finite,
    /// `fs > 0`, `0 < f0 < fs / 2`, `q > 0`, and the resulting coefficients
    /// are finite and stable.
    pub fn try_rbj(
        kind: BiquadKind,
        fs: f64,
        f0: f64,
        q: f64,
        gain_db: f64,
    ) -> Result<Self, DspError> {
        if !fs.is_finite()
            || !f0.is_finite()
            || !q.is_finite()
            || !gain_db.is_finite()
            || fs <= 0.0
            || f0 <= 0.0
            || f0 >= fs * 0.5
            || q <= 0.0
        {
            return Err(DspError::InvalidParam(
                "biquad coefficient inputs must be finite and inside the filter domain",
            ));
        }
        let coeffs = Self::rbj_unchecked(kind, fs, f0, q, gain_db);
        if ![coeffs.b0, coeffs.b1, coeffs.b2, coeffs.a1, coeffs.a2]
            .into_iter()
            .all(f64::is_finite)
            || !coeffs.is_stable()
        {
            return Err(DspError::InvalidParam(
                "biquad inputs must produce finite stable coefficients",
            ));
        }
        Ok(coeffs)
    }

    fn rbj_unchecked(kind: BiquadKind, fs: f64, f0: f64, q: f64, gain_db: f64) -> Self {
        let w0 = 2.0 * PI * f0 / fs;
        let cos_w0 = math::cos(w0);
        let sin_w0 = math::sin(w0);
        let alpha = sin_w0 / (2.0 * q);
        // Shelf and peak amplitude.
        let a = math::pow(10.0, gain_db / 40.0);
        // Unnormalized coefficients, normalized by a0 below.
        let (b0, b1, b2, a0, a1, a2) = match kind {
            BiquadKind::Lowpass => {
                let body = 1.0 - cos_w0;
                (
                    body * 0.5,
                    body,
                    body * 0.5,
                    1.0 + alpha,
                    -2.0 * cos_w0,
                    1.0 - alpha,
                )
            }
            BiquadKind::Highpass => {
                let body = 1.0 + cos_w0;
                (
                    body * 0.5,
                    -body,
                    body * 0.5,
                    1.0 + alpha,
                    -2.0 * cos_w0,
                    1.0 - alpha,
                )
            }
            BiquadKind::Peaking => (
                1.0 + alpha * a,
                -2.0 * cos_w0,
                1.0 - alpha * a,
                1.0 + alpha / a,
                -2.0 * cos_w0,
                1.0 - alpha / a,
            ),
            BiquadKind::LowShelf => {
                let am1 = a - 1.0;
                let ap1 = a + 1.0;
                let two_sqrt_a_alpha = 2.0 * math::sqrt(a) * alpha;
                (
                    a * (ap1 - am1 * cos_w0 + two_sqrt_a_alpha),
                    2.0 * a * (am1 - ap1 * cos_w0),
                    a * (ap1 - am1 * cos_w0 - two_sqrt_a_alpha),
                    ap1 + am1 * cos_w0 + two_sqrt_a_alpha,
                    -2.0 * (am1 + ap1 * cos_w0),
                    ap1 + am1 * cos_w0 - two_sqrt_a_alpha,
                )
            }
            BiquadKind::HighShelf => {
                let am1 = a - 1.0;
                let ap1 = a + 1.0;
                let two_sqrt_a_alpha = 2.0 * math::sqrt(a) * alpha;
                (
                    a * (ap1 + am1 * cos_w0 + two_sqrt_a_alpha),
                    -2.0 * a * (am1 + ap1 * cos_w0),
                    a * (ap1 + am1 * cos_w0 - two_sqrt_a_alpha),
                    ap1 - am1 * cos_w0 + two_sqrt_a_alpha,
                    2.0 * (am1 - ap1 * cos_w0),
                    ap1 - am1 * cos_w0 - two_sqrt_a_alpha,
                )
            }
        };
        let inv = 1.0 / a0;
        Self {
            b0: b0 * inv,
            b1: b1 * inv,
            b2: b2 * inv,
            a1: a1 * inv,
            a2: a2 * inv,
        }
    }

    /// The complex response `H(e^{jw})` at normalized angular frequency `w`.
    ///
    /// `w` is in radians per sample in `[0, pi]`. Returns `(real, imag)`.
    fn eval(&self, w: f64) -> (f64, f64) {
        // z^-1 = cos(w) - j sin(w). z^-2 = cos(2w) - j sin(2w).
        let (cos1, sin1) = (math::cos(w), math::sin(w));
        let (cos2, sin2) = (math::cos(2.0 * w), math::sin(2.0 * w));
        let num_re = self.b0 + self.b1 * cos1 + self.b2 * cos2;
        let num_im = -(self.b1 * sin1 + self.b2 * sin2);
        let den_re = 1.0 + self.a1 * cos1 + self.a2 * cos2;
        let den_im = -(self.a1 * sin1 + self.a2 * sin2);
        let denom = den_re * den_re + den_im * den_im;
        (
            (num_re * den_re + num_im * den_im) / denom,
            (num_im * den_re - num_re * den_im) / denom,
        )
    }

    /// Magnitude (linear) of the response at `w` radians/sample. The usual
    /// analysis interval is `0.0..=PI`.
    #[must_use]
    pub fn magnitude(&self, w: f64) -> f64 {
        let (re, im) = self.eval(w);
        math::hypot(re, im)
    }

    /// Phase (radians) of the response at `w` radians/sample. The usual
    /// analysis interval is `0.0..=PI`.
    #[must_use]
    pub fn phase(&self, w: f64) -> f64 {
        let (re, im) = self.eval(w);
        math::atan2(im, re)
    }

    /// Approximate group delay in samples at `w` radians/sample, calculated by
    /// a fixed central difference. It is not meaningful where the response
    /// magnitude is zero.
    #[must_use]
    pub fn group_delay(&self, w: f64) -> f64 {
        let step = 1e-4;
        let (re_hi, im_hi) = self.eval(w + step);
        let (re_lo, im_lo) = self.eval(w - step);
        // arg(H(w + step) * conj(H(w - step))) avoids phase-wrap artifacts.
        let prod_re = re_hi * re_lo + im_hi * im_lo;
        let prod_im = im_hi * re_lo - re_hi * im_lo;
        -math::atan2(prod_im, prod_re) / (2.0 * step)
    }

    /// Whether both poles are strictly inside the unit circle.
    #[must_use]
    pub fn is_stable(&self) -> bool {
        self.a2.abs() < 1.0 && self.a1.abs() < 1.0 + self.a2
    }
}

/// One channel's Direct Form I state.
#[derive(Clone, Copy, Debug, Default)]
struct State {
    x1: f64,
    x2: f64,
    y1: f64,
    y2: f64,
}

/// An RBJ biquad with automatable cutoff, Q, and gain.
///
/// Coefficients are recomputed once per sub-block from cutoff, Q, and gain.
/// Gain is declared on all shapes and ignored by low-pass and high-pass.
/// Per-channel state is allocated in `prepare` and cleared by `reset`.
/// The effective cutoff is clamped to `1.0..=0.999 * Nyquist`; at lower sample
/// rates, values near the declared 24 kHz upper bound share that ceiling.
///
/// `tail` reports a constant, conservative upper bound computed from the
/// declared range extremes: the slowest pole pair the cutoff, Q, and gain
/// ranges allow, decaying from `1e3` down to the `1e-6` floor. The bound is a
/// decay guarantee, not a silence guarantee: the drain covers at least
/// `1e3 / 1e-6` (180 dB) of decay relative to the state at end of input, so
/// state within the `1e3` headroom ends below the absolute floor, and hotter
/// (still finite) input decays by the same ratio before `done`. Actual drains
/// end far earlier: `flush` continues the recursion with silent input, using
/// the coefficients last seen by `render`, and reports `done` as soon as
/// every state value has decayed below `1e-6`. New input starts a new drain;
/// a host that wants a shorter drain caps the frames it requests.
#[derive(Clone, Debug)]
pub struct Biquad {
    kind: BiquadKind,
    /// Cutoff, Q, and gain. Every shape declares all three.
    params: [ParamInfo; 3],
    fs: f64,
    state: Vec<State>,
    flush_coeffs: BiquadCoeffs, // coefficients last seen by render
    tail_bound: usize,          // declared worst-case tail frames, set in prepare
    flushed: usize,             // tail frames drained since the last process/reset
}

impl Biquad {
    /// Cutoff or center frequency in Hz.
    pub const CUTOFF_HZ: ParamId = BiquadParams::CUTOFF_HZ;
    /// Quality factor or shelf transition control.
    pub const Q: ParamId = BiquadParams::Q;
    /// Shelf or peak gain in dB. Declared on all shapes and ignored by low-pass
    /// and high-pass.
    pub const GAIN_DB: ParamId = BiquadParams::GAIN_DB;

    /// A low-pass biquad with default settings.
    #[must_use]
    pub fn new() -> Self {
        Self::with_settings(BiquadSettings::new())
    }

    /// A biquad configured from `settings`.
    #[must_use]
    pub fn with_settings(settings: BiquadSettings) -> Self {
        let params = [
            ParamInfo::new(
                Self::CUTOFF_HZ,
                "cutoff",
                (10.0, 24_000.0),
                settings.cutoff_hz,
                Unit::Hz,
            ),
            ParamInfo::new(Self::Q, "q", (0.1, 16.0), settings.q, Unit::Q),
            ParamInfo::new(
                Self::GAIN_DB,
                "gain",
                (-24.0, 24.0),
                settings.gain_db,
                Unit::Db,
            ),
        ];
        Self {
            kind: settings.kind,
            params,
            fs: 0.0,
            state: Vec::new(),
            flush_coeffs: BiquadCoeffs {
                b0: 1.0,
                b1: 0.0,
                b2: 0.0,
                a1: 0.0,
                a2: 0.0,
            },
            tail_bound: 0,
            flushed: 0,
        }
    }

    /// Restore the flush coefficient cache to the `param_info` defaults and
    /// clear the drained-frame counter. Called from `prepare` and `reset`.
    fn reset_flush_state(&mut self) {
        self.flushed = 0;
        if self.fs > 0.0 {
            self.flush_coeffs = self.runtime_coeffs(
                self.params[0].default,
                self.params[1].default,
                self.params[2].default,
            );
        }
    }

    /// A low-pass biquad.
    #[must_use]
    pub fn lowpass() -> Self {
        Self::with_settings(BiquadSettings::lowpass())
    }

    /// A high-pass biquad.
    #[must_use]
    pub fn highpass() -> Self {
        Self::with_settings(BiquadSettings::highpass())
    }

    /// A low-shelf biquad.
    #[must_use]
    pub fn low_shelf() -> Self {
        Self::with_settings(BiquadSettings::low_shelf())
    }

    /// A high-shelf biquad.
    #[must_use]
    pub fn high_shelf() -> Self {
        Self::with_settings(BiquadSettings::high_shelf())
    }

    /// A peaking biquad.
    #[must_use]
    pub fn peaking() -> Self {
        Self::with_settings(BiquadSettings::peaking())
    }

    /// The coefficients this filter would use at cutoff or center `f0`, quality
    /// `q`, and optional `gain_db`. Values are clamped to the declared parameter
    /// ranges and the prepared sample-rate ceiling, exactly as they are during
    /// rendering.
    ///
    /// # Errors
    /// Returns [`DspError::UnsupportedSpec`] before `prepare`, or
    /// [`DspError::InvalidParam`] for a non-finite input.
    pub fn try_coeffs(&self, f0: f64, q: f64, gain_db: f64) -> Result<BiquadCoeffs, DspError> {
        if self.fs < 3.0 {
            return Err(DspError::UnsupportedSpec(
                "biquad coefficient readouts require prepare",
            ));
        }
        if !f0.is_finite() || !q.is_finite() || !gain_db.is_finite() {
            return Err(DspError::InvalidParam(
                "biquad coefficient readout inputs must be finite",
            ));
        }
        Ok(self.runtime_coeffs(f0, q, gain_db))
    }

    /// Convert a physical frequency (Hz) to normalized angular frequency
    /// (radians/sample) for the readouts. Valid after `prepare`.
    #[must_use]
    pub fn omega(&self, f_hz: f64) -> f64 {
        2.0 * PI * f_hz / self.fs
    }

    /// Clamp a requested cutoff to `(0, fs/2)`.
    fn clamp_cutoff(&self, f0: f64) -> f64 {
        f0.clamp(1.0, self.fs * 0.5 * 0.999)
    }

    fn runtime_coeffs(&self, f0: f64, q: f64, gain_db: f64) -> BiquadCoeffs {
        let f0 = f0.clamp(self.params[0].range.0, self.params[0].range.1);
        let q = q.clamp(self.params[1].range.0, self.params[1].range.1);
        let gain_db = gain_db.clamp(self.params[2].range.0, self.params[2].range.1);
        BiquadCoeffs::rbj_unchecked(self.kind, self.fs, self.clamp_cutoff(f0), q, gain_db)
    }
}

impl Default for Biquad {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Sample> Kernel<T> for Biquad {
    type Params = BiquadParams;

    fn prepare(&mut self, spec: ProcessSpec) -> Result<(), DspError> {
        // Below 3 Hz the cutoff clamp band (1.0, fs/2 * 0.999) inverts and
        // `f64::clamp` would panic in render, so the spec is rejected here.
        if spec.sample_rate < 3 {
            return Err(DspError::UnsupportedSpec("sample rate must be at least 3"));
        }
        MemoryLayout::new()
            .array::<State>(spec.channels)
            .preflight(spec.max_memory)?;
        self.fs = f64::from(spec.sample_rate);
        self.state = vec![State::default(); spec.channels];
        // A constant, conservative tail bound from the declared range
        // extremes: the slowest reachable pole pair is the peaking denominator
        // (1 - alpha/A) at maximum Q and gain, at whichever cutoff extreme
        // yields the smaller sin(w0). The drain budgets decay from
        // TAIL_HEADROOM down to TAIL_FLOOR.
        let w_lo = 2.0 * PI * self.clamp_cutoff(self.params[0].range.0) / self.fs;
        let w_hi = 2.0 * PI * self.clamp_cutoff(self.params[0].range.1) / self.fs;
        let sin_min = math::sin(w_lo).min(math::sin(w_hi));
        let q_max = self.params[1].range.1;
        let a_max = math::pow(10.0, self.params[2].range.1 / 40.0);
        let alpha = sin_min / (2.0 * q_max * a_max);
        let r = math::sqrt((1.0 - alpha) / (1.0 + alpha)); // slowest pole radius
        self.tail_bound = (math::ln(TAIL_HEADROOM / TAIL_FLOOR) / -math::ln(r)).ceil() as usize;
        self.reset_flush_state();
        Ok(())
    }

    fn reset(&mut self) {
        for s in &mut self.state {
            *s = State::default();
        }
        self.reset_flush_state();
    }

    fn tail(&self) -> Tail {
        Tail::Frames(self.tail_bound)
    }

    fn io_mode(&self) -> IoMode {
        IoMode::InPlace
    }

    fn memory_footprint(&self) -> usize {
        self.state.len() * std::mem::size_of::<State>()
    }

    fn param_info(&self) -> &[ParamInfo] {
        &self.params
    }

    fn render(&mut self, io: &mut SubBlock<'_, '_, '_, T>, params: &BiquadParams) {
        // New input starts a new drain: the drained-frame counter resets.
        self.flushed = 0;
        // Coefficients are constant for this sub-block.
        // Low-pass and high-pass ignore gain inside the coefficient calculation.
        let c = self.runtime_coeffs(params.cutoff_hz, params.q, params.gain_db);
        // Cache the last rendered coefficients so flush can continue the
        // recursion.
        self.flush_coeffs = c;
        for (ch, st) in self.state.iter_mut().enumerate() {
            for sample in io.channel_mut(ch).iter_mut() {
                let x0 = finite_or_zero(sample.to_f64());
                // Direct Form I with separate multiply and add operations.
                let y0 = c.b0 * x0 + c.b1 * st.x1 + c.b2 * st.x2 - c.a1 * st.y1 - c.a2 * st.y2;
                let y0 = flush_denormal(y0);
                st.x2 = st.x1;
                st.x1 = x0;
                st.y2 = st.y1;
                st.y1 = y0;
                *sample = T::from_f64(y0);
            }
        }
    }

    fn flush(&mut self, out: &mut AudioBlockMut<'_, '_, T>) -> Produced {
        let bound = self.tail_bound;
        let state_peak = self.state.iter().fold(0.0f64, |m, s| {
            m.max(s.x1.abs())
                .max(s.x2.abs())
                .max(s.y1.abs())
                .max(s.y2.abs())
        });
        if self.flushed >= bound || state_peak < TAIL_FLOOR {
            self.flushed = bound;
            return Produced {
                frames: 0,
                done: true,
            };
        }
        let want = out.frames().min(bound.saturating_sub(self.flushed));
        let c = self.flush_coeffs;
        for (ch, st) in self.state.iter_mut().enumerate() {
            let buf = out.channel_mut(ch);
            for slot in buf.iter_mut().take(want) {
                // Direct Form I continued with silent input (x0 = 0).
                let y0 = flush_denormal(c.b1 * st.x1 + c.b2 * st.x2 - c.a1 * st.y1 - c.a2 * st.y2);
                st.x2 = st.x1;
                st.x1 = 0.0;
                st.y2 = st.y1;
                st.y1 = y0;
                *slot = T::from_f64(y0);
            }
        }
        self.flushed += want;
        // Deterministic early exit: the tail is done once every state value is
        // below the decay floor or the declared bound has been drained.
        let peak = self.state.iter().fold(0.0f64, |m, s| {
            m.max(s.x1.abs())
                .max(s.x2.abs())
                .max(s.y1.abs())
                .max(s.y2.abs())
        });
        let done = self.flushed >= bound || peak < TAIL_FLOOR;
        if done {
            self.flushed = bound;
        }
        Produced { frames: want, done }
    }
}

#[cfg(test)]
mod tests {
    //! Unit tests for private helper boundaries.

    use super::{Biquad, BiquadCoeffs, BiquadKind, BiquadSettings, TAIL_FLOOR};
    use crate::dsp::sanitize::{flush_denormal, DENORMAL_FLOOR};
    use crate::processor::{AudioBlockMut, Kernel, ProcessSpec};

    #[test]
    fn convenience_constructors_select_their_documented_shapes() {
        for (filter, expected) in [
            (Biquad::lowpass(), BiquadKind::Lowpass),
            (Biquad::highpass(), BiquadKind::Highpass),
            (Biquad::low_shelf(), BiquadKind::LowShelf),
            (Biquad::high_shelf(), BiquadKind::HighShelf),
            (Biquad::peaking(), BiquadKind::Peaking),
        ] {
            assert_eq!(filter.kind, expected);
        }
    }

    #[test]
    fn gain_is_ignored_by_lowpass_and_highpass_only() {
        // Every shape declares gain, but low-pass and high-pass coefficients
        // do not depend on it. Shelves and peaking do.
        for kind in [BiquadKind::Lowpass, BiquadKind::Highpass] {
            let mut bq = Biquad::with_settings(BiquadSettings::new().kind(kind));
            bq.fs = 48_000.0;
            assert_eq!(
                bq.try_coeffs(1_000.0, 0.707, 0.0).expect("coefficients"),
                bq.try_coeffs(1_000.0, 0.707, 12.0).expect("coefficients"),
                "{kind:?} coefficients must not depend on gain"
            );
        }
        for kind in [
            BiquadKind::LowShelf,
            BiquadKind::HighShelf,
            BiquadKind::Peaking,
        ] {
            let mut bq = Biquad::with_settings(BiquadSettings::new().kind(kind));
            bq.fs = 48_000.0;
            assert_ne!(
                bq.try_coeffs(1_000.0, 0.707, 0.0).expect("coefficients"),
                bq.try_coeffs(1_000.0, 0.707, 12.0).expect("coefficients"),
                "{kind:?} coefficients must read gain"
            );
        }
    }

    #[test]
    fn clamp_cutoff_hits_the_exact_band_edges() {
        // The clamp band is (1.0, fs * 0.5 * 0.999).
        let mut bq = Biquad::lowpass();
        bq.fs = 48_000.0;
        let upper = 48_000.0 * 0.5 * 0.999; // 23_976.0
        assert_eq!(
            bq.clamp_cutoff(1.0e9),
            upper,
            "a huge cutoff clamps to fs/2 * 0.999"
        );
        assert_eq!(bq.clamp_cutoff(0.0), 1.0, "a zero cutoff clamps up to 1 Hz");
        // Values inside the band are unchanged.
        assert_eq!(bq.clamp_cutoff(1_000.0), 1_000.0, "in-band cutoff is kept");
    }

    #[test]
    fn flush_done_decides_at_the_exact_decay_floor() {
        // A zero-frame flush only scans the state, so the done flag isolates
        // the floor comparison: state exactly at the floor still drains
        // (kills < -> <=), state below it is done.
        let mut bq = Biquad::lowpass();
        Kernel::<f32>::prepare(
            &mut bq,
            ProcessSpec {
                sample_rate: 48_000,
                channels: 1,
                max_block: 32,
                max_memory: None,
            },
        )
        .expect("prepare");

        bq.state[0].y1 = TAIL_FLOOR;
        let mut planes: [&mut [f32]; 1] = [&mut []];
        let mut out = AudioBlockMut::new(&mut planes);
        assert!(
            !Kernel::<f32>::flush(&mut bq, &mut out).done,
            "state at the floor still drains"
        );

        bq.state[0].y1 = TAIL_FLOOR * 0.5;
        let mut planes: [&mut [f32]; 1] = [&mut []];
        let mut out = AudioBlockMut::new(&mut planes);
        assert!(
            Kernel::<f32>::flush(&mut bq, &mut out).done,
            "state below the floor is done"
        );
    }

    #[test]
    fn flush_detects_state_that_decays_below_the_floor() {
        let mut bq = Biquad::lowpass();
        Kernel::<f32>::prepare(
            &mut bq,
            ProcessSpec {
                sample_rate: 48_000,
                channels: 1,
                max_block: 32,
                max_memory: None,
            },
        )
        .expect("prepare");
        bq.state[0].y1 = TAIL_FLOOR;
        bq.flush_coeffs = BiquadCoeffs {
            b0: 0.0,
            b1: 0.0,
            b2: 0.0,
            a1: 0.0,
            a2: 0.0,
        };
        let mut samples = [0.0f32; 2];
        let mut planes = [samples.as_mut_slice()];
        let mut out = AudioBlockMut::new(&mut planes);
        let produced = Kernel::<f32>::flush(&mut bq, &mut out);
        assert_eq!(produced.frames, 2);
        assert!(produced.done);
    }

    #[test]
    fn flush_accumulates_its_frame_budget_across_calls() {
        let mut bq = Biquad::lowpass();
        Kernel::<f32>::prepare(
            &mut bq,
            ProcessSpec {
                sample_rate: 48_000,
                channels: 1,
                max_block: 32,
                max_memory: None,
            },
        )
        .expect("prepare");
        bq.tail_bound = 3;
        bq.state[0].y1 = 1.0;
        bq.flush_coeffs = BiquadCoeffs {
            b0: 0.0,
            b1: 0.0,
            b2: 0.0,
            a1: -1.0,
            a2: 0.0,
        };

        for expected_done in [false, false, true] {
            let mut samples = [0.0f32; 1];
            let mut planes = [samples.as_mut_slice()];
            let mut out = AudioBlockMut::new(&mut planes);
            let produced = Kernel::<f32>::flush(&mut bq, &mut out);
            assert_eq!(produced.frames, 1);
            assert_eq!(produced.done, expected_done);
        }

        let mut samples = [0.0f32; 1];
        let mut planes = [samples.as_mut_slice()];
        let mut out = AudioBlockMut::new(&mut planes);
        let produced = Kernel::<f32>::flush(&mut bq, &mut out);
        assert_eq!(produced.frames, 0);
        assert!(produced.done);
    }

    #[test]
    fn flush_denormal_zeroes_below_the_floor_only() {
        // Values strictly below DENORMAL_FLOOR are zeroed.
        assert_eq!(flush_denormal(1.0e-40), 0.0, "deep subnormal flushes to 0");
        assert_eq!(flush_denormal(-1.0e-40), 0.0, "sign-agnostic via abs()");
        assert_eq!(flush_denormal(f64::NAN), 0.0);
        assert_eq!(flush_denormal(f64::INFINITY), 0.0);
        // Values exactly at the floor are kept.
        assert_eq!(
            flush_denormal(DENORMAL_FLOOR),
            DENORMAL_FLOOR,
            "the floor itself is not flushed (kills < -> <=)"
        );
        // Values above the floor are unchanged.
        assert_eq!(flush_denormal(0.5), 0.5, "audible state is preserved");
    }
}
