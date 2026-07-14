// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Airclay LLC

//! Feedback delay.

use crate::parameter::{ParamId, ParamInfo, Unit};
use crate::processor::{
    AudioBlockMut, DspError, IoMode, Kernel, ProcessSpec, Produced, Sample, SubBlock, Tail,
};
use crate::{
    dsp::math,
    dsp::memory::MemoryLayout,
    dsp::sanitize::{finite_or_zero, flush_denormal},
};

/// Tail decay floor. The tail is treated as ended once every ring value is
/// below this magnitude.
const TAIL_FLOOR: f64 = 1e-6;

/// Retained-state magnitude covered by the declared tail bound.
///
/// This includes the `1 / (1 - MAX_FEEDBACK)` buildup from sustained
/// full-scale input and leaves additional margin for hotter finite input.
const TAIL_HEADROOM: f64 = 1e3;

/// The declared maximum of the `FEEDBACK` parameter range.
const MAX_FEEDBACK: f64 = 0.95;

/// Worst-case number of feedback passes before [`TAIL_HEADROOM`] decays below
/// [`TAIL_FLOOR`] at the declared feedback maximum.
///
/// Computed through the vendored [`math::ln`], not the platform library, so
/// the constant is identical on every platform.
fn k_max() -> usize {
    (math::ln(TAIL_FLOOR / TAIL_HEADROOM) / math::ln(MAX_FEEDBACK)).ceil() as usize
}

fn tail_bound(ring_len: usize) -> usize {
    (1 + k_max()).saturating_mul(ring_len)
}

/// Crossfade length in frames for a delay-time change: one control-rate cell.
///
/// Delay-time targets can only change at control-grid boundaries (the driver
/// hands `render` fixed-parameter runs), so a fade this long begun at one
/// boundary completes by the next and at most one fade is ever in flight.
const XFADE_FRAMES: usize = crate::dsp::driver::MAX_RUN_FRAMES;

/// Read the wet (delayed) sample for write position `p`.
///
/// Settled (`k >= XFADE_FRAMES`): the committed integer tap `d_new`, exactly
/// as an uncrossfaded delay line reads it. Fading (`1 <= k < XFADE_FRAMES`):
/// an equal-gain linear blend `old * (1 - k/32) + new * (k/32)` of the two
/// integer taps, so consecutive frames move the wet path by at most
/// `1/XFADE_FRAMES` of the tap difference plus the signal's own frame-to-frame
/// change. The weights are exact binary fractions, so the blend is
/// deterministic.
#[inline]
fn read_delayed(
    ring: &[f64],
    ring_len: usize,
    p: usize,
    d_new: usize,
    d_old: usize,
    k: usize,
) -> f64 {
    let tap = |d: usize| {
        let read = if p >= d { p - d } else { p + ring_len - d };
        finite_or_zero(ring[read])
    };
    if k >= XFADE_FRAMES {
        tap(d_new)
    } else {
        let w = k as f64 / XFADE_FRAMES as f64;
        tap(d_old) * (1.0 - w) + tap(d_new) * w
    }
}

/// Construction settings for [`Delay`].
#[derive(Clone, Copy, Debug, PartialEq)]
#[non_exhaustive]
pub struct DelaySettings {
    /// Initial delay time in milliseconds.
    pub delay_ms: f64,
    /// Feedback amount.
    pub feedback: f64,
    /// Wet/dry mix where `0.0` is dry and `1.0` is wet.
    pub mix: f64,
    /// Maximum delay time in milliseconds. This sizes the ring buffer.
    pub max_delay_ms: f64,
}

impl Default for DelaySettings {
    fn default() -> Self {
        Self {
            delay_ms: 250.0,
            feedback: 0.3,
            mix: 0.5,
            max_delay_ms: 1000.0,
        }
    }
}

impl DelaySettings {
    /// Default delay settings.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the initial delay time in milliseconds.
    #[must_use]
    pub fn delay_ms(mut self, delay_ms: f64) -> Self {
        self.delay_ms = delay_ms;
        self
    }

    /// Set the feedback amount.
    #[must_use]
    pub fn feedback(mut self, feedback: f64) -> Self {
        self.feedback = feedback;
        self
    }

    /// Set the wet/dry mix.
    #[must_use]
    pub fn mix(mut self, mix: f64) -> Self {
        self.mix = mix;
        self
    }

    /// Set the maximum delay time in milliseconds.
    #[must_use]
    pub fn max_delay_ms(mut self, max_delay_ms: f64) -> Self {
        self.max_delay_ms = max_delay_ms;
        self
    }
}

crate::params! {
    /// Smoothed parameter values for [`Delay`].
    pub struct DelayParams {
        /// Delay time in milliseconds.
        pub delay_ms => DELAY_MS,
        /// Feedback amount.
        pub feedback => FEEDBACK,
        /// Wet/dry mix where `0.0` is dry and `1.0` is wet.
        pub mix => MIX,
    }
}

/// A feedback delay. Each sample is mixed with a copy of itself from `delay`
/// milliseconds ago, and a `feedback` fraction is fed back into the line to make
/// the echo repeat and decay. `mix` crossfades dry (input) and wet (delayed).
///
/// Delay time, feedback, and mix are automatable.
///
/// The maximum delay is fixed at construction and sizes the ring buffer. Delay
/// time is rounded to whole samples, and at a settled value the line reads one
/// integer tap with no fractional interpolation, exactly as before any
/// automation. A delay-time change is click-safe: the wet path crossfades from
/// the old integer tap to the new one over 32 frames (one control-rate cell)
/// with equal-gain linear weights, bounding each frame's wet-path step to
/// `1/32` of the tap difference. While a fade is active, later target changes
/// wait for it to finish (checked once per fixed-parameter run), so continuous
/// automation advances in bounded 32-frame transitions and may skip
/// intermediate integer taps. The feedback path is clean feedback without
/// damping filters or saturation.
///
/// `tail` reports a constant, conservative upper bound computed from the
/// declared range maxima: delay time and feedback are automatable, so the
/// bound assumes worst-case automation to the full ring capacity at the
/// maximum feedback of `0.95`, giving `(1 + 270) * ring_capacity` frames
/// (`270 = ceil(ln(1e-6) / ln(0.95))` feedback passes to decay below `1e-6`).
/// Actual drains end far earlier: `flush` continues the feedback recursion
/// with silent input (an in-flight delay-time crossfade keeps fading), using
/// the last feedback and mix values seen by `render`, and reports `done` as
/// soon as every ring value has decayed below `1e-6`. New input starts a new
/// drain; a host that wants a shorter drain caps the frames it requests.
#[derive(Debug, Clone)]
pub struct Delay {
    params: [ParamInfo; 3],
    max_delay_ms: f64,
    fs: f64,
    rings: Vec<Vec<f64>>, // per channel, length = max delay + 1
    pos: usize,           // write position
    tap_cur: usize,       // committed integer tap (the fade target while fading)
    tap_prev: usize,      // fade-source tap, read only while a fade is active
    xfade_pos: usize,     // fade frames completed; >= XFADE_FRAMES when settled
    flush_feedback: f64,  // feedback last seen by render
    flush_mix: f64,       // mix last seen by render
    flushed: usize,       // tail frames drained since the last process/reset
}

impl Delay {
    /// Delay time in milliseconds.
    pub const DELAY_MS: ParamId = DelayParams::DELAY_MS;
    /// Feedback amount.
    pub const FEEDBACK: ParamId = DelayParams::FEEDBACK;
    /// Wet/dry mix.
    pub const MIX: ParamId = DelayParams::MIX;

    /// A delay configured from `settings`. `max_delay_ms` sizes the ring buffer
    /// and caps the delay-time range.
    ///
    /// The settings are stored as given, not clamped: `prepare` returns
    /// [`DspError::InvalidParam`] for a non-finite or sub-1 ms `max_delay_ms`,
    /// and (through the parameter-metadata validation) for a `delay_ms`,
    /// `feedback`, or `mix` outside its declared range, since each becomes
    /// that parameter's default.
    #[must_use]
    pub fn with_settings(settings: DelaySettings) -> Self {
        Self {
            params: [
                ParamInfo::new(
                    Self::DELAY_MS,
                    "delay",
                    (1.0, settings.max_delay_ms),
                    settings.delay_ms,
                    Unit::Ms,
                ),
                ParamInfo::new(
                    Self::FEEDBACK,
                    "feedback",
                    (0.0, 0.95),
                    settings.feedback,
                    Unit::Linear,
                ),
                ParamInfo::new(Self::MIX, "mix", (0.0, 1.0), settings.mix, Unit::Linear),
            ],
            max_delay_ms: settings.max_delay_ms,
            fs: 0.0,
            rings: Vec::new(),
            pos: 0,
            tap_cur: 1,
            tap_prev: 1,
            xfade_pos: XFADE_FRAMES,
            flush_feedback: 0.0,
            flush_mix: 0.0,
            flushed: 0,
        }
    }

    /// A delay with 250 ms delay, 0.3 feedback, 0.5 mix, and 1 s maximum.
    #[must_use]
    pub fn new() -> Self {
        Self::with_settings(DelaySettings::default())
    }

    /// Restore the flush parameter cache and the tap/crossfade state to the
    /// `param_info` defaults and clear the drained-frame counter. Called from
    /// `prepare` and `reset`.
    fn reset_flush_state(&mut self) {
        self.flushed = 0;
        if let Some(ring) = self.rings.first() {
            let ring_len = ring.len();
            let max_delay = ring_len.saturating_sub(1);
            self.tap_cur = ((finite_or_zero(self.params[0].default) * 1e-3 * self.fs).round()
                as usize)
                .clamp(1, max_delay);
            self.tap_prev = self.tap_cur;
            self.xfade_pos = XFADE_FRAMES;
            self.flush_feedback = finite_or_zero(self.params[1].default).clamp(0.0, MAX_FEEDBACK);
            self.flush_mix = finite_or_zero(self.params[2].default).clamp(0.0, 1.0);
        }
    }
}

impl Default for Delay {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Sample> Kernel<T> for Delay {
    type Params = DelayParams;

    fn prepare(&mut self, spec: ProcessSpec) -> Result<(), DspError> {
        if spec.sample_rate == 0 {
            return Err(DspError::UnsupportedSpec("sample rate must be non-zero"));
        }
        if !self.max_delay_ms.is_finite() || self.max_delay_ms < 1.0 {
            return Err(DspError::InvalidParam(
                "delay max_delay_ms must be finite and at least 1.0",
            ));
        }
        self.fs = f64::from(spec.sample_rate);
        let max_samples = (self.max_delay_ms * 1e-3 * self.fs).round().max(1.0) as usize;
        let ring_len = max_samples.checked_add(1).ok_or(DspError::InvalidParam(
            "delay max_delay_ms produces an unaddressable delay line",
        ))?;
        MemoryLayout::new()
            .repeated_array::<f64>(spec.channels, ring_len)
            .preflight(spec.max_memory)?;
        self.rings = vec![vec![0.0; ring_len]; spec.channels];
        self.pos = 0;
        self.reset_flush_state();
        Ok(())
    }

    fn reset(&mut self) {
        for ring in &mut self.rings {
            ring.fill(0.0);
        }
        self.pos = 0;
        self.reset_flush_state();
    }

    fn tail(&self) -> Tail {
        // A constant, conservative bound from the declared range maxima: delay
        // time can be automated up to the full ring capacity and feedback up to
        // MAX_FEEDBACK, so the last input can echo k_max() more times at the
        // longest delay before decaying below TAIL_FLOOR. Actual drains end far
        // earlier via the flush early exit.
        let ring = self.rings.first().map_or(0, Vec::len);
        Tail::Frames(tail_bound(ring))
    }

    fn io_mode(&self) -> IoMode {
        IoMode::InPlace
    }

    fn memory_footprint(&self) -> usize {
        self.rings.iter().map(Vec::len).sum::<usize>() * std::mem::size_of::<f64>()
    }

    fn param_info(&self) -> &[ParamInfo] {
        &self.params
    }

    fn render(&mut self, io: &mut SubBlock<'_, '_, '_, T>, params: &DelayParams) {
        // New input starts a new drain: the drained-frame counter resets.
        self.flushed = 0;
        let ring_len = self.rings[0].len();
        let max_delay = ring_len.saturating_sub(1);
        let d_target = ((params.delay_ms * 1e-3 * self.fs).round() as usize).clamp(1, max_delay);
        // Begin a crossfade toward a new integer tap only when settled. A
        // target that changes mid-fade waits for the active fade to finish
        // (re-checked at each fixed-parameter run), so at most one fade is in
        // flight and the blend never needs more than two read heads. Runs are
        // grid-anchored, so this decision is host-block-split invariant.
        if self.xfade_pos >= XFADE_FRAMES && d_target != self.tap_cur {
            self.tap_prev = self.tap_cur;
            self.tap_cur = d_target;
            self.xfade_pos = 0;
        }
        let feedback = params.feedback.clamp(0.0, MAX_FEEDBACK);
        let mix = params.mix.clamp(0.0, 1.0);
        let dry = 1.0 - mix;
        // Cache the last rendered values so flush can continue the recursion.
        self.flush_feedback = feedback;
        self.flush_mix = mix;
        let (d_new, d_old, fade_start) = (self.tap_cur, self.tap_prev, self.xfade_pos);
        let len = io.frames();
        let start = self.pos;
        let mut end = start;
        for (ch, ring) in self.rings.iter_mut().enumerate() {
            let mut p = start;
            for (i, slot) in io.channel_mut(ch).iter_mut().enumerate() {
                // The fade step for frame `i` is a pure function of the run's
                // starting fade position, so every channel replays the same
                // per-frame weights.
                let fade_step = fade_start.saturating_add(i).saturating_add(1);
                let delayed = read_delayed(ring, ring_len, p, d_new, d_old, fade_step);
                let x = finite_or_zero(slot.to_f64());
                ring[p] = flush_denormal(x + feedback * delayed);
                p = if p + 1 == ring_len { 0 } else { p + 1 };
                *slot = T::from_f64(flush_denormal(x * dry + delayed * mix));
            }
            // Every channel starts at `start` and advances one ring slot per
            // frame, so each iteration lands on the same end position.
            // Carrying the last channel's value is correct only because of
            // that per-channel identity.
            end = p;
        }
        self.pos = end;
        // The fade advances one step per rendered frame, for all channels at
        // once.
        self.xfade_pos = (fade_start + len).min(XFADE_FRAMES);
    }

    fn flush(&mut self, out: &mut AudioBlockMut<'_, '_, T>) -> Produced {
        let ring_len = self.rings.first().map_or(0, Vec::len);
        if ring_len == 0 {
            return Produced {
                frames: 0,
                done: true,
            };
        }
        // The declared worst-case bound; the decay scan below usually ends the
        // drain long before it is reached.
        let bound = tail_bound(ring_len);
        let state_peak = self
            .rings
            .iter()
            .flat_map(|ring| ring.iter())
            .fold(0.0f64, |m, &v| m.max(v.abs()));
        if self.flushed >= bound || state_peak < TAIL_FLOOR {
            self.flushed = bound;
            return Produced {
                frames: 0,
                done: true,
            };
        }
        let want = out.frames().min(bound.saturating_sub(self.flushed));
        let (d_new, d_old, fade_start) = (self.tap_cur, self.tap_prev, self.xfade_pos);
        let feedback = self.flush_feedback;
        let mix = self.flush_mix;
        let start = self.pos;
        let mut end = start;
        for (ch, ring) in self.rings.iter_mut().enumerate() {
            let buf = out.channel_mut(ch);
            let mut p = start;
            for (i, slot) in buf.iter_mut().take(want).enumerate() {
                // The recursion continues with silent input: no dry term. An
                // in-flight delay-time crossfade keeps fading during the
                // drain, so end-of-input mid-fade stays click-safe.
                let fade_step = fade_start.saturating_add(i).saturating_add(1);
                let delayed = read_delayed(ring, ring_len, p, d_new, d_old, fade_step);
                ring[p] = flush_denormal(feedback * delayed);
                p = if p + 1 == ring_len { 0 } else { p + 1 };
                *slot = T::from_f64(flush_denormal(delayed * mix));
            }
            // Every channel advances `want` slots from `start`; carrying the
            // last channel's end position relies on that identity.
            end = p;
        }
        self.pos = end;
        self.xfade_pos = (fade_start + want).min(XFADE_FRAMES);
        self.flushed += want;
        // Deterministic early exit: one O(channels * ring) scan per flush call
        // (not per frame). The tail is done once every ring value is below the
        // decay floor or the declared bound has been drained.
        let peak = self
            .rings
            .iter()
            .flat_map(|ring| ring.iter())
            .fold(0.0f64, |m, &v| m.max(v.abs()));
        let done = self.flushed >= bound || peak < TAIL_FLOOR;
        if done {
            self.flushed = bound;
        }
        Produced { frames: want, done }
    }
}

#[cfg(test)]
mod tests {
    use super::{
        k_max, tail_bound, Delay, DelayParams, DelaySettings, MAX_FEEDBACK, TAIL_FLOOR,
        TAIL_HEADROOM, XFADE_FRAMES,
    };
    use crate::processor::{AudioBlockMut, DspError, Io, Kernel, ProcessSpec, Produced, SubBlock};

    fn spec() -> ProcessSpec {
        ProcessSpec {
            sample_rate: 48_000,
            channels: 1,
            max_block: 32,
            max_memory: None,
        }
    }

    fn prepared_delay() -> Delay {
        let mut delay = Delay::with_settings(
            DelaySettings::new()
                .delay_ms(1.0)
                .feedback(0.0)
                .mix(1.0)
                .max_delay_ms(4.0),
        );
        Kernel::<f32>::prepare(&mut delay, spec()).expect("prepare");
        delay
    }

    fn flush(delay: &mut Delay, frames: usize) -> (Vec<f32>, Produced) {
        let mut samples = vec![0.0f32; frames];
        let produced = {
            let mut planes = [samples.as_mut_slice()];
            let mut out = AudioBlockMut::new(&mut planes);
            Kernel::<f32>::flush(delay, &mut out)
        };
        (samples, produced)
    }

    fn render_one(delay: &mut Delay, params: &DelayParams) -> f32 {
        let mut samples = [0.0f32; 1];
        {
            let mut planes = [samples.as_mut_slice()];
            let mut io = Io::InPlace(AudioBlockMut::new(&mut planes));
            let mut sub = SubBlock {
                io: &mut io,
                sc: &[],
                start: 0,
                len: 1,
            };
            Kernel::<f32>::render(delay, &mut sub, params);
        }
        samples[0]
    }

    /// The tail bound covers the documented retained-state headroom at maximum
    /// feedback and is computed through the vendored math wrappers.
    #[test]
    fn k_max_covers_the_declared_headroom() {
        assert_eq!(k_max(), 405);
        assert!(TAIL_HEADROOM * MAX_FEEDBACK.powi(k_max() as i32) < TAIL_FLOOR);
        assert!(TAIL_HEADROOM * MAX_FEEDBACK.powi(k_max() as i32 - 1) >= TAIL_FLOOR);
    }

    #[test]
    fn tail_bound_covers_sustained_full_scale_buildup() {
        let steady_state = 1.0 / (1.0 - MAX_FEEDBACK);
        assert!(steady_state * MAX_FEEDBACK.powi(k_max() as i32) < TAIL_FLOOR);
    }

    #[test]
    fn huge_delay_is_rejected_by_budget_before_ring_allocation() {
        let mut delay = Delay::with_settings(DelaySettings::new().max_delay_ms(1.0e12));
        let result = Kernel::<f32>::prepare(
            &mut delay,
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
    fn max_delay_validation_preserves_the_one_millisecond_boundary() {
        for bad in [0.5, f64::NAN] {
            let mut delay =
                Delay::with_settings(DelaySettings::new().delay_ms(1.0).max_delay_ms(bad));
            assert!(matches!(
                Kernel::<f32>::prepare(&mut delay, spec()),
                Err(DspError::InvalidParam(_))
            ));
        }

        let mut minimum =
            Delay::with_settings(DelaySettings::new().delay_ms(1.0).max_delay_ms(1.0));
        Kernel::<f32>::prepare(&mut minimum, spec()).expect("1 ms is the valid minimum");
    }

    #[test]
    fn settled_render_adopts_a_new_tap() {
        let mut delay = prepared_delay();
        delay.xfade_pos = XFADE_FRAMES;
        let params = DelayParams {
            delay_ms: 2.0,
            feedback: 0.0,
            mix: 1.0,
        };
        let _ = render_one(&mut delay, &params);
        assert_eq!(delay.tap_prev, 48);
        assert_eq!(delay.tap_cur, 96);
        assert_eq!(delay.xfade_pos, 1);
    }

    #[test]
    fn active_render_finishes_its_current_crossfade_first() {
        let mut delay = prepared_delay();
        let ring_len = delay.rings[0].len();
        delay.tap_prev = 48;
        delay.tap_cur = 96;
        delay.xfade_pos = 5;
        delay.rings[0][ring_len - delay.tap_prev] = 0.0;
        delay.rings[0][ring_len - delay.tap_cur] = 1.0;
        delay.rings[0][ring_len - 144] = 0.5;
        let params = DelayParams {
            delay_ms: 3.0,
            feedback: 0.0,
            mix: 1.0,
        };
        let sample = render_one(&mut delay, &params);
        assert_eq!(sample, 6.0 / XFADE_FRAMES as f32);
        assert_eq!(delay.tap_prev, 48);
        assert_eq!(delay.tap_cur, 96);
        assert_eq!(delay.xfade_pos, 6);
    }

    #[test]
    fn flush_reports_done_without_writing_when_state_is_below_floor() {
        let mut delay = prepared_delay();
        delay.rings[0][0] = TAIL_FLOOR * 0.5;
        let (_, produced) = flush(&mut delay, 4);
        assert_eq!(produced.frames, 0);
        assert!(produced.done);
    }

    #[test]
    fn flush_keeps_state_exactly_at_the_floor_live() {
        let mut delay = prepared_delay();
        delay.rings[0][1] = TAIL_FLOOR;
        let (_, produced) = flush(&mut delay, 1);
        assert_eq!(produced.frames, 1);
        assert!(!produced.done);
    }

    #[test]
    fn flush_detects_decay_below_floor_after_writing() {
        let mut delay = prepared_delay();
        delay.rings[0][0] = TAIL_FLOOR;
        let (_, produced) = flush(&mut delay, 1);
        assert_eq!(produced.frames, 1);
        assert!(produced.done);
    }

    #[test]
    fn flush_finishes_exactly_at_the_declared_bound_with_live_state() {
        let mut delay = prepared_delay();
        let bound = tail_bound(delay.rings[0].len());
        delay.flushed = bound - 1;
        delay.rings[0][1] = 1.0;
        let (_, produced) = flush(&mut delay, 1);
        assert_eq!(produced.frames, 1);
        assert!(produced.done);
        assert_eq!(delay.flushed, bound);
    }

    #[test]
    fn flush_advances_an_active_crossfade_by_exactly_the_written_frames() {
        let mut delay = prepared_delay();
        let ring_len = delay.rings[0].len();
        delay.tap_prev = 48;
        delay.tap_cur = 96;
        delay.xfade_pos = 5;
        delay.rings[0][ring_len - delay.tap_prev] = 0.0;
        delay.rings[0][ring_len - delay.tap_cur] = 1.0;
        let (samples, produced) = flush(&mut delay, 1);
        assert_eq!(produced.frames, 1);
        assert_eq!(samples[0], 6.0 / XFADE_FRAMES as f32);
        assert_eq!(delay.xfade_pos, 6);
    }

    #[test]
    fn flush_applies_the_latched_wet_mix_as_a_gain() {
        let mut delay = prepared_delay();
        let ring_len = delay.rings[0].len();
        delay.tap_prev = 1;
        delay.tap_cur = 1;
        delay.xfade_pos = XFADE_FRAMES;
        delay.flush_feedback = 0.0;
        delay.flush_mix = 0.5;
        delay.rings[0][ring_len - 1] = 0.25;

        let (samples, produced) = flush(&mut delay, 1);
        assert_eq!(produced.frames, 1);
        assert_eq!(samples[0], 0.125);
    }
}
