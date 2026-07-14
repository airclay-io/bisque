// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Airclay LLC

//! DC removal.

use std::f64::consts::TAU;

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

/// Headroom for each retained state value when computing the tail bound.
const TAIL_HEADROOM: f64 = 1e3;

/// The first silent recurrence can add the magnitudes of `x1` and `r * y1`.
const TAIL_INITIAL_HEADROOM: f64 = 2.0 * TAIL_HEADROOM;

/// Construction settings for [`DcBlocker`].
#[derive(Clone, Copy, Debug, PartialEq)]
#[non_exhaustive]
pub struct DcBlockerSettings {
    /// Cutoff frequency in Hz.
    pub cutoff_hz: f64,
}

impl Default for DcBlockerSettings {
    fn default() -> Self {
        Self { cutoff_hz: 20.0 }
    }
}

impl DcBlockerSettings {
    /// Default DC-blocker settings.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the cutoff frequency in Hz.
    #[must_use]
    pub fn cutoff_hz(mut self, cutoff_hz: f64) -> Self {
        self.cutoff_hz = cutoff_hz;
        self
    }
}

crate::params! {
    /// Smoothed parameter values for [`DcBlocker`].
    pub struct DcBlockerParams {
        /// Cutoff frequency in Hz.
        pub cutoff_hz => CUTOFF_HZ,
    }
}

/// One channel's DC-blocker state.
#[derive(Clone, Copy, Debug, Default)]
struct State {
    x1: f64,
    y1: f64,
}

/// An IIR DC blocker.
///
/// The recurrence is `y[n] = x[n] - x[n-1] + R * y[n-1]`, with
/// `R = exp(-2*pi*fc/fs)`. It has a zero at DC and near-unity gain in the audio
/// band. The cutoff parameter is automatable and recomputed once per sub-block.
/// Its effective value is clamped to `0.01..=0.49 * sample_rate`, so unusually
/// low sample rates can lower the declared range's effective upper limit.
///
/// `tail` reports a constant, conservative upper bound computed from the
/// declared range minimum: the slowest pole the cutoff range allows, with up to
/// `1e3` in each retained state value. The first silent recurrence can approach
/// `2e3`, so the bound budgets its decay below the `1e-6` floor. The bound is a
/// decay guarantee, not a silence guarantee: hotter (still finite) state decays
/// by the same ratio before `done`. Actual drains end far earlier:
/// `flush` continues the recursion with silent input, using the pole radius
/// last seen by `render`, and reports `done` as soon as every state value has
/// decayed below `1e-6`. New input starts a new drain; a host that wants a
/// shorter drain caps the frames it requests.
#[derive(Clone, Debug)]
pub struct DcBlocker {
    params: [ParamInfo; 1],
    fs: f64,
    state: Vec<State>,
    flush_r: f64,      // pole radius last seen by render
    tail_bound: usize, // declared worst-case tail frames, set in prepare
    flushed: usize,    // tail frames drained since the last process/reset
}

impl DcBlocker {
    /// Cutoff frequency in Hz.
    pub const CUTOFF_HZ: ParamId = DcBlockerParams::CUTOFF_HZ;

    /// A DC blocker configured from `settings`.
    #[must_use]
    pub fn with_settings(settings: DcBlockerSettings) -> Self {
        Self {
            params: [ParamInfo::new(
                Self::CUTOFF_HZ,
                "cutoff",
                (1.0, 1_000.0),
                settings.cutoff_hz,
                Unit::Hz,
            )],
            fs: 0.0,
            state: Vec::new(),
            flush_r: 0.0,
            tail_bound: 0,
            flushed: 0,
        }
    }

    /// A DC blocker with a 20 Hz cutoff.
    #[must_use]
    pub fn new() -> Self {
        Self::with_settings(DcBlockerSettings::default())
    }

    /// Restore the flush parameter cache to the `param_info` default and clear
    /// the drained-frame counter. Called from `prepare` and `reset`.
    fn reset_flush_state(&mut self) {
        self.flushed = 0;
        if self.fs > 0.0 {
            let fc = clamp_cutoff(finite_or_zero(self.params[0].default), self.fs);
            self.flush_r = math::exp(-TAU * fc / self.fs);
        }
    }
}

impl Default for DcBlocker {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Sample> Kernel<T> for DcBlocker {
    type Params = DcBlockerParams;

    fn prepare(&mut self, spec: ProcessSpec) -> Result<(), DspError> {
        if spec.sample_rate == 0 {
            return Err(DspError::UnsupportedSpec("sample rate must be non-zero"));
        }
        MemoryLayout::new()
            .array::<State>(spec.channels)
            .preflight(spec.max_memory)?;
        self.fs = f64::from(spec.sample_rate);
        self.state = vec![State::default(); spec.channels];
        // A constant, conservative tail bound from the declared range minimum:
        // the slowest reachable pole is at the lowest cutoff, and the drain
        // budgets the first silent recurrence from twice TAIL_HEADROOM down to
        // TAIL_FLOOR. Add one because the first emitted frame has not yet
        // received a recursive decay step.
        let fc_min = clamp_cutoff(self.params[0].range.0, self.fs);
        let decay_frames = (math::ln(TAIL_INITIAL_HEADROOM / TAIL_FLOOR) * self.fs / (TAU * fc_min))
            .ceil() as usize;
        self.tail_bound = decay_frames.saturating_add(1);
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

    fn render(&mut self, io: &mut SubBlock<'_, '_, '_, T>, params: &DcBlockerParams) {
        // New input starts a new drain: the drained-frame counter resets.
        self.flushed = 0;
        // Pole radius for this sub-block, cached so flush can continue the
        // recursion.
        let fc = clamp_cutoff(params.cutoff_hz, self.fs);
        let r = math::exp(-TAU * fc / self.fs);
        self.flush_r = r;
        for (ch, st) in self.state.iter_mut().enumerate() {
            for sample in io.channel_mut(ch).iter_mut() {
                let x = finite_or_zero(sample.to_f64());
                // Keep multiply and addition separate.
                let y = flush_denormal(x - st.x1 + r * st.y1);
                st.x1 = x;
                st.y1 = y;
                *sample = T::from_f64(y);
            }
        }
    }

    fn flush(&mut self, out: &mut AudioBlockMut<'_, '_, T>) -> Produced {
        let bound = self.tail_bound;
        let state_peak = self
            .state
            .iter()
            .fold(0.0f64, |m, s| m.max(s.x1.abs()).max(s.y1.abs()));
        if self.flushed >= bound || state_peak < TAIL_FLOOR {
            self.flushed = bound;
            return Produced {
                frames: 0,
                done: true,
            };
        }
        let want = out.frames().min(bound.saturating_sub(self.flushed));
        let r = self.flush_r;
        for (ch, st) in self.state.iter_mut().enumerate() {
            let buf = out.channel_mut(ch);
            for slot in buf.iter_mut().take(want) {
                // The recurrence continues with silent input.
                let y = flush_denormal(-st.x1 + r * st.y1);
                st.x1 = 0.0;
                st.y1 = y;
                *slot = T::from_f64(y);
            }
        }
        self.flushed += want;
        // Deterministic early exit: the tail is done once every state value is
        // below the decay floor or the declared bound has been drained.
        let peak = self
            .state
            .iter()
            .fold(0.0f64, |m, s| m.max(s.x1.abs()).max(s.y1.abs()));
        let done = self.flushed >= bound || peak < TAIL_FLOOR;
        if done {
            self.flushed = bound;
        }
        Produced { frames: want, done }
    }
}

/// Clamp a requested cutoff above DC and below Nyquist.
fn clamp_cutoff(cutoff_hz: f64, fs: f64) -> f64 {
    cutoff_hz.clamp(0.01, fs * 0.49)
}

#[cfg(test)]
mod tests {
    use super::clamp_cutoff;
    use crate::dsp::sanitize::{flush_denormal, DENORMAL_FLOOR};
    use crate::processor::{AudioBlockMut, Kernel, ProcessSpec, Produced};

    fn prepared_blocker() -> super::DcBlocker {
        let mut dc = super::DcBlocker::new();
        Kernel::<f32>::prepare(
            &mut dc,
            ProcessSpec {
                sample_rate: 48_000,
                channels: 1,
                max_block: 32,
                max_memory: None,
            },
        )
        .expect("prepare");
        dc
    }

    fn flush(dc: &mut super::DcBlocker, frames: usize) -> Produced {
        let mut samples = vec![0.0f32; frames];
        let mut planes = [samples.as_mut_slice()];
        let mut out = AudioBlockMut::new(&mut planes);
        Kernel::<f32>::flush(dc, &mut out)
    }

    #[test]
    fn clamp_cutoff_binds_both_ends_and_passes_the_middle() {
        // Clamp above Nyquist and below DC. Keep in-range values unchanged.
        assert_eq!(clamp_cutoff(1e9, 48_000.0), 48_000.0 * 0.49);
        assert_eq!(clamp_cutoff(0.0, 48_000.0), 0.01);
        assert_eq!(clamp_cutoff(20.0, 48_000.0), 20.0);
    }

    #[test]
    fn flush_done_decides_at_the_exact_decay_floor() {
        // A zero-frame flush only scans the state, so the done flag isolates
        // the floor comparison: state exactly at the floor still drains
        // (kills < -> <=), state below it is done.
        let mut dc = prepared_blocker();

        dc.state[0].y1 = super::TAIL_FLOOR;
        let mut planes: [&mut [f32]; 1] = [&mut []];
        let mut out = AudioBlockMut::new(&mut planes);
        assert!(
            !Kernel::<f32>::flush(&mut dc, &mut out).done,
            "state at the floor still drains"
        );

        dc.state[0].y1 = super::TAIL_FLOOR * 0.5;
        let mut planes: [&mut [f32]; 1] = [&mut []];
        let mut out = AudioBlockMut::new(&mut planes);
        assert!(
            Kernel::<f32>::flush(&mut dc, &mut out).done,
            "state below the floor is done"
        );
    }

    #[test]
    fn flush_writes_nothing_when_state_is_already_below_floor() {
        let mut dc = prepared_blocker();
        dc.state[0].y1 = super::TAIL_FLOOR * 0.5;
        let produced = flush(&mut dc, 4);
        assert_eq!(produced.frames, 0);
        assert!(produced.done);
    }

    #[test]
    fn flush_detects_state_that_decays_below_floor() {
        let mut dc = prepared_blocker();
        dc.state[0].y1 = super::TAIL_FLOOR;
        dc.flush_r = 0.0;
        let produced = flush(&mut dc, 1);
        assert_eq!(produced.frames, 1);
        assert!(produced.done);
    }

    #[test]
    fn flush_finishes_exactly_at_the_bound_with_live_state() {
        let mut dc = prepared_blocker();
        dc.flushed = dc.tail_bound - 1;
        dc.state[0].y1 = 1.0;
        dc.flush_r = 1.0;
        let produced = flush(&mut dc, 1);
        assert_eq!(produced.frames, 1);
        assert!(produced.done);
        assert_eq!(dc.flushed, dc.tail_bound);
    }

    #[test]
    fn flush_denormal_floors_below_but_keeps_at() {
        // Values below the floor are zeroed. Values at the floor are kept.
        assert_eq!(flush_denormal(1e-40), 0.0);
        assert_eq!(flush_denormal(-1e-40), 0.0);
        assert_eq!(flush_denormal(f64::NAN), 0.0);
        assert_eq!(flush_denormal(f64::INFINITY), 0.0);
        assert_eq!(flush_denormal(DENORMAL_FLOOR), DENORMAL_FLOOR);
        assert_eq!(flush_denormal(0.5), 0.5);
    }
}
