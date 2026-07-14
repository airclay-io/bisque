// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Airclay LLC

//! Streaming STFT brick-wall band filter.
//!
//! [`SpectralFilter`] buffers input to its hop, analyzes each frame, zeros bins
//! outside a frequency band, synthesizes the frame, overlap-adds, and emits output
//! as a host-driven [`Processor`].
//!
//! Latency and tail length are one FFT window. `flush` drains the tail. Streaming
//! reconstruction uses periodic steady-state window overlap from
//! [`window::cola_sum`].

use crate::processor::{
    AudioBlockMut, DspError, Io, IoMode, ProcessContext, ProcessSpec, Processor, Produced, Sample,
    Tail,
};
use crate::{
    dsp::driver::{debug_validate_flush_geometry, debug_validate_geometry, PreparedContract},
    dsp::memory::MemoryLayout,
    dsp::sanitize::finite_or_zero,
};

use super::fft::{Complex, Fft};
use super::window::{self, Window};

const MIN_OVERLAP_GAIN: f64 = 1e-12;

/// Construction settings for [`SpectralFilter`].
#[derive(Clone, Copy, Debug, PartialEq)]
#[non_exhaustive]
pub struct SpectralFilterSettings {
    /// FFT window size in samples.
    pub size: usize,
    /// Hop between analysis frames in samples.
    pub hop: usize,
    /// Analysis and synthesis window.
    pub window: Window,
    /// Lowest retained bin-center frequency in Hz.
    pub low_hz: f64,
    /// Highest retained bin-center frequency in Hz.
    pub high_hz: f64,
}

impl Default for SpectralFilterSettings {
    fn default() -> Self {
        Self {
            size: 1024,
            hop: 512,
            window: Window::Hann,
            low_hz: 0.0,
            high_hz: f64::INFINITY,
        }
    }
}

impl SpectralFilterSettings {
    /// Default full-band settings with a 1024-sample Hann window and 512-sample hop.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the FFT window size in samples.
    #[must_use]
    pub fn size(mut self, size: usize) -> Self {
        self.size = size;
        self
    }

    /// Set the analysis hop in samples.
    #[must_use]
    pub fn hop(mut self, hop: usize) -> Self {
        self.hop = hop;
        self
    }

    /// Set the analysis and synthesis window.
    #[must_use]
    pub fn window(mut self, window: Window) -> Self {
        self.window = window;
        self
    }

    /// Set the lowest retained bin-center frequency in Hz.
    #[must_use]
    pub fn low_hz(mut self, low_hz: f64) -> Self {
        self.low_hz = low_hz;
        self
    }

    /// Set the highest retained bin-center frequency in Hz.
    #[must_use]
    pub fn high_hz(mut self, high_hz: f64) -> Self {
        self.high_hz = high_hz;
        self
    }

    /// Retain bin-center frequencies in `[low_hz, high_hz]`.
    #[must_use]
    pub fn band(mut self, low_hz: f64, high_hz: f64) -> Self {
        self.low_hz = low_hz;
        self.high_hz = high_hz;
        self
    }
}

fn inverse_overlap_gain(gain: f64) -> f64 {
    debug_assert!(gain.is_finite() && gain > MIN_OVERLAP_GAIN);
    1.0 / gain
}

/// A streaming STFT brick-wall band filter.
///
/// Bins whose center falls outside `[low_hz, high_hz]` are zeroed each frame.
/// Events are ignored.
#[derive(Debug)]
pub struct SpectralFilter {
    // Configuration.
    size: usize,
    hop: usize,
    window: Window,
    low_hz: f64,
    high_hz: f64,

    // Prepared constants.
    channels: usize,
    fft: Fft,
    w_a: Vec<f64>,
    w_s: Vec<f64>,
    inv_wsum: Vec<f64>, // [hop] reciprocal periodic window overlap
    bin_lo: usize,      // first kept bin (inclusive)
    bin_hi: usize,      // last kept bin (inclusive)

    // Per-frame scratch.
    frame: Vec<f64>,             // [N] windowed time-domain / fft in+out
    spectrum: Vec<Complex<f64>>, // [N/2+1]

    // Per-channel persistent state.
    frame_in: Vec<Vec<f64>>, // [ch][N] analysis fill buffer
    acc: Vec<Vec<f64>>,      // [ch][N] overlap-add accumulator
    fifo: Vec<Vec<f64>>,     // [ch][N+hop] fixed ring of finalized outputs

    // Framing state shared by all channels.
    wpos: usize,      // fill cursor into frame_in. A frame fires at N
    fifo_head: usize, // ring read cursor
    fifo_len: usize,  // ring occupancy
    flushed: usize,   // tail frames drained since reset

    // Debug-time host-geometry contract (inline scalar state, not counted in
    // `memory_footprint`).
    prepared: Option<PreparedContract>,
}

impl SpectralFilter {
    /// A spectral filter configured from `settings`.
    ///
    /// The configured window is used for both analysis and synthesis. The configuration is
    /// stored as given: `prepare` rejects a size below two, a hop outside
    /// `1..=size`, incompatible window overlap, or invalid band edges with
    /// [`DspError::InvalidParam`] instead of silently adjusting them.
    #[must_use]
    pub fn with_settings(settings: SpectralFilterSettings) -> Self {
        Self {
            size: settings.size,
            hop: settings.hop,
            window: settings.window,
            low_hz: settings.low_hz,
            high_hz: settings.high_hz,
            channels: 0,
            fft: Fft::new(2), // real plan is built in `prepare`
            w_a: Vec::new(),
            w_s: Vec::new(),
            inv_wsum: Vec::new(),
            bin_lo: 0,
            bin_hi: 0,
            frame: Vec::new(),
            spectrum: Vec::new(),
            frame_in: Vec::new(),
            acc: Vec::new(),
            fifo: Vec::new(),
            wpos: 0,
            fifo_head: 0,
            fifo_len: 0,
            flushed: 0,
            prepared: None,
        }
    }

    /// A band filter using a Hann window.
    #[must_use]
    pub fn band(size: usize, hop: usize, low_hz: f64, high_hz: f64) -> Self {
        Self::with_settings(
            SpectralFilterSettings::new()
                .size(size)
                .hop(hop)
                .band(low_hz, high_hz),
        )
    }

    /// A low-pass filter keeping DC through `cutoff_hz`.
    #[must_use]
    pub fn low_pass(size: usize, hop: usize, cutoff_hz: f64) -> Self {
        Self::band(size, hop, 0.0, cutoff_hz)
    }

    /// A high-pass filter keeping `cutoff_hz` through Nyquist.
    #[must_use]
    pub fn high_pass(size: usize, hop: usize, cutoff_hz: f64) -> Self {
        Self::band(size, hop, cutoff_hz, f64::INFINITY)
    }

    /// The configured FFT size.
    #[must_use]
    pub fn size(&self) -> usize {
        self.size
    }

    /// The configured hop between analysis frames.
    #[must_use]
    pub fn hop(&self) -> usize {
        self.hop
    }

    /// Process one analysis frame for channel `ch` and push `hop` finalized
    /// samples into its FIFO at `push_base`.
    fn fire_channel(&mut self, ch: usize, push_base: usize) {
        let (n, h, cap) = (self.size, self.hop, self.size + self.hop);
        for i in 0..n {
            self.frame[i] = finite_or_zero(self.frame_in[ch][i]) * self.w_a[i];
        }
        self.fft.forward(&mut self.frame, &mut self.spectrum);
        for (b, s) in self.spectrum.iter_mut().enumerate() {
            if b < self.bin_lo || b > self.bin_hi {
                *s = Complex::new(0.0, 0.0);
            }
        }
        self.fft.inverse(&mut self.spectrum, &mut self.frame);
        for i in 0..n {
            self.acc[ch][i] += finite_or_zero(self.frame[i]) * self.w_s[i];
        }
        for k in 0..h {
            let y = finite_or_zero(self.acc[ch][k] * self.inv_wsum[k]);
            self.fifo[ch][(self.fifo_head + push_base + k) % cap] = y;
        }
        self.acc[ch].copy_within(h..n, 0);
        self.acc[ch][n - h..n].fill(0.0);
    }
}

impl Default for SpectralFilter {
    fn default() -> Self {
        Self::with_settings(SpectralFilterSettings::default())
    }
}

impl<T: Sample> Processor<T> for SpectralFilter {
    fn prepare(&mut self, spec: ProcessSpec) -> Result<(), DspError> {
        self.prepared = None;
        if spec.sample_rate == 0 {
            return Err(DspError::UnsupportedSpec("sample rate must be non-zero"));
        }
        if self.size < 2 {
            return Err(DspError::InvalidParam("STFT size must be at least 2"));
        }
        // Structural configuration is preserved as constructed and validated
        // here, not silently clamped.
        if self.hop == 0 || self.hop > self.size {
            return Err(DspError::InvalidParam("spectral hop must be in 1..=size"));
        }
        if !self.low_hz.is_finite() || self.low_hz < 0.0 {
            return Err(DspError::InvalidParam(
                "low_hz must be finite and non-negative",
            ));
        }
        // Positive infinity is the intentional unbounded edge used by
        // `high_pass`. Every other high edge must be finite.
        if !self.high_hz.is_finite() && self.high_hz != f64::INFINITY {
            return Err(DspError::InvalidParam(
                "high_hz must be finite or positive infinity",
            ));
        }
        if self.high_hz <= self.low_hz {
            return Err(DspError::InvalidParam("high_hz must exceed low_hz"));
        }
        let n = self.size;
        let h = self.hop;
        let bins = n / 2 + 1;
        let fifo_len = n.checked_add(h).ok_or(DspError::InvalidParam(
            "spectral size and hop exceed addressable memory",
        ))?;
        let fs = f64::from(spec.sample_rate);
        let channels = spec.channels;

        let state_layout = MemoryLayout::new()
            .array::<f64>(n) // frame
            .array::<Complex<f64>>(bins) // spectrum
            .array::<f64>(h) // inverse overlap sum
            .array::<f64>(n) // analysis window
            .array::<f64>(n) // synthesis window
            .repeated_array::<f64>(channels, n) // frame input
            .repeated_array::<f64>(channels, n) // OLA accumulator
            .repeated_array::<f64>(channels, fifo_len); // output FIFO

        // Reject oversized state before invoking the opaque backend
        // planner. Planning reveals the caller-owned scratch lengths; include
        // those in the final check before allocating any processor buffers.
        state_layout.preflight(spec.max_memory)?;

        let w_a = self.window.make(n);
        let w_s = self.window.make(n);
        let wprod: Vec<f64> = w_a.iter().zip(&w_s).map(|(a, s)| a * s).collect();
        let wsum = window::cola_sum(&wprod, h);
        if wsum
            .iter()
            .any(|&gain| !gain.is_finite() || gain <= MIN_OVERLAP_GAIN)
        {
            return Err(DspError::InvalidParam(
                "window and hop leave an unreconstructable output phase",
            ));
        }
        let inv_wsum = wsum
            .iter()
            .map(|&gain| inverse_overlap_gain(gain))
            .collect();

        let mut fft = Fft::plan(n);
        let (fwd_scratch, inv_scratch) = fft.scratch_lengths();
        state_layout
            .array::<Complex<f64>>(fwd_scratch)
            .array::<Complex<f64>>(inv_scratch)
            .preflight(spec.max_memory)?;
        fft.allocate_scratch();

        self.w_a = w_a;
        self.w_s = w_s;
        self.inv_wsum = inv_wsum;

        let hz_per_bin = fs / n as f64;
        self.bin_lo = (self.low_hz / hz_per_bin).ceil().max(0.0) as usize;
        self.bin_hi = ((self.high_hz / hz_per_bin).floor() as usize).min(bins - 1);

        self.frame = vec![0.0; n];
        self.spectrum = vec![Complex::new(0.0, 0.0); bins];
        self.frame_in = vec![vec![0.0; n]; channels];
        self.acc = vec![vec![0.0; n]; channels];
        self.fifo = vec![vec![0.0; fifo_len]; channels];
        self.fft = fft;
        self.channels = channels;

        self.wpos = 0;
        self.fifo_head = 0;
        self.fifo_len = 0;
        self.flushed = 0;

        // The pre-allocation estimate above matches the real layout, or
        // the budget boundary would drift from `memory_footprint()`.
        debug_assert!(
            spec.max_memory.is_none()
                || Processor::<T>::memory_footprint(self) <= spec.max_memory.unwrap_or(usize::MAX),
            "pre-allocation budget estimate must cover the real layout"
        );
        self.prepared = Some(PreparedContract {
            max_block: spec.max_block,
            channels: self.channels,
            io_mode: IoMode::Split,
            sidechain_inputs: 0,
        });
        Ok(())
    }

    fn reset(&mut self) {
        for c in &mut self.frame_in {
            c.fill(0.0);
        }
        for c in &mut self.acc {
            c.fill(0.0);
        }
        for c in &mut self.fifo {
            c.fill(0.0);
        }
        self.wpos = 0;
        self.fifo_head = 0;
        self.fifo_len = 0;
        self.flushed = 0;
    }

    fn latency(&self) -> usize {
        self.size
    }

    fn tail(&self) -> Tail {
        Tail::Frames(self.size)
    }

    fn io_mode(&self) -> IoMode {
        IoMode::Split
    }

    fn memory_footprint(&self) -> usize {
        let f = std::mem::size_of::<f64>();
        let c = std::mem::size_of::<Complex<f64>>();
        self.frame.len() * f
            + self.spectrum.len() * c
            + self.fft.scratch_footprint()
            + self.inv_wsum.len() * f
            + self.w_a.len() * f
            + self.w_s.len() * f
            + self.frame_in.iter().map(|v| v.len() * f).sum::<usize>()
            + self.acc.iter().map(|v| v.len() * f).sum::<usize>()
            + self.fifo.iter().map(|v| v.len() * f).sum::<usize>()
    }

    fn process(&mut self, ctx: &mut ProcessContext<'_, '_, T>) {
        debug_validate_geometry(self.prepared.as_ref(), ctx);
        let frames = ctx.frames;
        if frames == 0 {
            return;
        }
        // New input starts a new drain: the drained-frame counter resets.
        self.flushed = 0;
        let (n, h, cap) = (self.size, self.hop, self.size + self.hop);
        let nch = self.channels;
        let Io::Split { input, output } = &mut ctx.main else {
            debug_assert!(false, "SpectralFilter declared Split I/O");
            return;
        };
        for i in 0..frames {
            // Pop finalized output before ingesting the next input sample.
            if self.fifo_len > 0 {
                for ch in 0..nch {
                    output.channel_mut(ch)[i] =
                        T::from_f64(finite_or_zero(self.fifo[ch][self.fifo_head]));
                }
                self.fifo_head = (self.fifo_head + 1) % cap;
                self.fifo_len -= 1;
            } else {
                for ch in 0..nch {
                    output.channel_mut(ch)[i] = T::from_f64(0.0);
                }
            }
            // Ingest one input sample per channel.
            for ch in 0..nch {
                self.frame_in[ch][self.wpos] = finite_or_zero(input.channel(ch)[i].to_f64());
            }
            self.wpos += 1;
            // Fire a frame when the analysis window is full.
            if self.wpos == n {
                let push_base = self.fifo_len;
                for ch in 0..nch {
                    self.fire_channel(ch, push_base);
                }
                self.fifo_len += h;
                for ch in 0..nch {
                    self.frame_in[ch].copy_within(h..n, 0);
                }
                self.wpos = n - h;
            }
        }
    }

    fn flush(&mut self, out: &mut AudioBlockMut<'_, '_, T>) -> Produced {
        debug_validate_flush_geometry(self.prepared.as_ref(), out);
        let (n, h, cap) = (self.size, self.hop, self.size + self.hop);
        let nch = self.channels;
        // The tail is N frames of in-flight reconstruction.
        let want = out.frames().min(n - self.flushed);
        for i in 0..want {
            if self.fifo_len > 0 {
                for ch in 0..nch {
                    out.channel_mut(ch)[i] =
                        T::from_f64(finite_or_zero(self.fifo[ch][self.fifo_head]));
                }
                self.fifo_head = (self.fifo_head + 1) % cap;
                self.fifo_len -= 1;
            } else {
                for ch in 0..nch {
                    out.channel_mut(ch)[i] = T::from_f64(0.0);
                }
            }
            // Zero-pad input after end of stream.
            for ch in 0..nch {
                self.frame_in[ch][self.wpos] = 0.0;
            }
            self.wpos += 1;
            if self.wpos == n {
                let push_base = self.fifo_len;
                for ch in 0..nch {
                    self.fire_channel(ch, push_base);
                }
                self.fifo_len += h;
                for ch in 0..nch {
                    self.frame_in[ch].copy_within(h..n, 0);
                }
                self.wpos = n - h;
            }
        }
        self.flushed += want;
        Produced {
            frames: want,
            done: self.flushed >= n,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{inverse_overlap_gain, SpectralFilter, MIN_OVERLAP_GAIN};
    use crate::processor::{ProcessSpec, Processor};

    #[test]
    fn reported_footprint_includes_fft_scratch() {
        let base_spec = ProcessSpec {
            sample_rate: 48_000,
            channels: 2,
            max_block: 512,
            max_memory: None,
        };
        let mut prepared = SpectralFilter::low_pass(1014, 507, 8_000.0);
        Processor::<f32>::prepare(&mut prepared, base_spec).expect("prepare reference layout");
        let total = Processor::<f32>::memory_footprint(&prepared);
        let scratch = prepared.fft.scratch_footprint();
        let f = std::mem::size_of::<f64>();
        let c = std::mem::size_of::<super::Complex<f64>>();
        let non_fft = prepared.frame.len() * f
            + prepared.spectrum.len() * c
            + prepared.inv_wsum.len() * f
            + prepared.w_a.len() * f
            + prepared.w_s.len() * f
            + prepared.frame_in.iter().map(|v| v.len() * f).sum::<usize>()
            + prepared.acc.iter().map(|v| v.len() * f).sum::<usize>()
            + prepared.fifo.iter().map(|v| v.len() * f).sum::<usize>();
        assert_eq!(
            total,
            non_fft + scratch,
            "the processor footprint must add its FFT's owned scratch vectors"
        );
    }

    #[test]
    fn inverse_overlap_gain_returns_the_reciprocal_above_the_floor() {
        let gain = MIN_OVERLAP_GAIN * 2.0;
        assert_eq!(inverse_overlap_gain(gain), 1.0 / gain);
    }
}
