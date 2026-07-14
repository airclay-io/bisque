// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Airclay LLC

//! ITU-R BS.1770 loudness meter.

use std::f64::consts::{LN_10, PI};
use std::mem::size_of;

use crate::dsp::math;
use crate::dsp::memory::MemoryLayout;
use crate::dsp::sanitize::{finite_or_zero, flush_denormal};
use crate::processor::{AudioBlock, DspError, Measurer, ProcessSpec, Sample};

use super::{debug_validate_meter_geometry, PreparedMeterContract};

const MOMENTARY_MS: u32 = 400;
const SHORT_TERM_MS: u32 = 3_000;
const INTEGRATED_BLOCK_MS: u32 = 400;
const INTEGRATED_HOP_MS: u32 = 100;
const LOUDNESS_OFFSET_LU: f64 = -0.691;
const ABSOLUTE_GATE_LUFS: f64 = -70.0;
const RELATIVE_GATE_LU: f64 = 10.0;
const MIN_MAX_INTEGRATED_SECONDS: f64 = 0.4;
const MIN_K_WEIGHTING_SAMPLE_RATE: u32 = 3_364;
const SURROUND_CHANNEL_WEIGHT: f64 = 1.41;

/// Default integrated-loudness program duration.
///
/// The meter stores one 400 ms integrated-loudness block every 100 ms so
/// integrated LUFS can apply exact BS.1770 absolute and relative gating without
/// allocating during [`Measurer::observe`].
pub const DEFAULT_MAX_INTEGRATED_SECONDS: f64 = 3.0 * 60.0 * 60.0;

const K_SHELF_F0_HZ: f64 = 1_681.974_450_955_533;
const K_SHELF_GAIN_DB: f64 = 3.999_843_853_973_347;
const K_SHELF_Q: f64 = 0.707_175_236_955_419_6;
const K_SHELF_VB_POWER: f64 = 0.499_666_774_154_541_6;
const K_HIGHPASS_F0_HZ: f64 = 38.135_470_876_024_44;
const K_HIGHPASS_Q: f64 = 0.500_327_037_323_877_3;

/// Settings for [`LoudnessMeter`].
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct LoudnessMeterSettings {
    /// Per-channel contribution weights applied after K-weighting.
    ///
    /// An empty vector uses the meter's default weights for the prepared channel
    /// count when the channel count is unambiguous. Pass explicit weights for
    /// layouts with an LFE or non-standard channel order.
    pub channel_weights: Vec<f64>,
    /// Maximum program duration retained for exact integrated LUFS.
    ///
    /// Once this duration is exceeded, integrated loudness remains allocation
    /// free and [`LoudnessReading::integrated_complete`] becomes `false`.
    pub max_integrated_seconds: f64,
}

impl Default for LoudnessMeterSettings {
    fn default() -> Self {
        Self {
            channel_weights: Vec::new(),
            max_integrated_seconds: DEFAULT_MAX_INTEGRATED_SECONDS,
        }
    }
}

impl LoudnessMeterSettings {
    /// Default loudness meter settings.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Build settings with explicit per-channel weights.
    #[must_use]
    pub fn with_channel_weights(channel_weights: Vec<f64>) -> Self {
        Self {
            channel_weights,
            ..Self::default()
        }
    }

    /// Build settings with an explicit integrated-loudness duration.
    #[must_use]
    pub fn with_max_integrated_seconds(max_integrated_seconds: f64) -> Self {
        Self {
            max_integrated_seconds,
            ..Self::default()
        }
    }

    /// Set explicit per-channel weights.
    #[must_use]
    pub fn channel_weights(mut self, channel_weights: Vec<f64>) -> Self {
        self.channel_weights = channel_weights;
        self
    }

    /// Set the integrated-loudness history duration in seconds.
    #[must_use]
    pub fn max_integrated_seconds(mut self, max_integrated_seconds: f64) -> Self {
        self.max_integrated_seconds = max_integrated_seconds;
        self
    }

    /// Stereo weighting for left and right channels.
    #[must_use]
    pub fn stereo() -> Self {
        Self::with_channel_weights(vec![1.0, 1.0])
    }

    /// Five-channel weighting for left, right, center, left surround, and right
    /// surround.
    #[must_use]
    pub fn five_point_zero() -> Self {
        Self::with_channel_weights(vec![
            1.0,
            1.0,
            1.0,
            SURROUND_CHANNEL_WEIGHT,
            SURROUND_CHANNEL_WEIGHT,
        ])
    }

    /// Six-channel 5.1 weighting for left, right, center, LFE, left surround,
    /// and right surround.
    #[must_use]
    pub fn five_point_one() -> Self {
        Self::with_channel_weights(vec![
            1.0,
            1.0,
            1.0,
            0.0,
            SURROUND_CHANNEL_WEIGHT,
            SURROUND_CHANNEL_WEIGHT,
        ])
    }
}

/// Momentary, short-term, and integrated loudness in LUFS.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LoudnessReading {
    /// K-weighted loudness over the most recent 400 ms window.
    pub momentary_lufs: f64,
    /// K-weighted loudness over the most recent 3 second window.
    pub short_term_lufs: f64,
    /// Integrated loudness with BS.1770 absolute and relative gating.
    pub integrated_lufs: f64,
    /// Whether integrated loudness includes every integrated block since reset.
    pub integrated_complete: bool,
}

impl LoudnessReading {
    /// A reading for silence or for an empty meter.
    pub const SILENCE: Self = Self {
        momentary_lufs: f64::NEG_INFINITY,
        short_term_lufs: f64::NEG_INFINITY,
        integrated_lufs: f64::NEG_INFINITY,
        integrated_complete: true,
    };
}

/// ITU-R BS.1770 loudness meter.
///
/// The meter applies K-weighting, reports momentary and short-term loudness,
/// and stores one integrated-loudness energy value per 100 ms hop for exact
/// absolute and relative gating. Integrated history is bounded by
/// [`LoudnessMeterSettings::max_integrated_seconds`] so `observe` does not
/// allocate after `prepare`. Momentary and short-term readings are negative
/// infinity until their windows are full. Use explicit channel weights for
/// layouts where the default channel-order assumptions do not match the input.
/// `prepare` requires a sample rate of at least 3364 Hz so the K-weighting
/// shelf remains below Nyquist.
///
/// This type is intended for deterministic, allocation-free audio-path
/// metering. For offline conformance measurement, use a dedicated loudness
/// tool; see the [`analysis`](crate::analysis) module guidance.
#[derive(Debug, Clone)]
pub struct LoudnessMeter {
    settings: LoudnessMeterSettings,
    filters: Vec<KWeighting>,
    inferred_channel_weights: Vec<f64>,
    momentary: RollingEnergy,
    short_term: RollingEnergy,
    integrated_block: RollingEnergy,
    integrated_block_frames: usize,
    integrated_hop_frames: usize,
    integrated_block_capacity: usize,
    integrated_overflowed: bool,
    total_frames: u64,
    integrated_blocks: Vec<f64>,
    frame_energy: Vec<f64>,
    prepared: Option<PreparedMeterContract>,
}

impl LoudnessMeter {
    /// A loudness meter with inferred channel weights.
    #[must_use]
    pub fn new() -> Self {
        Self::with_settings(LoudnessMeterSettings::default())
    }

    /// A loudness meter with explicit settings.
    #[must_use]
    pub fn with_settings(settings: LoudnessMeterSettings) -> Self {
        Self {
            settings,
            filters: Vec::new(),
            inferred_channel_weights: Vec::new(),
            momentary: RollingEnergy::default(),
            short_term: RollingEnergy::default(),
            integrated_block: RollingEnergy::default(),
            integrated_block_frames: 1,
            integrated_hop_frames: 1,
            integrated_block_capacity: 1,
            integrated_overflowed: false,
            total_frames: 0,
            integrated_blocks: Vec::new(),
            frame_energy: Vec::new(),
            prepared: None,
        }
    }
}

impl Default for LoudnessMeter {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Sample> Measurer<T> for LoudnessMeter {
    type Reading = LoudnessReading;

    fn prepare(&mut self, spec: ProcessSpec) -> Result<(), DspError> {
        self.prepared = None;
        if spec.sample_rate < MIN_K_WEIGHTING_SAMPLE_RATE {
            return Err(DspError::UnsupportedSpec(
                "loudness meter sample rate is too low for K-weighting",
            ));
        }
        let channels = spec.channels;
        if channels == 0 {
            return Err(DspError::UnsupportedSpec(
                "loudness meter requires at least one channel",
            ));
        }

        // Validate the configured channel layout before computing or allocating
        // any owned buffers.
        validate_channel_weights(&self.settings, channels)?;
        let momentary_frames = frames_for_ms(spec.sample_rate, MOMENTARY_MS);
        let short_term_frames = frames_for_ms(spec.sample_rate, SHORT_TERM_MS);
        let integrated_block_frames = frames_for_ms(spec.sample_rate, INTEGRATED_BLOCK_MS);
        let integrated_hop_frames = frames_for_ms(spec.sample_rate, INTEGRATED_HOP_MS);
        let integrated_block_capacity =
            integrated_block_capacity(self.settings.max_integrated_seconds)?;

        loudness_memory_layout(
            channels,
            spec.max_block,
            momentary_frames,
            short_term_frames,
            integrated_block_frames,
            integrated_block_capacity,
        )
        .preflight(spec.max_memory)?;

        if self.settings.channel_weights.is_empty() {
            self.inferred_channel_weights =
                default_channel_weights(channels).ok_or(DspError::InvalidParam(
                    "explicit channel weights are required for this channel count",
                ))?;
        } else {
            self.inferred_channel_weights.clear();
        }
        self.integrated_block_frames = integrated_block_frames;
        self.integrated_hop_frames = integrated_hop_frames;
        self.integrated_block_capacity = integrated_block_capacity;
        self.filters = (0..channels)
            .map(|_| KWeighting::new(f64::from(spec.sample_rate)))
            .collect();
        self.momentary = RollingEnergy::new(momentary_frames);
        self.short_term = RollingEnergy::new(short_term_frames);
        self.integrated_block = RollingEnergy::new(self.integrated_block_frames);
        self.frame_energy = vec![0.0; spec.max_block];
        self.integrated_blocks = Vec::with_capacity(self.integrated_block_capacity);
        self.integrated_overflowed = false;
        self.total_frames = 0;
        self.prepared = Some(PreparedMeterContract::new(spec));
        Ok(())
    }

    fn reset(&mut self) {
        for filter in &mut self.filters {
            filter.reset();
        }
        self.momentary.reset();
        self.short_term.reset();
        self.integrated_block.reset();
        self.integrated_blocks.clear();
        self.integrated_overflowed = false;
        self.total_frames = 0;
    }

    fn memory_footprint(&self) -> usize {
        // The per-channel K-weighting filters plus every f64 buffer sized in
        // `prepare`: channel weights, the per-block frame-energy scratch, the
        // momentary, short-term, and integrated-block rings, and the reserved
        // integrated history. Matches the prepare-time budget estimate.
        self.filters.len() * size_of::<KWeighting>()
            + (self.channel_weights().len()
                + self.frame_energy.len()
                + self.momentary.ring.len()
                + self.short_term.ring.len()
                + self.integrated_block.ring.len()
                + self.integrated_block_capacity)
                * size_of::<f64>()
    }

    fn observe(&mut self, block: AudioBlock<'_, '_, T>) {
        debug_validate_meter_geometry(self.prepared.as_ref(), &block);
        let frames = block.frames();
        if frames == 0 {
            return;
        }

        {
            let frame_energy = &mut self.frame_energy[..frames];
            frame_energy.fill(0.0);

            let channel_weights = if self.settings.channel_weights.is_empty() {
                &self.inferred_channel_weights
            } else {
                &self.settings.channel_weights
            };
            for (ch, &weight) in channel_weights.iter().enumerate().take(block.channels()) {
                let filter = &mut self.filters[ch];
                if weight == 0.0 {
                    for &sample in block.channel(ch) {
                        let _ = filter.process(finite_or_zero(sample.to_f64()));
                    }
                    continue;
                }
                for (dst, &sample) in frame_energy.iter_mut().zip(block.channel(ch)) {
                    let weighted = filter.process(finite_or_zero(sample.to_f64()));
                    *dst += weight * weighted * weighted;
                }
            }
        }

        for frame in 0..frames {
            let energy = self.frame_energy[frame];
            self.push_weighted_energy(energy);
        }
    }

    fn read(&self) -> LoudnessReading {
        LoudnessReading {
            momentary_lufs: self
                .momentary
                .full_average()
                .map_or(f64::NEG_INFINITY, energy_to_loudness),
            short_term_lufs: self
                .short_term
                .full_average()
                .map_or(f64::NEG_INFINITY, energy_to_loudness),
            integrated_lufs: integrated_loudness(&self.integrated_blocks),
            integrated_complete: !self.integrated_overflowed,
        }
    }
}

impl LoudnessMeter {
    fn channel_weights(&self) -> &[f64] {
        if self.settings.channel_weights.is_empty() {
            &self.inferred_channel_weights
        } else {
            &self.settings.channel_weights
        }
    }

    fn push_weighted_energy(&mut self, energy: f64) {
        self.momentary.push(energy);
        self.short_term.push(energy);
        self.integrated_block.push(energy);
        self.total_frames += 1;

        let block_frames = self.integrated_block_frames as u64;
        let hop_frames = self.integrated_hop_frames as u64;
        if self.total_frames >= block_frames && (self.total_frames - block_frames) % hop_frames == 0
        {
            if let Some(block_energy) = self.integrated_block.average() {
                if self.integrated_blocks.len() < self.integrated_block_capacity {
                    self.integrated_blocks.push(block_energy);
                } else {
                    self.integrated_overflowed = true;
                }
            }
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct BiquadCoeffs {
    b0: f64,
    b1: f64,
    b2: f64,
    a1: f64,
    a2: f64,
}

#[derive(Debug, Clone, Copy)]
struct BiquadState {
    coeffs: BiquadCoeffs,
    z1: f64,
    z2: f64,
}

impl BiquadState {
    fn new(coeffs: BiquadCoeffs) -> Self {
        Self {
            coeffs,
            z1: 0.0,
            z2: 0.0,
        }
    }

    fn reset(&mut self) {
        self.z1 = 0.0;
        self.z2 = 0.0;
    }

    fn process(&mut self, x: f64) -> f64 {
        let y = self.coeffs.b0 * x + self.z1;
        self.z1 = self.coeffs.b1 * x - self.coeffs.a1 * y + self.z2;
        self.z2 = self.coeffs.b2 * x - self.coeffs.a2 * y;
        let y = flush_denormal(y);
        self.z1 = flush_denormal(self.z1);
        self.z2 = flush_denormal(self.z2);
        y
    }
}

#[derive(Debug, Clone, Copy)]
struct KWeighting {
    shelf: BiquadState,
    highpass: BiquadState,
}

impl KWeighting {
    fn new(sample_rate: f64) -> Self {
        Self {
            shelf: BiquadState::new(k_shelf_coeffs(sample_rate)),
            highpass: BiquadState::new(k_highpass_coeffs(sample_rate)),
        }
    }

    fn reset(&mut self) {
        self.shelf.reset();
        self.highpass.reset();
    }

    fn process(&mut self, x: f64) -> f64 {
        self.highpass.process(self.shelf.process(x))
    }
}

#[derive(Debug, Clone, Default)]
struct RollingEnergy {
    ring: Vec<f64>,
    pos: usize,
    len: usize,
    sum: f64,
    compensation: f64,
    updates_since_recompute: usize,
}

impl RollingEnergy {
    fn new(frames: usize) -> Self {
        Self {
            ring: vec![0.0; frames.max(1)],
            pos: 0,
            len: 0,
            sum: 0.0,
            compensation: 0.0,
            updates_since_recompute: 0,
        }
    }

    fn reset(&mut self) {
        self.ring.fill(0.0);
        self.pos = 0;
        self.len = 0;
        self.sum = 0.0;
        self.compensation = 0.0;
        self.updates_since_recompute = 0;
    }

    fn push(&mut self, energy: f64) {
        let energy = finite_or_zero(energy).max(0.0);
        if self.len < self.ring.len() {
            self.ring[self.pos] = energy;
            self.add_to_sum(energy);
            self.len += 1;
        } else {
            let old = self.ring[self.pos];
            self.ring[self.pos] = energy;
            self.add_to_sum(energy);
            self.add_to_sum(-old);
        }
        self.pos = if self.pos + 1 == self.ring.len() {
            0
        } else {
            self.pos + 1
        };
        self.updates_since_recompute += 1;
        if self.updates_since_recompute >= self.ring.len() {
            self.recompute_sum();
        }
    }

    fn average(&self) -> Option<f64> {
        if self.len == 0 {
            None
        } else {
            Some(self.corrected_sum() / self.len as f64)
        }
    }

    fn full_average(&self) -> Option<f64> {
        if self.len == self.ring.len() {
            self.average()
        } else {
            None
        }
    }

    fn add_to_sum(&mut self, value: f64) {
        let t = self.sum + value;
        if self.sum.abs() >= value.abs() {
            self.compensation += (self.sum - t) + value;
        } else {
            self.compensation += (value - t) + self.sum;
        }
        self.sum = t;
    }

    fn corrected_sum(&self) -> f64 {
        finite_or_zero(self.sum + self.compensation).max(0.0)
    }

    fn recompute_sum(&mut self) {
        self.sum = self.ring[..self.len].iter().copied().sum();
        self.compensation = 0.0;
        self.updates_since_recompute = 0;
    }
}

fn validate_channel_weights(
    settings: &LoudnessMeterSettings,
    channels: usize,
) -> Result<(), DspError> {
    if settings.channel_weights.is_empty() {
        if !(1..=5).contains(&channels) {
            return Err(DspError::InvalidParam(
                "loudness requires explicit channel weights for this channel count",
            ));
        }
    } else if settings.channel_weights.len() != channels {
        return Err(DspError::InvalidParam(
            "loudness channel weight count must match channel count",
        ));
    } else if settings
        .channel_weights
        .iter()
        .any(|weight| !weight.is_finite() || *weight < 0.0)
    {
        return Err(DspError::InvalidParam(
            "loudness channel weights must be finite and non-negative",
        ));
    }
    Ok(())
}

fn default_channel_weights(channels: usize) -> Option<Vec<f64>> {
    match channels {
        1 => Some(vec![1.0]),
        2 => Some(vec![1.0, 1.0]),
        3 => Some(vec![1.0, 1.0, 1.0]),
        4 => Some(vec![
            1.0,
            1.0,
            SURROUND_CHANNEL_WEIGHT,
            SURROUND_CHANNEL_WEIGHT,
        ]),
        5 => Some(vec![
            1.0,
            1.0,
            1.0,
            SURROUND_CHANNEL_WEIGHT,
            SURROUND_CHANNEL_WEIGHT,
        ]),
        _ => None,
    }
}

fn integrated_block_capacity(max_integrated_seconds: f64) -> Result<usize, DspError> {
    if !max_integrated_seconds.is_finite() || max_integrated_seconds < MIN_MAX_INTEGRATED_SECONDS {
        return Err(DspError::InvalidParam(
            "loudness max integrated seconds must be finite and at least 0.4",
        ));
    }

    let initial_hops = INTEGRATED_BLOCK_MS / INTEGRATED_HOP_MS - 1;
    let elapsed_hops = (max_integrated_seconds * 1_000.0 / f64::from(INTEGRATED_HOP_MS)).floor();
    let blocks = (elapsed_hops - f64::from(initial_hops)).max(1.0);
    // `usize::MAX as f64` rounds up on 64-bit targets, so equality is already
    // outside the exactly representable `usize` range.
    if blocks >= usize::MAX as f64 {
        return Err(DspError::InvalidParam(
            "loudness max integrated seconds is too large",
        ));
    }
    Ok(blocks.max(1.0) as usize)
}

fn loudness_memory_layout(
    channels: usize,
    max_block: usize,
    momentary_frames: usize,
    short_term_frames: usize,
    integrated_block_frames: usize,
    integrated_block_capacity: usize,
) -> MemoryLayout {
    MemoryLayout::new()
        .array::<KWeighting>(channels)
        .array::<f64>(channels) // channel weights
        .array::<f64>(max_block) // per-block frame energy
        .array::<f64>(momentary_frames)
        .array::<f64>(short_term_frames)
        .array::<f64>(integrated_block_frames)
        .array::<f64>(integrated_block_capacity)
}

fn frames_for_ms(sample_rate: u32, millis: u32) -> usize {
    let frames = (u64::from(sample_rate) * u64::from(millis) + 500) / 1_000;
    frames.max(1) as usize
}

fn k_shelf_coeffs(sample_rate: f64) -> BiquadCoeffs {
    let k = math::tan(PI * K_SHELF_F0_HZ / sample_rate);
    let vh = math::pow(10.0, K_SHELF_GAIN_DB / 20.0);
    let vb = math::pow(vh, K_SHELF_VB_POWER);
    let a0 = 1.0 + k / K_SHELF_Q + k * k;
    BiquadCoeffs {
        b0: (vh + vb * k / K_SHELF_Q + k * k) / a0,
        b1: 2.0 * (k * k - vh) / a0,
        b2: (vh - vb * k / K_SHELF_Q + k * k) / a0,
        a1: 2.0 * (k * k - 1.0) / a0,
        a2: (1.0 - k / K_SHELF_Q + k * k) / a0,
    }
}

fn k_highpass_coeffs(sample_rate: f64) -> BiquadCoeffs {
    let k = math::tan(PI * K_HIGHPASS_F0_HZ / sample_rate);
    let a0 = 1.0 + k / K_HIGHPASS_Q + k * k;
    BiquadCoeffs {
        b0: 1.0 / a0,
        b1: -2.0 / a0,
        b2: 1.0 / a0,
        a1: 2.0 * (k * k - 1.0) / a0,
        a2: (1.0 - k / K_HIGHPASS_Q + k * k) / a0,
    }
}

fn integrated_loudness(blocks: &[f64]) -> f64 {
    let absolute_threshold = loudness_to_energy(ABSOLUTE_GATE_LUFS);
    let (absolute_sum, absolute_count) = gated_energy(blocks, absolute_threshold);
    if absolute_count == 0 {
        return f64::NEG_INFINITY;
    }

    let absolute_loudness = energy_to_loudness(absolute_sum / absolute_count as f64);
    let relative_threshold = loudness_to_energy(absolute_loudness - RELATIVE_GATE_LU);
    let threshold = absolute_threshold.max(relative_threshold);
    let (gated_sum, gated_count) = gated_energy(blocks, threshold);
    if gated_count == 0 {
        f64::NEG_INFINITY
    } else {
        energy_to_loudness(gated_sum / gated_count as f64)
    }
}

fn gated_energy(blocks: &[f64], threshold: f64) -> (f64, usize) {
    let mut sum = 0.0;
    let mut count = 0;
    for &energy in blocks {
        let energy = finite_or_zero(energy).max(0.0);
        if energy > threshold {
            sum += energy;
            count += 1;
        }
    }
    (sum, count)
}

fn energy_to_loudness(energy: f64) -> f64 {
    let energy = finite_or_zero(energy);
    if energy <= 0.0 {
        f64::NEG_INFINITY
    } else {
        LOUDNESS_OFFSET_LU + 10.0 * (math::ln(energy) / LN_10)
    }
}

fn loudness_to_energy(lufs: f64) -> f64 {
    math::pow(10.0, (lufs - LOUDNESS_OFFSET_LU) / 10.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_close(got: f64, want: f64) {
        assert!((got - want).abs() < 1e-12, "got {got}, wanted {want}");
    }

    #[test]
    fn settings_builders_preserve_each_requested_value() {
        assert_eq!(DEFAULT_MAX_INTEGRATED_SECONDS, 10_800.0);

        let weights = vec![0.5, 1.5];
        let settings = LoudnessMeterSettings::new()
            .channel_weights(weights.clone())
            .max_integrated_seconds(12.5);
        assert_eq!(settings.channel_weights, weights);
        assert_eq!(settings.max_integrated_seconds, 12.5);

        assert_eq!(
            LoudnessMeterSettings::stereo().channel_weights,
            vec![1.0, 1.0]
        );
        assert_eq!(
            LoudnessMeterSettings::five_point_zero().channel_weights,
            vec![1.0, 1.0, 1.0, 1.41, 1.41]
        );
    }

    #[test]
    fn k_weighting_coefficients_match_reference_at_48k() {
        let shelf = k_shelf_coeffs(48_000.0);
        assert_close(shelf.b0, 1.535_124_859_586_97);
        assert_close(shelf.b1, -2.691_696_189_406_38);
        assert_close(shelf.b2, 1.198_392_810_852_85);
        assert_close(shelf.a1, -1.690_659_293_182_41);
        assert_close(shelf.a2, 0.732_480_774_215_85);

        let highpass = k_highpass_coeffs(48_000.0);
        assert_close(highpass.b0, 0.995_029_926_300_047);
        assert_close(highpass.b1, -1.990_059_852_600_09);
        assert_close(highpass.b2, 0.995_029_926_300_047);
        assert_close(highpass.a1, -1.990_047_454_833_98);
        assert_close(highpass.a2, 0.990_072_250_366_21);
    }

    #[test]
    fn default_channel_weights_cover_common_layouts() {
        assert_eq!(default_channel_weights(1), Some(vec![1.0]));
        assert_eq!(default_channel_weights(2), Some(vec![1.0, 1.0]));
        assert_eq!(default_channel_weights(3), Some(vec![1.0, 1.0, 1.0]));
        assert_eq!(default_channel_weights(4), Some(vec![1.0, 1.0, 1.41, 1.41]));
        assert_eq!(
            default_channel_weights(5),
            Some(vec![1.0, 1.0, 1.0, 1.41, 1.41])
        );
        assert_eq!(default_channel_weights(6), None);
    }

    #[test]
    fn rolling_energy_sanitizes_and_recomputes_sum() {
        let mut rolling = RollingEnergy::new(3);
        rolling.push(1.0);
        rolling.push(f64::NAN);
        rolling.push(3.0);
        assert_eq!(rolling.full_average(), Some(4.0 / 3.0));

        rolling.push(5.0);
        rolling.push(f64::INFINITY);
        rolling.push(7.0);
        assert_eq!(rolling.full_average(), Some(4.0));
        assert_eq!(rolling.updates_since_recompute, 0);
    }

    #[test]
    fn rolling_energy_tracks_update_cadence_and_compensated_sum() {
        let mut rolling = RollingEnergy::new(4);
        rolling.push(1.0);
        assert_eq!(rolling.updates_since_recompute, 1);

        let mut compensated = RollingEnergy::new(4);
        compensated.add_to_sum(1.0e16);
        compensated.add_to_sum(1.0);
        compensated.add_to_sum(-1.0e16);
        assert_eq!(compensated.corrected_sum(), 1.0);

        let mut larger_addend = RollingEnergy::new(4);
        larger_addend.add_to_sum(1.0);
        larger_addend.add_to_sum(1.0e16);
        larger_addend.add_to_sum(-1.0e16);
        assert_eq!(larger_addend.corrected_sum(), 1.0);
    }

    #[test]
    fn integrated_blocks_are_stored_on_the_declared_hop() {
        let mut meter = LoudnessMeter::new();
        meter.momentary = RollingEnergy::new(1);
        meter.short_term = RollingEnergy::new(1);
        meter.integrated_block = RollingEnergy::new(4);
        meter.integrated_block_frames = 4;
        meter.integrated_hop_frames = 3;
        meter.integrated_block_capacity = 4;

        for _ in 0..4 {
            meter.push_weighted_energy(1.0);
        }
        assert_eq!(meter.integrated_blocks, vec![1.0]);
        for _ in 0..2 {
            meter.push_weighted_energy(1.0);
        }
        assert_eq!(meter.integrated_blocks.len(), 1);
        meter.push_weighted_energy(1.0);
        assert_eq!(meter.integrated_blocks, vec![1.0, 1.0]);
    }

    fn assert_near(got: f64, want: f64, tolerance: f64) {
        assert!(
            (got - want).abs() <= tolerance,
            "got {got}, wanted {want} +/- {tolerance}"
        );
    }

    /// Squared magnitude of a biquad transfer function at angular frequency `w`,
    /// evaluated in closed form (independent of the time-domain recursion).
    fn biquad_mag_sq(c: &BiquadCoeffs, w: f64) -> f64 {
        let (c1, s1) = (math::cos(w), math::sin(w));
        let (c2, s2) = (math::cos(2.0 * w), math::sin(2.0 * w));
        let num_re = c.b0 + c.b1 * c1 + c.b2 * c2;
        let num_im = -(c.b1 * s1 + c.b2 * s2);
        let den_re = 1.0 + c.a1 * c1 + c.a2 * c2;
        let den_im = -(c.a1 * s1 + c.a2 * s2);
        (num_re * num_re + num_im * num_im) / (den_re * den_re + den_im * den_im)
    }

    /// Loudness the BS.1770 equation predicts for a steady `amp`-peak sine at
    /// `freq` in `channels` identical channels, from the closed-form K-weighting
    /// magnitude. A sine's mean square is `amp^2 / 2`.
    fn expected_tone_lufs(freq: f64, fs: f64, amp: f64, channels: usize) -> f64 {
        let w = 2.0 * PI * freq / fs;
        let k_gain_sq =
            biquad_mag_sq(&k_shelf_coeffs(fs), w) * biquad_mag_sq(&k_highpass_coeffs(fs), w);
        energy_to_loudness(channels as f64 * k_gain_sq * amp * amp / 2.0)
    }

    /// Drive the public meter over a steady tone and read it back.
    fn measure_tone(
        freq: f64,
        fs: u32,
        amp: f64,
        channels: usize,
        seconds: f64,
    ) -> LoudnessReading {
        let frames = (f64::from(fs) * seconds) as usize;
        let w = 2.0 * PI * freq / f64::from(fs);
        let wave: Vec<f32> = (0..frames)
            .map(|n| (amp * math::sin(w * n as f64)) as f32)
            .collect();
        let spec = ProcessSpec {
            sample_rate: fs,
            channels,
            max_block: 1024,
            max_memory: None,
        };
        let mut meter = LoudnessMeter::new();
        Measurer::<f32>::prepare(&mut meter, spec).expect("prepare");
        let mut start = 0;
        while start < frames {
            let len = 1024.min(frames - start);
            let slice = &wave[start..start + len];
            let planes: Vec<&[f32]> = (0..channels).map(|_| slice).collect();
            Measurer::<f32>::observe(&mut meter, AudioBlock::new(&planes));
            start += len;
        }
        Measurer::<f32>::read(&meter)
    }

    #[test]
    fn absolute_calibration_matches_bs1770_closed_form() {
        // EBU Tech 3341/3342 compliance is an absolute check. A steady tone must
        // read the loudness the BS.1770 equation predicts from the closed-form
        // K-weighting magnitude. The coefficients are pinned to the published
        // reference by `k_weighting_coefficients_match_reference_at_48k`, so
        // matching the closed form here pins the rest of the chain: filter
        // realization, channel summation, the -0.691 LU offset, and the gated
        // mean at two sample rates and across the K-weighting curve.
        for &(fs, freq, amp) in &[
            (48_000u32, 997.0, 0.5),
            (48_000, 1_000.0, 0.25),
            (48_000, 100.0, 0.5),
            (44_100, 997.0, 0.5),
        ] {
            let reading = measure_tone(freq, fs, amp, 2, 5.0);
            let expected = expected_tone_lufs(freq, f64::from(fs), amp, 2);
            assert_near(reading.momentary_lufs, expected, 0.1);
            assert_near(reading.short_term_lufs, expected, 0.1);
            assert_near(reading.integrated_lufs, expected, 0.1);
        }
    }

    /// `memory_footprint` equals the byte count derived from the allocation
    /// layout and matches the prepare-time budget estimate.
    #[test]
    fn footprint_is_the_exact_layout_byte_count() {
        let f = size_of::<f64>();
        let spec = ProcessSpec {
            sample_rate: 48_000,
            channels: 2,
            max_block: 1024,
            max_memory: None,
        };
        let seconds = 10.0;
        let mut meter = LoudnessMeter::with_settings(
            LoudnessMeterSettings::with_max_integrated_seconds(seconds),
        );
        Measurer::<f32>::prepare(&mut meter, spec).expect("prepare");

        // 400 ms, 3 s, and 400 ms windows at 48 kHz. A 10 second program
        // contains 97 complete 400 ms blocks on 100 ms hops.
        let (momentary, short_term, integrated_block) = (19_200, 144_000, 19_200);
        let capacity = 97;
        let ch = 2usize;
        let expected = ch * size_of::<KWeighting>()
            + (ch + 1024 + momentary + short_term + integrated_block + capacity) * f;
        assert_eq!(Measurer::<f32>::memory_footprint(&meter), expected);
        assert_eq!(
            expected,
            loudness_memory_layout(ch, 1024, momentary, short_term, integrated_block, capacity)
                .preflight(None)
                .expect("test layout is addressable"),
            "the reported footprint matches the prepare-time budget estimate"
        );
    }

    #[test]
    fn integrated_capacity_matches_complete_block_end_times() {
        assert_eq!(integrated_block_capacity(0.4), Ok(1));
        assert_eq!(integrated_block_capacity(0.5), Ok(2));
        assert_eq!(integrated_block_capacity(10.0), Ok(97));

        let first_unrepresentable = (usize::MAX as f64 + 3.0) * 0.1;
        assert!(matches!(
            integrated_block_capacity(first_unrepresentable),
            Err(DspError::InvalidParam(
                "loudness max integrated seconds is too large"
            ))
        ));
        assert!(matches!(
            integrated_block_capacity(f64::MAX),
            Err(DspError::InvalidParam(
                "loudness max integrated seconds is too large"
            ))
        ));
    }

    #[test]
    fn loudness_conversion_and_gating_preserve_their_boundaries() {
        for lufs in [-70.0, -23.0, 0.0, 12.0] {
            assert_close(energy_to_loudness(loudness_to_energy(lufs)), lufs);
        }
        assert_eq!(gated_energy(&[1.0, 2.0, 3.0], 2.0), (3.0, 1));
    }

    #[test]
    fn integrated_history_overflows_at_the_first_hop_after_the_limit() {
        let mut meter =
            LoudnessMeter::with_settings(LoudnessMeterSettings::with_max_integrated_seconds(0.5));
        let spec = ProcessSpec {
            sample_rate: 48_000,
            channels: 1,
            max_block: 4_800,
            max_memory: None,
        };
        Measurer::<f32>::prepare(&mut meter, spec).expect("prepare");
        let silence = vec![0.0f32; 4_800];
        for _ in 0..5 {
            Measurer::<f32>::observe(&mut meter, AudioBlock::new(&[&silence]));
        }
        assert!(Measurer::<f32>::read(&meter).integrated_complete);

        Measurer::<f32>::observe(&mut meter, AudioBlock::new(&[&silence]));
        assert!(!Measurer::<f32>::read(&meter).integrated_complete);
    }

    #[test]
    fn rlb_highpass_normalization_stays_within_ebu_tolerance() {
        // The RLB high-pass is normalized to unity passband (b0 = 1/a0) rather
        // than the literal published numerator [1, -2, 1]. The poles match the
        // reference, so only the numerator scale differs. Quantify the resulting
        // absolute-calibration shift and confirm it stays inside the EBU
        // Tech 3341 +/-0.1 LU tolerance.
        let fs = 48_000.0;
        let w = 2.0 * PI * 997.0 / fs;
        let normalized = biquad_mag_sq(&k_highpass_coeffs(fs), w);
        let published = biquad_mag_sq(
            &BiquadCoeffs {
                b0: 1.0,
                b1: -2.0,
                b2: 1.0,
                a1: -1.990_047_454_833_98,
                a2: 0.990_072_250_366_21,
            },
            w,
        );
        let delta_lu = 10.0 * math::ln(published / normalized) / LN_10;
        assert!(
            delta_lu > 0.0 && delta_lu < 0.1,
            "RLB normalization shifts calibration by {delta_lu} LU; must stay within EBU +/-0.1 LU"
        );
    }
}
