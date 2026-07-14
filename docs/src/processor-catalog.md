<!-- SPDX-License-Identifier: Apache-2.0 -->

# Processor Catalog

This page summarizes the public processors and utilities currently implemented.
All same-rate processors use `f32` or `f64` sample storage through the `Sample`
trait. Most built-in effects implement `Kernel` and can be wrapped with
`into_processor()`.

## Filters

Feature `filters`

| Type | Trait | I/O | Notes |
| --- | --- | --- | --- |
| `Biquad` | `Kernel` | In-place | RBJ low-pass, high-pass, low shelf, high shelf, and peaking shapes |
| `BiquadCoeffs` | Utility | None | Response readouts for magnitude, phase, group delay, and stability |
| `MovingAverage` | `Kernel` | Split | FIR moving average with whole-frame latency `floor((taps - 1) / 2)` |

`Biquad` parameters

| Constant | Name | Range | Unit | Shapes |
| --- | --- | --- | --- | --- |
| `Biquad::CUTOFF_HZ` | `cutoff` | `10.0..=24000.0` | Hz | All shapes |
| `Biquad::Q` | `q` | `0.1..=16.0` | Q | All shapes |
| `Biquad::GAIN_DB` | `gain` | `-24.0..=24.0` | dB | Low shelf, high shelf, peaking |

The declared cutoff range is host metadata shared across sample rates. During
rendering, the effective cutoff is clamped to `1.0..=0.999 * Nyquist` so the
coefficient calculation remains inside the representable audio band. At sample
rates below approximately 48.05 kHz, part of the declared upper range therefore
maps to that sample-rate-dependent ceiling.
Q controls resonance for low-pass, high-pass, and peaking shapes. For shelf
shapes it controls transition steepness and possible resonance.

`BiquadCoeffs::try_rbj` checks raw coefficient inputs. A prepared
`Biquad::try_coeffs` readout clamps cutoff, Q, and gain exactly as rendering does.

`Biquad` reports a finite tail. The bound is constant and conservative,
computed from
the declared range extremes (the slowest pole pair the cutoff, Q, and gain
ranges allow) at a decay floor of `1e-6`. `flush` continues the recursion on
silent input with the last-rendered coefficients and reports `done` early once
the state has decayed below the floor, so real drains end far sooner
than the declared bound.

`MovingAverage` reports a tail of exactly `taps - 1` frames, the FIR response
of the final inputs. `flush` continues the convolution with silent input, so
process-plus-flush reconstructs the complete causal response. Odd tap counts
align to the reported integer latency. Even tap counts have a half-frame
residual because the host latency contract uses whole frames.
`group_delay_frames` reports the exact fractional delay. A compensated running
sum gives amortized constant work per sample and is recomputed at every ring
wrap to bound drift. The tap count is stored as constructed; `prepare` rejects
zero taps.

Constructors

```rust
# extern crate bisque;
use bisque::filters::{Biquad, BiquadSettings, MovingAverage};

let lowpass = Biquad::lowpass();
let peaking = Biquad::with_settings(BiquadSettings::peaking());
let average = MovingAverage::new(16);
```

## Dynamics

Feature `dynamics`

| Type | Trait | I/O | Notes |
| --- | --- | --- | --- |
| `Compressor` | `Kernel` | In-place | Peak-detected hard-knee downward compressor |
| `Expander` | `Kernel` | In-place | Peak-detected hard-knee downward expander |
| `Gate` | `Kernel` | In-place | Peak-detected hard-knee gate with floor range |

All dynamics processors use linked peak detection with attack and release
ballistics on the detected level before evaluating the static curve. They do not
provide RMS detection, soft knee, program-dependent release, or gain-reduction
smoothing. Sidechain detection is optional at construction.

`Compressor` parameters

| Constant | Name | Range | Unit |
| --- | --- | --- | --- |
| `Compressor::THRESHOLD_DB` | `threshold` | `-60.0..=0.0` | dB |
| `Compressor::RATIO` | `ratio` | `1.0..=20.0` | Linear |
| `Compressor::MAKEUP_DB` | `makeup` | `0.0..=24.0` | dB |

`Expander` parameters

| Constant | Name | Range | Unit |
| --- | --- | --- | --- |
| `Expander::THRESHOLD_DB` | `threshold` | `-80.0..=0.0` | dB |
| `Expander::RATIO` | `ratio` | `1.0..=20.0` | Linear |

`Gate` parameters

| Constant | Name | Range | Unit |
| --- | --- | --- | --- |
| `Gate::THRESHOLD_DB` | `threshold` | `-80.0..=0.0` | dB |
| `Gate::RATIO` | `ratio` | `1.0..=20.0` | Linear |
| `Gate::RANGE_DB` | `range` | `-120.0..=0.0` | dB |

Constructors

```rust
# extern crate bisque;
use bisque::dynamics::{Compressor, CompressorSettings, Expander, Gate};

let compressor = Compressor::new();
let keyed_compressor = Compressor::with_sidechain();
let fast_compressor = Compressor::with_settings(
    CompressorSettings::new().attack_ms(1.0).release_ms(40.0)
);
let expander = Expander::new();
let keyed_expander = Expander::with_sidechain();
let gate = Gate::new();
let keyed_gate = Gate::with_sidechain();
```

The dynamics settings structs include `use_sidechain` for keyed detection.
Positive attack and release values are one-pole time constants. Zero applies
the detected level immediately. For the expander and gate, a ratio of 2.0 moves
a level 10 dB below threshold toward 20 dB below threshold. Gate `range_db` is
the minimum gain below threshold and uses a negative value from -120.0 to 0.0 dB.

## Mastering

Feature `mastering`

| Type | Trait | I/O | Notes |
| --- | --- | --- | --- |
| `Gain` | `Kernel` | In-place | Automatable gain in dB |
| `Scale` | `Kernel` | In-place | Fixed unbounded linear factor (dB, linear, or polarity) |
| `Dither` | `Kernel` | In-place | Seeded TPDF dither and mid-tread quantizer |
| `Limiter` | `Kernel` | In-place | Lookahead true-peak limiter with sample-rate gain |

`Scale` has no automatable parameters. The factor is fixed at construction with
`Scale::new` (linear), `Scale::from_db` (dB), or `Scale::inverted` (polarity), and
is unbounded except that `prepare` rejects a non-finite resolved factor.

`Gain` parameters

| Constant | Name | Range | Unit |
| --- | --- | --- | --- |
| `Gain::GAIN_DB` | `gain` | `-96.0..=24.0` | dB |

`Limiter` parameters

| Constant | Name | Range | Unit |
| --- | --- | --- | --- |
| `Limiter::THRESHOLD_DB` | `threshold` | `-30.0..=0.0` | dB |

`Dither` has no automatable parameters. It is configured by bit depth and seed
through `DitherSettings` (default: 16 bits, fixed default seed). `prepare`
validates bit depth in `2..=24`.

Constructors

```rust
# extern crate bisque;
use bisque::mastering::{
    Dither, DitherSettings, Gain, GainSettings, Limiter, LimiterSettings, Scale,
};

let gain = Gain::new();
let quiet = Gain::with_settings(GainSettings::new().gain_db(-6.0));
let scale = Scale::from_db(-6.0); // or Scale::new(0.5), or Scale::inverted()
let dither = Dither::new(); // 16-bit, default seed
let seeded_dither = Dither::with_settings(DitherSettings::new().bits(16).seed(0x1234));
let limiter = Limiter::new();
let custom_limiter = Limiter::with_settings(
    LimiterSettings::new()
        .threshold_db(-1.0)
        .true_peak_margin_db(0.1)
        .lookahead_ms(1.5)
        .release_ms(50.0)
);
```

`Limiter` latency and tail length are both equal to its lookahead. The gain
trajectory is a sliding minimum over the lookahead window smoothed with a
normalized Hann attack curve precomputed in `prepare`. The
attack uses the lookahead remaining after the true-peak detector tail. An exact
transparency fast path keeps below-threshold passthrough bit-exact.
The lookahead must round to at least 11 frames so the delay retains the complete
true-peak detector response. A zero release applies release immediately.
Detection uses the maximum of sample peak and oversampled true peak, but the
oversampler's small FIR group delay is not compensated relative to the
sample-domain peak detector. Gain is applied at sample rate, so fast gain
changes and difficult transients can still reconstruct above the detector
target. The real-time detector is not an offline conformance meter.
`LimiterSettings::true_peak_margin_db` subtracts a safety margin from the
internal gain target. The default margin is `0.1` dB.

`flush` drains the delay line while the true-peak detector keeps running over
silent input with the gain target last seen by `render`. An inter-sample peak
within the detector's group delay of end-of-input is only reported during the
drain, while the samples that produced it are still in the delay line, so the
drained tail follows the same detector target as the body.

## Analysis

Feature `analysis`

Meters implement `Measurer`, not `Processor`.

| Type | Trait | Reading | Notes |
| --- | --- | --- | --- |
| `PeakMeter` | `Measurer` | `f64` | Maximum absolute sample since reset |
| `RmsMeter` | `Measurer` | `f64` | RMS over all samples observed since reset |
| `MeanMeter` | `Measurer` | `f64` | Max absolute per-channel mean; signed per channel via `channel_mean` |
| `CrestMeter` | `Measurer` | `f64` | Peak divided by RMS |
| `TruePeakMeter` | `Measurer` | `f64` | Oversampled inter-sample peak estimate |
| `WindowedRmsMeter` | `Measurer` | `f64` | RMS over the most recent window |
| `LoudnessMeter` | `Measurer` | `LoudnessReading` | BS.1770 momentary, short-term, and integrated LUFS |
| `ClipMeter` | `Measurer` | `u64` | Count of samples at or above threshold |

Meters report `latency` and `memory_footprint` alongside their readings. Most
meters have zero latency. `TruePeakMeter`'s reading lags the newest observed
input by its oversampling FIR group delay (6 input frames).

Use `linear_to_dbfs` to convert linear readings to dBFS.
`WindowedRmsMeter` is configured with `WindowedRmsMeterSettings` (window length
in frames, default 512) and `ClipMeter` with `ClipMeterSettings` (linear
threshold, default 1.0). Preparation rejects a zero window or a clip threshold
that is not finite and positive.
`LoudnessMeter` applies K-weighting, reports 400 ms momentary loudness and
3 second short-term loudness once those windows are full, and computes
integrated loudness with absolute and relative gating. It stores one
integrated-history value per 100 ms hop up to
`LoudnessMeterSettings::max_integrated_seconds`. If that history fills,
`LoudnessReading::integrated_complete` becomes `false`.
K-weighting requires a sample rate of at least 3364 Hz.

For offline file measurement, use a full loudness tool. bisque uses the
`ebur128` crate as the conformance reference in tests. bisque's loudness meter
is for audio-path metering with allocation-free observe, block-size invariance,
and deterministic output.

Default channel weights are inferred only for unambiguous mono, stereo, 3.0,
4.0, and 5.0 layouts. Use `LoudnessMeterSettings::five_point_one()` or explicit
`channel_weights` for 5.1, LFE, or non-standard channel orders.

```rust
# extern crate bisque;
# let left = [0.0_f32; 512];
# let right = [0.0_f32; 512];
use bisque::analysis::{linear_to_dbfs, PeakMeter};
use bisque::processor::{AudioBlock, Measurer, ProcessSpec};

let mut meter = PeakMeter::new();
Measurer::<f32>::prepare(&mut meter, ProcessSpec {
    sample_rate: 48_000,
    channels: 2,
    max_block: 512,
    max_memory: None,
}).expect("prepare");

let planes: [&[f32]; 2] = [&left, &right];
Measurer::<f32>::observe(&mut meter, AudioBlock::new(&planes));
let peak_dbfs = linear_to_dbfs(Measurer::<f32>::read(&meter));
```

## Generators

Feature `generators`

Generators declare output-only I/O and do not require a main input signal. Use
`Kernel::into_processor()` when you want framework parameter smoothing and
block driving.

| Type | Trait | I/O | Notes |
| --- | --- | --- | --- |
| `SineOsc` | `Kernel` | Output-only | Sine oscillator |
| `WhiteNoise` | `Kernel` | Output-only | Seeded per-channel white noise |
| `PolyBlepOsc` | `Kernel` | Output-only | Alias-reduced PolyBLEP saw or square oscillator |
| `Waveform` | Enum | None | `Saw` or `Square` |

Oscillator parameters

| Constant | Name | Range | Unit | Types |
| --- | --- | --- | --- | --- |
| `SineOsc::FREQUENCY_HZ`, `PolyBlepOsc::FREQUENCY_HZ` | `frequency` | `1.0..=24000.0` | Hz | `SineOsc`, `PolyBlepOsc` |
| `SineOsc::AMPLITUDE`, `PolyBlepOsc::AMPLITUDE` | `amplitude` | `0.0..=1.0` | Linear | `SineOsc`, `PolyBlepOsc` |

`WhiteNoise` parameters

| Constant | Name | Range | Unit |
| --- | --- | --- | --- |
| `WhiteNoise::AMPLITUDE` | `amplitude` | `0.0..=1.0` | Linear |

The declared oscillator frequency range is shared across sample rates. At
render time, `SineOsc` and `PolyBlepOsc` clamp the effective frequency to
`0.999 * Nyquist`. The guarded ceiling keeps the maximum setting audible while
preventing an out-of-band oscillator at sample rates below 48 kHz.

Constructors

```rust
# extern crate bisque;
use bisque::generators::{
    PolyBlepOsc, PolyBlepOscSettings, SineOsc, Waveform, WhiteNoise,
    WhiteNoiseSettings,
};

let sine = SineOsc::new();
let noise = WhiteNoise::with_settings(WhiteNoiseSettings::new().amplitude(0.25).seed(42));
let saw = PolyBlepOsc::new(); // default waveform: saw at 440 Hz
let also_saw = PolyBlepOsc::saw();
let square = PolyBlepOsc::with_settings(
    PolyBlepOscSettings::new()
        .waveform(Waveform::Square)
        .frequency_hz(110.0)
        .amplitude(0.5)
);
```

## Time

Feature `time`

| Type | Trait | I/O | Notes |
| --- | --- | --- | --- |
| `Delay` | `Kernel` | In-place | Integer-sample feedback delay with wet/dry mix |
| `TimeStretch` | `VariableRate` | Pull source | Plain overlap-add time stretch |

`Delay` parameters

| Constant | Name | Range | Unit |
| --- | --- | --- | --- |
| `Delay::DELAY_MS` | `delay` | `1.0..=max_delay_ms` | ms |
| `Delay::FEEDBACK` | `feedback` | `0.0..=0.95` | Linear |
| `Delay::MIX` | `mix` | `0.0..=1.0` | Linear |

`Delay` rounds delay time to whole samples and does not interpolate fractional
positions: at settled values the line reads one integer tap, exactly as before
any automation. A delay-time change is click-safe: the wet path crossfades
from the old integer tap to the new one over 32 frames (one control-rate cell)
with equal-gain linear weights, so each frame moves the wet blend by at most
1/32 of the tap difference. While a fade is active, later target changes wait
for it to finish, so continuous automation advances in bounded 32-frame
transitions and may skip intermediate integer taps. The feedback path is clean
feedback without damping filters or saturation.

`Delay` settings are stored as constructed: `prepare` rejects a non-finite or
sub-1 ms `max_delay_ms`, and a `delay_ms`, `feedback`, or `mix` outside its
declared range, rather than clamping.

`Delay` reports a finite tail. The bound is constant and conservative,
computed from
the declared range maxima (delay time up to the ring capacity, feedback up to
`0.95`) with retained-state headroom of `1e3` and a −120 dBFS decay floor. The
headroom covers the buildup from sustained full-scale input at maximum
feedback. `flush` continues the feedback recursion on silent input with the
last-rendered parameter values and reports `done` early once the ring has
decayed below the floor, so real drains end far sooner than the declared bound.

Constructors

```rust
# extern crate bisque;
use bisque::time::{Delay, DelaySettings, TimeStretch, TimeStretchSettings};

let delay = Delay::new();
let short_delay = Delay::with_settings(
    DelaySettings::new()
        .delay_ms(80.0)
        .feedback(0.2)
        .mix(0.35)
        .max_delay_ms(500.0)
);
let stretch = TimeStretch::<f32>::with_settings(TimeStretchSettings::new().stretch(1.5));
```

`TimeStretch` validates the stretch ratio in `0.5..=2.0` during `prepare`. It
supports 1 to 16 channels and has no automatable parameters. It uses
Hann-window overlap-add with a fixed synthesis hop and variable analysis hop.
Boundary overlap normalization and constant edge extension prevent leading and
trailing amplitude tapers.
It is not a phase vocoder and does not preserve phase or transients.
Non-unity ratios can smear attacks and move sustained tones among grain-rate
sidebands, which can sound phasey or pitch-unstable.
It reports zero latency. The pull model absorbs the analysis window internally,
and at unity ratio it reconstructs the complete finite input at the same length
with no leading shift.

The analysis hop is rounded to a whole frame. `TimeStretch::stretch()` returns
the requested ratio. After a successful `prepare`,
`TimeStretch::effective_stretch()` returns the ratio produced by the rounded
hop. Finite output length is input length multiplied by that effective ratio
and rounded to the nearest frame.

## Repair

Feature `repair`

| Type | Trait | I/O | Notes |
| --- | --- | --- | --- |
| `DcBlocker` | `Kernel` | In-place | One-pole, one-zero high-pass for DC removal |
| `DcOffset` | `Kernel` | In-place | Fixed uniform or per-channel additive offset |

`DcOffset` has no automatable parameters. `DcOffset::broadcast` applies one
offset to every channel. `DcOffset::per_channel` and
`DcOffset::per_channel_from_slice` require exactly one offset for each prepared
channel. It is the exact, spectrum-preserving counterpart to
`DcBlocker`: measure a per-channel mean with `MeanMeter`, then apply the negated
means here. It removes one constant offset, not time-varying DC, and leaves a
residual at the numerical floor, not bitwise zero.

`DcBlocker` parameters

| Constant | Name | Range | Unit |
| --- | --- | --- | --- |
| `DcBlocker::CUTOFF_HZ` | `cutoff` | `1.0..=1000.0` | Hz |

During rendering, the effective cutoff is clamped to
`0.01..=0.49 * sample_rate`. The declared range is unchanged, but unusually low
sample rates can lower its effective upper limit.

`DcBlocker` reports a finite tail. The bound is constant and conservative,
from the
slowest pole its cutoff range allows at a decay floor of `1e-6`. `flush`
continues the recurrence on silent input with the last-rendered pole radius
and reports `done` early once the state has settled below the floor.

Constructors

```rust
# extern crate bisque;
use bisque::repair::{DcBlocker, DcBlockerSettings, DcOffset};

let blocker = DcBlocker::new(); // 20 Hz cutoff
let low_cut = DcBlocker::with_settings(DcBlockerSettings::new().cutoff_hz(10.0));
let uniform_offset = DcOffset::broadcast(0.1);
let channel_offsets = DcOffset::per_channel(vec![0.1, -0.1]);
```

## Spectral

Feature `spectral`

The spectral module uses `realfft` and is optional.

| Type | Trait | I/O | Notes |
| --- | --- | --- | --- |
| `Fft` | Utility | None | Real-input FFT wrapper with owned plans and scratch |
| `Stft` | Utility | None | Offline STFT analysis and synthesis |
| `Window` | Enum | None | Rectangular, Hann, Hamming, Blackman, and Sine windows |
| `SpectralFilter` | `Processor` | Split | Streaming STFT brick-wall band filter |
| `SpectralFilterSettings` | Settings | None | FFT size, hop, window, and retained frequency band |

`SpectralFilter` declares split I/O, ignores events, and reports latency and
tail equal to one FFT window. Its configuration is stored as constructed:
`prepare` rejects a size below two, a hop outside `1..=size`, window overlap
with an unreconstructable output phase, or invalid band edges. Sizes may be odd
or even. The low band edge must be finite and nonnegative. The high edge may be
finite or positive infinity. `Fft::new` panics on a zero size. `Stft::new`
panics on a zero size or a hop outside `1..=size`.

Constructors

```rust
# extern crate bisque;
use bisque::spectral::{SpectralFilter, SpectralFilterSettings, Stft, Window};

let mut stft = Stft::new(1024, 512, Window::Hann);
let low_pass = SpectralFilter::low_pass(1024, 512, 4_000.0);
let band = SpectralFilter::band(1024, 512, 200.0, 5_000.0);
let rectangular = SpectralFilter::with_settings(
    SpectralFilterSettings::new()
        .size(1024)
        .hop(1024)
        .window(Window::Rectangular)
        .band(200.0, 5_000.0),
);
```

Spectral processors currently use behavior and contract tests, not committed
cross-platform snapshots.
