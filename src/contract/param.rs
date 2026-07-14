// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Airclay LLC

//! Parameter identity, declared metadata, and runtime events.

use std::fmt;

/// A parameter's stable identity within a processor.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ParamId(pub u32);

/// The physical unit of a parameter.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum Unit {
    /// Decibels.
    Db,
    /// Frequency in Hz.
    Hz,
    /// Time in milliseconds.
    Ms,
    /// Resonance or quality factor.
    Q,
    /// A dimensionless linear value.
    Linear,
}

/// How a raw parameter value ramps toward a new target.
///
/// Each variant defines its own use of [`ParamInfo::smoothing_ms`]. With
/// `steps = smoothing_ms * 1e-3 * sample_rate / 32` (the control-rate steps in
/// one smoothing time), the per-control-step updates are exactly:
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum Smoothing {
    /// No smoothing: `cur = target` at the next control-rate boundary.
    ///
    /// `smoothing_ms` is ignored (a positive value is still required for
    /// metadata uniformity).
    Step,
    /// Constant ramp from the current value to each new target:
    /// `cur += abs(target - cur) / steps` toward the target, snapping on
    /// arrival. Without another target change, every ramp completes no later
    /// than the `ceil(steps)` control update.
    Linear,
    /// One-pole asymptotic approach:
    /// `cur += (target - cur) * (1 - exp(-1 / steps))`, so `smoothing_ms` is
    /// the time constant (about 63% of the remaining distance is covered per
    /// `smoothing_ms`). The target is approached but reached only
    /// asymptotically, never exactly.
    OnePole,
    /// Log-domain ramp from the current value to each new target. It uses the
    /// constant multiplicative factor
    /// `(max(cur, target) / min(cur, target))^(1 / steps)` per control update
    /// and snaps on arrival. Without another target change, every ramp
    /// completes no later than the `ceil(steps)` update. This is the natural
    /// shape for frequency parameters. It requires a strictly positive range
    /// minimum, which is validated in `prepare`.
    Exponential,
}

impl Smoothing {
    /// The default ramp shape for a unit: frequencies ramp log-domain
    /// ([`Exponential`](Self::Exponential)), everything else ramps
    /// [`Linear`](Self::Linear) over the raw value (dB parameters are already
    /// logarithmic in their raw form).
    #[must_use]
    pub const fn default_for(unit: Unit) -> Self {
        match unit {
            Unit::Hz => Self::Exponential,
            Unit::Db | Unit::Ms | Unit::Q | Unit::Linear => Self::Linear,
        }
    }
}

/// How a host maps a physical parameter value to and from `[0, 1]`.
///
/// This is deterministic parameter behavior, not presentation policy. Units,
/// normalized mapping, and smoothing are independent axes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum ValueScale {
    /// Uniform spacing in the physical value.
    Linear,
    /// Uniform spacing in the logarithm of a strictly positive value.
    Logarithmic,
}

impl ValueScale {
    /// The conventional mapping for `unit`: frequency is logarithmic and all
    /// other current units are linear.
    #[must_use]
    pub const fn default_for(unit: Unit) -> Self {
        match unit {
            Unit::Hz => Self::Logarithmic,
            Unit::Db | Unit::Ms | Unit::Q | Unit::Linear => Self::Linear,
        }
    }
}

/// Declared metadata for one automatable parameter. Returned by `param_info()`.
#[derive(Clone, Copy, Debug)]
#[non_exhaustive]
pub struct ParamInfo {
    /// Stable identity.
    pub id: ParamId,
    /// Human-readable name.
    pub name: &'static str,
    /// Inclusive `(min, max)` range. Runtime values are clamped to it.
    pub range: (f64, f64),
    /// Default value.
    pub default: f64,
    /// Physical unit.
    pub unit: Unit,
    /// Physical-to-normalized mapping used by generic hosts.
    pub value_scale: ValueScale,
    /// Ramp shape.
    pub smoothing: Smoothing,
    /// Smoothing time in milliseconds. Its exact meaning depends on
    /// `smoothing` (see [`Smoothing`] for the per-variant math).
    /// [`Smoothing::Linear`] and [`Smoothing::Exponential`] use it as the
    /// nominal duration of each target change. [`Smoothing::OnePole`] uses it
    /// as the asymptotic time constant, so a target is never reached exactly.
    /// [`Smoothing::Step`] ignores it. The 32-frame control grid quantizes the
    /// effective timing.
    ///
    /// Must be finite and positive for every shape, including `Step`
    /// (validated in `prepare`; the uniform requirement keeps metadata
    /// shape-independent).
    pub smoothing_ms: f64,
}

impl ParamInfo {
    /// Declare a parameter with conventional mapping and smoothing defaults.
    ///
    /// Construction preserves the supplied values so this method remains
    /// const-capable. `prepare` validates ranges, defaults, smoothing, and
    /// logarithmic-scale requirements before processing.
    #[must_use]
    pub const fn new(
        id: ParamId,
        name: &'static str,
        range: (f64, f64),
        default: f64,
        unit: Unit,
    ) -> Self {
        Self {
            id,
            name,
            range,
            default,
            unit,
            value_scale: ValueScale::default_for(unit),
            smoothing: Smoothing::default_for(unit),
            smoothing_ms: 5.0,
        }
    }

    /// Override the parameter's smoothing shape.
    #[must_use]
    pub const fn with_smoothing(mut self, smoothing: Smoothing) -> Self {
        self.smoothing = smoothing;
        self
    }

    /// Override the parameter's smoothing time in milliseconds.
    #[must_use]
    pub const fn with_smoothing_ms(mut self, smoothing_ms: f64) -> Self {
        self.smoothing_ms = smoothing_ms;
        self
    }

    /// Override the physical-to-normalized mapping.
    #[must_use]
    pub const fn with_value_scale(mut self, value_scale: ValueScale) -> Self {
        self.value_scale = value_scale;
        self
    }

    /// Map a physical value into `[0, 1]`.
    ///
    /// Finite values are clamped to the declared physical range. Non-finite
    /// values return an error. Interior round trips use deterministic `f64`
    /// math and are tested to a relative tolerance of `1e-12`. Metadata
    /// validity is checked during `prepare`.
    pub fn normalize(&self, value: f64) -> Result<f64, ParamValueError> {
        if !value.is_finite() {
            return Err(ParamValueError {
                param: self.id,
                value,
            });
        }
        let value = value.clamp(self.range.0, self.range.1);
        if value == self.range.0 {
            return Ok(0.0);
        }
        if value == self.range.1 {
            return Ok(1.0);
        }
        Ok(match self.value_scale {
            ValueScale::Linear => {
                let span = self.range.1 - self.range.0;
                if span.is_finite() {
                    (value - self.range.0) / span
                } else {
                    // Scaling both differences avoids overflow when a valid
                    // finite range crosses nearly the whole f64 domain.
                    (value * 0.5 - self.range.0 * 0.5) / (self.range.1 * 0.5 - self.range.0 * 0.5)
                }
            }
            ValueScale::Logarithmic => {
                let min_ln = crate::dsp::math::ln(self.range.0);
                (crate::dsp::math::ln(value) - min_ln)
                    / (crate::dsp::math::ln(self.range.1) - min_ln)
            }
        })
    }

    /// Map a normalized value into the declared physical range.
    ///
    /// Finite values are clamped to `[0, 1]`. Non-finite values return an
    /// error. Exact endpoint branches preserve the declared bounds bit-for-bit;
    /// interior round trips are tested to a relative tolerance of `1e-12`.
    pub fn denormalize(&self, normalized: f64) -> Result<f64, ParamValueError> {
        if !normalized.is_finite() {
            return Err(ParamValueError {
                param: self.id,
                value: normalized,
            });
        }
        let normalized = normalized.clamp(0.0, 1.0);
        if normalized == 0.0 {
            return Ok(self.range.0);
        }
        if normalized == 1.0 {
            return Ok(self.range.1);
        }
        Ok(match self.value_scale {
            ValueScale::Linear => self.range.0 * (1.0 - normalized) + self.range.1 * normalized,
            ValueScale::Logarithmic => {
                let min_ln = crate::dsp::math::ln(self.range.0);
                let max_ln = crate::dsp::math::ln(self.range.1);
                crate::dsp::math::exp(min_ln * (1.0 - normalized) + max_ln * normalized)
            }
        })
    }
}

/// A sample-stamped parameter change within a block.
///
/// Hosts pass events through `ProcessContext::events`, sorted by nondecreasing
/// `offset`. Offsets must be less than the block frame count and values must be
/// finite. The stamp selects the first 32-frame control-grid boundary at or
/// after it; target latching is not sample-accurate. Unknown ids are ignored.
/// Non-finite and out-of-block events are debug contract failures and release
/// skips. Malformed ordering is a debug contract failure with no promised
/// per-event release behavior.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ParamEvent {
    /// Frame offset within the block. Must be less than `ProcessContext::frames`.
    pub offset: u32,
    /// Which parameter changes.
    pub param: ParamId,
    /// The new target value.
    pub value: f64,
}

/// Failure returned by direct parameter writes.
///
/// Process-time [`ParamEvent`] application is best-effort because
/// [`Processor::process`](crate::processor::Processor::process) returns `()`.
/// This error is for checked direct restoration through
/// [`Processor::set_parameter_immediate`](crate::processor::Processor::set_parameter_immediate).
#[derive(Clone, Copy, Debug)]
#[non_exhaustive]
pub enum ParamSetError {
    /// The processor did not declare this parameter id.
    UnknownParam(ParamId),
    /// The supplied value was NaN or infinite.
    NonFiniteValue {
        /// Which parameter was being written.
        param: ParamId,
        /// The rejected value.
        value: f64,
    },
}

impl fmt::Display for ParamSetError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnknownParam(id) => write!(f, "unknown parameter id {}", id.0),
            Self::NonFiniteValue { param, value } => {
                write!(f, "non-finite value {value} for parameter id {}", param.0)
            }
        }
    }
}

impl std::error::Error for ParamSetError {}

/// A non-finite value rejected by normalized parameter mapping.
#[derive(Clone, Copy, Debug)]
#[non_exhaustive]
pub struct ParamValueError {
    /// The parameter whose mapping rejected the value.
    pub param: ParamId,
    /// The rejected NaN or infinity.
    pub value: f64,
}

impl fmt::Display for ParamValueError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "non-finite value {} for parameter id {}",
            self.value, self.param.0
        )
    }
}

impl std::error::Error for ParamValueError {}

/// A typed snapshot of smoothed parameter values, loaded from a
/// [`SmootherBank`](crate::dsp::SmootherBank) once per fixed-parameter run.
///
/// Implemented by the structs the [`params!`](crate::params) macro generates.
/// The macro is the supported implementation path for downstream kernels.
/// Manual implementations must preserve the count, ordering, and infallible
/// loading invariants described below.
/// Fields are loaded by declaration index, so `param_info()` must list the same
/// parameters in the same order with sequential ids starting at `0`. `prepare`
/// validates both the count (`COUNT == param_info().len()`) and the ordering
/// before any render runs.
pub trait Params: Copy + Send + 'static {
    /// The number of declared parameters.
    const COUNT: usize;
    /// Load the current smoothed values from `bank` by declaration index.
    fn from_bank(bank: &crate::dsp::SmootherBank) -> Self;
}

/// The empty parameter set for processors with no automatable parameters.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct NoParams;

impl Params for NoParams {
    const COUNT: usize = 0;
    fn from_bank(_bank: &crate::dsp::SmootherBank) -> Self {
        Self
    }
}

/// Declare a processor's typed parameter struct.
///
/// Generates the struct (one `f64` field per parameter), a [`ParamId`] constant
/// per field numbered sequentially in declaration order, and a [`Params`] impl
/// whose `from_bank` reads each field by that index. The processor's
/// `param_info()` must list the same parameters in the same order; `prepare`
/// rejects a metadata list whose length differs from the declared field count
/// and rejects out-of-order ids.
///
/// ```
/// bisque::params! {
///     /// Smoothed parameter values for a gain stage.
///     pub struct ExampleParams {
///         /// Gain in dB.
///         pub gain_db => GAIN_DB,
///     }
/// }
/// assert_eq!(
///     ExampleParams::GAIN_DB,
///     bisque::parameter::ParamId(0),
/// );
/// ```
#[macro_export]
macro_rules! params {
    (
        $(#[$struct_meta:meta])*
        $vis:vis struct $name:ident {
            $(
                $(#[$field_meta:meta])*
                $field_vis:vis $field:ident => $konst:ident
            ),+ $(,)?
        }
    ) => {
        $(#[$struct_meta])*
        #[derive(Clone, Copy, Debug, PartialEq)]
        $vis struct $name {
            $(
                $(#[$field_meta])*
                $field_vis $field: f64,
            )+
        }

        impl $name {
            $crate::params!(@consts [] $(($field, $konst))+);
        }

        impl $crate::parameter::Params for $name {
            const COUNT: usize = 0 $(+ $crate::params!(@one $field))+;
            fn from_bank(bank: &$crate::dsp::SmootherBank) -> Self {
                Self {
                    $(
                        // invariant: prepare validated that the typed field
                        // count equals param_info().len() and that ids are
                        // sequential declaration indices, so every field index
                        // resolves to a smoother.
                        $field: bank
                            .value_at(Self::$konst.0 as usize)
                            .expect("parameter count and ids validated during prepare"),
                    )+
                }
            }
        }
    };
    (@one $x:ident) => {
        1usize
    };
    (@consts [$($done:ident)*]) => {};
    (@consts [$($done:ident)*] ($field:ident, $konst:ident) $($rest:tt)*) => {
        #[doc = concat!("Id of `", stringify!($field), "` (its declaration index).")]
        pub const $konst: $crate::parameter::ParamId =
            $crate::parameter::ParamId((0usize $(+ $crate::params!(@one $done))*) as u32);
        $crate::params!(@consts [$($done)* $field] $($rest)*);
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    const LINEAR: ParamInfo = ParamInfo::new(ParamId(7), "gain", (-12.0, 12.0), 0.0, Unit::Db)
        .with_smoothing(Smoothing::OnePole)
        .with_smoothing_ms(10.0);
    const LOG: ParamInfo =
        ParamInfo::new(ParamId(8), "frequency", (20.0, 20_000.0), 1_000.0, Unit::Hz);

    #[test]
    fn constructor_defaults_and_const_overrides_are_stable() {
        assert_eq!(LINEAR.value_scale, ValueScale::Linear);
        assert_eq!(LINEAR.smoothing, Smoothing::OnePole);
        assert_eq!(LINEAR.smoothing_ms, 10.0);
        assert_eq!(LOG.value_scale, ValueScale::Logarithmic);
        assert_eq!(LOG.smoothing, Smoothing::Exponential);
    }

    #[test]
    fn mapping_endpoints_are_exact_and_finite_inputs_clamp() {
        assert_eq!(LINEAR.normalize(-99.0).unwrap(), 0.0);
        assert_eq!(LINEAR.normalize(99.0).unwrap(), 1.0);
        assert_eq!(LINEAR.denormalize(-1.0).unwrap(), LINEAR.range.0);
        assert_eq!(LINEAR.denormalize(2.0).unwrap(), LINEAR.range.1);
        assert_eq!(LOG.normalize(LOG.range.0).unwrap(), 0.0);
        assert_eq!(LOG.normalize(LOG.range.1).unwrap(), 1.0);
        assert_eq!(LOG.denormalize(0.0).unwrap(), LOG.range.0);
        assert_eq!(LOG.denormalize(1.0).unwrap(), LOG.range.1);
    }

    #[test]
    fn mapping_round_trips_interior_values() {
        for (info, value) in [(LINEAR, 3.25), (LOG, 440.0)] {
            let round_trip = info.denormalize(info.normalize(value).unwrap()).unwrap();
            assert!((round_trip - value).abs() <= value.abs().max(1.0) * 1e-12);
        }
    }

    #[test]
    fn mapping_stays_finite_across_extreme_valid_ranges() {
        let linear = ParamInfo::new(
            ParamId(9),
            "wide linear",
            (-f64::MAX, f64::MAX),
            0.0,
            Unit::Linear,
        );
        assert_eq!(linear.normalize(0.0).unwrap(), 0.5);
        assert_eq!(linear.denormalize(0.5).unwrap(), 0.0);

        let asymmetric = ParamInfo::new(
            ParamId(11),
            "asymmetric wide linear",
            (-f64::MAX, f64::MAX / 2.0),
            0.0,
            Unit::Linear,
        );
        let value = f64::MAX / 4.0;
        let normalized = asymmetric.normalize(value).unwrap();
        assert!((normalized - 5.0 / 6.0).abs() <= 2.0 * f64::EPSILON);
        let round_trip = asymmetric.denormalize(normalized).unwrap();
        assert!((round_trip - value).abs() <= value * 4.0 * f64::EPSILON);

        let logarithmic = ParamInfo::new(
            ParamId(10),
            "wide logarithmic",
            (1e-300, 1e300),
            1.0,
            Unit::Hz,
        );
        let normalized = logarithmic.normalize(1.0).unwrap();
        assert!((normalized - 0.5).abs() <= 1e-15);
        let round_trip = logarithmic.denormalize(normalized).unwrap();
        assert!(round_trip.is_finite());
        assert!((round_trip - 1.0).abs() <= 1e-12);
    }

    #[test]
    fn mapping_rejects_non_finite_values_with_the_parameter_id() {
        for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let err = LINEAR.normalize(value).unwrap_err();
            assert_eq!(err.param, LINEAR.id);
            assert!(err.value.is_nan() || err.value == value);
        }
        let err = LOG.denormalize(f64::NAN).unwrap_err();
        assert_eq!(err.param, LOG.id);
        assert!(err.value.is_nan());
    }

    #[test]
    fn parameter_errors_have_actionable_messages() {
        assert_eq!(
            ParamSetError::UnknownParam(ParamId(7)).to_string(),
            "unknown parameter id 7"
        );
        assert_eq!(
            ParamSetError::NonFiniteValue {
                param: ParamId(3),
                value: f64::INFINITY,
            }
            .to_string(),
            "non-finite value inf for parameter id 3"
        );
        assert_eq!(
            ParamValueError {
                param: ParamId(5),
                value: f64::NEG_INFINITY,
            }
            .to_string(),
            "non-finite value -inf for parameter id 5"
        );
    }
}
