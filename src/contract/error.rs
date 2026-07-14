// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Airclay LLC

//! Error type returned by preparation.

use std::fmt;

/// The error type returned by `prepare`.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum DspError {
    /// A parameter was outside its valid range or otherwise invalid.
    InvalidParam(&'static str),
    /// The [`ProcessSpec`](crate::processor::ProcessSpec) (rate, channels, block size) is unsupported.
    UnsupportedSpec(&'static str),
    /// The configuration's minimum footprint exceeds `ProcessSpec::max_memory`.
    OverBudget {
        /// Bytes the processor needs at minimum.
        needed: usize,
        /// The cap from `ProcessSpec::max_memory`.
        cap: usize,
    },
}

impl fmt::Display for DspError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidParam(m) => write!(f, "invalid parameter: {m}"),
            Self::UnsupportedSpec(m) => write!(f, "unsupported spec: {m}"),
            Self::OverBudget { needed, cap } => {
                write!(f, "over memory budget: needs {needed} bytes, cap is {cap}")
            }
        }
    }
}

impl std::error::Error for DspError {}

#[cfg(test)]
mod tests {
    use super::DspError;

    #[test]
    fn display_messages_carry_their_content() {
        // Display output includes the relevant message fields.
        let p = DspError::InvalidParam("freq").to_string();
        assert!(p.contains("freq") && p.contains("parameter"), "{p}");
        let s = DspError::UnsupportedSpec("rate").to_string();
        assert!(s.contains("rate") && s.contains("spec"), "{s}");
        let b = DspError::OverBudget {
            needed: 4096,
            cap: 1024,
        }
        .to_string();
        assert!(
            b.contains("4096") && b.contains("1024") && b.contains("budget"),
            "{b}"
        );
    }
}
