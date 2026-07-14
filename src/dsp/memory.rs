// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Airclay LLC

//! Checked logical-buffer footprint calculations for prepare-time preflight.

use crate::processor::DspError;

/// A checked sum of logical reserved payload bytes.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct MemoryLayout {
    bytes: Option<usize>,
    components_addressable: bool,
}

impl MemoryLayout {
    pub(crate) const fn new() -> Self {
        Self {
            bytes: Some(0),
            components_addressable: true,
        }
    }

    pub(crate) fn array<T>(self, elements: usize) -> Self {
        self.repeated_bytes(1, elements.checked_mul(std::mem::size_of::<T>()))
    }

    #[cfg(any(
        test,
        feature = "analysis",
        feature = "filters",
        feature = "mastering",
        feature = "spectral",
        feature = "time"
    ))]
    pub(crate) fn repeated_array<T>(self, arrays: usize, elements: usize) -> Self {
        if arrays == 0 {
            return self.repeated_bytes(0, Some(0));
        }
        self.repeated_bytes(arrays, elements.checked_mul(std::mem::size_of::<T>()))
    }

    fn repeated_bytes(self, arrays: usize, component: Option<usize>) -> Self {
        let component_addressable = component.is_some_and(|n| isize::try_from(n).is_ok());
        let additional = component.and_then(|n| n.checked_mul(arrays));
        Self {
            bytes: self
                .bytes
                .and_then(|total| additional.and_then(|n| total.checked_add(n))),
            components_addressable: self.components_addressable && component_addressable,
        }
    }

    /// Return the required bytes after enforcing the optional budget.
    ///
    /// Layout overflow cannot be represented by `OverBudget::needed`. When a
    /// finite cap already proves the layout cannot fit, `usize::MAX` is the
    /// conservative reported requirement; otherwise the layout is rejected as
    /// an unsupported address-space request.
    pub(crate) fn preflight(self, cap: Option<usize>) -> Result<usize, DspError> {
        let Some(needed) = self.bytes else {
            return match cap {
                Some(cap) if cap < usize::MAX => Err(DspError::OverBudget {
                    needed: usize::MAX,
                    cap,
                }),
                _ => Err(DspError::UnsupportedSpec(
                    "state layout exceeds addressable memory",
                )),
            };
        };
        if let Some(cap) = cap {
            if needed > cap {
                return Err(DspError::OverBudget { needed, cap });
            }
        }
        // A single `Vec` allocation cannot exceed `isize::MAX` bytes even when
        // usize is wider. The aggregate may legitimately exceed that across
        // several independently addressable buffers.
        if !self.components_addressable {
            return Err(DspError::UnsupportedSpec(
                "state layout exceeds addressable memory",
            ));
        }
        Ok(needed)
    }
}

#[cfg(test)]
mod tests {
    use super::MemoryLayout;
    use crate::processor::DspError;

    #[test]
    fn sums_typed_arrays_and_enforces_cap() {
        let layout = MemoryLayout::new()
            .array::<u32>(3)
            .repeated_array::<f64>(2, 4);
        assert_eq!(layout.preflight(None), Ok(76));
        assert_eq!(
            layout.preflight(Some(75)),
            Err(DspError::OverBudget {
                needed: 76,
                cap: 75
            })
        );
    }

    #[test]
    fn multiplication_overflow_is_rejected_before_allocation() {
        let layout = MemoryLayout::new().repeated_array::<f64>(usize::MAX, 2);
        assert!(matches!(
            layout.preflight(Some(1024)),
            Err(DspError::OverBudget {
                needed: usize::MAX,
                cap: 1024
            })
        ));
    }

    #[test]
    fn one_component_cannot_exceed_the_addressable_allocation_limit() {
        let too_large = isize::MAX as usize + 1;
        let layout = MemoryLayout::new().array::<u8>(too_large);
        assert!(matches!(
            layout.preflight(None),
            Err(DspError::UnsupportedSpec(
                "state layout exceeds addressable memory"
            ))
        ));
    }

    #[test]
    fn independently_addressable_arrays_may_exceed_isize_in_aggregate() {
        let per_array = isize::MAX as usize / 2 + 1;
        let total = per_array * 2;
        let layout = MemoryLayout::new().repeated_array::<u8>(2, per_array);
        assert_eq!(layout.preflight(None), Ok(total));
    }

    #[test]
    fn overflow_with_an_unlimited_cap_is_an_unsupported_layout() {
        let layout = MemoryLayout::new().repeated_array::<u8>(usize::MAX, 2);
        assert!(matches!(
            layout.preflight(Some(usize::MAX)),
            Err(DspError::UnsupportedSpec(
                "state layout exceeds addressable memory"
            ))
        ));
    }
}
