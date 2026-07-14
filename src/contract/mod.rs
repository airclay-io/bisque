// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Airclay LLC

//! Private implementation modules re-exported through the public
//! `processor` and `parameter` contracts.

pub(crate) mod block;
pub(crate) mod context;
pub(crate) mod error;
pub(crate) mod param;
pub(crate) mod sample;
pub(crate) mod source;
pub(crate) mod spec;
pub(crate) mod traits;

pub use block::{AudioBlock, AudioBlockMut, Io, IoMode};
pub use context::{ProcessContext, Produced, SubBlock, Tail};
pub use error::DspError;
pub use param::{
    NoParams, ParamEvent, ParamId, ParamInfo, ParamSetError, ParamValueError, Params, Smoothing,
    Unit, ValueScale,
};
pub use sample::Sample;
pub use source::RingSource;
pub use spec::ProcessSpec;
pub use traits::{Kernel, Measurer, Processor, Source, VariableRate};
