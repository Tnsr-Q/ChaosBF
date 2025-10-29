//! ChaosBF v4.0.1 WASM - Complete thermodynamic evolution platform
//!
//! Features all Phase 2 and 3 advanced capabilities:
//! - AURORA learned descriptors with InfoNCE contrastive loss
//! - Lyapunov estimation with bootstrap CI
//! - Edge-band routing (marginal elite selection)
//! - Island ecology (multi-population speciation)
//! - Reproducibility spine (snapshots & manifests)
//! - Critic-in-the-loop (self-bootstrapping semantics)
//! - PID + variance shaping dual-loop control
//! - Metropolis MCMC acceptance
//! - Landauer-exact costing

#![allow(clippy::manual_range_contains)]
#![allow(clippy::unnecessary_cast)]
#![allow(clippy::assign_op_pattern)]
#![allow(clippy::get_first)]
#![allow(clippy::manual_is_multiple_of)]
#![allow(clippy::or_fun_call)]
#![allow(clippy::missing_const_for_thread_local)]
#![allow(clippy::not_unsafe_ptr_arg_deref)]
#![allow(clippy::unwrap_or_default)]

pub mod rng;
pub mod state;
pub mod ops;
pub mod thermo;
pub mod aurora;
pub mod lyapunov;
pub mod edge_band;
pub mod island;
pub mod critic;
pub mod repro;
pub mod wasm_api;

// Re-export the frozen WASM API
pub use crate::wasm_api::*;
