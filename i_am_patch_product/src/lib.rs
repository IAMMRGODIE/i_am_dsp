//! The patcher products: the graph engine behind a host and a page.
//!
//! `i_am_patch` is the library; this crate is the two products a host loads.
//! They share everything but the boundary: the effect takes audio in, the
//! generator does not.

pub mod patcher;
pub mod save;
pub mod shared;
pub mod web;

pub use patcher::{PatcherEffect, PatcherGenerator, SIZE};
