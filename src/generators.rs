//! This file contains all available generators in the crate.

pub mod sampler;
pub mod prelude;
pub mod wavetable;
pub mod adsr;
pub mod additive;
pub mod stereo_generator;

pub(crate) struct Note {
	// pub channel: u8,
	pub note: u8,
	pub velocity: f32,	
}