//! A tiny editor server: the graph page and the ops behind it.
//!
//! cargo run -p i_am_patch --example editor
//! then open the URL it prints.

use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use i_am_dsp::prelude::NoteEvent;
use i_am_patch::block::AudioBlock;
use i_am_patch::compile::voice_group;
use i_am_patch::engine::Engine;
use i_am_patch::poly::Poly;
use i_am_patch::scope::Scope;

use i_am_patch::graph::{Graph, NodeId};
use serde_json::{Value, json};

mod audio;
mod demo;
mod imports;
mod ops;
mod routes;
mod shared;

// One name per thing, so a module reaches its neighbours through the root.
pub(crate) use audio::spawn_audio;
pub(crate) use demo::{demo, patch_json};
pub(crate) use imports::{Import, import_route, sample_route};
pub(crate) use ops::op;
pub(crate) use routes::handle;
pub(crate) use shared::Shared;

const PAGE_CSS: &str = concat!(
	include_str!("../../../i_am_web_gui/src/page/card.css"),
	"\n",
	include_str!("../../src/page/patcher.css"),
);

fn main() {
	let shared = Shared {
		graph: Arc::new(Mutex::new(demo())),
		generation: Arc::new(AtomicU64::new(1)),
		rate: 48_000,
		taps: Arc::new(Mutex::new(Vec::new())),
		phases: Arc::new(Mutex::new(Vec::new())),
		midi_inputs: Arc::new(Mutex::new(Vec::new())),
		error: Arc::new(Mutex::new(None)),
		slots: Default::default(),
		edits: Arc::new(Mutex::new(Vec::new())),
	};
	spawn_audio(shared.clone());

	let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
	println!("http://{}", listener.local_addr().expect("address"));

	for stream in listener.incoming().flatten() {
		handle(stream, &shared);
	}
}
