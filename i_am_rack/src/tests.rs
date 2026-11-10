//! M1 acceptance tests.

use i_am_dsp::Processor;
use i_am_dsp::prelude::{
	NoteEvent, Parameters, ProcessContext, ProcessInfos, SetValue, Value, from_binary, to_binary,
};
use i_am_plugin::Plugin;
use i_am_plugin::wry::{ProtocolRequest, ProtocolResponse};
use i_am_web_gui::Channels;

use super::*;
use crate::state::{Assignment, AssignmentMode, MACRO_BASE, SourceKind, Target};

fn command(rack: &Rack, body: &str) -> ProtocolResponse {
	let request = ProtocolRequest {
		method: "POST",
		path: "/rack-command",
		query: None,
		body: body.as_bytes(),
	};

	web::route(&request, &rack.shared).expect("the rack command route")
}

fn get(rack: &Rack, path: &str) -> ProtocolResponse {
	let request = ProtocolRequest {
		method: "GET",
		path,
		query: None,
		body: b"",
	};

	web::route(&request, &rack.shared).expect("the rack route")
}

/// The live value of one modulation source, as the page would read it.
fn modulation_value(rack: &Rack, id: u32) -> f64 {
	let response = get(rack, "/modulation");
	let json: serde_json::Value = serde_json::from_slice(&response.body).expect("JSON");

	json["sources"]
		.as_array()
		.expect("sources")
		.iter()
		.find(|source| source["id"] == id)
		.expect("the source")["value"]
		.as_f64()
		.expect("a value")
}

fn process(rack: &mut Rack, frame: [f32; 2]) -> [f32; 2] {
	process_with(rack, frame, Vec::new())
}

struct EventContext {
	infos: ProcessInfos,
	events: Vec<NoteEvent>,
}

impl ProcessContext for EventContext {
	fn infos(&self) -> &ProcessInfos {
		&self.infos
	}

	fn next_event(&mut self) -> Option<NoteEvent> {
		if self.events.is_empty() {
			None
		} else {
			Some(self.events.remove(0))
		}
	}

	fn send_event(&mut self, event: NoteEvent) {
		self.events.push(event);
	}

	fn events(&self) -> &[NoteEvent] {
		&self.events
	}

	fn clear_events(&mut self) {
		self.events.clear();
	}
}

fn process_with(rack: &mut Rack, frame: [f32; 2], events: Vec<NoteEvent>) -> [f32; 2] {
	let mut frame = frame;
	let mut context: Box<dyn ProcessContext> = Box::new(EventContext {
		infos: ProcessInfos::default(),
		events,
	});

	Processor::process(rack, &mut frame, &[], &mut context);

	frame
}

fn process_engine(engine: &mut crate::engine::Engine, frame: [f32; 2]) -> [f32; 2] {
	let mut frame = frame;
	let mut context: Box<dyn ProcessContext> = Box::new(());

	engine.process(&mut frame, &[], &mut context);
	frame
}

mod catalog;
mod history;
mod modulation;
mod modules;
mod page;
mod sources;
mod state;
