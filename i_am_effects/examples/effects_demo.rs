//! The effects, in a window, with something in front of them to hear.
//!
//! An effect with nothing to process is silent, so this puts a test source in front of
//! one of them: a saw at 110 Hz, which has enough harmonics that a filter, a crusher
//! and a drive all have something to do to it.
//!
//! ```text
//! cargo run -p i_am_effects --example effects_demo --features standalone
//! cargo run -p i_am_effects --example effects_demo --features standalone -- drive
//! cargo run -p i_am_effects --example effects_demo --features standalone -- crusher
//! ```
//!
//! The window is the effect's own faceplate -- the knobs, the sections, and the picture
//! each effect asked for -- because that is what a product shows a person. Nothing in
//! this file draws a control.
//!
//! **Space** starts and stops the source, so a filter can be heard opening and closing
//! against something that is not moving.

#[cfg(feature = "standalone")]
fn main() {
	match std::env::args().nth(1).as_deref() {
		Some("crusher") => demo::run::<i_am_effects::Crusher>(),
		Some("drive") => demo::run::<i_am_effects::Drive>(),
		Some("filter") | None => demo::run::<i_am_effects::Filter>(),
		Some(other) => {
			println!("no effect called {other}: running the filter");

			demo::run::<i_am_effects::Filter>()
		},
	}
}

#[cfg(not(feature = "standalone"))]
fn main() {
	println!("the standalone feature is off: nothing to run");
}

/// Everything that needs the demo host, which is what the feature is for.
#[cfg(feature = "standalone")]
mod demo {
	use std::sync::atomic::{AtomicBool, Ordering};

	use i_am_dsp::prelude::{Parameter, Parameters, SetValue};
	use i_am_dsp::ProcessContext;
	use i_am_dsp_iced::demo::Demo;
	use i_am_dsp_iced::iced;
	use i_am_dsp_iced::iced::Subscription;
	use i_am_dsp_iced::styles::theme;
	use i_am_dsp_iced::Processor;

	/// Whether the source is running.
	///
	/// A static rather than something the key listener carries, because iced takes a
	/// listener as a plain function and it cannot capture the flag it has to flip. One
	/// demo runs per process, so one flag is enough.
	static RUNNING: AtomicBool = AtomicBool::new(true);

	pub fn run<P>()
	where
		P: Processor + i_am_plugin::Plugin + Default + 'static,
	{
		// The window is the size and carries the title the effect asks for in its own
		// window options: a product that says how it wants to be met is met that way.
		let options = P::window_options();
		let size = options.window_size;
		let title = options.title.clone();

		iced::application(
			|| Demo::new(|sample_rate| Source::new(P::default(), sample_rate)),
			Demo::<Source<P>>::update,
			Demo::<Source<P>>::view,
		)
		.subscription(|_| {
			// The key is handled where it is seen, the way the kit's modifiers listener
			// keeps the keyboard state: a subscription is the only place a key arrives,
			// and this needs no message to travel anywhere.
			let space = iced::event::listen_with(|event, _, _| {
				if let iced::Event::Keyboard(iced::keyboard::Event::KeyPressed {
					key: iced::keyboard::Key::Named(iced::keyboard::key::Named::Space),
					repeat: false,
					..
				}) = event
				{
					let running = !RUNNING.load(Ordering::SeqCst);

					RUNNING.store(running, Ordering::SeqCst);
					println!("{}", if running { "playing" } else { "paused" });
				}

				None
			});

			Subscription::batch([Demo::<Source<P>>::subscriber(), space])
		})
		.theme(theme())
		.window_size(iced::Size::new(size.width as f32, size.height as f32))
		// Annotated because iced asks for a title for *any* borrow of the state, and an
		// unannotated closure is only inferred for one of them.
		.title(move |_: &Demo<Source<P>>| title.clone())
		.run()
		.expect("the demo window opens");
	}

	/// The test source, in front of the effect.
	///
	/// It is built for the rate the sound card runs at, which is what a host does: the
	/// plug-in is made first and told what it will run at afterwards, through the same
	/// hook a CLAP host uses.
	struct Source<P> {
		effect: P,
		phase: f32,
		sample_rate: usize,
	}

	impl<P: i_am_plugin::Plugin> Source<P> {
		fn new(mut effect: P, sample_rate: usize) -> Self {
			effect.on_activate(sample_rate);

			Self {
				effect,
				phase: 0.0,
				sample_rate,
			}
		}
	}

	impl<P: Processor> Parameters for Source<P> {
		fn get_parameters(&self) -> Vec<Parameter> {
			self.effect.get_parameters()
		}

		fn set_parameter(&mut self, identifier: &str, value: SetValue) -> bool {
			self.effect.set_parameter(identifier, value)
		}
	}

	impl<P: Processor> Processor for Source<P> {
		type Message = P::Message;
		type SyncedView = P::SyncedView;

		fn delay(&self) -> usize {
			self.effect.delay()
		}

		fn process(
			&mut self,
			samples: &mut [f32; 2],
			other: &[[f32; 2]],
			process_context: &mut Box<dyn ProcessContext>,
		) {
			// A saw: every harmonic of 110 Hz, so a filter has something to take away, a
			// crusher something to fold and a drive something to bend.
			//
			// One sample per call, because that is how a processor is handed audio here:
			// the demo host calls this once per frame and the source keeps its phase.
			//
			// Stopped means silence rather than no phase: the effect still runs, so a
			// tail rings out, and starting again carries on from where it was.
			let value = if RUNNING.load(Ordering::SeqCst) {
				let value = (self.phase * 2.0 - 1.0) * 0.25;

				self.phase = (self.phase + 110.0 / self.sample_rate as f32) % 1.0;
				value
			} else {
				0.0
			};

			samples[0] = value;
			samples[1] = value;

			// And then whatever the window is showing, done to it.
			self.effect.process(samples, other, process_context);
		}

		fn on_message(&self, message: Self::Message) {
			self.effect.on_message(message);
		}

		fn synced_view(&self) -> Self::SyncedView {
			self.effect.synced_view()
		}
	}
}
