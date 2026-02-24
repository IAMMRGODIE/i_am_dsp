//! A simple fixed-points equalizer plugin.

use i_am_dsp::{Effect, prelude::{Biquad, Paramed, Parameters}};
use iced::{Border, Length, Theme, alignment::{Horizontal, Vertical}, widget::{button, canvas, column, container, row}};

use crate::{Processor, SyncedView, styles::{BORDER_WIDTH, PADDING}, tools::{equalizer::EqualizerEditor, knob::knob}};

/// A simple fixed-points equalizer plugin.
pub struct Equalizer {
	biquads: Paramed<Vec<Biquad>>,
	editor: EqualizerView,
}

#[derive(Debug, Clone)]
/// The view of the equalizer.
pub struct EqualizerView {
	editor: EqualizerEditor,
}

impl Parameters for Equalizer {
	fn get_parameters(&self) -> Vec<i_am_dsp::prelude::Parameter> {
		self.biquads.get_parameters()
	}

	fn set_parameter(&mut self, identifier: &str, value: i_am_dsp::prelude::SetValue) -> bool {
		self.biquads.set_parameter(identifier, value)
	}
}

impl Equalizer {
	/// Create a new equalizer with the given number of biquads.
	pub fn new(sample_rate: usize, total_biquads: usize) -> Self {
		let (editor, biquads) = EqualizerEditor::new(total_biquads, sample_rate);
		Self {
			biquads,
			editor: EqualizerView { editor },
		}
	}
}

impl SyncedView for EqualizerView {
	type Message = ();

	fn update(&mut self, _: &Self::Message) {
		
	}

	fn view(&self) -> iced::Element<'_, Self::Message> {
		#[inline(always)]
		fn theme_func(theme: &Theme) -> iced::widget::container::Style {
			iced::widget::container::Style::default()
				.border(Border::default()
				.color(theme.extended_palette().background.strongest.color)
				.width(BORDER_WIDTH)
				.rounded(PADDING)
			)
			.background(theme.extended_palette().background.weakest.color)
		}

		column![
			canvas(&self.editor).height(Length::FillPortion(4)).width(Length::FillPortion(1)),
			
			container({
				let (cutoff, q, range_q, gain) = self.editor.edit_current(|settings, _| {
					(settings.get_cutoff(), settings.get_q(), settings.get_range_q(), settings.get_gain_db())
				});

				let sample_rate = self.editor.sample_rate as f32;
				// let current_node = self.editor.current_editing.load(std::sync::atomic::Ordering::SeqCst);

				row![
					// text!("Current node: {}", current_node),
					button("-").width(32.0).height(32.0).on_press_with(|| {
						self.editor.edit_current(|settings, _| {
							settings.set_to_prev();
						});
					}),
					knob(10.0..=sample_rate / 2.0 - 10.0, cutoff, |new_cutoff| { 
						self.editor.edit_current(|settings, sample_rate| {
							settings.set_cutoff(new_cutoff, sample_rate);
						});
					}).width(32.0).height(32.0),
					knob(range_q, q, |new_q| { 
						self.editor.edit_current(|settings, _| {
							settings.set_q(new_q);
						});
					}).width(32.0).height(32.0),
					gain.map(|gain| {
						knob(-12.0..=12.0, gain, |new_gain| { 
							self.editor.edit_current(|settings, _| {
								settings.set_gain_db(new_gain);
							});
						}).width(32.0).height(32.0)
					}),
					button("+").width(32.0).height(32.0).on_press_with(|| {
						self.editor.edit_current(|settings, _| {
							settings.set_to_next();
						});
					}),
				].align_y(Vertical::Center)
					.padding(16.0)
					.spacing(16.0)
					.height(64.0)
			}
			).align_x(Horizontal::Center).style(theme_func).width(Length::FillPortion(1))
		].align_x(Horizontal::Center).padding(16.0).spacing(16.0).into()
	}
}

impl Processor for Equalizer {
	type Message = ();
	type SyncedView = EqualizerView;


	fn delay(&self) -> usize {
		0
	}

	fn on_message(&self, _: Self::Message) {
		
	}

	fn process(&mut self, samples: &mut [f32; 2], other: &[[f32; 2]], process_context: &mut Box<dyn i_am_dsp::ProcessContext>) {
		self.editor.editor.sample_rate = process_context.infos().sample_rate;
		self.biquads.value.process(samples, other.iter().collect::<Vec<_>>().as_slice(), process_context);
	}

	fn synced_view(&self) -> Self::SyncedView {
		self.editor.clone()
	}
}
