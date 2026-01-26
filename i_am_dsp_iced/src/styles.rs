//! Default colors used in the application.

use iced::{Color, Theme, theme::palette::{Background, Danger, Extended, Pair, Primary, Secondary, Success, Warning, mix}};

/// The default padding for the application.
pub const PADDING: f32 = 6.0;

pub(crate) const BORDER_WIDTH: f32 = 2.0;

/// The alpha factor for the waveform display.
pub const ALPHA_FACTOR: f32 = 0.3;

/// The default background color.
pub static BACKGROUND_COLOR: Color = Color::from_rgba(0x1E as f32 / 255.0, 0x1E as f32 / 255.0, 0x1E as f32 / 255.0, 1.0);
/// The default background color of the card.
pub static CARD_COLOR: Color = Color::from_rgba(0x27 as f32 / 255.0, 0x27 as f32 / 255.0, 0x27 as f32 / 255.0, 1.0);
/// The default border color of the card.
pub static CARD_BORDER_COLOR: Color = Color::from_rgba(0x3D as f32 / 255.0, 0x3D as f32 / 255.0, 0x3D as f32 / 255.0, 1.0);

/// The primary color for the application.
pub static PRIMARY_COLOR: Color = Color::from_rgba(0x8A as f32 / 255.0, 0x6A as f32 / 255.0, 0xFF as f32 / 255.0, 1.0);
/// The secondary color for the application.
pub static SECONDARY_COLOR: Color = Color::from_rgba(0x6B as f32 / 255.0, 0xB5 as f32 / 255.0, 0xFF as f32 / 255.0, 1.0);

/// The default colors for the error message.
pub static ERROR_COLOR: Color = Color::from_rgba(0xFF as f32 / 255.0, 0x4D as f32 / 255.0, 0x6D as f32 / 255.0, 1.0);
/// The default colors for the success message.
pub static SUCCESS_COLOR: Color = Color::from_rgba(0x00 as f32 / 255.0, 0xC8 as f32 / 255.0, 0x97 as f32 / 255.0, 1.0);
/// The default colors for the warning message.
pub static WARNING_COLOR: Color = Color::from_rgba(0xFF as f32 / 255.0, 0xB8 as f32 / 255.0, 0x5C as f32 / 255.0, 1.0);

/// The default title colors for the application.
pub static PRIMARY_TEXT_COLOR: Color = Color::from_rgba(0xE0 as f32 / 255.0, 0xE0 as f32 / 255.0, 0xE0 as f32 / 255.0, 1.0);
/// The default text colors for the application.
pub static SECONDARY_TEXT_COLOR: Color = Color::from_rgba(0xB0 as f32 / 255.0, 0xB0 as f32 / 255.0, 0xB0 as f32 / 255.0, 1.0);
/// The default disabled text colors for the application.
pub static DISABLE_TEXT_COLOR: Color = Color::from_rgba(0x70 as f32 / 255.0, 0x70 as f32 / 255.0, 0x70 as f32 / 255.0, 1.0);

/// The default theme for the application.
pub fn theme() -> iced::Theme {
	let palette = iced::theme::Palette { 
		background: CARD_COLOR, 
		text: PRIMARY_TEXT_COLOR, 
		primary: PRIMARY_COLOR, 
		success: SUCCESS_COLOR, 
		warning: WARNING_COLOR, 
		danger: ERROR_COLOR 
	};

	Theme::custom_with_fn("Nablo", palette, |_| {
		Extended {
			is_dark: true,
			success: Success::generate(SUCCESS_COLOR, BACKGROUND_COLOR, PRIMARY_TEXT_COLOR),
			warning: Warning::generate(WARNING_COLOR, BACKGROUND_COLOR, PRIMARY_TEXT_COLOR),
			danger: Danger::generate(ERROR_COLOR, BACKGROUND_COLOR, PRIMARY_TEXT_COLOR),
			secondary: Secondary::generate(SECONDARY_COLOR, SECONDARY_TEXT_COLOR),
			primary: Primary::generate(PRIMARY_COLOR, BACKGROUND_COLOR, PRIMARY_TEXT_COLOR),

			background: Background {
				base: Pair::new(CARD_COLOR, PRIMARY_TEXT_COLOR),
				weakest: Pair::new(BACKGROUND_COLOR, SECONDARY_TEXT_COLOR),
				weaker: Pair::new(mix(CARD_COLOR, BACKGROUND_COLOR, 0.3), SECONDARY_TEXT_COLOR),
				weak: Pair::new(mix(CARD_COLOR, BACKGROUND_COLOR, 0.7), SECONDARY_TEXT_COLOR),
				neutral: Pair::new(CARD_COLOR, PRIMARY_TEXT_COLOR),
				strong: Pair::new(mix(CARD_COLOR, CARD_BORDER_COLOR, 0.3), PRIMARY_TEXT_COLOR),
				stronger: Pair::new(mix(CARD_COLOR, CARD_BORDER_COLOR, 0.7), PRIMARY_TEXT_COLOR),
				strongest: Pair::new(CARD_BORDER_COLOR, PRIMARY_TEXT_COLOR),
			}
		}
	})
}

// /// The border color for input fields while unfocused (e.g., text boxes).
// pub static INPUT_BORDER_COLOR: Color = Color::from_rgba(0x44 as f32 / 255.0, 0x44 as f32 / 255.0, 0x44 as f32 / 255.0, 1.0);
// /// The color for selected text in input fields (e.g., text boxes).
// pub static SELECTED_TEXT_COLOR: Color = Color::from_rgba(0x8A as f32 / 255.0, 0x6A as f32 / 255.0, 0xFF as f32 / 255.0, 0.3);
// /// The default background color of the button, selectable label, and other clickable elements when disabled.
// pub static DISABLE_COLOR: Color = Color::from_rgba(0x5A as f32 / 255.0, 0x4A as f32 / 255.0, 0x8F as f32 / 255.0, 1.0);
/// The default bright factoe of the widget's background color when hovered.
pub static BRIGHT_FACTOR: f32 = 0.075;