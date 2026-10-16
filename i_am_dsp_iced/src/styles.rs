//! Design tokens, and the default dark theme.
//!
//! Two ideas hold this file together:
//!
//! * **A design unit.** Every size is written as a multiple of [DESIGN_UNIT]
//!   (four logical pixels at 100 %), never as a bare pixel count, so a faceplate
//!   scales as one piece instead of one widget at a time.
//! * **Semantic colours.** A widget asks for "the modulation colour" or "the
//!   negative half of a bipolar value", never for a literal hue, so the whole
//!   family can be re-skinned — or given a per-effect accent — in one place.
//!
//! The scale is process-global. A plug-in instance owns one window and one scale
//! factor, so calling [set_ui_scale] once at start-up is enough, and no widget
//! has to thread a metrics value through every view function.

use iced::{
	Color, Theme,
	theme::palette::{Background, Danger, Extended, Pair, Primary, Secondary, Success, Warning, mix},
};

/// The default padding for the application.
pub const PADDING: f32 = 6.0;

/// The width of a panel border, in design units. Public: a product that draws its
/// own panel uses the same one the kit does.
pub const BORDER_WIDTH: f32 = 2.0;

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

/// The default bright factoe of the widget's background color when hovered.
pub static BRIGHT_FACTOR: f32 = 0.075;

// ---------------------------------------------------------------------------
// Design units
// ---------------------------------------------------------------------------

/// One design unit: four logical pixels at 100 % scale.
pub const DESIGN_UNIT: f32 = 4.0;

/// The smallest scale a faceplate may be drawn at.
pub const MIN_UI_SCALE: f32 = 0.75;
/// The largest scale a faceplate may be drawn at.
pub const MAX_UI_SCALE: f32 = 3.0;

/// The bit pattern of 1.0, written out because the bits are needed in a static.
const SCALE_ONE_BITS: u32 = 0x3F80_0000;

static UI_SCALE_BITS: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(SCALE_ONE_BITS);

/// The scale every design token is currently multiplied by.
pub fn ui_scale() -> f32 {
	f32::from_bits(UI_SCALE_BITS.load(std::sync::atomic::Ordering::Relaxed))
}

/// Sets the scale every design token is multiplied by, clamped to a sane range.
///
/// Call this once when the window is created, from the host's scale factor.
pub fn set_ui_scale(scale: f32) {
	UI_SCALE_BITS.store(
		scale.clamp(MIN_UI_SCALE, MAX_UI_SCALE).to_bits(),
		std::sync::atomic::Ordering::Relaxed,
	);
}

/// The given number of design units, in logical pixels, at the current scale.
pub fn u(units: f32) -> f32 {
	units * DESIGN_UNIT * ui_scale()
}

/// A raw pixel length, scaled to the current UI scale.
///
/// Only for values that are genuinely not on the design grid: font sizes and
/// hairline widths. Anything a faceplate lays out should use [u].
pub fn px(pixels: f32) -> f32 {
	pixels * ui_scale()
}

/// The spacing scale, in design units.
pub mod space {
	/// 4 px: the gap between a knob and its label.
	pub const XS: f32 = 1.0;
	/// 8 px: the gap between controls inside a group.
	pub const S: f32 = 2.0;
	/// 12 px: the gap between groups.
	pub const M: f32 = 3.0;
	/// 16 px: the padding inside a card.
	pub const L: f32 = 4.0;
	/// 24 px: the padding inside a window.
	pub const XL: f32 = 6.0;
}

/// The sizes controls come in, in design units.
pub mod size {
	/// A 40 px knob, for a faceplate that only has a few controls.
	pub const KNOB_S: f32 = 10.0;
	/// A 56 px knob, the default.
	pub const KNOB_M: f32 = 14.0;
	/// A 72 px knob, for a faceplate where one control is the story.
	pub const KNOB_L: f32 = 18.0;
	/// The height of a row control: a selector, a button, a text field.
	pub const ROW: f32 = 8.0;
	/// The height of a header.
	pub const HEADER: f32 = 7.0;
	/// The side of a square icon button.
	pub const ICON: f32 = 6.0;
	/// The width of the modulation ring drawn around a knob.
	pub const MOD_RING: f32 = 0.75;
	/// The gap between the modulation ring and the value ring.
	pub const MOD_GAP: f32 = 0.5;
	/// The corner radius of a card or a module. Deliberately soft: at 8 px a
	/// panel reads as a card rather than as a rectangle with a clipped corner.
	pub const RADIUS: f32 = 2.0;
	/// The corner radius of a row control: a button, a selector, a slider track.
	pub const CONTROL_RADIUS: f32 = 1.5;
	/// The thickness of a slider track.
	pub const TRACK: f32 = 1.5;
}

/// Type sizes, in logical pixels at 100 % scale.
pub mod font {
	/// The name of a parameter.
	pub const LABEL: f32 = 11.0;
	/// The value of a parameter. Monospaced, so digits do not jitter while a
	/// knob is dragged.
	pub const VALUE: f32 = 12.0;
	/// The title of a group of controls.
	pub const SECTION: f32 = 11.0;
	/// The title of a window or a module.
	pub const TITLE: f32 = 14.0;
	/// Anything that is present but not meant to be read closely.
	pub const CAPTION: f32 = 10.0;

	/// The font a parameter name is set in.
	pub const LABEL_FONT: iced::Font = iced::Font::DEFAULT;
	/// The font a parameter value is set in.
	pub const VALUE_FONT: iced::Font = iced::Font::MONOSPACE;
}

/// The colour of anything that is modulation, and of nothing else.
///
/// Reserved: no effect may use it as an identity colour, or a modulated
/// parameter stops being recognisable at a glance.
pub const MOD_COLOR: Color = Color::from_rgb(0.36, 0.84, 0.95);

/// The colour of the negative half of a bipolar control.
pub const NEGATIVE_COLOR: Color = SECONDARY_COLOR;

/// What a faceplate draws itself in when it has no identity colour of its own.
pub const IDENTITY_COLOR: Color = PRIMARY_COLOR;

/// How big a knob is drawn.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum KnobSize {
	/// A 40 px knob.
	Small,
	/// A 56 px knob: the default.
	#[default]
	Medium,
	/// A 72 px knob.
	Large,
}

impl KnobSize {
	/// The diameter of the ring, in design units.
	pub fn units(self) -> f32 {
		match self {
			Self::Small => size::KNOB_S,
			Self::Medium => size::KNOB_M,
			Self::Large => size::KNOB_L,
		}
	}

	/// The diameter of the ring, in logical pixels at the current scale.
	pub fn diameter(self) -> f32 {
		u(self.units())
	}

	/// The height of a knob that carries a label and a value underneath it.
	pub fn labeled_height(self) -> f32 {
		self.diameter() + u(space::XS) + px(font::LABEL) * 1.3 + px(font::VALUE) * 1.3
	}
}

/// The look of every push button in the family.
///
/// iced styles its own buttons from the theme, which is fine until the accent
/// colour is an effect's identity hue; this keeps the shape — radius, border,
/// padding — on the design grid instead, so a button scales with everything
/// else.
pub fn button_style(theme: &Theme, status: iced::widget::button::Status) -> iced::widget::button::Style {
	use iced::widget::button;

	let palette = theme.extended_palette();
	let radius = u(size::CONTROL_RADIUS).into();
	let border = |color| iced::Border {
		color,
		width: px(1.0),
		radius,
	};

	match status {
		button::Status::Hovered => button::Style {
			background: Some(palette.primary.strong.color.into()),
			text_color: palette.primary.strong.text,
			border: border(palette.primary.base.color),
			..Default::default()
		},
		button::Status::Pressed => button::Style {
			background: Some(palette.primary.weak.color.into()),
			text_color: palette.primary.weak.text,
			border: border(palette.primary.base.color),
			..Default::default()
		},
		_ => button::Style {
			background: Some(palette.background.strong.color.into()),
			text_color: palette.background.strong.text,
			border: border(palette.background.strongest.color),
			..Default::default()
		},
	}
}

/// The default theme for the application.
pub fn theme() -> Theme {
	theme_with_accent(IDENTITY_COLOR)
}

/// The default theme, with the primary colour replaced by an effect's own.
///
/// This is how a whole family shares one look while each member keeps a
/// recognisable accent: only the accent moves.
pub fn theme_with_accent(accent: Color) -> Theme {
	let palette = iced::theme::Palette {
		background: CARD_COLOR,
		text: PRIMARY_TEXT_COLOR,
		primary: accent,
		success: SUCCESS_COLOR,
		warning: WARNING_COLOR,
		danger: ERROR_COLOR,
	};

	Theme::custom_with_fn("Nablo", palette, move |_| {
		Extended {
			is_dark: true,
			success: Success::generate(SUCCESS_COLOR, BACKGROUND_COLOR, PRIMARY_TEXT_COLOR),
			warning: Warning::generate(WARNING_COLOR, BACKGROUND_COLOR, PRIMARY_TEXT_COLOR),
			danger: Danger::generate(ERROR_COLOR, BACKGROUND_COLOR, PRIMARY_TEXT_COLOR),
			secondary: Secondary::generate(SECONDARY_COLOR, SECONDARY_TEXT_COLOR),
			primary: Primary::generate(accent, BACKGROUND_COLOR, PRIMARY_TEXT_COLOR),

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
