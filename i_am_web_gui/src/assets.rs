pub(crate) const PAGE: &str = include_str!("page/index.html");
/// The page's styles.
pub(crate) const STYLE: &str = include_str!("page/kit.css");
/// The page's script, one IIFE split over the files in page/kit.
pub(crate) const SCRIPT: &str = concat!(
	include_str!("page/kit/state.js"),
	include_str!("page/kit/format.js"),
	include_str!("page/kit/controls.js"),
	include_str!("page/kit/eq.js"),
	include_str!("page/kit/heroes.js"),
	// What a hero's buttons and gestures mean, shared with the patcher cards.
	include_str!("page/card/hero-behaviour.js"),
	include_str!("page/kit/polls.js"),
	include_str!("page/kit/pitch.js"),
	include_str!("page/kit/amp.js"),
	include_str!("page/kit/face.js"),
	include_str!("page/kit/build.js"),
	include_str!("page/kit/boot.js"),
);

/// The card a faceplate and a rack module share, served at `/card.css`.
pub const CARD_STYLE: &str = include_str!("page/card.css");
/// The card's drawing code, served at `/card.js`.
///
/// The pieces are one IIFE split over several files so that a picture lives next
/// to the pictures it belongs to: no piece can stand alone, and the page loads
/// the concatenation as one script.
pub const CARD_SCRIPT: &str = concat!(
	include_str!("page/card/base.js"),
	include_str!("page/card/analysis.js"),
	include_str!("page/card/heroes-spectrum.js"),
	include_str!("page/card/heroes-scope.js"),
	include_str!("page/card/heroes-dynamics.js"),
	include_str!("page/card/heroes-stage.js"),
	include_str!("page/card/heroes-voice.js"),
	include_str!("page/card/heroes-drums.js"),
	include_str!("page/card/registry.js"),
	include_str!("page/card/controls.js"),
	include_str!("page/card/card.js"),
	// The control-source pictures shared by the patch and the rack page.
	include_str!("page/card/sources.js"),
	// What a hero's buttons and gestures mean, shared by every page that draws
	// one: the product's own page and a patcher node differ only in where an
	// edit goes.
	include_str!("page/card/hero-behaviour.js"),
);

/// The three assets a page is made of.
///
/// A product that draws something the metadata cannot describe -- a rack, a
/// curve editor, a browser of its own -- hands its own assets here. The routes
/// stay the same (`/`, `/kit.css`, `/kit.js`), so the page can be embedded
/// without adding another protocol or another editor backend.
#[derive(Debug, Clone, Copy)]
pub struct PageAssets {
	/// The HTML the editor loads.
	pub html: &'static str,
	/// The stylesheet served at `/kit.css`.
	pub css: &'static str,
	/// The script served at `/kit.js`.
	pub js: &'static str,
}

impl PageAssets {
	/// The faceplate page: controls generated from parameter metadata.
	pub const FACEPLATE: Self = Self {
		html: PAGE,
		css: STYLE,
		js: SCRIPT,
	};
}
