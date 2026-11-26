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
	include_str!("page/card/heroes-tuned.js"),
	include_str!("page/card/heroes-history.js"),
	include_str!("page/card/heroes-filter-table.js"),
	include_str!("page/card/filter-table-editor.js"),
	include_str!("page/card/heroes-voice.js"),
	include_str!("page/card/heroes-sampler.js"),
	include_str!("page/card/heroes-additive.js"),
	include_str!("page/card/heroes-canvas.js"),
	include_str!("page/card/heroes-unison.js"),
	include_str!("page/card/heroes-morph.js"),
	include_str!("page/card/heroes-drums.js"),
	include_str!("page/card/heroes-spectral.js"),
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

/// A short FNV-1a of the asset's bytes, as the URL's cache stamp.
fn stamp(text: &str) -> String {
	let mut hash: u32 = 2_166_136_261;

	for byte in text.bytes() {
		hash ^= byte as u32;
		hash = hash.wrapping_mul(16_777_619);
	}

	format!("{hash:08x}")
}

/// Replaces the token after each `path?v=` with the asset's content stamp.
fn stamp_urls(html: &str, assets: &[(&str, &str)]) -> String {
	let mut out = html.to_string();

	for (path, content) in assets {
		let needle = format!("{path}?v=");
		let value = stamp(content);

		while let Some(index) = out.find(&needle) {
			let start = index + needle.len();
			let end = out[start..]
				.find(['"', '&', ' '])
				.map(|offset| start + offset)
				.unwrap_or(out.len());

			if out[start..end] == value {
				break;
			}

			out.replace_range(start..end, &value);
		}
	}

	out
}

impl PageAssets {
	/// The page with each asset URL stamped by that asset's content.
	///
	/// WebView2 caches a script by its URL: a rebuild that changes the bytes but
	/// not the URL keeps running the old page until the host forgets it. Stamping
	/// the URL with the content makes the cache expire exactly when it should.
	pub fn versioned_html(&self) -> String {
		stamp_urls(
			self.html,
			&[
				("/card.css", CARD_STYLE),
				("/kit.css", self.css),
				("/card.js", CARD_SCRIPT),
				("/kit.js", self.js),
			],
		)
	}

	/// A stamp of the whole page, for the document URL a rebuild has to move.
	pub fn version(&self) -> String {
		stamp(&format!("{}{}{}", self.html, self.css, self.js))
	}

	/// The faceplate page: controls generated from parameter metadata.
	pub const FACEPLATE: Self = Self {
		html: PAGE,
		css: STYLE,
		js: SCRIPT,
	};
}

#[cfg(test)]
mod tests {
	use super::*;

	/// Every asset URL carries its content's stamp, so a rebuilt page cannot
	/// keep running a script the WebView cached from the previous build.
	#[test]
	fn the_asset_urls_carry_a_content_stamp() {
		let html = PageAssets::FACEPLATE.versioned_html();

		for path in ["/card.css", "/kit.css", "/card.js", "/kit.js"] {
			let needle = format!("{path}?v=");
			let start = html
				.find(&needle)
				.unwrap_or_else(|| panic!("{path} missing"))
				+ needle.len();
			let end = html[start..].find('"').expect("a quoted url") + start;
			let stamp = &html[start..end];

			assert_eq!(stamp.len(), 8, "{path} stamp {stamp}");
			assert!(
				stamp.chars().all(|character| character.is_ascii_hexdigit()),
				"{path} stamp {stamp}"
			);
		}
	}

	/// The stamp is the content, so two different assets never share one.
	#[test]
	fn a_different_asset_gets_a_different_stamp() {
		assert_ne!(stamp("one"), stamp("two"));
		assert_eq!(stamp("one"), stamp("one"));
	}
}
