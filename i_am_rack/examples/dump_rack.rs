//! Writes the rack page and the JSON /rack would answer, for a browser preview.
//!
//! The host serves the same assets over its custom protocol; this is what lets
//! the page be looked at without a DAW:
//!
//! ```text
//! cargo run -p i_am_rack --example dump_rack -- target/rack-preview
//! ```

use std::fs;
use std::path::PathBuf;

fn main() {
	let out = PathBuf::from(
		std::env::args()
			.nth(1)
			.unwrap_or_else(|| "target/rack-preview".to_string()),
	);

	fs::create_dir_all(&out).expect("the preview folder");

	fs::write(out.join("index.html"), include_str!("../src/page/rack.html"))
		.expect("the page");
	fs::write(out.join("kit.css"), include_str!("../src/page/rack.css"))
		.expect("the styles");
	fs::write(out.join("kit.js"), include_str!("../src/page/rack.js"))
		.expect("the script");
	fs::write(out.join("card.css"), i_am_web_gui::CARD_STYLE)
		.expect("the shared card styles");
	fs::write(out.join("card.js"), i_am_web_gui::CARD_SCRIPT)
		.expect("the shared card script");
	fs::write(
		out.join("rack"),
		i_am_rack::Rack::default().rack_json(),
	)
	.expect("the rack JSON");

	println!("{}", out.display());
}
