//! Snapshots of the widget gallery.
//!
//! iced_test draws an interface in software, with no window and no input device
//! involved, so a faceplate can be checked on a machine that has no desktop at
//! all, and a click can be aimed at an exact point instead of being injected
//! into whatever window happens to be in front.
//!
//! The first run writes the reference image; every run after it compares
//! against that image and fails when a pixel moved. To accept a change, delete
//! the reference image under tests/snapshots and run again: the next run writes
//! the new one and passes. (iced_test appends the name of the renderer to the
//! file it is given, so the files on disk end in -wgpu.png.)

#[allow(dead_code)]
#[path = "../examples/widget_kit.rs"]
mod gallery;

use i_am_dsp_iced::styles;

/// Renders the gallery at a scale and compares it with its reference image.
fn render(scale: f32, name: &str) {
	styles::set_ui_scale(scale);

	let gallery = gallery::WidgetKit::default();
	let mut ui = iced_test::simulator(gallery.view());

	let snapshot = match ui.snapshot(&styles::theme()) {
		Ok(snapshot) => snapshot,
		Err(error) => panic!("could not render the gallery: {error:?}"),
	};

	let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
		.join("tests")
		.join("snapshots")
		.join(name);

	std::fs::create_dir_all(path.parent().expect("a snapshot has a directory"))
		.expect("could not create the snapshot directory");

	assert!(
		snapshot.matches_image(&path).expect("could not compare the snapshot"),
		"the gallery at {scale} no longer matches its reference image; delete {} and run again to accept the change",
		path.display()
	);
}

/// One test, not two: the UI scale is process-global, and two tests run in
/// parallel would fight over it and render each other at the wrong size.
#[test]
fn the_gallery_renders_the_same() {
	render(1.0, "gallery-100.png");
	render(1.5, "gallery-150.png");
	render(1.0, "gallery-100.png");
}
