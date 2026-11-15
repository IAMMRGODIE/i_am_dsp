use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use i_am_plugin::wry::WryEditor;
use i_am_plugin::{Editor, Plugin};

use crate::assets::{CARD_SCRIPT, CARD_STYLE, PageAssets};
use crate::face;
use crate::{Faceplate, ProtocolRequest, ProtocolResponse, Scope};

/// The editor for a plug-in whose controls the metadata describes.
///
/// Returns nothing when the plug-in has no controls a person can move: a page with
/// no knobs is a blank window, which is worse than telling the host there is none.
///
/// The bypass flag is shared with the product's processor, so turning the switch
/// off passes the input through untouched.
pub fn editor<P: Plugin>(
	processor: &P,
	face: Faceplate,
	bypass: Arc<AtomicBool>,
	scope: Arc<Scope>,
	rate: usize,
) -> Option<Box<dyn Editor<P>>> {
	editor_with(
		processor,
		face,
		bypass,
		scope,
		rate,
		|_: &ProtocolRequest<'_>| None,
		|_: &str| false,
	)
}

/// The editor for a plug-in that also serves routes and commands of its own.
///
/// `extra` sees every request before the kit's routes do, and answers the ones
/// the product owns: an upload route, or the state of a resource the page draws.
/// Returning `None` passes the request on to the kit.
///
/// `command` sees every IPC string before the parameter parser does. A command
/// that needs the host -- a native file dialog, say -- returns true and is not
/// mistaken for `id=value`.
pub fn editor_with<P: Plugin>(
	processor: &P,
	face: Faceplate,
	bypass: Arc<AtomicBool>,
	scope: Arc<Scope>,
	rate: usize,
	extra: impl Fn(&ProtocolRequest<'_>) -> Option<ProtocolResponse> + Send + Sync + 'static,
	command: impl Fn(&str) -> bool + Send + Sync + 'static,
) -> Option<Box<dyn Editor<P>>> {
	if face::sections(&processor.get_parameters()).is_empty() {
		return None;
	}

	editor_with_page(
		processor,
		face,
		PageAssets::FACEPLATE,
		bypass,
		scope,
		rate,
		extra,
		command,
	)
}

/// The editor for a plug-in with a page of its own.
///
/// This is [editor_with] with the three embedded assets replaced. The routes the
/// kit serves -- `/face`, `/state`, `/scope`, `/stereo`, `/note`, `/meter` --
/// are still there, and `extra` still answers first, so a page that draws its own
/// state can add `/rack` and keep the scope if it wants one.
///
/// Unlike [editor_with], a page with no metadata controls is not refused: a rack
/// may take all of its controls from its own route.
// The argument list matches editor_with, plus the page: splitting it into a
// builder would make the one call site that matters longer, not shorter.
#[allow(clippy::too_many_arguments)]
pub fn editor_with_page<P: Plugin>(
	processor: &P,
	face: Faceplate,
	page: PageAssets,
	bypass: Arc<AtomicBool>,
	scope: Arc<Scope>,
	rate: usize,
	extra: impl Fn(&ProtocolRequest<'_>) -> Option<ProtocolResponse> + Send + Sync + 'static,
	command: impl Fn(&str) -> bool + Send + Sync + 'static,
) -> Option<Box<dyn Editor<P>>> {
	let sections = face::sections(&processor.get_parameters());
	let page_face = face::face_json_with_rate(&face, &sections, Some(rate), true);
	let map = processor.param_map();

	let state_sections = sections.clone();
	let state_map = map.clone();
	let state_bypass = bypass.clone();
	let command_sections = sections;
	let command_map = map;
	let command_bypass = bypass;

	let page_html = page.versioned_html();

	Some(Box::new(
		WryEditor::<P>::new(page_html.clone(), move |body| {
			if command(body) {
				return None;
			}

			if !face::apply_bypass(&command_bypass, body) {
				face::apply_command(&command_map, &command_sections, body);
			}

			// The page writes the value into the shared map itself, which is what
			// the audio thread reads; there is no message to carry it.
			None
		})
		.with_custom_protocol_request("webgui", move |request| {
			if let Some(response) = extra(request) {
				return response;
			}

			match request.path {
				"/" | "/index.html" => ProtocolResponse::html(page_html.clone()),
				"/kit.css" => ProtocolResponse {
					status: 200,
					content_type: "text/css; charset=utf-8",
					body: page.css.as_bytes().to_vec(),
				},
				"/kit.js" => ProtocolResponse {
					status: 200,
					content_type: "text/javascript; charset=utf-8",
					body: page.js.as_bytes().to_vec(),
				},
				"/card.css" => ProtocolResponse {
					status: 200,
					content_type: "text/css; charset=utf-8",
					body: CARD_STYLE.as_bytes().to_vec(),
				},
				"/card.js" => ProtocolResponse {
					status: 200,
					content_type: "text/javascript; charset=utf-8",
					body: CARD_SCRIPT.as_bytes().to_vec(),
				},
				"/face" => ProtocolResponse::json(page_face.clone()),
				"/state" => ProtocolResponse::json(face::state_json_with_bypass(
					&state_map,
					&state_sections,
					state_bypass.load(Ordering::Relaxed),
				)),
				"/scope" => ProtocolResponse {
					status: 200,
					content_type: "application/octet-stream",
					body: scope.snapshot(),
				},
				"/stereo" => ProtocolResponse {
					status: 200,
					content_type: "application/octet-stream",
					body: scope.stereo_snapshot(),
				},
				"/note" => ProtocolResponse::json(format!("{{\"note\":{}}}", scope.note())),
				"/allowed" => {
					ProtocolResponse::json(format!("{{\"notes\":{}}}", scope.allowed_notes()))
				}
				"/meter" => {
					let (input, sidechain) = scope.meter();
					ProtocolResponse::json(format!(
						"{{\"input\":{input:.2},\"sidechain\":{sidechain:.2}}}"
					))
				}
				_ => ProtocolResponse::not_found(),
			}
		})
		// The document itself is cached by URL too, so a rebuild moves the URL:
		// otherwise the page keeps the script URLs of the build before it.
		.with_url(format!("webgui://localhost/?v={}", page.version())),
	))
}
