#!/usr/bin/env node
// The one preview server for the web faceplates.
//
// The assets are rebuilt from the sources on every run, so what a browser shows
// is the page the plug-in would embed -- never a copy that has drifted.
//
//   node tools/preview/preview.mjs cards            the card/hero gallery (the default)
//   node tools/preview/preview.mjs product gain     one product's real faceplate
//   node tools/preview/preview.mjs rack             the Multipass page
//   node tools/preview/preview.mjs patch            the patcher's live editor
//
// Flags:
//   --port N     the port to listen on (default 8950)
//   --dir DIR    where to build the static preview (default target/preview/<mode>)
//   --no-build   serve what is already on disk instead of rebuilding
//
// The product and rack pages are static: the page fetches the same JSON the
// plug-in serves over its custom protocol, and this server hands it the files
// the dump examples wrote. The patcher is not: its page edits a live graph, so
// this mode starts i_am_patch's own editor server and prints its URL.

import { execFileSync, spawn } from "node:child_process";
import { createServer } from "node:http";
import { mkdirSync, readFileSync, writeFileSync, copyFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const REPO = path.resolve(HERE, "..", "..");
const GALLERY = path.resolve(REPO, "..", ".ui-preview");
const BUILD = path.join(REPO, "target", "preview");

// Reads a "pub const NAME: &str = include_str!(...);" asset.
function includeAsset(library, name) {
	const file = path.join(REPO, library);
	const text = readFileSync(file, "utf8");
	const pattern = "(?:pub )?const " + name + "[^=]*=\\s*include_str!\\(\"([^\"]+)\"\\);";
	const match = text.match(new RegExp(pattern));
	if (!match) throw new Error(name + " is not an include_str! in " + library);
	return readFileSync(path.join(path.dirname(file), match[1]), "utf8");
}

// Concatenates the include_str! list of a "pub const NAME: &str = concat!(...);".
function concatAsset(library, name) {
	const file = path.join(REPO, library);
	const text = readFileSync(file, "utf8");
	const declaration = text.match(new RegExp("(?:pub )?const " + name + "[^=]*=\\s*concat!\\("));
	if (!declaration) throw new Error(name + " is not a concat! in " + library);
	const open = text.indexOf("concat!(", declaration.index);
	const close = text.indexOf("\n);", open);
	const chunk = text.slice(open, close);
	const parts = [...chunk.matchAll(/include_str!\("([^"]+)"\)/g)];
	return parts
		.map((part) => readFileSync(path.join(path.dirname(file), part[1]), "utf8"))
		.join("");
}

// Writes the shared page assets into dir.
//
// kind picks the kit: a faceplate page loads the metadata-driven kit, a rack
// page loads the rack's own page script and styles.
function refreshAssets(dir, kind) {
	mkdirSync(dir, { recursive: true });
	const files = {
		"card.css": includeAsset("i_am_web_gui/src/lib.rs", "CARD_STYLE"),
		"card.js": concatAsset("i_am_web_gui/src/lib.rs", "CARD_SCRIPT"),
	};

	if (kind === "rack") {
		files["kit.css"] = readFileSync(path.join(REPO, "i_am_rack/src/page/rack.css"), "utf8");
		files["kit.js"] = concatAsset("i_am_rack/src/lib.rs", "PAGE_JS");
	} else {
		files["kit.css"] = includeAsset("i_am_web_gui/src/lib.rs", "STYLE");
		files["kit.js"] = concatAsset("i_am_web_gui/src/lib.rs", "SCRIPT");
		// The card gallery also lays out a rack card, and that sheet is shared.
		if (kind === "gallery") {
			files["rack.css"] = readFileSync(path.join(REPO, "i_am_rack/src/page/rack.css"), "utf8");
		}
	}

	for (const [name, body] of Object.entries(files)) {
		writeFileSync(path.join(dir, name), body);
	}

	return Object.keys(files);
}

function cargo(args) {
	execFileSync("cargo", args, { cwd: REPO, stdio: "inherit" });
}

const MIME = {
	".html": "text/html; charset=utf-8",
	".css": "text/css; charset=utf-8",
	".js": "text/javascript; charset=utf-8",
	".json": "application/json; charset=utf-8",
};

// Serves dir until the process is killed.
function serve(dir, port, index = "index.html") {
	const root = path.resolve(dir);
	const server = createServer((request, response) => {
		const url = decodeURIComponent((request.url || "/").split("?")[0]);
		const relative = url === "/" ? "/" + index : url;
		const file = path.join(root, relative);

		// Nothing leaves the preview folder.
		if (!path.resolve(file).startsWith(root)) {
			response.writeHead(403).end();
			return;
		}

		let body;
		try {
			body = readFileSync(file);
		} catch {
			response.writeHead(404, { "content-type": "text/plain" });
			response.end("not in the preview: " + relative + "\n");
			return;
		}

		response.writeHead(200, {
			"content-type": MIME[path.extname(file)] || "application/octet-stream",
			"cache-control": "no-store",
		});
		response.end(body);
	});

	server.on("error", (error) => {
		if (error.code === "EADDRINUSE") {
			console.error("port " + port + " is busy; pass --port to pick another");
			process.exit(1);
		}
		throw error;
	});

	server.listen(port, "127.0.0.1", () => {
		console.log("preview on http://127.0.0.1:" + port + "/  (serving " + root + ")");
	});
}

function main() {
	const args = process.argv.slice(2);
	const flagValue = (flag) => {
		const at = args.indexOf(flag);
		return at >= 0 ? args[at + 1] : null;
	};
	const noBuild = args.includes("--no-build");
	const port = Number(flagValue("--port") || 8950);
	const mode = args.find((arg, at) => {
		if (arg.startsWith("--")) return false;
		const before = args[at - 1];
		return before !== "--port" && before !== "--dir";
	}) || "cards";

	if (mode === "patch") {
		// The patcher edits a live graph, so it runs its own server. This mode is a
		// convenience wrapper: same command, the real thing behind it.
		console.log("patch: starting i_am_patch's editor server (its own port)");
		spawn("cargo", ["run", "-q", "-p", "i_am_patch", "--example", "editor"], {
			cwd: REPO,
			stdio: "inherit",
		});
		return;
	}

	const dir = flagValue("--dir")
		? path.resolve(flagValue("--dir"))
		: mode === "cards"
			? GALLERY
			: path.join(BUILD, mode);

	if (mode === "cards") {
		const written = noBuild ? [] : refreshAssets(dir, "gallery");
		if (written.length) console.log("refreshed " + written.join(", ") + " from src/page");
		serve(dir, port, "heroes.html");
		return;
	}

	if (mode === "product" || mode === "rack") {
		const name = mode === "product" ? args[args.indexOf("product") + 1] : null;
		if (mode === "product" && (!name || name.startsWith("--"))) {
			console.error("product needs a name, for example: preview.mjs product gain");
			process.exit(2);
		}

		const out = name ? path.join(dir, name) : dir;
		if (!noBuild) {
			mkdirSync(out, { recursive: true });
			if (mode === "product") {
				cargo(["run", "-q", "-p", "i_am_effects", "--example", "dump_face", "--", out, name]);
				copyFileSync(path.join(REPO, "i_am_web_gui/src/page/index.html"), path.join(out, "index.html"));
				refreshAssets(out, "faceplate");
			} else {
				cargo(["run", "-q", "-p", "i_am_rack", "--example", "dump_rack", "--", out]);
				refreshAssets(out, "rack");
			}
			console.log("built " + out);
		}

		serve(out, port);
		return;
	}

	console.error("unknown mode: " + mode);
	console.error("usage: preview.mjs [cards|product <name>|rack|patch] [--port N] [--dir DIR] [--no-build]");
	process.exit(2);
}

main();
