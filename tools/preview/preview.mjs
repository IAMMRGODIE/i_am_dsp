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
import { copyFileSync, mkdirSync, readFileSync, readdirSync, writeFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const REPO = path.resolve(HERE, "..", "..");
const GALLERY = path.resolve(REPO, "..", ".ui-preview");
const BUILD = path.join(REPO, "target", "preview");

// Walks the Rust files of one crate, so a constant can move between modules
// without breaking the preview build.
function* rustFiles(directory) {
	for (const entry of readdirSync(directory, { withFileTypes: true })) {
		const file = path.join(directory, entry.name);
		if (entry.isDirectory()) yield* rustFiles(file);
		else if (entry.name.endsWith(".rs")) yield file;
	}
}

// Finds the module that declares "const NAME".
function declaration(crateDirectory, name) {
	const pattern = new RegExp("(?:pub )?const " + name + "\\b");

	for (const file of rustFiles(path.join(REPO, crateDirectory))) {
		const text = readFileSync(file, "utf8");
		if (pattern.test(text)) return { file, text };
	}

	throw new Error(name + " is not declared under " + crateDirectory);
}

// Reads a "const NAME: &str = include_str!(...);" asset.
function includeAsset(crateDirectory, name) {
	const { file, text } = declaration(crateDirectory, name);
	const match = text.match(
		new RegExp("(?:pub )?const " + name + "[^=]*=\\s*include_str!\\(\"([^\"]+)\"\\);"),
	);
	if (!match) throw new Error(name + " is not an include_str! in " + file);
	return readFileSync(path.join(path.dirname(file), match[1]), "utf8");
}

// Concatenates the include_str! list of a "const NAME: &str = concat!(...);".
function concatAsset(crateDirectory, name) {
	const { file, text } = declaration(crateDirectory, name);
	const declarationAt = text.match(new RegExp("(?:pub )?const " + name + "[^=]*=\\s*concat!\\("));
	if (!declarationAt) throw new Error(name + " is not a concat! in " + file);
	const open = text.indexOf("concat!(", declarationAt.index);
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
		"card.css": includeAsset("i_am_web_gui/src", "CARD_STYLE"),
		"card.js": concatAsset("i_am_web_gui/src", "CARD_SCRIPT"),
	};

	if (kind === "rack") {
		files["kit.css"] = readFileSync(path.join(REPO, "i_am_rack/src/page/rack.css"), "utf8");
		files["kit.js"] = concatAsset("i_am_rack/src", "PAGE_JS");
	} else {
		files["kit.css"] = includeAsset("i_am_web_gui/src", "STYLE");
		files["kit.js"] = concatAsset("i_am_web_gui/src", "SCRIPT");
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


// The preview has no audio engine, so the live routes answer with noise: enough
// for the heroes and meters to move while two pages are compared. The phase
// drifts per request, so a meter follows the level instead of freezing.
const NOISE_FRAMES = 1024;
let noisePhase = 0;

function noiseLevel() {
	noisePhase += 0.05;
	return 0.18 + 0.12 * Math.sin(noisePhase);
}

function noiseFrames(level) {
	const data = new Float32Array(NOISE_FRAMES);
	for (let index = 0; index < data.length; index++) {
		data[index] = (Math.random() * 2 - 1) * level;
	}
	return data;
}

function binary(response, parts) {
	const total = parts.reduce((sum, part) => sum + part.length, 0);
	const body = Buffer.concat(parts.map((part) => Buffer.from(part.buffer, part.byteOffset, part.byteLength)));
	response.writeHead(200, {
		"content-type": "application/octet-stream",
		"content-length": total,
		"cache-control": "no-store",
	});
	response.end(body);
}

function dbOf(level) {
	return 20 * Math.log10(Math.max(level, 1e-6));
}

// The live routes a faceplate polls; each answers with synthetic input.
function liveRoute(url, response) {
	const level = noiseLevel();

	if (url === "/scope") {
		const input = noiseFrames(level);
		const output = noiseFrames(level * 0.6);
		const sidechain = noiseFrames(level * 0.35);
		binary(response, [input, output, sidechain]);
		return true;
	}

	if (url === "/stereo") {
		const left = noiseFrames(level);
		const right = noiseFrames(level);
		const outLeft = noiseFrames(level * 0.6);
		const outRight = noiseFrames(level * 0.6);
		binary(response, [left, right, outLeft, outRight]);
		return true;
	}

	if (url === "/meter") {
		const body = JSON.stringify({
			input: dbOf(level),
			output: dbOf(level * 0.6),
			sidechain: dbOf(level * 0.35),
		});
		response.writeHead(200, { "content-type": "application/json", "cache-control": "no-store" });
		response.end(body);
		return true;
	}

	if (url === "/spectrum") {
		// A drawn spectrum page is a set of partial levels and start phases, not
		// a plain array: the hero sums the levels into the wave.
		const pages = [1, 2, 3].map((page) => ({
			levels: Array.from({ length: 512 }, (_, index) => {
				const n = index + 1;
				return Math.max(0, Math.sin(n * page * 0.7) * 0.5 + 0.5) / Math.sqrt(n);
			}),
			phases: Array.from({ length: 512 }, (_, index) => (index * page) % 512 / 512),
		}));
		response.writeHead(200, { "content-type": "application/json", "cache-control": "no-store" });
		response.end(JSON.stringify({ generation: Math.floor(Date.now() / 1000), frames: pages }));
		return true;
	}

	if (url === "/note") {
		response.writeHead(200, { "content-type": "application/json", "cache-control": "no-store" });
		response.end(JSON.stringify({ note: -1 }));
		return true;
	}

	if (url === "/allowed") {
		response.writeHead(200, { "content-type": "application/json", "cache-control": "no-store" });
		response.end(JSON.stringify({ notes: -1 }));
		return true;
	}

	return false;
}

// Serves dir until the process is killed.
function serve(dir, port, index = "index.html") {
	const root = path.resolve(dir);
	const server = createServer((request, response) => {
		const url = decodeURIComponent((request.url || "/").split("?")[0]);

		if (liveRoute(url, response)) return;

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
