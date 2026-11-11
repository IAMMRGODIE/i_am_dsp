#!/usr/bin/env node
// The one preview server for the web faceplates.
//
// The assets are rebuilt from the sources on every run, so what a browser shows
// is the page the plug-in would embed -- never a copy that has drifted.
//
//   node tools/preview/preview.mjs cards            the card/hero gallery (the default)
//   node tools/preview/preview.mjs product gain     one product's real faceplate
//   node tools/preview/preview.mjs card gain        the same face as a patch card
//   node tools/preview/preview.mjs compare gain     product page and card side by side
//   node tools/preview/preview.mjs compare-all      every product, one picker page
//   node tools/preview/preview.mjs rack             the Multipass page
//   node tools/preview/preview.mjs patch            the patcher's live editor
//
// Flags:
//   --port N     the port to listen on (default 8950)
//   --dir DIR    where to build the static preview (default target/preview/<mode>)
//   --no-build   serve what is already on disk instead of rebuilding
//   --synth      the name is an instrument, not an effect (card/compare)
//   --limit N    build only the first N products (compare-all)
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
function liveRoute(url, response, request) {
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

	if (url === "/script") {
		if (request.method === "POST") {
			response.writeHead(200, { "content-type": "application/json", "cache-control": "no-store" });
			response.end(JSON.stringify({ ok: true }));
		} else {
			response.writeHead(200, { "content-type": "application/json", "cache-control": "no-store" });
			response.end(JSON.stringify({ source: "// the preview has no worker: this is the script text area\nfn process(x) { x }\n" }));
		}
		return true;
	}

	if (url === "/clear") {
		response.writeHead(200, { "content-type": "application/json", "cache-control": "no-store" });
		response.end(JSON.stringify({ ok: true }));
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



// The product names the dump examples know: the effects' match arms and the
// instruments (strings is the synth dump's fallback, so it is listed by hand).
function productNames() {
	const effects = [...readFileSync(path.join(REPO, "i_am_effects/examples/dump_face.rs"), "utf8")
		.matchAll(/"([a-z0-9_]+)" =>/g)].map((match) => match[1]);
	const synths = [...readFileSync(path.join(REPO, "i_am_synths/examples/dump_synths.rs"), "utf8")
		.matchAll(/"([a-z0-9_]+)" =>/g)].map((match) => match[1]);

	return {
		effects,
		synths: [...new Set(["strings", ...synths])],
	};
}

// Builds one product's compare page into dir.
function buildCompare(name, synth, dir) {
	mkdirSync(dir, { recursive: true });
	cargo([
		"run", "-q", "-p", synth ? "i_am_synths" : "i_am_effects",
		"--example", synth ? "dump_synths" : "dump_face", "--", dir, name,
	]);
	refreshAssets(dir, "faceplate");
	copyFileSync(path.join(REPO, "i_am_patch/src/page/patcher.css"), path.join(dir, "patcher.css"));
	const product = path.join(dir, "product");
	mkdirSync(product, { recursive: true });
	copyFileSync(path.join(REPO, "i_am_web_gui/src/page/index.html"), path.join(product, "index.html"));
	writeFileSync(path.join(dir, "index.html"), COMPARE_PAGE);
}

// A page that opens one product's compare view at a time. Every product lives
// under /p/<name>/, and the server routes /face and /state by the referrer.
function galleryPage(products) {
	const rows = products
		.map(({ name, synth }) => `    <li><a href="/p/${name}/">${name}<small>${synth ? "instrument" : "effect"}</small></a></li>`)
		.join("\n");

	return `<!doctype html>
<html lang="en">
<meta charset="utf-8">
<title>Compare</title>
<style>
  body{margin:0;padding:24px;background:#141414;color:#ddd;font:13px system-ui,sans-serif}
  h1{font-size:16px;font-weight:600;margin:0 0 16px}
  ul{display:grid;grid-template-columns:repeat(auto-fill,minmax(220px,1fr));gap:6px;list-style:none;margin:0;padding:0}
  a{display:flex;justify-content:space-between;padding:8px 10px;border:1px solid #2a2a2a;border-radius:6px;color:#ddd;text-decoration:none}
  a:hover{border-color:#444;background:#1c1c1c}
  small{color:#777}
</style>
<h1>Compare: product page and patch card</h1>
<ul>
${rows}
</ul>
</html>
`;
}

// The same face, drawn the way a patcher node or a rack module draws it.
const CARD_PAGE = `<!doctype html>
<html lang="en">
<meta charset="utf-8">
<title>card</title>
<link rel="stylesheet" href="/card.css">
<link rel="stylesheet" href="/patcher.css">
<style>
  body{margin:0;min-height:100vh;display:flex;align-items:center;justify-content:center;background:#141414}
  #card-stage{padding:28px;max-width:980px;width:100%;display:flex;justify-content:center}
  #card-stage .mcard{width:100%;height:100%}
</style>
<div id="card-stage" class="card-host"></div>
<script src="/card.js"></script>
<script>
  // /face describes a product page; a card takes the flat controls list.
  function asCard(face) {
    var hidden = face.hidden || [];
    var controls = [];
    (face.sections || []).forEach(function (section) {
      section.controls = (section.params || []).filter(function (p) {
        return hidden.indexOf(p.id) < 0;
      }).map(function (p) {
        return {
          id: p.id, label: p.label, kind: p.kind, unit: p.unit,
          min: p.min, max: p.max, default: p.default,
          log: p.log, bipolar: p.bipolar, step: p.step,
          widget: p.widget, size: p.size, order: p.order, choices: p.choices
        };
      });
      controls = controls.concat(section.controls);
    });
    face.controls = controls;
    return face;
  }
  Promise.all([fetch("/face").then(r => r.json()), fetch("/state").then(r => r.json())]).then(([face, state]) => {
    const stage = document.getElementById("card-stage");
    const product = document.querySelector("iframe");
    function syncCompact() {
      const doc = product && product.contentDocument;
      if (doc && doc.querySelector(".face.compact")) stage.classList.add("compact");
    }
    if (product) {
      product.addEventListener("load", syncCompact);
      setTimeout(syncCompact, 1200);
    }
    const context = {
      rate: face.rate || 48000,
      scope: { input: null, output: null, sidechain: null },
      spectra: {},
      meter: {},
      envHistory: { input: [], output: [] },
      envWindow: 128
    };
    const card = IAmCard.faceCard({
      face: asCard(face),
      values: state,
      context: context,
      clear: !!face.clear,
      onClear: function () {
        fetch("/clear", { method: "POST" }).catch(function () {});
      },
      script: !!face.script,
      loadScript: function () {
        fetch("/script", { cache: "no-store" })
          .then(function (response) { return response.ok ? response.json() : null; })
          .then(function (state) { if (state) card.setScript(state.source || ""); })
          .catch(function () {});
      },
      applyScript: function (source) {
        fetch("/script", { method: "POST", body: source })
          .then(function (response) { return response.ok ? response.json() : null; })
          .then(function (state) {
            if (state && state.error) card.setScriptError(state.error);
            else card.setScriptError("");
          })
          .catch(function () {});
      },
      activePage: 0,
      onPage: function () {},
      onEdit: function () {}
    });
    const faceWidth = face.size ? face.size[0] : 620;
    const faceHeight = face.size ? face.size[1] : 420;
    const box = document.createElement("div");
    box.style.width = faceWidth + "px";
    box.style.height = faceHeight + "px";
    box.style.transformOrigin = "top center";
    const productFace =
      product && product.contentDocument && product.contentDocument.querySelector(".face");
    const fit = productFace
      ? productFace.getBoundingClientRect().width / faceWidth
      : Math.min((stage.clientWidth || 620) / faceWidth, 1);
    box.style.transform = "scale(" + fit + ")";
    box.appendChild(card);
    stage.appendChild(box);

    // The live routes answer with noise here, so the card's heroes move the
    // same way the product page's do.
    function levelOf(samples) {
      var energy = 0;
      for (var i = 0; i < samples.length; i++) energy += samples[i] * samples[i];
      return Math.sqrt(energy / Math.max(samples.length, 1));
    }
    function pollLive() {
      fetch("/scope", { cache: "no-store" })
        .then(function (response) { return response.arrayBuffer(); })
        .then(function (buffer) {
          var data = new Float32Array(buffer);
          var third = data.length / 3;
          if (third <= 64) return;
          context.scope.input = data.subarray(0, third);
          context.scope.output = data.subarray(third, third * 2);
          context.scope.sidechain = data.subarray(third * 2);
          context.envHistory.input.push(levelOf(context.scope.input));
          context.envHistory.output.push(levelOf(context.scope.output));
          if (context.envHistory.input.length > context.envWindow) {
            context.envHistory.input.shift();
            context.envHistory.output.shift();
          }
          card.repaintHero();
        })
        .catch(function () {})
        .then(function () { setTimeout(pollLive, 60); });
    }
    pollLive();

    // A drawn spectrum lives outside the parameter map, so the page asks for
    // it the way the product page does.
    function pollSpectrum() {
      fetch("/spectrum", { cache: "no-store" })
        .then(function (response) { return response.ok ? response.json() : null; })
        .then(function (state) {
          if (!state || !state.frames) return;
          context.spectrum = { generation: state.generation, frames: state.frames, page: 0 };
          card.repaintHero();
        })
        .catch(function () {})
        .then(function () { setTimeout(pollSpectrum, 1500); });
    }
    pollSpectrum();
  });
</script>
</html>
`;

// The product page and the card side by side, on one face and one state.
const COMPARE_PAGE = `<!doctype html>
<html lang="en">
<meta charset="utf-8">
<title>compare</title>
<link rel="stylesheet" href="/card.css">
<link rel="stylesheet" href="/patcher.css">
<style>
  html,body{margin:0;height:100%;background:#0f0f0f;color:#bbb;font:12px system-ui,sans-serif}
  #panes{display:grid;grid-template-columns:1fr 1fr;height:100%}
  .pane{display:flex;flex-direction:column;min-width:0;border-right:1px solid #222}
  .pane:last-child{border-right:0}
  .label{padding:6px 10px;color:#777;border-bottom:1px solid #222}
  .body{flex:1;min-height:0;display:flex;align-items:center;justify-content:center;padding:18px}
  iframe{width:100%;height:100%;border:0;background:transparent}
  #card-stage{width:100%;max-width:980px}
  #card-stage .mcard{width:100%;height:100%}
</style>
<div id="panes">
  <div class="pane"><div class="label">product page</div><div class="body"><iframe src="product/index.html"></iframe></div></div>
  <div class="pane"><div class="label">patch card</div><div class="body"><div id="card-stage" class="card-host"></div></div></div>
</div>
<script src="/card.js"></script>
<script>
  function asCard(face) {
    var hidden = face.hidden || [];
    var controls = [];
    (face.sections || []).forEach(function (section) {
      section.controls = (section.params || []).filter(function (p) {
        return hidden.indexOf(p.id) < 0;
      }).map(function (p) {
        return {
          id: p.id, label: p.label, kind: p.kind, unit: p.unit,
          min: p.min, max: p.max, default: p.default,
          log: p.log, bipolar: p.bipolar, step: p.step,
          widget: p.widget, size: p.size, order: p.order, choices: p.choices
        };
      });
      controls = controls.concat(section.controls);
    });
    face.controls = controls;
    return face;
  }
  Promise.all([fetch("/face").then(r => r.json()), fetch("/state").then(r => r.json())]).then(([face, state]) => {
    const stage = document.getElementById("card-stage");
    stage.style.width = Math.min(face.size ? face.size[0] : 620, stage.clientWidth || 620) + "px";
    const product = document.querySelector("iframe");
    function syncCompact() {
      const doc = product && product.contentDocument;
      if (doc && doc.querySelector(".face.compact")) stage.classList.add("compact");
    }
    if (product) {
      product.addEventListener("load", syncCompact);
      setTimeout(syncCompact, 1200);
    }
    const context = {
      rate: face.rate || 48000,
      scope: { input: null, output: null, sidechain: null },
      spectra: {},
      meter: {},
      envHistory: { input: [], output: [] },
      envWindow: 128
    };
    const card = IAmCard.faceCard({
      face: asCard(face),
      values: state,
      context: context,
      clear: !!face.clear,
      onClear: function () {
        fetch("/clear", { method: "POST" }).catch(function () {});
      },
      script: !!face.script,
      loadScript: function () {
        fetch("/script", { cache: "no-store" })
          .then(function (response) { return response.ok ? response.json() : null; })
          .then(function (state) { if (state) card.setScript(state.source || ""); })
          .catch(function () {});
      },
      applyScript: function (source) {
        fetch("/script", { method: "POST", body: source })
          .then(function (response) { return response.ok ? response.json() : null; })
          .then(function (state) {
            if (state && state.error) card.setScriptError(state.error);
            else card.setScriptError("");
          })
          .catch(function () {});
      },
      activePage: 0,
      onPage: function () {},
      onEdit: function () {}
    });
    const faceWidth = face.size ? face.size[0] : 620;
    const faceHeight = face.size ? face.size[1] : 420;
    const box = document.createElement("div");
    box.style.width = faceWidth + "px";
    box.style.height = faceHeight + "px";
    box.style.transformOrigin = "top center";
    const productFace =
      product && product.contentDocument && product.contentDocument.querySelector(".face");
    const fit = productFace
      ? productFace.getBoundingClientRect().width / faceWidth
      : Math.min((stage.clientWidth || 620) / faceWidth, 1);
    box.style.transform = "scale(" + fit + ")";
    box.appendChild(card);
    stage.appendChild(box);

    // The live routes answer with noise here, so the card's heroes move the
    // same way the product page's do.
    function levelOf(samples) {
      var energy = 0;
      for (var i = 0; i < samples.length; i++) energy += samples[i] * samples[i];
      return Math.sqrt(energy / Math.max(samples.length, 1));
    }
    function pollLive() {
      fetch("/scope", { cache: "no-store" })
        .then(function (response) { return response.arrayBuffer(); })
        .then(function (buffer) {
          var data = new Float32Array(buffer);
          var third = data.length / 3;
          if (third <= 64) return;
          context.scope.input = data.subarray(0, third);
          context.scope.output = data.subarray(third, third * 2);
          context.scope.sidechain = data.subarray(third * 2);
          context.envHistory.input.push(levelOf(context.scope.input));
          context.envHistory.output.push(levelOf(context.scope.output));
          if (context.envHistory.input.length > context.envWindow) {
            context.envHistory.input.shift();
            context.envHistory.output.shift();
          }
          card.repaintHero();
        })
        .catch(function () {})
        .then(function () { setTimeout(pollLive, 60); });
    }
    pollLive();

    // A drawn spectrum lives outside the parameter map, so the page asks for
    // it the way the product page does.
    function pollSpectrum() {
      fetch("/spectrum", { cache: "no-store" })
        .then(function (response) { return response.ok ? response.json() : null; })
        .then(function (state) {
          if (!state || !state.frames) return;
          context.spectrum = { generation: state.generation, frames: state.frames, page: 0 };
          card.repaintHero();
        })
        .catch(function () {})
        .then(function () { setTimeout(pollSpectrum, 1500); });
    }
    pollSpectrum();
  });
</script>
</html>
`;

// Serves dir until the process is killed.
function serve(dir, port, index = "index.html") {
	const root = path.resolve(dir);
	const server = createServer((request, response) => {
		const url = decodeURIComponent((request.url || "/").split("?")[0]);

		if (liveRoute(url, response, request)) return;

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


// Serves the compare gallery: every product under /p/<name>/, and the /face
// and /state a page fetches absolutely resolved by the referrer.
function serveGallery(dir, port) {
	const root = path.resolve(dir);
	const server = createServer((request, response) => {
		const url = decodeURIComponent((request.url || "/").split("?")[0]);

		if (liveRoute(url, response, request)) return;

		const referer = request.headers.referer || "";
		const match = referer.match(/\/p\/([^/]+)\//);
		const productRoot = match ? path.join(root, "p", match[1]) : null;
		const wanted = url === "/" ? "/index.html" : url.endsWith("/") ? url + "index.html" : url;
		let file = path.join(root, wanted);
		let body;

		try {
			body = readFileSync(file);
		} catch {
			body = null;
		}

		// The page's own routes are absolute; the referrer says which product.
		if (!body && productRoot) {
			try {
				file = path.join(productRoot, wanted);
				body = readFileSync(file);
			} catch {
				body = null;
			}
		}

		if (!body) {
			response.writeHead(404, { "content-type": "text/plain" });
			response.end("not in the gallery: " + wanted + "\n");
			return;
		}

		if (!path.resolve(file).startsWith(root)) {
			response.writeHead(403).end();
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
		console.log("compare gallery on http://127.0.0.1:" + port + "/  (serving " + root + ")");
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

	if (mode === "compare-all") {
		const names = productNames();
		const list = [
			...names.effects.map((name) => ({ name, synth: false })),
			...names.synths.map((name) => ({ name, synth: true })),
		];
		const limit = Number(flagValue("--limit") || 0);
		const chosen = limit > 0 ? list.slice(0, limit) : list;

		if (!noBuild) {
			mkdirSync(dir, { recursive: true });

			for (const { name, synth } of chosen) {
				buildCompare(name, synth, path.join(dir, "p", name));
				console.log("built " + name);
			}

			const first = path.join(dir, "p", chosen[0].name);
			for (const asset of ["card.css", "card.js", "kit.css", "kit.js", "patcher.css"]) {
				copyFileSync(path.join(first, asset), path.join(dir, asset));
			}

			writeFileSync(path.join(dir, "index.html"), galleryPage(chosen));
		}

		serveGallery(dir, port);
		return;
	}


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

	if (mode === "card" || mode === "compare") {
		const name = args[args.indexOf(mode) + 1];
		if (!name || name.startsWith("--")) {
			console.error(mode + " needs a name, for example: preview.mjs " + mode + " gain");
			process.exit(2);
		}

		const synth = args.includes("--synth");
		const out = path.join(dir, name);

		if (!noBuild) {
			mkdirSync(out, { recursive: true });
			if (synth) {
				cargo(["run", "-q", "-p", "i_am_synths", "--example", "dump_synths", "--", out, name]);
			} else {
				cargo(["run", "-q", "-p", "i_am_effects", "--example", "dump_face", "--", out, name]);
			}
			refreshAssets(out, "faceplate");
			// The card's tabs and body live in the patcher sheet.
			copyFileSync(
				path.join(REPO, "i_am_patch/src/page/patcher.css"),
				path.join(out, "patcher.css"),
			);

			if (mode === "card") {
				writeFileSync(path.join(out, "index.html"), CARD_PAGE);
			} else {
				// The page is one directory down, but it fetches /face, /state and
				// /kit.js from the root, which is where the face already lives.
				const product = path.join(out, "product");
				mkdirSync(product, { recursive: true });
				copyFileSync(
					path.join(REPO, "i_am_web_gui/src/page/index.html"),
					path.join(product, "index.html"),
				);
				writeFileSync(path.join(out, "index.html"), COMPARE_PAGE);
			}
			console.log("built " + out);
		}

		serve(out, port);
		return;
	}

	console.error("unknown mode: " + mode);
	console.error("usage: preview.mjs [cards|product <name>|card <name>|compare <name>|compare-all|rack|patch] [--port N] [--dir DIR] [--no-build] [--synth] [--limit N]");
	process.exit(2);
}

main();
