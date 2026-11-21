// The graph editor: every node is its own card on the canvas.
//
// A card is drawn at a base size and scaled by itself, so zooming a node scales
// its face the way the product's faceplate scales to its window. The page talks
// to /catalog, /patch and /op; the graph stays the truth.
(() => {
  const svg = document.getElementById("graph");
  const menu = document.getElementById("menu");
  const NS = "http://www.w3.org/2000/svg";

  const BASE_W = 280;
  const BASE_H = 300;

  let catalog = [];
  let patch = { nodes: [], edges: [], mods: [] };
  // What is selected: a set, plus the one clicked last. The scope strip and
  // the scale handle follow the primary; Delete and copy take the set.
  const selection = new Set();
  let primary = null;
  let selectedEdge = null;
  let selectedMod = null;
  let view = { x: 30, y: 30, zoom: 1 };
  let spaceHeld = false;
  let clipboard = null;
  let suspendSnapshot = 0;
  let drag = null;
  // Each card scales by itself: x and y are independent, so a wide filter and
  // a narrow one can sit side by side at the sizes they need.
  const scales = {};

  const cards = {};
  const pagesOf = {};

  // The host's macro values, read from /macros. A macro card draws the one its
  // index names; dragging it writes the atomics a host writes, so a drag is
  // automation rather than a graph edit.
  let macroValues = new Array(64).fill(0);
  let macroPending = null;
  let macroTimer = null;

  function queueMacro(index, value) {
    macroPending = { index, value };

    if (macroTimer) return;
    macroTimer = setTimeout(flushMacro, 30);
  }

  function flushMacro() {
    macroTimer = null;
    const pending = macroPending;
    macroPending = null;

    if (!pending) return;

    fetch("/macro?index=" + pending.index + "&value=" + pending.value, { method: "POST" })
      .catch(() => {});
  }

  async function readMacros() {
    try {
      const answer = await (await fetch("/macros", { cache: "no-store" })).json();
      if (answer && Array.isArray(answer.values)) macroValues = answer.values.slice(0, 64);
    } catch {
      // The next tick tries again.
    }
  }

  function sourceName(id) {
    const source = nodeOf(id);
    const kind = source ? kindOf(source.kind) : null;
    return kind ? kind.name : "source";
  }

  function modsOf(id) {
    return patch.mods
      .map((mod, index) => ({
        slot: index,
        from: mod.from,
        to: mod.to,
        depth: mod.depth,
        mode: mod.mode || "center",
      }))
      .filter((mod) => mod.to[0] === id);
  }

  // The rack's source pictures, ported: a 300x70 drawing stretched over the
  // card's hero, with the live trace and the playhead on top.
  const SOURCE_KINDS = ["lfo", "envelope", "random", "table", "pitch", "midi", "map"];

  /// A table node draws a product's table page, so the product's own wiring
  /// (imports, hero editors, routes) has to reach it the same way.
  function tableProduct(kind) {
    return kind && typeof kind.id === "string" && kind.id.startsWith("table:")
      ? kind.id.slice("table:".length)
      : "";
  }

  function isProductCard(kind, product) {
    return kind.id === "synth:" + product || tableProduct(kind) === product;
  }

  // A source card's trace outlives the card. A rebuild happens on every
  // parameter change, and a history kept on the entry would restart the picture
  // from nothing under the hand.
  const sourceHistories = new Map();

  // How far each source's block history has been read. The server sends only
  // the blocks after this, so a poll is a handful of values, not the whole ring.
  const sourceCursors = new Map();

  function historyOf(node) {
    let history = sourceHistories.get(node);

    if (!history) {
      history = [];
      sourceHistories.set(node, history);
    }

    return history;
  }

  /// The mapping a MIDI source starts with: the note itself, unchanged.
  const MIDI_MAPPING = [
    { time: 0, value: 0, bend: 0 },
    { time: 1, value: 1, bend: 0 },
  ];

  function rackNode(name, attrs) {
    const node = document.createElementNS(NS, name);

    for (const [key, value] of Object.entries(attrs || {})) {
      node.setAttribute(key, String(value));
    }

    return node;
  }

  

  function drawSourceStatic(svg, node, kind, dots, values, points) {
    const params = values || (node && node.params) || {};

    if (kind === "lfo") {
      const shape = Math.round(params.shape ?? 0);
      const curve = points || (node && node.points) || [];

      IAmCard.sources.drawPreview(svg, "lfo", {
        points: curve.length >= 2 ? curve : IAmCard.sources.preset(shape),
        shape,
      }, dots);
    } else if (kind === "table") {
      IAmCard.sources.drawPreview(svg, "table", {
        shape: Math.round(params.shape ?? 0),
        morph: params.morph ?? 0,
      }, dots);
    } else if (kind === "midi" || kind === "map") {
      // A mapping, not a value: an input goes in on the x axis and the control
      // value comes out on the y. A MIDI source maps a note; a map node maps
      // whatever control reaches its input port.
      if (dots) dots.textContent = "";
      const ink = kind === "map" ? "#8ac926" : "#ff6f91";
      const halo = kind === "map" ? "#d6efb0" : "#ffe1ea";
      svg.appendChild(rackNode("line", { x1: 8, y1: 60, x2: 292, y2: 60, stroke: "#333", "stroke-width": 1, "stroke-dasharray": "2 4" }));
      svg.appendChild(rackNode("line", { x1: 8, y1: 8, x2: 8, y2: 60, stroke: "#333", "stroke-width": 1, "stroke-dasharray": "2 4" }));

      const curve = points && points.length >= 2 ? points : MIDI_MAPPING;
      const bend = IAmCard.sources.bend;
      let d = "";

      // The segment bends toward its end value, so the picture and the engine
      // read the same curve. A straight segment is just the same formula with
      // a zero bend.
      for (let index = 0; index < curve.length - 1; index++) {
        const from = curve[index];
        const to = curve[index + 1];

        for (let step = 0; step <= 14; step++) {
          const unit = step / 14;
          const time = from.time + (to.time - from.time) * unit;
          const value = from.value + (to.value - from.value) * bend(unit, from.bend || 0);

          d += (d ? " L " : "M ") + (8 + time * 284).toFixed(1) + " " +
            (60 - value * 52).toFixed(1);
        }
      }

      svg.appendChild(rackNode("path", {
        d,
        fill: "none",
        stroke: ink,
        "stroke-width": 2,
        "stroke-linecap": "round",
        "vector-effect": "non-scaling-stroke",
      }));

      // The points are handles: they live in the HTML layer so the card's
      // aspect cannot squash them, the way the LFO's own points do.
      for (const point of curve) {
        if (!dots) break;

        const dot = document.createElement("div");
        dot.className = "hero-pip";
        dot.style.left = ((8 + point.time * 284) / 3).toFixed(2) + "%";
        dot.style.top = ((60 - point.value * 52) / 0.7).toFixed(2) + "%";
        dot.style.width = "7px";
        dot.style.height = "7px";
        dot.style.background = halo;
        dot.style.border = "1.4px solid " + ink;
        dots.appendChild(dot);
      }

      // A hollow dot at each segment's middle is the bend handle; editLfo's
      // bendAt finds it at the same place.
      for (let index = 0; index < curve.length - 1; index++) {
        if (!dots) break;

        const from = curve[index];
        const to = curve[index + 1];
        const time = (from.time + to.time) / 2;
        const value = from.value + (to.value - from.value) * bend(0.5, from.bend || 0);
        const handle = document.createElement("div");

        handle.className = "hero-pip";
        handle.style.left = ((8 + time * 284) / 3).toFixed(2) + "%";
        handle.style.top = ((60 - value * 52) / 0.7).toFixed(2) + "%";
        handle.style.width = "5px";
        handle.style.height = "5px";
        handle.style.background = "transparent";
        handle.style.border = "1.2px solid " + ink;
        dots.appendChild(handle);
      }
    } else {
      IAmCard.sources.drawPreview(svg, kind, {}, dots);
    }
  }

  // The live overlay follows the rack's: a playhead for the LFO and the table,
  // a trace for the others. It sits in its own layer, so a poll repaints only
  // this and leaves the static drawing and the editor's handlers alone.
  function drawSourceLive(entry) {
    const layer = entry.live;

    if (!layer) return;

    layer.textContent = "";

    if (entry.liveDots) entry.liveDots.textContent = "";

    if (entry.kind.id === "midi") {
      // The mapping is note or velocity -> value: the live mark is the point
      // where the input the node is reading meets the value it sends. The
      // crosshair ties that point back to both axes.
      const value = typeof entry.lastValue === "number" ? entry.lastValue : 0;
      const y = 60 - value * 52;
      const input = typeof entry.input === "number" ? entry.input : null;
      const x = input === null ? null : 8 + input * 284;

      if (x !== null) {
        layer.appendChild(rackNode("line", { x1: x.toFixed(1), y1: 8, x2: x.toFixed(1), y2: 60, stroke: "#ff6f91", "stroke-opacity": ".22", "stroke-width": 1, "stroke-dasharray": "3 4" }));
      }

      layer.appendChild(rackNode("line", { x1: 8, y1: y.toFixed(1), x2: 292, y2: y.toFixed(1), stroke: "#ff6f91", "stroke-opacity": ".38", "stroke-width": 1, "stroke-dasharray": "3 4" }));

      if (entry.liveDots) {
        const dot = document.createElement("div");
        dot.className = "hero-pip";
        dot.style.left = ((x === null ? 292 : x) / 3).toFixed(2) + "%";
        dot.style.top = (y / 0.7).toFixed(2) + "%";
        dot.style.width = "7px";
        dot.style.height = "7px";
        // The mapping's own points are light handles with a pink ring; the
        // live point is solid so the two do not read as the same mark.
        dot.style.background = "#ff6f91";
        dot.style.border = "1.4px solid #1a1a1a";
        entry.liveDots.appendChild(dot);

        // The level line is a bare height; the number makes it a value, as the
        // other source cards' live label does.
        if (typeof entry.lastValue === "number") {
          const label = document.createElement("div");
          label.className = "hero-label";
          label.style.left = "10px";
          label.style.top = "auto";
          label.style.bottom = "2px";

          // The mapping's input is a MIDI number (a note or a velocity), and
          // the output is the value it sends; both in one readout.
          const input = typeof entry.input === "number"
            ? Math.round(entry.input * 127) + " · "
            : "";

          label.textContent = input + entry.lastValue.toFixed(2);
          entry.liveDots.appendChild(label);
        }
      }

      return;
    }

    if (!entry.history || !entry.history.length) return;

    IAmCard.sources.drawLive(layer, entry.kind.id, {
      history: { patch: entry.history },
      phases: { patch: typeof entry.phase === "number" ? entry.phase : 0 },
      values: { patch: typeof entry.lastValue === "number" ? entry.lastValue : 0 },
    }, "patch", false, entry.liveDots);

    // A trace that sits at one level says nothing on its own: the number is
    // what tells a held value from a flat line.
    if (entry.liveDots && typeof entry.lastValue === "number") {
      const label = document.createElement("div");
      label.className = "hero-label";
      label.style.left = "10px";
      label.style.top = "auto";
      label.style.bottom = "2px";
      label.textContent = entry.lastValue.toFixed(2);
      entry.liveDots.appendChild(label);
    }
  }

  const CARD_HEROES = {
    lfo: "patch-source",
    envelope: "patch-source",
    random: "patch-source",
    table: "patch-source",
    pitch: "patch-source",
    midi: "patch-source",
    map: "patch-source",
  };

  // The source hero is a drawing surface the page fills after every repaint.
  // The picture is stretched over the card, so the dots it would draw (an LFO's
  // points, a playhead) live in their own HTML layer above it, where the card's
  // aspect cannot squash them.
  if (window.IAmCard) {
    window.IAmCard.registerHero("patch-source", () =>
      '<div class="source-hero">' +
        '<svg viewBox="0 0 300 70" preserveAspectRatio="none" class="rack-hero"></svg>' +
        '<div class="source-dots"></div>' +
        '<div class="source-live-dots"></div>' +
      '</div>',
    );
  }


  // Heroes for the constructor nodes. The card kit draws product pictures from
  // their own metadata; these nodes have no product, so the page draws what the
  // node is set to: the table, the phase map, the modulation, the envelope.
  function patchSvg(content) {
    // `mcard-graph` makes the svg take the hero's full box; without it the
    // waveform keeps its own aspect and the box's lower half stays empty.
    return '<svg class="mcard-graph" viewBox="0 0 520 160" preserveAspectRatio="none">' + content + '</svg>';
  }

  function patchValue(values, name, fallback) {
    var value = values && values[name];
    return typeof value === "number" ? value : fallback;
  }

  function patchTrace(points) {
    var path = "";

    for (var i = 0; i < points.length; i++) {
      path += (i ? " L " : "M ") + points[i][0].toFixed(1) + " " + points[i][1].toFixed(1);
    }

    return path;
  }

  /// A sampler over one cycle of samples, linearly interpolated.
  function patchSamples(values) {
    var length = values.length;

    return function (t) {
      var x = patchFrac(t) * length;
      var index = Math.floor(x) % length;
      var next = (index + 1) % length;
      var fraction = x - Math.floor(x);

      return values[index] + (values[next] - values[index]) * fraction;
    };
  }

  /// The sample function of a node: the engine's own rendering when it has been
  /// fetched, a plain table's shape otherwise.
  function patchSourceShape(source) {
    if (!source) return null;

    const samples = source.id !== undefined ? waveCache.get(source.id) : null;

    if (samples) return patchSamples(samples);

    const kind = source.kind;
    const shape = Math.round((source.params && source.params.shape) || 0);

    if (kind === "table:shape" || kind === "osc:shape") {
      return function (t) {
        return patchShapeValue(shape, t);
      };
    }

    return null;
  }

  function patchShapeValue(shape, t) {
    var cycle = t - Math.floor(t);

    if (shape === 1) return 2 * Math.abs(2 * cycle - 1) - 1;
    if (shape === 2) return 2 * cycle - 1;
    if (shape === 3) return cycle < 0.5 ? 1 : -1;
    if (shape === 4) {
      var noise = Math.sin(cycle * 12345.678) * 43758.5453;
      return (noise - Math.floor(noise)) * 2 - 1;
    }

    return Math.sin(cycle * Math.PI * 2);
  }

  function patchTableHero(values) {
    var shapes = ["SINE", "TRIANGLE", "SAW", "SQUARE", "NOISE"];
    var shape = Math.round(patchValue(values, "shape", 0));
    var points = [];

    for (var i = 0; i <= 240; i++) {
      var t = i / 240;
      points.push([20 + t * 480, 80 - patchShapeValue(shape, t) * 52]);
    }

    return patchSvg(
      '<line x1="20" y1="80" x2="500" y2="80" stroke="#2f2f2f" stroke-dasharray="3 5"/>' +
      '<path d="' + patchTrace(points) + '" fill="none" stroke="var(--accent)" stroke-width="2" vector-effect="non-scaling-stroke"/>'
    ) + '<div class="hero-label" style="left:2%;top:2%">' + shapes[shape] + '</div>';
  }

  /// The table a warp reads, sampled through the warp: the actual waveform, not
  /// the phase map that produced it.
  function patchWarpHero(mode, values, context) {
    var amount = patchValue(values, "amount", 0);
    var power = Math.max(patchValue(values, "power", 1), 0.25);
    var ratio = Math.max(patchValue(values, "ratio", 1), 1);
    var steps = Math.max(patchValue(values, "steps", 1), 1);
    var shape = patchSourceShape(context && context.source(0)) ||
      function (t) { return patchShapeValue(0, t); };
    var points = [];

    for (var i = 0; i <= 240; i++) {
      var t = i / 240;
      var folded = 1 - Math.abs(2 * t - 1);
      var warped = mode === "bend" ? Math.pow(t, power)
        : mode === "mirror" ? t + (folded - t) * amount
        : mode === "sync" ? t * ratio - Math.floor(t * ratio)
        : mode === "quantize" ? Math.floor(t * steps) / steps
        : t + (0.5 + 0.5 * Math.sin(t * Math.PI * 2) - t) * amount;

      points.push([20 + t * 480, 80 - shape(warped) * 56]);
    }

    var names = { bend: "BEND", mirror: "MIRROR", sync: "SYNC", quantize: "QUANTIZE", remap: "REMAP" };

    return patchSvg(
      '<line x1="20" y1="80" x2="500" y2="80" stroke="#2f2f2f" stroke-dasharray="3 5"/>' +
      '<path d="' + patchTrace(points) + '" fill="none" stroke="var(--accent)" stroke-width="2" vector-effect="non-scaling-stroke"/>'
    ) + '<div class="hero-label" style="left:2%;top:2%">' + names[mode] + '</div>';
  }

  function patchFrac(t) {
    return t - Math.floor(t);
  }

  /// The two tables a modulation reads, combined the way the node combines them.
  function patchModHero(mode, values, context) {
    var depth = patchValue(values, "depth", 0);
    var ratio = Math.max(patchValue(values, "ratio", 1), 0.0625);
    // A modulator slower than the carrier moves almost not at all over two
    // cycles, so the window grows until one modulator cycle fits.
    var cycles = Math.min(Math.max(2, Math.ceil(1 / ratio)), 32);
    var sine = function (t) { return Math.sin(t * Math.PI * 2); };
    var carrier = patchSourceShape(context && context.source(0)) || sine;
    var modulator = patchSourceShape(context && context.source(1)) || sine;
    var points = [];
    var steps = 480;
    var integral = 0;

    for (var i = 0; i <= steps; i++) {
      var t = i / steps * cycles;
      var m = modulator(patchFrac(t * ratio));
      var value;

      if (mode === "am") value = carrier(patchFrac(t)) * (1 + depth * m);
      else if (mode === "rm") value = carrier(patchFrac(t)) * (1 - depth + depth * m);
      else if (mode === "pm") value = carrier(patchFrac(t + depth * m));
      else {
        integral += m * (cycles / steps);
        value = carrier(patchFrac(t + depth * integral));
      }

      points.push([20 + i / steps * 480, 80 - value * 56]);
    }

    var names = { am: "AM", rm: "RM", pm: "PM", fm: "FM" };

    return patchSvg(
      '<line x1="20" y1="80" x2="500" y2="80" stroke="#2f2f2f" stroke-dasharray="3 5"/>' +
      '<path d="' + patchTrace(points) + '" fill="none" stroke="var(--accent)" stroke-width="2" vector-effect="non-scaling-stroke"/>'
    ) + '<div class="hero-label" style="left:2%;top:2%">' +
        names[mode] +
        " · " +
        depth.toFixed(2) +
        (Math.abs(ratio - 1) > 0.001 ? " × " + ratio.toFixed(2) : "") +
        "</div>";
  }

  function patchEnvelopeHero(values) {
    var attack = Math.min(Math.max(patchValue(values, "attack_time", 0.01), 0.001), 0.9);
    var decay = Math.min(Math.max(patchValue(values, "decay_time", 0.2), 0.001), 0.9);
    var sustain = Math.min(Math.max(patchValue(values, "sustain_level", 0.7), 0), 1);
    var release = Math.min(Math.max(patchValue(values, "release_time", 0.3), 0.001), 1);
    var x = function (time) { return 20 + time * 300; };
    var y = function (level) { return 148 - level * 120; };
    var points = [
      [x(0), y(0)],
      [x(attack), y(1)],
      [x(Math.min(attack + decay, 0.98)), y(sustain)],
      [x(1), y(sustain)],
      [x(1 + release), y(0)]
    ];

    return patchSvg(
      '<line x1="20" y1="148" x2="500" y2="148" stroke="#2a2a2a"/>' +
      '<path d="' + patchTrace(points) + ' L 20 148 Z" fill="var(--accent)" fill-opacity=".08"/>' +
      '<path d="' + patchTrace(points) + '" fill="none" stroke="var(--accent)" stroke-width="2" vector-effect="non-scaling-stroke"/>'
    ) + '<div class="hero-label" style="left:2%;top:2%">ADSR</div>';
  }

  function patchProductHero(name, shape) {
    return function () {
      var points = [];

      for (var i = 0; i <= 240; i++) {
        var t = i / 240;
        points.push([20 + t * 480, 80 - patchShapeValue(shape, t) * 50]);
      }

      return patchSvg(
        '<line x1="20" y1="80" x2="500" y2="80" stroke="#2f2f2f" stroke-dasharray="3 5"/>' +
        '<path d="' + patchTrace(points) + '" fill="none" stroke="var(--accent)" stroke-width="2" vector-effect="non-scaling-stroke"/>'
      ) + '<div class="hero-label" style="left:2%;top:2%">' + name + '</div>';
    };
  }

  if (window.IAmCard) {
    IAmCard.registerHero("patch-wavetable", patchTableHero);
    IAmCard.registerHero("patch-warp-bend", function (values, context) { return patchWarpHero("bend", values, context); });
    IAmCard.registerHero("patch-warp-mirror", function (values, context) { return patchWarpHero("mirror", values, context); });
    IAmCard.registerHero("patch-warp-sync", function (values, context) { return patchWarpHero("sync", values, context); });
    IAmCard.registerHero("patch-warp-quantize", function (values, context) { return patchWarpHero("quantize", values, context); });
    IAmCard.registerHero("patch-warp-remap", function (values, context) { return patchWarpHero("remap", values, context); });
    IAmCard.registerHero("patch-mod-am", function (values, context) { return patchModHero("am", values, context); });
    IAmCard.registerHero("patch-mod-rm", function (values, context) { return patchModHero("rm", values, context); });
    IAmCard.registerHero("patch-mod-pm", function (values, context) { return patchModHero("pm", values, context); });
    IAmCard.registerHero("patch-mod-fm", function (values, context) { return patchModHero("fm", values, context); });
    IAmCard.registerHero("patch-envelope", patchEnvelopeHero);
    // The envelope's shape page draws the oscillator it is actually fed, not a
    // fixed sine: the source's kind and parameters decide the picture.
    IAmCard.registerHero("patch-osc", function (values, context) {
      const source = context && context.oscSource;

      if (!source) return patchProductHero("OSC", 0)(values, context);

      const kind = source.kind;
      const params = source.params || {};
      const sampled = waveCache.get(source.id);

      // The engine's own cycle beats any hand-drawn guess.
      if (sampled) {
        const sample = patchSamples(sampled);
        const points = [];

        for (let index = 0; index <= 240; index++) {
          const t = index / 240;
          points.push([20 + t * 480, 80 - sample(t) * 56]);
        }

        return patchSvg(
          '<line x1="20" y1="80" x2="500" y2="80" stroke="#2f2f2f" stroke-dasharray="3 5"/>' +
          '<path d="' + patchTrace(points) + '" fill="none" stroke="var(--accent)" stroke-width="2" vector-effect="non-scaling-stroke"/>'
        ) + '<div class="hero-label" style="left:2%;top:2%">' +
          String(kind.split(":")[1] || kind).toUpperCase() + '</div>';
      }

      if (kind === "table:shape" || kind === "osc:shape") return patchTableHero(params);

      const mode = kind.indexOf("osc:") === 0 ? kind.slice(4) : "";

      if (["am", "rm", "pm", "fm"].includes(mode)) return patchModHero(mode, params, context);

      if (kind.indexOf("table:") === 0) {
        const product = kind.slice(6);
        const heroes = { shapes: "shape", additive: "partials", spectrum: "spectra" };
        const hero = heroes[product];

        if (hero) {
          const html = IAmCard.hero(hero, params, context);

          if (html) return html;
        }
      }

      return patchProductHero("OSC", 0)(values, context);
    });
    IAmCard.registerHero("patch-osc-shapes", patchProductHero("SHAPES", 0));
    IAmCard.registerHero("patch-osc-additive", patchProductHero("ADDITIVE", 2));
    IAmCard.registerHero("patch-osc-spectrum", patchProductHero("SPECTRUM", 1));
    IAmCard.registerHero("patch-osc-script", patchProductHero("SCRIPT", 3));
    IAmCard.registerHero("patch-osc-canvas", patchProductHero("CANVAS", 4));
    IAmCard.registerHero("patch-canvas-table", patchProductHero("CANVAS", 4));
  }

  CARD_HEROES["table:shape"] = "patch-wavetable";
  CARD_HEROES["table:canvas"] = "patch-canvas-table";
  CARD_HEROES["osc:shape"] = "patch-wavetable";
  CARD_HEROES["osc:am"] = "patch-mod-am";
  CARD_HEROES["osc:rm"] = "patch-mod-rm";
  CARD_HEROES["osc:pm"] = "patch-mod-pm";
  CARD_HEROES["osc:fm"] = "patch-mod-fm";
  CARD_HEROES["warp:bend"] = "patch-warp-bend";
  CARD_HEROES["warp:mirror"] = "patch-warp-mirror";
  CARD_HEROES["warp:sync"] = "patch-warp-sync";
  CARD_HEROES["warp:quantize"] = "patch-warp-quantize";
  CARD_HEROES["warp:remap"] = "patch-warp-remap";
  CARD_HEROES["adsr"] = "patch-envelope";
  CARD_HEROES["table:shapes"] = "patch-osc-shapes";
  CARD_HEROES["table:additive"] = "patch-osc-additive";
  CARD_HEROES["table:spectrum"] = "patch-osc-spectrum";
  CARD_HEROES["table:script"] = "patch-osc-script";

  // A source card's history comes from the engine's tap for that node, and a
  // tap exists only after it is asked for once.
  const tapped = new Set();

  function tapSource(id) {
    if (tapped.has(id)) return;
    tapped.add(id);
    fetch(`/scope?node=${id}`, { cache: "no-store" }).catch(() => {});
  }

  // `values` is the card's own control map, passed when a control moved: the
  // page's copy of the node is only refreshed by the next full redraw, so the
  // hero has to be drawn from what the control just said.
  function drawSourceCard(entry, values) {
    if (!entry || !entry.card || !entry.kind) return;

    tapSource(entry.node.id);

    const svg = entry.card.querySelector(".mcard-hero svg");
    if (!svg) return;

    const dots = entry.card.querySelector(".mcard-hero .source-dots");
    const liveDots = entry.card.querySelector(".mcard-hero .source-live-dots");

    entry.dots = dots;
    entry.liveDots = liveDots;

    if (dots) dots.textContent = "";

    const params = Object.assign({}, (entry.node && entry.node.params) || {}, values || {});
    const midiMode = Math.round(params.mode ?? 0);
    const velocity = entry.kind.id === "midi" && midiMode === 1;
    const active = entry.kind.id === "midi"
      ? ((entry.node && (velocity ? entry.node.velocity_points : entry.node.points)) || [])
      : ((entry.node && entry.node.points) || []);
    const pasted = active.length >= 2;
    // A pasted curve wins; otherwise the shape decides. The derived curve stays
    // local: writing it back into the node would outrank the shape for good and
    // make the page disagree with the server on every poll.
    const points = pasted
      ? active
      : entry.kind.id === "midi" || entry.kind.id === "map"
        ? MIDI_MAPPING
        : IAmCard.sources.preset(Math.round(params.shape ?? 0));

    svg.textContent = "";
    drawSourceStatic(svg, entry.node, entry.kind.id, dots, params, points);

    if (entry.kind.id === "lfo") {
      IAmCard.sources.editLfo(svg, { points: points.slice() }, {
        shape: params.shape,
        dots,
        commit: (next) => {
          op({
            op: "set_points",
            id: entry.node.id,
            points: next.map((point) => [point.time, point.value, point.bend || 0]),
          }, false);
        },
        onStart: () => { editingPoints = true; },
        onEnd: () => { editingPoints = false; },
      });
    } else if (entry.kind.id === "midi") {
      // The mapping is the node's curve: a point is a note (or a velocity) and
      // the value it sends, so dragging one is how the map is edited. The two
      // modes keep separate point lists; the bend edits the segment after it.
      IAmCard.sources.editLfo(svg, { points: points.slice() }, {
        dots,
        paint: (next) => {
          svg.textContent = "";
          drawSourceStatic(svg, entry.node, "midi", dots, params, next);
          entry.live = el("g", { class: "source-live" });
          svg.appendChild(entry.live);
          drawSourceLive(entry);
        },
        commit: (next) => {
          op({
            op: "set_points",
            id: entry.node.id,
            curve: velocity ? "velocity" : "note",
            points: next.map((point) => [point.time, point.value, point.bend || 0]),
          }, false);
        },
        onStart: () => { editingPoints = true; },
        onEnd: () => { editingPoints = false; },
      });
    } else if (entry.kind.id === "map") {
      // The curve is the whole node, so dragging a point is how the mapping is
      // written: the engine reads the same points when it next builds.
      IAmCard.sources.editLfo(svg, { points: points.slice() }, {
        dots,
        paint: (next) => {
          svg.textContent = "";
          drawSourceStatic(svg, entry.node, "map", dots, params, next);
        },
        commit: (next) => {
          op({
            op: "set_points",
            id: entry.node.id,
            points: next.map((point) => [point.time, point.value, point.bend || 0]),
          }, false);
        },
        onStart: () => { editingPoints = true; },
        onEnd: () => { editingPoints = false; },
      });
    }

    // Last, so the editor's repaint of the points does not erase the playhead.
    // A MIDI card's editor paints its own live layer while it draws the
    // mapping; drop that one before adding the layer the poll updates, or the
    // old level line stays frozen under the moving one.
    for (const stale of svg.querySelectorAll(".source-live")) stale.remove();

    entry.live = el("g", { class: "source-live" });
    svg.appendChild(entry.live);
    drawSourceLive(entry);
  }

  // The rack's source cards put these choices in the header instead of a knob.
  const GENERIC_HEADER = {
    lfo: ["shape", "sync", "one_shot"],
    table: ["shape", "sync"],
    random: ["sync"],
    midi: ["mode"],
  };
  const NODE_NOTE = {
    "input:midi": "Host MIDI input",
    "input:audio": "Host audio input",
    "input:sidechain": "Host sidechain input",
    "out:audio": "Host audio output",
    "voice:in": "A note starts the voice group here",
    "voice:out": "The voice group's audio leaves here",
    "split:ms": "Mid and side, each one mono",
    "split:lr": "Left and right, each copied into both channels",
    "mix:ms": "Mid and side back into left and right",
    "mix:lr": "Left and right back into a stereo pair",
    "mix:bands": "Low, mid and high added back together",
  };

  function labelOf(key) {
    return key.replace(/_/g, " ").replace(/\b\w/g, (letter) => letter.toUpperCase());
  }


  function genericCard(node, kind, w, h) {
    let entry = cards[node.id];
    const mods = modsOf(node.id);

    // A macro's value is a host parameter, not a graph one: the card carries a
    // control for it anyway, and an edit of that control goes to /macro, the same
    // atomic a host writes. It is in the rebuild key so a host-driven move shows.
    const macroIndex = kind.id === "macro"
      ? Math.min(64, Math.max(1, Math.round(node.params.index ?? 1)))
      : 0;
    const macroLevel = macroIndex ? (macroValues[macroIndex - 1] || 0) : 0;

    const key = JSON.stringify([
      mods.map((mod) => [mod.slot, mod.to[1], Math.round(mod.depth * 1000), mod.mode]),
      node.points || [],
      node.params,
      macroLevel,
    ]);

    if (entry && (entry.key !== key || entry.kindId !== node.kind)) {
      entry.host.remove();
      delete cards[node.id];
      entry = null;
    }

    if (entry) {
      entry.node = node;
      entry.kind = kind;
      entry.host.style.setProperty(
        "--k",
        String(Math.max(0.05, Math.min(w / baseWidth(node), h / baseHeight(node)))),
      );
      entry.host.classList.toggle("compact", w < 430);
      // Only a source card's picture is redrawn here. Every other card keeps the
      // hero its kind registered; painting the source picture over it would wipe
      // that hero on the next redraw.
      if (SOURCE_KINDS.includes(kind.id)) drawSourceCard(entry);

      return entry.host;
    }

    const context = contextOf(node);
    const meta = Object.fromEntries((kind.param_meta || []).map((item) => [item.name, item]));
    const specs = (kind.params || []).map(([id, fallback]) => {
      // The catalog carries the range the DSP itself declares; the fallbacks
      // only matter for a kind whose parameters were written by hand.
      const range = meta[id] || {};
      const choices = range.choices || null;
      const span = Math.max(0.001, Math.abs(fallback) || 1);
      const control = {
        id,
        label: labelOf(id),
        kind: range.boolean ? "bool" : choices || range.step === 1 ? "int" : "float",
        min: range.min ?? (range.boolean ? 0 : -span * 2),
        max: range.max ?? (choices ? choices.length - 1 : span * 2),
        default: fallback,
        log: !!range.log,
        step: choices ? 1 : range.step,
        choices,
      };

      // A depth is a share of the parameter's own range, which is the unit the
      // card's ring already works in: a logarithmic control then swings by a
      // ratio, the way its knob moves.
      const assignments = modsOf(node.id)
        .filter((mod) => mod.to[1] === control.id)
        .map((mod) => ({
          slot: mod.slot,
          label: sourceName(mod.from[0]),
          depth: mod.depth,
          mode: mod.mode === "base" ? "frombase" : "centered",
        }));

      return IAmCard.modulate(control, {
        assignments,
        onDepth: (slot, next) => {
          op({ op: "mod_depth", index: slot, depth: next }, false);
        },
        onMode: (slots, next) => {
          op({ op: "mod_mode", index: slots[0], mode: next === "center" ? "center" : "base" }, false);
        },
        // Ctrl + right click on the knob: every source comes off it.
        onClearModulations: (slots) => {
          for (const slot of slots) op({ op: "unmodulate", index: slot }, false);
        },
        dropTarget: node.id + ":" + id,
      });
    });

    // The macro's value rides beside the node's own parameters: it is a host
    // parameter, so it is written through /macro and never through `op param`.
    if (macroIndex) {
      specs.push({
        id: "value",
        label: "Value",
        kind: "float",
        min: 0,
        max: 1,
        default: macroLevel,
        log: false,
        step: 0.01,
        choices: null,
      });
    }

    const host = document.createElement("div");
    host.className = "card-host";
    const scale = document.createElement("div");
    scale.className = "card-scale";
    scale.style.width = `${baseWidth(node)}px`;
    scale.style.height = `${baseHeight(node)}px`;
    scale.style.flexBasis = `${baseWidth(node)}px`;

    const card = window.IAmCard.create({
      title: kind.name,
      accent: kind.accent || "#e07a2f",
      hero: CARD_HEROES[kind.id] || null,
      header: GENERIC_HEADER[kind.id] || [],
      switches: kind.switches || [],
      controls: specs,
      values: {
        ...node.params,
        ...(macroIndex ? { value: macroLevel } : {}),
        points: JSON.stringify((node.points || []).map((point) => (Array.isArray(point) ? point : [point.time, point.value, point.bend || 0]))),
      },
      context,
      onEdit: (identifier, value) => {
        // The macro value is not in the graph: it goes to the host parameter the
        // node names, so a drag lands in the automation lane the host reads.
        if (macroIndex && identifier === "value") {
          macroValues[macroIndex - 1] = value;
          queueMacro(macroIndex, value);
          return;
        }

        op({ op: "param", id: node.id, key: identifier, value }, false);
      },
    });

    if (!specs.length) {
      const body = card.querySelector(".mcard-body");
      const note = NODE_NOTE[kind.id];

      if (body && note) {
        const row = document.createElement("div");
        row.className = "card-note";
        row.textContent = note;
        body.appendChild(row);
      }
    }

    scale.appendChild(card);
    host.appendChild(scale);
    entry = { host, key, kindId: node.kind, context, card, node, kind, samples: null, phase: null };
    cards[node.id] = entry;

    // A node that already holds a file shows it as soon as the card is built,
    // not only right after the drop that loaded it.
    if (IMPORT_ROUTES[kind.id]) refreshImport(entry, IMPORT_ROUTES[kind.id].path);

    if (SOURCE_KINDS.includes(kind.id)) {
      const repaint = card.repaintHero;
      card.repaintHero = function (values) {
        repaint(values);
        drawSourceCard(entry, values);
      };
      drawSourceCard(entry);
    }
    entry.host.style.setProperty(
      "--k",
      String(Math.max(0.2, Math.min(w / baseWidth(node), h / baseHeight(node)))),
    );
    entry.host.classList.toggle("compact", w < 430);

    return entry.host;
  }

  // A recorded note dropped or picked on a Modal card: the page posts it to
  // the node's route, which fits it and selects the result.
  let sampleInput = null;

  function pickSampleFile(onFile) {
    if (!sampleInput) {
      sampleInput = document.createElement("input");
      sampleInput.type = "file";
      sampleInput.accept = ".wav,.aif,.aiff";
      sampleInput.style.display = "none";
      document.body.appendChild(sampleInput);
    }

    sampleInput.onchange = () => {
      const file = sampleInput.files && sampleInput.files[0];
      sampleInput.value = "";
      if (file) onFile(file);
    };
    sampleInput.click();
  }

  function postSample(node, file) {
    const reader = new FileReader();

    reader.onload = async () => {
      const response = await fetch(
        "/sample?node=" + node + "&name=" + encodeURIComponent(file.name),
        { method: "POST", body: reader.result },
      );
      const result = response.ok ? await response.json() : null;

      if (result && result.ok) {
        patch = readPatch(await (await fetch("/patch")).json());
        draw();
      } else {
        showStatus((result && result.error) || "the sample could not be fitted");
      }
    };
    reader.readAsArrayBuffer(file);
  }

  // The kinds whose card takes a file, and the route that loads it.
  const IMPORT_ROUTES = {
    "synth:sampler": { path: "sampler", hint: "Load a sample: click or drop a .wav" },
    "synth:shapes": { path: "waves", hint: "Build the table from a .wav: click or drop a file" },
    "table:shapes": { path: "waves", hint: "Build the table from a .wav: click or drop a file" },
    "mod:convolver": { path: "ir", hint: "Load an impulse response: click or drop a file" },
  };

  /// Posts a file to a node's import route, then redraws what the card shows.
  function postImport(node, path, file) {
    const reader = new FileReader();

    reader.onload = async () => {
      const entry = cards[node];
      const mode = entry && entry.context && entry.context.waveMode;
      const response = await fetch(
        "/" + path + "?node=" + node + "&name=" + encodeURIComponent(file.name) +
          (mode ? "&mode=" + encodeURIComponent(mode) : ""),
        { method: "POST", body: reader.result },
      );
      const result = response.ok ? await response.json() : null;

      if (!result || !result.ok) {
        showStatus((result && result.error) || "the file could not be loaded");
        return;
      }

      showStatus("");
      await refreshImport(cards[node], path);
    };
    reader.readAsArrayBuffer(file);
  }

  /// The live state a card's heroes read, kept across the rebuilds of that card.
  const cardContexts = new Map();

  function contextOf(node) {
    let context = cardContexts.get(node.id);

    if (!context) {
      context = {
        rate: 48000,
        scope: { input: null, output: null },
        spectra: {},
        eqSpectrum: {},
      };

      // What an envelope card draws in its Shape hero: whatever node feeds its
      // oscillator input, read when the picture is painted so it never lags the
      // source's own parameters behind.
      Object.defineProperty(context, "oscSource", {
        get() {
          const edge = patch.edges.find(
            (candidate) => candidate.to[0] === node.id && candidate.to[1] === 1,
          );
          const upstream = edge ? nodeOf(edge.from[0]) : null;

          return upstream ? { kind: upstream.kind, params: upstream.params || {} } : null;
        },
      });

      // What feeds one of this node's inputs, read when the picture is painted.
      context.source = (port) => {
        const edge = patch.edges.find(
          (candidate) => candidate.to[0] === node.id && candidate.to[1] === port,
        );

        return edge ? nodeOf(edge.from[0]) : null;
      };

      cardContexts.set(node.id, context);
    }

    return context;
  }

  /// Where a press sits along a hero, as a share of it.
  function heroShare(hero, event) {
    const rect = hero.getBoundingClientRect();

    return Math.min(1, Math.max(0, ((event.clientX - rect.left) / rect.width * 520 - 20) / 480));
  }

  /// What a hero's behaviour acts on for a patch node: the card's own values and
  /// the graph behind them, instead of the product page's parameter map.
  function heroContextFor(node, kind) {
    const route = IMPORT_ROUTES[kind.id];
    const entryOf = () => cards[node.id];
    const cardOf = () => (entryOf() && entryOf().card) || null;
    const state = () => (entryOf() && entryOf().context) || {};
    const heroOf = () => {
      const card = cardOf();

      return card ? card.querySelector(".mcard-hero") : null;
    };

    // The card paints the gesture at once; the graph hears every step, in the
    // order the hand made them, so the sound follows instead of waiting for the
    // pointer to come up.
    const set = (id, value) => {
      node.params[id] = value;

      const card = cardOf();

      if (card && card.setValue) card.setValue(id, value);

      postParam(node.id, id, value);
    };

    const postFades = () => {
      const ir = state().ir || {};

      return queued(() =>
        fetch(
          "/ir?node=" + node.id + "&fade_in=" + (ir.fade_in || 0) +
            "&fade_out=" + (ir.fade_out || 0),
          { method: "POST", body: "fades" },
        ).catch(() => {}),
      );
    };

    const post = async (ids) => {
      // The gesture is over: wait for its writes, then rebuild the card from the
      // graph, which is now what the hand left it at.
      await writeChain;

      if (kind.id === "mod:convolver") return;

      for (const id of ids) await postParam(node.id, id, node.params[id]);

      await reload();
    };

    const frames = () => (state().waves && state().waves.count) || 0;

    return {
      read: () => {
        const current = state();

        if (kind.id === "mod:convolver") return current.ir;
        if (kind.id === "synth:shapes" || tableProduct(kind) === "shapes") return current.waves;

        return current.sample;
      },
      value: (id) => node.params[id] ?? 0,
      set,
      toggle: (id) => set(id, (node.params[id] ?? 0) > 0.5 ? 0 : 1),
      commit: post,
      action: async (name, query) => {
        if (!route) return;

        await postImportAction(node.id, route.path, name, query);
      },
      pick: (file) => {
        if (!route) return;

        if (file) postImport(node.id, route.path, file);
        else pickSampleFile((picked) => postImport(node.id, route.path, picked));
      },
      share: (event) => {
        const hero = heroOf();

        return hero ? heroShare(hero, event) : 0;
      },
      near: (at) => {
        const sample = state().sample;

        return !!(
          sample &&
          sample.slices &&
          sample.slices.some((mark) => Math.abs(mark - at) < 0.02)
        );
      },
      repaint: () => {
        // The fades are the convolver's own state, so the picture and the graph
        // move together.
        if (kind.id === "mod:convolver") postFades();

        const card = cardOf();

        if (card && card.repaintHero) card.repaintHero();
      },
      frame: (step) => stepFrame(node, step),
      frameIndex: () => IAmPage.index(node.params["oscillator.smooth_factor"] ?? 0, frames()),
      cycleMode: () => {
        const entry = entryOf();

        if (!entry) return;

        const modes = ["auto", "fixed", "pitch", "single"];
        const index = modes.indexOf(entry.context.waveMode || "auto");

        entry.context.waveMode = modes[(index + 1) % modes.length];

        const card = cardOf();

        if (card && card.repaintHero) card.repaintHero();
      },
    };
  }

  /// Moves the morph to the next whole frame, the way the shape hero's arrows
  /// do.
  async function stepFrame(node, step) {
    const entry = cards[node.id];
    const count = (entry && entry.context.waves && entry.context.waves.count) || 0;

    if (count <= 1) return;

    const current = node.params["oscillator.smooth_factor"] ?? 0;
    const frame = Math.round(current * (count - 1)) + step;
    const clamped = Math.min(Math.max(frame, 0), count - 1);

    await op(
      {
        op: "param",
        id: node.id,
        key: "oscillator.smooth_factor",
        value: clamped / (count - 1),
      },
      false,
    );
    await reload();
  }

  /// Posts one of the hero's edit actions and redraws the picture.
  async function postImportAction(node, path, action, query) {
    const extra = Object.entries(query || {})
      .map(([key, value]) => "&" + key + "=" + value)
      .join("");
    const response = await fetch(
      "/" + path + "?node=" + node + "&action=" + encodeURIComponent(action) + extra,
      { method: "POST", body: action },
    );
    const result = response.ok ? await response.json() : null;

    if (!result || !result.ok) {
      showStatus((result && result.error) || "the action did not run");
      return;
    }

    showStatus("");
    await refreshImport(cards[node], path);
  }

  /// Writes one parameter without the page's undo snapshot or a redraw.
  ///
  /// A gesture that is still running posts as it goes, so the sound follows the
  /// hand and the graph ends up with what the card already shows. A parameter
  /// is cheap now: the engine takes the value without rebuilding.
  /// A card that draws another node's state (the envelope's Shape hero) follows
  /// a parameter edit on that node without a full redraw.
  function repaintDependents(nodeId) {
    for (const entry of Object.values(cards)) {
      if (!entry || !entry.card || entry.kindId !== "adsr") continue;

      const edge = patch.edges.find(
        (candidate) => candidate.to[0] === entry.node.id && candidate.to[1] === 1,
      );

      if (edge && edge.from[0] === nodeId && entry.card.repaintHero) {
        entry.card.repaintHero();
      }
    }
  }

  // One cycle of each constructor node, as the engine's own builder renders it.
  const waveCache = new Map();
  const waveKeys = new Map();

  /// Asks for the sampled cycle of every constructor node, so a hero draws the
  /// DSP's output instead of a guess.
  function refreshWaves(force) {
    const live = new Set(patch.nodes.map((node) => node.id));

    for (const id of [...waveCache.keys()]) {
      if (!live.has(id)) {
        waveCache.delete(id);
        waveKeys.delete(id);
      }
    }

    for (const node of patch.nodes) {
      if (
        !(
          node.kind.startsWith("table:") ||
          node.kind.startsWith("osc:") ||
          node.kind.startsWith("warp:")
        )
      ) {
        continue;
      }

      const key = JSON.stringify(node.params);

      if (!force && waveKeys.get(node.id) === key) continue;

      waveKeys.set(node.id, key);
      fetch("/wave?node=" + node.id + "&points=192", { cache: "no-store" })
        .then((response) => (response.ok ? response.json() : null))
        .then((body) => {
          if (!body || !body.values) return;

          waveCache.set(node.id, body.values);
          repaintHeroes();
        })
        .catch(() => {});
    }
  }

  /// Heroes that draw another node (an envelope's source, a warp's table) follow
  /// a structural edit too: rewiring changes what they should show.
  function repaintHeroes() {
    for (const entry of Object.values(cards)) {
      if (entry && entry.card && entry.card.repaintHero) entry.card.repaintHero();
    }
  }

  function postParam(node, key, value) {
    repaintDependents(node);

    return queued(() =>
      fetch("/op", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ op: "param", id: node, key, value }),
      }).catch(() => {}),
    );
  }

  /// Reloads the graph from the server and draws it, so a card that shows a
  /// parameter picks the new value up.
  async function reload() {
    patch = readPatch(await (await fetch("/patch")).json());
    refreshWaves();
    draw();
  }

  /// A hero's editor attaches to the markup a repaint replaces: watch the box
  /// and re-attach to whatever the repaint puts there, or only the first drag
  /// works.
  function watchHero(entry, wire) {
    const hero = entry.card && entry.card.querySelector(".mcard-hero");

    if (!hero || entry.heroWatch) return;

    entry.heroWatch = new MutationObserver(() => {
      if (!entry.host || !entry.host.isConnected) {
        entry.heroWatch.disconnect();
        return;
      }

      wire(entry);
    });

    entry.heroWatch.observe(hero, { childList: true });
  }

  /// The Canvas node's drawing is not a parameter: the product keeps it, and
  /// the card asks for it the way the product page does.
  function refreshCanvas(entry) {
    if (!entry || !isProductCard(entry.kind, "canvas")) return;

    fetch("/canvas?node=" + entry.node.id, { cache: "no-store" })
      .then((response) => (response.ok ? response.json() : null))
      .then((state) => {
        if (!state || !state.shape) return;

        const same = entry.context.canvas && entry.context.canvas.shape === state.shape;
        entry.context.canvas = state;

        if (same) wireCanvasCard(entry);
        else if (entry.card.repaintHero) entry.card.repaintHero();
      })
      .catch(() => {});
  }

  /// A canvas hero on a patch card edits this node's drawing and morph, not a
  /// page-wide canvas, so its writes carry the node id.
  function wireCanvasCard(entry) {
    if (!entry || !isProductCard(entry.kind, "canvas") || !entry.context.canvas) return;
    const hero = entry.card && entry.card.querySelector(".mcard-hero");

    if (!hero) return;

    const svg = hero.querySelector("svg");

    if (!svg || svg.__canvasWired) return;

    svg.__canvasWired = true;
    watchHero(entry, wireCanvasCard);

    IAmCard.wire(hero, "canvas", entry.node.params, {
      canvas: entry.context.canvas,
      set: (id, value) => {
        entry.node.params[id] = value;
        postParam(entry.node.id, id, value);

        // The product page repaints through its dirty flag; a patch card has to
        // move its own control and picture, or a page button would leave the old
        // reading up.
        requestAnimationFrame(() => {
          if (!entry.host.isConnected) return;

          if (entry.card.setValue) entry.card.setValue(id, value);
          else if (entry.card.repaintHero) entry.card.repaintHero();
        });
      },
      post: (text) => {
        fetch("/canvas?node=" + entry.node.id, { method: "POST", body: text })
          .then((response) => (response.ok ? response.json() : null))
          .then((result) => {
            if (result && result.error) showStatus(result.error);
          })
          .catch(() => {});
      },
    });
  }

  /// The Spectrum node's pages are outside the parameter map too: the card
  /// asks the product's route and the editor writes them back to the same node.
  function refreshSpectrum(entry) {
    if (!entry || !isProductCard(entry.kind, "spectrum")) return;

    fetch("/spectrum?node=" + entry.node.id, { cache: "no-store" })
      .then((response) => (response.ok ? response.json() : null))
      .then((state) => {
        if (!state || !state.frames) return;

        // The hero's handlers hold this object, so its pages are replaced in
        // place: a fresh object would leave them editing the old snapshot and
        // posting it back over the new one.
        const holder = entry.context.spectrum || { page: 0 };
        holder.generation = state.generation;
        holder.gains = state.gains;
        holder.frames = state.frames;
        entry.context.spectrum = holder;

        if (entry.card.repaintHero) entry.card.repaintHero();
      })
      .catch(() => {});
  }

  /// A spectrum hero on a patch card edits this node's pages and morph, not a
  /// page-wide spectrum, so its writes carry the node id.
  function wireSpectrumCard(entry) {
    if (!entry || !isProductCard(entry.kind, "spectrum") || !entry.context.spectrum) return;
    const hero = entry.card && entry.card.querySelector(".mcard-hero");

    if (!hero) return;

    const svg = hero.querySelector("svg");

    if (!svg || svg.__spectrumWired) return;

    svg.__spectrumWired = true;
    watchHero(entry, wireSpectrumCard);

    IAmCard.wire(hero, "spectrum", entry.node.params, {
      spectrum: entry.context.spectrum,
      set: (id, value) => {
        entry.node.params[id] = value;
        postParam(entry.node.id, id, value);

        requestAnimationFrame(() => {
          if (!entry.host.isConnected) return;

          if (entry.card.setValue) entry.card.setValue(id, value);
          else if (entry.card.repaintHero) entry.card.repaintHero();
        });
      },
      post: (body) => {
        fetch("/spectrum?node=" + entry.node.id, { method: "POST", body })
          .then((response) => (response.ok ? response.json() : null))
          .then((result) => {
            if (result && result.error) showStatus(result.error);
          })
          .catch(() => {});
      },
    });
  }

  /// The Drummer's click and six slots are the product's own state, so the
  /// card asks for both and its buttons post back through the node.
  function refreshDrummer(entry) {
    if (!entry || entry.kind.id !== "synth:drummer") return;

    Promise.all([
      fetch("/click?node=" + entry.node.id, { cache: "no-store" })
        .then((response) => (response.ok ? response.json() : null))
        .catch(() => null),
      fetch("/samples?node=" + entry.node.id, { cache: "no-store" })
        .then((response) => (response.ok ? response.json() : null))
        .catch(() => null),
      fetch("/pitch?node=" + entry.node.id, { cache: "no-store" })
        .then((response) => (response.ok ? response.json() : null))
        .catch(() => null),
      fetch("/amp?node=" + entry.node.id, { cache: "no-store" })
        .then((response) => (response.ok ? response.json() : null))
        .catch(() => null),
    ]).then(([click, samples, pitch, amp]) => {
      if (click) entry.context.click = click;
      if (samples) entry.context.samples = samples;
      if (pitch && pitch.points) entry.context.fall = pitch;
      if (amp && amp.curves) entry.context.ampCurves = amp.curves;

      if (entry.card.repaintHero) entry.card.repaintHero();
    });
  }

  /// Where a hero's 520x160 picture sits after its meet scaling.
  function drummerMap(hero) {
    const svg = hero.querySelector("svg");

    if (!svg) return null;

    const rect = svg.getBoundingClientRect();

    if (!rect.width || !rect.height) return null;

    const scale = Math.min(rect.width / 520, rect.height / 160);

    return {
      scale,
      left: rect.left + (rect.width - 520 * scale) / 2,
      top: rect.top + (rect.height - 160 * scale) / 2,
    };
  }

  /// The amp axis: a cymbal rings for ten seconds, the snare's pitch is short,
  /// and every other kit piece gets at least one.
  function drummerAmpSpan(points, drum) {
    const last = points && points.length ? points[points.length - 1][0] : 1;

    if (drum === 8) return Math.max(0.03, last * 1.1);
    if (drum === 5) return Math.max(10, last * 1.1);

    return Math.max(1, last * 1.1);
  }

  function drummerCurveBody(points) {
    return points
      .map((point) => point[0].toFixed(4) + ":" + point[1].toFixed(4) + ":" + (point[2] || 0).toFixed(3))
      .join(";");
  }

  function clamp(value, low, high) {
    return Math.min(high, Math.max(low, value));
  }

  /// A Drummer card loads a click, its slots and its curves through its own
  /// node's routes. The buttons, rows and curve handles are one delegated
  /// gesture on the hero, so every page the card can show is covered.
  function wireDrummerCard(entry) {
    if (entry.drummerWired) return;

    const hero = entry.card && entry.card.querySelector(".mcard-hero");

    if (!hero) return;

    entry.drummerWired = true;

    function finish() {
      const drag = entry.drumDrag;

      if (!drag) return;

      entry.drumDrag = null;
      window.removeEventListener("pointermove", onMove);
      window.removeEventListener("pointerup", onUp);
      window.removeEventListener("pointercancel", onUp);

      const body = drummerCurveBody(drag.points);
      const url = drag.isFall
        ? "/pitch?node=" + entry.node.id
        : "/amp?node=" + entry.node.id + "&drum=" + drag.drum;

      fetch(url, { method: "POST", body })
        .then((response) => (response.ok ? response.json() : null))
        .then((result) => {
          if (result && result.error) showStatus(String(result.error));
          refreshDrummer(entry);
        })
        .catch(() => {});
    }

    function onUp(event) {
      finish();
      event.preventDefault();
    }

    function onMove(event) {
      const drag = entry.drumDrag;

      if (!drag) return;

      const map = drummerMap(drag.hero);

      if (!map) return;

      const x = (event.clientX - map.left) / map.scale;
      const y = (event.clientY - map.top) / map.scale;
      const points = drag.points;

      if (drag.kind === "point" && drag.isFall) {
        const time = clamp((x - 26) / 468, 0, 1) * 0.25;
        const ratio = Math.pow(2, clamp((132 - y) / 112, 0, 1) * 10);
        const lower = drag.index > 0 ? points[drag.index - 1][0] + 0.0005 : 0;
        const upper = drag.index < points.length - 1 ? points[drag.index + 1][0] - 0.0005 : 0.25;

        points[drag.index][0] = drag.index === 0 ? 0 : clamp(time, lower, Math.max(lower, upper));
        points[drag.index][1] = ratio;
      } else if (drag.kind === "point") {
        const span = drummerAmpSpan(points, drag.drum);
        const time = clamp((x - 26) / 468, 0, 1) * span;
        const value = clamp((132 - y) / 112, 0, 1) * (drag.drum === 8 ? 10 : 1);
        const lower = drag.index === 0 ? 0 : points[drag.index - 1][0] + 0.001;
        const upper = drag.index === points.length - 1
          ? span
          : points[drag.index + 1][0] - 0.001;

        points[drag.index][0] = clamp(time, lower, Math.max(lower, upper));
        points[drag.index][1] = value;
      } else {
        const travel = (drag.startY - event.clientY) / Math.max(map.scale, 1e-6);

        points[drag.index][2] = clamp(
          drag.startBend + travel * 0.09 * drag.direction,
          -10,
          10,
        );
      }

      entry.card.repaintHero();
      event.preventDefault();
    }

    function pageCurve() {
      const page = entry.pageName || "";
      const isFall = page === "Kick" && (entry.context.kickMode || "pitch") !== "amp";
      const drum = page === "Kick"
        ? 0
        : page === "Snare"
          ? ({ base: 6, noise: 7, pitch: 8 }[entry.context.snareMode || "base"] ?? 6)
          : ({ Hat: 2, Clap: 3, Tom: 4, Cymbal: 5 }[page] ?? -1);
      const points = isFall
        ? (entry.context.fall && entry.context.fall.points)
        : (drum >= 0 && entry.context.ampCurves ? entry.context.ampCurves[drum] : null);

      return { isFall, drum, points };
    }

    function pointOf(event, isFall, points, drum) {
      const map = drummerMap(hero);

      if (!map) return null;

      const x = (event.clientX - map.left) / map.scale;
      const y = (event.clientY - map.top) / map.scale;

      if (isFall) {
        return {
          time: clamp((x - 26) / 468, 0, 1) * 0.25,
          value: Math.pow(2, clamp((132 - y) / 112, 0, 1) * 10),
        };
      }

      const span = drummerAmpSpan(points, drum);

      return {
        time: clamp((x - 26) / 468, 0, 1) * span,
        value: clamp((132 - y) / 112, 0, 1) * (drum === 8 ? 10 : 1),
      };
    }

    function postCurve(isFall, drum, points) {
      const body = drummerCurveBody(points);
      const url = isFall
        ? "/pitch?node=" + entry.node.id
        : "/amp?node=" + entry.node.id + "&drum=" + drum;

      fetch(url, { method: "POST", body })
        .then((response) => (response.ok ? response.json() : null))
        .then((result) => {
          if (result && result.error) showStatus(String(result.error));
          refreshDrummer(entry);
        })
        .catch(() => {});
    }

    hero.addEventListener("pointerdown", (event) => {
      if (event.button !== 0) return;

      const action = event.target.closest("[data-action]");
      const load = event.target.closest("[data-load]");
      const clear = event.target.closest("[data-clear]");
      const kick = event.target.closest("[data-kick-mode]");
      const snare = event.target.closest("[data-snare-mode]");

      if (action && action.getAttribute("data-action") === "load") {
        event.preventDefault();
        pickSampleFile((file) => postDrummer(entry, "click", "", file));
        return;
      }

      if (load) {
        event.preventDefault();
        const slot = load.getAttribute("data-load");

        pickSampleFile((file) => postDrummer(entry, "samples", slot, file));
        return;
      }

      if (clear) {
        event.preventDefault();
        const slot = clear.getAttribute("data-clear");

        fetch("/samples?node=" + entry.node.id + "&slot=" + encodeURIComponent(slot) + "&clear=1", {
          method: "POST",
          body: "clear",
        })
          .then((response) => (response.ok ? response.json() : null))
          .then(() => refreshDrummer(entry))
          .catch(() => {});
        return;
      }

      if (kick) {
        entry.context.kickMode = kick.getAttribute("data-kick-mode");
        event.preventDefault();
        entry.card.repaintHero();
        return;
      }

      if (snare) {
        entry.context.snareMode = snare.getAttribute("data-snare-mode");
        event.preventDefault();
        entry.card.repaintHero();
        return;
      }

      const page = entry.pageName || "";
      const isFall = page === "Kick" && (entry.context.kickMode || "pitch") !== "amp";
      const ampDrum = page === "Kick"
        ? 0
        : page === "Snare"
          ? ({ base: 6, noise: 7, pitch: 8 }[entry.context.snareMode || "base"] ?? 6)
          : ({ Hat: 2, Clap: 3, Tom: 4, Cymbal: 5 }[page] ?? -1);

      if (!isFall && ampDrum < 0) return;

      const point = event.target.closest(isFall ? "[data-point]" : "[data-amp-point]");
      const bend = event.target.closest(isFall ? "[data-bend]" : "[data-amp-bend]");

      if (!point && !bend) return;

      const map = drummerMap(hero);

      if (!map) return;

      const points = isFall
        ? (entry.context.fall && entry.context.fall.points)
        : (entry.context.ampCurves && entry.context.ampCurves[ampDrum]);

      if (!points) return;

      const join = (point || bend).getAttribute(
        isFall ? (point ? "data-point" : "data-bend") : (point ? "data-amp-point" : "data-amp-bend"),
      );
      const index = Number(join);

      if (!Number.isFinite(index) || !points[index]) return;

      entry.drumDrag = {
        kind: point ? "point" : "bend",
        isFall,
        drum: ampDrum,
        points,
        index,
        hero,
        startY: event.clientY,
        startBend: points[index][2] || 0,
        direction: points[index + 1] && points[index + 1][1] > points[index][1] ? -1 : 1,
      };

      try {
        hero.setPointerCapture(event.pointerId);
      } catch (error) {
        // A synthetic pointer has nothing to capture.
      }

      window.addEventListener("pointermove", onMove);
      window.addEventListener("pointerup", onUp);
      window.addEventListener("pointercancel", onUp);
      event.preventDefault();
      event.stopPropagation();
    });

    hero.addEventListener("dblclick", (event) => {
      const curve = pageCurve();

      if (!curve.points || curve.points.length >= 10) return;
      if (event.target.closest("[data-point],[data-bend],[data-amp-point],[data-amp-bend]")) return;

      const position = pointOf(event, curve.isFall, curve.points, curve.drum);

      if (!position) return;

      curve.points.push([position.time, position.value, 0]);
      curve.points.sort((a, b) => a[0] - b[0]);
      entry.card.repaintHero();
      postCurve(curve.isFall, curve.drum, curve.points);
      event.preventDefault();
      event.stopPropagation();
    });

    hero.addEventListener("contextmenu", (event) => {
      const curve = pageCurve();

      if (!curve.points) return;

      const target = event.target.closest(curve.isFall ? "[data-point]" : "[data-amp-point]");

      if (!target) return;

      const index = Number(target.getAttribute(curve.isFall ? "data-point" : "data-amp-point"));

      if (!Number.isFinite(index) || curve.points.length <= 2) return;

      curve.points.splice(index, 1);
      entry.card.repaintHero();
      postCurve(curve.isFall, curve.drum, curve.points);
      event.preventDefault();
      event.stopPropagation();
    });
  }

  /// Posts one file to the click or a sample slot, then redraws the card.
  function postDrummer(entry, route, slot, file) {
    const reader = new FileReader();

    reader.onload = () => {
      const query = "?node=" + entry.node.id + (slot ? "&slot=" + encodeURIComponent(slot) : "");

      fetch("/" + route + query, { method: "POST", body: reader.result })
        .then((response) => (response.ok ? response.json() : null))
        .then((result) => {
          if (result && result.error) showStatus(String(result.error));
          else showStatus("");
          refreshDrummer(entry);
        })
        .catch(() => {});
    };

    reader.readAsArrayBuffer(file);
  }

  /// The Script's source, error and spectrum are the product's own state. The
  /// card asks for it through the node's route and shows it in the panel and
  /// the hero.
  function refreshScript(entry) {
    if (!entry || !isProductCard(entry.kind, "script")) return;

    fetch("/script?node=" + entry.node.id, { cache: "no-store" })
      .then((response) => (response.ok ? response.json() : null))
      .then((state) => {
        if (!state) return;

        entry.context.script = state;
        if (entry.card.setScript) entry.card.setScript(state.source || "");
        if (entry.card.setScriptError) entry.card.setScriptError(state.error || "");
        if (entry.card.repaintHero) entry.card.repaintHero();
      })
      .catch(() => {});
  }

  /// The Filter Table frame, fetched the same way the product page does.
  function refreshFilterTable(entry) {
    if (!entry || entry.kind.id !== "mod:filter_table") return;

    fetch("/filter-table?node=" + entry.node.id, { cache: "no-store" })
      .then((response) => (response.ok ? response.json() : null))
      .then((state) => {
        if (!state || !state.pages) return;

        state.view = entry.context.filterTable && entry.context.filterTable.view;
        state.windowSamples = entry.context.filterTable && entry.context.filterTable.windowSamples;
        state.frozen = entry.context.filterTable && entry.context.filterTable.frozen;
        state.importMode = entry.context.filterTable && entry.context.filterTable.importMode;
        state.sliceMode = entry.context.filterTable && entry.context.filterTable.sliceMode;
        entry.context.filterTable = state;
        if (entry.card.repaintHero) entry.card.repaintHero();
      })
      .catch(() => {});
  }

  /// A Filter Table hero edits this node's frame.
  function wireFilterTableCard(entry) {
    if (!entry || entry.kind.id !== "mod:filter_table" || !entry.context.filterTable) return;
    const hero = entry.card && entry.card.querySelector(".mcard-hero");

    if (!hero) return;

    const svg = hero.querySelector("svg");

    if (!svg || svg.__filterTableWired) return;

    svg.__filterTableWired = true;
    watchHero(entry, wireFilterTableCard);

    IAmCard.wire(hero, "filtertable", entry.node.params, {
      filterTable: entry.context.filterTable,
      set: (id, value) => {
        entry.node.params[id] = value;
        postParam(entry.node.id, id, value);

        requestAnimationFrame(() => {
          if (!entry.host.isConnected) return;

          if (entry.card.setValue) entry.card.setValue(id, value);
          else if (entry.card.repaintHero) entry.card.repaintHero();
        });
      },
      post: (body) => {
        fetch("/filter-table?node=" + entry.node.id, { method: "POST", body })
          .then((response) => (response.ok ? response.json() : null))
          .catch(() => {});
      },
      import: (body, window, page, mode, slice) =>
        fetch(
          "/filter-table/import?node=" + entry.node.id + "&window=" + window + "&page=" + page + "&mode=" + mode + "&slice=" + slice,
          { method: "POST", body }
        ).then((response) => (response.ok ? response.json() : null)),
      importProgress: () =>
        fetch("/filter-table/import/progress?node=" + entry.node.id, { cache: "no-store" })
          .then((response) => (response.ok ? response.json() : null)),
      reload: () =>
        fetch("/filter-table?node=" + entry.node.id, { cache: "no-store" })
          .then((response) => (response.ok ? response.json() : null)),
      edit: (action, page) =>
        fetch(
          "/filter-table/edit?node=" + entry.node.id + "&action=" + action + "&page=" + page,
          { method: "POST" }
        ).then((response) => (response.ok ? response.json() : null)),
    });
  }

  /// The History pages are the product's state, so the card asks its node for
  /// them the way the product's own page does.
  function refreshHistory(entry) {
    if (!entry || entry.kind.id !== "mod:history") return;

    fetch("/history?node=" + entry.node.id, { cache: "no-store" })
      .then((response) => (response.ok ? response.json() : null))
      .then((state) => {
        if (!state || !state.pages) return;

        entry.context.history = state;
        if (entry.card.repaintHero) entry.card.repaintHero();
      })
      .catch(() => {});
  }

  /// A history hero edits this node's pages, not a page-wide history.
  function wireHistoryCard(entry) {
    if (!entry || entry.kind.id !== "mod:history" || !entry.context.history) return;
    const hero = entry.card && entry.card.querySelector(".mcard-hero");

    if (!hero) return;

    const svg = hero.querySelector("svg");

    if (!svg || svg.__historyWired) return;

    svg.__historyWired = true;
    watchHero(entry, wireHistoryCard);

    IAmCard.wire(hero, "history", entry.node.params, {
      history: entry.context.history,
      set: (id, value) => {
        entry.node.params[id] = value;
        postParam(entry.node.id, id, value);

        requestAnimationFrame(() => {
          if (!entry.host.isConnected) return;

          if (entry.card.setValue) entry.card.setValue(id, value);
          else if (entry.card.repaintHero) entry.card.repaintHero();
        });
      },
      post: (body) => {
        fetch("/history?node=" + entry.node.id, { method: "POST", body })
          .then((response) => (response.ok ? response.json() : null))
          .then((result) => {
            if (result && result.error) showStatus(result.error);
          })
          .catch(() => {});
      },
    });

    if (!entry.historyTimer) {
      entry.historyTimer = setInterval(() => {
        fetch("/history-phase?node=" + entry.node.id, { cache: "no-store" })
          .then((response) => (response.ok ? response.json() : null))
          .then((state) => {
            if (!state || typeof state.phase !== "number") return;

            entry.context.history.phase = state.phase;
            const line = entry.card.querySelector(".history-playhead");
            if (line) line.style.left = (state.phase * 100).toFixed(2) + "%";
          })
          .catch(() => {});
      }, 100);
    }
  }

  /// Fetches what a node's slot now holds and repaints the card's picture.
  async function refreshImport(entry, path) {
    if (!entry || !entry.card || !entry.context) return;

    const response = await fetch("/" + path + "?node=" + entry.node.id, {
      cache: "no-store",
    });

    if (!response.ok) return;

    const state = await response.json();
    const key = path === "ir" ? "ir" : path === "waves" ? "waves" : "sample";
    entry.context[key] = state;

    if (entry.card.repaintHero) entry.card.repaintHero();
  }

  // A product card is the product's own face: the shared builder draws its
  // sections, pages, heroes and controls, so the patch cannot drift from the
  // product page. The patch adds the modulation rings around the controls.
  function productCard(node, kind, w, h) {
    let entry = cards[node.id];
    const mods = modsOf(node.id);
    const pages = kind.pages || [];
    const activePage = pages.length
      ? Math.min(pagesOf[node.id] ?? 0, pages.length - 1)
      : 0;
    const key = JSON.stringify([
      mods.map((mod) => [mod.slot, mod.to[1], Math.round(mod.depth * 1000), mod.mode]),
      activePage,
      node.params,
    ]);

    if (entry && (entry.key !== key || entry.kindId !== node.kind)) {
      entry.host.remove();
      delete cards[node.id];
      entry = null;
    }

    if (entry) {
      entry.host.style.setProperty(
        "--k",
        String(Math.max(0.2, Math.min(w / baseWidth(node), h / baseHeight(node)))),
      );
      entry.host.classList.toggle("compact", w < 430);

      // The pages live outside the parameter map, so they can change without a
      // graph change: keep this card's copy current or its next edit would post
      // a stale snapshot back over them.
      if (isProductCard(kind, "spectrum")) refreshSpectrum(entry);

      return entry.host;
    }

    // The heroes that draw live data read it from here. The context outlives the
    // card: a rebuild happens on every graph change, and a fresh context would
    // flash the hero's fallback picture until the next poll filled the live one
    // back in.
    const context = contextOf(node);
    const importRoute = IMPORT_ROUTES[kind.id];

    const card = IAmCard.faceCard({
      face: kind,
      values: node.params,
      context,
      clear: !!kind.clear,
      onClear: () => {
        fetch("/clear?node=" + node.id, { method: "POST" }).catch(() => {});
      },
      script: !!kind.script,
      // The fitted instrument fits a recorded note; the sampler, the shapes and
      // the convolver load the file straight into the slot their voice reads.
      sample: kind.id === "synth:fitted" || !!importRoute,
      sampleHint: importRoute
        ? importRoute.hint
        : "Load a recorded note (.wav) to fit: click or drop a file",
      heroContext: heroContextFor(node, kind),
      pickSample: (file) => {
        if (kind.id === "synth:fitted") {
          if (file) postSample(node.id, file);
          else pickSampleFile((picked) => postSample(node.id, picked));
          return;
        }

        if (!importRoute) return;

        if (file) postImport(node.id, importRoute.path, file);
        else pickSampleFile((picked) => postImport(node.id, importRoute.path, picked));
      },
      loadScript: () => {
        fetch("/script?node=" + node.id, { cache: "no-store" })
          .then((response) => (response.ok ? response.json() : null))
          .then((state) => {
            if (!state) return;

            card.setScript(state.source || "");
            card.setScriptError(state.error || "");

            if (entry) {
              entry.context.script = state;
              if (card.repaintHero) card.repaintHero();
            }
          })
          .catch(() => {});
      },
      applyScript: (source) => {
        fetch("/script?node=" + node.id, { method: "POST", body: source })
          .then((response) => (response.ok ? response.json() : null))
          .then((state) => {
            if (!state) return;

            card.setScriptError(state.error || "");

            if (state.error) {
              if (entry) {
                entry.context.script = Object.assign({}, entry.context.script, { error: state.error, spectrum: [] });
                if (card.repaintHero) card.repaintHero();
              }
            } else if (entry) {
              refreshScript(entry);
            }
          })
          .catch(() => {});
      },
      activePage,
      onPage: (index) => {
        pagesOf[node.id] = index;
        delete cards[node.id];
        draw();
      },
      onEdit: (identifier, value) => {
        // The card already shows the value; rebuilding the graph here would
        // take the focus away from the control being dragged.
        op({ op: "param", id: node.id, key: identifier, value }, false);

        if (isProductCard(kind, "script") && entry) {
          clearTimeout(entry.scriptTimer);
          entry.scriptTimer = setTimeout(() => refreshScript(entry), 150);
        }
      },
      wrapControl: (control) => {
        const assignments = mods
          .filter((mod) => mod.to[1] === control.id)
          .map((mod) => ({
            slot: mod.slot,
            label: sourceName(mod.from[0]),
            depth: mod.depth,
            mode: mod.mode === "base" ? "frombase" : "centered",
          }));

        return IAmCard.modulate(control, {
          assignments,
          onDepth: (slot, next) => op({ op: "mod_depth", index: slot, depth: next }, false),
          onMode: (slots, next) => op({ op: "mod_mode", index: slots[0], mode: next === "center" ? "center" : "base" }, false),
          // Ctrl + right click on the knob: every source comes off it.
          onClearModulations: (slots) => {
            for (const slot of slots) op({ op: "unmodulate", index: slot }, false);
          },
          dropTarget: node.id + ":" + control.id,
        });
      },
    });

    const host = document.createElement("div");
    host.className = "card-host";
    const scale = document.createElement("div");
    scale.className = "card-scale";
    scale.style.width = `${baseWidth(node)}px`;
    scale.style.height = `${baseHeight(node)}px`;
    scale.style.flexBasis = `${baseWidth(node)}px`;
    scale.appendChild(card);
    host.appendChild(scale);
    entry = { host, key, kindId: node.kind, context, card, node, kind, samples: null, phase: null };

    if (isProductCard(kind, "canvas")) {
      const repaint = card.repaintHero;

      card.repaintHero = function (values) {
        repaint(values);
        wireCanvasCard(entry);
      };

      refreshCanvas(entry);
    }

    cards[node.id] = entry;

    // A node that already holds a file shows it as soon as the card is built,
    // not only right after the drop that loaded it.
    if (IMPORT_ROUTES[kind.id]) refreshImport(entry, IMPORT_ROUTES[kind.id].path);
    if (isProductCard(kind, "spectrum")) {
      const repaint = card.repaintHero;

      card.repaintHero = function (values) {
        repaint(values);
        wireSpectrumCard(entry);
      };

      refreshSpectrum(entry);
    }

    if (kind.id === "synth:drummer") {
      entry.page = activePage;
      entry.pageName = ((kind.pages || [])[activePage] || {}).name || "";
      wireDrummerCard(entry);
      refreshDrummer(entry);
    }

    if (isProductCard(kind, "script")) {
      refreshScript(entry);
    }

    if (kind.id === "mod:history") {
      const repaint = card.repaintHero;

      card.repaintHero = function (values) {
        repaint(values);
        wireHistoryCard(entry);
      };

      refreshHistory(entry);
    }

    if (kind.id === "mod:filter_table") {
      const repaint = card.repaintHero;

      card.repaintHero = function (values) {
        repaint(values);
        wireFilterTableCard(entry);
      };

      refreshFilterTable(entry);
    }

    entry.host.style.setProperty(
      "--k",
      String(Math.max(0.2, Math.min(w / baseWidth(node), h / baseHeight(node)))),
    );
    entry.host.classList.toggle("compact", w < 430);
    return entry.host;
  }

  const kindOf = (id) => catalog.find((kind) => kind.id === id);

  /// The depth a fresh modulation starts with, in the parameter's normalized
  /// range. A logarithmic parameter gets about one octave; a linear one gets
  /// half its travel, which is the unit the card's ring already works in.
  function defaultDepth(kind, key) {
    const meta = (kind.param_meta || []).find((item) => item.name === key);

    if (!meta) return 0.5;

    if (meta.log && meta.min > 0 && meta.max > meta.min) {
      const span = Math.log(meta.max / meta.min);

      if (span > 0.0001) return Math.min(0.5, Math.LN2 / span);
    }

    return 0.5;
  }
  const nodeOf = (id) => patch.nodes.find((node) => node.id === id);
  const scaleOf = (id) => scales[id] ?? { x: 1, y: 1 };

  function el(name, attrs = {}) {
    const node = document.createElementNS(NS, name);
    for (const [key, value] of Object.entries(attrs)) node.setAttribute(key, value);
    return node;
  }

  function toWorld(event) {
    const rect = svg.getBoundingClientRect();
    return {
      x: (event.clientX - rect.left - view.x) / view.zoom,
      y: (event.clientY - rect.top - view.y) / view.zoom,
    };
  }

  // A product card is wider and taller than the plain ones: the filter's six
  // controls need the room its own faceplate gives them.
  function faceOf(node) {
    const kind = kindOf(node.kind);
    return kind && kind.face ? kind.face : null;
  }

  // The kinds with no product face get a size that fits what they draw: a note
  // and a header for the host ports, a hero and its knobs for a source, knobs
  // and nothing else for the small utilities.
  const COMPACT_SIZES = {
    "input:midi": [260, 150],
    "input:audio": [260, 150],
    "input:sidechain": [260, 150],
    "out:audio": [260, 150],
    "voice:in": [260, 190],
    "voice:out": [260, 150],
    // Two knobs and nothing else: the default card leaves half of it empty.
    macro: [260, 175],
    // The source cards are wide like the rack's: a full-width hero and the
    // knobs in a row under it.
    lfo: [640, 340],
    envelope: [640, 340],
    random: [640, 340],
    table: [640, 340],
    pitch: [600, 320],
    midi: [600, 320],
    // A curve and nothing else: there are no knobs to stack under the hero,
    // so it takes the height the others give their controls.
    map: [600, 260],
    // The constructor cards draw one waveform and a knob or two: a tall card
    // leaves the hero's lower half empty.
    "table:shape": [280, 200],
    "table:canvas": [280, 200],
    "warp:bend": [280, 200],
    "warp:mirror": [280, 200],
    "warp:sync": [280, 200],
    "warp:quantize": [280, 200],
    "warp:remap": [280, 200],
    "osc:am": [280, 230],
    "osc:rm": [280, 230],
    "osc:pm": [280, 230],
    "osc:fm": [280, 230],
    "synth:additive": [300, 210],
    mixer: [280, 190],
    "split:ms": [280, 170],
    "split:lr": [280, 170],
    "split:bands": [300, 210],
    "mix:ms": [280, 170],
    "mix:lr": [280, 170],
    "mix:bands": [340, 190],
    delay: [280, 210],
  };

  function baseSize(node) {
    const face = faceOf(node);

    if (face) return face;

    return COMPACT_SIZES[node.kind] || [BASE_W, BASE_H];
  }

  function baseWidth(node) {
    return baseSize(node)[0];
  }

  function baseHeight(node) {
    return baseSize(node)[1];
  }

  function size(node) {
    const scale = scaleOf(node.id);
    return {
      w: baseWidth(node) * scale.x,
      h: baseHeight(node) * scale.y,
      sx: scale.x,
      sy: scale.y,
    };
  }

  // The server stores a point as \`[time, value, bend]\`; the page edits objects.
  function readPatch(patch) {
    for (const node of patch.nodes) {
      node.points = (node.points || []).map((point) => (
        Array.isArray(point)
          ? { time: point[0], value: point[1], bend: point[2] || 0 }
          : point
      ));
      node.velocity_points = (node.velocity_points || []).map((point) => (
        Array.isArray(point)
          ? { time: point[0], value: point[1], bend: point[2] || 0 }
          : point
      ));
    }

    return patch;
  }

  async function load() {
    catalog = await (await fetch("/catalog")).json();
    patch = readPatch(await (await fetch("/patch")).json());
    refreshWaves(true);
    draw();
  }

  // A control inside a card owns the focus while it is being used; the poll
  // must not rebuild the page under it, or the select closes and the knob
  // stops mid-drag.
  let interacting = false;
  let editingPoints = false;

  function busy() {
    if (drag || interacting || editingPoints) return true;
    const active = document.activeElement;
    return !!(active && active.closest && active.closest(".card-host"));
  }

  // Undo keeps the page's text form: the server already round-trips it, and
  // every op is a graph change. Consecutive moves of one control coalesce, so
  // one drag is one undo step.
  const undoStack = [];
  const redoStack = [];
  let lastRecord = { key: "", at: 0 };
  let serverError = "";
  let opError = "";

  function mutationKey(body) {
    if (body.op === "param") return "param:" + body.id + ":" + body.key;
    if (body.op === "mod_depth") return "depth:" + body.index;
    if (body.op === "mod_mode") return "mode:" + (body.index || []).join(",");
    if (body.op === "move") return "move:" + body.id;
    return "";
  }

  function paintStatus() {
    const bar = document.getElementById("status");
    const text = serverError || opError;
    bar.textContent = text;
    bar.hidden = !text;
  }

  function showStatus(text) {
    opError = text || "";
    paintStatus();
  }

  function updateHistory() {
    document.getElementById("undo").disabled = undoStack.length === 0;
    document.getElementById("redo").disabled = redoStack.length === 0;
  }

  async function snapshot(body) {
    const key = mutationKey(body);
    const now = Date.now();

    if (key && key === lastRecord.key && now - lastRecord.at < 600) {
      lastRecord.at = now;
      return;
    }

    lastRecord = { key: key, at: now };
    undoStack.push(await (await fetch("/text")).json());

    if (undoStack.length > 100) undoStack.shift();
    redoStack.length = 0;
    updateHistory();
  }

  async function restore(text) {
    await fetch("/text", { method: "POST", body: text });
    patch = readPatch(await (await fetch("/patch")).json());
    updateHistory();
    draw();
  }

  async function undo() {
    if (!undoStack.length) return;
    redoStack.push(await (await fetch("/text")).json());
    await restore(undoStack.pop());
  }

  async function redo() {
    if (!redoStack.length) return;
    undoStack.push(await (await fetch("/text")).json());
    await restore(redoStack.pop());
  }

  /// Copies the selected nodes, the edges between them and the modulations
  /// inside the selection.
  function copySelection() {
    copyNodes([...selection]);
  }

  /// Copies the given nodes, with the wires and links between them.
  function copyNodes(ids) {
    if (!ids.length) return;

    const index = new Map(ids.map((id, position) => [id, position]));

    clipboard = {
      nodes: ids.map((id) => {
        const node = nodeOf(id);

        return {
          kind: node.kind,
          x: node.x,
          y: node.y,
          params: Object.entries(node.params || {}),
          points: (node.points || []).map((point) => [point.time, point.value, point.bend || 0]),
          velocity_points: (node.velocity_points || []).map((point) => [point.time, point.value, point.bend || 0]),
        };
      }),
      edges: patch.edges
        .filter((edge) => index.has(edge.from[0]) && index.has(edge.to[0]))
        .map((edge) => ({
          from: index.get(edge.from[0]),
          fromPort: edge.from[1],
          to: index.get(edge.to[0]),
          toPort: edge.to[1],
        })),
      mods: patch.mods
        .filter((mod) => index.has(mod.from[0]) && index.has(mod.to[0]))
        .map((mod) => ({
          from: index.get(mod.from[0]),
          fromPort: mod.from[1],
          to: index.get(mod.to[0]),
          param: mod.to[1],
          depth: mod.depth,
          mode: mod.mode,
        })),
    };
  }

  /// Pastes the clipboard a little off the original, as one undo step.
  async function pasteClipboard() {
    if (!clipboard || !clipboard.nodes.length) return;

    await snapshot({ op: "paste" });
    suspendSnapshot += 1;

    const ids = [];

    try {
      for (const item of clipboard.nodes) {
        const added = await op(
          { op: "add", kind: item.kind, x: item.x + 24, y: item.y + 24 },
          false,
        );
        const id = added && added.id;

        ids.push(id);

        if (id === undefined) continue;

        for (const [key, value] of item.params) {
          await op({ op: "param", id, key, value }, false);
        }

        if (item.points.length) {
          await op({ op: "set_points", id, points: item.points, curve: "note" }, false);
        }

        if (item.velocity_points && item.velocity_points.length) {
          await op({ op: "set_points", id, points: item.velocity_points, curve: "velocity" }, false);
        }
      }

      for (const edge of clipboard.edges) {
        if (ids[edge.from] === undefined || ids[edge.to] === undefined) continue;

        await op(
          { op: "connect", from: [ids[edge.from], edge.fromPort], to: [ids[edge.to], edge.toPort] },
          false,
        );
      }

      for (const mod of clipboard.mods) {
        if (ids[mod.from] === undefined || ids[mod.to] === undefined) continue;

        await op(
          { op: "modulate", from: [ids[mod.from], mod.fromPort], to: [ids[mod.to], mod.param], depth: mod.depth },
          false,
        );
      }
    } finally {
      suspendSnapshot -= 1;
    }

    patch = readPatch(await (await fetch("/patch")).json());

    // A centered modulation is a mode on the link; the op adds the link first.
    for (const mod of clipboard.mods) {
      if (mod.mode !== "center") continue;
      if (ids[mod.from] === undefined || ids[mod.to] === undefined) continue;

      const index = patch.mods.findIndex((link) =>
        link.from[0] === ids[mod.from] && link.to[0] === ids[mod.to] && link.to[1] === mod.param);

      if (index >= 0) await op({ op: "mod_mode", index, mode: "center" }, false);
    }

    selection.clear();

    for (const id of ids) {
      if (id !== undefined) selection.add(id);
    }

    primary = selection.size ? [...selection][selection.size - 1] : null;
    draw();
  }

  // The page's writes go out one at a time, in the order the hand made them.
  //
  // A drag posts many of them, and each one first reads the patch for the undo
  // step: left to race, a slower earlier request can land after a later one and
  // put the graph back where the drag started, while the card still shows the
  // new value.
  let writeChain = Promise.resolve();

  function queued(task) {
    writeChain = writeChain.then(task, task);

    return writeChain;
  }

  async function op(body, redraw = true) {
    return queued(async () => {
      if (suspendSnapshot === 0) await snapshot(body);

      const response = await fetch("/op", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify(body),
      });
      const result = await response.json();
      showStatus(result.error ? String(result.error) : "");
      patch = readPatch(await (await fetch("/patch")).json());

      if (body.op === "param" && typeof body.id === "number") {
        waveCache.delete(body.id);
        waveKeys.delete(body.id);
        repaintDependents(body.id);
        refreshWaves(true);
      }

      if (["connect", "disconnect", "replace", "remove", "add", "duplicate"].includes(body.op)) {
        refreshWaves(true);
      }

      if (redraw) draw();

      return result;
    });
  }

  // The port circles sit on the card's edge, in base coordinates; the world
  // point includes the node's own scale.
  function portY(port, index) {
    return 44 + 20 * index;
  }

  function portXY(node, port, output) {
    const kind = kindOf(node.kind);
    const list = output ? kind.outputs : kind.inputs;
    const index = list.indexOf(port);
    const { w, sy } = size(node);
    return {
      x: node.x + (output ? w : 0),
      y: node.y + portY(port, index) * sy,
    };
  }

  // The biquad lowpass response, in dB, at a frequency.
  // The voice group is one underlined stretch of the graph: a card inside it
  // plays once per note.
  function drawVoiceBox(world) {
    const members = (patch.voice || []).map((id) => nodeOf(id)).filter(Boolean);

    if (members.length < 1) return;

    let minX = Infinity;
    let minY = Infinity;
    let maxX = -Infinity;
    let maxY = -Infinity;

    for (const node of members) {
      const { w, h } = size(node);
      minX = Math.min(minX, node.x);
      minY = Math.min(minY, node.y);
      maxX = Math.max(maxX, node.x + w);
      maxY = Math.max(maxY, node.y + h);
    }

    const pad = 18;
    world.appendChild(el("rect", {
      x: minX - pad,
      y: minY - pad - 18,
      width: maxX - minX + pad * 2,
      height: maxY - minY + pad * 2 + 18,
      rx: 10,
      class: "voice-box",
    }));

    const label = el("text", { x: minX - pad + 12, y: minY - pad - 4, class: "voice-label" });
    label.textContent = "VOICE";
    world.appendChild(label);
  }

  function draw() {
    // A context and a history belong to a node; one the graph no longer holds
    // is dropped.
    for (const id of cardContexts.keys()) {
      if (!patch.nodes.some((node) => node.id === id)) cardContexts.delete(id);
    }

    for (const id of sourceHistories.keys()) {
      if (!patch.nodes.some((node) => node.id === id)) sourceHistories.delete(id);
    }

    for (const id of sourceCursors.keys()) {
      if (!patch.nodes.some((node) => node.id === id)) sourceCursors.delete(id);
    }

    while (svg.firstChild) svg.removeChild(svg.firstChild);

    const world = el("g", {
      transform: `translate(${view.x} ${view.y}) scale(${view.zoom})`,
    });
    svg.appendChild(world);

    drawVoiceBox(world);

    for (const edge of patch.edges) {
      const from = nodeOf(edge.from[0]);
      const to = nodeOf(edge.to[0]);
      if (!from || !to) continue;
      const a = portXY(from, kindOf(from.kind).outputs[edge.from[1]], true);
      const b = portXY(to, kindOf(to.kind).inputs[edge.to[1]], false);
      const mid = (a.x + b.x) / 2;
      const d = `M ${a.x} ${a.y} C ${mid} ${a.y}, ${mid} ${b.y}, ${b.x} ${b.y}`;
      const index = patch.edges.indexOf(edge);

      // A wire that crosses the voice boundary carries nothing: the group's
      // audio only leaves through its own output, so the split drops the edge.
      // Drawing it apart is what says so.
      const voice = new Set(patch.voice || []);
      const crossed = voice.has(edge.from[0]) !== voice.has(edge.to[0]);

      world.appendChild(el("path", { d, class: "wire-hit", "data-edge": index }));
      world.appendChild(el("path", {
        d,
        class: "wire" +
          (crossed ? " crossed" : "") +
          (selectedEdge === index ? " selected" : ""),
      }));
    }

    for (const mod of patch.mods) {
      const from = nodeOf(mod.from[0]);
      const to = nodeOf(mod.to[0]);
      if (!from || !to) continue;
      const a = portXY(from, kindOf(from.kind).outputs[mod.from[1]], true);
      const { w, h } = size(to);
      const b = { x: to.x + w * 0.5, y: to.y + h };
      const d = `M ${a.x} ${a.y} C ${a.x + 60} ${a.y}, ${b.x} ${b.y - 50}, ${b.x} ${b.y}`;
      const index = patch.mods.indexOf(mod);

      world.appendChild(el("path", { d, class: "mod-hit", "data-mod": index }));
      world.appendChild(el("path", {
        d,
        class: "mod" + (selectedMod === index ? " selected" : ""),
      }));
    }

    for (const node of patch.nodes) {
      const kind = kindOf(node.kind);
      if (!kind) continue;

      const { w, h, sx, sy } = size(node);
      const u = Math.min(sx, sy);
      const group = el("g", {
        "data-node": node.id,
        transform: `translate(${node.x} ${node.y})`,
      });

      // The box resizes with the handle; the face inside keeps its proportions,
      // so text and knobs are never stretched on one axis only.
      group.appendChild(el("rect", {
        class: "node" + (selection.has(node.id) ? " selected" : ""),
        x: 0, y: 0, width: w, height: h, rx: 6,
      }));

      if (window.IAmCard) {
        const foreign = el("foreignObject", { x: 0, y: 0, width: w, height: h });
        const host = kind.controls && kind.controls.length
          ? productCard(node, kind, w, h)
          : genericCard(node, kind, w, h);
        foreign.appendChild(host);
        group.appendChild(foreign);
      } else {
        group.appendChild(el("rect", {
          x: 0, y: 0, width: w, height: 22 * u, rx: 6, class: "card-head",
        }));

        const title = el("text", { class: "node-name", x: 10, y: 15 * u, "font-size": 12 * u });
        title.textContent = kind.name;
        group.appendChild(title);

        const tag = el("text", {
          class: "node-kind", x: w - 10, y: 15 * u, "font-size": 10 * u, "text-anchor": "end",
        });
        tag.textContent = node.kind;
        group.appendChild(tag);

        const content = el("g", { transform: `scale(${u})` });
        drawBody(content, node, kind);
        group.appendChild(content);
      }

      drawPorts(group, kind, w, sy, u);

      if (selection.size === 1 && selection.has(node.id)) {
        group.appendChild(el("rect", {
          class: "scale-handle", x: w - 14, y: h - 14, width: 12, height: 12, rx: 2,
        }));
      }

      world.appendChild(group);
    }

    if (drag && drag.kind === "marquee") {
      const a = drag.from;
      const b = drag.to;
      world.appendChild(el("rect", {
        x: Math.min(a.x, b.x), y: Math.min(a.y, b.y),
        width: Math.abs(b.x - a.x), height: Math.abs(b.y - a.y),
        class: "marquee",
      }));
    }

    // A blade of wire follows the pointer while a port is dragged, and the
    // ports it could land on light up.
    if (drag && drag.kind === "port") {
      const source = nodeOf(drag.id);
      const kind = kindOf(source.kind);
      const list = drag.output ? kind.outputs : kind.inputs;
      const start = portXY(source, list[drag.index], drag.output);
      const end = drag.point || start;
      const mid = (start.x + end.x) / 2;

      world.appendChild(el("path", {
        d: `M ${start.x} ${start.y} C ${mid} ${start.y}, ${mid} ${end.y}, ${end.x} ${end.y}`,
        class: "wire-temp",
      }));
      world.appendChild(el("circle", {
        cx: end.x, cy: end.y, r: 4.5, class: "wire-dot",
      }));

      const signal = list[drag.index].kind;

      for (const other of patch.nodes) {
        if (other.id === drag.id) continue;
        const otherKind = kindOf(other.kind);
        const ports = drag.output ? otherKind.inputs : otherKind.outputs;

        ports.forEach((port, index) => {
          // A table may land on an oscillator port: the engine reads it through
          // a table oscillator, and the page has to offer the same pair.
          const feeds = port.kind === signal
            || (signal === "wavetable" && port.kind === "osc")
            || (signal === "osc" && port.kind === "wavetable");
          if (!feeds) return;
          const at = portXY(other, port, !drag.output);
          const hovered = drag.hoverPort && drag.hoverPort.id === other.id &&
            drag.hoverPort.index === index && drag.hoverPort.output === !drag.output;

          world.appendChild(el("circle", {
            cx: at.x, cy: at.y, r: hovered ? 8 : 6,
            class: "port-target" + (hovered ? " hover" : ""),
          }));
        });
      }

      // A control drag lands on a knob, so the cards that have one glow.
      if (drag.control && drag.output) {
        for (const other of patch.nodes) {
          if (other.id === drag.id) continue;
          const otherKind = kindOf(other.kind);
          if (!otherKind.params || !otherKind.params.length) continue;
          const { w, h } = size(other);
          world.appendChild(el("rect", {
            x: other.x - 3, y: other.y - 3, width: w + 6, height: h + 6,
            rx: 9, class: "mod-target-hint",
          }));
        }
      }
    }
  }

  function drawBody(group, node, kind) {
    {
      const rows = kind.params.slice(0, 6);
      const meta = new Map((kind.param_meta || []).map((entry) => [entry.name, entry]));

      rows.forEach(([key, fallback], index) => {
        const y = 42 + index * 20;
        const row = el("g", { "data-mod-target": `${node.id}:${key}` });
        const label = el("text", { class: "card-param", x: 12, y });
        label.textContent = key;
        row.appendChild(label);
        const value = el("text", { class: "card-value", x: BASE_W - 12, y, "text-anchor": "end" });
        const entry = meta.get(key);
        const raw = node.params[key] ?? fallback;
        const choice = entry && entry.choices && entry.step === 1
          ? entry.choices[Math.round(raw)]
          : null;
        value.textContent = choice ?? raw.toFixed(2);
        row.appendChild(value);
        row.appendChild(el("rect", { x: 6, y: y - 13, width: BASE_W - 12, height: 18, fill: "transparent" }));
        group.appendChild(row);
      });

    }
  }

  function describeArc(cx, cy, r, from, to) {
    const start = polar(cx, cy, r, to);
    const end = polar(cx, cy, r, from);
    const large = Math.abs(to - from) <= 180 ? "0" : "1";

    return `M ${start.x} ${start.y} A ${r} ${r} 0 ${large} 0 ${end.x} ${end.y}`;
  }

  function polar(cx, cy, r, angle) {
    const radians = (angle - 90) * Math.PI / 180;
    return { x: cx + r * Math.cos(radians), y: cy + r * Math.sin(radians) };
  }

  function formatKnob(key, value) {
    if (key === "cutoff") {
      return value >= 1000 ? (value / 1000).toFixed(2) + "k" : Math.round(value) + "";
    }

    return value.toFixed(2);
  }

  function drawPorts(group, kind, w, sy, u) {
    kind.inputs.forEach((port, index) => {
      const y = portY(port, index) * sy;
      group.appendChild(el("circle", {
        "data-port": "in", "data-index": index, cx: 0, cy: y, r: 5,
        class: "port " + port.kind,
      }));
      const label = el("text", { class: "port-label", x: 8, y: y + 3, "font-size": 9 * u });
      label.textContent = port.name;
      group.appendChild(label);
    });

    kind.outputs.forEach((port, index) => {
      const y = portY(port, index) * sy;
      group.appendChild(el("circle", {
        "data-port": "out", "data-index": index, cx: w, cy: y, r: 5,
        class: "port " + port.kind,
      }));
      const label = el("text", {
        class: "port-label", x: w - 8, y: y + 3, "font-size": 9 * u, "text-anchor": "end",
      });
      label.textContent = port.name;
      group.appendChild(label);
    });
  }

  // The sources whose card already draws their live picture: the scope strip
  // would be the same picture twice.
  const CONTROL_KINDS = ["lfo", "envelope", "random", "table", "pitch", "midi", "map"];

  // A right-click opens the node library: categories, then the nodes inside
  // them, alphabetically. No list takes space from the graph until it is asked.
  // Four groups: an effect processes audio, a generator makes it, a control
  // steers a parameter, and the host ports and the voice boundaries are the
  // wiring around them.
  function categoryOf(kind) {
    const id = kind.id;

    // Splitters and mixers are the graph's routing, not effects and not
    // generators: their own shelf keeps the list short.
    if (id.startsWith("split:") || id.startsWith("mix:") || id === "mixer") return "Routing";

    if (id.startsWith("mod:")) return "Effect";

    // The block-alignment node delays by one block and reports that as latency;
    // it is not the echo product, so it has its own entry.
    if (id === "delay") return "Effect";

    if (
      [
        "mixer",
        "split:ms",
        "split:lr",
        "split:bands",
        "mix:ms",
        "mix:lr",
        "mix:bands",
      ].includes(id)
    ) {
      return "Effect";
    }
    if (id.startsWith("gen:") || id.startsWith("synth:")) return "Generator";
    if (id.startsWith("table:") || id.startsWith("warp:")) return "Wavetable";
    if (id.startsWith("osc:") || id === "adsr") return "Oscillator";
    if (CONTROL_KINDS.includes(id)) return "Control";

    return "I/O";
  }

  /// The flyout lifted out of the menu, and the timer that gives the pointer
  /// time to travel from the row into it.
  let flyout = null;
  let flyoutTimer = null;

  function closeMenu() {
    menu.style.display = "none";

    if (flyout) {
      flyout.style.display = "none";
      flyout = null;
    }
  }

  function placeMenu(clientX, clientY) {
    menu.style.display = "block";
    menu.style.left = "0px";
    menu.style.top = "0px";
    menu.style.maxHeight = "";

    // The menu hangs inside the stage, which clips what leaves it: the window's
    // size is not the limit that matters, the space this element has is. The
    // pointer is in viewport coordinates, so the parent's origin comes off first.
    const parent = menu.offsetParent || document.body;
    const bounds = parent.getBoundingClientRect();
    const left = clientX - bounds.left;
    const top = clientY - bounds.top;

    // Room below, room above: a menu pinned to the bottom of the stage would
    // otherwise be a 140 px scroll box, so it flips up when that side is longer.
    const below = bounds.height - top - 8;
    const above = top - 8;
    const flip = below < 260 && above > below;

    menu.style.maxHeight = `${Math.max(140, flip ? above : below)}px`;

    const width = menu.offsetWidth;
    const height = menu.offsetHeight;
    const y = flip ? top - height : top;

    menu.style.left = `${Math.max(8, Math.min(left, bounds.width - width - 8))}px`;
    menu.style.top = `${Math.max(8, Math.min(y, bounds.height - height - 8))}px`;
  }

  /// Opens a category's flyout on the side of the window that has room.
  ///
  /// A flyout that always opens right runs off the edge of a narrow window, so
  /// it mirrors when the right side is short and rides up when the category is
  /// near the bottom.
  /// Shows a category's flyout beside its row.
  ///
  /// The flyout is lifted out of the menu and parked on the body before it is
  /// shown: the menu scrolls its own list, and anything that scrolls clips what
  /// leaves it — a flyout opened near the bottom of the list would be cut in
  /// half. From the body it can hang over the edge and still be seen.
  function showFlyout(category) {
    clearTimeout(flyoutTimer);

    const sub = category.flyout;

    if (!sub) return;

    if (flyout && flyout !== sub) flyout.style.display = "none";

    flyout = sub;
    document.body.appendChild(sub);
    sub.style.display = "block";

    const rect = category.getBoundingClientRect();
    const margin = 8;

    // Whatever the stylesheet says, a flyout may not be taller than the window:
    // the rest is its own scroll box. This is set before measuring so the clamp
    // below uses the height it will really have.
    sub.style.maxHeight = `${Math.max(120, window.innerHeight - margin * 2)}px`;

    const width = sub.offsetWidth;
    const height = sub.offsetHeight;

    // Placed in viewport coordinates: it opens to the right of its row, mirrors
    // when that side is short, and rides up when the row is near the bottom.
    let left = rect.right;

    if (left + width > window.innerWidth - margin) {
      left = Math.max(margin, rect.left - width);
    }

    const top = Math.max(margin, Math.min(rect.top, window.innerHeight - margin - height));

    sub.style.left = `${left}px`;
    sub.style.top = `${top}px`;
  }

  /// Hides a flyout after a short grace, so the pointer can cross the gap
  /// between the row and the flyout without it closing under the cursor.
  function hideFlyout(sub) {
    clearTimeout(flyoutTimer);

    flyoutTimer = setTimeout(() => {
      if (sub) sub.style.display = "none";
    }, 160);
  }

  function menuButton(label, action) {
    const item = document.createElement("button");
    item.className = "menu-item";
    item.textContent = label;
    item.onclick = action;

    return item;
  }

  /// Right-clicking a node offers what can be done to it.
  function openNodeMenu(nodeId, clientX, clientY) {
    const wired = patch.edges.some(
      (edge) => edge.from[0] === nodeId || edge.to[0] === nodeId,
    );

    menu.innerHTML = "";
    // Replacing keeps the node's id, its place and the wires the new kind can
    // still carry, so a patch can be rerouted without redrawing the graph.
    menu.appendChild(menuButton("Replace…", () => {
      openLibrary(clientX, clientY, (kind) => {
        op({ op: "replace", id: nodeId, kind });
      });
    }));
    // A right-click does not select, so Copy here is how one node is copied.
    menu.appendChild(menuButton("Copy", () => {
      copyNodes([nodeId]);
      closeMenu();
    }));
    menu.appendChild(menuButton("Duplicate", () => {
      op({ op: "duplicate", id: nodeId });
      closeMenu();
    }));

    if (wired) {
      menu.appendChild(menuButton("Disconnect", () => {
        disconnectNode(nodeId);
        closeMenu();
      }));
    }

    menu.appendChild(menuButton("Delete", () => {
      op({ op: "remove", id: nodeId });
      delete cards[nodeId];
      closeMenu();
    }));

    placeMenu(clientX, clientY);
  }

  /// Takes every wire off one node, as one undo step.
  ///
  /// The edges are cut from the end of the list: each cut shifts the ones after
  /// it, so going backwards keeps the indices the page is holding valid.
  async function disconnectNode(id) {
    const indices = patch.edges
      .map((edge, index) => ({ edge, index }))
      .filter(({ edge }) => edge.from[0] === id || edge.to[0] === id)
      .map(({ index }) => index)
      .sort((a, b) => b - a);

    if (!indices.length) return;

    await snapshot({ op: "disconnect" });
    suspendSnapshot += 1;

    try {
      for (const index of indices) {
        await op({ op: "disconnect", index }, false);
      }
    } finally {
      suspendSnapshot -= 1;
    }

    draw();
  }

  /// Right-clicking a wire offers to cut it, or to put a node in the middle.
  function openWireMenu(index, clientX, clientY) {
    const edge = patch.edges[index];
    const from = edge && nodeOf(edge.from[0]);
    const to = edge && nodeOf(edge.to[0]);
    const audio = from && to &&
      kindOf(from.kind).outputs[edge.from[1]].kind === "audio" &&
      kindOf(to.kind).inputs[edge.to[1]].kind === "audio";

    menu.innerHTML = "";

    if (audio) {
      menu.appendChild(menuButton("Insert…", () => {
        openLibrary(
          clientX,
          clientY,
          (kind) => insertInto(index, kind),
          // One audio in and one audio out: the processors a wire can carry.
          (kind) => categoryOf(kind) === "Effect" &&
            kind.inputs.length === 1 && kind.outputs.length === 1 &&
            kind.inputs[0].kind === "audio" && kind.outputs[0].kind === "audio",
        );
      }));
    }

    menu.appendChild(menuButton("Delete connection", () => {
      op({ op: "disconnect", index });
      closeMenu();
    }));

    placeMenu(clientX, clientY);
  }

  /// Puts a node in the middle of one wire, keeping what was on both sides.
  async function insertInto(index, kind) {
    const edge = patch.edges[index];
    const from = edge && nodeOf(edge.from[0]);
    const to = edge && nodeOf(edge.to[0]);

    if (!from || !to) return;

    const a = portXY(from, kindOf(from.kind).outputs[edge.from[1]], true);
    const b = portXY(to, kindOf(to.kind).inputs[edge.to[1]], false);
    const [w, h] = baseSize({ kind });
    const fromSize = baseSize(from);
    const toSize = baseSize(to);
    const mid = { x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 };
    // Between the two cards when the gap takes it, otherwise under them: a node
    // dropped on top of its neighbours hides the very wires it joined.
    const between = mid.x - w / 2 > from.x + fromSize[0] + 20 &&
      mid.x + w / 2 < to.x - 20;
    const x = Math.round(between ? mid.x - w / 2 : from.x + fromSize[0] + 40);
    const y = Math.round(
      between
        ? mid.y - 44
        : Math.max(from.y, to.y) + Math.max(fromSize[1], toSize[1]) + 40,
    );

    await snapshot({ op: "insert" });
    suspendSnapshot += 1;

    try {
      const added = await op({ op: "add", kind, x, y }, false);
      const id = added && added.id;

      if (id !== undefined) {
        // The old wire goes first: cutting it later would use an index that the
        // two connects have already moved.
        await op({ op: "disconnect", index }, false);
        await op(
          { op: "connect", from: [edge.from[0], edge.from[1]], to: [id, 0] },
          false,
        );
        await op(
          { op: "connect", from: [id, 0], to: [edge.to[0], edge.to[1]] },
          false,
        );
      }
    } finally {
      suspendSnapshot -= 1;
    }

    draw();
  }

  /// Right-clicking a modulation wire offers to take the assignment off.
  function openModMenu(index, clientX, clientY) {
    menu.innerHTML = "";
    menu.appendChild(menuButton("Remove modulation", () => {
      op({ op: "unmodulate", index });
      closeMenu();
    }));

    placeMenu(clientX, clientY);
  }

  /// The node library, with a search box: the canvas adds what it picks, a
  /// node's menu replaces that node with it, a wire's menu puts it in between.
  function openLibrary(clientX, clientY, onPick, fits) {
    const allowed = fits || (() => true);
    menu.innerHTML = "";

    const search = document.createElement("input");
    search.className = "menu-search";
    search.placeholder = "Search nodes";
    menu.appendChild(search);

    const list = document.createElement("div");
    list.className = "menu-list";
    menu.appendChild(list);

    const pick = (kind) => {
      onPick(kind);
      closeMenu();
    };

    const render = (query) => {
      // A lifted flyout lives on the body, so clearing the list does not take it
      // with it.
      if (flyout) {
        flyout.remove();
        flyout = null;
      }

      list.innerHTML = "";
      list.classList.toggle("flat", !!query);

      if (query) {
        const matches = catalog
          .filter((kind) =>
            categoryOf(kind) && allowed(kind) && kind.name.toLowerCase().includes(query))
          .sort((a, b) => a.name.localeCompare(b.name));

        for (const kind of matches) {
          list.appendChild(menuButton(kind.name, () => pick(kind.id)));
        }

        return;
      }

      const groups = new Map();

      for (const kind of catalog) {
        const category = categoryOf(kind);
        if (!category || !allowed(kind)) continue;
        if (!groups.has(category)) groups.set(category, []);
        groups.get(category).push(kind);
      }

      for (const category of [...groups.keys()].sort((a, b) => a.localeCompare(b))) {
        const items = groups.get(category).sort((a, b) => a.name.localeCompare(b.name));
        const row = document.createElement("div");
        row.className = "menu-cat";
        const label = document.createElement("span");
        label.textContent = category;
        const arrow = document.createElement("span");
        arrow.className = "menu-arrow";
        arrow.textContent = "\u25B8";
        row.appendChild(label);
        row.appendChild(arrow);

        const sub = document.createElement("div");
        sub.className = "menu-sub";
        row.flyout = sub;
        row.addEventListener("mouseenter", () => showFlyout(row));
        row.addEventListener("mouseleave", () => hideFlyout(sub));
        sub.addEventListener("mouseenter", () => clearTimeout(flyoutTimer));
        sub.addEventListener("mouseleave", () => hideFlyout(sub));

        for (const kind of items) {
          sub.appendChild(menuButton(kind.name, () => pick(kind.id)));
        }

        row.appendChild(sub);
        list.appendChild(row);
      }
    };

    search.oninput = () => render(search.value.trim().toLowerCase());
    search.onkeydown = (event) => {
      if (event.key !== "Enter") return;
      const first = list.querySelector(".menu-item");
      if (first) first.click();
    };

    render("");
    placeMenu(clientX, clientY);
    search.focus();
  }

  svg.addEventListener("contextmenu", (event) => {
    event.preventDefault();
    const nodeEl = event.target.closest("[data-node]");
    const wireEl = event.target.closest("[data-edge]");
    const modEl = event.target.closest("[data-mod]");

    if (wireEl) {
      openWireMenu(Number(wireEl.getAttribute("data-edge")), event.clientX, event.clientY);
    } else if (modEl) {
      openModMenu(Number(modEl.getAttribute("data-mod")), event.clientX, event.clientY);
    } else if (nodeEl) {
      openNodeMenu(Number(nodeEl.getAttribute("data-node")), event.clientX, event.clientY);
    } else {
      const world = toWorld(event);
      openLibrary(event.clientX, event.clientY, (kind) => {
        op({ op: "add", kind, x: Math.round(world.x), y: Math.round(world.y) });
      });
    }
  });

  document.addEventListener("pointerdown", (event) => {
    if (!event.target.closest("#menu")) closeMenu();
  });

  // The card the page is watching. Its two sides ride along with the sources'
  // rings in one request, so every live picture on the page is the same age.
  let scopeNode = null;
  let scopeCanvas = null;

  function stopScope() {
    scopeNode = null;

    if (scopeCanvas) {
      scopeCanvas.remove();
      scopeCanvas = null;
    }
  }

  function startScope(id) {
    if (scopeNode === id) return;

    stopScope();

    const node = nodeOf(id);

    if (node && CONTROL_KINDS.includes(node.kind)) return;

    scopeNode = id;
  }

  // One tick for every live picture: the source cards' histories and the card
  // being watched. A picture read by the eye is enough at twenty a second, and
  // a hidden window does not need that many.
  const LIVE_INTERVAL = 20;
  const HIDDEN_INTERVAL = 150;
  let liveTimer = null;

  function startScopes() {
    if (liveTimer !== null) return;

    liveTimer = 0;
    pollLive();
  }

  async function pollLive() {
    const started = performance.now();

    try {
      const cursors = [];

      for (const [node, block] of sourceCursors) cursors.push(node + ":" + block);

      let query = "?since=" + cursors.join(",");

      if (scopeNode !== null) query += "&node=" + scopeNode;

      const data = await (await fetch("/scopes" + query, { cache: "no-store" })).json();

      applySourceScopes(data);

      if (data.live) applyLiveScope(data.live, data.rate);
    } catch {
      // The next tick tries again.
    }

    // Keep the cadence: the fetch's own time is not part of the interval, or
    // every point would arrive a little later than the one before it.
    const interval = document.hidden ? HIDDEN_INTERVAL : LIVE_INTERVAL;
    const elapsed = performance.now() - started;

    // A slow fetch or a heavy repaint is not a reason to start the next poll
    // immediately; a short floor keeps the loop from running flat out.
    liveTimer = setTimeout(pollLive, Math.max(4, interval - elapsed));
  }

  /// Draws the control sources' histories from one tick's payload.
  function applySourceScopes(data) {
    // A point drag owns the card until it ends: redrawing the hero here would
    // replace the handlers the drag is using.
    if (editingPoints) return;

    const blockFrames = (typeof data.block_frames === "number" && data.block_frames > 0)
      ? data.block_frames
      : 64;
    const rate = (typeof data.rate === "number" && data.rate > 0) ? data.rate : 44100;
    const blockMs = blockFrames / rate * 1000;

    for (const scope of data.scopes || []) {
      const entry = cards[scope.node];

      if (!entry || !SOURCE_KINDS.includes(entry.kind ? entry.kind.id : "")) continue;

      entry.phase = typeof scope.phase === "number" ? scope.phase : null;
      entry.input = typeof scope.input === "number" ? scope.input : null;
      entry.history = historyOf(scope.node);

      // The LFO editor's repaint clears the hero while dragging, which takes
      // the live layer with it; put it back before drawing into it again.
      if (entry.live && !entry.live.isConnected) {
        const svg = entry.card.querySelector(".mcard-hero svg");
        if (svg) svg.appendChild(entry.live);
      }

      const blocks = scope.blocks || [];
      const total = typeof scope.block === "number" ? scope.block : 0;

      if (blocks.length) {
        const cursor = sourceCursors.get(scope.node) || 0;
        const first = total - blocks.length;

        // A new scope, or a gap longer than the ring, starts a new picture: the
        // old points do not belong on the same line.
        if (total < cursor || first > cursor) entry.history.length = 0;

        // One point per block, timed by the block's own index: the trace is
        // drawn against audio time, so the poll's jitter never reaches the x
        // axis. Three seconds of points keep a little more than the 2.4 s the
        // card draws.
        const keep = Math.ceil(3000 / blockMs);

        for (let i = 0; i < blocks.length; i++) {
          entry.history.push({ t: (first + i) * blockMs, v: blocks[i] });
        }

        if (entry.history.length > keep) {
          entry.history.splice(0, entry.history.length - keep);
        }

        entry.lastValue = blocks[blocks.length - 1];
        sourceCursors.set(scope.node, total);
      }

      drawSourceLive(entry);
    }
  }

  /// Draws the card the page is watching from the same tick's payload.
  function applyLiveScope(live, rate) {
    const id = live.node;
    const samples = live.output || [];
    const arriving = live.input || [];

    // A product card's hero draws both sides of the node, the way the product
    // page's does: the filter's spectrum sits on its response curve, with
    // what arrived at it behind what left it.
    const entry = cards[id];

    if (entry && entry.context && entry.card && samples.length > 1) {
      const mono = (frames) => frames.map(([left, right]) => (left + right) * 0.5);

      // The rate the host runs at, so a spectrum lands on the right axis.
      if (rate) entry.context.rate = rate;

      entry.context.scope.input = arriving.length > 1 ? mono(arriving) : null;
      entry.context.scope.output = mono(samples);
      entry.card.repaintHero();
    }

    // The strip is drawn last: repainting the hero replaces its markup, which
    // takes the canvas with it, and drawing it first flashed the waveform on
    // every poll.
    drawScope(id, samples);
  }

  function drawScope(id, samples) {
    const node = document.querySelector(`[data-node="${id}"]`);
    const card = node && node.querySelector(".mcard");

    if (!card) return;

    // A card that draws its own picture already shows the signal: a strip on top
    // of the hero would cover the picture and say the same thing twice.
    if (card.querySelector(".mcard-hero")) return;

    if (!scopeCanvas || !scopeCanvas.isConnected || scopeCanvas.parentElement !== card) {
      scopeCanvas = document.createElement("canvas");
      scopeCanvas.className = "scope-canvas";
      scopeCanvas.width = 512;
      scopeCanvas.height = 88;
      card.appendChild(scopeCanvas);
    }

    const context = scopeCanvas.getContext("2d");
    const width = scopeCanvas.width;
    const height = scopeCanvas.height;
    context.clearRect(0, 0, width, height);

    if (samples.length < 2) return;

    let peak = 0.0001;

    for (const [, right] of samples) {
      peak = Math.max(peak, Math.abs(right));
    }

    const scale = Math.min(1, 0.5 / peak);
    const accent = getComputedStyle(card).getPropertyValue("--accent").trim();
    context.beginPath();
    context.strokeStyle = accent || "rgba(224,122,47,.9)";
    context.lineWidth = 1.5;

    const step = samples.length / width;

    for (let x = 0; x < width; x++) {
      const index = Math.floor(x * step);
      const value = samples[index] ? samples[index][1] : 0;
      const y = height / 2 - value * scale * (height / 2 - 4);

      if (x === 0) context.moveTo(x, y);
      else context.lineTo(x, y);
    }

    context.stroke();
  }

  svg.addEventListener("pointerdown", (event) => {
    if (event.target.closest(".card-host")) {
      interacting = true;
    }

    const knobEl = event.target.closest("[data-knob]");
    const wire = event.target.closest("[data-edge]");
    const mod = event.target.closest("[data-mod]");

    if (event.target.closest(".mcard .knob, .mcard button, .mcard select, .mcard input, .mcard textarea, .mcard .mcard-select, .mcard .mcard-script, .mcard-hero")) {
      return;
    }

    if (knobEl) {
      const id = Number(knobEl.getAttribute("data-node"));
      const key = knobEl.getAttribute("data-knob");
      const node = nodeOf(id);

      drag = {
        kind: "knob", id, key,
        startY: event.clientY,
        start: node.params[key] ?? 0,
        min: key === "cutoff" ? 20 : 0.3,
        max: key === "cutoff" ? 20000 : 10,
      };
      return;
    }

    if (wire) {
      selection.clear();
      primary = null;
      selectedEdge = Number(wire.getAttribute("data-edge"));
      selectedMod = null;
      draw();
      return;
    }

    if (mod) {
      selection.clear();
      primary = null;
      selectedEdge = null;
      selectedMod = Number(mod.getAttribute("data-mod"));
      draw();
      return;
    }

    const port = event.target.closest("[data-port]");
    const nodeEl = event.target.closest("[data-node]");

    if (port) {
      const id = Number(nodeEl.getAttribute("data-node"));
      drag = {
        kind: "port", id,
        output: port.getAttribute("data-port") === "out",
        control: port.classList.contains("control"),
        index: Number(port.getAttribute("data-index")),
        start: toWorld(event),
        point: toWorld(event),
        hoverPort: null,
        hoverKnob: null,
      };
      document.getElementById("stage").classList.toggle("connecting-control", drag.control && drag.output);
      draw();
      return;
    }

    if (nodeEl) {
      const id = Number(nodeEl.getAttribute("data-node"));
      const point = toWorld(event);
      const node = nodeOf(id);
      const additive = event.shiftKey || event.metaKey || event.ctrlKey;

      if (additive) {
        if (selection.has(id)) {
          selection.delete(id);
          primary = selection.size ? [...selection][selection.size - 1] : null;
          draw();
          return;
        }

        selection.add(id);
      } else if (!selection.has(id)) {
        selection.clear();
        selection.add(id);
      }

      primary = id;
      draw();
      startScope(id);

      // The scale handle scales the card, not moves it.
      if (event.target.closest(".scale-handle")) {
        drag = { kind: "scale", id, start: { ...scaleOf(id) }, from: point };
        return;
      }

      const start = [...selection].map((nid) => {
        const item = nodeOf(nid);
        return { id: nid, x: item.x, y: item.y };
      });

      drag = {
        kind: "node", id,
        dx: point.x - node.x, dy: point.y - node.y,
        start, origin: point,
      };
      return;
    }

    selection.clear();
    primary = null;
    stopScope();

    if (event.button === 1 || spaceHeld) {
      drag = { kind: "pan", x: event.clientX, y: event.clientY, vx: view.x, vy: view.y };
    } else {
      const point = toWorld(event);
      drag = { kind: "marquee", from: point, to: point };
    }

    draw();
  });

  svg.addEventListener("pointermove", (event) => {
    if (!drag) return;

    if (drag.kind === "node") {
      const point = toWorld(event);

      if (drag.start.length > 1) {
        const dx = point.x - drag.origin.x;
        const dy = point.y - drag.origin.y;

        for (const item of drag.start) {
          const node = nodeOf(item.id);
          node.x = Math.round(item.x + dx);
          node.y = Math.round(item.y + dy);
        }
      } else {
        const node = nodeOf(drag.id);
        node.x = Math.round(point.x - drag.dx);
        node.y = Math.round(point.y - drag.dy);
      }

      draw();
    } else if (drag.kind === "marquee") {
      drag.to = toWorld(event);
      draw();
    } else if (drag.kind === "knob") {
      const node = nodeOf(drag.id);
      const span = drag.max - drag.min;
      let value = drag.start + ((drag.startY - event.clientY) / 120) * span;
      value = Math.min(drag.max, Math.max(drag.min, value));
      if (drag.key === "cutoff") value = Math.round(value);
      node.params[drag.key] = value;
      draw();
    } else if (drag.kind === "scale") {
      const point = toWorld(event);
      const sx = Math.max(0.1, drag.start.x * (1 + (point.x - drag.from.x) / 160));
      const sy = Math.max(0.1, drag.start.y * (1 + (point.y - drag.from.y) / 120));
      scales[drag.id] = { x: sx, y: sy };
      draw();
    } else if (drag.kind === "pan") {
      view.x = drag.vx + (event.clientX - drag.x);
      view.y = drag.vy + (event.clientY - drag.y);
      draw();
    } else if (drag.kind === "port") {
      drag.point = toWorld(event);

      const source = nodeOf(drag.id);
      const kind = kindOf(source.kind);
      const signal = (drag.output ? kind.outputs : kind.inputs)[drag.index].kind;
      let best = null;
      let bestDistance = 18;

      for (const other of patch.nodes) {
        if (other.id === drag.id) continue;
        const otherKind = kindOf(other.kind);
        const ports = drag.output ? otherKind.inputs : otherKind.outputs;

        ports.forEach((port, index) => {
          // A table may land on an oscillator port: the engine reads it through
          // a table oscillator, and the page has to offer the same pair.
          const feeds = port.kind === signal
            || (signal === "wavetable" && port.kind === "osc")
            || (signal === "osc" && port.kind === "wavetable");
          if (!feeds) return;
          const at = portXY(other, port, !drag.output);
          const distance = Math.hypot(at.x - drag.point.x, at.y - drag.point.y);

          if (distance < bestDistance) {
            bestDistance = distance;
            best = { id: other.id, index, output: !drag.output };
          }
        });
      }

      drag.hoverPort = best;

      if (drag.control && drag.output) {
        const under = document.elementFromPoint(event.clientX, event.clientY);
        const knobRow = (under && under.closest("[data-drop-target]")) ||
          event.target.closest("[data-mod-target]");

        if (drag.hoverKnob !== knobRow) {
          if (drag.hoverKnob) drag.hoverKnob.classList.remove("mod-hover");
          drag.hoverKnob = knobRow || null;
          if (drag.hoverKnob) drag.hoverKnob.classList.add("mod-hover");
        }
      }

      draw();
    }
  });

  svg.addEventListener("pointerup", (event) => {
    if (!drag) return;

    if (drag.kind === "node") {
      // The positions are read now: each op replaces the page's patch with the
      // server's, which would undo the moves that have not been sent yet.
      const moves = (drag.start.length ? drag.start : [{ id: drag.id }]).map((item) => {
        const node = nodeOf(item.id);
        return { id: item.id, x: node.x, y: node.y };
      });

      (async () => {
        for (const move of moves) {
          await op({ op: "move", id: move.id, x: move.x, y: move.y }, false);
        }

        draw();
      })();
    } else if (drag.kind === "marquee") {
      const a = drag.from;
      const b = drag.to;
      const left = Math.min(a.x, b.x);
      const right = Math.max(a.x, b.x);
      const top = Math.min(a.y, b.y);
      const bottom = Math.max(a.y, b.y);

      selection.clear();

      for (const node of patch.nodes) {
        const { w, h } = size(node);
        const inside = node.x + w >= left && node.x <= right &&
          node.y + h >= top && node.y <= bottom;

        if (inside) selection.add(node.id);
      }

      primary = selection.size ? [...selection][selection.size - 1] : null;
      draw();
    } else if (drag.kind === "knob") {
      const node = nodeOf(drag.id);
      op({ op: "param", id: node.id, key: drag.key, value: node.params[drag.key] });
    } else if (drag.kind === "scale") {
      // The scale is a page affordance, not part of the patch.
    } else if (drag.kind === "port" && drag.control && drag.output) {
      // A control wire ends either on a parameter, which is a modulation, or
      // on a control input port, which is a node that maps the signal. The
      // port is the more specific target, so it wins when it is under the
      // pointer.
      if (drag.hoverPort && drag.hoverPort.output !== drag.output) {
        const voice = patch.voice || [];

        if (voice.includes(drag.hoverPort.id) === voice.includes(drag.id)) {
          op({
            op: "connect",
            from: [drag.id, drag.index],
            to: [drag.hoverPort.id, drag.hoverPort.index],
          });
        } else {
          showStatus(
            "A control wire cannot cross the VOICE group: move both nodes into the same group.",
          );
        }

        document.getElementById("stage").classList.remove("connecting-control");
        drag = null;
        draw();
        return;
      }

      const row = event.target.closest("[data-mod-target]");
      const target = event.target.closest("[data-node]");
      const under = document.elementFromPoint(event.clientX, event.clientY);
      const knob = under ? under.closest("[data-drop-target]") : null;

      let id = null;
      let key = null;

      if (row) {
        [id, key] = row.getAttribute("data-mod-target").split(":");
        id = Number(id);
      } else if (knob) {
        const [owner, param] = knob.getAttribute("data-drop-target").split(":");
        if (Number(owner) !== drag.id) {
          id = Number(owner);
          key = param;
        }
      } else if (target && Number(target.getAttribute("data-node")) !== drag.id) {
        id = Number(target.getAttribute("data-node"));
        key = kindOf(nodeOf(id).kind).params[0]?.[0] ?? null;
      }

      if (id !== null && key) {
        // A source that modulates a voice node joins the voice group itself, so
        // everything else it drives has to be in the group too; the compiler
        // rejects the old link otherwise.
        const voice = patch.voice || [];
        const targetInside = voice.includes(id);
        const sourceInside = voice.includes(drag.id);

        if (targetInside && !sourceInside) {
          const outside = patch.mods.filter(
            (mod) => mod.from[0] === drag.id && !voice.includes(mod.to[0]),
          );

          if (outside.length) {
            const other = nodeOf(outside[0].to[0]);
            const name = (other && kindOf(other.kind).name) || "a node";

            showStatus(
              "The source would join the VOICE group, but it also modulates " +
                name +
                " outside it: move that node inside the group, or use another source.",
            );

            return;
          }
        } else if (!targetInside && sourceInside) {
          showStatus("A control inside the VOICE group cannot modulate a node outside it.");

          return;
        }

        const depth = defaultDepth(kindOf(nodeOf(id).kind), key);

        op({ op: "modulate", from: [drag.id, drag.index], to: [id, key], depth });
      }
    } else if (drag.kind === "port") {
      const target = drag.hoverPort;

      if (target && target.output !== drag.output) {
        const from = drag.output ? [drag.id, drag.index] : [target.id, target.index];
        const to = drag.output ? [target.id, target.index] : [drag.id, drag.index];
        op({ op: "connect", from, to });
      }
    }

    if (drag.kind === "port") {
      document.getElementById("stage").classList.remove("connecting-control");
      if (drag.hoverKnob) drag.hoverKnob.classList.remove("mod-hover");
    }

    const wasPort = drag.kind === "port";
    drag = null;

    if (wasPort) draw();
  });

  function zoomAt(factor, clientX, clientY) {
    const rect = svg.getBoundingClientRect();
    const point = { x: clientX - rect.left, y: clientY - rect.top };
    const before = { x: (point.x - view.x) / view.zoom, y: (point.y - view.y) / view.zoom };
    view.zoom = Math.min(2, Math.max(0.35, view.zoom * factor));
    view.x = point.x - before.x * view.zoom;
    view.y = point.y - before.y * view.zoom;
    draw();
  }

  function zoomBy(factor) {
    const rect = svg.getBoundingClientRect();
    zoomAt(factor, rect.left + rect.width / 2, rect.top + rect.height / 2);
  }

  /// Frames every node in the view.
  function fitView() {
    if (!patch.nodes.length) {
      view = { x: 30, y: 30, zoom: 1 };
      draw();
      return;
    }

    let minX = Infinity;
    let minY = Infinity;
    let maxX = -Infinity;
    let maxY = -Infinity;

    for (const node of patch.nodes) {
      const { w, h } = size(node);
      minX = Math.min(minX, node.x);
      minY = Math.min(minY, node.y);
      maxX = Math.max(maxX, node.x + w);
      maxY = Math.max(maxY, node.y + h);
    }

    const rect = svg.getBoundingClientRect();
    const pad = 60;
    const zoom = Math.min(
      2,
      Math.max(0.35, Math.min(
        (rect.width - pad * 2) / Math.max(1, maxX - minX),
        (rect.height - pad * 2) / Math.max(1, maxY - minY),
      )),
    );

    view.zoom = zoom;
    view.x = rect.width / 2 - (minX + maxX) / 2 * zoom;
    view.y = rect.height / 2 - (minY + maxY) / 2 * zoom;
    draw();
  }

  svg.addEventListener("wheel", (event) => {
    event.preventDefault();
    zoomAt(event.deltaY < 0 ? 1.1 : 0.9, event.clientX, event.clientY);
  }, { passive: false });

  window.addEventListener("pointerup", () => {
    interacting = false;
  });

  window.addEventListener("keydown", (event) => {
    if (event.key === "Escape") {
      closeMenu();
      return;
    }

    if ((event.ctrlKey || event.metaKey) && !event.altKey) {
      const key = event.key.toLowerCase();

      if (key === "z") {
        if (event.shiftKey) redo();
        else undo();
        event.preventDefault();
        return;
      }

      if (key === "y") {
        redo();
        event.preventDefault();
        return;
      }

      if (key === "c") {
        copySelection();
        event.preventDefault();
        return;
      }

      if (key === "v") {
        pasteClipboard();
        event.preventDefault();
        return;
      }
    }

    if (event.key === " " && !event.target.closest("input, textarea, select")) {
      spaceHeld = true;
      event.preventDefault();
      return;
    }

    if (event.key !== "Delete" && event.key !== "Backspace") return;

    if (selectedEdge !== null) {
      op({ op: "disconnect", index: selectedEdge });
      selectedEdge = null;
    } else if (selectedMod !== null) {
      op({ op: "unmodulate", index: selectedMod });
      selectedMod = null;
    } else if (selection.size) {
      const ids = [...selection];
      selection.clear();
      primary = null;

      (async () => {
        await snapshot({ op: "remove" });
        suspendSnapshot += 1;

        try {
          for (const id of ids) {
            delete cards[id];
            await op({ op: "remove", id }, false);
          }
        } finally {
          suspendSnapshot -= 1;
        }

        draw();
      })();
    }
  });

  window.addEventListener("keyup", (event) => {
    if (event.key === " ") spaceHeld = false;
  });

  // The patch saves as the text the host state stores, and opens from one.
  document.getElementById("undo").onclick = undo;
  document.getElementById("redo").onclick = redo;
  document.getElementById("zoom-in").onclick = () => zoomBy(1.2);
  document.getElementById("zoom-out").onclick = () => zoomBy(1 / 1.2);
  document.getElementById("zoom-fit").onclick = fitView;
  updateHistory();

  document.getElementById("save").onclick = async () => {
    const text = await (await fetch("/text")).json();
    const link = document.createElement("a");
    link.href = URL.createObjectURL(new Blob([text], { type: "text/plain" }));
    link.download = "patch.txt";
    link.click();
    URL.revokeObjectURL(link.href);
  };

  document.getElementById("open").onclick = () => document.getElementById("file").click();

  document.getElementById("file").onchange = async (event) => {
    const file = event.target.files && event.target.files[0];

    if (!file) return;

    const text = await file.text();
    const result = await (await fetch("/text", { method: "POST", body: text })).json();

    if (result.error) console.warn(result.error);

    patch = readPatch(await (await fetch("/patch")).json());
    draw();
    event.target.value = "";
  };

  startScopes();

  load();
  readMacros().then(draw);

  // A host moving a macro does not touch the graph, so the revision poll below
  // never sees it. The values are small enough to read on their own faster timer,
  // and a card is redrawn only when one of them actually moved.
  setInterval(async () => {
    if (busy() || drag) return;

    const before = macroValues.join(",");
    await readMacros();

    if (macroValues.join(",") !== before) draw();
  }, 120);

  // The graph changes from outside the page too: the host restores its saved
  // state, another client posts text, the fitting route writes parameters back.
  // One small request says whether anything moved, and the graph itself is only
  // read when it did.
  let patchRevision = -1;
  let spectrumTick = 0;

  setInterval(async () => {
    if (busy()) return;

    try {
      const state = await (await fetch("/state", { cache: "no-store" })).json();

      serverError = state.error || "";
      paintStatus();

      // The drawn pages are not part of the graph, so a page edited in another
      // window does not move the revision: ask for them on a slow tick.
      if (++spectrumTick >= 8) {
        spectrumTick = 0;

        for (const entry of Object.values(cards)) {
          if (entry && entry.kind && isProductCard(entry.kind, "spectrum")) refreshSpectrum(entry);
        }

      }

      if (state.revision === patchRevision) return;

      patchRevision = state.revision;

      const next = readPatch(await (await fetch("/patch", { cache: "no-store" })).json());

      if (JSON.stringify(next) === JSON.stringify(patch)) return;

      patch = next;
      refreshWaves();
      draw();
    } catch {
      // The next tick tries again.
    }
  }, 250);
})();
