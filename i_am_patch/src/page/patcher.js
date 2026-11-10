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
  let selected = null;
  let selectedEdge = null;
  let selectedMod = null;
  let view = { x: 30, y: 30, zoom: 1 };
  let drag = null;
  // Each card scales by itself: x and y are independent, so a wide filter and
  // a narrow one can sit side by side at the sizes they need.
  const scales = {};

  const cards = {};
  const pagesOf = {};

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
  const SOURCE_KINDS = ["lfo", "envelope", "random", "table", "pitch", "midi"];

  function rackNode(name, attrs) {
    const node = document.createElementNS(NS, name);

    for (const [key, value] of Object.entries(attrs || {})) {
      node.setAttribute(key, String(value));
    }

    return node;
  }

  

  function drawSourceStatic(svg, node, kind) {
    const params = (node && node.params) || {};

    if (kind === "lfo") {
      const shape = Math.round(params.shape ?? 0);
      const points = node && node.points && node.points.length >= 2
        ? node.points
        : IAmCard.sources.preset(shape);

      IAmCard.sources.drawPreview(svg, "lfo", { points, shape });
    } else if (kind === "table") {
      IAmCard.sources.drawPreview(svg, "table", {
        shape: Math.round(params.shape ?? 0),
        morph: params.morph ?? 0,
      });
    } else {
      IAmCard.sources.drawPreview(svg, kind, {});
    }
  }

  function drawSourceLive(svg, node, kind, samples, phase, _accent) {
    if (!samples || samples.length < 2) return;

    const values = samples.map((frame) => frame[0]);
    const key = "patch";

    IAmCard.sources.drawLive(svg, kind, {
      history: { [key]: values },
      phases: { [key]: typeof phase === "number" ? phase : 0 },
      values: { [key]: values[values.length - 1] },
    }, key, false);
  }

  // The control sources' pictures, the ones the rack's modulation page draws:
  // a wave for the LFO, a stage shape for the envelope, a table and its morph,
  // and a smoothed random walk.
  if (window.IAmCard) {
    const heroSvg = (inner) =>
      `<svg viewBox="0 0 520 160" preserveAspectRatio="none">${inner}</svg>`;

    const shapeAt = (shape, phase) => IAmCard.sources.shape(shape, phase);

    window.IAmCard.registerHero("patch-lfo", (values, context) => {
      const table = values.points ? JSON.parse(values.points) : [];
      const phase = context && typeof context.phase === "number" ? context.phase : null;
      const playhead = (value, y) => {
        if (phase === null) return "";

        const x = (20 + phase * 480).toFixed(1);

        return `<line x1="${x}" y1="20" x2="${x}" y2="140" stroke="var(--accent)" stroke-width="1" opacity=".5"/>` +
          `<circle cx="${x}" cy="${y.toFixed(1)}" r="4" fill="var(--accent)"/>`;
      };

      if (table.length >= 2) {
        let line = "";

        for (const [time, value] of table) {
          line += (line ? " L " : "M ") + (20 + time * 480).toFixed(1) + " " +
            (140 - value * 120).toFixed(1);
        }

        const dots = table.map(([time, value]) =>
          `<circle cx="${(20 + time * 480).toFixed(1)}" cy="${(140 - value * 120).toFixed(1)}" r="2.5" fill="var(--accent)"/>`,
        ).join("");

        let value = 0;

        if (phase !== null) {
          for (let index = 0; index + 1 < table.length; index++) {
            if (phase >= table[index][0] && phase <= table[index + 1][0]) {
              const span = table[index + 1][0] - table[index][0] || 1;
              value = table[index][1] + (table[index + 1][1] - table[index][1]) *
                ((phase - table[index][0]) / span);
            }
          }
        }

        return heroSvg(
          '<line x1="20" y1="80" x2="500" y2="80" stroke="#2f2f2f" stroke-dasharray="3 5"/>' +
          `<path d="${line}" fill="none" stroke="var(--accent)" stroke-width="2" vector-effect="non-scaling-stroke"/>` +
          dots + playhead(value, 140 - value * 120),
        );
      }

      const shape = Math.round(values.shape ?? 0);
      let path = "";

      for (let index = 0; index <= 240; index++) {
        const t = index / 240;
        const value = shapeAt(shape, (t * 2) % 1);
        path += (index ? " L " : "M ") + (20 + t * 480).toFixed(1) + " " +
          (80 - value * 55).toFixed(1);
      }

      const value = phase === null ? 0 : shapeAt(shape, phase);

      return heroSvg(
        '<line x1="20" y1="80" x2="500" y2="80" stroke="#2f2f2f" stroke-dasharray="3 5"/>' +
        `<path d="${path}" fill="none" stroke="var(--accent)" stroke-width="2" vector-effect="non-scaling-stroke"/>` +
        playhead(value, 80 - value * 55),
      );
    });

    window.IAmCard.registerHero("patch-envelope", (values) => {
      // The envelope source follows the input: its picture is an input pulse
      // with the follower riding on it, not an ADSR's decay.
      const attack = Math.max(values.attack ?? 10, 0.1);
      const release = Math.max(values.release ?? 100, 0.1);
      const total = attack + release;
      const rise = Math.min(0.55, (attack / total) * 0.9);
      const fall = Math.min(0.55, (release / total) * 0.9);
      const on = 0.12;
      const off = 1 - fall;
      const top = 30;
      const bottom = 140;
      const x = (t) => (20 + t * 480).toFixed(1);
      const y = (value) => (bottom - value * (bottom - top)).toFixed(1);
      const path = `M ${x(0)} ${y(0)} L ${x(on)} ${y(0)} ` +
        `L ${x(on + rise)} ${y(1)} L ${x(off)} ${y(1)} L ${x(1)} ${y(0)}`;
      const pulse = `<rect x="${x(on)}" y="${y(1)}" width="${(off - on) * 480}" ` +
        `height="${bottom - top}" fill="#2a2a2a"/>`;

      return heroSvg(
        pulse +
        `<path d="${path}" fill="none" stroke="var(--accent)" stroke-width="2" vector-effect="non-scaling-stroke"/>`,
      );
    });

    window.IAmCard.registerHero("patch-table", (values) => {
      const shape = Math.round(values.shape ?? 0);
      const morph = Math.min(Math.max(values.morph ?? 0, 0), 1);
      let path = "";

      for (let index = 0; index <= 240; index++) {
        const phase = (index / 240) % 1;
        const value = shapeAt(shape, phase) * (1 - morph) + shapeAt(shape + 1, phase) * morph;
        path += (index ? " L " : "M ") + (20 + (index / 240) * 480).toFixed(1) + " " +
          (80 - value * 55).toFixed(1);
      }

      return heroSvg(
        '<line x1="20" y1="80" x2="500" y2="80" stroke="#2f2f2f" stroke-dasharray="3 5"/>' +
        `<path d="${path}" fill="none" stroke="var(--accent)" stroke-width="2" vector-effect="non-scaling-stroke"/>`,
      );
    });

    window.IAmCard.registerHero("patch-random", (values) => {
      const smooth = Math.max(values.smooth ?? 20, 1);
      let seed = 0x1f3d5b79;
      let value = 0;
      let target = 0;
      let path = "";
      const rows = 48;

      for (let index = 0; index <= rows; index++) {
        if (index % 6 === 0) {
          seed ^= seed << 13;
          seed ^= seed >>> 17;
          seed ^= seed << 5;
          target = ((seed >>> 0) / 0xffffffff) * 2 - 1;
        }

        value += (target - value) * Math.min(1, 6 / smooth);
        path += (index ? " L " : "M ") + (20 + (index / rows) * 480).toFixed(1) + " " +
          (80 - value * 55).toFixed(1);
      }

      return heroSvg(
        '<line x1="20" y1="80" x2="500" y2="80" stroke="#2f2f2f" stroke-dasharray="3 5"/>' +
        `<path d="${path}" fill="none" stroke="var(--accent)" stroke-width="2" vector-effect="non-scaling-stroke"/>`,
      );
    });
  }

  const CARD_HEROES = {
    lfo: "patch-source",
    envelope: "patch-source",
    random: "patch-source",
    table: "patch-source",
    pitch: "patch-source",
    midi: "patch-source",
  };

  // The source hero is a drawing surface the page fills after every repaint.
  if (window.IAmCard) {
    window.IAmCard.registerHero("patch-source", () =>
      '<svg viewBox="0 0 300 70" preserveAspectRatio="none" class="rack-hero"></svg>',
    );
  }

  function drawSourceCard(entry) {
    if (!entry || !entry.card || !entry.kind) return;

    const svg = entry.card.querySelector(".mcard-hero svg");
    if (!svg) return;

    const params = (entry.node && entry.node.params) || {};

    if (entry.kind.id === "lfo" && (!entry.node.points || entry.node.points.length < 2)) {
      entry.node.points = IAmCard.sources.preset(Math.round(params.shape ?? 0));
    }

    svg.textContent = "";
    drawSourceStatic(svg, entry.node, entry.kind.id);

    const accent = getComputedStyle(entry.card).getPropertyValue("--accent").trim() || "#e07a2f";
    drawSourceLive(svg, entry.node, entry.kind.id, entry.samples, entry.phase, accent);

    if (entry.kind.id === "lfo") {
      IAmCard.sources.editLfo(svg, entry.node, {
        shape: params.shape,
        commit: (points) => {
          op({
            op: "set_points",
            id: entry.node.id,
            points: points.map((point) => [point.time, point.value, point.bend || 0]),
          }, false);
        },
        onStart: () => { editingPoints = true; },
        onEnd: () => { editingPoints = false; },
      });
    }
  }

  // Ranges for the kinds that are not products: their catalog entry only has a
  // default, so the page gives the knob a sensible travel.
  const PARAM_RANGES = {
    beats: { min: 0.125, max: 8, log: true },
    sync: { min: 0, max: 1, step: 1 },
    rate: { min: 0.01, max: 200, log: true },
    smooth: { min: 0.1, max: 500, log: true },
    shape: { min: 0, max: 3, step: 1 },
    morph: { min: 0, max: 1 },
    partials: { min: 1, max: 64, step: 1 },
    spread: { min: 0, max: 1 },
    decay: { min: 0.05, max: 30, log: true },
    gain: { min: 0, max: 2, log: true },
    pan: { min: -1, max: 1 },
    a: { min: 0, max: 2 },
    b: { min: 0, max: 2 },
    time: { min: 1, max: 1000 },
    feedback: { min: 0, max: 0.95 },
    attack: { min: 0.1, max: 2000, log: true },
    release: { min: 0.1, max: 2000, log: true },
    clarity: { min: 0, max: 1 },
    glide: { min: 0.1, max: 1000, log: true },
    mode: { min: 0, max: 1, step: 1 },
    transpose: { min: -24, max: 24, step: 1 },
  };

  // The rack's source cards put these choices in the header instead of a knob.
  const GENERIC_HEADER = {
    lfo: ["shape", "sync"],
    table: ["shape", "sync"],
    random: ["sync"],
    midi: ["mode"],
  };
  const GENERIC_CHOICES = {
    shape: ["Sine", "Triangle", "Saw", "Square"],
    mode: ["Note", "Velocity"],
  };

  const NODE_NOTE = {
    "input:midi": "Host MIDI input",
    "input:audio": "Host audio input",
    "input:sidechain": "Host sidechain input",
    "out:audio": "Host audio output",
    "voice:in": "A note starts the voice group here",
    "voice:out": "The voice group's audio leaves here",
  };

  function labelOf(key) {
    return key.replace(/_/g, " ").replace(/\b\w/g, (letter) => letter.toUpperCase());
  }

  /// The modulation ring and the two gestures a knob takes, for one parameter.
  function modulationSpec(node, control) {
    const range = Math.max(1e-9, control.max - control.min);
    const modulations = modsOf(node.id)
      .filter((mod) => mod.to[1] === control.id)
      .map((mod) => ({
        slot: mod.slot,
        label: sourceName(mod.from[0]),
        depth: mod.depth / range,
        mode: mod.mode === "base" ? "frombase" : "centered",
      }));

    return {
      modulations,
      onModulation: (slot, next) => {
        op({ op: "mod_depth", index: slot, depth: next * range }, false);
      },
      onModeToggle: (slots, next) => {
        op({ op: "mod_mode", index: slots[0], mode: next === "center" ? "center" : "base" }, false);
      },
    };
  }

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
    }

    if (!entry) {
      // A paged product shows one page's sections at a time; the tab lives in
      // the card's header, where the product's own header controls are.
      const headerControls = (kind.header || []).flatMap((id) => {
        const control = (kind.controls || []).find((item) => item.id === id);

        return control ? [control] : [];
      });

      const shown = pages.length
        ? (pages[activePage].sections || []).flatMap((name) => {
            const section = (kind.sections || []).find((item) => item.name === name);

            return section ? section.controls : [];
          }).concat(headerControls.filter((control) =>
            !(pages[activePage].sections || []).some((name) => {
              const section = (kind.sections || []).find((item) => item.name === name);
              return section && section.controls.some((item) => item.id === control.id);
            })))
        : (kind.controls || []);

      const specs = shown.map((control) => ({
        id: control.id,
        label: control.label,
        kind: control.kind,
        unit: control.unit,
        min: control.min,
        max: control.max,
        default: control.default,
        log: control.log,
        bipolar: control.bipolar,
        step: control.step,
        widget: control.widget,
        size: control.size,
        choices: control.choices,
        ...modulationSpec(node, control),
      }));

      const host = document.createElement("div");
      host.className = "card-host";
      const scale = document.createElement("div");
      scale.className = "card-scale";
      scale.style.width = `${baseWidth(node)}px`;
      scale.style.height = `${baseHeight(node)}px`;
      scale.style.flexBasis = `${baseWidth(node)}px`;
      const context = { rate: 48000, scope: { input: null, output: null }, spectra: {} };
      const pageHero = pages.length ? pages[activePage].hero : null;
      const card = window.IAmCard.create({
        title: kind.title || kind.name,
        accent: kind.accent || "#e07a2f",
        hero: pageHero || kind.hero || "spectrum",
        mode: kind.mode || null,
        header: kind.header || [],
        visible: kind.visible || null,
        switches: kind.switches || [],
        controls: specs,
        values: node.params,
        context,
        onEdit: (identifier, value) => {
          // The card already shows the value; rebuilding the graph here would
          // take the focus away from the control being dragged.
          op({ op: "param", id: node.id, key: identifier, value }, false);
        },
      });
      if (pages.length > 1) {
        const tabs = document.createElement("div");
        tabs.className = "mcard-tabs";

        pages.forEach((page, index) => {
          const tab = document.createElement("button");
          tab.className = "mcard-tab" + (index === activePage ? " active" : "");
          tab.textContent = page.name;
          tab.onclick = () => {
            pagesOf[node.id] = index;
            delete cards[node.id];
            draw();
          };
          tabs.appendChild(tab);
        });

        const head = card.querySelector(".mcard-head");

        if (head) head.after(tabs);
        else card.insertBefore(tabs, card.firstChild);
      }

      scale.appendChild(card);
      host.appendChild(scale);
      entry = { host, key, kindId: node.kind, context, card };
      cards[node.id] = entry;
      entry.host.style.setProperty(
        "--k",
        String(Math.max(0.2, Math.min(w / baseWidth(node), h / baseHeight(node)))),
      );
      entry.host.classList.toggle("compact", w < 430);
    }

    return entry.host;
  }


  /// A card for a kind that is not a product: knobs from its catalog params,
  /// and a line of text when it has none.
  function genericCard(node, kind, w, h) {
    let entry = cards[node.id];
    const mods = modsOf(node.id);
    const key = JSON.stringify([
      mods.map((mod) => [mod.slot, mod.to[1], Math.round(mod.depth * 1000), mod.mode]),
      node.points || [],
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
        String(Math.max(0.2, Math.min(w / baseWidth(node), h / baseHeight(node)))),
      );
      entry.host.classList.toggle("compact", w < 430);
      drawSourceCard(entry);

      return entry.host;
    }

    const context = { rate: 48000 };
    const specs = (kind.params || []).map(([id, fallback]) => {
      const range = PARAM_RANGES[id] || {};
      const choices = GENERIC_CHOICES[id] || null;
      const span = Math.max(0.001, Math.abs(fallback) || 1);
      const control = {
        id,
        label: labelOf(id),
        kind: id === "sync" ? "bool" : choices || range.step === 1 ? "int" : "float",
        min: id === "sync" ? 0 : range.min ?? -span * 2,
        max: id === "sync" ? 1 : choices ? choices.length - 1 : range.max ?? span * 2,
        default: fallback,
        log: !!range.log,
        step: choices ? 1 : range.step,
        choices,
      };

      return { ...control, ...modulationSpec(node, control) };
    });

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
      values: { ...node.params, points: JSON.stringify((node.points || []).map((point) => (Array.isArray(point) ? point : [point.time, point.value, point.bend || 0]))) },
      context,
      onEdit: (identifier, value) => {
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

    if (SOURCE_KINDS.includes(kind.id)) {
      const repaint = card.repaintHero;
      card.repaintHero = function () {
        repaint();
        drawSourceCard(entry);
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

  const kindOf = (id) => catalog.find((kind) => kind.id === id);
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
    "voice:in": [260, 150],
    "voice:out": [260, 150],
    // The source cards are wide like the rack's: a full-width hero and the
    // knobs in a row under it.
    lfo: [640, 340],
    envelope: [640, 340],
    random: [640, 340],
    table: [640, 340],
    pitch: [600, 320],
    midi: [600, 320],
    "gen:additive": [300, 210],
    mixer: [280, 190],
    "split:ms": [280, 160],
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
    }

    return patch;
  }

  async function load() {
    catalog = await (await fetch("/catalog")).json();
    patch = readPatch(await (await fetch("/patch")).json());
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

  async function op(body, redraw = true) {
    const response = await fetch("/op", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(body),
    });
    const result = await response.json();
    if (result.error) console.warn(result.error);
    patch = readPatch(await (await fetch("/patch")).json());
    if (redraw) draw();
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
  function filterMagnitude(state, frequency, rate) {
    const cutoff = Math.min(Math.max(state.cutoff ?? 2000, 20), rate * 0.45);
    const q = Math.min(Math.max(state.q ?? 0.707, 0.3), 10);
    const w0 = 2 * Math.PI * cutoff / rate;
    const alpha = Math.sin(w0) / (2 * q);
    const cos = Math.cos(w0);
    const a0 = 1 + alpha;
    const b0 = (1 - cos) / 2 / a0;
    const b1 = (1 - cos) / a0;
    const b2 = b0;
    const a1 = -2 * cos / a0;
    const a2 = (1 - alpha) / a0;
    const w = 2 * Math.PI * frequency / rate;
    const c1 = Math.cos(w), s1 = Math.sin(w);
    const c2 = Math.cos(2 * w), s2 = Math.sin(2 * w);
    const num = Math.hypot(b0 + b1 * c1 + b2 * c2, -(b1 * s1 + b2 * s2));
    const den = Math.hypot(1 + a1 * c1 + a2 * c2, -(a1 * s1 + a2 * s2));
    return 20 * Math.log10(num / Math.max(den, 1e-9));
  }

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

      world.appendChild(el("path", { d, class: "wire-hit", "data-edge": index }));
      world.appendChild(el("path", {
        d,
        class: "wire" + (selectedEdge === index ? " selected" : ""),
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
        class: "node" + (selected === node.id ? " selected" : ""),
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

      if (selected === node.id) {
        group.appendChild(el("rect", {
          class: "scale-handle", x: w - 14, y: h - 14, width: 12, height: 12, rx: 2,
        }));
      }

      world.appendChild(group);
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
          if (port.kind !== signal) return;
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
    if (node.kind === "filter") {
      const width = 216, height = 76;
      const path = el("path", {
        fill: "none",
        stroke: "#4d9dff",
        "stroke-width": "1.8",
      });
      const points = [];

      for (let x = 0; x <= width; x += 2) {
        const frequency = 20 * Math.pow(1000, x / width);
        const db = filterMagnitude(node.params, frequency, 48000);
        const y = height * (1 - (Math.max(-48, Math.min(6, db)) + 48) / 54);
        points.push(`${12 + x},${30 + y}`);
      }

      path.setAttribute("d", "M " + points.join(" L "));
      group.appendChild(path);
      group.appendChild(knob(node, 70, 132, 20, "cutoff", node.params.cutoff ?? 2000, 20, 20000));
      group.appendChild(knob(node, 170, 132, 20, "resonance", node.params.resonance ?? 0.707, 0.3, 10));
    } else {
      const rows = kind.params.slice(0, 6);

      rows.forEach(([key, fallback], index) => {
        const y = 42 + index * 20;
        const row = el("g", { "data-mod-target": `${node.id}:${key}` });
        const label = el("text", { class: "card-param", x: 12, y });
        label.textContent = key;
        row.appendChild(label);
        const value = el("text", { class: "card-value", x: BASE_W - 12, y, "text-anchor": "end" });
        value.textContent = (node.params[key] ?? fallback).toFixed(2);
        row.appendChild(value);
        row.appendChild(el("rect", { x: 6, y: y - 13, width: BASE_W - 12, height: 18, fill: "transparent" }));
        group.appendChild(row);
      });
    }
  }

  function knob(node, cx, cy, r, key, value, min, max) {
    const group = el("g", { "data-knob": key, "data-node": node.id, "data-mod-target": `${node.id}:${key}` });
    group.appendChild(el("circle", { cx, cy, r: r + 3, fill: "transparent" }));
    group.appendChild(el("circle", { cx, cy, r, fill: "#1a1a1a", stroke: "#333", "stroke-width": 2 }));

    const t = Math.min(1, Math.max(0, (value - min) / (max - min)));
    const angle = -135 + 270 * t;
    const arc = el("path", {
      d: describeArc(cx, cy, r, -135, angle),
      fill: "none",
      stroke: "#e07a2f",
      "stroke-width": 3,
      "stroke-linecap": "round",
    });
    group.appendChild(arc);

    const text = el("text", { class: "card-value", x: cx, y: cy + 34, "text-anchor": "middle" });
    text.textContent = formatKnob(key, value);
    group.appendChild(text);

    return group;
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

  // A right-click opens the node library: categories, then the nodes inside
  // them, alphabetically. No list takes space from the graph until it is asked.
  // Four groups: an effect processes audio, a generator makes it, a control
  // steers a parameter, and the host ports and the voice boundaries are the
  // wiring around them.
  function categoryOf(kind) {
    const id = kind.id;

    if (id.startsWith("mod:")) return "Effect";

    // The bare filter, gain and delay are aliases of the modules, kept for old
    // patches and the tests; the menu offers the modules themselves.
    if (["filter", "gain", "delay"].includes(id)) return null;

    // The bare additive generator is the small version of the Additive product,
    // which can do everything it can; old patches keep parsing, the menu offers
    // one Additive.
    if (id === "gen:additive") return null;

    if (["mixer", "split:ms"].includes(id)) return "Effect";
    if (id.startsWith("gen:") || id.startsWith("synth:")) return "Generator";
    if (["lfo", "envelope", "random", "table", "pitch", "midi"].includes(id)) return "Control";

    return "I/O";
  }

  function closeMenu() {
    menu.style.display = "none";
  }

  function placeMenu(clientX, clientY) {
    menu.style.display = "block";
    menu.style.left = `${Math.min(clientX, window.innerWidth - 260)}px`;
    menu.style.top = `${Math.min(clientY, window.innerHeight - 320)}px`;
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
    menu.innerHTML = "";
    menu.appendChild(menuButton("Duplicate", () => {
      op({ op: "duplicate", id: nodeId });
      closeMenu();
    }));
    menu.appendChild(menuButton("Delete", () => {
      op({ op: "remove", id: nodeId });
      delete cards[nodeId];
      closeMenu();
    }));

    placeMenu(clientX, clientY);
  }

  /// Right-clicking the canvas offers the library, with a search box.
  function openAddMenu(clientX, clientY, world) {
    menu.innerHTML = "";

    const search = document.createElement("input");
    search.className = "menu-search";
    search.placeholder = "Search nodes";
    menu.appendChild(search);

    const list = document.createElement("div");
    list.className = "menu-list";
    menu.appendChild(list);

    const addAt = (kind) => {
      op({ op: "add", kind, x: Math.round(world.x), y: Math.round(world.y) });
      closeMenu();
    };

    const render = (query) => {
      list.innerHTML = "";
      list.classList.toggle("flat", !!query);

      if (query) {
        const matches = catalog
          .filter((kind) => categoryOf(kind) && kind.name.toLowerCase().includes(query))
          .sort((a, b) => a.name.localeCompare(b.name));

        for (const kind of matches) {
          list.appendChild(menuButton(kind.name, () => addAt(kind.id)));
        }

        return;
      }

      const groups = new Map();

      for (const kind of catalog) {
        const category = categoryOf(kind);
        if (!category) continue;
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

        for (const kind of items) {
          sub.appendChild(menuButton(kind.name, () => addAt(kind.id)));
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

    if (nodeEl) {
      openNodeMenu(Number(nodeEl.getAttribute("data-node")), event.clientX, event.clientY);
    } else {
      openAddMenu(event.clientX, event.clientY, toWorld(event));
    }
  });

  document.addEventListener("pointerdown", (event) => {
    if (!event.target.closest("#menu")) closeMenu();
  });

  // The selected node's last output, drawn along the bottom of its card.
  let scopeTimer = null;
  let scopeNode = null;
  let scopeCanvas = null;

  function stopScope() {
    if (scopeTimer !== null) {
      clearInterval(scopeTimer);
      scopeTimer = null;
    }

    scopeNode = null;

    if (scopeCanvas) {
      scopeCanvas.remove();
      scopeCanvas = null;
    }

    fetch("/scope?node=0").catch(() => {});
  }

  function startScope(id) {
    if (scopeNode === id) return;

    stopScope();
    scopeNode = id;
    scopeTimer = setInterval(() => pollScope(id), 120);
    pollScope(id);
  }

  // The control sources' histories: the engine keeps a ring for each of them,
  // and the page draws every one, whether it is selected or not.
  let scopesTimer = null;

  function startScopes() {
    if (scopesTimer !== null) return;

    scopesTimer = setInterval(pollScopes, 160);
    pollScopes();
  }

  async function pollScopes() {
    try {
      const data = await (await fetch("/scopes", { cache: "no-store" })).json();

      for (const scope of data.scopes || []) {
        const entry = cards[scope.node];

        if (!entry || !SOURCE_KINDS.includes(entry.kind ? entry.kind.id : "")) continue;

        entry.samples = scope.samples || [];
        entry.phase = typeof scope.phase === "number" ? scope.phase : null;
        drawSourceCard(entry);
      }
    } catch {
      // The next tick tries again.
    }
  }

  async function pollScope(id) {
    try {
      const response = await fetch(`/scope?node=${id}`, { cache: "no-store" });
      const data = await response.json();
      const samples = data.samples || [];
      drawScope(id, samples);

      // A product card's hero draws the same samples: the filter's spectrum
      // sits on its response curve, the way the product page's does.
      const entry = cards[id];

      if (entry && entry.context && entry.card && samples.length > 1) {
        const mono = samples.map(([left, right]) => (left + right) * 0.5);
        entry.context.scope.input = mono;
        entry.context.scope.output = mono;
        entry.card.repaintHero();
      }
    } catch {
      // The page closed or the node went away; the next tick tries again.
    }
  }

  function drawScope(id, samples) {
    const node = document.querySelector(`[data-node="${id}"]`);
    const card = node && node.querySelector(".mcard");

    if (!card) return;


    const hero = card.querySelector(".mcard-hero");
    const host = hero || card;

    if (!scopeCanvas || !scopeCanvas.isConnected || scopeCanvas.parentElement !== host) {
      scopeCanvas = document.createElement("canvas");
      scopeCanvas.className = "scope-canvas";
      scopeCanvas.width = 512;
      scopeCanvas.height = hero ? 150 : 88;
      host.appendChild(scopeCanvas);
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

    if (event.target.closest(".mcard .knob, .mcard button, .mcard select, .mcard input, .mcard .mcard-select, .mcard-hero")) {
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
      selected = null;
      selectedEdge = Number(wire.getAttribute("data-edge"));
      selectedMod = null;
      draw();
      return;
    }

    if (mod) {
      selected = null;
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

      selected = id;
      draw();
      startScope(id);

      // The scale handle scales the card, not moves it.
      if (event.target.closest(".scale-handle")) {
        drag = { kind: "scale", id, start: { ...scaleOf(id) }, from: point };
        return;
      }

      drag = { kind: "node", id, dx: point.x - node.x, dy: point.y - node.y };
      return;
    }

    selected = null;
    stopScope();
    drag = { kind: "pan", x: event.clientX, y: event.clientY, vx: view.x, vy: view.y };
    draw();
  });

  svg.addEventListener("pointermove", (event) => {
    if (!drag) return;

    if (drag.kind === "node") {
      const point = toWorld(event);
      const node = nodeOf(drag.id);
      node.x = Math.round(point.x - drag.dx);
      node.y = Math.round(point.y - drag.dy);
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
      const sx = Math.max(0.3, drag.start.x * (1 + (point.x - drag.from.x) / 160));
      const sy = Math.max(0.3, drag.start.y * (1 + (point.y - drag.from.y) / 120));
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
          if (port.kind !== signal) return;
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
        const knobRow = event.target.closest("[data-mod-target]");

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
      const point = toWorld(event);
      const node = nodeOf(drag.id);
      op({ op: "move", id: node.id, x: node.x, y: node.y });
    } else if (drag.kind === "knob") {
      const node = nodeOf(drag.id);
      op({ op: "param", id: node.id, key: drag.key, value: node.params[drag.key] });
    } else if (drag.kind === "scale") {
      // The scale is a page affordance, not part of the patch.
    } else if (drag.kind === "port" && drag.control && drag.output) {
      const row = event.target.closest("[data-mod-target]");
      const target = event.target.closest("[data-node]");

      let id = null;
      let key = null;

      if (row) {
        [id, key] = row.getAttribute("data-mod-target").split(":");
        id = Number(id);
      } else if (target && Number(target.getAttribute("data-node")) !== drag.id) {
        id = Number(target.getAttribute("data-node"));
        key = kindOf(nodeOf(id).kind).params[0]?.[0] ?? null;
      }

      if (id !== null && key) {
        const fallback = kindOf(nodeOf(id).kind).params.find(([name]) => name === key)?.[1] ?? 1;
        const depth = Math.max(0.1, Math.abs(fallback) * 0.5);
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

  svg.addEventListener("wheel", (event) => {
    event.preventDefault();
    const rect = svg.getBoundingClientRect();
    const point = { x: event.clientX - rect.left, y: event.clientY - rect.top };
    const before = { x: (point.x - view.x) / view.zoom, y: (point.y - view.y) / view.zoom };
    view.zoom = Math.min(2, Math.max(0.35, view.zoom * (event.deltaY < 0 ? 1.1 : 0.9)));
    view.x = point.x - before.x * view.zoom;
    view.y = point.y - before.y * view.zoom;
    draw();
  }, { passive: false });

  window.addEventListener("pointerup", () => {
    interacting = false;
  });

  window.addEventListener("keydown", (event) => {
    if (event.key === "Escape") {
      closeMenu();
      return;
    }

    if (event.key !== "Delete" && event.key !== "Backspace") return;

    if (selectedEdge !== null) {
      op({ op: "disconnect", index: selectedEdge });
      selectedEdge = null;
    } else if (selectedMod !== null) {
      op({ op: "unmodulate", index: selectedMod });
      selectedMod = null;
    } else if (selected !== null) {
      op({ op: "remove", id: selected });
      delete cards[selected];
      selected = null;
    }
  });

  // The patch saves as the text the host state stores, and opens from one.
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
  setInterval(async () => {
    if (busy()) return;
    const next = readPatch(await (await fetch("/patch")).json());
    if (JSON.stringify(next) === JSON.stringify(patch)) return;
    patch = next;
    draw();
  }, 1200);
})();
