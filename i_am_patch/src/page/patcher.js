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
    } else if (kind === "midi") {
      // A mapping, not a value: the note (or velocity) goes in on the x axis
      // and the control value comes out on the y, like the product's page.
      svg.appendChild(rackNode("line", { x1: 8, y1: 60, x2: 292, y2: 60, stroke: "#333", "stroke-width": 1, "stroke-dasharray": "2 4" }));
      svg.appendChild(rackNode("line", { x1: 8, y1: 8, x2: 8, y2: 60, stroke: "#333", "stroke-width": 1, "stroke-dasharray": "2 4" }));
      svg.appendChild(rackNode("path", {
        d: "M 8 60 L 292 8",
        fill: "none",
        stroke: "#ff6f91",
        "stroke-width": 2,
        "stroke-linecap": "round",
        "vector-effect": "non-scaling-stroke",
      }));
    } else {
      IAmCard.sources.drawPreview(svg, kind, {});
    }
  }

  // The live overlay follows the rack's: a playhead for the LFO and the table,
  // a trace for the others. It sits in its own layer, so a poll repaints only
  // this and leaves the static drawing and the editor's handlers alone.
  function drawSourceLive(entry) {
    const layer = entry.live;

    if (!layer) return;

    layer.textContent = "";

    if (entry.kind.id === "midi") {
      // The mapping's live dot: input equals output on the default curve.
      const value = typeof entry.lastValue === "number" ? entry.lastValue : 0;
      const x = (8 + value * 284).toFixed(1);
      const y = (60 - value * 52).toFixed(1);

      layer.appendChild(rackNode("line", { x1: x, y1: 60, x2: x, y2: y, stroke: "#ff6f91", "stroke-opacity": ".35", "stroke-width": 1 }));
      layer.appendChild(rackNode("circle", { cx: x, cy: y, r: "3.5", fill: "#ffe1ea", stroke: "#ff6f91", "stroke-width": "1.5" }));
      return;
    }

    if (!entry.history || !entry.history.length) return;

    IAmCard.sources.drawLive(layer, entry.kind.id, {
      history: { patch: entry.history },
      phases: { patch: typeof entry.phase === "number" ? entry.phase : 0 },
      values: { patch: typeof entry.lastValue === "number" ? entry.lastValue : 0 },
    }, "patch", false);
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

  // A source card's history comes from the engine's tap for that node, and a
  // tap exists only after it is asked for once.
  const tapped = new Set();

  function tapSource(id) {
    if (tapped.has(id)) return;
    tapped.add(id);
    fetch(`/scope?node=${id}`, { cache: "no-store" }).catch(() => {});
  }

  function drawSourceCard(entry) {
    if (!entry || !entry.card || !entry.kind) return;

    tapSource(entry.node.id);

    const svg = entry.card.querySelector(".mcard-hero svg");
    if (!svg) return;

    const params = (entry.node && entry.node.params) || {};

    if (entry.kind.id === "lfo" && (!entry.node.points || entry.node.points.length < 2)) {
      entry.node.points = IAmCard.sources.preset(Math.round(params.shape ?? 0));
    }

    svg.textContent = "";
    drawSourceStatic(svg, entry.node, entry.kind.id);

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

    // Last, so the editor's repaint of the points does not erase the playhead.
    entry.live = el("g", { class: "source-live" });
    svg.appendChild(entry.live);
    drawSourceLive(entry);
  }

  // The rack's source cards put these choices in the header instead of a knob.
  const GENERIC_HEADER = {
    lfo: ["shape", "sync"],
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
  };

  function labelOf(key) {
    return key.replace(/_/g, " ").replace(/\b\w/g, (letter) => letter.toUpperCase());
  }


  // A product whose faceplate splits into sections draws one hero per section,
  // side by side, exactly as the product's own page does. The single-hero card
  // would show only the first of them.
  function sectionedCard(node, kind, w, h) {
    let entry = cards[node.id];
    const mods = modsOf(node.id);
    const key = JSON.stringify([
      mods.map((mod) => [mod.slot, mod.to[1], Math.round(mod.depth * 1000), mod.mode]),
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
      return entry.host;
    }

    const context = { rate: 48000, scope: { input: null, output: null }, spectra: {} };
    const values = node.params;
    const card = document.createElement("section");
    card.className = "mcard";
    card.style.setProperty("--accent", kind.accent || "#e07a2f");

    const head = document.createElement("header");
    head.className = "mcard-head";
    const name = document.createElement("span");
    name.className = "mcard-name";
    name.textContent = kind.title || kind.name;
    head.appendChild(name);

    const edit = (identifier, value) => {
      op({ op: "param", id: node.id, key: identifier, value }, false);
    };

    const byId = {};
    (kind.controls || []).forEach((control) => { byId[control.id] = control; });

    (kind.header || []).forEach((id) => {
      const control = byId[id];

      if (!control) return;

      const handle = control.choices && control.choices.length
        ? IAmCard.choiceHandle(control, values[id], (_, value) => edit(id, value))
        : control.kind === "bool"
          ? IAmCard.dotHandle(control, values[id], (_, value) => edit(id, value))
          : null;

      if (handle) head.appendChild(handle);
    });

    card.appendChild(head);

    const sections = document.createElement("div");
    sections.className = "mcard-sections";
    const heroes = [];
    const bodies = [];

    (kind.sections || []).forEach((section) => {
      const box = document.createElement("div");
      box.className = "mcard-section";

      if (section.hero) {
        const hero = document.createElement("div");
        hero.className = "mcard-hero";
        hero.innerHTML = IAmCard.hero(section.hero, values, context);
        box.appendChild(hero);
        heroes.push({ el: hero, name: section.hero });
      }

      const body = document.createElement("div");
      body.className = "mcard-body";

      (section.controls || []).forEach((control) => {
        if ((kind.header || []).indexOf(control.id) >= 0) return;

        const range = Math.max(1e-9, control.max - control.min);
        const assignments = mods
          .filter((mod) => mod.to[1] === control.id)
          .map((mod) => ({
            slot: mod.slot,
            label: sourceName(mod.from[0]),
            depth: mod.depth / range,
            mode: mod.mode === "base" ? "frombase" : "centered",
          }));
        const spec = IAmCard.modulate(control, {
          assignments,
          onDepth: (slot, next) => op({ op: "mod_depth", index: slot, depth: next * range }, false),
          onMode: (slots, next) => op({ op: "mod_mode", index: slots[0], mode: next === "center" ? "center" : "base" }, false),
          dropTarget: node.id + ":" + control.id,
        });

        body.appendChild(IAmCard.control(spec, values[control.id], edit));
      });

      box.appendChild(body);
      sections.appendChild(box);
      bodies.push(body);
    });

    card.appendChild(sections);
    card.repaintHero = () => {
      heroes.forEach((hero) => {
        hero.el.innerHTML = IAmCard.hero(hero.name, values, context);
      });
    };

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
    cards[node.id] = entry;
    return host;
  }

  function productCard(node, kind, w, h) {
    if (kind.sections && kind.sections.length > 1 && !(kind.pages && kind.pages.length)) {
      return sectionedCard(node, kind, w, h);
    }

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

      const specs = shown.map((control) => {
        const range = Math.max(1e-9, control.max - control.min);
        const assignments = modsOf(node.id)
          .filter((mod) => mod.to[1] === control.id)
          .map((mod) => ({
            slot: mod.slot,
            label: sourceName(mod.from[0]),
            depth: mod.depth / range,
            mode: mod.mode === "base" ? "frombase" : "centered",
          }));

        return IAmCard.modulate(control, {
          assignments,
          onDepth: (slot, next) => {
            op({ op: "mod_depth", index: slot, depth: next * range }, false);
          },
          onMode: (slots, next) => {
            op({ op: "mod_mode", index: slots[0], mode: next === "center" ? "center" : "base" }, false);
          },
          dropTarget: node.id + ":" + control.id,
        });
      });

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
      node.params,
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

      const rangeSpan = Math.max(1e-9, control.max - control.min);
      const assignments = modsOf(node.id)
        .filter((mod) => mod.to[1] === control.id)
        .map((mod) => ({
          slot: mod.slot,
          label: sourceName(mod.from[0]),
          depth: mod.depth / rangeSpan,
          mode: mod.mode === "base" ? "frombase" : "centered",
        }));

      return IAmCard.modulate(control, {
        assignments,
        onDepth: (slot, next) => {
          op({ op: "mod_depth", index: slot, depth: next * rangeSpan }, false);
        },
        onMode: (slots, next) => {
          op({ op: "mod_mode", index: slots[0], mode: next === "center" ? "center" : "base" }, false);
        },
        dropTarget: node.id + ":" + id,
      });
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
    "synth:additive": [300, 210],
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
    if (!selection.size) return;

    const ids = [...selection];
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
          await op({ op: "set_points", id, points: item.points }, false);
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

  async function op(body, redraw = true) {
    if (suspendSnapshot === 0) await snapshot(body);

    const response = await fetch("/op", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(body),
    });
    const result = await response.json();
    showStatus(result.error ? String(result.error) : "");
    patch = readPatch(await (await fetch("/patch")).json());
    if (redraw) draw();

    return result;
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
    {
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
  const CONTROL_KINDS = ["lfo", "envelope", "random", "table", "pitch", "midi"];

  // A right-click opens the node library: categories, then the nodes inside
  // them, alphabetically. No list takes space from the graph until it is asked.
  // Four groups: an effect processes audio, a generator makes it, a control
  // steers a parameter, and the host ports and the voice boundaries are the
  // wiring around them.
  function categoryOf(kind) {
    const id = kind.id;

    if (id.startsWith("mod:")) return "Effect";

    // The block-alignment node delays by one block and reports that as latency;
    // it is not the echo product, so it has its own entry.
    if (id === "delay") return "Effect";

    if (["mixer", "split:ms"].includes(id)) return "Effect";
    if (id.startsWith("gen:") || id.startsWith("synth:")) return "Generator";
    if (CONTROL_KINDS.includes(id)) return "Control";

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

    const node = nodeOf(id);

    if (node && CONTROL_KINDS.includes(node.kind)) return;

    scopeNode = id;
    scopeTimer = setInterval(() => pollScope(id), 120);
    pollScope(id);
  }

  // The control sources' histories: the engine keeps a ring for each of them,
  // and the page draws every one, whether it is selected or not.
  let scopesTimer = null;

  function startScopes() {
    if (scopesTimer !== null) return;

    scopesTimer = setInterval(pollScopes, 20);
    pollScopes();
  }

  async function pollScopes() {
    // A point drag owns the card until it ends: redrawing the hero here would
    // replace the handlers the drag is using.
    if (editingPoints) return;

    try {
      const data = await (await fetch("/scopes", { cache: "no-store" })).json();

      for (const scope of data.scopes || []) {
        const entry = cards[scope.node];

        if (!entry || !SOURCE_KINDS.includes(entry.kind ? entry.kind.id : "")) continue;

        entry.samples = scope.samples || [];
        entry.phase = typeof scope.phase === "number" ? scope.phase : null;

        // The LFO editor's repaint clears the hero while dragging, which takes
        // the live layer with it; put it back before drawing into it again.
        if (entry.live && !entry.live.isConnected) {
          const svg = entry.card.querySelector(".mcard-hero svg");
          if (svg) svg.appendChild(entry.live);
        }

        if (entry.samples !== entry.lastSamples) {
          entry.lastSamples = entry.samples;

          if (entry.samples.length) {
            entry.lastValue = entry.samples[entry.samples.length - 1][0];
            entry.history = entry.history || [];
            entry.ticks = (entry.ticks || 0) + 1;

            // One point per four polls: a 20 ms playhead, and a window of about
            // ten seconds, the way the rack's modulation cards read.
            if (entry.ticks % 4 === 1) {
              entry.history.push(entry.lastValue);

              if (entry.history.length > 120) entry.history.shift();
            }
          }
        }

        drawSourceLive(entry);
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
  setInterval(async () => {
    if (busy()) return;

    const status = await fetch("/status");

    if (status.ok) {
      const body = await status.json();
      serverError = body.error || "";
      paintStatus();
    }

    const next = readPatch(await (await fetch("/patch")).json());
    if (JSON.stringify(next) === JSON.stringify(patch)) return;
    patch = next;
    draw();
  }, 1200);
})();
