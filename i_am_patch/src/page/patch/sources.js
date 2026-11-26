// Part two: the source cards' pictures.
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


