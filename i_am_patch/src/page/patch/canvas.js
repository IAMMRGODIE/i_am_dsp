// Part seven: the canvas, its sizes, the drawing, undo and the queue that orders the writes.
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
  // Whether the bar is saying where something went rather than what failed.
  let statusNote = false;

  function mutationKey(body) {
    if (body.op === "param") return "param:" + body.id + ":" + body.key;
    if (body.op === "mod_depth") return "depth:" + body.index;
    // The gesture used to send every slot of the knob and the op sends the one slot
    // it changes, so the key takes either: a number here was calling join and
    // throwing, which killed the op before it reached the server.
    if (body.op === "mod_mode") {
      return "mode:" + (Array.isArray(body.index) ? body.index.join(",") : body.index);
    }
    if (body.op === "move") return "move:" + body.id;
    // A note typed in one sitting is one undo step, not one per pause.
    if (body.op === "text") return "text:" + body.id;
    return "";
  }

  function paintStatus() {
    const bar = document.getElementById("status");
    const text = serverError || opError;
    bar.textContent = text;
    // A refusal and a note share the bar; only one of them is red.
    bar.classList.toggle("note", statusNote && !serverError);
    bar.hidden = !text;
  }

  function showStatus(text, note) {
    opError = text || "";
    statusNote = !!text && !!note;
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
          // A note travels with its node: a copied note pasted without its text is
          // an empty card, which is not what was copied.
          text: node.text || "",
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

        if (item.text) {
          await op({ op: "text", id, text: item.text }, false);
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

  /// Duplicates the selection a little off the original, as one undo step.
  ///
  /// It is what Copy then Paste would make, kept inside the page so the clipboard the
  /// user filled stays where it was. Copying the whole selection is what keeps a
  /// duplicated voice group whole: its wires and the modulations inside it come along,
  /// which the node menu's Duplicate (one node, its parameters) cannot do.
  async function duplicateSelection() {
    const ids = [...selection];

    if (!ids.length) return;

    const kept = clipboard;

    copyNodes(ids);

    try {
      await pasteClipboard();
    } finally {
      clipboard = kept;
    }
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
      // The undo step is a courtesy; a bug in it must not swallow the edit itself,
      // which is how a mode change used to reach the card and never the graph.
      if (suspendSnapshot === 0) {
        try {
          await snapshot(body);
        } catch (error) {
          console.warn("the undo step was skipped:", error);
        }
      }

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

    for (const id of envelopeCards.keys()) {
      if (!patch.nodes.some((node) => node.id === id)) envelopeCards.delete(id);
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

