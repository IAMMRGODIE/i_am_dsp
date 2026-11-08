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
  let view = { x: 230, y: 30, zoom: 1 };
  let drag = null;
  // Each card scales by itself: x and y are independent, so a wide filter and
  // a narrow one can sit side by side at the sizes they need.
  const scales = {};

  const cards = {};

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

  function productCard(node, kind, w, h) {
    let entry = cards[node.id];
    const mods = modsOf(node.id);
    const key = JSON.stringify(
      mods.map((mod) => [mod.slot, mod.to[1], Math.round(mod.depth * 1000), mod.mode]),
    );

    if (entry && entry.key !== key) {
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
      const specs = (kind.controls && kind.controls.length ? kind.controls : [
        { id: "cutoff", label: "Cutoff", kind: "float", min: 20, max: 20000, default: 1000, unit: "hz", log: true },
        { id: "q", label: "Resonance", kind: "float", min: 0.01, max: 10, default: 0.707 },
      ]).map((control) => {
        const range = Math.max(1e-9, control.max - control.min);
        const assignments = mods
          .filter((mod) => mod.to[1] === control.id)
          .map((mod) => ({
            slot: mod.slot,
            label: sourceName(mod.from[0]),
            depth: mod.depth / range,
            mode: mod.mode === "base" ? "frombase" : "centered",
          }));

        return {
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
          modulations: assignments,
          onModulation: (slot, next) => {
            op({ op: "mod_depth", index: slot, depth: next * range }, false);
          },
          onModeToggle: (slots, next) => {
            op({ op: "mod_mode", index: slots[0], mode: next === "center" ? "center" : "base" }, false);
          },
        };
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
        hero: kind.hero || "spectrum",
        mode: kind.mode || null,
        header: kind.header || [],
        visible: kind.visible || null,
        switches: kind.switches || [],
        controls: specs,
        values: node.params,
        context: { rate: 48000 },
        onEdit: (identifier, value) => {
          // The card already shows the value; rebuilding the graph here would
          // take the focus away from the control being dragged.
          op({ op: "param", id: node.id, key: identifier, value }, false);
        },
      });
      scale.appendChild(card);
      host.appendChild(scale);
      entry = { host, key };
      cards[node.id] = entry;
      entry.host.style.setProperty(
        "--k",
        String(Math.max(0.2, Math.min(w / baseWidth(node), h / baseHeight(node)))),
      );
      entry.host.classList.toggle("compact", w < 430);
    }

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

  function baseWidth(node) {
    const face = faceOf(node);
    return face ? face[0] : BASE_W;
  }

  function baseHeight(node) {
    const face = faceOf(node);
    return face ? face[1] : BASE_H;
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

  async function load() {
    catalog = await (await fetch("/catalog")).json();
    patch = await (await fetch("/patch")).json();
    draw();
  }

  // A control inside a card owns the focus while it is being used; the poll
  // must not rebuild the page under it, or the select closes and the knob
  // stops mid-drag.
  let interacting = false;

  function busy() {
    if (drag || interacting) return true;
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
    patch = await (await fetch("/patch")).json();
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

  function draw() {
    while (svg.firstChild) svg.removeChild(svg.firstChild);

    const world = el("g", {
      transform: `translate(${view.x} ${view.y}) scale(${view.zoom})`,
    });
    svg.appendChild(world);

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

      if (window.IAmCard && kind.controls && kind.controls.length) {
        const foreign = el("foreignObject", { x: 0, y: 0, width: w, height: h });
        const host = productCard(node, kind, w, h);
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
  function categoryOf(kind) {
    const id = kind.id;

    if (id.startsWith("mod:")) {
      const name = id.slice(4);
      if (/compressor|limiter|gate|expander|ott|transient|tape_stop/.test(name)) return "Dynamics";
      if (/reverb|schroeder|convolver|ir_reverb|comb|allpass/.test(name)) return "Reverb & Space";
      if (/chorus|flanger|phaser|disperser|unstablizer/.test(name)) return "Modulation";
      if (/filter|eq|formant|resonator/.test(name)) return "Filters";
      if (/distortion|crusher/.test(name)) return "Distortion";
      if (/pitch|autotune|spectral|shifter|vocoder|stft/.test(name)) return "Pitch & Voice";
      if (/width|balance|particle|router/.test(name)) return "Stereo";
      if (/script/.test(name)) return "Script";
      return "Other";
    }

    if (id.startsWith("input:") || id === "out:audio") return "Input & Output";
    if (id.startsWith("voice:")) return "Voice";
    if (id.startsWith("gen:")) return "Generators";

    return "Utilities";
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
          .filter((kind) => kind.name.toLowerCase().includes(query))
          .sort((a, b) => a.name.localeCompare(b.name));

        for (const kind of matches) {
          list.appendChild(menuButton(kind.name, () => addAt(kind.id)));
        }

        return;
      }

      const groups = new Map();

      for (const kind of catalog) {
        const category = categoryOf(kind);
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

  svg.addEventListener("pointerdown", (event) => {
    if (event.target.closest(".card-host")) {
      interacting = true;
    }

    const knobEl = event.target.closest("[data-knob]");
    const wire = event.target.closest("[data-edge]");
    const mod = event.target.closest("[data-mod]");

    if (event.target.closest(".mcard .knob, .mcard button, .mcard select, .mcard input, .mcard .mcard-select")) {
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
        line: null,
      };
      draw();
      return;
    }

    if (nodeEl) {
      const id = Number(nodeEl.getAttribute("data-node"));
      const point = toWorld(event);
      const node = nodeOf(id);

      selected = id;
      draw();

      // The scale handle scales the card, not moves it.
      if (event.target.closest(".scale-handle")) {
        drag = { kind: "scale", id, start: { ...scaleOf(id) }, from: point };
        return;
      }

      drag = { kind: "node", id, dx: point.x - node.x, dy: point.y - node.y };
      return;
    }

    selected = null;
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
      const port = event.target.closest("[data-port]");

      if (port) {
        const nodeEl = event.target.closest("[data-node]");
        const id = Number(nodeEl.getAttribute("data-node"));
        const index = Number(port.getAttribute("data-index"));
        const output = port.getAttribute("data-port") === "out";

        if (output !== drag.output) {
          const from = drag.output ? [drag.id, drag.index] : [id, index];
          const to = drag.output ? [id, index] : [drag.id, drag.index];
          op({ op: "connect", from, to });
        }
      }
    }

    drag = null;
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
      selected = null;
    }
  });

  load();
  setInterval(async () => {
    if (busy()) return;
    const next = await (await fetch("/patch")).json();
    if (JSON.stringify(next) === JSON.stringify(patch)) return;
    patch = next;
    draw();
  }, 1200);
})();
