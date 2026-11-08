// The graph editor: an SVG canvas over the /patch JSON, with an op per gesture.
(() => {
  const svg = document.getElementById("graph");
  const palette = document.getElementById("palette");
  const params = document.getElementById("params");
  const NS = "http://www.w3.org/2000/svg";

  let catalog = [];
  let patch = { nodes: [], edges: [], mods: [] };
  let selected = null;
  let selectedEdge = null;
  let selectedMod = null;
  let view = { x: 210, y: 40, zoom: 1 };
  let drag = null;

  const kindOf = (id) => catalog.find((kind) => kind.id === id);
  const nodeOf = (id) => patch.nodes.find((node) => node.id === id);

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

  async function load() {
    catalog = await (await fetch("/catalog")).json();
    patch = await (await fetch("/patch")).json();
    draw();
  }

  async function op(body) {
    const response = await fetch("/op", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(body),
    });
    const result = await response.json();
    if (result.error) console.warn(result.error);
    patch = await (await fetch("/patch")).json();
    draw();
  }

  function nodeSize(node) {
    const kind = kindOf(node.kind);
    const ports = Math.max(kind ? kind.inputs.length : 0, kind ? kind.outputs.length : 0);
    return { w: 132, h: 28 + ports * 16 };
  }

  function portXY(node, port, output) {
    const { w, h } = nodeSize(node);
    const kind = kindOf(node.kind);
    const list = output ? kind.outputs : kind.inputs;
    const index = list.indexOf(port);
    return {
      x: node.x + (output ? w : 0),
      y: node.y + 26 + 16 * index,
    };
  }

  function draw() {
    while (svg.firstChild) svg.removeChild(svg.firstChild);
    const world = el("g", {
      transform: `translate(${view.x} ${view.y}) scale(${view.zoom})`,
    });
    svg.appendChild(world);

    // Wires first, so the nodes sit on top.
    for (const edge of patch.edges) {
      const from = nodeOf(edge.from[0]);
      const to = nodeOf(edge.to[0]);
      if (!from || !to) continue;
      const fromKind = kindOf(from.kind);
      const toKind = kindOf(to.kind);
      const a = portXY(from, fromKind.outputs[edge.from[1]], true);
      const b = portXY(to, toKind.inputs[edge.to[1]], false);
      const mid = (a.x + b.x) / 2;
      const d = `M ${a.x} ${a.y} C ${mid} ${a.y}, ${mid} ${b.y}, ${b.x} ${b.y}`;

      world.appendChild(el("path", { d, class: "wire-hit", "data-edge": patch.edges.indexOf(edge) }));
      world.appendChild(el("path", {
        d,
        class: "wire" + (selectedEdge === patch.edges.indexOf(edge) ? " selected" : ""),
      }));
    }

    for (const mod of patch.mods) {
      const from = nodeOf(mod.from[0]);
      const to = nodeOf(mod.to[0]);
      if (!from || !to) continue;
      const fromKind = kindOf(from.kind);
      const toKind = kindOf(to.kind);
      const a = portXY(from, fromKind.outputs[mod.from[1]], true);
      const b = { x: to.x + 60, y: to.y + nodeSize(to).h - 4 };
      const d = `M ${a.x} ${a.y} C ${a.x + 60} ${a.y}, ${b.x} ${b.y - 40}, ${b.x} ${b.y}`;

      world.appendChild(el("path", { d, class: "mod-hit", "data-mod": patch.mods.indexOf(mod) }));
      world.appendChild(el("path", {
        d,
        class: "mod" + (selectedMod === patch.mods.indexOf(mod) ? " selected" : ""),
      }));
    }

    for (const node of patch.nodes) {
      const kind = kindOf(node.kind);
      if (!kind) continue;
      const { w, h } = nodeSize(node);
      const group = el("g", { "data-node": node.id, transform: `translate(${node.x} ${node.y})` });
      group.appendChild(el("rect", {
        class: "node" + (selected === node.id ? " selected" : ""),
        width: w, height: h, rx: 5,
      }));
      const name = el("text", { class: "node-name", x: 8, y: 15 });
      name.textContent = kind.name;
      group.appendChild(name);

      kind.inputs.forEach((port, index) => {
        const circle = el("circle", {
          "data-port": "in", "data-index": index, cx: 0, cy: 26 + index * 16, r: 5,
          class: "port " + port.kind,
        });
        group.appendChild(circle);
        const label = el("text", { class: "port-label", x: 8, y: 29 + index * 16 });
        label.textContent = port.name;
        group.appendChild(label);
      });

      kind.outputs.forEach((port, index) => {
        const circle = el("circle", {
          "data-port": "out", "data-index": index, cx: w, cy: 26 + index * 16, r: 5,
          class: "port " + port.kind,
        });
        group.appendChild(circle);
        const label = el("text", { class: "port-label", x: w - 8, y: 29 + index * 16, "text-anchor": "end" });
        label.textContent = port.name;
        group.appendChild(label);
      });

      if (drag && drag.kind === "port" && drag.control && drag.output && node.id !== drag.id) {
        kind.params.forEach(([key, value], row) => {
          const y = h + 6 + row * 16;
          group.appendChild(el("rect", {
            class: "drop-row", x: -6, y, width: w + 12, height: 14, rx: 3,
          }));
          const text = el("text", { class: "drop-label", x: 6, y: y + 11 });
          text.textContent = key;
          group.appendChild(text);
          const id = `${node.id}:${key}:${value}`;
          const hit = el("rect", {
            x: -6, y, width: w + 12, height: 14, fill: "transparent", "data-mod-target": id,
          });
          group.appendChild(hit);
        });
      }

      world.appendChild(group);
    }

    drawPalette();
  }

  function drawPalette() {
    palette.innerHTML = "<h3>Add node</h3>";
    for (const kind of catalog) {
      const button = document.createElement("button");
      button.className = "kind";
      button.textContent = kind.name;
      button.onclick = () => {
        const rect = svg.getBoundingClientRect();
        const x = (rect.width / 2 - view.x) / view.zoom - 60;
        const y = (rect.height / 2 - view.y) / view.zoom - 20;
        op({ op: "add", kind: kind.id, x: Math.round(x), y: Math.round(y) });
      };
      palette.appendChild(button);
    }
  }

  function drawParams(node) {
    params.innerHTML = "";
    const kind = kindOf(node.kind);
    if (!kind) { params.style.display = "none"; return; }
    params.style.display = "block";
    const title = document.createElement("h3");
    title.textContent = kind.name;
    params.appendChild(title);

    for (const [key, value] of kind.params) {
      const label = document.createElement("label");
      label.textContent = key;
      params.appendChild(label);
      const input = document.createElement("input");
      input.type = "range";
      input.min = "0";
      input.max = String(Math.max(2, Math.abs(value) * 4));
      input.step = "0.001";
      input.value = String(node.params[key] ?? value);
      const readout = document.createElement("span");
      readout.className = "value";
      readout.textContent = input.value;
      input.oninput = () => { readout.textContent = input.value; };
      input.onchange = () => op({ op: "param", id: node.id, key, value: Number(input.value) });
      params.appendChild(input);
      params.appendChild(readout);
    }
  }

  svg.addEventListener("pointerdown", (event) => {
    const wire = event.target.closest("[data-edge]");
    const mod = event.target.closest("[data-mod]");

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
      selected = id;
      const node = nodeOf(id);
      draw();
      drawParams(node);
      const point = toWorld(event);
      drag = { kind: "node", id, dx: point.x - node.x, dy: point.y - node.y };
      return;
    }

    selected = null;
    params.style.display = "none";
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
    } else if (drag.kind === "pan") {
      // nothing to save
    } else if (drag.kind === "port" && drag.control && drag.output) {
      const target = event.target.closest("[data-mod-target]");

      if (target) {
        const [id, key, fallback] = target.getAttribute("data-mod-target").split(":");
        const depth = Math.max(0.1, Math.abs(Number(fallback)) * 0.5);
        op({ op: "modulate", from: [drag.id, drag.index], to: [Number(id), key], depth });
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

  window.addEventListener("keydown", (event) => {
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
      params.style.display = "none";
    }
  });

  load();
  setInterval(async () => {
    if (drag) return;
    patch = await (await fetch("/patch")).json();
    draw();
  }, 1200);
})();
