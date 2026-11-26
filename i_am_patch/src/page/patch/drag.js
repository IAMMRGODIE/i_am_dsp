// Part ten: the pointer, dragging cards, knobs, wires and ports.
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

