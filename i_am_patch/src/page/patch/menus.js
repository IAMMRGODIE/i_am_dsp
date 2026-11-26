// Part eight: the right-click menus and the node library.
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
    // A note is nothing the graph plays: it explains the patch beside it.
    if (id === "text") return "Note";

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

  /// The knob a modulation pushes, if the card is showing it.
  function modulationKnob(node, control) {
    const host = document.querySelector('[data-node="' + node + '"]');
    if (!host) return null;

    return host.querySelector('.knob[data-drop-target="' + node + ":" + control + '"]');
  }

  /// The page of a card that shows one control, or -1 when no page does.
  function controlPage(node, control) {
    const kind = kindOf(node.kind);
    const pages = (kind && kind.pages) || [];

    if (!pages.length) return 0;

    // A section lists either the ids or the controls themselves, depending on how
    // the catalog was asked; both say the same thing.
    const section = ((kind && kind.sections) || []).find((item) =>
      (item.controls || []).some((entry) => entry === control || (entry && entry.id === control)),
    );

    if (!section) return -1;

    return pages.findIndex((page) => (page.sections || []).includes(section.name));
  }

  /// Opens the depth editor of one knob, the panel alt + right click shows there.
  ///
  /// The panel belongs to the knob, not to the wire: opening it from the wire is the
  /// same gesture, so it is dispatched as one rather than read off a copy of the code.
  function openDepthsFor(node, control) {
    let knob = modulationKnob(node.id, control);

    if (!knob) {
      // The knob lives on another page of a product card: turn to it first.
      const page = controlPage(node, control);

      if (page >= 0 && (pagesOf[node.id] ?? 0) !== page) {
        pagesOf[node.id] = page;
        delete cards[node.id];
        draw();
        knob = modulationKnob(node.id, control);
      }
    }

    if (!knob) return false;

    knob.dispatchEvent(new MouseEvent("contextmenu", {
      bubbles: true,
      cancelable: true,
      altKey: true,
    }));

    return true;
  }

  /// Right-clicking a modulation wire opens its depth editor, or takes the
  /// assignment off.
  function openModMenu(index, clientX, clientY) {
    const mod = patch.mods[index];
    menu.innerHTML = "";

    if (mod) {
      const node = nodeOf(mod.to[0]);
      const reachable =
        !!node && (!!modulationKnob(mod.to[0], mod.to[1]) || controlPage(node, mod.to[1]) >= 0);

      if (reachable) {
        menu.appendChild(menuButton("Edit", () => {
          closeMenu();

          // The card is rebuilt by the page turn above, so the knob is looked up
          // again inside there rather than held from before the click.
          openDepthsFor(node, mod.to[1]);
        }));
      }
    }

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
    // A category's flyout is lifted onto the body so the menu's own scroll
    // cannot cut it off, which means a press on one of its items is outside
    // the menu element: closing on that press would take the item away before
    // the click could land.
    if (!event.target.closest("#menu, .menu-sub")) closeMenu();
  });

