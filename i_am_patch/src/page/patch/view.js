// Part eleven: zoom, fit and the keyboard.
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

  // The capture phase, because a hero or a knob answers a release of its own and
  // stops it there: the flag would stay up, the poll would skip every redraw after
  // it, and the page would look frozen until something else cleared it.
  window.addEventListener("pointerup", () => {
    interacting = false;
    editingPoints = false;
  }, true);

  window.addEventListener("pointercancel", () => {
    interacting = false;
    editingPoints = false;
  }, true);

  /// Whether a key was typed into a field rather than at the canvas.
  ///
  /// A card's value editor and the node library's search are inputs: Backspace there
  /// edits the number, and it must not also take the selected node off the graph.
  function typingIn(event) {
    return !!(event.target && event.target.closest && event.target.closest("input, textarea, select"));
  }

  window.addEventListener("keydown", (event) => {
    if (event.key === "Escape") {
      closeMenu();
      return;
    }

    if ((event.ctrlKey || event.metaKey) && !event.altKey) {
      const key = event.key.toLowerCase();

      // A field owns its own editing keys: the search in the node library and the
      // script editor must not duplicate a node, or undo the graph, under the caret.
      if (typingIn(event)) return;

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

      // The browser would save the page itself, which is not a patch: Ctrl+S asks
      // the same question the button does.
      if (key === "s") {
        openSaveMenu(document.getElementById("save"));
        event.preventDefault();
        return;
      }

      if (key === "d") {
        duplicateSelection();
        event.preventDefault();
        return;
      }
    }

    if (event.key === " " && !typingIn(event)) {
      spaceHeld = true;
      event.preventDefault();
      return;
    }

    if (event.key !== "Delete" && event.key !== "Backspace") return;

    // The value editor is a text field: Backspace belongs to the number in it.
    if (typingIn(event)) return;

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

