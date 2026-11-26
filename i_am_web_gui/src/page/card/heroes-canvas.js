// The canvas picture and its editor: the drawing the oscillator plays.
  function parseCanvas(shape) {
    if (!shape) return null;

    var tokens = String(shape).trim().split(/\s+/).map(Number);
    var at = 0;
    var next = function () { return tokens[at++]; };
    var pageCount = next();

    if (!(pageCount >= 1)) return null;

    var pages = [];

    for (var page = 0; page < pageCount; page++) {
      var start = [next(), next()];
      var count = next();

      if (!(count >= 1) || at + count * 6 > tokens.length) return null;

      var nodes = [];
      for (var index = 0; index < count; index++) {
        nodes.push([next(), next(), next(), next(), next(), next()]);
      }
      pages.push({ start: start, nodes: nodes });
    }

    return { pages: pages };
  }

  function serializeCanvas(model) {
    var parts = [String(model.pages.length)];
    model.pages.forEach(function (page) {
      parts.push(page.start[0] + " " + page.start[1] + " " + page.nodes.length);
      page.nodes.forEach(function (node) { parts.push(node.join(" ")); });
    });
    return parts.join(" ");
  }

  /// The canvas' [-1, 1] field inside a square view box: 160 units with a 16
  /// unit margin, so a circle stays a circle at any window shape.
  function canvasPoint(point) {
    return [80 + point[0] * 64, 80 - point[1] * 64];
  }

  // The last click on the canvas, for counting double clicks across rebuilds.
  var canvasClick = { time: 0, x: 0, y: 0 };

  function canvasPath(page) {
    var first = canvasPoint(page.start);
    var d = "M " + first[0].toFixed(1) + " " + first[1].toFixed(1);
    var previous = page.start;

    page.nodes.forEach(function (node) {
      // The six numbers are the two handle offsets and the end point: the
      // handles are measured from the ends they pull, so they follow them.
      var c1 = canvasPoint([previous[0] + node[0], previous[1] + node[1]]);
      var c2 = canvasPoint([node[4] + node[2], node[5] + node[3]]);
      var to = canvasPoint([node[4], node[5]]);

      d += " C " + c1[0].toFixed(1) + " " + c1[1].toFixed(1) + " " +
        c2[0].toFixed(1) + " " + c2[1].toFixed(1) + " " +
        to[0].toFixed(1) + " " + to[1].toFixed(1);

      previous = [node[4], node[5]];
    });

    return d;
  }

  /// The handles of one page and the lines that say which end each one pulls.
  ///
  /// Kept out of the hero so a drag can redraw this layer in place: the handles
  /// belong to the ends they pull, and dragging an end has to move them with it.
  function canvasHandles(page, index) {
    var handle = function (node, part, point) {
      var screen = canvasPoint(point);

      return '<circle class="canvas-handle" data-page="' + index + '" data-node="' + node +
        '" data-part="' + part + '" cx="' + screen[0].toFixed(1) + '" cy="' +
        screen[1].toFixed(1) + '" r="4" fill="var(--accent)"/>';
    };

    var guide = function (from, to) {
      var a = canvasPoint(from);
      var b = canvasPoint(to);

      return '<line x1="' + a[0].toFixed(1) + '" y1="' + a[1].toFixed(1) + '" x2="' +
        b[0].toFixed(1) + '" y2="' + b[1].toFixed(1) + '" stroke="#777" ' +
        'stroke-width="1" vector-effect="non-scaling-stroke"/>';
    };

    var markup = handle(-1, 0, page.start);
    var previous = page.start;

    page.nodes.forEach(function (node, nodeIndex) {
      var control1 = [previous[0] + node[0], previous[1] + node[1]];
      var control2 = [node[4] + node[2], node[5] + node[3]];
      var end = [node[4], node[5]];

      markup += handle(nodeIndex, 0, control1) + guide(previous, control1);
      markup += handle(nodeIndex, 1, control2) + guide(end, control2);
      markup += handle(nodeIndex, 2, end);
      previous = end;
    });

    return markup;
  }

  /// The drawn pages. The left and right channels are the curve's x and y, so a
  /// circle is a quadrature pair and a line is mono; the selected page is the one
  /// whose handles are shown.
  function canvasHero(values, context) {
    var state = context && context.canvas;
    var model = parseCanvas(state && state.shape);

    if (!model) {
      return heroSvg("") + '<div class="hero-empty">no canvas</div>';
    }

    // The page being drawn is the one the morph sits on, the same way the
    // spectrum page works: the knob is the position, the arrows are exact.
    var morph = Math.max(0, Math.min(1, valueOf(values, "oscillator.smooth_factor", 0)));
    var position = morph * (model.pages.length - 1);
    var from = Math.floor(position);
    var to = Math.min(from + 1, model.pages.length - 1);
    var blend = position - from;
    var selected = IAmPage.index(morph, model.pages.length);
    var paths = "";

    model.pages.forEach(function (page, index) {
      var active = index === selected;
      var blended = index === from || index === to;
      paths += '<path class="canvas-path" data-page="' + index + '" d="' + canvasPath(page) +
        '" fill="none" stroke="' +
        (active ? "var(--accent)" : blended ? "#9a9a9a" : "#555") + '" stroke-width="' +
        (active ? 2 : 1.25) + '" vector-effect="non-scaling-stroke" style="pointer-events:stroke"/>';

    });

    var pageLabel = blend > 0.001
      ? "PAGE " + (from + 1) + "\u2192" + (to + 1) + " / " + model.pages.length
      : "PAGE " + (selected + 1) + " / " + model.pages.length;

    return '<div class="canvas-hero"><div class="canvas-box">' +
        '<svg class="mcard-graph" viewBox="0 0 160 160" preserveAspectRatio="xMidYMid meet">' +
          '<rect x="16" y="16" width="128" height="128" fill="none" stroke="#2b2b2b"/>' +
          '<line x1="80" y1="16" x2="80" y2="144" stroke="#242424"/>' +
          '<line x1="16" y1="80" x2="144" y2="80" stroke="#242424"/>' +
          paths + '<g class="canvas-edit">' + canvasHandles(model.pages[selected], selected) +
          '</g>' +
        '</svg>' +
      '</div>' +
      '<div class="hero-label" style="top:4%">STEREO CANVAS \u00b7 ' + pageLabel + '</div>' +
      '<div class="hero-label" style="top:4%; left:auto; right:14px">DRAG · DBL-CLICK ADD · RIGHT-CLICK REMOVE</div>' +
      '<div class="hero-actions">' +
        '<button type="button" class="hero-action" data-action="page-prev">\u2039</button>' +
        '<button type="button" class="hero-action" data-action="page-next">\u203a</button>' +
        '<button type="button" class="hero-action" data-action="page-add">+ PAGE</button>' +
        '<button type="button" class="hero-action" data-action="page-remove">- PAGE</button>' +
      '</div>' +
      '<div class="hero-label" style="bottom:4%; left:auto; right:14px">MORPH ' +
        Math.round(morph * 100) + '%</div>' +
    '</div>';
  }

  /// The canvas editor: drag a handle, double-click to add a node to the selected
  /// page, right-click an end node to remove it, or use the page buttons.
  function wireCanvas(el, kind, values, state) {
    // The wrapper carries the drawn pages and the control setter, the same shape
    // the spectrum editor takes: the page follows the morph, the arrows move it.
    var canvasState = state && state.canvas;
    if (kind !== "canvas" || !canvasState) return;

    var model = parseCanvas(canvasState.shape);
    var svg = el.querySelector("svg");
    if (!model || !svg) return;

    var drag = null;

    var selected = function () {
      var morph = Math.max(0, Math.min(1, valueOf(values, "oscillator.smooth_factor", 0)));
      return IAmPage.index(morph, model.pages.length);
    };

    var setMorph = function (value) {
      value = Math.max(0, Math.min(1, value));

      if (state.set) state.set("oscillator.smooth_factor", value);
      else refresh();
    };

    var pointer = function (event) {
      var rect = svg.getBoundingClientRect();
      var x = ((event.clientX - rect.left) / Math.max(rect.width, 1)) * 160;
      var y = ((event.clientY - rect.top) / Math.max(rect.height, 1)) * 160;

      return [
        Math.max(-1, Math.min(1, (x - 80) / 64)),
        Math.max(-1, Math.min(1, (80 - y) / 64))
      ];
    };

    // Double clicks are counted here rather than left to the dblclick event: the
    // hero is rebuilt between clicks, which replaces the node under the pointer
    // and costs the browser its pair of clicks.
    var addNode = function (point) {
      var page = model.pages[selected()];
      var last = page.nodes.length
        ? [page.nodes[page.nodes.length - 1][4], page.nodes[page.nodes.length - 1][5]]
        : page.start;

      var pull = 0.5 * (point[0] - last[0]);

      page.nodes.push([
        pull, 0.0,
        -pull, 0.0,
        point[0], point[1]
      ]);

      canvasClick.time = 0;
      post();
      refresh();
    };

    var redraw = function (pageIndex) {
      var path = svg.querySelector('.canvas-path[data-page="' + pageIndex + '"]');
      if (path) path.setAttribute("d", canvasPath(model.pages[pageIndex]));

      // The handles belong to the ends they pull: moving an end moves them, and
      // the layer has to be rebuilt, not just the dragged dot.
      var editing = svg.querySelector(".canvas-edit");
      if (editing) editing.innerHTML = canvasHandles(model.pages[pageIndex], pageIndex);
    };

    var post = function () {
      var text = serializeCanvas(model);
      canvasState.shape = text;

      // The product page owns one canvas, so its POST has no node. A patch card
      // shares the page with many, and its state carries the route to use.
      if (state && state.post) {
        state.post(text);
        return;
      }

      fetch("/canvas", { method: "POST", body: text })
        .then(function (response) { return response.ok ? response.json() : null; })
        .then(function (result) {
          if (result && result.error) console.warn(result.error);
        })
        .catch(function () {});
    };

    var refresh = function () {
      el.innerHTML = canvasHero(values, { canvas: canvasState });
      wireCanvas(el, "canvas", values, state);
    };

    var select = function (page) {
      var last = model.pages.length - 1;

      setMorph(last > 0 ? page / last : 0);
    };

    // The page buttons are HTML and handled on pointerdown: the hero is rebuilt
    // on every repaint, so a click would lose its node before the release.
    el.onpointerdown = function (event) {
      var action = event.target.closest && event.target.closest("[data-action]");
      if (!action) return;

      var what = action.getAttribute("data-action");
      var last = model.pages.length - 1;
      var step = last > 0 ? 1 / last : 0;
      var page = selected();

      event.preventDefault();

      if (what === "page-prev") setMorph(Math.max(page - 1, 0) * step);
      else if (what === "page-next") setMorph(Math.min(page + 1, last) * step);
      else if (what === "page-add") {
        model.pages.push(JSON.parse(JSON.stringify(model.pages[model.pages.length - 1])));
        var added = model.pages.length - 1;
        post();
        setMorph(added / Math.max(added, 1));
      } else if (what === "page-remove" && model.pages.length > 1) {
        model.pages.splice(page, 1);
        var left = model.pages.length - 1;
        post();
        setMorph(left > 0 ? Math.min(page, left) / left : 0);
      }
    };

    svg.addEventListener("pointerdown", function (event) {
      var handle = event.target.closest && event.target.closest("[data-node]");
      var point = pointer(event);

      if (!handle) {
        var now = performance.now();
        var again = now - canvasClick.time < 400 &&
          Math.abs(point[0] - canvasClick.x) < 0.12 &&
          Math.abs(point[1] - canvasClick.y) < 0.12;

        canvasClick.time = now;
        canvasClick.x = point[0];
        canvasClick.y = point[1];

        if (again) {
          addNode(point);
          return;
        }
      }

      if (!handle) {
        var path = event.target.closest && event.target.closest("[data-page]");
        if (path) select(Number(path.getAttribute("data-page")));
        return;
      }

      drag = {
        page: Number(handle.getAttribute("data-page")),
        node: Number(handle.getAttribute("data-node")),
        part: Number(handle.getAttribute("data-part"))
      };

      el.dataset.busy = "1";
      event.preventDefault();

      if (svg.setPointerCapture) {
        try { svg.setPointerCapture(event.pointerId); } catch (error) { }
      }
    });

    svg.addEventListener("pointermove", function (event) {
      if (!drag) return;

      var point = pointer(event);
      var page = model.pages[drag.page];

      if (drag.node < 0) {
        page.start = point;
      } else {
        var node = page.nodes[drag.node];
        var previous = drag.node > 0
          ? [page.nodes[drag.node - 1][4], page.nodes[drag.node - 1][5]]
          : page.start;

        if (drag.part === 0) {
          node[0] = point[0] - previous[0];
          node[1] = point[1] - previous[1];
        } else if (drag.part === 1) {
          node[2] = point[0] - node[4];
          node[3] = point[1] - node[5];
        } else {
          node[4] = point[0];
          node[5] = point[1];
        }
      }

      redraw(drag.page);
    });

    svg.addEventListener("pointerup", function () {
      if (!drag) return;
      drag = null;
      delete el.dataset.busy;
      post();
    });

    svg.addEventListener("pointercancel", function () {
      if (!drag) return;
      drag = null;
      delete el.dataset.busy;
      post();
    });

    svg.addEventListener("contextmenu", function (event) {
      var handle = event.target.closest && event.target.closest("[data-node]");
      if (!handle) return;

      event.preventDefault();

      var pageIndex = Number(handle.getAttribute("data-page"));
      var node = Number(handle.getAttribute("data-node"));
      var part = Number(handle.getAttribute("data-part"));

      if (part !== 2 || node < 0 || model.pages[pageIndex].nodes.length <= 1) return;

      var page = model.pages[pageIndex];
      var removed = page.nodes[node];
      var previous = node > 0
        ? [page.nodes[node - 1][4], page.nodes[node - 1][5]]
        : page.start;
      var next = page.nodes[node + 1];

      if (next) {
        // The next handle was measured from the removed point; move the offset
        // so the curve stays where it was drawn.
        next[0] += removed[4] - previous[0];
        next[1] += removed[5] - previous[1];
      }

      page.nodes.splice(node, 1);
      post();
      refresh();
    });
  }

  /// The voice's unison: every copy as the same wave played at its detuned pitch,
  /// so the traces fan apart across the three periods drawn here, and the wave the
  /// copies add up to on the right. Random Phase is not drawn on the copies
  /// themselves - it would just look like noise - but as the window a copy may
  /// start in, and only while its knob is being held. The window is one band that
  /// runs on into the later periods instead of wrapping back to the left edge.
