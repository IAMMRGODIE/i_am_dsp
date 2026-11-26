// The history picture and the editor that pages through it.
  function historySelected(values, pages) {
    return IAmPage.index(valueOf(values, "morph", 0), pages.length);
  }

  function historyPath(points) {
    if (!points || !points.length) return '';

    var sorted = points.slice().sort(function (a, b) { return a.time - b.time; });
    var path = "";
    var samples = 36;

    for (var index = 0; index < sorted.length - 1; index++) {
      var left = sorted[index];
      var right = sorted[index + 1];

      for (var step = 0; step <= samples; step++) {
        var unit = step / samples;
        var time = left.time + (right.time - left.time) * unit;
        var value = left.value + (right.value - left.value) *
          IAmCard.sources.bend(unit, left.bend);
        var x = 8 + Math.max(0, Math.min(1, time)) * 284;
        var y = 60 - Math.max(0, Math.min(1, value)) * 52;
        path += (path ? " L " : "M ") + x.toFixed(1) + " " + y.toFixed(1);
      }
    }

    return path;
  }

  function historyHero(values, context) {
    var history = context && context.history;
    var pages = (history && history.pages) || [{ points: [] }];
    var selected = historySelected(values, pages);
    var points = (pages[selected] && pages[selected].points) || pages[selected] || [];
    var path = historyPath(points);
    var phase = history && typeof history.phase === "number" ? history.phase : 0;

    return '<svg class="mcard-graph" viewBox="0 0 300 64" preserveAspectRatio="xMidYMid meet">' +
      '<line x1="8" y1="60" x2="292" y2="60" stroke="#333" stroke-width="1" stroke-dasharray="2 4"/>' +
      (path ? '<path d="' + path + '" fill="none" stroke="var(--accent)" stroke-width="2" vector-effect="non-scaling-stroke"/>' : '') +
      '</svg>' +
      '<div class="hero-actions">' +
        '<button type="button" class="hero-action" data-action="history-prev">‹</button>' +
        '<button type="button" class="hero-action" data-action="history-next">›</button>' +
        '<button type="button" class="hero-action" data-action="history-add">+ PAGE</button>' +
        '<button type="button" class="hero-action" data-action="history-remove">- PAGE</button>' +
      '</div>' +
      '<div class="history-playhead" style="left:' + (phase * 100).toFixed(2) + '%"></div>' +
      '<div class="hero-label" style="left:2%;top:2%">HISTORY · PAGE ' +
        (selected + 1) + ' / ' + pages.length + '</div>';
  }

  /// The page editor: editLfo moves the points of one page; the page buttons
  /// move the morph and add or remove whole pages.
  function wireHistory(el, kind, values, state) {
    var history = state && state.history;

    if (kind !== "history" || !history || !history.pages || !history.pages.length) return;

    var pages = history.pages;
    var selected = historySelected(values, pages);
    var svg = el.querySelector("svg");

    if (!svg) return;

    var page = pages[selected];
    var holder = { points: page.points || page };
    var accent = (getComputedStyle(el).getPropertyValue("--accent") || "#4c8dff").trim();
    var paint = function (points) {
      if (page.points) page.points = points;
      else pages[selected] = points;
      IAmCard.sources.drawPreview(svg, "lfo", { points: points, color: accent, segments: 36 }, null);
    };

    var post = function () {
      if (state.post) state.post(JSON.stringify(pages));
    };

    // The action row can sit under the SVG inside a scaled foreignObject, so a
    // press that misses the button is matched to its rectangle here.
    svg.addEventListener("pointerdown", function (event) {
      var buttons = el.querySelectorAll(".hero-action");

      for (var index = 0; index < buttons.length; index++) {
        var rect = buttons[index].getBoundingClientRect();

        if (event.clientX >= rect.left && event.clientX <= rect.right &&
            event.clientY >= rect.top && event.clientY <= rect.bottom) {
          buttons[index].dispatchEvent(new PointerEvent("pointerdown", {
            bubbles: true,
            cancelable: true,
            clientX: event.clientX,
            clientY: event.clientY,
            pointerId: event.pointerId,
          }));
          event.preventDefault();
          event.stopPropagation();
          return;
        }
      }
    });

    IAmCard.sources.editLfo(svg, holder, {
      paint: paint,
      commit: post,
      dots: true,
      onStart: function () { el.dataset.busy = "1"; },
      onEnd: function () { delete el.dataset.busy; },
    });

    el.onpointerdown = function (event) {
      var button = event.target.closest && event.target.closest("[data-action]");

      if (!button) return;

      var action = button.getAttribute("data-action");
      var last = pages.length - 1;

      if (action === "history-prev") {
        state.set("morph", last > 0 ? Math.max(selected - 1, 0) / last : 0);
      } else if (action === "history-next") {
        state.set("morph", last > 0 ? Math.min(selected + 1, last) / last : 0);
      } else if (action === "history-add") {
        var copy = { points: (page.points || page).map(function (point) { return Object.assign({}, point); }) };
        pages.splice(selected + 1, 0, copy);
        post();
        state.set("morph", (selected + 1) / Math.max(pages.length - 1, 1));
      } else if (action === "history-remove" && pages.length > 1) {
        pages.splice(selected, 1);
        post();
        state.set("morph", Math.min(selected, pages.length - 1) / Math.max(pages.length - 1, 1));
      }

      event.preventDefault();
    };

    paint(holder.points);
  }

  /// The frame a Filter Table uses as its convolution kernel.
