// The spectrum pages and the morph picture, with the editor that picks a page.
  function spectrumPagePoints(frame) {
    var points = 256;
    var partials = Math.min(frame.levels.length, 512);
    var values = [];
    var peak = 1e-6;

    for (var p = 0; p < points; p++) {
      var t = p / points;
      var sum = 0;

      for (var n = 0; n < partials; n++) {
        var level = frame.levels[n] || 0;
        if (level <= 0) continue;
        sum += level * Math.sin(2 * Math.PI * ((n + 1) * t + (frame.phases[n] || 0)));
      }

      values.push(sum);
      if (Math.abs(sum) > peak) peak = Math.abs(sum);
    }

    for (var q = 0; q < points; q++) values[q] /= peak;

    return values;
  }

  function spectrumLevelsPath(frame) {
    var count = frame.levels.length;
    var parts = [];

    for (var i = 0; i < count; i++) {
      var x = 20 + (count === 1 ? 0.5 : i / (count - 1)) * 480;
      var height = Math.max(0, Math.min(1, frame.levels[i] || 0)) * 40;

      parts.push("M " + x.toFixed(1) + " " + (116 - height).toFixed(1) + " L " +
        x.toFixed(1) + " 116");
    }

    return parts.join(" ");
  }

  function spectrumPhasesPath(frame) {
    var count = frame.phases.length;
    var parts = [];

    for (var i = 0; i < count; i++) {
      var x = 20 + (count === 1 ? 0.5 : i / (count - 1)) * 480;
      var height = Math.max(0, Math.min(1, frame.phases[i] || 0)) * 16;

      parts.push("M " + x.toFixed(1) + " " + (136 - height).toFixed(1) + " L " +
        x.toFixed(1) + " 136");
    }

    return parts.join(" ");
  }

  /// The drawn spectrum: the wave Morph makes right now, the levels of the page
  /// it sits on, and that page's start phases. Pages are what Morph fades
  /// between, so the page being edited follows the knob.
  function morphSpectrumHero(values, context) {
    var state = context && context.spectrum;
    var frames = state && state.frames;

    if (!frames || !frames.length) {
      return heroSvg("") + '<div class="hero-empty">no spectrum</div>';
    }

    var morph = clamp(valueOf(values, "oscillator.morph", 0), 0, 1);
    var position = morph * (frames.length - 1);
    var from = Math.floor(position);
    var to = Math.min(from + 1, frames.length - 1);
    var blend = position - from;
    var selected = IAmPage.index(position / Math.max(frames.length - 1, 1), frames.length);
    var frame = frames[selected];

    var cache = spectrumWaveCache;

    if (cache.generation !== state.generation) {
      cache.generation = state.generation;
      cache.pages = {};
    }

    var pointsOf = function (page) {
      if (!cache.pages[page]) cache.pages[page] = spectrumPagePoints(frames[page]);
      return cache.pages[page];
    };

    var gains = state.gains || [];
    var gainOf = function (page) { return gains[page] || 1; };
    var first = pointsOf(from);
    var second = pointsOf(to);
    var path = "";

    for (var q = 0; q <= 256; q++) {
      var index = q % 256;
      // The points are the drawn wave normalised to its own peak, and each page
      // plays at full scale: the picture is that normalised wave, unsealed by
      // the page's gain, which only says how far the table was turned.
      var value = first[index] * (1 - blend) + second[index] * blend;

      path += (q ? " L " : "M ") + (20 + q / 256 * 480).toFixed(1) + " " +
        (39 - value * 27).toFixed(1);
    }

    var page = blend > 0.001
      ? "PAGE " + (from + 1) + "\u2192" + (to + 1) + " / " + frames.length
      : "PAGE " + (selected + 1) + " / " + frames.length;

    return heroSvg(
      '<line x1="20" y1="39" x2="500" y2="39" stroke="#1f1f1f"/>' +
      '<path d="' + path + '" fill="none" stroke="var(--accent)" stroke-width="1.8" ' +
        'vector-effect="non-scaling-stroke"/>' +
      '<line x1="20" y1="76" x2="500" y2="76" stroke="#242424"/>' +
      '<path class="spectrum-levels" d="' + spectrumLevelsPath(frame) + '" stroke="var(--accent)" ' +
        'stroke-opacity=".75" stroke-width="1" vector-effect="non-scaling-stroke"/>' +
      '<line x1="20" y1="116" x2="500" y2="116" stroke="#2c2c2c"/>' +
      '<path class="spectrum-phases" d="' + spectrumPhasesPath(frame) + '" stroke="#8a8a8a" ' +
        'stroke-width="1" vector-effect="non-scaling-stroke"/>' +
      '<line x1="20" y1="136" x2="500" y2="136" stroke="#2c2c2c"/>'
    ) +
      '<div class="hero-label" style="top:2%">SPECTRUM \u00b7 ' + frame.levels.length +
        ' PARTIALS' +
        (Math.abs(gainOf(selected) - 1) > 0.005
          ? ' \u00b7 \u00d7' + gainOf(selected).toFixed(2)
          : '') + '</div>' +
      '<div class="hero-label" style="top:2%; left:auto; right:10px">' + page +
        ' \u00b7 MORPH ' + Math.round(morph * 100) + '%</div>' +
      '<div class="hero-actions">' +
        '<button type="button" class="hero-action" data-action="spectrum-prev">\u2039</button>' +
        '<button type="button" class="hero-action" data-action="spectrum-next">\u203a</button>' +
        '<button type="button" class="hero-action" data-action="spectrum-add">+ PAGE</button>' +
        '<button type="button" class="hero-action" data-action="spectrum-remove">- PAGE</button>' +
      '</div>' +
      '<div class="hero-label" style="bottom:2%; left:auto; right:10px">' +
        'DRAG LEVELS \u00b7 SHIFT DRAG PHASES</div>';
  }

  /// The spectrum editor: drag across the levels or the phase strip, add and
  /// remove pages, and post the pages back to the product.
  function wireSpectrum(el, kind, values, state) {
    // The wrapper carries the pages and the control setter: the editor edits one
    // and moves the morph through the other.
    var spectrum = state && state.spectrum;

    if (kind !== "spectrum" || !spectrum || !spectrum.frames || !spectrum.frames.length) return;

    var svg = el.querySelector("svg");
    if (!svg) return;

    var frames = spectrum.frames;
    var morph = clamp(valueOf(values, "oscillator.morph", 0), 0, 1);
    var selected = IAmPage.index(morph, frames.length);
    var frame = frames[selected];
    var drag = null;

    var setMorph = function (value) {
      value = Math.max(0, Math.min(1, value));

      if (state.set) state.set("oscillator.morph", value);
      else {
        spectrum.generation = (spectrum.generation || 0) + 1;
        el.innerHTML = morphSpectrumHero(values, { spectrum: spectrum });
        wireSpectrum(el, "spectrum", values, state);
      }
    };

    var point = function (event) {
      var rect = svg.getBoundingClientRect();
      return {
        x: ((event.clientX - rect.left) / Math.max(rect.width, 1)) * 520,
        y: ((event.clientY - rect.top) / Math.max(rect.height, 1)) * 160
      };
    };

    var paint = function () {
      var levels = svg.querySelector(".spectrum-levels");
      var phases = svg.querySelector(".spectrum-phases");

      if (levels) levels.setAttribute("d", spectrumLevelsPath(frame));
      if (phases) phases.setAttribute("d", spectrumPhasesPath(frame));
    };

    var redraw = function () {
      spectrum.generation = (spectrum.generation || 0) + 1;
      el.innerHTML = morphSpectrumHero(values, { spectrum: spectrum });
      wireSpectrum(el, "spectrum", values, state);
    };

    var post = function () {
      var body = JSON.stringify(spectrum.frames);

      // The product page owns one spectrum, so its POST has no node. A patch
      // card shares the page with many, and its state carries the route to use.
      if (state && state.post) {
        state.post(body);
        return;
      }

      fetch("/spectrum", { method: "POST", body })
        .then(function (response) { return response.ok ? response.json() : null; })
        .then(function (result) { if (result && result.error) console.warn(result.error); })
        .catch(function () {});
    };

    /// Paints the bars the pointer passed over since the last event, not just
    /// the one under it: a fast drag draws a line instead of a picket fence.
    var edit = function (event) {
      // The page's poll replaces the pages while this hero stays wired, so the
      // arrays are read at gesture time: an edit must not post the copy the
      // hero was wired with back over newer pages.
      frames = spectrum.frames;
      selected = Math.min(selected, frames.length - 1);
      frame = frames[selected] || frame;

      var at = point(event);
      var count = frame.levels.length;
      var index = Math.max(0, Math.min(count - 1,
        Math.round((at.x - 20) / 480 * (count - 1))));
      var phase = event.shiftKey || at.y >= 120;

      if (!phase && at.y < 84) return;

      var value = phase
        ? Math.max(0, Math.min(1, (136 - at.y) / 16))
        : Math.max(0, Math.min(1, (116 - at.y) / 40));

      if (!drag || drag.phase !== phase) {
        drag = { phase: phase, index: index, value: value };
      }

      var steps = Math.abs(index - drag.index);
      var direction = index >= drag.index ? 1 : -1;

      for (var step = 0; step <= steps; step++) {
        var partial = drag.index + direction * step;
        var t = steps === 0 ? 1 : step / steps;
        var filled = drag.value + (value - drag.value) * t;

        if (phase) frame.phases[partial] = filled;
        else frame.levels[partial] = filled;
      }

      drag.index = index;
      drag.value = value;
      paint();
    };

    // One handler, not one per repaint, and on pointerdown rather than click: a
    // repaint replaces the button between press and release, so a click would
    // never land. The arrows set the exact morph of the page they move to.
    el.onpointerdown = function (event) {
      var action = event.target.closest && event.target.closest("[data-action]");
      if (!action) return;

      var what = action.getAttribute("data-action");
      var last = frames.length - 1;
      var step = last > 0 ? 1 / last : 0;

      event.preventDefault();

      if (what === "spectrum-prev") setMorph(Math.max(selected - 1, 0) * step);
      else if (what === "spectrum-next") setMorph(Math.min(selected + 1, last) * step);
      else if (what === "spectrum-add") {
        frames.splice(selected + 1, 0, {
          levels: frame.levels.slice(),
          phases: frame.phases.slice()
        });
        spectrum.generation = (spectrum.generation || 0) + 1;
        setMorph((selected + 1) / Math.max(frames.length - 1, 1));
        post();
      } else if (what === "spectrum-remove" && frames.length > 1) {
        frames.splice(selected, 1);
        spectrum.generation = (spectrum.generation || 0) + 1;
        setMorph(Math.min(selected, frames.length - 1) / Math.max(frames.length - 1, 1));
        post();
      }
    };

    /// A card's action row can be painted under the SVG inside a scaled
    /// foreignObject, so a real press lands on the picture. Match the press to
    /// the buttons' own rectangles and hand it to the one it covers.
    var actionAt = function (event) {
      var buttons = el.querySelectorAll(".hero-action");

      for (var i = 0; i < buttons.length; i++) {
        var rect = buttons[i].getBoundingClientRect();

        if (event.clientX >= rect.left && event.clientX <= rect.right &&
            event.clientY >= rect.top && event.clientY <= rect.bottom) {
          return buttons[i];
        }
      }

      return null;
    };

    svg.addEventListener("pointerdown", function (event) {
      var button = actionAt(event);

      if (button) {
        button.dispatchEvent(new PointerEvent("pointerdown", {
          bubbles: true,
          cancelable: true,
          clientX: event.clientX,
          clientY: event.clientY,
          pointerId: event.pointerId
        }));
        event.preventDefault();
        return;
      }

      event.preventDefault();
      el.dataset.busy = "1";
      drag = null;
      edit(event);

      if (svg.setPointerCapture) {
        try { svg.setPointerCapture(event.pointerId); } catch (error) {}
      }
    });

    svg.addEventListener("pointermove", function (event) {
      if (el.dataset.busy !== "1") return;
      event.preventDefault();
      edit(event);
    });

    var finish = function () {
      if (el.dataset.busy !== "1") return;
      el.dataset.busy = "";
      drag = null;
      post();
      redraw();
    };

    svg.addEventListener("pointerup", finish);
    svg.addEventListener("pointercancel", finish);
  }
