// The card a faceplate and a rack module agree on.
//
// A product page is one card: a header with the mode and header choices, an
// optional picture, and the controls the effect's metadata declares. A rack
// module is the same card without the product title. This file keeps no global
// parameter map, so several cards on one page cannot overwrite each other.
(function () {
  "use strict";

  var START = -135;
  var SWEEP = 270;

  function clamp(value, low, high) {
    return Math.min(high, Math.max(low, value));
  }

  function norm(spec, value) {
    var low = spec.min;
    var high = spec.max;
    if (spec.log && low > 0 && high > low) {
      return clamp(Math.log(value / low) / Math.log(high / low), 0, 1);
    }
    return high > low ? clamp((value - low) / (high - low), 0, 1) : 0;
  }

  function at(spec, position) {
    var low = spec.min;
    var high = spec.max;
    var value = spec.log && low > 0 && high > low
      ? low * Math.pow(high / low, position)
      : low + (high - low) * position;
    return clamp(value, low, high);
  }

  function format(spec, value) {
    if (spec.kind === "bool") return value ? "On" : "Off";
    if (spec.kind === "int" && spec.choices && spec.choices.length) {
      return spec.choices[clamp(Math.round(value), 0, spec.choices.length - 1)];
    }
    if (spec.unit === "hz") {
      return value >= 1000 ? (value / 1000).toFixed(2) + " kHz" : value.toFixed(0) + " Hz";
    }
    if (spec.unit === "percent") return (value * 100).toFixed(0) + " %";
    if (spec.unit === "db") return (value > 0 ? "+" : "") + value.toFixed(1) + " dB";
    if (spec.unit === "ms") return value.toFixed(1) + " ms";
    if (spec.unit === "s") return value.toFixed(3) + " s";
    if (spec.unit === "beats") return value.toFixed(2) + " beats";
    return spec.kind === "int" ? String(Math.round(value)) : value.toFixed(2);
  }

  function svg(tag, attributes) {
    var node = document.createElementNS("http://www.w3.org/2000/svg", tag);
    for (var key in attributes) {
      node.setAttribute(key, attributes[key]);
    }
    return node;
  }

  function polar(cx, cy, radius, degrees) {
    var radians = degrees * Math.PI / 180;
    return {
      x: cx + radius * Math.cos(radians),
      y: cy + radius * Math.sin(radians)
    };
  }

  function arcPath(cx, cy, radius, from, to) {
    if (Math.abs(to - from) < 0.01) return "";
    var start = polar(cx, cy, radius, from);
    var end = polar(cx, cy, radius, to);
    var large = Math.abs(to - from) > 180 ? 1 : 0;
    var sweep = to > from ? 1 : 0;
    return "M " + start.x.toFixed(2) + " " + start.y.toFixed(2) +
      " A " + radius + " " + radius + " 0 " + large + " " + sweep + " " +
      end.x.toFixed(2) + " " + end.y.toFixed(2);
  }

  function valueOf(values, id, fallback) {
    var value = values ? values[id] : undefined;
    return typeof value === "number" ? value : fallback;
  }

  function frame(inner) {
    return '<svg class="mcard-graph" viewBox="0 0 500 150" preserveAspectRatio="none">' +
      '<line x1="18" y1="10" x2="18" y2="140" stroke="#333" stroke-width="1"/>' +
      '<line x1="18" y1="75" x2="482" y2="75" stroke="#2e2e2e" stroke-width="1"/>' +
      inner +
      '<line x1="18" y1="140" x2="482" y2="140" stroke="#333" stroke-width="1"/>' +
      '</svg>';
  }

  function spectrumHero(values) {
    var cutoff = Math.max(valueOf(values, "cutoff", 1000), 10);
    var q = Math.max(valueOf(values, "q", 0.707), 0.05);
    var mode = Math.round(valueOf(values, "mode", 0));
    var gainDb = valueOf(values, "gain_db", 0);
    var slope = Math.max(valueOf(values, "slope", 1), 0.5);
    var sections = mode < 4
      ? clamp(Math.round(valueOf(values, "stages", 0)) + 1, 1, 4)
      : 1;
    var magnitudes = [];

    for (var i = 0; i < 96; i++) {
      var frequency = 20 * Math.pow(1000, i / 95);
      var ratio = frequency / cutoff;
      var denominator = Math.sqrt(
        Math.pow(1 - ratio * ratio, 2) + Math.pow(ratio / q, 2)
      );
      var magnitude;

      if (mode === 1) magnitude = (ratio * ratio) / denominator;
      else if (mode === 2) magnitude = (ratio / q) / denominator;
      else if (mode === 3) magnitude = Math.abs(1 - ratio * ratio) / denominator;
      else if (mode === 4) {
        var peakGain = Math.pow(10, gainDb / 20);
        var octaves = Math.max(0.08, Math.log2(1 + 1 / q));
        var bump = Math.exp(-0.5 * Math.pow(Math.log2(ratio) / octaves, 2));
        magnitude = 1 + (peakGain - 1) * bump;
      } else if (mode === 5 || mode === 6) {
        var shelfGain = Math.pow(10, gainDb / 20);
        var transition = 1 / (1 + Math.pow(ratio, 4 * slope));
        magnitude = mode === 5
          ? 1 + (shelfGain - 1) * transition
          : 1 + (shelfGain - 1) * (1 - transition);
      } else magnitude = 1 / denominator;

      if (mode < 4) magnitude = Math.pow(magnitude, sections);

      magnitudes.push(magnitude);
    }

    var loudest = Math.max.apply(null, magnitudes);
    var points = [];

    for (var j = 0; j < magnitudes.length; j++) {
      var decibels = 20 * Math.log10(Math.max(magnitudes[j] / loudest, 1e-6));
      var x = 18 + j / 95 * 464;
      var y = clamp(10 + (1 - (1 + decibels / 40)) * 130, 10, 140);
      points.push((j ? "L " : "M ") + x.toFixed(1) + " " + y.toFixed(1));
    }

    return frame(
      '<path d="' + points.join(" ") + '" fill="none" stroke="var(--accent)" ' +
      'stroke-width="2" vector-effect="non-scaling-stroke"/>'
    );
  }

  function shaper(mode, value, values) {
    if (mode === 1) {
      var alpha = Math.max(valueOf(values, "alpha", 0.5), 0.01);
      return Math.tanh(value / alpha);
    }
    if (mode === 2) {
      var k = Math.max(valueOf(values, "k", 2.0), 0.01);
      return value * k / (1 + Math.abs(value * k));
    }
    if (mode === 3) {
      var a = Math.max(valueOf(values, "a", 1.0), 0.01);
      var p = Math.max(valueOf(values, "p", 2.0), 1.0);
      return value / (1 + Math.pow(Math.abs(value) / a, p)) * (1 + 1 / Math.pow(a, p));
    }
    if (mode === 4) {
      var bits = Math.max(valueOf(values, "bits", 8), 1);
      var steps = Math.pow(2, bits) / 2;
      return Math.round(value * steps) / steps;
    }
    if (mode === 5) {
      return value;
    }
    var threshold = Math.max(valueOf(values, "threshold", 0.8), 0.01);
    return clamp(value, -threshold, threshold);
  }

  function transferHero(values) {
    var mode = Math.round(valueOf(values, "mode", 0));
    var points = [];

    for (var i = 0; i <= 80; i++) {
      var input = -1 + 2 * i / 80;
      var output = shaper(mode, input, values);
      var x = 18 + (input + 1) / 2 * 464;
      var y = 75 - output * 65;
      points.push((i ? "L " : "M ") + x.toFixed(1) + " " + y.toFixed(1));
    }

    return frame(
      '<line x1="18" y1="10" x2="482" y2="140" stroke="#3a3a3a" stroke-width="1" ' +
      'stroke-dasharray="3 4"/>' +
      '<path d="' + points.join(" ") + '" fill="none" stroke="var(--accent)" ' +
      'stroke-width="2" vector-effect="non-scaling-stroke"/>'
    );
  }

  function waveformHero(values) {
    var gain = clamp(valueOf(values, "gain", 1) / 4, 0.05, 1);
    var amplitude = 62 * gain;
    var points = [];

    for (var i = 0; i <= 160; i++) {
      var x = 18 + i / 160 * 464;
      var y = 75 - Math.sin(i / 160 * Math.PI * 6) * amplitude;
      points.push((i ? "L " : "M ") + x.toFixed(1) + " " + y.toFixed(1));
    }

    return frame(
      '<path d="' + points.join(" ") + '" fill="none" stroke="var(--accent)" ' +
      'stroke-width="2" vector-effect="non-scaling-stroke"/>'
    );
  }

  function hero(kind, values) {
    if (kind === "spectrum") return spectrumHero(values);
    if (kind === "transfer") return transferHero(values);
    if (kind === "waveform") return waveformHero(values);
    return "";
  }

  function makeKnob(spec, value, onEdit) {
    var wrap = document.createElement("div");
    wrap.className = "knob-wrap";

    var dial = document.createElement("div");
    dial.className = "knob";
    dial.tabIndex = 0;
    dial.setAttribute("role", "slider");
    dial.setAttribute("aria-label", spec.label);
    dial.setAttribute("aria-valuemin", String(spec.min));
    dial.setAttribute("aria-valuemax", String(spec.max));

    var drawing = svg("svg", { viewBox: "0 0 64 64" });
    drawing.appendChild(svg("circle", { class: "knob-bg", cx: 32, cy: 32, r: 25 }));
    drawing.appendChild(svg("path", {
      class: "knob-track",
      d: arcPath(32, 32, 25, START, START + SWEEP)
    }));

    var fill = svg("path", { class: "knob-arc" });
    var hand = svg("line", { class: "knob-hand" });
    drawing.appendChild(fill);
    drawing.appendChild(hand);
    dial.appendChild(drawing);

    var label = document.createElement("div");
    label.className = "knob-label";
    label.textContent = spec.label;

    var readout = document.createElement("div");
    readout.className = "knob-value";

    var current = clamp(value, spec.min, spec.max);

    function paint() {
      var position = clamp(norm(spec, current), 0, 1);
      var angle = START + SWEEP * position;
      fill.setAttribute("d", arcPath(32, 32, 25, START, angle));
      var tip = polar(32, 32, 16, angle);
      hand.setAttribute("x1", "32");
      hand.setAttribute("y1", "32");
      hand.setAttribute("x2", tip.x.toFixed(2));
      hand.setAttribute("y2", tip.y.toFixed(2));
      readout.textContent = format(spec, current);
      dial.setAttribute("aria-valuenow", String(current));
      dial.setAttribute("aria-valuetext", format(spec, current));
    }

    function set(value, send) {
      current = clamp(value, spec.min, spec.max);
      if (spec.kind === "int") current = Math.round(current);
      paint();
      if (send && onEdit) onEdit(spec.id, current);
    }

    var drag = null;

    dial.addEventListener("pointerdown", function (event) {
      if (event.button !== 0) return;
      drag = { y: event.clientY, position: norm(spec, current) };
      dial.setPointerCapture(event.pointerId);
      event.preventDefault();
    });

    dial.addEventListener("pointermove", function (event) {
      if (!drag) return;
      var position = clamp(drag.position + (drag.y - event.clientY) / 150, 0, 1);
      set(at(spec, position), true);
    });

    function stop(event) {
      if (!drag) return;
      drag = null;
      try {
        dial.releasePointerCapture(event.pointerId);
      } catch (error) {
        // Released outside the window.
      }
    }

    dial.addEventListener("pointerup", stop);
    dial.addEventListener("pointercancel", stop);
    dial.addEventListener("dblclick", function () { set(spec.default, true); });

    dial.addEventListener("keydown", function (event) {
      var step = (spec.max - spec.min) / 100;
      if (event.key === "ArrowUp" || event.key === "ArrowRight") {
        set(current + step, true);
        event.preventDefault();
      } else if (event.key === "ArrowDown" || event.key === "ArrowLeft") {
        set(current - step, true);
        event.preventDefault();
      } else if (event.key === "Home") {
        set(spec.min, true);
        event.preventDefault();
      } else if (event.key === "End") {
        set(spec.max, true);
        event.preventDefault();
      }
    });

    paint();
    wrap.appendChild(dial);
    wrap.appendChild(label);
    wrap.appendChild(readout);

    return { el: wrap, set: set };
  }

  function makeChoice(spec, value, onEdit, className) {
    var select = document.createElement("select");
    select.className = className || "mcard-select";
    select.setAttribute("aria-label", spec.label);

    spec.choices.forEach(function (name, index) {
      var option = document.createElement("option");
      option.value = String(index);
      option.textContent = name;
      select.appendChild(option);
    });

    select.value = String(Math.round(value));
    select.addEventListener("change", function () {
      if (onEdit) onEdit(spec.id, Number(select.value));
    });

    return select;
  }

  function makeToggle(spec, value, onEdit) {
    var button = document.createElement("button");
    button.type = "button";
    button.className = "mcard-toggle";
    var current = !!value;
    button.textContent = current ? "On" : "Off";
    button.setAttribute("aria-pressed", current ? "true" : "false");
    button.addEventListener("click", function () {
      current = !current;
      button.textContent = current ? "On" : "Off";
      button.setAttribute("aria-pressed", current ? "true" : "false");
      if (onEdit) onEdit(spec.id, current ? 1 : 0);
    });
    return button;
  }

  function makeBodyControl(spec, value, onEdit) {
    if (spec.kind === "bool") {
      var row = document.createElement("div");
      row.className = "mcard-row";
      var toggleLabel = document.createElement("span");
      toggleLabel.className = "mcard-row-label";
      toggleLabel.textContent = spec.label;
      row.appendChild(toggleLabel);
      row.appendChild(makeToggle(spec, value, onEdit));
      return row;
    }

    if (spec.widget === "choice" || (spec.kind === "int" && spec.choices && spec.choices.length)) {
      var choiceRow = document.createElement("div");
      choiceRow.className = "mcard-row";
      var choiceLabel = document.createElement("span");
      choiceLabel.className = "mcard-row-label";
      choiceLabel.textContent = spec.label;
      choiceRow.appendChild(choiceLabel);
      choiceRow.appendChild(makeChoice(spec, value, onEdit));
      return choiceRow;
    }

    return makeKnob(spec, value, onEdit).el;
  }

  function setting(label, control) {
    var box = document.createElement("div");
    box.className = "mcard-setting";
    var name = document.createElement("span");
    name.className = "mcard-setting-label";
    name.textContent = label.toUpperCase();
    box.appendChild(name);
    box.appendChild(control);
    return box;
  }

  function create(options) {
    var card = document.createElement("section");
    card.className = "mcard";
    card.style.setProperty("--accent", options.accent || "#8a6aff");

    var values = {};
    (options.controls || []).forEach(function (spec) {
      values[spec.id] = valueOf(options.values, spec.id, spec.default);
    });

    var byId = {};
    (options.controls || []).forEach(function (spec) { byId[spec.id] = spec; });

    var head = document.createElement("header");
    head.className = "mcard-head";

    var dot = document.createElement("span");
    dot.className = "mcard-dot";
    head.appendChild(dot);

    var name = document.createElement("span");
    name.className = "mcard-name";
    name.textContent = options.title;
    head.appendChild(name);

    var badge = document.createElement("span");
    badge.className = "mcard-badge";
    badge.textContent = options.badge || "";
    head.appendChild(badge);

    var heroElement = null;
    function paintHero() {
      if (heroElement) heroElement.innerHTML = hero(options.hero, values);
    }

    function edited(id, value) {
      values[id] = value;
      paintHero();
      if (options.onEdit) options.onEdit(id, value);
    }

    var modeSpec = options.mode ? byId[options.mode] : null;
    var currentMode = modeSpec ? Math.round(values[modeSpec.id]) : 0;

    if (modeSpec) {
      var modeSelect = makeChoice(modeSpec, values[modeSpec.id], function (_, value) {
        currentMode = Math.round(value);
        applyVisibility();
        edited(modeSpec.id, currentMode);
      }, "mcard-select mcard-mode");
      head.appendChild(setting("Mode", modeSelect));
    }

    (options.header || []).forEach(function (id) {
      var spec = byId[id];
      if (!spec) return;
      var control;
      if (spec.kind === "int" && spec.choices && spec.choices.length) {
        control = makeChoice(spec, values[id], function (_, value) {
          edited(id, value);
        });
      } else {
        control = makeToggle(spec, values[id], function (_, value) {
          edited(id, value);
        });
      }
      head.appendChild(setting(spec.label, control));
    });

    var spacer = document.createElement("span");
    spacer.className = "mcard-spacer";
    head.appendChild(spacer);

    var collapse = document.createElement("button");
    collapse.type = "button";
    collapse.className = "mcard-collapse";
    collapse.textContent = "\u2304";
    collapse.title = "Collapse";
    collapse.setAttribute("aria-label", "Collapse");
    collapse.addEventListener("click", function () {
      var collapsed = card.classList.toggle("collapsed");
      collapse.textContent = collapsed ? "\u2305" : "\u2304";
      collapse.title = collapsed ? "Expand" : "Collapse";
      collapse.setAttribute("aria-label", collapse.title);
    });
    head.appendChild(collapse);

    var bypassed = !!options.bypass;
    var toggle = document.createElement("button");
    toggle.className = "mcard-bypass";
    toggle.type = "button";
    toggle.textContent = bypassed ? "Bypassed" : "Active";
    if (bypassed) toggle.classList.add("off");
    toggle.addEventListener("click", function () {
      bypassed = !bypassed;
      toggle.textContent = bypassed ? "Bypassed" : "Active";
      toggle.classList.toggle("off", bypassed);
      card.classList.toggle("bypassed", bypassed);
      if (options.onBypass) options.onBypass(bypassed);
    });
    head.appendChild(toggle);

    card.appendChild(head);

    if (options.hero) {
      heroElement = document.createElement("div");
      heroElement.className = "mcard-hero";
      card.appendChild(heroElement);
      paintHero();
    }

    var body = document.createElement("div");
    body.className = "mcard-body";

    var bodyControls = [];
    (options.controls || []).forEach(function (spec) {
      if (modeSpec && spec.id === modeSpec.id) return;
      if ((options.header || []).indexOf(spec.id) >= 0) return;
      var el = makeBodyControl(spec, values[spec.id], edited);
      body.appendChild(el);
      bodyControls.push({ spec: spec, el: el });
    });

    card.appendChild(body);

    function applyVisibility() {
      if (!options.visible) return;
      var allowed = null;
      for (var i = 0; i < options.visible.length; i++) {
        if (Number(options.visible[i][0]) === currentMode) {
          allowed = options.visible[i][1];
        }
      }
      bodyControls.forEach(function (entry) {
        var show = !allowed || allowed.indexOf(entry.spec.id) >= 0;
        entry.el.style.display = show ? "" : "none";
      });
    }

    applyVisibility();

    return card;
  }

  window.IAmCard = {
    create: create,
    hero: hero,
    format: format,
    norm: norm,
    at: at
  };
})();
