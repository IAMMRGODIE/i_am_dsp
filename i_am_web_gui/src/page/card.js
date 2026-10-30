// The card a faceplate and a rack module agree on.
//
// A product page is one card: a header, an optional picture, and the controls
// the effect's metadata declares. A rack module is the same card without the
// product header. This file is the drawing code both use; it keeps no global
// parameter map, so two cards on one page cannot overwrite each other.
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
    var points = [];

    for (var i = 0; i < 96; i++) {
      var frequency = 20 * Math.pow(1000, i / 95);
      var ratio = frequency / cutoff;
      var magnitude = 1 / Math.sqrt(
        Math.pow(1 - ratio * ratio, 2) + Math.pow(ratio / q, 2)
      );
      var decibels = 20 * Math.log10(Math.max(magnitude, 1e-6));
      var x = 18 + i / 95 * 464;
      var y = clamp(10 + (1 - (1 + decibels / 40)) * 130, 10, 140);
      points.push((i ? "L " : "M ") + x.toFixed(1) + " " + y.toFixed(1));
    }

    return frame(
      '<path d="' + points.join(" ") + '" fill="none" stroke="var(--accent)" ' +
      'stroke-width="2" vector-effect="non-scaling-stroke"/>'
    );
  }

  function transferHero(values) {
    var amount = Math.max(valueOf(values, "amount", 0.5), 0.01);
    var points = [];

    for (var i = 0; i <= 80; i++) {
      var input = -1 + 2 * i / 80;
      var output = Math.tanh(input / amount);
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
        // The pointer may have been released outside the window.
      }
    }

    dial.addEventListener("pointerup", stop);
    dial.addEventListener("pointercancel", stop);

    dial.addEventListener("dblclick", function () {
      set(spec.default, true);
    });

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

  function create(options) {
    var card = document.createElement("section");
    card.className = "mcard";

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

    var picture = hero(options.hero, options.values);
    if (picture) {
      var box = document.createElement("div");
      box.className = "mcard-hero";
      box.innerHTML = picture;
      card.appendChild(box);
    }

    var body = document.createElement("div");
    body.className = "mcard-body";

    (options.controls || []).forEach(function (spec) {
      var value = valueOf(options.values, spec.id, spec.default);
      body.appendChild(makeKnob(spec, value, options.onEdit).el);
    });

    card.appendChild(body);

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
