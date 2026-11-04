// The shared card script, part one: the number formatting, the knob editor and the SVG helpers.
// Every part is one file here and one IIFE once concatenated.
// The card a faceplate and a rack module agree on.
//
// A product page is one card: a header with the mode and header choices, an
// optional picture, and the controls the effect's metadata declares. A rack
// module is the same card without the product title. This file keeps no global
// parameter map, so several cards on one page cannot overwrite each other.
(function () {
  "use strict";

  var START = -225;
  var SWEEP = 270;
  var knobEditor = null;

  function clamp(value, low, high) {
    return Math.min(high, Math.max(low, value));
  }

  /// The scale of the smooth logarithmic map used when a logarithmic range
  /// reaches zero or goes negative. It is 1% of the larger endpoint, so the
  /// middle of a bipolar knob spends more travel near zero without making the
  /// ends feel linear.
  function symlogScale(spec) {
    return Math.max(Math.abs(spec.min), Math.abs(spec.max), 1e-6) / 100;
  }

  function symlogNorm(spec, value) {
    var scale = symlogScale(spec);
    var low = Math.asinh(spec.min / scale);
    var high = Math.asinh(spec.max / scale);
    return high > low ? clamp((Math.asinh(value / scale) - low) / (high - low), 0, 1) : 0;
  }

  function symlogAt(spec, position) {
    var scale = symlogScale(spec);
    var low = Math.asinh(spec.min / scale);
    var high = Math.asinh(spec.max / scale);
    return high > low ? Math.sinh(low + clamp(position, 0, 1) * (high - low)) * scale : spec.min;
  }

  function norm(spec, value) {
    var low = spec.min;
    var high = spec.max;

    if (spec.log && high > low) {
      return low > 0
        ? clamp(Math.log(value / low) / Math.log(high / low), 0, 1)
        : symlogNorm(spec, value);
    }

    return high > low ? clamp((value - low) / (high - low), 0, 1) : 0;
  }

  /// One notch of a control. An explicit step keeps its own units (semitones,
  /// kernel lengths), an integer counts by one, and everything else moves a
  /// fraction of its *normalised* range: a log or symmetric-log knob then takes
  /// a constant ratio per notch instead of a constant number of hertz.
  function stepBy(spec, value, direction, fine) {
    var scale = fine ? 0.1 : 1;

    if (spec.step && spec.step > 0) {
      return value + direction * spec.step * scale;
    }

    if (spec.kind === "int") {
      return value + direction;
    }

    // A log knob spans a much wider range than its pixel width suggests, so it
    // takes a finer share of that range per notch.
    var amount = spec.log ? 0.005 : 0.0125;
    var position = norm(spec, value) + direction * amount * scale;

    return at(spec, clamp(position, 0, 1));
  }

  function at(spec, position) {
    var low = spec.min;
    var high = spec.max;
    position = clamp(position, 0, 1);

    if (spec.log && high > low) {
      return low > 0
        ? low * Math.pow(high / low, position)
        : symlogAt(spec, position);
    }

    return low + (high - low) * position;
  }

  /// A control can override its readout without changing its unit metadata.
  function display(spec, value) {
    return typeof spec.format === "function" ? spec.format(value) : format(spec, value);
  }

  function format(spec, value) {
    if (spec.kind === "bool") return value ? "On" : "Off";
    if (spec.kind === "int" && spec.choices && spec.choices.length) {
      return spec.choices[clamp(Math.round(value), 0, spec.choices.length - 1)];
    }
    if (spec.unit === "hz") {
      if (value >= 1000) return (value / 1000).toFixed(2) + " kHz";
      if (value < 100) return value.toFixed(2) + " Hz";
      return value.toFixed(0) + " Hz";
    }
    if (spec.unit === "percent") return (value * 100).toFixed(0) + " %";
    if (spec.unit === "db") return (value > 0 ? "+" : "") + value.toFixed(1) + " dB";
    if (spec.unit === "ms") return value.toFixed(1) + " ms";
    if (spec.unit === "s") return value.toFixed(3) + " s";
    if (spec.unit === "beats") return value.toFixed(2) + " beats";
    if (spec.unit === "samples") return Math.round(value) + " smp";
    if (spec.unit === "st") return (value > 0 ? "+" : "") + Math.round(value) + " st";
    if (spec.unit === "ct") return (value > 0 ? "+" : "") + Math.round(value) + " ct";
    return spec.kind === "int" ? String(Math.round(value)) : value.toFixed(2);
  }

  function closeKnobEditor() {
    if (knobEditor) {
      knobEditor.remove();
      knobEditor = null;
    }
  }

  /// A small text field over a knob, for entering an exact value.
  function openKnobEditor(spec, current, set, anchor) {
    closeKnobEditor();

    var box = document.createElement("div");
    box.className = "knob-edit";

    var input = document.createElement("input");
    input.type = "text";
    input.value = String(Number(current.toFixed(4)));
    input.spellcheck = false;
    input.setAttribute("aria-label", spec.label + " value");

    var unit = document.createElement("span");
    unit.className = "unit";
    unit.textContent = spec.unit || "";

    box.appendChild(input);
    box.appendChild(unit);
    document.body.appendChild(box);

    var rect = anchor.getBoundingClientRect();
    var width = 118;
    var top = rect.bottom + 4;

    if (top + 30 > window.innerHeight) top = Math.max(4, rect.top - 30);

    box.style.left = Math.min(window.innerWidth - width, Math.max(4, rect.left)) + "px";
    box.style.top = top + "px";

    input.focus();
    input.select();

    input.addEventListener("keydown", function (event) {
      if (event.key === "Enter") {
        var parsed = Number.parseFloat(input.value);
        if (Number.isFinite(parsed)) set(parsed, true);
        closeKnobEditor();
      } else if (event.key === "Escape") {
        closeKnobEditor();
      }
    });

    input.addEventListener("blur", function () {
      setTimeout(closeKnobEditor, 0);
    });

    knobEditor = box;
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

    // Switches are booleans in the faceplate's control map, while hero code
    // reads them as 0/1. Accepting both also fixes the rack's own heroes.
    if (typeof value === "boolean") return value ? 1 : 0;
    return typeof value === "number" ? value : fallback;
  }

  function heroSvg(inner) {
    return '<svg class="mcard-graph" viewBox="0 0 520 160" preserveAspectRatio="none">' +
      inner +
      '</svg>';
  }

  function heroSvgMeter(inner) {
    return '<svg class="mcard-graph" viewBox="0 0 320 200" preserveAspectRatio="xMidYMid meet">' +
      inner +
      '</svg>';
  }
