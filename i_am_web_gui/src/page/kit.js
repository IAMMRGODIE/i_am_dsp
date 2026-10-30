(function () {
  "use strict";

  var app = document.getElementById("app");
  // Whether the page is a browser preview or lives in a host. The custom
  // protocol WebView2 actually exposes is an http:// origin, so the protocol
  // cannot tell the difference; the server says so in /face instead.
  var isPreview = false;
  var face = null;
  var accent = "#8a6aff";
  var negative = "#9a8fbf";
  var controls = {};
  var heroes = [];
  var dragging = null;
  var drag = null;
  var dirty = false;
  var polling = false;
  var bypassed = false;
  var powerButton = null;
  var modeSpec = null;
  var wantsScope = false;
var wantsMeter = false;
var wantsStereo = false;
var wantsParticles = false;
var wantsIr = false;
var irState = null;
var irInput = null;
var wantsScript = false;
var scriptState = null;
var scriptArea = null;
var scriptStatus = null;
var scriptError = null;
var scriptDirty = false;
var scriptEpoch = 0;
var wantsCurve = false;
var curveState = null;
var wantsOtt = false;
var ottState = null;
var wantsVocoder = false;
var vocoderState = null;
var noteBadge = null;
var stereoScope = {
  inputLeft: null,
  inputRight: null,
  outputLeft: null,
  outputRight: null
};
var meter = { input: null, sidechain: null };
// A short level history so the right-hand envelope can scroll; the wheel
// changes how much of it is visible, which changes how fast it appears to move.
var envHistory = { input: [], output: [] };
var envWindow = 128;
var ENV_MAX = 512;
  var scope = { input: null, output: null };
  var knobs = [];
  var scale = 1;
  var spectra = { input: null, output: null };

  var START = -135;
  var SWEEP = 270;

  function clamp(value, low, high) {
    return Math.min(high, Math.max(low, value));
  }

  function element(tag, className) {
    var node = document.createElement(tag);
    if (className) node.className = className;
    return node;
  }

  function mixHex(from, to, amount) {
    var a = [1, 3, 5].map(function (i) { return parseInt(from.slice(i, i + 2), 16); });
    var b = [1, 3, 5].map(function (i) { return parseInt(to.slice(i, i + 2), 16); });
    return "#" + a.map(function (value, i) {
      return Math.round(value + (b[i] - value) * amount).toString(16).padStart(2, "0");
    }).join("");
  }

  // ------------------------------------------------------------- formatting
  function beats(value) {
    var forms = [[1, ""], [1.5, " dotted"], [2 / 3, " triplet"]];
    for (var i = 0; i < forms.length; i++) {
      var note = value / forms[i][0];
      if (note <= 0) continue;
      var denominator = Math.round(4 / note);
      if (denominator >= 1 && denominator <= 128 &&
          (denominator & (denominator - 1)) === 0 &&
          Math.abs(4 / denominator - note) < 1e-4) {
        return "1/" + denominator + forms[i][1];
      }
    }
    return value.toFixed(2) + " beats";
  }

  function format(spec, value) {
    if (spec.kind === "bool") return value ? "on" : "off";
    if (spec.kind === "int" && spec.choices && spec.choices.length) {
      return spec.choices[clamp(Math.round(value), 0, spec.choices.length - 1)] || String(value);
    }
    switch (spec.unit) {
      case "db": return (value > 0 ? "+" : "") + value.toFixed(2) + " dB";
      case "hz":
        if (value >= 1000) return (value / 1000).toFixed(2) + " kHz";
        if (value >= 10) return value.toFixed(0) + " Hz";
        return value.toFixed(2) + " Hz";
      case "percent": return (value * 100).toFixed(1) + " %";
      case "ms": return value.toFixed(1) + " ms";
      case "s": return value.toFixed(3) + " s";
      case "samples": return value.toFixed(0);
      case "beats": return beats(value);
      default: return spec.kind === "int" ? String(Math.round(value)) : value.toFixed(2);
    }
  }

  /// The number shown in a knob's editor, in the unit the readout uses.
  function input_text(spec, value) {
    switch (spec.unit) {
      case "db": return value.toFixed(2);
      case "hz": return value.toFixed(2);
      case "percent": return (value * 100).toFixed(1);
      case "ms": return value.toFixed(1);
      case "s": return value.toFixed(3);
      case "samples": return String(Math.round(value));
      case "beats": return value.toFixed(3);
      default: return spec.kind === "int" ? String(Math.round(value)) : value.toFixed(2);
    }
  }

  function unit_label(spec) {
    switch (spec.unit) {
      case "db": return "dB";
      case "hz": return "Hz";
      case "percent": return "%";
      case "ms": return "ms";
      case "s": return "s";
      case "samples": return "samples";
      case "beats": return "beats";
      default: return "";
    }
  }

  /// Parses what a person typed into the raw parameter value. Units are
  /// optional and the Hz field accepts k/kHz.
  function parse_input(spec, text) {
    var value = String(text).trim().toLowerCase().replace(/\s+/g, "");

    if (!value) return null;

    var multiplier = 1;

    if (spec.unit === "hz") {
      if (value.endsWith("khz")) { value = value.slice(0, -3); multiplier = 1000; }
      else if (value.endsWith("hz")) { value = value.slice(0, -2); }
      else if (value.endsWith("k")) { value = value.slice(0, -1); multiplier = 1000; }
    } else if (spec.unit === "db") {
      value = value.replace(/db$/, "");
    } else if (spec.unit === "ms") {
      value = value.replace(/ms$/, "");
    } else if (spec.unit === "s") {
      value = value.replace(/s$/, "");
    } else if (spec.unit === "samples") {
      value = value.replace(/samples$/, "");
    } else if (spec.unit === "beats") {
      value = value.replace(/beats$/, "");
    } else if (spec.unit === "percent") {
      value = value.replace(/%$/, "");
    }

    var number = parseFloat(value);

    if (!isFinite(number)) return null;

    number *= multiplier;

    if (spec.unit === "percent") number /= 100;

    return number;
  }

  var knob_editor = null;

  function close_knob_editor() {
    if (knob_editor) {
      knob_editor.remove();
      knob_editor = null;
    }
  }

  /// A small text field over the control, for entering an exact value.
  function open_knob_editor(control, anchor) {
    close_knob_editor();

    var spec = control.spec;
    var box = element("div", "knob-edit");
    var input = document.createElement("input");
    input.type = "text";
    input.value = input_text(spec, control.value);
    input.spellcheck = false;
    input.setAttribute("aria-label", spec.label + " value");

    var unit = element("span", "unit");
    unit.textContent = unit_label(spec);

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
        var parsed = parse_input(spec, input.value);
        if (parsed !== null) setValue(control, parsed, true);
        close_knob_editor();
      } else if (event.key === "Escape") {
        close_knob_editor();
      }
    });

    input.addEventListener("blur", function () {
      // Enter closes first; a click elsewhere just discards the edit.
      setTimeout(close_knob_editor, 0);
    });

    knob_editor = box;
  }

  // ------------------------------------------------------------ value space
  function curveOf(spec) {
    return face && face.curves ? face.curves[spec.id] : null;
  }

  function atanKnee(spec) {
    var limit = Math.max(Math.abs(spec.min), Math.abs(spec.max), 1);
    return { limit: limit, knee: Math.max(limit * 0.1, 1e-3) };
  }

  function norm(spec, value) {
    if (spec.max === spec.min) return 0;

    if (curveOf(spec) === "atan") {
      var k = atanKnee(spec);
      return 0.5 + Math.atan(value / k.knee) / (2 * Math.atan(k.limit / k.knee));
    }

    if (spec.log && spec.min > 0) return Math.log(value / spec.min) / Math.log(spec.max / spec.min);
    return (value - spec.min) / (spec.max - spec.min);
  }

  function at(spec, position) {
    position = clamp(position, 0, 1);

    if (curveOf(spec) === "atan") {
      var k = atanKnee(spec);
      return k.knee * Math.tan((position - 0.5) * 2 * Math.atan(k.limit / k.knee));
    }

    if (spec.log && spec.min > 0) return spec.min * Math.pow(spec.max / spec.min, position);
    return spec.min + position * (spec.max - spec.min);
  }

  function snap(spec, value) {
    if (spec.step && spec.step > 0) value = Math.round(value / spec.step) * spec.step;
    return clamp(value, spec.min, spec.max);
  }

  function setValue(control, value, send) {
    value = snap(control.spec, value);

    // An integer parameter has to go back as an integer: the host parses it with
    // the parameter's own kind, and "8.37" is not an integer.
    if (control.spec.kind === "int") value = Math.round(value);

    control.value = value;
    control.paint();
    markDirty();
    applySwitches();

    if (send && window.ipc) {
      var wire = control.spec.kind === "int" ? Math.round(value) : value;
      window.ipc.postMessage(control.spec.id + "=" + wire);
    }
  }

  // --------------------------------------------------------------- controls
  function makeControl(spec, header) {
    var control = {
      spec: spec,
      value: spec.default,
      el: null,
      paint: function () {}
    };

    controls[spec.id] = control;

    var el;

    if (header && spec.kind === "bool") el = buildHeaderDot(control);
    else if (spec.widget === "toggle" || spec.kind === "bool") el = buildToggle(control);
    else if (spec.widget === "choice") el = buildChoice(control);
    else if (spec.widget === "slider") el = buildSlider(control);
    else el = buildKnob(control);

    control.el = el;
    return el;
  }

  /// A boolean in the header is a dot, like the bypass one: no label and no row
  /// taken from the faceplate.
  function buildHeaderDot(control) {
    var spec = control.spec;
    var el = element("button", "header-dot");
    el.type = "button";
    el.title = spec.label;
    el.setAttribute("aria-label", spec.label);
    el.appendChild(element("span", "dot"));

    control.paint = function () {
      var on = !!control.value;
      el.classList.toggle("on", on);
      el.setAttribute("aria-pressed", on ? "true" : "false");
    };

    el.addEventListener("click", function () {
      setValue(control, control.value ? 0 : 1, true);
    });

    control.paint();
    return el;
  }

  function knobGeometry(diameter) {
    var radius = diameter / 2;
    var ring = Math.max(radius / 64 * 10, 3);
    var arcRadius = radius - ring / 2 - 2;
    return {
      radius: radius,
      ring: ring,
      arc: arcRadius,
      body: arcRadius - ring / 2
    };
  }

  function polar(cx, cy, radius, degrees) {
    var radians = (degrees - 90) * Math.PI / 180;
    return [cx + Math.cos(radians) * radius, cy + Math.sin(radians) * radius];
  }

  function arcPath(cx, cy, radius, from, to) {
    if (to - from < 0.01) return "";
    var a = polar(cx, cy, radius, from);
    var b = polar(cx, cy, radius, to);
    var large = to - from > 180 ? 1 : 0;
    return "M " + a[0].toFixed(2) + " " + a[1].toFixed(2) +
      " A " + radius.toFixed(2) + " " + radius.toFixed(2) + " 0 " + large + " 1 " +
      b[0].toFixed(2) + " " + b[1].toFixed(2);
  }

  function buildKnob(control) {
    var spec = control.spec;
    // One knob size across the kit: a mixed row of 40/56/72 reads as a layout
    // accident, not as emphasis.
    var diameter = 56;
    var geometry = knobGeometry(diameter);
    var radius = geometry.radius;
    var zero = spec.bipolar ? clamp(norm(spec, 0), 0, 1) : 0;

    var el = element("div", "knob");
    var ns = "http://www.w3.org/2000/svg";
    var svg = document.createElementNS(ns, "svg");
    svg.setAttribute("width", diameter);
    svg.setAttribute("height", diameter);
    svg.setAttribute("viewBox", "0 0 " + diameter + " " + diameter);
    svg.setAttribute("tabindex", "0");
    svg.setAttribute("role", "slider");
    svg.setAttribute("aria-label", spec.label);

    // The ring is drawn in a viewBox of its design size; the CSS size follows
    // the window, so a bigger window gets a bigger knob rather than a hole.
    var resize = function (factor) {
      svg.style.width = (diameter * factor) + "px";
      svg.style.height = (diameter * factor) + "px";
    };
    resize(scale);
    knobs.push(resize);

    var track = document.createElementNS(ns, "path");
    track.setAttribute("d", arcPath(radius, radius, geometry.arc, START, START + SWEEP));
    track.setAttribute("fill", "none");
    track.setAttribute("stroke", "#2a2a2a");
    track.setAttribute("stroke-width", geometry.ring);
    track.setAttribute("stroke-linecap", "round");
    svg.appendChild(track);

    var valueArc = document.createElementNS(ns, "path");
    valueArc.setAttribute("fill", "none");
    valueArc.setAttribute("stroke-width", geometry.ring);
    valueArc.setAttribute("stroke-linecap", "round");
    svg.appendChild(valueArc);

    var body = document.createElementNS(ns, "circle");
    body.setAttribute("cx", radius);
    body.setAttribute("cy", radius);
    body.setAttribute("r", geometry.body);
    body.setAttribute("fill", "#333");
    svg.appendChild(body);

    var indicator = document.createElementNS(ns, "line");
    indicator.setAttribute("stroke", "#e0e0e0");
    indicator.setAttribute("stroke-width", Math.max(1.5, diameter * 0.05));
    indicator.setAttribute("stroke-linecap", "round");
    svg.appendChild(indicator);

    var label = element("div", "label");
    label.textContent = spec.label;
    var readout = element("div", "readout");

    control.paint = function () {
      var position = clamp(norm(spec, control.value), 0, 1);
      var low = Math.min(zero, position);
      var high = Math.max(zero, position);

      valueArc.setAttribute("d", arcPath(radius, radius, geometry.arc, START + low * SWEEP, START + high * SWEEP));
      valueArc.setAttribute("stroke", spec.bipolar && control.value < 0 ? negative : accent);

      var angle = START + position * SWEEP;
      var a = polar(radius, radius, geometry.body * 0.35, angle);
      var b = polar(radius, radius, geometry.arc + geometry.ring / 2, angle);
      indicator.setAttribute("x1", a[0].toFixed(2));
      indicator.setAttribute("y1", a[1].toFixed(2));
      indicator.setAttribute("x2", b[0].toFixed(2));
      indicator.setAttribute("y2", b[1].toFixed(2));

      readout.textContent = format(spec, control.value);
      svg.setAttribute("aria-valuemin", spec.min);
      svg.setAttribute("aria-valuemax", spec.max);
      svg.setAttribute("aria-valuenow", control.value);
      svg.setAttribute("aria-valuetext", format(spec, control.value));
    };

    function nudge(amount) {
      setValue(control, at(spec, norm(spec, control.value) + amount), true);
    }

    svg.addEventListener("pointerdown", function (event) {
      if (event.button !== 0) return;
      event.preventDefault();
      dragging = control;
      drag = { y: event.clientY, value: control.value };
      try { svg.setPointerCapture(event.pointerId); } catch (ignore) {}
    });

    svg.addEventListener("pointermove", function (event) {
      if (dragging !== control || !drag) return;
      var distance = (drag.y - event.clientY) / 180;
      var move = distance * (event.shiftKey ? 0.1 : 1);
      setValue(control, at(spec, norm(spec, drag.value) + move), true);
    });

    function release(event) {
      if (dragging !== control) return;
      dragging = null;
      drag = null;
      try { svg.releasePointerCapture(event.pointerId); } catch (ignore) {}
    }

    svg.addEventListener("pointerup", release);
    svg.addEventListener("pointercancel", release);

    svg.addEventListener("dblclick", function () {
      setValue(control, spec.default, true);
    });

    el.title = "Right-click to type a value";
    el.addEventListener("contextmenu", function (event) {
      event.preventDefault();
      open_knob_editor(control, el);
    });

    svg.addEventListener("wheel", function (event) {
      event.preventDefault();
      if (spec.step && spec.step > 0) {
        setValue(control, control.value + (event.deltaY < 0 ? 1 : -1) * spec.step, true);
      } else {
        nudge((event.deltaY < 0 ? 1 : -1) * (event.shiftKey ? 0.001 : 0.01));
      }
    }, { passive: false });

    svg.addEventListener("keydown", function (event) {
      var step = event.shiftKey ? 0.001 : 0.01;
      if (event.key === "ArrowUp" || event.key === "ArrowRight") { nudge(step); event.preventDefault(); }
      else if (event.key === "ArrowDown" || event.key === "ArrowLeft") { nudge(-step); event.preventDefault(); }
      else if (event.key === "Home") { setValue(control, spec.min, true); event.preventDefault(); }
      else if (event.key === "End") { setValue(control, spec.max, true); event.preventDefault(); }
    });

    el.appendChild(svg);
    el.appendChild(label);
    el.appendChild(readout);
    control.paint();

    return el;
  }

  function buildToggle(control) {
    var spec = control.spec;
    var el = element("div", "toggle");
    var track = element("div", "switch");
    track.appendChild(element("i"));
    var label = element("div", "label");
    label.textContent = spec.label;
    var readout = element("div", "readout");

    control.paint = function () {
      el.classList.toggle("on", !!control.value);
      readout.textContent = format(spec, control.value);
    };

    el.addEventListener("click", function () {
      setValue(control, control.value ? 0 : 1, true);
    });

    el.appendChild(track);
    el.appendChild(label);
    el.appendChild(readout);
    control.paint();

    return el;
  }

  function buildChoice(control) {
    var spec = control.spec;
    var el = element("div", "choice");
    var label = element("div", "label");
    label.textContent = spec.label;

    var select = document.createElement("select");
    (spec.choices || []).forEach(function (name, index) {
      var option = document.createElement("option");
      option.value = String(index);
      option.textContent = name;
      select.appendChild(option);
    });

    control.paint = function () {
      select.value = String(Math.round(control.value));
    };

    select.addEventListener("change", function () {
      setValue(control, parseInt(select.value, 10), true);
    });

    el.appendChild(label);
    el.appendChild(select);
    control.paint();

    return el;
  }

  function buildSlider(control) {
    var spec = control.spec;
    var zero = spec.bipolar ? clamp(norm(spec, 0), 0, 1) : 0;

    var el = element("div", "slider");
    var top = element("div", "top");
    var label = element("div", "label");
    label.textContent = spec.label;
    var readout = element("div", "readout");
    top.appendChild(label);
    top.appendChild(readout);

    var track = element("div", "track");
    track.setAttribute("tabindex", "0");
    track.setAttribute("role", "slider");
    track.setAttribute("aria-label", spec.label);
    var fill = element("div", "fill");
    var handle = element("div", "handle");
    track.appendChild(fill);
    track.appendChild(handle);

    control.paint = function () {
      var position = clamp(norm(spec, control.value), 0, 1);
      fill.style.left = (Math.min(zero, position) * 100) + "%";
      fill.style.width = (Math.abs(position - zero) * 100) + "%";
      handle.style.left = (position * 100) + "%";
      readout.textContent = format(spec, control.value);
      track.setAttribute("aria-valuenow", control.value);
      track.setAttribute("aria-valuetext", format(spec, control.value));
    };

    function apply(event) {
      var box = track.getBoundingClientRect();
      var position = (event.clientX - box.left) / box.width;
      setValue(control, at(spec, position), true);
    }

    track.addEventListener("pointerdown", function (event) {
      if (event.button !== 0) return;
      event.preventDefault();
      dragging = control;
      try { track.setPointerCapture(event.pointerId); } catch (ignore) {}
      apply(event);
    });

    track.addEventListener("pointermove", function (event) {
      if (dragging === control) apply(event);
    });

    function release(event) {
      if (dragging !== control) return;
      dragging = null;
      try { track.releasePointerCapture(event.pointerId); } catch (ignore) {}
    }

    track.addEventListener("pointerup", release);
    track.addEventListener("pointercancel", release);

    el.appendChild(top);
    el.appendChild(track);
    control.paint();

    return el;
  }

  // ----------------------------------------------------------------- heroes
  function valueOf(id, fallback) {
    var control = controls[id];
    return control ? control.value : fallback;
  }

  function heroSvg(content) {
    return '<svg viewBox="0 0 520 160" preserveAspectRatio="none">' + content + '</svg>';
  }

  /// A hero that keeps its proportions: a circle stays a circle whatever shape
  /// the hero box is.
  function heroSvgSquare(content) {
    return '<svg viewBox="0 0 520 160" preserveAspectRatio="xMidYMid meet">' + content + '</svg>';
  }

  function heroSvgMeter(content) {
    return '<svg viewBox="0 0 320 200" preserveAspectRatio="xMidYMid meet">' + content + '</svg>';
  }

  function wavePath(amplitude, middle, cycles) {
    var path = "";
    for (var i = 0; i <= 180; i++) {
      var x = 20 + i / 180 * 480;
      var y = middle - Math.sin(i / 180 * Math.PI * 2 * cycles) * amplitude;
      path += (i ? " L " : "M ") + x.toFixed(1) + " " + y.toFixed(1);
    }
    return path;
  }

  function trace(samples, middle, amplitude, stride) {
    stride = stride || 1;

    var last = samples.length - 1;
    var path = "";
    var point = function (index) {
      var x = 20 + index / last * 480;
      var y = middle - clamp(samples[index], -1.2, 1.2) / 1.2 * amplitude;
      return (path ? " L " : "M ") + x.toFixed(1) + " " + y.toFixed(1);
    };

    for (var i = 0; i <= last; i += stride) {
      path += point(i);
    }

    if (last % stride !== 0) path += point(last);

    return path;
  }

  function waveformSvg() {
    if (scope.input && scope.output) {
      return heroSvg(
        '<line x1="0" y1="80" x2="520" y2="80" stroke="#2c2c2c" stroke-width="1"/>' +
        '<path d="' + trace(scope.input, 80, 58, 8) + '" fill="none" stroke="#6a6a6a" ' +
          'stroke-width="1.5" vector-effect="non-scaling-stroke"/>' +
        '<path d="' + trace(scope.output, 80, 58, 8) + '" fill="none" stroke="' + accent + '" ' +
          'stroke-width="2" vector-effect="non-scaling-stroke"/>'
      );
    }

    var gain = valueOf("gain", 1);
    var db = 20 * Math.log10(Math.max(gain, 1e-4));
    var output = clamp(20 * Math.pow(10, db / 20), 3, 58);
    var input = wavePath(16, 80, 3);

    return heroSvg(
      '<line x1="0" y1="80" x2="520" y2="80" stroke="#2c2c2c" stroke-width="1"/>' +
      // The input is an area, not a second line: at unity the two traces coincide,
      // and a line under a line is invisible.
      '<path d="' + input + ' L 500 80 L 20 80 Z" fill="#8a8a8a" fill-opacity=".14"/>' +
      '<path d="' + input + '" fill="none" stroke="#6a6a6a" stroke-width="1.5" ' +
        'vector-effect="non-scaling-stroke"/>' +
      '<path d="' + wavePath(output, 80, 3) + '" fill="none" stroke="' + accent + '" ' +
        'stroke-width="2" vector-effect="non-scaling-stroke"/>'
    );
  }

  function rmsOf(samples) {
    if (!samples || samples.length === 0) return 0;

    var sum = 0;
    for (var i = 0; i < samples.length; i++) sum += samples[i] * samples[i];

    return Math.sqrt(sum / samples.length);
  }

  function dbText(rms) {
    if (!(rms > 1e-6)) return "-inf";
    return (20 * Math.log10(rms)).toFixed(1);
  }

  function rmsBadge() {
    if (!scope.input || !scope.output) {
      return '<div class="hero-badge">RMS --</div>';
    }

    return '<div class="hero-badge">RMS ' + dbText(rmsOf(scope.input)) +
      ' \u2192 ' + dbText(rmsOf(scope.output)) + ' dB</div>';
  }

  function transferSvg() {
    var mode = Math.round(valueOf("mode", 0));
    var threshold = clamp(valueOf("threshold", 0.8), 0.01, 4);
    var gain = valueOf("gain", 1);
    var path = "";
    var bands = "";

    // Where the signal actually lives: a vertical band for how far the input
    // swings, a horizontal one for the output, both behind the curve.
    if (scope.input && scope.output) {
      var inputRms = Math.min(rmsOf(scope.input), 2);
      var outputRms = Math.min(rmsOf(scope.output), 2);

      if (inputRms > 0.0005) {
        var left = 20 + (-inputRms + 2) / 4 * 480;
        var right = 20 + (inputRms + 2) / 4 * 480;
        bands += '<rect x="' + left.toFixed(1) + '" y="0" width="' +
          (right - left).toFixed(1) + '" height="160" fill="#9a9a9a" fill-opacity=".10"/>';
      }

      if (outputRms > 0.0005) {
        var top = 80 - outputRms / 2 * 66;
        var bottom = 80 + outputRms / 2 * 66;
        bands += '<rect x="0" y="' + top.toFixed(1) + '" width="520" height="' +
          (bottom - top).toFixed(1) + '" fill="' + accent + '" fill-opacity=".08"/>';
      }
    }

    var alpha = Math.max(valueOf("alpha", 0.5), 0.01);
    var k = valueOf("k", 2);
    var a = Math.max(valueOf("a", 1), 0.01);
    var p = valueOf("p", 2);
    var bits = Math.max(1, valueOf("bits", 8));

    // The domain is wider than the clip point on purpose: a hard clipper's curve
    // only shows its knees when the flat part has somewhere to run.
    for (var i = 0; i <= 140; i++) {
      var x = -2 + i / 140 * 4;
      var shaped;

      if (mode === 1) shaped = Math.tanh(x / alpha);
      else if (mode === 2) shaped = 2 / (1 + Math.exp(-x * k)) - 1;
      else if (mode === 3) {
        var factor = 1 + 1 / Math.pow(a, p);
        var norm = Math.pow(Math.abs(x) / a, p);
        shaped = (x / (1 + norm)) * factor;
      } else if (mode === 4) shaped = Math.round(bits * x) / bits;
      else if (mode === 5) shaped = x;
      else shaped = clamp(x, -threshold, threshold);

      var y = clamp(shaped * gain, -2, 2);
      path += (i ? " L " : "M ") + (20 + (x + 2) / 4 * 480).toFixed(1) +
              " " + (80 - y / 2 * 66).toFixed(1);
    }

    return heroSvg(
      bands +
      '<path d="M 20 146 L 500 14" stroke="#333" stroke-width="1" stroke-dasharray="3 5" ' +
        'vector-effect="non-scaling-stroke"/>' +
      '<line x1="260" y1="12" x2="260" y2="148" stroke="#2b2b2b"/>' +
      '<line x1="20" y1="80" x2="500" y2="80" stroke="#2b2b2b"/>' +
      '<path d="' + path + '" fill="none" stroke="' + accent + '" stroke-width="2.4" ' +
        'vector-effect="non-scaling-stroke"/>'
    );
  }

  var BANDS = 96;

  function fft(real, imag) {
    var n = real.length;

    for (var i = 1, j = 0; i < n; i++) {
      var bit = n >> 1;
      for (; j & bit; bit >>= 1) j ^= bit;
      j ^= bit;

      if (i < j) {
        var tr = real[i]; real[i] = real[j]; real[j] = tr;
        var ti = imag[i]; imag[i] = imag[j]; imag[j] = ti;
      }
    }

    for (var length = 2; length <= n; length <<= 1) {
      var angle = -2 * Math.PI / length;
      var stepR = Math.cos(angle);
      var stepI = Math.sin(angle);
      var half = length >> 1;

      for (var start = 0; start < n; start += length) {
        var wr = 1, wi = 0;

        for (var k = 0; k < half; k++) {
          var ar = real[start + k], ai = imag[start + k];
          var br = real[start + k + half], bi = imag[start + k + half];
          var vr = br * wr - bi * wi;
          var vi = br * wi + bi * wr;

          real[start + k] = ar + vr;
          imag[start + k] = ai + vi;
          real[start + k + half] = ar - vr;
          imag[start + k + half] = ai - vi;

          var nextR = wr * stepR - wi * stepI;
          wi = wr * stepI + wi * stepR;
          wr = nextR;
        }
      }
    }
  }

  /// Log-spaced bands in dBFS, so a spectrum can share the response's x axis.
  ///
  /// Four times zero padding gives every band several bins, so a band can keep
  /// the loudest bin without looking like a comb; the triangular filter below
  /// smooths the result across neighbouring bands. Averaging the power over a
  /// band would make the same tone quieter the higher it sits, because a high
  /// band covers far more bins than a low one.
  function bandsOf(samples, rate) {
    var n = samples.length;
    if (!n || (n & (n - 1)) !== 0) return null;

    var fftLength = n * 4;
    var real = new Float32Array(fftLength);
    var imag = new Float32Array(fftLength);

    for (var i = 0; i < n; i++) {
      var window = 0.5 - 0.5 * Math.cos(2 * Math.PI * i / (n - 1));
      real[i] = samples[i] * window;
    }

    fft(real, imag);

    var raw = new Float32Array(BANDS);

    for (var k = 1; k < fftLength / 2; k++) {
      var frequency = k * rate / fftLength;
      if (frequency < 20 || frequency > 20000) continue;

      var magnitude = 4 * Math.sqrt(real[k] * real[k] + imag[k] * imag[k]) / n;
      var band = Math.floor(Math.log(frequency / 20) / Math.log(1000) * BANDS);

      if (band < 0) band = 0;
      if (band >= BANDS) band = BANDS - 1;
      if (magnitude > raw[band]) raw[band] = magnitude;
    }

    for (var b = 0; b < BANDS; b++) {
      raw[b] = 20 * Math.log10(Math.max(raw[b], 1e-7));
    }

    // A short triangular filter across neighbouring bands. The response curve
    // is analytic; the spectrum is measured, and a few pixels of smoothing make
    // the two readable together without hiding the shape of either.
    var smooth = new Float32Array(BANDS);
    var weights = [1, 2, 1];

    for (var c = 0; c < BANDS; c++) {
      var sum = 0;
      var weight = 0;

      for (var w = -1; w <= 1; w++) {
        var index = clamp(c + w, 0, BANDS - 1);
        var factor = weights[w + 1];
        sum += raw[index] * factor;
        weight += factor;
      }

      // A tone can live in a single band; averaging it with empty neighbours
      // would make it quieter the higher it is. Keep the band's own peak and use
      // the average only to lift the gaps between measured bands.
      smooth[c] = Math.max(raw[c], sum / weight);
    }

    return smooth;
  }

  function smoothBands(previous, next) {
    if (!previous || previous.length !== next.length) return next;

    for (var i = 0; i < next.length; i++) {
      previous[i] += (next[i] - previous[i]) * 0.28;
    }

    return previous;
  }

  function bandsPath(bands) {
    var path = "";

    for (var i = 0; i < bands.length; i++) {
      var x = 20 + i / (BANDS - 1) * 480;
      var y = clamp(10 + clamp(-bands[i] / 60, 0, 1) * 140, 10, 150);
      path += (i ? " L " : "M ") + x.toFixed(1) + " " + y.toFixed(1);
    }

    return path;
  }

  function spectrumSvg() {
    var cutoff = Math.max(valueOf("cutoff", 1000), 10);
    var q = Math.max(valueOf("q", 0.707), 0.05);
    var mode = Math.round(valueOf("mode", 0));
    var gainDb = valueOf("gain_db", 0);
    var slope = Math.max(valueOf("slope", 1), 0.5);
    var sections = mode < 4
      ? clamp(Math.round(valueOf("stages", 0)) + 1, 1, 4)
      : 1;
    var magnitudes = [];

    for (var i = 0; i < 80; i++) {
      var frequency = 20 * Math.pow(1000, i / 79);
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
    var response = "";

    for (var j = 0; j < magnitudes.length; j++) {
      var decibels = 20 * Math.log10(Math.max(magnitudes[j] / loudest, 1e-6));
      var x = 20 + j / 79 * 480;
      var y = clamp(10 + (1 - (1 + decibels / 40)) * 140, 10, 150);
      response += (j ? " L " : "M ") + x.toFixed(1) + " " + y.toFixed(1);
    }

    var inputPath = "";
    var outputPath = "";
    var rate = face.rate || 48000;

    if (scope.input && scope.output) {
      var inputBands = bandsOf(scope.input, rate);
      var outputBands = bandsOf(scope.output, rate);

      if (inputBands && outputBands) {
        spectra.input = smoothBands(spectra.input, inputBands);
        spectra.output = smoothBands(spectra.output, outputBands);
        inputPath = bandsPath(spectra.input);
        outputPath = bandsPath(spectra.output);
      }
    }

    return heroSvg(
      (inputPath
        ? '<path d="' + inputPath + '" fill="none" stroke="#8a8a8a" stroke-width="1.2" ' +
          'opacity=".7" vector-effect="non-scaling-stroke"/>'
        : "") +
      (outputPath
        ? '<path d="' + outputPath + ' L 500 150 L 20 150 Z" fill="' + accent + '" ' +
            'fill-opacity=".06"/>' +
          '<path d="' + outputPath + '" fill="none" stroke="' + accent + '" stroke-width="1.5" ' +
            'opacity=".9" vector-effect="non-scaling-stroke"/>'
        : "") +
      '<path d="' + response + '" fill="none" stroke="' + accent + '" stroke-width="2" ' +
        'opacity=".95" vector-effect="non-scaling-stroke"/>'
    );
  }

  /// A stereo meter: input and output L/R plotted on one goniometer. The x axis
  /// is the side field (L - R) and y is the mid field (L + R); a wider Width
  /// opens the accent cloud horizontally. Grey is before, accent is after.
  function correlation(left, right) {
    if (!left || !right || left.length < 2) return 0;

    var n = Math.min(left.length, right.length);
    var sumLeft = 0;
    var sumRight = 0;

    for (var i = 0; i < n; i++) {
      sumLeft += left[i];
      sumRight += right[i];
    }

    var meanLeft = sumLeft / n;
    var meanRight = sumRight / n;
    var covariance = 0;
    var leftEnergy = 0;
    var rightEnergy = 0;

    for (var j = 0; j < n; j++) {
      var dl = left[j] - meanLeft;
      var dr = right[j] - meanRight;
      covariance += dl * dr;
      leftEnergy += dl * dl;
      rightEnergy += dr * dr;
    }

    var denominator = Math.sqrt(leftEnergy * rightEnergy);
    return denominator > 1e-12
      ? Math.max(-1, Math.min(1, covariance / denominator))
      : 0;
  }

  /// The particle wider's schematic: one pair of L/R lines per bin, each
  /// scattered by a fresh random factor inside the Low/High band. Depth and
  /// Side set how far they jump, and Negative lets a line cross the axis.
  function particlesSvg() {
    var depth = clamp(valueOf("depth", 0.5), 0, 1);
    var low = clamp(valueOf("low_frequency", 200), 20, 20000);
    var high = clamp(valueOf("high_frequency", 8000), 20, 20000);
    var side = clamp(valueOf("side_gain", 1), 0, 2);
    var allowNegative = !!valueOf("allow_negative", false);
    var fftIndex = clamp(Math.round(valueOf("fft_size", 2)), 0, 3);
    var windowSize = [512, 1024, 2048, 4096][fftIndex];
    var count = Math.max(16, Math.min(128, windowSize / 32));
    var left = "";
    var right = "";
    var lower = Math.min(low, high);
    var upper = Math.max(low, high);
    var frequencyToX = function (frequency) {
      return 30 + Math.log(frequency / 20) / Math.log(1000) * 460;
    };
    var bandLeft = frequencyToX(lower);
    var bandRight = frequencyToX(upper);
    var grid =
      '<line x1="18" y1="80" x2="502" y2="80" stroke="#343434"/>' +
      '<line x1="260" y1="14" x2="260" y2="146" stroke="#2a2a2a"/>' +
      '<rect x="' + bandLeft.toFixed(1) + '" y="22" width="' +
        (bandRight - bandLeft).toFixed(1) + '" height="116" fill="' +
        accent + '" fill-opacity=".05"/>' +
      '<line x1="' + bandLeft.toFixed(1) + '" y1="22" x2="' + bandLeft.toFixed(1) +
        '" y2="138" stroke="' + accent + '" stroke-opacity=".25"/>' +
      '<line x1="' + bandRight.toFixed(1) + '" y1="22" x2="' + bandRight.toFixed(1) +
        '" y2="138" stroke="' + accent + '" stroke-opacity=".25"/>';

    for (var index = 0; index < count; index++) {
      var x = 30 + index / (count - 1) * 460;
      var frequency = 20 * Math.pow(1000, index / (count - 1));
      var active = frequency >= lower && frequency <= upper;
      var range = (0.12 + 0.88 * depth) * (0.25 + 0.75 * Math.min(side, 2) / 2);
      var signLeft = allowNegative ? Math.random() * 2 - 1 : Math.random();
      var signRight = allowNegative ? Math.random() * 2 - 1 : Math.random();
      var jumpLeft = active ? signLeft * range * 60 : 0;
      var jumpRight = active ? signRight * range * 60 : 0;

      left += '<line x1="' + x.toFixed(1) + '" y1="80" x2="' + x.toFixed(1) +
        '" y2="' + (80 - jumpLeft).toFixed(1) + '" stroke="' + accent +
        '" stroke-width="1.4" opacity="' + (active ? 0.7 : 0.18) + '"/>';
      right += '<line x1="' + (x + 1.5).toFixed(1) + '" y1="80" x2="' +
        (x + 1.5).toFixed(1) + '" y2="' + (80 - jumpRight).toFixed(1) +
        '" stroke="#b7f0e2" stroke-width="1.1" opacity="' + (active ? 0.5 : 0.12) +
        '"/>';
    }

    return heroSvgSquare(grid + left + right);
  }

  function stereoSvg() {
    var cx = 160;
    var cy = 72;
    var radius = 62;
    var grid = "";

    for (var r = 20; r <= 60; r += 20) {
      grid += '<circle cx="' + cx + '" cy="' + cy + '" r="' + r +
        '" fill="none" stroke="#3a3a3a" stroke-width="1"/>';
    }

    grid += '<line x1="' + (cx - 62) + '" y1="' + cy + '" x2="' + (cx + 62) + '" y2="' +
      cy + '" stroke="#2f2f2f"/>' +
      '<line x1="' + cx + '" y1="' + (cy - 62) + '" x2="' + cx + '" y2="' + (cy + 62) +
      '" stroke="#2f2f2f"/>';

    var diagonal = 62 / Math.SQRT2;
    grid += '<line x1="' + (cx - diagonal).toFixed(1) + '" y1="' + (cy - diagonal).toFixed(1) +
      '" x2="' + (cx + diagonal).toFixed(1) + '" y2="' + (cy + diagonal).toFixed(1) +
      '" stroke="#2a2a2a"/>' +
      '<line x1="' + (cx - diagonal).toFixed(1) + '" y1="' + (cy + diagonal).toFixed(1) +
      '" x2="' + (cx + diagonal).toFixed(1) + '" y2="' + (cy - diagonal).toFixed(1) +
      '" stroke="#2a2a2a"/>';

    var input = "";
    var output = "";
    var labels = "";
    var points = stereoScope.inputLeft ? stereoScope.inputLeft.length : 0;

    if (points > 1) {
      // Keep a quiet signal from filling the meter, but let a loud one use it.
      var peak = 0.4;

      for (var p = 0; p < points; p += 2) {
        peak = Math.max(
          peak,
          Math.abs(stereoScope.inputLeft[p]),
          Math.abs(stereoScope.inputRight[p]),
          Math.abs(stereoScope.outputLeft[p]),
          Math.abs(stereoScope.outputRight[p])
        );
      }

      var scale = radius / peak;
      var step = Math.max(1, Math.floor(points / 512));

      for (var i = 0; i < points; i += step) {
        var il = stereoScope.inputLeft[i];
        var ir = stereoScope.inputRight[i];
        var ol = stereoScope.outputLeft[i];
        var or = stereoScope.outputRight[i];

        input += '<circle cx="' + (cx + (il - ir) * 0.5 * scale).toFixed(1) +
          '" cy="' + (cy - (il + ir) * 0.5 * scale).toFixed(1) +
          '" r="1.1" fill="#a8a8a8" fill-opacity=".42"/>';
        output += '<circle cx="' + (cx + (ol - or) * 0.5 * scale).toFixed(1) +
          '" cy="' + (cy - (ol + or) * 0.5 * scale).toFixed(1) +
          '" r="1.2" fill="' + accent + '" fill-opacity=".52"/>';
      }

      labels =
        '<text x="238" y="24" fill="#8a8a8a" font-size="8">IN</text>' +
        '<text x="278" y="24" fill="' + accent + '" font-size="8">OUT</text>';
    }

    var corrInput = correlation(stereoScope.inputLeft, stereoScope.inputRight);
    var corrOutput = correlation(stereoScope.outputLeft, stereoScope.outputRight);
    var barX = 60;
    var barY = 164;
    var barWidth = 200;
    var barHeight = 8;
    var bar =
      '<text x="' + barX + '" y="' + (barY - 6) +
        '" fill="#8a8a8a" font-size="7">CORR</text>' +
      '<text x="' + (barX + 42) + '" y="' + (barY - 6) +
        '" fill="#a8a8a8" font-size="7">IN ' + corrInput.toFixed(2) + '</text>' +
      '<text x="' + (barX + 126) + '" y="' + (barY - 6) +
        '" fill="' + accent + '" font-size="7">OUT ' + corrOutput.toFixed(2) + '</text>' +
      '<rect x="' + barX + '" y="' + barY + '" width="' + barWidth + '" height="' + barHeight +
        '" rx="4" fill="#0b0b0b" stroke="#3a3a3a"/>' +
      '<rect x="' + (barX + 1) + '" y="' + (barY + 1) + '" width="' + (barWidth - 2) +
        '" height="' + (barHeight - 2) + '" rx="3" fill="#050505"/>' +
      '<line x1="' + (barX + barWidth / 2) + '" y1="' + barY + '" x2="' + (barX + barWidth / 2) +
        '" y2="' + (barY + barHeight) + '" stroke="#3a3a3a"/>';

    bar +=
      '<circle cx="' + (barX + (corrInput + 1) * 0.5 * barWidth).toFixed(1) + '" cy="' +
        (barY + barHeight / 2) + '" r="2.2" fill="#a8a8a8"/>' +
      '<circle cx="' + (barX + (corrOutput + 1) * 0.5 * barWidth).toFixed(1) + '" cy="' +
        (barY + barHeight / 2) + '" r="2.2" fill="' + accent + '"/>' +
      '<text x="' + barX + '" y="' + (barY + 20) +
        '" fill="#6a6a6a" font-size="7">-1</text>' +
      '<text x="' + (barX + barWidth / 2 - 3) + '" y="' + (barY + 20) +
        '" fill="#6a6a6a" font-size="7">0</text>' +
      '<text x="' + (barX + barWidth - 8) + '" y="' + (barY + 20) +
        '" fill="#6a6a6a" font-size="7">+1</text>';

    return heroSvgMeter(grid + input + output + labels + bar);
  }

  /// A semitone ruler: the pitch factor is a ratio, so the picture reads it as
  /// the semitones a person actually hears.
  function pitchSvg() {
    var factor = clamp(valueOf("pitch_shift_factor", 1), 0.25, 4);
    var semitones = 12 * Math.log2(factor);
    var marker = 40 + (semitones + 24) / 48 * 440;
    var ticks = "";

    for (var step = -24; step <= 24; step += 3) {
      var x = 40 + (step + 24) / 48 * 440;
      var major = step % 12 === 0;
      ticks += '<line x1="' + x.toFixed(1) + '" y1="' + (major ? 58 : 66) + '" x2="' +
        x.toFixed(1) + '" y2="' + (major ? 102 : 94) + '" stroke="' +
        (major ? "#4a4a4a" : "#333") + '" stroke-width="1"/>';
    }

    return heroSvg(
      '<line x1="40" y1="80" x2="480" y2="80" stroke="#3a3a3a" stroke-width="1"/>' +
      ticks +
      '<line x1="' + marker.toFixed(1) + '" y1="30" x2="' + marker.toFixed(1) +
        '" y2="130" stroke="' + accent + '" stroke-width="2"/>' +
      '<circle cx="' + marker.toFixed(1) + '" cy="80" r="4" fill="' + accent + '"/>'
    );
  }

  function pitchBadge() {
    var factor = clamp(valueOf("pitch_shift_factor", 1), 0.25, 4);
    var semitones = 12 * Math.log2(factor);
    var sign = semitones > 0.05 ? "+" : "";

    return '<div class="hero-badge">' + sign + semitones.toFixed(1) + ' st</div>';
  }

  /// The loaded impulse response, as min/max columns. Before anything is loaded
  /// the hero itself is the load button, so the picture says what to do.
  function impulseSvg() {
    if (!irState || !irState.peaks || !irState.peaks.length) {
      var failed = irState && irState.error;
      var pending = irState && irState.pending;
      var message = failed
        ? irState.error
        : (pending ? "Choosing an impulse response..." : "Click to load an impulse response");
      var color = failed ? "#ff7a6b" : (pending ? accent : "#8a8a8a");

      return heroSvg(
        '<line x1="20" y1="80" x2="500" y2="80" stroke="#2c2c2c"/>'
      ) + '<div class="hero-empty" style="color:' + color + '">' +
        escapeXml(message) + '</div>';
    }

    var peaks = irState.peaks;
    var last = Math.max(peaks.length - 1, 1);
    var top = "M 20 80";
    var bottom = "";

    for (var i = 0; i < peaks.length; i++) {
      var x = 20 + i / last * 480;
      var low = clamp(peaks[i][0], -1, 1);
      var high = clamp(peaks[i][1], -1, 1);
      top += " L " + x.toFixed(1) + " " + (80 - high * 62).toFixed(1);
      bottom = " L " + x.toFixed(1) + " " + (80 - low * 62).toFixed(1) + bottom;
    }

    var seconds = irState.rate ? irState.samples / irState.rate : 0;
    var label = (irState.name || "impulse") + "  -  " + seconds.toFixed(2) + " s";

    return heroSvg(
      '<line x1="20" y1="80" x2="500" y2="80" stroke="#2f2f2f"/>' +
      '<path d="' + top + bottom + ' Z" fill="' + accent + '" fill-opacity=".14" stroke="' +
        accent + '" stroke-width="1" vector-effect="non-scaling-stroke"/>'
    ) + '<div class="hero-label" style="top:2%">' + escapeXml(label) + '</div>';
  }

  /// A frequency mapping: input to output frequency on a log/log plot at the top,
  /// and the amplitude the script returns along the bottom.
  function mappingSvg() {
    var grid =
      '<rect x="20" y="12" width="480" height="82" fill="none" stroke="#2a2a2a"/>' +
      '<line x1="20" y1="94" x2="500" y2="12" stroke="#3a3a3a" stroke-dasharray="3 5"/>' +
      '<line x1="20" y1="148" x2="500" y2="148" stroke="#2f2f2f"/>' +
      '<line x1="20" y1="116" x2="500" y2="116" stroke="#242424"/>';
    var points = curveState && curveState.points;

    if (!points || points.length < 2) {
      return heroSvg(grid) +
        '<div class="hero-empty">Waiting for the script...</div>';
    }

    var span = Math.log(1000);
    var frequencyPath = "";
    var amplitudePath = "";

    for (var i = 0; i < points.length; i++) {
      var input = clamp(points[i][0], 20, 20000);
      var mapped = clamp(points[i][1], 20, 20000);
      var amplitude = clamp(points[i][2], 0, 2);
      var x = 20 + Math.log(input / 20) / span * 480;
      var y = 94 - Math.log(mapped / 20) / span * 82;
      var amp = 148 - amplitude / 2 * 32;

      frequencyPath += (i ? " L " : "M ") + x.toFixed(1) + " " + y.toFixed(1);
      amplitudePath += (i ? " L " : "M ") + x.toFixed(1) + " " + amp.toFixed(1);
    }

    return heroSvg(
      grid +
      '<path d="' + frequencyPath + '" fill="none" stroke="' + accent + '" stroke-width="1.6" ' +
      'vector-effect="non-scaling-stroke"/>' +
      '<path d="' + amplitudePath + '" fill="none" stroke="#9a9a9a" stroke-width="1.2" ' +
      'vector-effect="non-scaling-stroke"/>'
    ) +
      '<div class="hero-label" style="top:3%">frequency in / out</div>' +
      '<div class="hero-label" style="top:68%">amplitude</div>';
  }

  function escapeXml(value) {
    return String(value === null || value === undefined ? "" : value)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  /// The scripted transfer function: a shaped curve over the input range, with
  /// the identity line for reference.
  function curveSvg() {
    var grid =
      '<line x1="20" y1="80" x2="500" y2="80" stroke="#2f2f2f"/>' +
      '<line x1="260" y1="14" x2="260" y2="146" stroke="#2a2a2a"/>' +
      '<line x1="20" y1="142" x2="500" y2="18" stroke="#3a3a3a" stroke-dasharray="3 5"/>';
    var points = curveState && curveState.points;

    if (!points || points.length < 2) {
      return heroSvg(grid) +
        '<div class="hero-empty">Waiting for the script...</div>';
    }

    var path = "";

    for (var i = 0; i < points.length; i++) {
      var x = clamp(points[i][0], -1, 1);
      var y = clamp(points[i][1], -2, 2);
      var px = 20 + (x + 1) * 0.5 * 480;
      var py = 80 - y / 2 * 62;
      path += (i ? " L " : "M ") + px.toFixed(1) + " " + py.toFixed(1);
    }

    return heroSvg(
      grid +
      '<path d="' + path + '" fill="none" stroke="' + accent + '" stroke-width="1.8" ' +
      'vector-effect="non-scaling-stroke"/>'
    ) + '<div class="hero-label" style="top:2%">in -1..1 / out -2..2</div>';
  }

  /// Before and after spectra only: a frequency shifter has no response curve to
  /// draw, and the two traces are the whole story.
  function spectraSvg() {
    if (!scope.input || !scope.output) {
      return heroSvg(
        '<line x1="20" y1="140" x2="500" y2="140" stroke="#2c2c2c" stroke-width="1"/>'
      );
    }

    var rate = face.rate || 48000;
    var inputBands = bandsOf(scope.input, rate);
    var outputBands = bandsOf(scope.output, rate);

    if (!inputBands || !outputBands) return "";

    spectra.input = smoothBands(spectra.input, inputBands);
    spectra.output = smoothBands(spectra.output, outputBands);

    return heroSvg(
      '<path d="' + bandsPath(spectra.input) + '" fill="none" stroke="#8a8a8a" ' +
        'stroke-width="1.2" opacity=".7" vector-effect="non-scaling-stroke"/>' +
      '<path d="' + bandsPath(spectra.output) + '" fill="none" stroke="' + accent + '" ' +
        'stroke-width="1.8" vector-effect="non-scaling-stroke"/>'
    );
  }

  /// A group-delay bump around the disperser's cutoff: the wider the bandwidth,
  /// the broader the region the allpass chain delays.
  function phaseSvg() {
    var cutoff = Math.max(valueOf("cutoff", 440), 10);
    var bandwidth = Math.max(valueOf("bandwidth", 1000), 10);
    var amount = clamp(valueOf("amount", 0.5), 0, 1);
    var width = Math.max(0.15, Math.abs(Math.log2(bandwidth / cutoff)) * 0.35);
    var strength = 0.35 + 0.65 * amount;
    var path = "";

    for (var i = 0; i <= 120; i++) {
      var frequency = 20 * Math.pow(1000, i / 120);
      var octaves = Math.log2(frequency / cutoff);
      var delay = 1 / (1 + Math.pow(octaves / width, 2)) * strength;
      var x = 20 + i / 120 * 480;
      var y = 150 - delay * 120;
      path += (i ? " L " : "M ") + x.toFixed(1) + " " + y.toFixed(1);
    }

    return heroSvg(
      '<line x1="20" y1="150" x2="500" y2="150" stroke="#2a2a2a"/>' +
      '<path d="' + path + ' L 500 150 L 20 150 Z" fill="' + accent + '" fill-opacity=".08"/>' +
      '<path d="' + path + '" fill="none" stroke="' + accent + '" stroke-width="2" ' +
        'vector-effect="non-scaling-stroke"/>'
    );
  }

  /// Two channel gains as meters; balance is independent left and right levels.
  function balanceSvg() {
    var left = clamp(valueOf("left_gain", 1), 0, 4);
    var right = clamp(valueOf("right_gain", 1), 0, 4);
    var base = 140;
    var top = 20;
    var full = base - top;
    var height = function (gain) { return Math.max(4, clamp(gain / 2, 0, 1) * full); };
    var lh = height(left);
    var rh = height(right);
    var unity = base - height(1);

    var meter = function (x, value) {
      return '<rect x="' + x + '" y="' + top + '" width="50" height="' + full +
        '" rx="3" fill="#242424"/>' +
        '<rect x="' + x + '" y="' + (base - value).toFixed(1) + '" width="50" height="' +
        value.toFixed(1) + '" rx="3" fill="' + accent + '" fill-opacity=".22" stroke="' +
        accent + '" stroke-width="1.4"/>';
    };

    return heroSvg(
      '<line x1="20" y1="' + unity.toFixed(1) + '" x2="500" y2="' + unity.toFixed(1) +
        '" stroke="#3a3a3a" stroke-dasharray="3 5"/>' +
      '<line x1="20" y1="' + base + '" x2="500" y2="' + base + '" stroke="#3d3d3d"/>' +
      meter(190, lh) +
      meter(280, rh)
    );
  }

  /// The delay's repeats as a short time line. Synced repeats follow the beats at
  /// 120 BPM for the picture; the audio, of course, follows the host's tempo.
  function tapsSvg() {
    var sync = !!valueOf("tempo.sync", 0);
    var beats = Math.max(valueOf("tempo.beats", 1), 0.03125);
    var free = Math.max(valueOf("delay_time", 250), 0);
    var time = sync ? beats * 500 : free;
    var feedback = clamp(valueOf("decay_factor", 0.8), 0, 0.95);
    var mix = clamp(valueOf("wet_gain", 1), 0, 2);
    var base = 140;
    var taps = "";

    for (var tap = 0; tap < 8; tap++) {
      var at = time * (tap + 1);
      if (at <= 0 || at > 2000) break;

      var x = 20 + at / 2000 * 480;
      var level = Math.pow(feedback, tap) * Math.min(mix, 1);
      var height = 16 + 100 * level;

      taps += '<line x1="' + x.toFixed(1) + '" y1="' + base + '" x2="' + x.toFixed(1) +
        '" y2="' + (base - height).toFixed(1) + '" stroke="' + accent + '" stroke-width="2" ' +
        'opacity="' + (0.3 + 0.7 * level).toFixed(2) + '"/>' +
        '<circle cx="' + x.toFixed(1) + '" cy="' + (base - height).toFixed(1) +
        '" r="2" fill="' + accent + '"/>';
    }

    return heroSvg(
      '<line x1="20" y1="' + base + '" x2="500" y2="' + base + '" stroke="#3a3a3a"/>' +
      taps
    );
  }

  /// The LFO as it is set now: the free rate in hertz, or the beat period converted
  /// to hertz at 120 BPM for the picture.
  function lfoSvg() {
    var sync = !!valueOf("tempo.sync", 0);
    var beats = Math.max(valueOf("tempo.beats", 1), 0.03125);
    var free = Math.max(valueOf("lfo_frequency", valueOf("lfo_freq", 1)), 0.01);
    var rate = clamp(sync ? 1 / (beats * 0.5) : free, 0.01, 20);
    var amplitude = valueOf("lfo_amplitude", 0);
    var depth = amplitude > 0 ? clamp(amplitude / 10, 0.1, 1) : 0.6;
    var scale = 45 * depth;
    var path = "";

    for (var i = 0; i <= 520; i++) {
      var t = i / 520 * 2.0;
      var x = 20 + i / 520 * 480;
      var y = 80 - Math.sin(t * rate * Math.PI * 2) * scale;
      path += (i ? " L " : "M ") + x.toFixed(1) + " " + y.toFixed(1);
    }

    return heroSvg(
      '<line x1="20" y1="80" x2="500" y2="80" stroke="#2f2f2f" stroke-dasharray="3 5"/>' +
      '<path d="' + path + '" fill="none" stroke="' + accent + '" stroke-width="2" ' +
        'vector-effect="non-scaling-stroke"/>'
    );
  }

  /// The FDN tail: a pre-delay gap, a few early reflections, then a T60 decay
  /// with a lighter high-frequency trace. Decay and damping drive the shape.
  function tailSvg() {
    var decay = clamp(valueOf("decay_time", 2200), 100, 10000);
    var pre = clamp(valueOf("pre_delay", 20), 0, 200);
    var size = clamp(valueOf("size", 10), 1, 50);
    var damping = clamp(valueOf("damping", 8000), 200, 20000);
    var diffusion = clamp(valueOf("diffusion", 0.7), 0, 1);
    var total = Math.max(pre + decay * 1.15, 100) * 1.02;
    var xAt = function (ms) { return 20 + ms / total * 480; };
    var yAt = function (level) { return 150 - level * 140; };
    var preX = xAt(pre);
    var ticks = "";
    var tail = "";
    var high = "";

    for (var k = 0; k < 5; k++) {
      var at = pre + size * (0.35 + k * 0.55);
      if (at >= total) break;
      var height = (0.55 - k * 0.08) * (0.65 + diffusion * 0.35);
      ticks += '<line x1="' + xAt(at).toFixed(1) + '" y1="150" x2="' + xAt(at).toFixed(1) +
        '" y2="' + yAt(height).toFixed(1) + '" stroke="' + accent +
        '" stroke-width="2" opacity="' + (0.35 + k * 0.1).toFixed(2) + '"/>';
    }

    for (var i = 0; i <= 180; i++) {
      var t = pre + (total - pre) * i / 180;
      var level = Math.exp(-6.9 * (t - pre) / decay);
      tail += (i ? " L " : "M ") + xAt(t).toFixed(1) + " " + yAt(level).toFixed(1);
    }

    var hfDecay = decay * (0.35 + 0.65 * clamp(damping / 20000, 0.1, 1));

    for (var j = 0; j <= 180; j++) {
      var ht = pre + (total - pre) * j / 180;
      var hlevel = Math.exp(-6.9 * (ht - pre) / hfDecay);
      high += (j ? " L " : "M ") + xAt(ht).toFixed(1) + " " + yAt(hlevel).toFixed(1);
    }

    return heroSvg(
      '<line x1="20" y1="150" x2="500" y2="150" stroke="#3a3a3a"/>' +
      (pre > 0
        ? '<line x1="20" y1="150" x2="' + preX.toFixed(1) + '" y2="150" stroke="#343434"/>'
        : "") +
      '<path d="' + tail + ' L ' + xAt(total).toFixed(1) + ' 150 L ' + xAt(pre).toFixed(1) +
        ' 150 Z" fill="' + accent + '" fill-opacity=".07"/>' +
      '<path d="' + tail + '" fill="none" stroke="' + accent + '" stroke-width="2" ' +
        'vector-effect="non-scaling-stroke"/>' +
      '<path d="' + high + '" fill="none" stroke="#ffffff" stroke-opacity=".22" ' +
        'stroke-width="1" vector-effect="non-scaling-stroke"/>' +
      ticks
    ) + (pre > 0 ? '<div class="hero-label" style="left:4.6%;top:88%">pre</div>' : "");
  }

  /// Left: the transfer curve on a square that stays left. Right: a scrolling
  /// before/after level envelope that fills the rest. The wheel over the right
  /// panel changes the visible time window, which changes how fast it moves.
  function compressorSvg() {
    var threshold = clamp(valueOf("threshold", -24), -60, 0);
    var ratio = Math.max(valueOf("ratio", 4), 1);
    var makeup = valueOf("makeup_db", 0);
    var sidechainOn = !!valueOf("sidechain", 0);
    var level = sidechainOn ? meter.sidechain : meter.input;
    var path = "";

    for (var i = 0; i <= 120; i++) {
      var input = -60 + i / 120 * 60;
      var output = input <= threshold ? input : threshold + (input - threshold) / ratio;
      output = clamp(output + makeup, -60, 0);
      var x = 8 + i / 120 * 124;
      var y = 132 - (output + 60) / 60 * 124;
      path += (i ? " L " : "M ") + x.toFixed(1) + " " + y.toFixed(1);
    }

    var knee = 8 + (threshold + 60) / 60 * 124;
    var marker = "";

    if (level !== null && level !== undefined && isFinite(level)) {
      var db = clamp(level, -60, 0);
      var mx = 8 + (db + 60) / 60 * 124;
      var out = db <= threshold ? db : threshold + (db - threshold) / ratio;
      out = clamp(out + makeup, -60, 0);
      var my = 132 - (out + 60) / 60 * 124;
      var tag = sidechainOn ? "SC " : "IN ";
      var textX = mx > 92 ? mx - 4 : mx + 4;
      var anchor = mx > 92 ? "end" : "start";

      marker =
        '<line x1="' + mx.toFixed(1) + '" y1="8" x2="' + mx.toFixed(1) + '" y2="132" stroke="' + accent + '" stroke-opacity=".22" stroke-dasharray="2 4"/>' +
        '<line x1="8" y1="' + my.toFixed(1) + '" x2="132" y2="' + my.toFixed(1) + '" stroke="' + accent + '" stroke-opacity=".12" stroke-dasharray="2 4"/>' +
        '<circle cx="' + mx.toFixed(1) + '" cy="' + my.toFixed(1) + '" r="2.4" fill="' + accent + '"/>' +
        '<text x="' + textX.toFixed(1) + '" y="17" fill="' + accent + '" font-size="7" text-anchor="' + anchor + '">' + tag + db.toFixed(1) + ' dB</text>';
    }

    var left =
      '<svg viewBox="0 0 140 140" preserveAspectRatio="xMidYMid meet">' +
        '<rect x="8" y="8" width="124" height="124" fill="none" stroke="#2a2a2a"/>' +
        '<line x1="8" y1="132" x2="132" y2="8" stroke="#2f2f2f" stroke-dasharray="3 5"/>' +
        '<line x1="' + knee.toFixed(1) + '" y1="8" x2="' + knee.toFixed(1) + '" y2="132" stroke="#343434" stroke-dasharray="3 5"/>' +
        '<line x1="8" y1="132" x2="132" y2="132" stroke="#3a3a3a"/>' +
        '<path d="' + path + '" fill="none" stroke="' + accent + '" stroke-width="2" vector-effect="non-scaling-stroke"/>' +
        marker +
      '</svg>';

    var right = "";
    var count = Math.min(envWindow, envHistory.input.length);

    if (count > 1) {
      var loudest = 1e-6;

      for (var k = 0; k < count; k++) {
        var index = envHistory.input.length - count + k;
        loudest = Math.max(loudest, envHistory.input[index], envHistory.output[index]);
      }

      var inPath = "";
      var outPath = "";

      for (var s = 0; s < count; s++) {
        var idx = envHistory.input.length - count + s;
        var px = 8 + s / (count - 1) * 384;
        var iy = 132 - envHistory.input[idx] / loudest * 124;
        var oy = 132 - envHistory.output[idx] / loudest * 124;
        inPath += (s ? " L " : "M ") + px.toFixed(1) + " " + iy.toFixed(1);
        outPath += (s ? " L " : "M ") + px.toFixed(1) + " " + oy.toFixed(1);
      }

      right =
        '<path d="' + outPath + ' L 392 132 L 8 132 Z" fill="' + accent + '" fill-opacity=".06"/>' +
        '<path d="' + inPath + '" fill="none" stroke="#8a8a8a" stroke-width="1.1" opacity=".65" vector-effect="non-scaling-stroke"/>' +
        '<path d="' + outPath + '" fill="none" stroke="' + accent + '" stroke-width="1.6" vector-effect="non-scaling-stroke"/>';
    } else {
      right = '<line x1="8" y1="132" x2="392" y2="132" stroke="#3a3a3a"/>';
    }

    var seconds = (envWindow * 0.05).toFixed(1);

    return '<div class="compressor-hero">' +
      '<div class="transfer-box">' + left + '</div>' +
      '<div class="env-box" title="Scroll to change the envelope time window">' +
        '<svg viewBox="0 0 400 140" preserveAspectRatio="none">' + right + '</svg>' +
        (count > 1
          ? '<div class="hero-label" style="left:4%;top:2%">IN</div>' +
            '<div class="hero-label" style="left:13%;top:2%;color:' + accent + '">OUT</div>'
          : "") +
        '<div class="hero-badge">' + seconds + ' s</div></div>' +
      '</div>';
  }

  function updateHeroes() {
    heroes.forEach(function (hero) {
      if (hero.kind === "compressor") hero.el.innerHTML = compressorSvg();
      else if (hero.kind === "tail") {
        var seconds = clamp(valueOf("decay_time", 2200), 100, 10000) / 1000;
        hero.el.innerHTML = tailSvg() +
          '<div class="hero-badge">T60 ' + seconds.toFixed(1) + ' s</div>';
      }
      else if (hero.kind === "lfo") {
        hero.el.innerHTML = lfoSvg() +
          (valueOf("tempo.sync", 0) ? '<div class="hero-badge">SYNC</div>' : "");
      }
      else if (hero.kind === "taps") {
        hero.el.innerHTML = tapsSvg() +
          (valueOf("tempo.sync", 0) ? '<div class="hero-badge">SYNC</div>' : "");
      }
      else if (hero.kind === "waveform") hero.el.innerHTML = waveformSvg();
      else if (hero.kind === "transfer") hero.el.innerHTML = transferSvg() + rmsBadge();
      else if (hero.kind === "spectrum") hero.el.innerHTML = spectrumSvg();
      else if (hero.kind === "spectra") hero.el.innerHTML = spectraSvg();
      else if (hero.kind === "phase") hero.el.innerHTML = phaseSvg();
      else if (hero.kind === "balance") hero.el.innerHTML = balanceSvg();
      else if (hero.kind === "stereo") hero.el.innerHTML = stereoSvg();
  else if (hero.kind === "particles") hero.el.innerHTML = particlesSvg();
      else if (hero.kind === "pitch") hero.el.innerHTML = pitchSvg() + pitchBadge();
      else if (hero.kind === "impulse") hero.el.innerHTML = impulseSvg();
      else if (hero.kind === "curve") hero.el.innerHTML = curveSvg();
      else if (hero.kind === "mapping") hero.el.innerHTML = mappingSvg();
      else if (hero.kind === "ott") hero.el.innerHTML = ottSvg();
      else if (hero.kind === "vocodex") hero.el.innerHTML = vocoderSvg();
    });
  }

  function markDirty() {
    if (dirty) return;
    dirty = true;
    requestAnimationFrame(function () {
      dirty = false;
      updateHeroes();
    });
  }

  // -------------------------------------------------------------------- poll
  function poll() {
    if (polling) return;
    polling = true;

    if (document.hidden) {
      polling = false;
      setTimeout(poll, 250);
      return;
    }

    fetch("/state", { cache: "no-store" })
      .then(function (response) {
        if (!response.ok) throw new Error("state " + response.status);
        return response.json();
      })
      .then(function (state) {
        if (typeof state.__bypass !== "undefined") setBypassed(state.__bypass);

        Object.keys(state).forEach(function (id) {
          if (id === "__bypass") return;

          var control = controls[id];
          if (control && control !== dragging) {
            control.value = control.spec.kind === "bool"
              ? state[id]
              : snap(control.spec, state[id]);
            control.paint();
          }
        });
        applySwitches();
        markDirty();
      })
      .catch(function (error) {
        console.warn("i_am_web_gui: " + error);
      })
      .then(function () {
        polling = false;
        setTimeout(poll, 60);
      });
  }

  function levelOf(samples, start, count) {
    var energy = 0;

    for (var i = 0; i < count; i++) {
      var sample = samples[start + i];
      energy += sample * sample;
    }

    return Math.sqrt(energy / Math.max(count, 1));
  }

  // Fill the history from the scope the first time, so the envelope has a
  // complete picture before the first few live frames arrive.
  function seedEnvelope(input, output) {
    if (envHistory.input.length) return;

    var segments = 128;
    var step = Math.floor(input.length / segments);

    if (step < 1) step = 1;

    for (var s = 0; s < segments; s++) {
      var start = Math.min(s * step, input.length - step);
      envHistory.input.push(levelOf(input, start, step));
      envHistory.output.push(levelOf(output, start, step));
    }
  }

  function pushEnvelope(input, output) {
    var step = Math.min(1024, input.length);
    var start = input.length - step;

    if (envHistory.input.length >= ENV_MAX) {
      envHistory.input.shift();
      envHistory.output.shift();
    }

    envHistory.input.push(levelOf(input, start, step));
    envHistory.output.push(levelOf(output, start, step));
  }

  // The scope is its own poll: it is a binary document, and a host with no audio
  // running still wants the rest of the page to keep working.
  function pollScope() {
    if (document.hidden || !wantsScope) {
      setTimeout(pollScope, 150);
      return;
    }

    fetch("/scope", { cache: "no-store" })
      .then(function (response) {
        if (!response.ok) throw new Error("scope " + response.status);
        return response.arrayBuffer();
      })
      .then(function (buffer) {
        var data = new Float32Array(buffer);
        var half = data.length / 2;

        if (half > 64) {
          scope.input = data.subarray(0, half);
          scope.output = data.subarray(half);
          seedEnvelope(scope.input, scope.output);
          pushEnvelope(scope.input, scope.output);
          markDirty();
        }
      })
      .catch(function () {})
      .then(function () {
        setTimeout(pollScope, 50);
      });
  }

  /// The note the host last sent, shown beside the title. A product without
  /// note input answers with -1 and the badge stays hidden.
  function pollNote() {
    if (document.hidden || !noteBadge) {
      setTimeout(pollNote, 250);
      return;
    }

    fetch("/note", { cache: "no-store" })
      .then(function (response) {
        return response.ok ? response.json() : null;
      })
      .then(function (data) {
        if (data && data.note >= 0) {
          var names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"];
          var index = ((Math.round(data.note) % 12) + 12) % 12;
          var octave = Math.floor(Math.round(data.note) / 12) - 1;
          noteBadge.textContent = names[index] + octave;
          noteBadge.style.display = "";
        } else {
          noteBadge.style.display = "none";
        }
      })
      .catch(function () {
        noteBadge.style.display = "none";
      })
      .then(function () {
        setTimeout(pollNote, 100);
      });
  }

  /// The loaded impulse response, with its picture. The product owns /ir; the
  /// preview server does not, so a failed fetch just leaves the empty hero.
  function pollIr() {
    if (document.hidden || !wantsIr) {
      setTimeout(pollIr, 500);
      return;
    }

    fetch("/ir", { cache: "no-store" })
      .then(function (response) {
        return response.ok ? response.json() : null;
      })
      .then(function (value) {
        if (!value) return;
        var changed = JSON.stringify(value) !== JSON.stringify(irState);
        irState = value;
        if (changed) markDirty();
      })
      .catch(function () {})
      .then(function () {
        setTimeout(pollIr, 500);
      });
  }

  /// Hands the picked file to the product's upload route as it is. The Rust side
  /// decodes and resamples it, so the page never has to parse an audio format.
  function uploadIr() {
    var file = irInput && irInput.files && irInput.files[0];
    if (!file) return;

    var reader = new FileReader();
    reader.onload = function () {
      fetch("/ir?name=" + encodeURIComponent(file.name), {
        method: "POST",
        body: reader.result
      })
        .then(function (response) {
          return response.ok ? response.json() : null;
        })
        .then(function (result) {
          if (result && result.error) {
            irState = { name: null, rate: 0, samples: 0, peaks: [], error: result.error };
            markDirty();
          }
        })
        .catch(function () {});
    };
    reader.readAsArrayBuffer(file);
    irInput.value = "";
  }

  /// The vocoder bank: the modulator bands above the centre line and the output
  /// bands below it, one pair per band.
  function vocoderSvg() {
    var state = vocoderState;
    var bands = state && state.bands ? Math.min(state.bands, 128) : 24;
    var input = state && state.input ? state.input : [];
    var output = state && state.output ? state.output : [];
    var grid = '<line x1="20" y1="80" x2="500" y2="80" stroke="#2f2f2f"/>';
    var bars = "";
    var slot = 480 / bands;
    // The bars widen with the same Bandwidth multiplier the filters use, so
    // the picture shows how much the bands overlap.
    var bandwidth = clamp(valueOf("bandwidth", 1.0), 0.25, 4.0);
    var bar_width = Math.max(1, slot * bandwidth - 1);

    for (var i = 0; i < bands; i++) {
      var x = 20 + (i + 0.5) * slot - bar_width / 2;
      var in_height = Math.sqrt(clamp(input[i] || 0, 0, 1)) * 60;
      var out_height = Math.sqrt(clamp(output[i] || 0, 0, 1)) * 60;

      bars += '<rect x="' + x.toFixed(1) + '" y="' + (80 - in_height).toFixed(1) +
        '" width="' + bar_width.toFixed(1) + '" height="' + in_height.toFixed(1) +
        '" fill="#8a8a8a" fill-opacity=".5"/>';
      bars += '<rect x="' + x.toFixed(1) + '" y="80" width="' + bar_width.toFixed(1) +
        '" height="' + out_height.toFixed(1) + '" fill="' + accent +
        '" fill-opacity=".65"/>';
    }

    var source = valueOf("sidechain", 1) ? "SIDECHAIN" : "SELF";

    return heroSvg(grid + bars) +
      '<div class="hero-label" style="left:2%;top:3%">IN ' + source + '</div>' +
      '<div class="hero-label" style="left:2%;top:55%">OUTPUT</div>';
  }

  function pollVocoder() {
    if (!wantsVocoder || document.hidden) {
      setTimeout(pollVocoder, 60);
      return;
    }

    fetch("/bands", { cache: "no-store" })
      .then(function (response) {
        return response.ok ? response.json() : null;
      })
      .then(function (state) {
        if (!state) return;
        var changed = JSON.stringify(state) !== JSON.stringify(vocoderState);
        vocoderState = state;
        if (changed) markDirty();
      })
      .catch(function () {})
      .then(function () {
        setTimeout(pollVocoder, 60);
      });
  }

  /// The OTT band meters, in dB of net gain: positive is attenuation, negative
  /// is the upward compressor's boost.
  function ottSvg() {
    var bands = ottState && ottState.bands ? ottState.bands : [0, 0, 0];
    var grid = '<line x1="260" y1="10" x2="260" y2="150" stroke="#2f2f2f"/>';
    var bars = "";

    for (var i = 0; i < 3; i++) {
      var y = 30 + i * 45;
      var db = bands[i] || 0;
      var x = db >= 0
        ? 260 + Math.min(db, 60) / 60 * 230
        : 260 + Math.max(db, -24) / 24 * 100;
      var left = Math.min(260, x);
      var width = Math.abs(x - 260);
      var color = db >= 0 ? accent : negative;

      bars += '<rect x="' + left.toFixed(1) + '" y="' + (y - 7) +
        '" width="' + width.toFixed(1) + '" height="14" fill="' + color +
        '" fill-opacity=".55" rx="2"/>';
    }

    return heroSvg(grid + bars) +
      '<div class="hero-label" style="left:2%;top:12%">LOW</div>' +
      '<div class="hero-label" style="left:2%;top:40%">MID</div>' +
      '<div class="hero-label" style="left:2%;top:68%">HIGH</div>';
  }

  function pollOtt() {
    if (!wantsOtt || document.hidden) {
      setTimeout(pollOtt, 60);
      return;
    }

    fetch("/bands", { cache: "no-store" })
      .then(function (response) {
        return response.ok ? response.json() : null;
      })
      .then(function (state) {
        if (!state) return;
        var changed = JSON.stringify(state) !== JSON.stringify(ottState);
        ottState = state;
        if (changed) markDirty();
      })
      .catch(function () {})
      .then(function () {
        setTimeout(pollOtt, 60);
      });
  }

  /// The script panel's text and status. The worker owns the script; the page
  /// only shows what it reports.
  function pollScript() {
    if (!wantsScript || document.hidden) {
      setTimeout(pollScript, 500);
      return;
    }

    var epoch = scriptEpoch;

    fetch("/script", { cache: "no-store" })
      .then(function (response) {
        return response.ok ? response.json() : null;
      })
      .then(function (state) {
        if (!state) return;

        // A poll that started before an edit or an apply is stale: writing its
        // source back would undo what the person just typed. The next poll reads
        // the applied source.
        if (epoch !== scriptEpoch) return;

        scriptState = state;

        if (scriptArea && !scriptDirty && document.activeElement !== scriptArea) {
          scriptArea.value = state.source || "";
        }

        if (scriptError) scriptError.textContent = state.error || "";

        if (scriptStatus) {
          scriptStatus.textContent = state.error
            ? "error"
            : "applied " + (state.applied || 0);
        }
      })
      .catch(function () {})
      .then(function () {
        setTimeout(pollScript, 500);
      });
  }

  function applyScript() {
    if (!scriptArea) return;

    if (scriptStatus) scriptStatus.textContent = "applying...";
    if (scriptError) scriptError.textContent = "";

    var source = scriptArea.value;
    scriptDirty = false;
    scriptEpoch += 1;

    fetch("/script", { method: "POST", body: source })
      .then(function (response) {
        return response.ok ? response.json() : null;
      })
      .then(function (result) {
        if (result && result.error && scriptError) scriptError.textContent = result.error;
        pollScript();
      })
      .catch(function () {});
  }

  function buildScriptPanel() {
    var panel = element("details", "script-panel");
    var summary = element("summary", "");
    summary.textContent = "Script (Rhai)";

    scriptArea = document.createElement("textarea");
    scriptArea.className = "script-area";
    scriptArea.spellcheck = false;
    scriptArea.rows = 7;
    scriptArea.addEventListener("input", function () {
      scriptDirty = true;
      scriptEpoch += 1;
    });

    var actions = element("div", "script-actions");
    var apply = element("button", "script-apply");
    apply.type = "button";
    apply.textContent = "Apply";
    apply.addEventListener("click", applyScript);

    scriptStatus = element("span", "script-status");
    scriptError = element("div", "script-error");

    actions.appendChild(apply);
    actions.appendChild(scriptStatus);
    panel.appendChild(summary);
    panel.appendChild(scriptArea);
    panel.appendChild(actions);
    panel.appendChild(scriptError);

    return panel;
  }

  /// The curve the worker rendered, downsampled by it for the picture.
  function pollCurve() {
    if (!wantsCurve || document.hidden) {
      setTimeout(pollCurve, 400);
      return;
    }

    fetch("/curve", { cache: "no-store" })
      .then(function (response) {
        return response.ok ? response.json() : null;
      })
      .then(function (state) {
        if (!state) return;
        var changed = JSON.stringify(state) !== JSON.stringify(curveState);
        curveState = state;
        if (changed) markDirty();
      })
      .catch(function () {})
      .then(function () {
        setTimeout(pollCurve, 400);
      });
  }

  // The goniometer needs L/R before and after the effect; the mono /scope is not
  // enough for it, so it has its own small poll.
  function pollStereo() {
    if (document.hidden || !wantsStereo) {
      setTimeout(pollStereo, 150);
      return;
    }

    fetch("/stereo", { cache: "no-store" })
      .then(function (response) {
        if (!response.ok) throw new Error("stereo " + response.status);
        return response.arrayBuffer();
      })
      .then(function (buffer) {
        var data = new Float32Array(buffer);
        var points = data.length / 4;

        if (points > 64) {
          stereoScope.inputLeft = data.subarray(0, points);
          stereoScope.inputRight = data.subarray(points, points * 2);
          stereoScope.outputLeft = data.subarray(points * 2, points * 3);
          stereoScope.outputRight = data.subarray(points * 3);
          markDirty();
        }
      })
      .catch(function () {})
      .then(function () {
        setTimeout(pollStereo, 50);
      });
  }

  // The level meter is tiny JSON, polled alongside the scope for the transfer
  // marker. A plug-in without one just answers 404.
  function pollMeter() {
    if (document.hidden || !wantsMeter) return;

    fetch("/meter", { cache: "no-store" })
      .then(function (response) {
        if (!response.ok) throw new Error("meter " + response.status);
        return response.json();
      })
      .then(function (value) {
        meter.input = typeof value.input === "number" ? value.input : null;
        meter.sidechain = typeof value.sidechain === "number" ? value.sidechain : null;
        markDirty();
      })
      .catch(function () {});
  }

  // ------------------------------------------------------------------- build
  function setBypassed(value) {
    bypassed = !!value;
    app.classList.toggle("bypassed", bypassed);
    if (powerButton) powerButton.setAttribute("aria-pressed", bypassed ? "true" : "false");
  }

  function toggleBypass() {
    setBypassed(!bypassed);
    if (window.ipc) window.ipc.postMessage("__bypass=" + (bypassed ? 1 : 0));
  }

  /// Show the controls the current mode has. A mode with no entry shows all.
  function applyVisibility(mode) {
    if (!face || !face.visible) return;

    var ids = null;

    for (var i = 0; i < face.visible.length; i++) {
      if (face.visible[i][0] === mode) {
        ids = face.visible[i][1];
        break;
      }
    }

    Object.keys(controls).forEach(function (id) {
      var control = controls[id];
      if (!control.el || id === face.mode) return;

      var show = !ids || ids.indexOf(id) >= 0;
      control.el.style.display = show ? "" : "none";
    });
  }

  /// Show the branch the current switch selects. Only the listed controls move;
  /// everything else is left to the mode's visibility.
  function applySwitches() {
    if (!face || !face.switches) return;

    face.switches.forEach(function (row) {
      var control = controls[row.control];
      if (!control) return;

      var on = !!control.value;
      var show = on ? row.on : row.off;
      var hide = on ? row.off : row.on;

      hide.forEach(function (id) {
        var target = controls[id];
        if (target && target.el) target.el.style.display = "none";
      });

      show.forEach(function (id) {
        var target = controls[id];
        if (target && target.el) target.el.style.display = "";
      });
    });
  }


  /// A choice in the header: a label and a drop-down, without taking a row
  /// from the faceplate.
  function buildHeaderChoice(spec) {
    var el = element("div", "choice");
    var label = element("span", "label");
    label.textContent = spec.label || "";

    var select = document.createElement("select");
    select.className = "mode-select";
    select.setAttribute("aria-label", spec.label || "Choice");

    var control = { spec: spec, value: spec.default, el: el, paint: function () {} };
    controls[spec.id] = control;

    (spec.choices || []).forEach(function (name, index) {
      var option = document.createElement("option");
      option.value = String(index);
      option.textContent = name;
      select.appendChild(option);
    });

    select.addEventListener("change", function () {
      control.value = parseInt(select.value, 10);
      control.paint();
      markDirty();
      if (window.ipc) window.ipc.postMessage(spec.id + "=" + control.value);
    });

    control.paint = function () {
      select.value = String(Math.round(control.value));
    };

    el.appendChild(label);
    el.appendChild(select);
    control.paint();
    return el;
  }

  function buildMode(spec) {
    var el = element("div", "mode");
    var buttons = [];

    var control = { spec: spec, value: spec.default, el: null, paint: function () {} };
    controls[spec.id] = control;

    var choices = spec.choices || [];

    // A long switch is a drop-down: six segments would either squeeze the title
    // or take a row of their own, and neither is worth the space.
    if (choices.length > 3) {
      el.classList.add("mode-drop");

      var select = document.createElement("select");
      select.className = "mode-select";
      select.setAttribute("aria-label", spec.label || "Mode");

      choices.forEach(function (name, index) {
        var option = document.createElement("option");
        option.value = String(index);
        option.textContent = name;
        select.appendChild(option);
      });

      select.addEventListener("change", function () {
        control.value = parseInt(select.value, 10);
        control.paint();
        if (spec.id === face.mode) applyVisibility(control.value);
        markDirty();
        if (window.ipc) window.ipc.postMessage(spec.id + "=" + control.value);
      });

      control.el = el;
      control.paint = function () {
        var active = Math.round(control.value);
        select.value = String(active);
        if (spec.id === face.mode) applyVisibility(active);
      };

      el.appendChild(select);
      control.paint();
      return el;
    }

    control.paint = function () {
      var active = Math.round(control.value);

      buttons.forEach(function (button, index) {
        button.classList.toggle("active", index === active);
        button.setAttribute("aria-pressed", index === active ? "true" : "false");
      });

      if (spec.id === face.mode) applyVisibility(active);
    };

    (spec.choices || []).forEach(function (name, index) {
      var button = element("button", "mode-choice");
      button.type = "button";
      button.textContent = name;
      button.addEventListener("click", function () {
        control.value = index;
        control.paint();
        if (spec.id === face.mode) applyVisibility(index);
        markDirty();
        if (window.ipc) window.ipc.postMessage(spec.id + "=" + index);
      });
      buttons.push(button);
      el.appendChild(button);
    });

    control.el = el;
    control.paint();
    return el;
  }

  function buildFace() {
    knobs = [];
    isPreview = !face.hosted;
    app.classList.toggle("hosted", !isPreview);
    accent = face.accent || accent;
    negative = mixHex(accent, "#9a9a9a", 0.5);

    var root = document.documentElement.style;
    root.setProperty("--accent", accent);
    root.setProperty("--negative", negative);
    document.title = face.title;

    if (isPreview) {
      document.body.classList.add("preview");
      app.style.width = face.size[0] + "px";
      app.style.height = face.size[1] + "px";
    }

    // A mode switch and the header's extra choices are parameters drawn in the
    // header, not knobs in the body.
    var headerIds = face.header || [];
    var headerSpecs = [];

    if (face.mode || headerIds.length) {
      face.sections.forEach(function (section) {
        section.params = section.params.filter(function (spec) {
          if (face.mode && spec.id === face.mode) {
            modeSpec = spec;
            return false;
          }

          if (headerIds.indexOf(spec.id) >= 0) {
            headerSpecs.push(spec);
            return false;
          }

          return true;
        });
      });

      face.sections = face.sections.filter(function (section) {
        return section.params.length > 0 || section.hero;
      });
    }

    // The downsampler cannot hold samples faster than the host is running; the
    // DSP clamps it, so the knob must not offer a range the effect cannot use.
    if (face.rate) {
      face.sections.forEach(function (section) {
        section.params.forEach(function (spec) {
          if (spec.id === "target_sample_rate" && spec.max > face.rate) {
            spec.max = face.rate;
          }
        });
      });
    }

    var header = element("header", "header");
    powerButton = element("button", "power");
    powerButton.type = "button";
    powerButton.title = "Bypass";
    powerButton.setAttribute("aria-label", "Bypass");
    powerButton.setAttribute("aria-pressed", "false");
    powerButton.appendChild(element("span", "dot"));
    powerButton.addEventListener("click", toggleBypass);

    var title = element("span", "title");
    title.textContent = face.title;

    header.appendChild(powerButton);
    header.appendChild(title);

    noteBadge = element("span", "note-badge");
    noteBadge.style.display = "none";
    header.appendChild(noteBadge);

    header.appendChild(element("span", "spacer"));

    var tools = element("div", "header-tools");
    var modeEl = modeSpec ? buildMode(modeSpec) : null;

    if (modeEl) tools.appendChild(modeEl);

    headerSpecs.forEach(function (spec) {
      if (spec.widget === "choice") {
        tools.appendChild(buildHeaderChoice(spec));
      } else if (spec.kind === "bool") {
        var dot = makeControl(spec, true);
        var wrap = element("div", "header-switch");
        var caption = element("span", "header-caption");
        caption.textContent = spec.label;
        wrap.appendChild(dot);
        wrap.appendChild(caption);
        tools.appendChild(wrap);
      } else {
        tools.appendChild(makeControl(spec, true));
      }
    });

    if (face.clear) {
      var clearTail = element("button", "clear-tail");
      clearTail.type = "button";
      clearTail.textContent = "CLEAR";
      clearTail.title = "Stop the tail immediately";
      clearTail.addEventListener("click", function () {
        fetch("/clear", { method: "POST" }).catch(function () {});
        clearTail.classList.add("cleared");
        setTimeout(function () {
          clearTail.classList.remove("cleared");
        }, 160);
      });
      tools.appendChild(clearTail);
    }

    if (modeEl || headerSpecs.length || face.clear) header.appendChild(tools);

    var body = element("div", "body");
    var many = face.sections.length > 1;
    var usedHeroes = {};

    face.sections.forEach(function (section) {
      var el = element("section", "section");

      if (many && section.name) {
        var caption = element("div", "caption");
        caption.textContent = section.name;
        el.appendChild(caption);
      }

      // One scope per page is enough: a nested effect can ask for the same hero
      // in a second section (the crusher's gain stage embeds Gain), and drawing
      // the same live trace twice only costs space.
      if (section.hero && !usedHeroes[section.hero]) {
        usedHeroes[section.hero] = true;
        var hero = element("div", "hero");
        hero.dataset.hero = section.hero;

        if (section.hero === "impulse") {
          hero.title = "Load an impulse response";
          hero.addEventListener("click", function () {
            // In a host, ask Rust for a native dialog owned by the DAW. The
            // browser's own file input has no owner there, which is why its
            // picker can open behind the window or appear not to open at all.
            // The preview server has no IPC, so it uses the input instead.
            if (window.ipc && window.ipc.postMessage) {
              window.ipc.postMessage("__load_ir");
              return;
            }

            if (irInput) irInput.click();
          });
        }

        if (section.hero === "compressor") {
          hero.addEventListener("wheel", function (event) {
            if (!event.target.closest(".env-box")) return;
            event.preventDefault();
            envWindow = clamp(
              Math.round(envWindow * (event.deltaY > 0 ? 1.25 : 0.8)),
              32,
              ENV_MAX
            );
            markDirty();
          }, { passive: false });
        }

        heroes.push({ el: hero, kind: section.hero });
        el.appendChild(hero);
      } else {
        el.classList.add("plain");
      }

      var row = element("div", "controls");
      section.params.forEach(function (spec) {
        row.appendChild(makeControl(spec));
      });
      el.appendChild(row);
      body.appendChild(el);
    });

    if (modeSpec) {
      var initialMode = Math.round(controls[modeSpec.id].value);
      applyVisibility(initialMode);

      // applyVisibility also covers the header, but re-apply it directly here so
      // a header choice built just before the body is hidden on the first paint.
      if (face.visible) {
        var initialIds = null;
        face.visible.forEach(function (row) {
          if (row[0] === initialMode) initialIds = row[1];
        });
        headerSpecs.forEach(function (spec) {
          var control = controls[spec.id];
          if (control && control.el && initialIds && initialIds.indexOf(spec.id) < 0) {
            control.el.style.display = "none";
          }
        });
      }
    }
    applySwitches();
    setTimeout(function () {
      if (modeSpec) applyVisibility(Math.round(controls[modeSpec.id].value));
    }, 0);
    setTimeout(pollNote, 100);

    // Every hero the page draws is live: a waveform, a transfer with its RMS
    // bands, or a response with the input and output spectra on it.
    wantsScope = heroes.some(function (hero) {
      return hero.kind === "waveform" || hero.kind === "transfer" ||
        hero.kind === "spectrum" || hero.kind === "spectra" ||
        hero.kind === "compressor";
    });
    wantsMeter = heroes.some(function (hero) {
      return hero.kind === "compressor";
    });
    wantsStereo = heroes.some(function (hero) {
      return hero.kind === "stereo";
    });
    wantsParticles = heroes.some(function (hero) {
      return hero.kind === "particles";
    });
    wantsIr = heroes.some(function (hero) {
      return hero.kind === "impulse";
    });
    if (wantsIr) {
      irInput = document.createElement("input");
      irInput.type = "file";
      irInput.accept = ".wav,.flac,.mp3,.ogg,audio/*";
      irInput.style.display = "none";
      irInput.addEventListener("change", uploadIr);
      document.body.appendChild(irInput);
      setTimeout(pollIr, 50);
    }
    wantsCurve = heroes.some(function (hero) {
      return hero.kind === "curve" || hero.kind === "mapping";
    });
    if (wantsCurve) setTimeout(pollCurve, 50);
    wantsOtt = heroes.some(function (hero) {
      return hero.kind === "ott";
    });
    if (wantsOtt) setTimeout(pollOtt, 50);
    wantsVocoder = heroes.some(function (hero) {
      return hero.kind === "vocodex";
    });
    if (wantsVocoder) setTimeout(pollVocoder, 50);
    if (face.script) {
      body.appendChild(buildScriptPanel());
      wantsScript = true;
      setTimeout(pollScript, 50);
    }
    if (wantsMeter) setInterval(pollMeter, 100);
    if (wantsStereo) setTimeout(pollStereo, 50);
    if (wantsParticles) {
      setInterval(function () {
        if (!document.hidden && wantsParticles) markDirty();
      }, 40);
    }

    var shell = element("div", "face");
    shell.appendChild(header);
    shell.appendChild(body);
    app.appendChild(shell);

    applyScale();
    markDirty();
    poll();
    if (wantsScope) pollScope();
  }

  /// Size the faceplate to the window: a bigger host window gets bigger knobs,
  /// labels and margins, not just a bigger picture area.
  function applyScale() {
    if (!face) return;

    // The card, not the window: in a host the faceplate is inset, and the
    // controls have to fit the card they sit on.
    var card = app.querySelector(".face");
    var width = (card ? card.clientWidth : app.clientWidth) || face.size[0];
    var height = (card ? card.clientHeight : app.clientHeight) || face.size[1];
    scale = clamp(Math.min(width / face.size[0], height / face.size[1]), 0.8, 1.9);
    document.documentElement.style.setProperty("--scale", scale);
    knobs.forEach(function (resize) { resize(scale); });
  }

  window.addEventListener("resize", function () {
    requestAnimationFrame(applyScale);
  });

  // A host resizes the WebView2 child, not the browser window: the window event
  // is not guaranteed there, but the app's own box always changes.
  if (window.ResizeObserver) {
    new ResizeObserver(function () {
      requestAnimationFrame(applyScale);
    }).observe(app);
  }

  fetch("/face", { cache: "no-store" })
    .then(function (response) {
      if (!response.ok) throw new Error("face " + response.status);
      return response.json();
    })
    .then(function (page) {
      face = page;
      buildFace();
    })
    .catch(function (error) {
      app.textContent = "faceplate unavailable";
      console.error("i_am_web_gui: " + error);
    });
})();