(function () {
  "use strict";

  var app = document.getElementById("app");
  var isPreview = location.protocol !== "webgui:";
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
  var scope = { input: null, output: null };

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
      case "hz": return value >= 1000 ? (value / 1000).toFixed(2) + " kHz" : value.toFixed(0) + " Hz";
      case "percent": return (value * 100).toFixed(1) + " %";
      case "ms": return value.toFixed(1) + " ms";
      case "s": return value.toFixed(3) + " s";
      case "samples": return value.toFixed(0);
      case "beats": return beats(value);
      default: return spec.kind === "int" ? String(Math.round(value)) : value.toFixed(2);
    }
  }

  // ------------------------------------------------------------ value space
  function norm(spec, value) {
    if (spec.max === spec.min) return 0;
    if (spec.log && spec.min > 0) return Math.log(value / spec.min) / Math.log(spec.max / spec.min);
    return (value - spec.min) / (spec.max - spec.min);
  }

  function at(spec, position) {
    position = clamp(position, 0, 1);
    if (spec.log && spec.min > 0) return spec.min * Math.pow(spec.max / spec.min, position);
    return spec.min + position * (spec.max - spec.min);
  }

  function snap(spec, value) {
    if (spec.step && spec.step > 0) value = Math.round(value / spec.step) * spec.step;
    return clamp(value, spec.min, spec.max);
  }

  function setValue(control, value, send) {
    value = snap(control.spec, value);
    control.value = value;
    control.paint();
    markDirty();
    if (send && window.ipc) window.ipc.postMessage(control.spec.id + "=" + value);
  }

  // --------------------------------------------------------------- controls
  function makeControl(spec) {
    var control = {
      spec: spec,
      value: spec.default,
      paint: function () {}
    };

    controls[spec.id] = control;

    if (spec.widget === "toggle" || spec.kind === "bool") return buildToggle(control);
    if (spec.widget === "choice") return buildChoice(control);
    if (spec.widget === "slider") return buildSlider(control);
    return buildKnob(control);
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
    var diameter = spec.size === "large" ? 72 : spec.size === "small" ? 40 : 56;
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

  function wavePath(amplitude, middle, cycles) {
    var path = "";
    for (var i = 0; i <= 180; i++) {
      var x = 20 + i / 180 * 480;
      var y = middle - Math.sin(i / 180 * Math.PI * 2 * cycles) * amplitude;
      path += (i ? " L " : "M ") + x.toFixed(1) + " " + y.toFixed(1);
    }
    return path;
  }

  function trace(samples, middle, amplitude) {
    var path = "";

    for (var i = 0; i < samples.length; i++) {
      var x = 20 + i / (samples.length - 1) * 480;
      var y = middle - clamp(samples[i], -1.2, 1.2) / 1.2 * amplitude;
      path += (i ? " L " : "M ") + x.toFixed(1) + " " + y.toFixed(1);
    }

    return path;
  }

  function waveformSvg() {
    if (scope.input && scope.output) {
      return heroSvg(
        '<line x1="0" y1="80" x2="520" y2="80" stroke="#2c2c2c" stroke-width="1"/>' +
        '<path d="' + trace(scope.input, 80, 58) + '" fill="none" stroke="#6a6a6a" ' +
          'stroke-width="1.5" vector-effect="non-scaling-stroke"/>' +
        '<path d="' + trace(scope.output, 80, 58) + '" fill="none" stroke="' + accent + '" ' +
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

  function transferSvg() {
    var mode = Math.round(valueOf("mode", 0));
    var threshold = clamp(valueOf("threshold", 0.8), 0.01, 4);
    var gain = valueOf("gain", 1);
    var path = "";

    // The domain is wider than the clip point on purpose: a hard clipper's curve
    // only shows its knees when the flat part has somewhere to run.
    for (var i = 0; i <= 140; i++) {
      var x = -2 + i / 140 * 4;
      var shaped = mode === 1
        ? Math.tanh(x / threshold)
        : clamp(x, -threshold, threshold);
      var y = clamp(shaped * gain, -2, 2);
      path += (i ? " L " : "M ") + (20 + (x + 2) / 4 * 480).toFixed(1) +
              " " + (80 - y / 2 * 66).toFixed(1);
    }

    return heroSvg(
      '<path d="M 20 146 L 500 14" stroke="#333" stroke-width="1" stroke-dasharray="3 5" ' +
        'vector-effect="non-scaling-stroke"/>' +
      '<line x1="260" y1="12" x2="260" y2="148" stroke="#2b2b2b"/>' +
      '<line x1="20" y1="80" x2="500" y2="80" stroke="#2b2b2b"/>' +
      '<path d="' + path + '" fill="none" stroke="' + accent + '" stroke-width="2.4" ' +
        'vector-effect="non-scaling-stroke"/>'
    );
  }

  function spectrumSvg() {
    var cutoff = Math.max(valueOf("cutoff", 1000), 10);
    var q = Math.max(valueOf("q", 0.707), 0.05);
    var mode = Math.round(valueOf("mode", 0));
    var magnitudes = [];

    for (var i = 0; i < 80; i++) {
      var frequency = 20 * Math.pow(1000, i / 79);
      var ratio = frequency / cutoff;
      var denominator = Math.sqrt(
        Math.pow(1 - ratio * ratio, 2) + Math.pow(ratio / q, 2)
      );
      var magnitude = mode === 1
        ? (ratio * ratio) / denominator
        : mode === 2
          ? (ratio / q) / denominator
          : 1 / denominator;

      magnitudes.push(magnitude);
    }

    var loudest = Math.max.apply(null, magnitudes);
    var path = "";

    for (var j = 0; j < magnitudes.length; j++) {
      var decibels = 20 * Math.log10(Math.max(magnitudes[j] / loudest, 1e-6));
      var x = 20 + j / 79 * 480;
      var y = clamp(10 + (1 - (1 + decibels / 40)) * 140, 10, 150);
      path += (j ? " L " : "M ") + x.toFixed(1) + " " + y.toFixed(1);
    }

    return heroSvg(
      '<path d="' + path + ' L 500 150 L 20 150 Z" fill="' + accent + '" fill-opacity=".07"/>' +
      '<path d="' + path + '" fill="none" stroke="' + accent + '" stroke-width="2" ' +
        'vector-effect="non-scaling-stroke"/>'
    );
  }

  function scopeSvg() {
    if (!scope.output) {
      return heroSvg(
        '<line x1="0" y1="80" x2="520" y2="80" stroke="#2c2c2c" stroke-width="1"/>'
      );
    }

    return heroSvg(
      '<line x1="0" y1="80" x2="520" y2="80" stroke="#2c2c2c" stroke-width="1"/>' +
      '<path d="' + trace(scope.input, 80, 58) + '" fill="none" stroke="#6a6a6a" ' +
        'stroke-width="1.2" vector-effect="non-scaling-stroke"/>' +
      '<path d="' + trace(scope.output, 80, 58) + '" fill="none" stroke="' + accent + '" ' +
        'stroke-width="1.8" vector-effect="non-scaling-stroke"/>'
    );
  }

  function updateHeroes() {
    heroes.forEach(function (hero) {
      if (hero.kind === "waveform") hero.el.innerHTML = waveformSvg();
      else if (hero.kind === "transfer") hero.el.innerHTML = transferSvg();
      else if (hero.kind === "spectrum") hero.el.innerHTML = spectrumSvg();
      else if (hero.kind === "scope") hero.el.innerHTML = scopeSvg();
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
            control.value = state[id];
            control.paint();
          }
        });
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

        if (data.length >= 512) {
          scope.input = data.subarray(0, 256);
          scope.output = data.subarray(256, 512);
          markDirty();
        }
      })
      .catch(function () {})
      .then(function () {
        setTimeout(pollScope, 33);
      });
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

  function buildMode(spec) {
    var el = element("div", "mode");
    var buttons = [];

    var control = { spec: spec, value: spec.default, paint: function () {} };
    controls[spec.id] = control;

    control.paint = function () {
      var active = Math.round(control.value);

      buttons.forEach(function (button, index) {
        button.classList.toggle("active", index === active);
        button.setAttribute("aria-pressed", index === active ? "true" : "false");
      });
    };

    (spec.choices || []).forEach(function (name, index) {
      var button = element("button", "mode-choice");
      button.type = "button";
      button.textContent = name;
      button.addEventListener("click", function () {
        control.value = index;
        control.paint();
        markDirty();
        if (window.ipc) window.ipc.postMessage(spec.id + "=" + index);
      });
      buttons.push(button);
      el.appendChild(button);
    });

    control.paint();
    return el;
  }

  function buildFace() {
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

    // A mode switch is one parameter drawn in the header, not a knob in the body.
    if (face.mode) {
      face.sections.forEach(function (section) {
        section.params = section.params.filter(function (spec) {
          if (spec.id !== face.mode) return true;
          modeSpec = spec;
          return false;
        });
      });

      face.sections = face.sections.filter(function (section) {
        return section.params.length > 0 || section.hero;
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
    header.appendChild(element("span", "spacer"));

    if (modeSpec) header.appendChild(buildMode(modeSpec));

    var body = element("div", "body");
    var many = face.sections.length > 1;

    face.sections.forEach(function (section) {
      var el = element("section", "section");

      if (many && section.name) {
        var caption = element("div", "caption");
        caption.textContent = section.name;
        el.appendChild(caption);
      }

      if (section.hero) {
        var hero = element("div", "hero");
        hero.dataset.hero = section.hero;
        heroes.push({ el: hero, kind: section.hero });
        el.appendChild(hero);
      }

      var row = element("div", "controls");
      section.params.forEach(function (spec) {
        row.appendChild(makeControl(spec));
      });
      el.appendChild(row);
      body.appendChild(el);
    });

    wantsScope = !!face.scope || heroes.some(function (hero) {
      return hero.kind === "waveform";
    });

    if (face.scope) {
      var strip = element("section", "section strip");
      var scopeHero = element("div", "hero");
      scopeHero.dataset.hero = "scope";
      heroes.push({ el: scopeHero, kind: "scope" });
      strip.appendChild(scopeHero);
      body.appendChild(strip);
    }

    var shell = element("div", "face");
    shell.appendChild(header);
    shell.appendChild(body);
    app.appendChild(shell);

    markDirty();
    poll();
    if (wantsScope) pollScope();
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
