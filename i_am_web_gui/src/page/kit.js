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
  var dirty = false;
  var polling = false;
  var bypassed = false;
  var powerButton = null;
  var modeSpec = null;
  var wantsScope = false;
var wantsAutotune = false;
var autotuneState = { notes: -1 };
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
var eqDrag = null;
var eqSpectrum = { input: null, output: null, rate: 0, inputPath: "", outputPath: "" };
var ENV_MAX = 512;
  var scope = { input: null, output: null, sidechain: null };
  var scale = 1;
  var spectra = { input: null, output: null };


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
  // ------------------------------------------------------------ value space
  /// The scale of the smooth logarithmic map used when a logarithmic range
  /// reaches zero or goes negative.
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
    if (spec.max === spec.min) return 0;

    if (spec.log && spec.max > spec.min) {
      return spec.min > 0
        ? Math.log(value / spec.min) / Math.log(spec.max / spec.min)
        : symlogNorm(spec, value);
    }

    return (value - spec.min) / (spec.max - spec.min);
  }

  function at(spec, position) {
    position = clamp(position, 0, 1);

    if (spec.log && spec.max > spec.min) {
      return spec.min > 0
        ? spec.min * Math.pow(spec.max / spec.min, position)
        : symlogAt(spec, position);
    }

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

    if (control.spec.id.indexOf("_type") >= 0) applyEqType();

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
    else if (spec.widget === "notes") el = buildNotePicker(control);
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

  function buildKnob(control) {
    var spec = control.spec;
    var handle = IAmCard.controlHandle(spec, control.value, function (_, value) {
      setValue(control, value, true);
    });

    handle.el.classList.add("face-knob");
    control.paint = function () { handle.set(control.value, false); };
    control.paint();
    return handle.el;
  }

  function buildNotePicker(control) {
    var handle = IAmCard.notePickerHandle(control.spec, control.value, function (_, value) {
      setValue(control, value, true);
    });
    handle.el.classList.add("face-notes");
    control.paint = function () { handle.set(control.value); };
    control.paint();
    return handle.el;
  }

  function buildToggle(control) {
    var spec = control.spec;
    var handle = IAmCard.switchHandle(spec, control.value, function (_, value) {
      setValue(control, value, true);
    });
    var el = element("div", "toggle");
    var label = element("div", "label");
    label.textContent = spec.label;
    var readout = element("div", "readout");
    handle.el.classList.add("face-switch");

    control.paint = function () {
      handle.set(control.value, false);
      readout.textContent = format(spec, control.value);
    };

    el.appendChild(handle.el);
    el.appendChild(label);
    el.appendChild(readout);
    control.paint();
    return el;
  }

  function buildChoice(control) {
    var spec = control.spec;
    var handle = IAmCard.choiceHandle(spec, control.value, function (_, value) {
      setValue(control, value, true);
    });
    var el = element("div", "choice");
    var label = element("div", "label");
    label.textContent = spec.label;
    handle.el.classList.add("face-choice");

    control.paint = function () { handle.set(control.value); };

    el.appendChild(label);
    el.appendChild(handle.el);
    control.paint();
    return el;
  }

  function buildSlider(control) {
    var spec = control.spec;
    var handle = IAmCard.sliderHandle(spec, control.value, function (_, value) {
      setValue(control, value, true);
    });
    var el = element("div", "slider");
    handle.el.classList.add("face-slider");
    control.paint = function () { handle.set(control.value, false); };
    el.appendChild(handle.el);
    control.paint();
    return el;
  }

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

  function wavePath(amplitude, middle, cycles) {
    var path = "";
    for (var i = 0; i <= 180; i++) {
      var x = 20 + i / 180 * 480;
      var y = middle - Math.sin(i / 180 * Math.PI * 2 * cycles) * amplitude;
      path += (i ? " L " : "M ") + x.toFixed(1) + " " + y.toFixed(1);
    }
    return path;
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
  /// The EQ's RBJ coefficients, mirrored from the DSP so the curve is the
  /// filter and not an approximation of it.
  /// The EQ curve, its frequency/dB grid and the six draggable band nodes.
  function eqBandIndex(target) {
    if (!target || !target.closest) return -1;
    var node = target.closest("[data-eq-band]");
    return node ? parseInt(node.getAttribute("data-eq-band"), 10) : -1;
  }

  function eqSelectBand(index) {
    var control = controls.band;
    if (control) setValue(control, index, true);
  }

  function eqSet(index, name, value) {
    var control = controls["b" + (index + 1) + "_" + name];
    if (control) setValue(control, value, true);
  }

  function eqPointerDown(event) {
    var index = eqBandIndex(event.target);
    if (index < 0) return;

    var hero = event.currentTarget;
    if (!hero.querySelector("svg")) return;

    eqDrag = { index: index, hero: hero, pointerId: event.pointerId };
    eqSelectBand(index);
    if (hero.setPointerCapture) hero.setPointerCapture(event.pointerId);
    event.preventDefault();
  }

  function eqPointerMove(event) {
    if (!eqDrag) return;

    var svg = eqDrag.hero.querySelector("svg");
    if (!svg) return;

    var rect = svg.getBoundingClientRect();
    if (!rect.width || !rect.height) return;

    var u = clamp((event.clientX - rect.left) / rect.width, 0, 1);
    var v = clamp((event.clientY - rect.top) / rect.height, 0, 1);
    var type = Math.round(valueOf("b" + (eqDrag.index + 1) + "_type", 0));
    eqSet(eqDrag.index, "freq", 20 * Math.pow(1000, u));
    if (type <= 2) eqSet(eqDrag.index, "gain", 24 - v * 48);
    event.preventDefault();
  }

  function eqPointerUp(event) {
    if (!eqDrag) return;

    var hero = event.currentTarget;
    if (hero.releasePointerCapture && eqDrag.pointerId !== undefined) {
      try {
        hero.releasePointerCapture(eqDrag.pointerId);
      } catch (error) {}
    }

    eqDrag = null;
  }

  function eqWheel(event) {
    var index = eqBandIndex(event.target);
    if (index < 0) return;

    event.preventDefault();
    eqSelectBand(index);
    var q = valueOf("b" + (index + 1) + "_q", 1);
    eqSet(index, "q", q * (event.deltaY < 0 ? 1.12 : 0.89));
  }

  function eqDoubleClick(event) {
    var index = eqBandIndex(event.target);
    if (index < 0) return;

    eqSelectBand(index);
    eqSet(index, "gain", 0);
    event.preventDefault();
  }

  /// The current control values, in the shape the shared card heroes expect.
  /// Passing this map is what keeps a hero tied to its parameters after the
  /// renderers moved out of this page and into card.js.
  function heroValues() {
    var values = {};

    Object.keys(controls).forEach(function (id) {
      values[id] = controls[id].value;
    });

    return values;
  }

  function updateHeroes() {
    if (!window.IAmCard || !IAmCard.hero) return;

    var values = heroValues();

    heroes.forEach(function (hero) {
      var content = IAmCard.hero(hero.kind, values, {
        ir: irState, curve: curveState, stereo: stereoScope, scope: scope,
        rate: (face && face.rate) || 48000, spectra: spectra,
        meter: meter, envHistory: envHistory, envWindow: envWindow,
        ott: ottState, vocoder: vocoderState, eqSpectrum: eqSpectrum,
        autotune: autotuneState
      });
      if (content) hero.el.innerHTML = content;
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
          if (control) {
            control.value = control.spec.kind === "bool"
              ? state[id]
              : snap(control.spec, state[id]);
            control.paint();
          }
        });
        applySwitches();
        applyEqType();
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
        var third = data.length / 3;

        if (third > 64) {
          scope.input = data.subarray(0, third);
          scope.output = data.subarray(third, third * 2);
          scope.sidechain = data.subarray(third * 2);
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
  /// The autotune publishes the mask it is actually using: MIDI held notes win
  /// over the faceplate mask. The preview server has no /allowed and the hero
  /// falls back to the note control.
  function pollAllowed() {
    if (!wantsAutotune) return;
    if (isPreview || document.hidden) {
      setTimeout(pollAllowed, 100);
      return;
    }

    fetch("/allowed", { cache: "no-store" })
      .then(function (response) {
        return response.ok ? response.json() : null;
      })
      .then(function (data) {
        if (data && typeof data.notes === "number") {
          autotuneState.notes = data.notes;
          updateHeroes();
        }
      })
      .catch(function () {
        autotuneState.notes = -1;
      })
      .then(function () {
        setTimeout(pollAllowed, 100);
      });
  }

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

    applyEqType();
  }

  /// A cut or a notch has no gain, so hide that knob for the selected band.
  function applyEqType() {
    if (!face || face.mode !== "band") return;

    var index = Math.round(valueOf("band", 0));
    var type = Math.round(valueOf("b" + (index + 1) + "_type", 0));
    var gain = controls["b" + (index + 1) + "_gain"];
    var slope = controls["b" + (index + 1) + "_slope"];

    if (gain && gain.el) gain.el.style.display = type <= 2 ? "" : "none";
    if (slope && slope.el) {
      slope.el.style.display = type === 3 || type === 4 ? "" : "none";
    }
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

    var control = { spec: spec, value: spec.default, el: el, paint: function () {} };
    controls[spec.id] = control;

    if (window.IAmCard && IAmCard.choiceHandle) {
      var handle = IAmCard.choiceHandle(spec, control.value, function (_, value) {
        setValue(control, value, true);
      });
      handle.el.className = "mcard-select mode-select";
      handle.el.setAttribute("aria-label", spec.label || "Choice");
      control.paint = function () { handle.set(control.value); };

      el.appendChild(label);
      el.appendChild(handle.el);
      control.paint();
      return el;
    }

    var select = document.createElement("select");
    select.className = "mode-select";
    select.setAttribute("aria-label", spec.label || "Choice");

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
    var el = element("div", "mode mode-drop");

    var control = { spec: spec, value: spec.default, el: null, paint: function () {} };
    controls[spec.id] = control;

    var caption = element("span", "header-caption");
    caption.textContent = spec.label || "";
    el.appendChild(caption);

    var handle = IAmCard.choiceHandle(spec, control.value, function (_, value) {
      setValue(control, value, true);
    });
    handle.el.className = "mcard-select mode-select";
    handle.el.setAttribute("aria-label", spec.label || "Mode");
    control.el = el;
    control.paint = function () {
      var active = Math.round(control.value);
      handle.set(active);
      if (spec.id === face.mode) applyVisibility(active);
    };

    el.appendChild(handle.el);
    control.paint();
    return el;
  }

  /// One strip of tabs, and only the sections the active tab names.
  ///
  /// A section that no page names stays visible on every page: the page list
  /// hides what it mentions and never a control nobody thought about.
  function buildPages(sectionEls) {
    var pages = face.pages;
    var mentioned = {};

    pages.forEach(function (page) {
      (page.sections || []).forEach(function (name) { mentioned[name] = true; });
    });

    var strip = element("div", "face-tabs");
    var buttons = [];

    function show(index) {
      var page = pages[index] || pages[0];
      var shown = {};

      (page.sections || []).forEach(function (name) { shown[name] = true; });

      buttons.forEach(function (button, position) {
        button.setAttribute("aria-pressed", position === index ? "true" : "false");
      });

      sectionEls.forEach(function (section) {
        var visible = !section.name || !mentioned[section.name] || !!shown[section.name];
        section.el.style.display = visible ? "" : "none";
      });
    }

    pages.forEach(function (page, index) {
      var button = element("button", "face-tab");
      button.type = "button";
      button.textContent = page.name;
      button.setAttribute("aria-pressed", "false");
      button.addEventListener("click", function () { show(index); });
      buttons.push(button);
      strip.appendChild(button);
    });

    show(0);
    return strip;
  }

  function buildFace() {
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
    // Parameters the face is told not to draw: the host still has them.
    var hiddenIds = face.hidden || [];

    if (face.mode || headerIds.length || hiddenIds.length) {
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

          if (hiddenIds.indexOf(spec.id) >= 0) return false;

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
        // A mode that hides this switch has to hide its caption too, so the
        // control the visibility rules move is the whole item, not just the dot.
        if (controls[spec.id]) controls[spec.id].el = wrap;
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

    // The shell exists before the body so the tab strip can sit between them.
    var shell = element("div", "face");

    // A paged faceplate packs its rows: the tabs name the page, so the room the
    // wide spacing used to take is not needed.
    if (face.pages && face.pages.length) shell.classList.add("compact");

    var body = element("div", "body");
    var many = face.sections.length > 1;
    // When every page holds a single section, the tab already names it: drawing
    // the section caption as well would say the same thing twice.
    var oneSectionPerPage =
      face.pages && face.pages.length > 0 &&
      face.pages.every(function (page) { return (page.sections || []).length === 1; });
    var usedHeroes = {};
    var sectionEls = [];
    // Which page a section is on, so a hero shared by two pages (the OTT's meter
    // is on both) can be drawn twice: only one page is visible at a time.
    var heroPageOf = {};

    if (face.pages) {
      face.pages.forEach(function (page, index) {
        (page.sections || []).forEach(function (name) { heroPageOf[name] = index; });
      });
    }

    face.sections.forEach(function (section) {
      var el = element("section", "section");

      if (many && section.name && !oneSectionPerPage) {
        var caption = element("div", "caption");
        caption.textContent = section.name;
        el.appendChild(caption);
      }

      // One hero per page is enough: a nested effect can ask for the same hero
      // in a second section on the same page (the crusher's gain stage embeds
      // Gain), and drawing the same live trace twice only costs space. Two
      // different pages may share it, because only one of them is ever shown.
      var heroPage = Object.prototype.hasOwnProperty.call(heroPageOf, section.name)
        ? heroPageOf[section.name]
        : -1;
      var heroUsed = usedHeroes[section.hero];
      var heroAllowed = section.hero && (heroUsed === undefined ||
        (heroPage >= 0 && heroUsed >= 0 && heroPage !== heroUsed));

      if (heroAllowed) {
        usedHeroes[section.hero] = heroPage;
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

        if (section.hero === "eq") {
          hero.title = "Drag a band: horizontal frequency, vertical gain; wheel changes Q";
          hero.style.touchAction = "none";
          hero.addEventListener("pointerdown", eqPointerDown);
          hero.addEventListener("pointermove", eqPointerMove);
          hero.addEventListener("pointerup", eqPointerUp);
          hero.addEventListener("pointercancel", eqPointerUp);
          hero.addEventListener("wheel", eqWheel, { passive: false });
          hero.addEventListener("dblclick", eqDoubleClick);
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
      sectionEls.push({ name: section.name, el: el });
    });

    // A faceplate with more controls than one window shows comfortably can hand
    // its sections to tabs; the hero and the switches stay in the sections they
    // belong to.
    var pagesStrip = face.pages && face.pages.length ? buildPages(sectionEls) : null;

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
        hero.kind === "compressor" || hero.kind === "gate" ||
        hero.kind === "expander" || hero.kind === "limiter" || hero.kind === "eq";
    });
    wantsMeter = heroes.some(function (hero) {
      return hero.kind === "compressor" || hero.kind === "gate" ||
        hero.kind === "expander" || hero.kind === "limiter";
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

    shell.appendChild(header);
    if (pagesStrip) shell.appendChild(pagesStrip);
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
      wantsAutotune = page.sections.some(function (section) {
        return section.hero === "autotune";
      });
      buildFace();
      pollAllowed();
    })
    .catch(function (error) {
      app.textContent = "faceplate unavailable: " + error;
      console.error("i_am_web_gui: " + error);
    });
})();