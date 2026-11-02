(function () {
  "use strict";

  var rack = document.getElementById("rack");
  var status = document.getElementById("status");

  // A page that dies on load leaves an empty rack and no clue; the footer is
  // the only place a person can see why.
  window.addEventListener("error", function (event) {
    if (status && event.message) status.textContent = "error: " + event.message;
  });
  var rate = document.getElementById("rate");
  var power = document.getElementById("power");
  var panelBody = document.getElementById("panel-body");
  var panelTabs = document.querySelectorAll(".panel-tab");
  var bottomPanel = document.getElementById("bottom");
  var splitter = document.getElementById("splitter");
  var panelToggle = document.getElementById("panel-toggle");
  /// The macro knobs and the automation readouts, refreshed from the host.
  var macroHandles = [];
  var macroValues = [];
  var automationSpans = [];
  var pointerDown = false;
  var panelCollapsed = false;
  var panelRestore = 0;
  var activePanel = "sources";
  var panelPages = null;
  var undoButton = document.getElementById("undo");
  var redoButton = document.getElementById("redo");
  var abButton = document.getElementById("ab");
  var presetInit = document.getElementById("preset-init");
  var presetSave = document.getElementById("preset-save");
  var presetLoad = document.getElementById("preset-load");
  var presetFile = document.getElementById("preset-file");

  var pending = Object.create(null);
  var timer = null;
  var bypassed = false;
  var current = null;
  var rangeLabels = [];
  var paintCrossover = null;
  var collapsedBands = Object.create(null);
  var collapsedGroups = Object.create(null);
  var collapsedSources = Object.create(null);
  var collapsedModules = Object.create(null);
  /// The source a person is looking at, when they picked one: every knob then
  /// shows only that source's modulation.
  var selectedSource = null;
  var sourceNames = Object.create(null);
  var sourceColors = Object.create(null);
  var slotScroll = Object.create(null);
  var sourceScroll = 0;
  var livePainters = [];
  var live = {
    values: Object.create(null),
    phases: Object.create(null),
    history: Object.create(null)
  };
  var liveTimer = null;
  var scopeTimer = null;
  var stereoTimer = null;
  var meterTimer = null;
  // The audio the pages draw: the same three documents the product page polls,
  // kept in one context object so a card can be redrawn whenever one arrives.
  var pageRate = 48000;
  var cardTimer = null;
  var irTimer = null;
  var liveCards = [];

  /// Every card gets its own rings: a module's card shows what entered and left
  /// that module, not the rack's own input and output.
  function cardContext() {
    return {
      scope: { input: null, output: null, sidechain: null },
      stereo: { inputLeft: null, inputRight: null, outputLeft: null, outputRight: null },
      meter: { input: null, sidechain: null },
      spectra: { input: null, output: null },
      eqSpectrum: { input: null, output: null, rate: 0, inputPath: "", outputPath: "" },
      envHistory: { input: [], output: [] },
      envWindow: 128,
      rate: pageRate
    };
  }

  function slotLetter(index) {
    return index < 26 ? String.fromCharCode(65 + index) : String(index + 1);
  }

  function el(tag, className, text) {
    var node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined) node.textContent = text;
    return node;
  }

  function queue(address, value) {
    pending[address] = address + "=" + value;
    if (timer) cancelAnimationFrame(timer);
    timer = requestAnimationFrame(flush);
  }

  function flush() {
    if (timer) {
      cancelAnimationFrame(timer);
      timer = null;
    }

    var addresses = Object.keys(pending);
    if (!addresses.length) return Promise.resolve();

    var body = addresses.map(function (address) {
      return pending[address];
    }).join("\n");

    pending = Object.create(null);

    return fetch("/rack-command", { method: "POST", body: body })
      .then(function (response) {
        if (!response.ok) throw new Error("command " + response.status);
        return response.json();
      })
      .then(function (result) {
        if (!result.ok) {
          status.textContent = "some commands were ignored";
          return;
        }

        // A normal edit is a new undo step and drops the redo branch; a
        // whole-state command refreshes the buttons from /rack right after.
        undoButton.disabled = false;
        redoButton.disabled = true;
      })
      .catch(function (error) {
        status.textContent = error.message;
      });
  }

  function sendNow(address, value) {
    pending[address] = address + "=" + value;
    return flush();
  }

  // A command that replaces the whole state must not share a batch with a
  // slider that is still in the queue: the server treats it as a lone line.
  function runSpecial(address, value) {
    return flush().then(function () {
      return sendNow(address, value);
    });
  }

  function refreshEditSlots(data) {
    undoButton.disabled = !data.history || !data.history.undo;
    redoButton.disabled = !data.history || !data.history.redo;

    var active = data.ab && data.ab.active === "B" ? "B" : "A";
    abButton.textContent = active;
    abButton.dataset.active = active;
    abButton.title = "A/B compare (now " + active + ")";
  }

  function historyStep(direction) {
    return runSpecial("history." + direction, "1").then(load);
  }

  function formatHz(value) {
    if (value >= 1000) {
      return (value / 1000).toFixed(value >= 10000 ? 1 : 2) + " kHz";
    }
    return Math.round(value) + " Hz";
  }

  function bandRange(data, index) {
    if (index > 2) return "20 Hz \u2013 20 kHz";

    var low = index === 0 ? 20 : index === 1 ? data.split_low : data.split_high;
    var high = index === 0 ? data.split_low : index === 1 ? data.split_high : 20000;
    return formatHz(low) + " \u2013 " + formatHz(high);
  }

  function updateRanges(data) {
    data.bands.forEach(function (band, index) {
      if (rangeLabels[index]) {
        rangeLabels[index].textContent = bandRange(data, index);
      }
    });
  }

  function setModulationDepth(slot, depth) {
    queue("assign." + slot + ".depth", depth);
  }

  function attachHeightResize(handle, target, minimum, maximum, key) {
    try {
      var saved = Number(localStorage.getItem(key));
      if (saved > 0) target.style.height = saved + "px";
    } catch (ignore) {}

    var drag = null;

    handle.addEventListener("pointerdown", function (event) {
      if (event.button !== 0) return;
      drag = {
        y: event.clientY,
        height: target.getBoundingClientRect().height
      };
      handle.classList.add("dragging");
      handle.setPointerCapture(event.pointerId);
      event.preventDefault();
    });

    handle.addEventListener("pointermove", function (event) {
      if (!drag) return;
      var height = Math.max(minimum, Math.min(maximum, drag.height + event.clientY - drag.y));
      target.style.height = height + "px";
      event.preventDefault();
    });

    function stop(event) {
      if (!drag) return;
      drag = null;
      handle.classList.remove("dragging");
      try { localStorage.setItem(key, String(Math.round(target.getBoundingClientRect().height))); } catch (ignore) {}
      try { handle.releasePointerCapture(event.pointerId); } catch (ignore) {}
    }

    handle.addEventListener("pointerup", stop);
    handle.addEventListener("pointercancel", stop);
    handle.addEventListener("dblclick", function () {
      target.style.height = "";
      try { localStorage.removeItem(key); } catch (ignore) {}
    });
  }

  function toggleModulationMode(slots, mode) {
    slots.forEach(function (slot) {
      queue("assign." + slot + ".mode", mode);
    });
    flush().then(load);
  }

  function linkModulation(data, target) {
    var source = "";

    if (data.indexOf("source:") === 0) {
      source = data.slice(7);
    } else if (data.indexOf("macro:") === 0) {
      source = "macro" + (Number(data.slice(6)) + 1);
    }

    if (!source || !target) return;
    sendNow("assign.add", source + "|" + target).then(load);
  }

  // The card owns the normalisation, including the symmetric-log mapping a
  // bipolar parameter uses; asking it keeps the page and the knobs identical.
  function normalizeValue(spec, value) {
    return window.IAmCard.norm(spec, value);
  }

  function denormalizeValue(spec, position) {
    return window.IAmCard.at(spec, position);
  }

  /// Whether a source only ever runs forward, which decides how its value reads
  /// as a modulation amount.
  function unipolar(source) {
    if (source >= 4) return true;

    var kind = current && current.sources
      ? (current.sources.find(function (entry) { return entry.id === source; }) || {}).kind
      : "";

    return kind === "envelope" || kind === "pitch" || kind === "midi";
  }

  /// The values a card should draw after modulation: the engine applies this
  /// arithmetic on the audio thread, and the card reproduces it so a hero moves
  /// with the source instead of sitting at the knob's base value.
  function liveModulation(entry) {
    var module = entry.module;
    if (!module) return null;

    var prefix = "m" + entry.id + ".";
    var sums = Object.create(null);
    var any = false;

    (current && current.assignments ? current.assignments : []).forEach(function (assignment) {
      if (!assignment || !assignment.target) return;
      if (assignment.target.indexOf(prefix) !== 0) return;

      var value = live.values[String(assignment.source)];
      if (typeof value !== "number") return;

      var offset = assignment.mode === "centered"
        ? value * assignment.depth
        : (unipolar(assignment.source) ? value : (value + 1) * 0.5) * assignment.depth;

      var parameter = assignment.target.slice(prefix.length);
      sums[parameter] = (sums[parameter] || 0) + offset;
      any = true;
    });

    if (!any) return null;

    var out = Object.create(null);

    Object.keys(sums).forEach(function (parameter) {
      var control = (module.controls || []).find(function (entry) {
        return entry.id === parameter;
      });

      if (!control || control.kind !== "float") return;

      var base = typeof module.values[parameter] === "number"
        ? module.values[parameter]
        : control.default;
      var normalized = normalizeValue(control, base) + sums[parameter];

      out[parameter] = denormalizeValue(control, normalized);
    });

    return out;
  }

  /// The colour a source kind uses, in its card and on the knobs it drives.
  function sourceAccent(kind) {
    return kind === "lfo" ? "#4c8dff" :
      kind === "envelope" ? "#35c77a" :
        kind === "random" ? "#ff9a3d" :
          kind === "table" ? "#c8e04a" :
            kind === "pitch" ? "#b08cff" : "#ff6f91";
  }

  function sourceLabel(source) {
    return sourceNames[source] || ("Source " + (source + 1));
  }

  function sourceColor(source) {
    return sourceColors[source] || "var(--accent)";
  }

  function selectSource(id) {
    selectedSource = selectedSource === id ? null : id;
    load();
  }

  function modulationsFor(data, target) {
    var result = [];

    (data.assignments || []).forEach(function (assignment, slot) {
      if (!assignment || assignment.target !== target) return;
      if (selectedSource !== null && assignment.source !== selectedSource) return;

      result.push({
        slot: slot,
        source: assignment.source,
        label: sourceLabel(assignment.source),
        color: sourceColor(assignment.source),
        depth: assignment.depth,
        mode: assignment.mode || "centered",
        selected: selectedSource !== null
      });
    });

    return result;
  }

  function makeCrossover(data) {
    var section = el("section", "crossover");
    // The server keeps the splits inside the band its filters can be designed
    // for: low <= 0.25 of nyquist, high >= low * 1.5. The knobs have to agree,
    // or a value set here would silently change on the next command.
    var nyquist = (data.rate || 44100) * 0.49;
    var controls = el("div", "crossover-controls");
    controls.appendChild(el("span", "crossover-label", "CROSSOVER"));

    var low = window.IAmCard.control({
      id: "split.low",
      label: "Low / Mid",
      kind: "float",
      min: 20,
      max: Math.min(20000, nyquist * 0.25),
      default: 120,
      unit: "hz",
      log: true
    }, data.split_low, function (_, value) {
      current.split_low = Math.min(value, current.split_high / 1.5);
      queue("split.low", current.split_low);
      updateRanges(current);
      if (paintCrossover) paintCrossover();
    });
    low.classList.add("xover-knob");
    controls.appendChild(low);

    controls.appendChild(el("span", "crossover-sep", "/"));

    var high = window.IAmCard.control({
      id: "split.high",
      label: "Mid / High",
      kind: "float",
      min: 20,
      max: Math.min(20000, nyquist),
      default: 2000,
      unit: "hz",
      log: true
    }, data.split_high, function (_, value) {
      current.split_high = Math.max(value, current.split_low * 1.5);
      queue("split.high", current.split_high);
      updateRanges(current);
      if (paintCrossover) paintCrossover();
    });
    high.classList.add("xover-knob");
    controls.appendChild(high);
    section.appendChild(controls);

    var viz = el("div", "crossover-viz");
    var svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
    svg.setAttribute("viewBox", "0 0 560 78");
    svg.setAttribute("role", "img");
    svg.setAttribute("aria-label", "Low, mid and high crossover responses");
    viz.appendChild(svg);
    section.appendChild(viz);

    function node(tag, attributes) {
      var value = document.createElementNS("http://www.w3.org/2000/svg", tag);
      for (var key in attributes) value.setAttribute(key, attributes[key]);
      return value;
    }

    function xFor(hz) {
      var low = 20;
      var high = 20000;
      var position = Math.log(Math.max(low, Math.min(high, hz)) / low) /
        Math.log(high / low);
      return 8 + position * 544;
    }

    function yFor(decibels) {
      var top = 8;
      var height = 48;
      var high = 3;
      var low = -48;
      return top + (high - Math.max(low, Math.min(high, decibels))) /
        (high - low) * height;
    }

    function responsePath(kind) {
      var path = "";
      var low = current.split_low;
      var high = current.split_high;

      for (var i = 0; i <= 160; i++) {
        var frequency = 20 * Math.pow(1000, i / 160);
        var lowRatio = frequency / low;
        var highRatio = frequency / high;
        var lowPower = Math.pow(lowRatio, 4);
        var highPower = Math.pow(highRatio, 4);
        var lowPassLow = 1 / (1 + lowPower);
        var highPassLow = lowPower / (1 + lowPower);
        var lowPassHigh = 1 / (1 + highPower);
        var highPassHigh = highPower / (1 + highPower);
        var magnitude;

        if (kind === "low") magnitude = lowPassLow;
        else if (kind === "mid") magnitude = highPassLow * lowPassHigh;
        else magnitude = highPassLow * highPassHigh;

        var decibels = 20 * Math.log10(Math.max(magnitude, 0.0001));
        path += (i ? " L " : "M ") + xFor(frequency).toFixed(1) +
          " " + yFor(decibels).toFixed(1);
      }

      return path;
    }

    function paint() {
      svg.textContent = "";

      var lowX = xFor(current.split_low);
      var highX = xFor(current.split_high);

      [0, -24, -48].forEach(function (decibels) {
        svg.appendChild(node("line", {
          x1: 8,
          y1: yFor(decibels).toFixed(1),
          x2: 552,
          y2: yFor(decibels).toFixed(1),
          stroke: "#333",
          "stroke-width": "1",
          "stroke-dasharray": decibels === 0 ? "2 4" : "1 5"
        }));
      });

      [["low", "#4c8dff"], ["mid", "#ff9a3d"], ["high", "#35c77a"]].forEach(
        function (entry) {
          svg.appendChild(node("path", {
            d: responsePath(entry[0]),
            fill: "none",
            stroke: entry[1],
            "stroke-width": "2",
            "stroke-linecap": "round",
            "vector-effect": "non-scaling-stroke"
          }));
        }
      );

      [lowX, highX].forEach(function (x, index) {
        svg.appendChild(node("line", {
          x1: x.toFixed(1),
          y1: 4,
          x2: x.toFixed(1),
          y2: 60,
          stroke: "#e0e0e0",
          "stroke-opacity": ".35",
          "stroke-width": "1",
          "stroke-dasharray": "2 3"
        }));

        var frequency = index === 0 ? current.split_low : current.split_high;
        var label = node("text", {
          x: Math.max(16, Math.min(544, x)).toFixed(1),
          y: 73,
          "text-anchor": "middle",
          fill: "#9a9a9a",
          "font-size": "9",
          "font-family": "Cascadia Mono, Consolas, monospace"
        });
        label.textContent = formatHz(frequency);
        svg.appendChild(label);
      });

      [["LOW", 10, "start", "#4c8dff"],
       ["MID", (lowX + highX) / 2, "middle", "#ff9a3d"],
       ["HIGH", 550, "end", "#35c77a"]].forEach(function (entry) {
        var label = node("text", {
          x: Math.max(10, Math.min(550, entry[1])).toFixed(1),
          y: 8,
          "text-anchor": entry[2],
          fill: entry[3],
          "font-size": "8",
          "font-family": "Segoe UI, system-ui, sans-serif",
          "letter-spacing": "1"
        });
        label.textContent = entry[0];
        svg.appendChild(label);
      });
    }

    paintCrossover = paint;
    paint();

    section.appendChild(el("span", "crossover-note", "LR4 \u00B7 4 lanes"));
    return section;
  }

  function makeBandHeader(data, bandIndex, section) {
    var band = data.bands[bandIndex];
    var head = el("div", "band-head");

    var collapsed = !!collapsedBands[bandIndex];
    var toggle = el("button", "band-collapse", collapsed ? "\u25B8" : "\u25BE");
    toggle.type = "button";
    toggle.title = collapsed ? "Expand band" : "Collapse band";
    toggle.setAttribute("aria-expanded", collapsed ? "false" : "true");
    toggle.setAttribute("aria-label", toggle.title);
    toggle.addEventListener("click", function () {
      var next = !collapsedBands[bandIndex];
      collapsedBands[bandIndex] = next;
      section.classList.toggle("collapsed", next);
      toggle.textContent = next ? "\u25B8" : "\u25BE";
      toggle.title = next ? "Expand band" : "Collapse band";
      toggle.setAttribute("aria-expanded", next ? "false" : "true");
      toggle.setAttribute("aria-label", toggle.title);
    });
    head.appendChild(toggle);

    var rangeText = bandRange(data, bandIndex);
    var name = el("span", "band-name", band.name);
    name.title = band.name + " \u00B7 " + rangeText;
    head.appendChild(name);

    var range = el("span", "band-range", rangeText);
    rangeLabels[bandIndex] = range;
    head.appendChild(range);
    head.appendChild(el("span", "band-spacer"));

    var controls = el("div", "band-controls");

    controls.appendChild(window.IAmCard.control({
      id: "band.gain",
      label: "Gain",
      kind: "float",
      min: 0,
      max: 2,
      default: 1,
      unit: "",
      modulations: modulationsFor(data, "band." + bandIndex + ".gain"),
      onModulation: setModulationDepth,
      dropTarget: "band." + bandIndex + ".gain",
      onModulationDrop: linkModulation,
      onModeToggle: toggleModulationMode,
      format: function (value) {
        if (value <= 0) return "-\u221E dB";
        return (20 * Math.log10(value)).toFixed(1) + " dB";
      }
    }, band.gain, function (_, value) {
      band.gain = value;
      queue("band." + bandIndex + ".gain", value);
    }));

    controls.appendChild(window.IAmCard.control({
      id: "band.pan",
      label: "Pan",
      kind: "float",
      min: -1,
      max: 1,
      default: 0,
      unit: "",
      bipolar: true,
      modulations: modulationsFor(data, "band." + bandIndex + ".pan"),
      onModulation: setModulationDepth,
      dropTarget: "band." + bandIndex + ".pan",
      onModulationDrop: linkModulation,
      onModeToggle: toggleModulationMode,
      format: function (value) {
        if (Math.abs(value) < 0.005) return "C";
        return (value < 0 ? "L" : "R") + Math.round(Math.abs(value) * 100);
      }
    }, band.pan, function (_, value) {
      band.pan = value;
      queue("band." + bandIndex + ".pan", value);
    }));

    head.appendChild(controls);

    var buttons = el("div", "band-buttons");

    var mute = el("button", "band-button" + (band.mute ? " on" : ""), "M");
    mute.type = "button";
    mute.title = "Mute " + band.name;
    mute.setAttribute("aria-pressed", band.mute ? "true" : "false");
    mute.addEventListener("click", function () {
      sendNow("band." + bandIndex + ".mute", band.mute ? 0 : 1).then(load);
    });
    buttons.appendChild(mute);

    var solo = el("button", "band-button" + (band.solo ? " on" : ""), "S");
    solo.type = "button";
    solo.title = "Solo " + band.name;
    solo.setAttribute("aria-pressed", band.solo ? "true" : "false");
    solo.addEventListener("click", function () {
      sendNow("band." + bandIndex + ".solo", band.solo ? 0 : 1).then(load);
    });
    buttons.appendChild(solo);

    head.appendChild(buttons);
    return head;
  }

  function groupIndex(modules) {
    var groups = Object.create(null);

    modules.forEach(function (module, slot) {
      if (!module || module.group === null || module.group === undefined) return;

      var key = String(module.group);

      if (!groups[key]) groups[key] = { first: slot, count: 0 };
      groups[key].count++;
    });

    return groups;
  }

  function groupColor(id) {
    return "hsl(" + ((id * 67) % 360) + " 62% 62%)";
  }

  function decorateGroup(wrap, bandIndex, slot, module, groups, collapsed) {
    var card = wrap.querySelector(".mcard");
    if (!card) return;

    var row = el("div", "group-row");

    if (module.group !== null && module.group !== undefined) {
      var info = groups[String(module.group)];
      if (!info) return;

      var key = String(module.group);
      var isFirst = info.first === slot;
      var isCollapsed = !!collapsed[key];

      card.classList.add("grouped");
      card.style.setProperty("--group", groupColor(module.group));

      if (isFirst) {
        var main = el("button", "group-main",
          "GROUP " + info.count + (isCollapsed ? " \u25B8" : " \u25BE"));
        main.type = "button";
        main.title = isCollapsed ? "Expand group" : "Collapse group";
        main.style.borderColor = groupColor(module.group);
        main.addEventListener("click", function () {
          if (!collapsedGroups[bandIndex]) collapsedGroups[bandIndex] = Object.create(null);
          collapsedGroups[bandIndex][key] = !isCollapsed;
          load();
        });
        row.appendChild(main);
      } else {
        row.appendChild(el("span", "group-tag", "GROUP " + info.count));
      }

      var leave = el("button", "group-leave", "Leave");
      leave.type = "button";
      leave.title = "Leave this group";
      leave.addEventListener("click", function () {
        sendNow("group." + bandIndex + "." + slot, 1).then(load);
      });
      row.appendChild(leave);
    } else if (slot > 0) {
      var join = el("button", "group-join", "\u26D3 Group with previous");
      join.type = "button";
      join.title = "Group this module with the one on its left";
      join.addEventListener("click", function () {
        sendNow("group." + bandIndex + "." + slot, 1).then(load);
      });
      row.appendChild(join);
    } else {
      return;
    }

    card.appendChild(row);
  }

  function makeSlot(data, bandIndex, slotIndex, module) {
    var wrap = el("div", "slot");
    var widthKey = "iamrack.slotWidth." + bandIndex + "." + slotIndex;

    try {
      var saved = Number(localStorage.getItem(widthKey));

      if (saved >= 260) wrap.style.setProperty("--slot-width", saved + "px");
    } catch (ignore) {}
    wrap.appendChild(el("div", "slot-label", "SLOT " + slotLetter(slotIndex)));
    wrap.title = "Drop a module here; hold Alt or Ctrl to copy";

    var dropHint = el("div", "slot-drop-hint", "");
    wrap.appendChild(dropHint);

    if (module) {
      wrap.classList.add("has-resize");

      var resizer = el("div", "slot-resize");
      resizer.title = "Drag to resize this card";
      wrap.appendChild(resizer);

      var sizeDrag = null;

      resizer.addEventListener("pointerdown", function (event) {
        if (event.button !== 0) return;

        sizeDrag = {
          x: event.clientX,
          width: wrap.getBoundingClientRect().width
        };
        resizer.classList.add("dragging");
        resizer.setPointerCapture(event.pointerId);
        event.preventDefault();
        event.stopPropagation();
      });

      resizer.addEventListener("pointermove", function (event) {
        if (!sizeDrag) return;

        var width = Math.min(900, Math.max(260, sizeDrag.width + event.clientX - sizeDrag.x));
        wrap.style.setProperty("--slot-width", width + "px");
      });

      function stopSize(event) {
        if (!sizeDrag) return;

        sizeDrag = null;
        resizer.classList.remove("dragging");

        try {
          localStorage.setItem(widthKey, String(Math.round(wrap.getBoundingClientRect().width)));
        } catch (ignore) {}

        try { resizer.releasePointerCapture(event.pointerId); } catch (ignore) {}
      }

      resizer.addEventListener("pointerup", stopSize);
      resizer.addEventListener("pointercancel", stopSize);
    }

    wrap.addEventListener("dragover", function (event) {
      event.preventDefault();
      var copy = event.altKey || event.ctrlKey || event.metaKey;
      if (event.dataTransfer) event.dataTransfer.dropEffect = copy ? "copy" : "move";
      wrap.classList.add("drop");
      wrap.classList.toggle("drop-copy", copy);
      dropHint.textContent = copy ? "Copy here" : "Move here";
    });
    wrap.addEventListener("dragleave", function () {
      wrap.classList.remove("drop", "drop-copy");
      dropHint.textContent = "";
    });
    wrap.addEventListener("drop", function (event) {
      event.preventDefault();
      wrap.classList.remove("drop", "drop-copy");
      dropHint.textContent = "";

      if (!event.dataTransfer) return;

      var source = event.dataTransfer.getData("text/plain");
      var target = bandIndex + "." + slotIndex;

      // A source or a macro dragged here belongs to a knob, not to the slot.
      if (!source || source.indexOf("source:") === 0 || source.indexOf("macro:") === 0) return;
      if (source === target) return;

      // Alt/Ctrl turns the drag into a copy instead of a move.
      var copy = event.altKey || event.ctrlKey || event.metaKey;
      sendNow("slot." + source + (copy ? ".copy" : ".move"), target).then(load);
    });

    if (!module) {
      var empty = el("div", "slot-empty");
      var select = document.createElement("select");
      select.className = "slot-add";
      select.setAttribute("aria-label", "Add module");

      var placeholder = document.createElement("option");
      placeholder.value = "";
      placeholder.textContent = "+ Add module";
      select.appendChild(placeholder);

      // Thirty-eight modules read better in their families than in one list.
      var groups = Object.create(null);

      (data.kinds || []).forEach(function (kind) {
        var option = document.createElement("option");
        option.value = kind.name;
        option.textContent = kind.name;

        if (!kind.group) {
          select.appendChild(option);
          return;
        }

        if (!groups[kind.group]) {
          groups[kind.group] = document.createElement("optgroup");
          groups[kind.group].label = kind.group;
          select.appendChild(groups[kind.group]);
        }

        groups[kind.group].appendChild(option);
      });

      select.addEventListener("change", function () {
        if (!select.value) return;
        sendNow("slot." + bandIndex + "." + slotIndex + ".kind", select.value).then(load);
      });

      empty.appendChild(select);
      empty.appendChild(el("span", "slot-hint", "empty slot"));
      wrap.appendChild(empty);
      return wrap;
    }

    var kinds = data.kinds || [];

    (module.controls || []).forEach(function (control) {
      // The server annotates every control with the assignments that drive it;
      // the page narrows that to the selected source here, and gives each one
      // the label and colour its arc is drawn in.
      if (control.modulations && control.modulations.length) {
        control.modulations = control.modulations
          .filter(function (assignment) {
            return selectedSource === null || assignment.source === selectedSource;
          })
          .map(function (assignment) {
            assignment.label = sourceLabel(assignment.source);
            assignment.color = sourceColor(assignment.source);
            assignment.selected = selectedSource !== null;
            return assignment;
          });
      }

      control.onModulation = setModulationDepth;

      if (control.kind === "float") {
        control.dropTarget = "m" + module.id + "." + control.id;
        control.onModulationDrop = linkModulation;
        control.onModeToggle = toggleModulationMode;
      }
    });

    var cardAudio = cardContext();
    var entry = {
      id: module.id,
      module: module,
      channels: module.scope,
      note: module.note,
      meter: module.meter,
      ir: module.ir,
      script: module.script,
      card: null,
      context: cardAudio,
      fetching: false
    };

    var card = window.IAmCard.create({
      title: module.kind,
      badge: slotLetter(slotIndex),
      accent: module.accent,
      hero: module.hero,
      mode: module.mode,
      header: module.header,
      visible: module.visible,
      switches: module.switches,
      bypass: module.bypass,
      controls: module.controls,
      values: module.values,
      context: cardAudio,
      note: !!module.note,
      ir: !!module.ir,
      script: !!module.script,
      pickIr: function () { pickImpulse(entry); },
      loadScript: function () { loadScript(entry, true); },
      applyScript: function (source) { applyScript(entry, source); },
      kinds: kinds,
      kind: module.kind,
      dragData: bandIndex + "." + slotIndex,
      collapsed: !!collapsedModules[module.id],
      onCollapse: function (next) { collapsedModules[module.id] = next; },
      onKind: function (kind) {
        sendNow("slot." + bandIndex + "." + slotIndex + ".kind", kind).then(load);
      },
      onDuplicate: function () {
        sendNow("slot." + bandIndex + "." + slotIndex + ".duplicate", 1).then(load);
      },
      onRemove: function () {
        sendNow("slot." + bandIndex + "." + slotIndex + ".clear", 1).then(load);
      },
      onEdit: function (identifier, value) {
        queue(module.id + "." + identifier, value);
      },
      onBypass: function (next) {
        sendNow(module.id + ".__bypass", next ? 1 : 0);
      }
    });

    wrap.appendChild(card);

    attachEqBands(card, module);

    // The engine decides what a card reads from the audio, and says so in the
    // module: a hero that draws parameters gets nothing allocated at all.
    entry.card = card;
    liveCards.push(entry);

    return wrap;
  }

  /// The EQ's picture carries six draggable nodes. The product page wires this
  /// in its own module; a card needs the same behaviour, or the dots do nothing.
  ///
  /// The listeners sit on the card, not on the picture: a repaint replaces the
  /// picture's contents while the pointer is down.
  function attachEqBands(card, module) {
    var hero = card.querySelector(".mcard-hero");
    if (!hero) return;

    var drag = null;

    function bandOf(target) {
      var node = target && target.closest ? target.closest("[data-eq-band]") : null;
      return node ? Number(node.getAttribute("data-eq-band")) : -1;
    }

    /// The node closest to a horizontal position in the picture. Grabbing the
    /// picture anywhere takes the nearest one, which is more forgiving than
    /// insisting on its six-pixel dot.
    function nearestBand(x) {
      var dots = hero.querySelectorAll(".hero-dot");
      var best = -1;
      var bestDistance = 48;

      for (var i = 0; i < dots.length; i++) {
        var rect = dots[i].getBoundingClientRect();
        var distance = Math.abs(rect.left + rect.width / 2 - x);

        if (distance < bestDistance) {
          bestDistance = distance;
          best = Number(dots[i].getAttribute("data-eq-band"));
        }
      }

      return best;
    }

    function insideHero(event) {
      var rect = hero.getBoundingClientRect();
      return event.clientX >= rect.left && event.clientX <= rect.right
        && event.clientY >= rect.top && event.clientY <= rect.bottom;
    }

    function value(name, fallback) {
      var stored = module.values[name];
      return typeof stored === "number" ? stored : fallback;
    }

    function begin(band, pointerId) {
      if (!hero.querySelector("svg")) return;

      // The picture's contents are replaced while a card is live (the scope
      // arrives every few frames), so nothing inside it is held on to: the
      // element measured below is the one on screen at that moment.
      drag = { band: band, pointerId: pointerId };
      queue(module.id + ".band", band);
    }

    function move(x, y) {
      if (!drag) return;

      var svg = hero.querySelector("svg");
      if (!svg) return;

      var rect = svg.getBoundingClientRect();
      if (!rect.width || !rect.height) return;

      var u = Math.min(1, Math.max(0, (x - rect.left) / rect.width));
      var v = Math.min(1, Math.max(0, (y - rect.top) / rect.height));
      var index = drag.band + 1;
      var frequency = 20 * Math.pow(1000, u);
      var type = Math.round(value("b" + index + "_type", 0));

      queue(module.id + ".b" + index + "_freq", frequency);
      module.values["b" + index + "_freq"] = frequency;

      if (type <= 2) {
        var gain = 24 - v * 48;
        queue(module.id + ".b" + index + "_gain", gain);
        module.values["b" + index + "_gain"] = gain;
      }

      // Show the move while the pointer is down; the next reload settles it.
      if (card.repaintHero) card.repaintHero();
    }

    function release() {
      drag = null;
    }

    card.addEventListener("pointerdown", function (event) {
      if (!hero.querySelector("svg")) return;

      var band = bandOf(event.target);
      if (band < 0 && insideHero(event)) band = nearestBand(event.clientX);
      if (band < 0) return;

      begin(band, event.pointerId);
      try { card.setPointerCapture(event.pointerId); } catch (ignore) {}
      event.preventDefault();
    });

    card.addEventListener("pointermove", function (event) {
      if (!drag) return;
      move(event.clientX, event.clientY);
      event.preventDefault();
    });

    // The picture redraws with every scope frame; while a node is held, those
    // redraws would fight the drag, so the card pauses them.
    card.addEventListener("pointerdown", function (event) {
      if (drag) card.classList.add("dragging-band");
    }, true);
    card.addEventListener("pointerup", function () {
      card.classList.remove("dragging-band");
    }, true);

    card.addEventListener("pointerup", release);
    card.addEventListener("pointercancel", release);

    // A backend that does not deliver pointer capture still sends mouse events.
    card.addEventListener("mousedown", function (event) {
      if (event.button !== 0) return;
      if (drag || !hero.querySelector("svg")) return;

      var band = bandOf(event.target);
      if (band < 0 && insideHero(event)) band = nearestBand(event.clientX);
      if (band < 0) return;

      begin(band, null);
      event.preventDefault();
    });

    card.addEventListener("mousemove", function (event) {
      if (!drag || drag.pointerId !== null) return;
      move(event.clientX, event.clientY);
    });

    card.addEventListener("mouseup", function () {
      if (drag && drag.pointerId === null) release();
    });
  }

  function makeBand(data, bandIndex) {
    var band = data.bands[bandIndex];
    var section = el("section", "band");
    var anySolo = data.bands.some(function (candidate) { return candidate.solo; });

    if (band.mute || (anySolo && !band.solo)) section.classList.add("muted");
    if (band.solo) section.classList.add("solo");
    if (collapsedBands[bandIndex]) section.classList.add("collapsed");

    section.appendChild(makeBandHeader(data, bandIndex, section));

    var slots = el("div", "slots");
    var modules = band.slots.map(function (_, slot) {
      return data.modules.find(function (candidate) {
        return candidate.band === bandIndex && candidate.slot === slot;
      }) || null;
    });
    var groups = groupIndex(modules);
    var collapsed = collapsedGroups[bandIndex] || Object.create(null);

    for (var slot = 0; slot < modules.length; slot++) {
      var module = modules[slot];

      if (module && module.group !== null && module.group !== undefined) {
        var info = groups[String(module.group)];

        if (info && collapsed[String(module.group)] && info.first !== slot) {
          continue;
        }
      }

      var slotElement = makeSlot(data, bandIndex, slot, module);

      if (module) decorateGroup(slotElement, bandIndex, slot, module, groups, collapsed);

      slots.appendChild(slotElement);
    }

    section.appendChild(slots);
    slots.scrollLeft = slotScroll[bandIndex] || 0;
    slots.addEventListener("scroll", function () {
      slotScroll[bandIndex] = slots.scrollLeft;
    });

    var resizer = el("div", "band-resizer");
    resizer.title = "Drag to resize this band";
    section.appendChild(resizer);
    attachHeightResize(resizer, slots, 260, 900, "iamrack.bandHeight." + bandIndex);

    return section;
  }

  function svgNode(tag, attributes) {
    var node = document.createElementNS("http://www.w3.org/2000/svg", tag);
    for (var key in attributes) node.setAttribute(key, attributes[key]);
    return node;
  }

  function paintCurve(svg, values, color) {
    svg.textContent = "";

    svg.appendChild(svgNode("line", {
      x1: 8, y1: 32, x2: 292, y2: 32,
      stroke: "#333", "stroke-width": "1", "stroke-dasharray": "2 4"
    }));

    var path = "";

    for (var i = 0; i < values.length; i++) {
      var x = 8 + i / (values.length - 1) * 284;
      var y = 32 - Math.max(-1, Math.min(1, values[i])) * 22;
      path += (i ? " L " : "M ") + x.toFixed(1) + " " + y.toFixed(1);
    }

    svg.appendChild(svgNode("path", {
      d: path, fill: "none", stroke: color, "stroke-width": "2",
      "stroke-linecap": "round", "vector-effect": "non-scaling-stroke"
    }));
  }

  function lfoValue(shape, phase) {
    switch (((shape % 4) + 4) % 4) {
      case 1: return 1 - 4 * Math.abs(phase - 0.5);
      case 2: return 2 * phase - 1;
      case 3: return phase < 0.5 ? 1 : -1;
      default: return Math.sin(phase * Math.PI * 2);
    }
  }

  function tableValue(shape, morph, phase) {
    var first = lfoValue(shape, phase);
    var second = lfoValue(shape + 1, phase);
    return first + (second - first) * Math.max(0, Math.min(1, morph));
  }

  function lfoPreset(shape) {
    var raw;

    switch (((shape % 4) + 4) % 4) {
      case 1: raw = [[0, 0], [0.5, 1], [1, 0]]; break;
      case 2: raw = [[0, 0], [1, 1]]; break;
      case 3: raw = [[0, 1], [0.5, 1], [0.5001, 0], [1, 0]]; break;
      default: raw = [[0, 0.5], [0.25, 1], [0.5, 0.5], [0.75, 0], [1, 0.5]];
    }

    return raw.map(function (point) {
      return { time: point[0], value: point[1], bend: 0 };
    });
  }

  function lfoPointsFromJson(points, shape) {
    if (Array.isArray(points) && points.length >= 2) {
      return points.map(function (point) {
        return {
          time: Number(point[0]),
          value: Number(point[1]),
          bend: Number(point[2] === undefined ? 0 : point[2])
        };
      });
    }

    return lfoPreset(shape);
  }

  function curvePointsFromJson(points) {
    if (Array.isArray(points) && points.length >= 2) {
      return points.map(function (point) {
        return { x: Number(point[0]), y: Number(point[1]) };
      });
    }

    return [{ x: 0, y: 0 }, { x: 1, y: 1 }];
  }

  function serializeCurve(points) {
    return points.map(function (point) {
      return point.x.toFixed(4) + ":" + point.y.toFixed(4);
    }).join(";");
  }

  function drawMidiCurve(svg, points, save) {
    function eventPoint(event) {
      var rect = svg.getBoundingClientRect();
      return {
        x: Math.max(0, Math.min(1, (event.clientX - rect.left) / rect.width)),
        y: Math.max(0, Math.min(1, 1 - (event.clientY - rect.top) / rect.height))
      };
    }

    function at(event) {
      var position = eventPoint(event);

      for (var index = 0; index < points.length; index++) {
        if (Math.abs(points[index].x - position.x) < 0.035 &&
            Math.abs(points[index].y - position.y) < 0.12) {
          return points[index];
        }
      }

      return null;
    }

    function paint() {
      svg.textContent = "";

      for (var line = 0; line <= 4; line++) {
        svg.appendChild(svgNode("line", {
          x1: 8, y1: 10 + line * 11, x2: 292, y2: 10 + line * 11,
          stroke: "#2a2a2a", "stroke-width": "1"
        }));
      }

      var sorted = points.slice().sort(function (a, b) { return a.x - b.x; });
      var path = "";

      sorted.forEach(function (point, index) {
        path += (index ? " L " : "M ") + (8 + point.x * 284).toFixed(1) +
          " " + (54 - point.y * 44).toFixed(1);
      });

      svg.appendChild(svgNode("path", {
        d: path, fill: "none", stroke: "#ff6f91", "stroke-width": "2",
        "stroke-linecap": "round", "vector-effect": "non-scaling-stroke"
      }));

      sorted.forEach(function (point) {
        var circle = svgNode("circle", {
          cx: (8 + point.x * 284).toFixed(1),
          cy: (54 - point.y * 44).toFixed(1),
          r: "3.2", fill: "#ff6f91", stroke: "#1a1a1a", "stroke-width": "1.4"
        });
        circle.style.pointerEvents = "none";
        svg.appendChild(circle);
      });
    }

    var drag = null;

    function commit() {
      save(points);
    }

    svg.onpointerdown = function (event) {
      if (event.button !== 0) return;

      var point = at(event);

      if (!point) {
        var position = eventPoint(event);
        point = { x: position.x, y: position.y };
        points.push(point);
        points.sort(function (a, b) { return a.x - b.x; });
      }

      drag = { point: point };
      svg.setPointerCapture(event.pointerId);
      paint();
      event.preventDefault();
    };

    svg.onpointermove = function (event) {
      if (!drag) return;
      var position = eventPoint(event);
      drag.point.x = position.x;
      drag.point.y = position.y;
      points.sort(function (a, b) { return a.x - b.x; });
      paint();
      event.preventDefault();
    };

    function finish(event) {
      if (!drag) return;
      drag = null;
      commit();
      event.preventDefault();
    }

    svg.onpointerup = finish;
    svg.onpointercancel = finish;

    function remove(event) {
      var point = at(event);
      if (!point || points.length <= 2) return;
      var index = points.indexOf(point);
      if (index >= 0) points.splice(index, 1);
      paint();
      commit();
      event.preventDefault();
    }

    svg.ondblclick = remove;
    svg.oncontextmenu = remove;

    paint();
  }

  function serializeLfo(points) {
    return points.map(function (point) {
      return point.time.toFixed(4) + ":" + point.value.toFixed(4) +
        ":" + point.bend.toFixed(4);
    }).join(";");
  }

  function lfoBend(t, bend) {
    if (Math.abs(bend) < 0.0001) return t;
    return (Math.exp(t * bend) - 1) / (Math.exp(bend) - 1);
  }

  function drawLfo(svg, lfo, id) {
    function eventPoint(event) {
      var x;
      var y;
      var matrix = svg.getScreenCTM();

      if (matrix) {
        var pointer = svg.createSVGPoint();
        pointer.x = event.clientX;
        pointer.y = event.clientY;
        var local = pointer.matrixTransform(matrix.inverse());
        x = local.x;
        y = local.y;
      } else {
        var rect = svg.getBoundingClientRect();
        x = (event.clientX - rect.left) / rect.width * 300;
        y = (event.clientY - rect.top) / rect.height * 64;
      }

      return {
        x: x,
        y: y,
        time: Math.max(0, Math.min(1, (x - 8) / 284)),
        value: Math.max(0, Math.min(1, (60 - y) / 52))
      };
    }

    function pointAt(event) {
      var position = eventPoint(event);

      for (var i = 0; i < lfo.points.length; i++) {
        var point = lfo.points[i];
        var x = 8 + point.time * 284;
        var y = 60 - point.value * 52;

        if (Math.abs(x - position.x) < 9 && Math.abs(y - position.y) < 9) {
          return point;
        }
      }

      return null;
    }

    function bendAt(event) {
      var position = eventPoint(event);
      var points = lfo.points.slice().sort(function (a, b) {
        return a.time - b.time;
      });

      for (var i = 0; i < points.length - 1; i++) {
        var point = points[i];
        var next = points[i + 1];
        var time = (point.time + next.time) / 2;
        var value = point.value + (next.value - point.value) * lfoBend(0.5, point.bend);
        var x = 8 + time * 284;
        var y = 60 - value * 52;

        if (Math.abs(x - position.x) < 8 && Math.abs(y - position.y) < 8) {
          return point;
        }
      }

      return null;
    }

    function paint() {
      svg.textContent = "";

      svg.appendChild(svgNode("line", {
        x1: 8, y1: 60, x2: 292, y2: 60,
        stroke: "#333", "stroke-width": "1", "stroke-dasharray": "2 4"
      }));

      var points = lfo.points.slice().sort(function (a, b) {
        return a.time - b.time;
      });
      var path = "";

      for (var i = 0; i < points.length; i++) {
        if (i === 0) {
          path = "M " + (8 + points[i].time * 284).toFixed(1) + " " +
            (60 - points[i].value * 52).toFixed(1);
          continue;
        }

        var previous = points[i - 1];
        var point = points[i];

        for (var step = 1; step <= 14; step++) {
          var unit = step / 14;
          var time = previous.time + (point.time - previous.time) * unit;
          var value = previous.value +
            (point.value - previous.value) * lfoBend(unit, previous.bend);
          path += " L " + (8 + time * 284).toFixed(1) + " " +
            (60 - value * 52).toFixed(1);
        }
      }

      if (points.length > 1) {
        svg.appendChild(svgNode("path", {
          d: path, fill: "none", stroke: "#4c8dff", "stroke-width": "2",
          "stroke-linecap": "round", "vector-effect": "non-scaling-stroke"
        }));
      }

      points.forEach(function (point) {
        var circle = svgNode("circle", {
          cx: (8 + point.time * 284).toFixed(1),
          cy: (60 - point.value * 52).toFixed(1),
          r: "3.5", fill: "#4c8dff", stroke: "#1a1a1a", "stroke-width": "1.5"
        });
        circle.style.pointerEvents = "none";
        svg.appendChild(circle);
      });

      for (var bendIndex = 0; bendIndex < points.length - 1; bendIndex++) {
        var start = points[bendIndex];
        var end = points[bendIndex + 1];
        var bendTime = (start.time + end.time) / 2;
        var bendValue = start.value +
          (end.value - start.value) * lfoBend(0.5, start.bend);
        var handle = svgNode("circle", {
          cx: (8 + bendTime * 284).toFixed(1),
          cy: (60 - bendValue * 52).toFixed(1),
          r: "2.6", fill: "none", stroke: "#8fb8ff", "stroke-width": "1.4"
        });
        handle.style.pointerEvents = "none";
        svg.appendChild(handle);
      }
    }

    var drag = null;

    function commit() {
      sendNow("source." + id + ".points", serializeLfo(lfo.points)).then(load);
    }

    function removePoint(point) {
      if (lfo.points.length <= 2) return;

      var index = lfo.points.indexOf(point);
      if (index >= 0) lfo.points.splice(index, 1);
      paint();
      commit();
    }

    svg.onpointerdown = function (event) {
      if (event.button !== 0) return;

      var point = pointAt(event);

      if (point) {
        drag = { kind: "point", point: point };
      } else {
        var bend = bendAt(event);

        if (bend) {
          var sorted = lfo.points.slice().sort(function (a, b) {
            return a.time - b.time;
          });
          var index = sorted.indexOf(bend);
          var next = sorted[index + 1];
          var position = eventPoint(event);

          drag = {
            kind: "bend",
            point: bend,
            startValue: position.value,
            startBend: bend.bend,
            direction: next && next.value > bend.value ? -1 : 1
          };
        }
      }

      if (!drag) return;

      svg.setPointerCapture(event.pointerId);
      paint();
      event.preventDefault();
    };

    svg.onpointermove = function (event) {
      if (!drag) return;

      var position = eventPoint(event);

      if (drag.kind === "point") {
        drag.point.time = position.time;
        drag.point.value = position.value;
        lfo.points.sort(function (a, b) { return a.time - b.time; });
      } else {
        var delta = position.value - drag.startValue;
        drag.point.bend = Math.max(-10, Math.min(10,
          drag.startBend + delta * 20 * drag.direction));
      }

      paint();
      event.preventDefault();
    };

    function finish(event) {
      if (!drag) return;
      drag = null;
      commit();
      event.preventDefault();
    }

    svg.onpointerup = finish;
    svg.onpointercancel = finish;
    svg.ondblclick = function (event) {
      // Adding is deliberate: a single click starts a drag on a point or bend,
      // while an empty double-click appends a new point. Right-click still
      // removes an existing point.
      if (pointAt(event)) return;

      var position = eventPoint(event);
      if (position.x < 8 || position.x > 292 || position.y < 8 || position.y > 60) return;
      var point = { time: position.time, value: position.value, bend: 0 };
      lfo.points.push(point);
      lfo.points.sort(function (a, b) { return a.time - b.time; });
      paint();
      commit();
      event.preventDefault();
    };
    svg.oncontextmenu = function (event) {
      var point = pointAt(event);
      if (point) {
        removePoint(point);
        event.preventDefault();
      }
    };

    paint();
  }

  function drawEnvelope(svg) {
    svg.textContent = "";
    svg.appendChild(svgNode("line", {
      x1: 8, y1: 58, x2: 292, y2: 58,
      stroke: "#333", "stroke-width": "1", "stroke-dasharray": "2 4"
    }));
  }

  function drawRandom(svg) {
    svg.textContent = "";
    svg.appendChild(svgNode("line", {
      x1: 8, y1: 32, x2: 292, y2: 32,
      stroke: "#333", "stroke-width": "1", "stroke-dasharray": "2 4"
    }));
  }

  function drawTable(svg, table) {
    var values = [];

    for (var i = 0; i < 120; i++) {
      values.push(tableValue(table.shape, table.morph, (i / 120 * 2) % 1));
    }

    paintCurve(svg, values, "#c8e04a");
  }

  function liveTrace(svg, values, color, base, height) {
    svg.textContent = "";
    if (values.length < 2) return;

    var path = "";

    for (var i = 0; i < values.length; i++) {
      var x = 8 + i / (values.length - 1) * 284;
      var y = base - values[i] * height;
      path += (i ? " L " : "M ") + x.toFixed(1) + " " + y.toFixed(1);
    }

    svg.appendChild(svgNode("path", {
      d: path, fill: "none", stroke: color, "stroke-width": "2",
      "stroke-linecap": "round", "vector-effect": "non-scaling-stroke"
    }));
  }

  function drawLiveEnvelope(svg, live, id) {
    liveTrace(svg, (live.history[id] || []).slice(-120), "#35c77a", 58, 48);
  }

  function drawLiveRandom(svg, live, id) {
    liveTrace(svg, (live.history[id] || []).slice(-120), "#ff9a3d", 32, 22);
  }

  function drawLivePitch(svg, live, id) {
    liveTrace(svg, (live.history[id] || []).slice(-120), "#b08cff", 58, 48);
  }

  function drawLiveMidi(svg, live, id) {
    liveTrace(svg, (live.history[id] || []).slice(-120), "#ff6f91", 58, 48);
  }

  function drawLiveLfo(svg, live, id) {
    svg.textContent = "";

    var phase = live.phases[id] || 0;
    var value = live.values[id] || 0;
    var x = 8 + phase * 284;
    var y = 60 - ((value + 1) / 2) * 52;

    svg.appendChild(svgNode("line", {
      x1: x.toFixed(1), y1: 6, x2: x.toFixed(1), y2: 60,
      stroke: "#4c8dff", "stroke-opacity": ".35", "stroke-width": "1"
    }));
    svg.appendChild(svgNode("circle", {
      cx: x.toFixed(1), cy: y.toFixed(1), r: "3.5",
      fill: "#dce9ff", stroke: "#4c8dff", "stroke-width": "1.5"
    }));
  }

  function drawLiveTable(svg, live, id) {
    svg.textContent = "";

    var phase = live.phases[id] || 0;
    var value = live.values[id] || 0;
    var x = 8 + phase * 284;
    var y = 32 - value * 22;

    svg.appendChild(svgNode("line", {
      x1: x.toFixed(1), y1: 6, x2: x.toFixed(1), y2: 60,
      stroke: "#c8e04a", "stroke-opacity": ".35", "stroke-width": "1"
    }));
    svg.appendChild(svgNode("circle", {
      cx: x.toFixed(1), cy: y.toFixed(1), r: "3.5",
      fill: "#f4ffd0", stroke: "#c8e04a", "stroke-width": "1.5"
    }));
  }

  function makeMacroName(knob, slot, fallback) {
    var label = knob.querySelector(".knob-label");
    if (!label) return;

    label.title = "Double-click to rename";
    label.classList.add("macro-name");
    label.addEventListener("dblclick", function (event) {
      event.preventDefault();

      var previous = label.textContent;
      var input = document.createElement("input");
      input.className = "macro-rename";
      input.type = "text";
      input.maxLength = 24;
      input.value = previous === fallback ? "" : previous;
      input.setAttribute("aria-label", "Macro " + (slot + 1) + " name");
      label.textContent = "";
      label.appendChild(input);
      input.focus();
      input.select();

      var done = false;

      function finish(commit) {
        if (done) return;
        done = true;

        var next = input.value.trim().slice(0, 24);
        label.textContent = commit ? (next || fallback) : previous;

        if (commit) {
          sendNow("macro." + slot + ".name", next).then(load);
        }
      }

      input.addEventListener("blur", function () { finish(true); });
      input.addEventListener("keydown", function (event) {
        if (event.key === "Enter") {
          event.preventDefault();
          finish(true);
        } else if (event.key === "Escape") {
          event.preventDefault();
          finish(false);
        }
      });
    });
  }

  function makeMacroBar(data) {
    var section = el("section", "macrobar");
    section.appendChild(el("span", "macro-label", "MACROS"));

    var controls = el("div", "macro-controls");
    var macros = data.macros || [];
    var names = data.macro_names || [];

    for (var index = 0; index < 8; index++) {
      (function (slot) {
        var value = typeof macros[slot] === "number" ? macros[slot] : 0;
        var fallback = "M" + (slot + 1);
        var name = names[slot] || fallback;

        var item = el("div", "macro-item");

        if (selectedSource === 4 + slot) item.classList.add("selected");

        // A single click picks the macro; a double click renames it. Waiting out
        // the double-click window keeps the rename from being torn down by the
        // re-render a selection causes.
        var selectTimer = null;

        item.addEventListener("click", function (event) {
          if (event.target.closest("button, select, input, textarea, .macro-name")) return;

          clearTimeout(selectTimer);
          selectTimer = setTimeout(function () {
            selectSource(4 + slot);
          }, 260);
        });
        item.addEventListener("dblclick", function () { clearTimeout(selectTimer); });
        var handle = el("span", "macro-drag", "\u28FF");
        handle.draggable = true;
        handle.title = "Drag macro to a knob";
        handle.setAttribute("aria-label", "Drag macro to a knob");
        handle.addEventListener("dragstart", function (event) {
          event.dataTransfer.setData("text/plain", "macro:" + slot);
          event.dataTransfer.effectAllowed = "link";
        });

        var knobHandle = window.IAmCard.controlHandle({
          id: "macro." + slot,
          label: name,
          kind: "float",
          min: 0,
          max: 1,
          default: 0,
          unit: "percent"
        }, value, function (_, next) {
          macroValues[slot] = next;
          queue("macro." + slot, next);
        });

        var knob = knobHandle.el;
        macroHandles[slot] = knobHandle;
        macroValues[slot] = value;

        makeMacroName(knob, slot, fallback);

        item.appendChild(handle);
        item.appendChild(knob);
        controls.appendChild(item);
      })(index);
    }

    section.appendChild(controls);
    return section;
  }

  function sourceControl(card, spec, value, address, onChange) {
    card.appendChild(window.IAmCard.control(spec, value, function (_, next) {
      queue(address, next);
      if (onChange) onChange(next);
    }));
  }

  function sourceChoice(head, label, choices, value, onChange) {
    var box = el("div", "mcard-setting");
    box.appendChild(el("span", "mcard-setting-label", label));

    var select = document.createElement("select");
    select.className = "mcard-select mod-choice";
    select.setAttribute("aria-label", label);

    choices.forEach(function (name, index) {
      var option = document.createElement("option");
      option.value = String(index);
      option.textContent = name;
      select.appendChild(option);
    });

    select.value = String(Math.round(value));
    select.addEventListener("change", function () {
      onChange(Number(select.value));
    });

    box.appendChild(select);
    head.appendChild(box);
  }

  function makeSourceCard(title, accent, paint, build, livePaint, chrome) {
    chrome = chrome || {};
    var card = el("div", "mod-card");
    card.style.setProperty("--accent", accent);
    if (title.indexOf("MIDI") === 0) card.classList.add("midi-card");

    var key = chrome.key || title;

    if (collapsedSources[key]) card.classList.add("collapsed");

    var head = el("div", "mod-card-head");

    if (chrome.dragId !== undefined) {
      if (selectedSource === chrome.dragId) card.classList.add("selected");
      card.title = "Click to show only this source's modulation";

      card.addEventListener("click", function (event) {
        if (event.target.closest("button, select, input, textarea, a, .mod-drag")) return;
        selectSource(chrome.dragId);
      });

      var handle = el("span", "mod-drag", "\u28FF");
      handle.draggable = true;
      handle.title = "Drag source";
      handle.setAttribute("aria-label", "Drag source");
      handle.addEventListener("dragstart", function (event) {
        event.dataTransfer.setData("text/plain", "source:" + chrome.dragId);
        event.dataTransfer.effectAllowed = "move";
        card.classList.add("dragging");
      });
      handle.addEventListener("dragend", function () {
        card.classList.remove("dragging");
      });
      head.appendChild(handle);
    }

    var collapse = el("button", "mod-collapse", collapsedSources[key] ? "\u25B8" : "\u25BE");
    collapse.type = "button";
    collapse.title = collapsedSources[key] ? "Expand source" : "Collapse source";
    collapse.setAttribute("aria-label", collapse.title);
    collapse.addEventListener("click", function () {
      var next = !collapsedSources[key];
      collapsedSources[key] = next;
      card.classList.toggle("collapsed", next);
      collapse.textContent = next ? "\u25B8" : "\u25BE";
      collapse.title = next ? "Expand source" : "Collapse source";
      collapse.setAttribute("aria-label", collapse.title);
    });
    head.appendChild(collapse);

    if (chrome.onBypass) {
      if (chrome.bypass) card.classList.add("bypassed");

      var power = document.createElement("button");
      power.type = "button";
      power.className = "mcard-power";
      power.title = chrome.bypass ? "Enable source" : "Bypass source";
      power.setAttribute("aria-label", power.title);
      power.setAttribute("aria-pressed", chrome.bypass ? "false" : "true");
      if (chrome.bypass) power.classList.add("off");

      var dot = el("span", "dot");
      power.appendChild(dot);
      power.addEventListener("click", function () { chrome.onBypass(!chrome.bypass); });
      head.appendChild(power);
    }

    head.appendChild(el("span", "mod-card-title", title));
    head.appendChild(el("span", "mcard-spacer"));

    card.appendChild(head);

    var hero = el("div", "mod-hero");
    var svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
    svg.setAttribute("viewBox", "0 0 300 64");
    svg.setAttribute("role", "img");
    svg.setAttribute("aria-label", title + " modulation shape");
    if (title.indexOf("LFO") === 0) {
      svg.classList.add("lfo-editor");
      hero.title = "Double-click to add a point; drag points or the small bend ring; right-click to remove";
    } else if (title.indexOf("MIDI") === 0) {
      svg.classList.add("midi-editor");
      hero.title = "Click to add a curve point; drag to move; double-click or right-click to remove";
    }
    hero.appendChild(svg);

    var liveSvg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
    liveSvg.setAttribute("viewBox", "0 0 300 64");
    liveSvg.setAttribute("class", "mod-live");

    if (title.indexOf("MIDI") === 0) {
      svg.setAttribute("preserveAspectRatio", "none");
      liveSvg.setAttribute("preserveAspectRatio", "none");
    }
    liveSvg.setAttribute("aria-hidden", "true");
    hero.appendChild(liveSvg);
    card.appendChild(hero);

    if (livePaint) {
      livePainters.push(function () { livePaint(liveSvg, live); });
    }

    var controls = el("div", "mod-card-controls");
    build(controls, head, function () { paint(svg); });
    paint(svg);
    card.appendChild(controls);

    if (chrome.sync !== undefined) {
      var sync = el("button", "mcard-toggle mod-sync", "Sync");
      sync.type = "button";
      sync.title = chrome.sync ? "Rate follows the host tempo" : "Rate is in hertz";
      sync.setAttribute("aria-pressed", chrome.sync ? "true" : "false");
      sync.addEventListener("click", function () { chrome.onSync(!chrome.sync); });
      head.appendChild(sync);
    }

    if (chrome.onRemove) {
      var remove = el("button", "mod-remove", "\u00D7");
      remove.type = "button";
      remove.title = "Remove source";
      remove.setAttribute("aria-label", remove.title);
      remove.addEventListener("click", chrome.onRemove);
      head.appendChild(remove);
    }

    return card;
  }

  function numberOr(value, fallback) {
    return typeof value === "number" ? value : fallback;
  }

  function makeModulation(data) {
    var sources = data.sources || [];
    var section = el("section", "modulation");
    var head = el("div", "mod-head");

    head.appendChild(el("span", "mod-title", "MODULATION"));
    head.appendChild(el("span", "mod-hint", "source slots \u00B7 assignments \u00B7 macros"));
    section.appendChild(head);

    var grid = el("div", "mod-sources");
    grid.addEventListener("scroll", function () { sourceScroll = grid.scrollLeft; });
    grid.addEventListener("dragover", function (event) {
      var card = event.target.closest(".mod-card");
      if (!card) return;
      event.preventDefault();
      if (event.dataTransfer) event.dataTransfer.dropEffect = "move";
      grid.querySelectorAll(".mod-card.drop").forEach(function (candidate) {
        candidate.classList.remove("drop");
      });
      card.classList.add("drop");
    });
    grid.addEventListener("dragleave", function (event) {
      var card = event.target.closest(".mod-card");
      if (card) card.classList.remove("drop");
    });
    grid.addEventListener("drop", function (event) {
      event.preventDefault();
      var card = event.target.closest(".mod-card");
      if (!card) return;
      var data = event.dataTransfer ? event.dataTransfer.getData("text/plain") : "";
      if (!data || data.indexOf("source:") !== 0) return;
      var from = data.slice(7);
      var to = card.getAttribute("data-source-id");
      card.classList.remove("drop");
      if (!to || from === to) return;
      sendNow("source." + from + ".move", to).then(load);
    });

    sources.forEach(function (source) {
      var id = source.id;
      var settings = source.settings || {};
      var kind = source.kind;
      var same = sources.filter(function (candidate) { return candidate.kind === kind; });
      var title = source.name + (same.length > 1 ? " " + (same.indexOf(source) + 1) : "");
      var accent = kind === "lfo" ? "#4c8dff" :
        kind === "envelope" ? "#35c77a" :
          kind === "random" ? "#ff9a3d" :
            kind === "table" ? "#c8e04a" :
              kind === "pitch" ? "#b08cff" : "#ff6f91";
      var paint = function () {};
      var build = function () {};
      var livePaint = null;
      var chrome;

      var onBypass = function (next) {
        sendNow("source." + id + ".bypass", next ? 1 : 0).then(load);
      };
      var onSync = function (next) {
        sendNow("source." + id + ".sync", next ? 1 : 0).then(load);
      };
      var onRemove = function () {
        sendNow("source." + id + ".remove", 1).then(load);
      };

      if (kind === "lfo") {
        var lfo = {
          rate: numberOr(settings.rate, 1),
          shape: numberOr(settings.shape, 0),
          points: lfoPointsFromJson(settings.points, numberOr(settings.shape, 0)),
          sync: !!settings.sync,
          beats: numberOr(settings.beats, 1)
        };

        paint = function (svg) { drawLfo(svg, lfo, id); };
        livePaint = function (svg, live) { drawLiveLfo(svg, live, id); };
        build = function (card, head, refresh) {
          sourceChoice(head, "Shape", ["Sine", "Triangle", "Saw", "Square", "Custom"],
            lfo.shape, function (value) {
              lfo.shape = value;
              if (value <= 3) lfo.points = lfoPreset(value);
              refresh();
              sendNow("source." + id + ".shape", value).then(load);
            });

          if (lfo.sync) {
            sourceControl(card, {
              id: "lfo." + id + ".beats", label: "Time", kind: "float",
              min: 0.03125, max: 32, default: 1, unit: "beats", log: true
            }, lfo.beats, "source." + id + ".beats", function (value) {
              lfo.beats = value;
              refresh();
            });
          } else {
            sourceControl(card, {
              id: "lfo." + id + ".rate", label: "Rate", kind: "float",
              min: 0.01, max: 20, default: 1, unit: "hz", log: true
            }, lfo.rate, "source." + id + ".rate", function (value) {
              lfo.rate = value;
              refresh();
            });
          }
        };

        chrome = {
          key: "source-" + id,
          sync: lfo.sync,
          bypass: !!source.bypass,
          onSync: onSync,
          onBypass: onBypass,
          onRemove: onRemove
        };
      } else if (kind === "envelope") {
        var env = {
          attack: numberOr(settings.attack, 10),
          release: numberOr(settings.release, 100)
        };

        paint = function (svg) { drawEnvelope(svg, env); };
        livePaint = function (svg, live) { drawLiveEnvelope(svg, live, id); };
        build = function (card) {
          sourceControl(card, {
            id: "env." + id + ".attack", label: "Attack", kind: "float",
            min: 1, max: 500, default: 10, unit: "ms", log: true
          }, env.attack, "source." + id + ".attack", function (value) {
            env.attack = value;
          });
          sourceControl(card, {
            id: "env." + id + ".release", label: "Release", kind: "float",
            min: 1, max: 500, default: 100, unit: "ms", log: true
          }, env.release, "source." + id + ".release", function (value) {
            env.release = value;
          });
        };

        chrome = {
          key: "source-" + id,
          bypass: !!source.bypass,
          onBypass: onBypass,
          onRemove: onRemove
        };
      } else if (kind === "random") {
        var random = {
          rate: numberOr(settings.rate, 4),
          smooth: numberOr(settings.smooth, 80),
          sync: !!settings.sync,
          beats: numberOr(settings.beats, 1)
        };

        paint = function (svg) { drawRandom(svg, random); };
        livePaint = function (svg, live) { drawLiveRandom(svg, live, id); };
        build = function (card, head, refresh) {
          if (random.sync) {
            sourceControl(card, {
              id: "random." + id + ".beats", label: "Time", kind: "float",
              min: 0.03125, max: 32, default: 1, unit: "beats", log: true
            }, random.beats, "source." + id + ".beats", function (value) {
              random.beats = value;
              refresh();
            });
          } else {
            sourceControl(card, {
              id: "random." + id + ".rate", label: "Rate", kind: "float",
              min: 0.01, max: 20, default: 4, unit: "hz", log: true
            }, random.rate, "source." + id + ".rate", function (value) {
              random.rate = value;
              refresh();
            });
          }

          sourceControl(card, {
            id: "random." + id + ".smooth", label: "Smooth", kind: "float",
            min: 0, max: 500, default: 80, unit: "ms", log: true
          }, random.smooth, "source." + id + ".smooth", function (value) {
            random.smooth = value;
            refresh();
          });
        };

        chrome = {
          key: "source-" + id,
          sync: random.sync,
          bypass: !!source.bypass,
          onSync: onSync,
          onBypass: onBypass,
          onRemove: onRemove
        };
      } else if (kind === "table") {
        var table = {
          rate: numberOr(settings.rate, 1),
          shape: numberOr(settings.shape, 0),
          morph: numberOr(settings.morph, 0),
          sync: !!settings.sync,
          beats: numberOr(settings.beats, 1)
        };

        paint = function (svg) { drawTable(svg, table); };
        livePaint = function (svg, live) { drawLiveTable(svg, live, id); };
        build = function (card, head, refresh) {
          sourceChoice(head, "Shape", ["Sine", "Triangle", "Saw", "Square"],
            table.shape, function (value) {
              table.shape = value;
              refresh();
              sendNow("source." + id + ".shape", value).then(load);
            });

          if (table.sync) {
            sourceControl(card, {
              id: "table." + id + ".beats", label: "Time", kind: "float",
              min: 0.03125, max: 32, default: 1, unit: "beats", log: true
            }, table.beats, "source." + id + ".beats", function (value) {
              table.beats = value;
              refresh();
            });
          } else {
            sourceControl(card, {
              id: "table." + id + ".rate", label: "Rate", kind: "float",
              min: 0.01, max: 20, default: 1, unit: "hz", log: true
            }, table.rate, "source." + id + ".rate", function (value) {
              table.rate = value;
              refresh();
            });
          }

          sourceControl(card, {
            id: "table." + id + ".morph", label: "Morph", kind: "float",
            min: 0, max: 1, default: 0, unit: "percent"
          }, table.morph, "source." + id + ".morph", function (value) {
            table.morph = value;
            refresh();
          });
        };

        chrome = {
          key: "source-" + id,
          sync: table.sync,
          bypass: !!source.bypass,
          onSync: onSync,
          onBypass: onBypass,
          onRemove: onRemove
        };
      } else if (kind === "pitch") {
        var pitch = {
          clarity: numberOr(settings.clarity, 0.5),
          glide: numberOr(settings.glide, 40)
        };

        paint = function (svg) { drawEnvelope(svg); };
        livePaint = function (svg, live) { drawLivePitch(svg, live, id); };
        build = function (card) {
          sourceControl(card, {
            id: "pitch." + id + ".clarity", label: "Clarity", kind: "float",
            min: 0, max: 1, default: 0.5, unit: "percent"
          }, pitch.clarity, "source." + id + ".clarity", function (value) {
            pitch.clarity = value;
          });
          sourceControl(card, {
            id: "pitch." + id + ".glide", label: "Glide", kind: "float",
            min: 0, max: 200, default: 40, unit: "ms"
          }, pitch.glide, "source." + id + ".glide", function (value) {
            pitch.glide = value;
          });
        };

        chrome = {
          key: "source-" + id,
          bypass: !!source.bypass,
          onBypass: onBypass,
          onRemove: onRemove
        };
      } else if (kind === "midi") {
        var midi = {
          mode: numberOr(settings.mode, 0),
          transpose: numberOr(settings.transpose, 0),
          fine: numberOr(settings.fine, 0),
          note: curvePointsFromJson(settings.note_curve),
          velocity: curvePointsFromJson(settings.velocity_curve)
        };
        var curve = midi.mode === 1 ? midi.velocity : midi.note;
        var curveName = midi.mode === 1 ? "velocity_curve" : "note_curve";

        paint = function (svg) {
          drawMidiCurve(svg, curve, function (points) {
            sendNow("source." + id + "." + curveName, serializeCurve(points)).then(load);
          });
        };
        livePaint = function (svg, live) { drawLiveMidi(svg, live, id); };
        build = function (card, head) {
          sourceChoice(head, "Value", ["Note", "Velocity"], midi.mode, function (value) {
            sendNow("source." + id + ".mode", value).then(load);
          });

          sourceControl(card, {
            id: "midi." + id + ".transpose", label: "Transpose", kind: "int",
            min: -24, max: 24, default: 0, unit: "st"
          }, midi.transpose, "source." + id + ".transpose", function (value) {
            midi.transpose = Math.round(value);
          });

          sourceControl(card, {
            id: "midi." + id + ".fine", label: "Fine", kind: "float",
            min: -100, max: 100, default: 0, unit: "ct"
          }, midi.fine, "source." + id + ".fine", function (value) {
            midi.fine = value;
          });
        };

        chrome = {
          key: "source-" + id,
          bypass: !!source.bypass,
          onBypass: onBypass,
          onRemove: onRemove
        };
      }

      chrome.dragId = id;
      var card = makeSourceCard(title, accent, paint, build, livePaint, chrome);
      card.setAttribute("data-source-id", String(id));
      grid.appendChild(card);
    });

    var addCard = el("div", "slot-empty mod-add");
    var addSelect = document.createElement("select");
    addSelect.className = "mcard-select";
    addSelect.setAttribute("aria-label", "Add modulation source");

    var addPlaceholder = document.createElement("option");
    addPlaceholder.value = "";
    addPlaceholder.textContent = "+ Add source";
    addSelect.appendChild(addPlaceholder);

    [["lfo", "LFO"], ["envelope", "Envelope"], ["random", "Random"], ["table", "Table"],
      ["pitch", "Pitch"], ["midi", "MIDI"]]
      .forEach(function (entry) {
        var option = document.createElement("option");
        option.value = entry[0];
        option.textContent = entry[1];
        addSelect.appendChild(option);
      });

    addSelect.addEventListener("change", function () {
      if (!addSelect.value) return;
      sendNow("source.add", addSelect.value).then(load);
    });

    addCard.appendChild(addSelect);
    grid.appendChild(addCard);

    section.appendChild(grid);

    var resizer = el("div", "mod-resizer");
    resizer.title = "Drag to resize modulation sources";
    section.appendChild(resizer);
    attachHeightResize(resizer, grid, 220, 800, "iamrack.modHeight");

    return section;
  }

  function makeSourcesPage(data) {
    var page = el("div", "sources-page");
    page.appendChild(makeMacroBar(data));
    page.appendChild(makeModulation(data));
    return page;
  }

  /// The heading a matrix row belongs under: the band, the module, or the
  /// band's own gain and pan.
  function targetGroup(data, target) {
    if (!target) {
      return { key: "", label: "Unassigned", rows: [] };
    }

    var module = target.match(/^m(\d+)\./);

    if (module) {
      var id = Number(module[1]);
      var found = (data.modules || []).find(function (entry) { return entry.id === id; });
      var band = found ? ((data.bands || [])[found.band] || {}).name : null;

      return {
        key: "m" + id,
        label: band ? band + " \u00B7 " + found.kind : "Module " + id,
        rows: []
      };
    }

    var lane = target.match(/^band\.(\d+)\./);

    if (lane) {
      var index = Number(lane[1]);
      var name = ((data.bands || [])[index] || {}).name || ("Band " + (index + 1));

      return { key: "band" + index, label: name, rows: [] };
    }

    return { key: target, label: target, rows: [] };
  }

  function makeMatrix(data) {
    var section = el("section", "matrix-page");
    var head = el("div", "mod-head");
    head.appendChild(el("span", "mod-title", "MODULATION MATRIX"));

    // The filter is the same selection the source cards drive: picking a source
    // here is what picking its card does, and the knobs follow both.
    var filter = document.createElement("select");
    filter.className = "mcard-select matrix-filter";
    filter.setAttribute("aria-label", "Filter the matrix by source");

    var all = document.createElement("option");
    all.value = "";
    all.textContent = "All sources";
    filter.appendChild(all);

    (data.sources || []).forEach(function (source) {
      var option = document.createElement("option");
      option.value = String(source.id);
      option.textContent = sourceNames[source.id] || source.name;
      filter.appendChild(option);
    });

    (data.macro_names || []).forEach(function (name, index) {
      var option = document.createElement("option");
      option.value = String(4 + index);
      option.textContent = name || ("M" + (index + 1));
      filter.appendChild(option);
    });

    if (selectedSource !== null) filter.value = String(selectedSource);
    filter.addEventListener("change", function () {
      selectedSource = filter.value === "" ? null : Number(filter.value);
      load();
    });
    head.appendChild(filter);

    var count = el("span", "mod-hint", "");
    head.appendChild(count);
    head.appendChild(el("span", "mod-spacer"));

    var add = el("button", "mcard-toggle", "+ Row");
    add.type = "button";
    add.addEventListener("click", function () {
      sendNow("assign.add", 1).then(load);
    });
    head.appendChild(add);
    section.appendChild(head);

    var list = el("div", "assignments");
    var assignments = data.assignments || [];
    var total = 0;
    var shown = 0;
    var groups = [];

    assignments.forEach(function (assignment, slot) {
      if (!assignment) return;

      total += 1;

      if (selectedSource !== null && assignment.source !== selectedSource) return;

      shown += 1;

      var key = targetGroup(data, assignment.target).key;
      var group = null;

      for (var i = 0; i < groups.length; i++) {
        if (groups[i].key === key) {
          group = groups[i];
          break;
        }
      }

      if (!group) {
        group = targetGroup(data, assignment.target);
        groups.push(group);
      }

      group.rows.push({ slot: slot, assignment: assignment });
    });

    groups.forEach(function (group) {
      list.appendChild(el("div", "assign-group", group.label));

      group.rows.forEach(function (row) {
        list.appendChild(makeAssignment(data, row.slot, row.assignment));
      });
    });

    count.textContent = selectedSource === null
      ? total + " links"
      : shown + " of " + total + " links";

    if (!list.children.length) {
      list.appendChild(el("div", "matrix-empty", "No links yet. Drag a source or macro onto any modulatable knob."));
    }

    section.appendChild(list);
    return section;
  }

  function makeAutomation(data) {
    var section = el("section", "automation");
    var head = el("div", "mod-head");
    head.appendChild(el("span", "mod-title", "DAW PARAMETERS"));
    head.appendChild(el("span", "mod-hint", "32 fixed host slots \u00B7 bind manually"));
    section.appendChild(head);

    var grid = el("div", "automation-grid");
    var bindings = data.automation || [];

    for (var slot = 0; slot < 32; slot++) {
      (function (index) {
        var binding = bindings[index] || { slot: index, target: null, value: 0 };
        var row = el("div", "automation-row");
        row.appendChild(el("span", "automation-slot", "A" + String(index + 1).padStart(2, "0")));

        var target = document.createElement("select");
        target.className = "mcard-select automation-target";
        target.setAttribute("aria-label", "Automation " + (index + 1) + " target");

        var placeholder = document.createElement("option");
        placeholder.value = "";
        placeholder.textContent = "Unbound";
        target.appendChild(placeholder);

        (data.targets || []).forEach(function (entry) {
          var option = document.createElement("option");
          option.value = entry.id;
          option.textContent = entry.label;
          target.appendChild(option);
        });

        if (binding.target) target.value = binding.target;

        target.addEventListener("change", function () {
          if (!target.value) {
            sendNow("automation." + index + ".clear", 1).then(load);
          } else {
            sendNow("automation." + index + ".target", target.value).then(load);
          }
        });

        row.appendChild(target);

        var readout = el("span", "automation-value", Math.round((binding.value || 0) * 100) + "%");
        automationSpans[index] = readout;
        row.appendChild(readout);
        grid.appendChild(row);
      })(slot);
    }

    section.appendChild(grid);
    return section;
  }

  function makeAssignment(data, slot, assignment) {
    var row = el("div", "assign-row");
    var sources = data.sources || [];
    var sourceNames = [];
    var sourceValues = [];

    sources.forEach(function (entry) {
      var same = sources.filter(function (candidate) { return candidate.kind === entry.kind; });
      sourceNames.push(entry.name + (same.length > 1 ? " " + (same.indexOf(entry) + 1) : ""));
      sourceValues.push(String(entry.id));
    });

    var macroNames = data.macro_names || [];

    for (var macro = 1; macro <= 8; macro++) {
      var macroName = macroNames[macro - 1];
      sourceNames.push(macroName ? macroName : "Macro " + macro);
      sourceValues.push("macro" + macro);
    }

    row.appendChild(el("span", "assign-slot", String(slot + 1).padStart(2, "0")));

    var source = document.createElement("select");
    source.className = "mcard-select assign-select";
    source.setAttribute("aria-label", "Modulation source");
    var placeholder = document.createElement("option");
    placeholder.value = "";
    placeholder.textContent = "\u2014";
    source.appendChild(placeholder);

    sourceValues.forEach(function (value, index) {
      var option = document.createElement("option");
      option.value = value;
      option.textContent = sourceNames[index];
      source.appendChild(option);
    });

    if (assignment) {
      source.value = assignment.source >= 1000000000
        ? "macro" + (assignment.source - 1000000000 + 1)
        : String(assignment.source);
    }

    source.addEventListener("change", function () {
      if (!source.value) {
        sendNow("assign." + slot + ".clear", 1).then(load);
      } else {
        sendNow("assign." + slot + ".source", source.value).then(load);
      }
    });
    row.appendChild(source);

    var target = document.createElement("select");
    target.className = "mcard-select assign-target";
    target.setAttribute("aria-label", "Modulation target");
    target.disabled = !assignment;

    var targetPlaceholder = document.createElement("option");
    targetPlaceholder.value = "";
    targetPlaceholder.textContent = "Target\u2026";
    target.appendChild(targetPlaceholder);

    (data.targets || []).forEach(function (entry) {
      var option = document.createElement("option");
      option.value = entry.id;
      option.textContent = entry.label;
      target.appendChild(option);
    });

    if (assignment && assignment.target) target.value = assignment.target;

    target.addEventListener("change", function () {
      if (!target.value) {
        sendNow("assign." + slot + ".clear", 1).then(load);
      } else {
        sendNow("assign." + slot + ".target", target.value).then(load);
      }
    });
    row.appendChild(target);

    var fromBase = assignment && assignment.mode === "frombase";
    var mode = el("button", "assign-mode" + (fromBase ? " from-base" : ""), fromBase ? "S" : "C");
    mode.type = "button";
    mode.disabled = !assignment;
    mode.title = fromBase
      ? "Modulation starts at the current value"
      : "Modulation is centred on the current value";
    mode.setAttribute("aria-label", mode.title);
    mode.addEventListener("click", function () {
      sendNow("assign." + slot + ".mode", fromBase ? "center" : "start").then(load);
    });
    row.appendChild(mode);

    var depth = document.createElement("input");
    depth.type = "range";
    depth.className = "assign-depth";
    depth.min = "-1";
    depth.max = "1";
    depth.step = "0.01";
    depth.value = String(assignment ? assignment.depth : 0);
    depth.disabled = !assignment;
    depth.setAttribute("aria-label", "Modulation depth");

    var value = el("span", "assign-value",
      assignment ? Math.round(assignment.depth * 100) + "%" : "\u2014");

    var depthWrap = el("div", "assign-depth-wrap");
    depthWrap.appendChild(depth);
    depthWrap.appendChild(value);
    row.appendChild(depthWrap);

    depth.addEventListener("input", function () {
      value.textContent = Math.round(Number(depth.value) * 100) + "%";
      queue("assign." + slot + ".depth", depth.value);
    });

    var clear = el("button", "assign-clear", "\u00D7");
    clear.type = "button";
    clear.title = "Clear assignment";
    clear.setAttribute("aria-label", "Clear assignment");
    clear.disabled = !assignment;
    clear.addEventListener("click", function () {
      sendNow("assign." + slot + ".clear", 1).then(load);
    });
    row.appendChild(clear);

    return row;
  }

  function showPanel(name) {
    if (!panelPages) return;
    activePanel = name;
    panelTabs.forEach(function (tab) {
      tab.classList.toggle("on", tab.getAttribute("data-panel") === name);
    });
    panelBody.textContent = "";
    panelBody.appendChild(panelPages[name]);

    if (name === "sources") {
      var sources = panelBody.querySelector(".mod-sources");
      if (sources) sources.scrollLeft = sourceScroll;
    }
  }

  function render(data) {
    current = data;
    rangeLabels = [];
    livePainters = [];
    liveCards = [];
    sourceNames = Object.create(null);
    sourceColors = Object.create(null);

    var counts = Object.create(null);
    var seen = Object.create(null);

    (data.sources || []).forEach(function (source) {
      counts[source.kind] = (counts[source.kind] || 0) + 1;
    });

    (data.sources || []).forEach(function (source) {
      seen[source.kind] = (seen[source.kind] || 0) + 1;
      sourceNames[source.id] =
        source.name + (counts[source.kind] > 1 ? " " + seen[source.kind] : "");
      sourceColors[source.id] = sourceAccent(source.kind);
    });

    (data.macro_names || []).forEach(function (name, index) {
      sourceNames[4 + index] = name || ("M" + (index + 1));
      sourceColors[4 + index] = "#9a9aff";
    });
    pageRate = data.rate;
    rack.textContent = "";

    rack.appendChild(makeCrossover(data));
    data.bands.forEach(function (band, index) {
      rack.appendChild(makeBand(data, index));
    });

    rack.querySelectorAll(".band .slots").forEach(function (element, index) {
      element.scrollLeft = slotScroll[index] || 0;
    });

    panelPages = {
      sources: makeSourcesPage(data),
      matrix: makeMatrix(data),
      daw: makeAutomation(data)
    };
    showPanel(activePanel);

    updateRanges(data);

    var mods = (data.assignments || []).filter(function (assignment) {
      return assignment;
    }).length;

    rate.textContent = (data.rate / 1000).toFixed(1) + " kHz";
    status.textContent = data.modules.length + " modules \u00B7 " + data.bands.length +
      " lanes \u00B7 " + mods + " mods";
    refreshEditSlots(data);
  }

  function load() {
    var top = rack.scrollTop;

    return fetch("/rack", { cache: "no-store" })
      .then(function (response) {
        if (!response.ok) throw new Error("rack " + response.status);
        return response.json();
      })
      .then(function (data) {
        render(data);
        rack.scrollTop = top;
      })
      .catch(function (error) {
        status.textContent = "rack unavailable: " + error.message;
      });
  }

  /// The level of a scope ring in dBFS, which is the unit the dynamics heroes
  /// mark their traces in.
  function levelOf(samples) {
    if (!samples || !samples.length) return null;

    var energy = 0;
    for (var i = 0; i < samples.length; i++) energy += samples[i] * samples[i];

    return 20 * Math.log10(Math.max(Math.sqrt(energy / samples.length), 1e-6));
  }

  function repaint(entry) {
    if (entry.card.repaintHero) entry.card.repaintHero();
  }

  /// One card's own input and output: the document is the same three rings the
  /// module scope keeps, so a card never shows another module's audio.
  function fetchScope(entry) {
    entry.fetching = true;

    fetch("/module-scope?id=" + entry.id, { cache: "no-store" })
      .then(function (response) {
        return response.ok ? response.arrayBuffer() : null;
      })
      .then(function (buffer) {
        if (!buffer) return;

        var data = new Float32Array(buffer);
        var third = data.length / 3;
        if (third <= 64) return;

        entry.context.scope.input = data.subarray(0, third);
        entry.context.scope.output = data.subarray(third, third * 2);
        entry.context.scope.sidechain = data.subarray(third * 2);
        entry.context.meter.input = levelOf(entry.context.scope.input);
        entry.context.meter.sidechain = levelOf(entry.context.scope.sidechain);
        pushEnvelope(entry.context);
        entry.context.rate = pageRate;
        entry.context.eqSpectrum.rate = pageRate;
        repaint(entry);
      })
      .catch(function () {})
      .then(function () {
        entry.fetching = false;
      });
  }

  /// The goniometer wants both channels before and after the module; the mono
  /// document cannot tell it anything about the stereo image.
  function fetchStereo(entry) {
    fetch("/module-stereo?id=" + entry.id, { cache: "no-store" })
      .then(function (response) {
        return response.ok ? response.arrayBuffer() : null;
      })
      .then(function (buffer) {
        if (!buffer) return;

        var data = new Float32Array(buffer);
        var points = data.length / 4;
        if (points <= 64) return;

        entry.context.stereo.inputLeft = data.subarray(0, points);
        entry.context.stereo.inputRight = data.subarray(points, points * 2);
        entry.context.stereo.outputLeft = data.subarray(points * 2, points * 3);
        entry.context.stereo.outputRight = data.subarray(points * 3);
        repaint(entry);
      })
      .catch(function () {});
  }

  /// Asks the person for an impulse response file and posts its bytes to the
  /// module. The page reads the file itself, so the rack never opens a native
  /// dialog and the host window cannot end up behind the DAW.
  function pickImpulse(entry) {
    var input = document.createElement("input");
    input.type = "file";
    input.accept = ".wav,.aif,.aiff,.flac,audio/*";

    input.addEventListener("change", function () {
      var file = input.files && input.files[0];
      if (!file) return;

      file
        .arrayBuffer()
        .then(function (bytes) {
          return fetch(
            "/module-ir?id=" + entry.id + "&name=" + encodeURIComponent(file.name),
            { method: "POST", body: bytes }
          );
        })
        .then(function () { return fetchIr(entry); })
        .catch(function () {});
    });

    input.click();
  }

  /// The impulse response a convolver module holds, with its picture.
  function fetchIr(entry) {
    fetch("/module-ir?id=" + entry.id, { cache: "no-store" })
      .then(function (response) {
        return response.ok ? response.json() : null;
      })
      .then(function (value) {
        if (!value) return;

        var changed = JSON.stringify(entry.context.ir) !== JSON.stringify(value);
        entry.context.ir = value;

        if (changed) repaint(entry);
      })
      .catch(function () {});
  }

  /// The source a scripted module runs, for its editor. The poll only refreshes
  /// the error line, so it cannot overwrite what a person is typing.
  function loadScript(entry, force) {
    fetch("/module-script?id=" + entry.id, { cache: "no-store" })
      .then(function (response) {
        return response.ok ? response.json() : null;
      })
      .then(function (value) {
        if (!value) return;

        entry.context.script = value;
        entry.card.setScript(value.source, !!force);

        if (entry.card.setScriptError) entry.card.setScriptError(value.error);
      })
      .catch(function () {});
  }

  /// Posts a new source; the worker renders it on its own thread.
  function applyScript(entry, source) {
    fetch("/module-script?id=" + entry.id, {
      method: "POST",
      body: source
    })
      .then(function () { loadScript(entry); })
      .catch(function () {});
  }

  /// The curve a script renders right now, which the card's picture draws.
  function fetchCurve(entry) {
    fetch("/module-curve?id=" + entry.id, { cache: "no-store" })
      .then(function (response) {
        return response.ok ? response.json() : null;
      })
      .then(function (value) {
        if (!value || !value.points) return;

        var changed = JSON.stringify(entry.context.curve) !== JSON.stringify(value);
        entry.context.curve = value;
        entry.context.mapping = value;

        if (changed) repaint(entry);
      })
      .catch(function () {});
  }

  function pollImpulses() {
    if (!document.hidden) {
      liveCards.forEach(function (entry) {
        if (entry.ir) fetchIr(entry);
        if (entry.script) {
          fetchCurve(entry);
          loadScript(entry, false);
        }
      });
    }

    irTimer = setTimeout(pollImpulses, 250);
  }

  /// The band meter a module draws: the OTT's three gain bars, or the vocoder's
  /// input and output level per band.
  function fetchBands(entry) {
    fetch("/module-bands?id=" + entry.id, { cache: "no-store" })
      .then(function (response) {
        return response.ok ? response.json() : null;
      })
      .then(function (value) {
        if (!value) return;

        var changed;

        if (value.gains) {
          changed = JSON.stringify(entry.context.ott) !== JSON.stringify({ bands: value.input });
          entry.context.ott = { bands: value.input };
        } else {
          var next = { bands: value.bands, input: value.input, output: value.output };
          changed = JSON.stringify(entry.context.vocoder) !== JSON.stringify(next);
          entry.context.vocoder = next;
        }

        if (changed) repaint(entry);
      })
      .catch(function () {});
  }

  /// The RMS of a block of samples, in the 0..1 the heroes plot.
  function levelOf(samples) {
    if (!samples || !samples.length) return 0;

    var energy = 0;
    for (var i = 0; i < samples.length; i++) energy += samples[i] * samples[i];

    return Math.sqrt(energy / samples.length);
  }

  /// Fills a card's envelope history: the first scope frame seeds a whole window
  /// so the picture starts complete, and every later frame appends one step.
  function pushEnvelope(context) {
    var scope = context.scope;
    var history = context.envHistory;

    if (!scope || !scope.input || !scope.output || !history) return;

    if (!history.input.length) {
      var segments = 128;
      var step = Math.max(1, Math.floor(scope.input.length / segments));

      for (var s = 0; s < segments; s++) {
        var start = Math.min(s * step, scope.input.length - step);
        history.input.push(levelOf(scope.input.subarray(start, start + step)));
        history.output.push(levelOf(scope.output.subarray(start, start + step)));
      }

      context.envWindow = segments;
      return;
    }

    var size = Math.min(1024, scope.input.length);
    var begin = scope.input.length - size;

    if (history.input.length >= 512) {
      history.input.shift();
      history.output.shift();
    }

    history.input.push(levelOf(scope.input.subarray(begin, begin + size)));
    history.output.push(levelOf(scope.output.subarray(begin, begin + size)));
  }

  /// The note a pitch-tracking module is hearing, and the mask it tunes to: the
  /// card shows the note in its header, the autotune hero lights the mask.
  function fetchNote(entry) {
    fetch("/module-note?id=" + entry.id, { cache: "no-store" })
      .then(function (response) {
        return response.ok ? response.json() : null;
      })
      .then(function (value) {
        if (!value) return;

        if (entry.card.setNote) {
          // A chord shows every note it is playing, not just the last one.
          var held = value.notes && value.notes.length
            ? value.notes
            : (value.note >= 0 ? [value.note] : []);

          entry.card.setNote(held);
        }

        if (typeof value.allowed === "number") {
          var changed =
            !entry.context.autotune || entry.context.autotune.notes !== value.allowed;
          entry.context.autotune = { notes: value.allowed };
          if (changed) repaint(entry);
        }
      })
      .catch(function () {});
  }

  /// Only the cards on screen and drawing audio are polled, one document each,
  /// so a rack of twenty-seven modules costs what the visible few cost.
  function pollCards() {
    if (!document.hidden) {
      liveCards.forEach(function (entry) {
        if (entry.fetching) return;
        if (!entry.card.isConnected || entry.card.offsetParent === null) return;

        // The audio polls are for the cards that draw audio; the note poll is
        // for the cards that follow the host's notes, and they do not overlap:
        // an autotune has nothing to draw but its mask.
        if (entry.channels) {
          fetchScope(entry);
          if (entry.channels === "stereo") fetchStereo(entry);
        }

        if (entry.note) fetchNote(entry);
        if (entry.meter) fetchBands(entry);
      });
    }

    cardTimer = setTimeout(pollCards, 60);
  }

  function startLive() {
    if (liveTimer) clearInterval(liveTimer);

    liveTimer = setInterval(function () {
      fetch("/modulation", { cache: "no-store" })
        .then(function (response) {
          if (!response.ok) throw new Error("modulation " + response.status);
          return response.json();
        })
        .then(function (data) {
          (data.sources || []).forEach(function (source) {
            var id = String(source.id);
            live.values[id] = source.value;
            live.phases[id] = source.phase;

            if (!live.history[id]) live.history[id] = [];
            live.history[id].push(source.value);
            if (live.history[id].length > 160) live.history[id].shift();
          });

          applyHostValues(data);

          liveCards.forEach(function (entry) {
            if (!entry.card || !entry.card.isConnected) return;

            var values = liveModulation(entry);
            if (values && entry.card.setLiveValues) entry.card.setLiveValues(values);
          });

          livePainters.forEach(function (paint) { paint(); });
        })
        .catch(function () {
          // The live route is absent in a static preview; the static heroes stay.
        });
    }, 50);
  }

  /// Pull the host's macro and automation values into what the page shows. A
  /// pointer that is down owns the control it grabbed, so a drag is never
  /// fought by a value arriving from the host.
  function applyHostValues(data) {
    if (pointerDown) return;

    (data.macros || []).forEach(function (value, index) {
      var handle = macroHandles[index];

      if (!handle || typeof value !== "number") return;
      if (Math.abs((macroValues[index] || 0) - value) < 0.0005) return;

      macroValues[index] = value;
      handle.set(value);
    });

    (data.automation || []).forEach(function (value, index) {
      var readout = automationSpans[index];

      if (!readout || typeof value !== "number") return;

      var text = Math.round(value * 100) + "%";
      if (readout.textContent !== text) readout.textContent = text;
    });
  }

  function panelLimits() {
    return { low: 140, high: Math.max(180, window.innerHeight - 150) };
  }

  function setPanelHeight(height, persist) {
    var limits = panelLimits();
    var clamped = Math.max(limits.low, Math.min(limits.high, height));
    bottomPanel.style.flex = "0 0 " + clamped + "px";

    if (persist) {
      try { localStorage.setItem("iamrack.panelHeight", String(Math.round(clamped))); } catch (ignore) {}
    }
  }

  /// Fold the panel down to its tab strip, or bring it back at the height it
  /// had. The strip stays: it is the handle that reopens the panel.
  function applyPanelCollapsed(collapsed, persist) {
    if (collapsed && !panelCollapsed) {
      panelRestore = Math.round(bottomPanel.getBoundingClientRect().height);
    }

    panelCollapsed = collapsed;
    bottomPanel.classList.toggle("collapsed", collapsed);
    document.body.classList.toggle("panel-collapsed", collapsed);

    if (panelToggle) {
      panelToggle.setAttribute("aria-expanded", String(!collapsed));
      panelToggle.title = collapsed ? "Expand the control panel" : "Collapse the control panel";
    }

    if (collapsed) {
      bottomPanel.style.flex = "";
    } else if (panelRestore > 0) {
      setPanelHeight(panelRestore, false);
    }

    if (persist) {
      try {
        localStorage.setItem("iamrack.panelCollapsed", collapsed ? "1" : "0");
      } catch (ignore) {}
    }
  }

  function resetPanelHeight() {
    bottomPanel.style.flex = "";
    try { localStorage.removeItem("iamrack.panelHeight"); } catch (ignore) {}
  }

  try {
    var savedPanelHeight = Number(localStorage.getItem("iamrack.panelHeight"));
    if (savedPanelHeight > 0) setPanelHeight(savedPanelHeight, false);
  } catch (ignore) {}

  if (panelToggle) {
    panelToggle.addEventListener("click", function (event) {
      event.stopPropagation();
      applyPanelCollapsed(!panelCollapsed, true);
    });
  }

  // Double-clicking the strip itself does the same; the tabs keep their job.
  var panelStrip = bottomPanel.querySelector(".panel-tabs");

  if (panelStrip) {
    panelStrip.addEventListener("dblclick", function (event) {
      if (event.target.closest(".panel-toggle, .panel-tab")) return;
      applyPanelCollapsed(!panelCollapsed, true);
    });
  }

  try {
    if (localStorage.getItem("iamrack.panelCollapsed") === "1") {
      applyPanelCollapsed(true, false);
    }
  } catch (ignore) {}

  // While a pointer is down, the control under it belongs to the person.
  document.addEventListener("pointerdown", function () { pointerDown = true; }, true);
  document.addEventListener("pointerup", function () { pointerDown = false; }, true);
  document.addEventListener("pointercancel", function () { pointerDown = false; }, true);

  var panelDrag = null;

  splitter.addEventListener("pointerdown", function (event) {
    if (event.button !== 0) return;

    if (panelCollapsed) applyPanelCollapsed(false, true);

    panelDrag = {
      y: event.clientY,
      height: bottomPanel.getBoundingClientRect().height
    };
    splitter.classList.add("dragging");
    splitter.setPointerCapture(event.pointerId);
    event.preventDefault();
  });

  splitter.addEventListener("pointermove", function (event) {
    if (!panelDrag) return;
    setPanelHeight(panelDrag.height - (event.clientY - panelDrag.y), false);
    event.preventDefault();
  });

  function stopPanelDrag(event) {
    if (!panelDrag) return;
    panelDrag = null;
    splitter.classList.remove("dragging");
    setPanelHeight(bottomPanel.getBoundingClientRect().height, true);
    try { splitter.releasePointerCapture(event.pointerId); } catch (ignore) {}
  }

  splitter.addEventListener("pointerup", stopPanelDrag);
  splitter.addEventListener("pointercancel", stopPanelDrag);
  splitter.addEventListener("dblclick", resetPanelHeight);
  splitter.addEventListener("keydown", function (event) {
    if (event.key === "ArrowUp" || event.key === "ArrowDown") {
      var delta = event.key === "ArrowUp" ? 20 : -20;
      setPanelHeight(bottomPanel.getBoundingClientRect().height + delta, true);
      event.preventDefault();
    }
  });

  window.addEventListener("resize", function () {
    if (bottomPanel.style.flex) {
      setPanelHeight(bottomPanel.getBoundingClientRect().height, false);
    }
  });

  panelTabs.forEach(function (tab) {
    tab.addEventListener("click", function () {
      showPanel(tab.getAttribute("data-panel"));
    });
  });

  undoButton.addEventListener("click", function () { historyStep("undo"); });
  redoButton.addEventListener("click", function () { historyStep("redo"); });
  abButton.addEventListener("click", function () {
    runSpecial("ab.toggle", "1").then(load);
  });

  presetInit.addEventListener("click", function () {
    runSpecial("preset.init", "1").then(load);
  });

  presetSave.addEventListener("click", function () {
    flush()
      .then(function () {
        return fetch("/state", { cache: "no-store" });
      })
      .then(function (response) {
        if (!response.ok) throw new Error("state " + response.status);
        return response.text();
      })
      .then(function (text) {
        var blob = new Blob([text], { type: "application/json" });
        var url = URL.createObjectURL(blob);
        var link = document.createElement("a");
        link.href = url;
        link.download = "I Am Multipass.json";
        document.body.appendChild(link);
        link.click();
        link.remove();
        setTimeout(function () { URL.revokeObjectURL(url); }, 1000);
        status.textContent = "preset saved";
      })
      .catch(function (error) {
        status.textContent = error.message;
      });
  });

  presetLoad.addEventListener("click", function () {
    presetFile.click();
  });

  presetFile.addEventListener("change", function () {
    var file = presetFile.files && presetFile.files[0];
    presetFile.value = "";
    if (!file) return;

    file.text()
      .then(function (text) {
        var parsed;

        try {
          parsed = JSON.parse(text);
        } catch (error) {
          status.textContent = "not a preset file";
          return;
        }

        return runSpecial("preset.load", JSON.stringify(parsed)).then(load);
      })
      .catch(function (error) {
        status.textContent = error.message;
      });
  });

  document.addEventListener("keydown", function (event) {
    if (!(event.ctrlKey || event.metaKey) || event.altKey) return;

    var target = event.target;
    if (target && (target.isContentEditable ||
      /^(INPUT|TEXTAREA|SELECT)$/.test(target.tagName))) {
      return;
    }

    var key = String(event.key).toLowerCase();

    if (key === "z") {
      event.preventDefault();
      historyStep(event.shiftKey ? "redo" : "undo");
    } else if (key === "y") {
      event.preventDefault();
      historyStep("redo");
    }
  });


  power.addEventListener("click", function () {
    bypassed = !bypassed;
    power.classList.toggle("off", bypassed);
    power.setAttribute("aria-pressed", bypassed ? "true" : "false");
    document.body.classList.toggle("bypassed", bypassed);
    if (window.ipc) window.ipc.postMessage("__bypass=" + (bypassed ? 1 : 0));
  });

  load();
  startLive();
  pollCards();
  pollImpulses();
})();
