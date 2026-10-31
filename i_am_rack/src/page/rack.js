(function () {
  "use strict";

  var rack = document.getElementById("rack");
  var status = document.getElementById("status");
  var rate = document.getElementById("rate");
  var power = document.getElementById("power");

  var pending = Object.create(null);
  var timer = null;
  var bypassed = false;
  var current = null;
  var rangeLabels = [];
  var paintCrossover = null;
  var collapsedBands = Object.create(null);

  function el(tag, className, text) {
    var node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined) node.textContent = text;
    return node;
  }

  function queue(address, value) {
    pending[address] = address + "=" + value;
    if (timer) clearTimeout(timer);
    timer = setTimeout(flush, 60);
  }

  function flush() {
    if (timer) {
      clearTimeout(timer);
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
        if (!result.ok) status.textContent = "some commands were ignored";
      })
      .catch(function (error) {
        status.textContent = error.message;
      });
  }

  function sendNow(address, value) {
    pending[address] = address + "=" + value;
    return flush();
  }

  function formatHz(value) {
    if (value >= 1000) {
      return (value / 1000).toFixed(value >= 10000 ? 1 : 2) + " kHz";
    }
    return Math.round(value) + " Hz";
  }

  function bandRange(data, index) {
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

  function modulationsFor(data, target) {
    return (data.assignments || []).filter(function (assignment) {
      return assignment && assignment.target === target;
    }).map(function (assignment) {
      return { source: assignment.source, depth: assignment.depth };
    });
  }

  function makeCrossover(data) {
    var section = el("section", "crossover");
    var controls = el("div", "crossover-controls");
    controls.appendChild(el("span", "crossover-label", "CROSSOVER"));

    var low = window.IAmCard.control({
      id: "split.low",
      label: "Low / Mid",
      kind: "float",
      min: 20,
      max: 20000,
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
      max: 20000,
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

    section.appendChild(el("span", "crossover-note", "LR4 \u00B7 3 bands"));
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

    head.appendChild(el("span", "band-name", band.name));

    var range = el("span", "band-range", bandRange(data, bandIndex));
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
      modulations: modulationsFor(data, "band." + bandIndex + ".gain")
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
      modulations: modulationsFor(data, "band." + bandIndex + ".pan")
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

  function makeSlot(data, bandIndex, slotIndex, module) {
    var wrap = el("div", "slot");
    wrap.appendChild(el("div", "slot-label", slotIndex ? "SLOT B" : "SLOT A"));

    if (!module) {
      var empty = el("div", "slot-empty");
      var select = document.createElement("select");
      select.className = "slot-add";
      select.setAttribute("aria-label", "Add module");

      var placeholder = document.createElement("option");
      placeholder.value = "";
      placeholder.textContent = "+ Add module";
      select.appendChild(placeholder);

      (data.kinds || []).forEach(function (kind) {
        var option = document.createElement("option");
        option.value = kind.name;
        option.textContent = kind.name;
        select.appendChild(option);
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

    var kinds = (data.kinds || []).map(function (kind) { return kind.name; });

    var card = window.IAmCard.create({
      title: module.kind,
      badge: slotIndex ? "B" : "A",
      accent: module.accent,
      hero: module.hero,
      mode: module.mode,
      header: module.header,
      visible: module.visible,
      bypass: module.bypass,
      controls: module.controls,
      values: module.values,
      kinds: kinds,
      kind: module.kind,
      onKind: function (kind) {
        sendNow("slot." + bandIndex + "." + slotIndex + ".kind", kind).then(load);
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
    return wrap;
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

    for (var slot = 0; slot < 2; slot++) {
      var module = data.modules.find(function (candidate) {
        return candidate.band === bandIndex && candidate.slot === slot;
      }) || null;
      slots.appendChild(makeSlot(data, bandIndex, slot, module));
    }

    section.appendChild(slots);
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

  function drawLfo(svg, lfo) {
    var cycles = Math.max(1, Math.min(4, Math.round(lfo.rate || 1)));
    var values = [];

    for (var i = 0; i < 120; i++) {
      values.push(lfoValue(lfo.shape, (i / 120 * cycles) % 1));
    }

    paintCurve(svg, values, "#4c8dff");
  }

  function drawEnvelope(svg, env) {
    var attack = Math.max(1, env.attack || 10);
    var release = Math.max(1, env.release || 100);
    var knee = Math.max(0.08, Math.min(0.92, attack / (attack + release)));
    var values = [];

    for (var i = 0; i < 120; i++) {
      var t = i / 119;
      values.push(t <= knee ? t / knee : Math.exp(-4 * (t - knee) / (1 - knee)));
    }

    paintCurve(svg, values, "#35c77a");
  }

  function drawRandom(svg, random) {
    var steps = Math.max(4, Math.min(32, Math.round((random.rate || 4) * 2)));
    var alpha = 0.08 + 0.65 * Math.exp(-(random.smooth || 0) / 120);
    var current = 0;
    var values = [];

    for (var i = 0; i < 120; i++) {
      var step = Math.floor(i / 120 * steps);
      var seed = Math.sin(step * 12.9898 + 4.1414) * 43758.5453;
      var target = (seed - Math.floor(seed)) * 2 - 1;
      current += (target - current) * alpha;
      values.push(current);
    }

    paintCurve(svg, values, "#ff9a3d");
  }

  function drawTable(svg, table) {
    var values = [];

    for (var i = 0; i < 120; i++) {
      values.push(tableValue(table.shape, table.morph, (i / 120 * 2) % 1));
    }

    paintCurve(svg, values, "#c8e04a");
  }

  function makeMacroBar(data) {
    var section = el("section", "macrobar");
    section.appendChild(el("span", "macro-label", "MACROS"));

    var controls = el("div", "macro-controls");
    var macros = data.macros || [];

    for (var index = 0; index < 8; index++) {
      (function (slot) {
        var value = typeof macros[slot] === "number" ? macros[slot] : 0;

        controls.appendChild(window.IAmCard.control({
          id: "macro." + slot,
          label: "M" + (slot + 1),
          kind: "float",
          min: 0,
          max: 1,
          default: 0,
          unit: "percent"
        }, value, function (_, next) {
          queue("macro." + slot, next);
        }));
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

  function makeSourceCard(title, accent, paint, build) {
    var card = el("div", "mod-card");
    card.style.setProperty("--accent", accent);
    card.appendChild(el("div", "mod-card-title", title));

    var hero = el("div", "mod-hero");
    var svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
    svg.setAttribute("viewBox", "0 0 300 64");
    svg.setAttribute("role", "img");
    svg.setAttribute("aria-label", title + " modulation shape");
    hero.appendChild(svg);
    card.appendChild(hero);

    var controls = el("div", "mod-card-controls");
    build(controls, function () { paint(svg); });
    paint(svg);
    card.appendChild(controls);
    return card;
  }

  function numberOr(value, fallback) {
    return typeof value === "number" ? value : fallback;
  }

  function makeModulation(data) {
    var sources = data.sources || {};
    var lfo = {
      rate: numberOr((sources.lfo || {}).rate, 1),
      shape: numberOr((sources.lfo || {}).shape, 0)
    };
    var env = {
      attack: numberOr((sources.env || {}).attack, 10),
      release: numberOr((sources.env || {}).release, 100)
    };
    var random = {
      rate: numberOr((sources.random || {}).rate, 4),
      smooth: numberOr((sources.random || {}).smooth, 80)
    };
    var table = {
      rate: numberOr((sources.table || {}).rate, 1),
      shape: numberOr((sources.table || {}).shape, 0),
      morph: numberOr((sources.table || {}).morph, 0)
    };
    var section = el("section", "modulation");
    var head = el("div", "mod-head");

    head.appendChild(el("span", "mod-title", "MODULATION"));
    head.appendChild(el("span", "mod-hint", "8 slots \u00B7 LFO / envelope / random / table"));
    section.appendChild(head);

    var grid = el("div", "mod-sources");

    grid.appendChild(makeSourceCard("LFO", "#4c8dff", function (svg) {
      drawLfo(svg, lfo);
    }, function (card, refresh) {
      sourceControl(card, {
        id: "lfo.rate", label: "Rate", kind: "float",
        min: 0.01, max: 20, default: 1, unit: "hz", log: true
      }, lfo.rate, "source.lfo.rate", function (value) {
        lfo.rate = value;
        refresh();
      });
      sourceControl(card, {
        id: "lfo.shape", label: "Shape", kind: "int",
        min: 0, max: 3, default: 0, widget: "choice",
        choices: ["Sine", "Triangle", "Saw", "Square"]
      }, lfo.shape, "source.lfo.shape", function (value) {
        lfo.shape = value;
        refresh();
      });
    }));

    grid.appendChild(makeSourceCard("Envelope", "#35c77a", function (svg) {
      drawEnvelope(svg, env);
    }, function (card, refresh) {
      sourceControl(card, {
        id: "env.attack", label: "Attack", kind: "float",
        min: 1, max: 500, default: 10, unit: "ms", log: true
      }, env.attack, "source.env.attack", function (value) {
        env.attack = value;
        refresh();
      });
      sourceControl(card, {
        id: "env.release", label: "Release", kind: "float",
        min: 1, max: 500, default: 100, unit: "ms", log: true
      }, env.release, "source.env.release", function (value) {
        env.release = value;
        refresh();
      });
    }));

    grid.appendChild(makeSourceCard("Random", "#ff9a3d", function (svg) {
      drawRandom(svg, random);
    }, function (card, refresh) {
      sourceControl(card, {
        id: "random.rate", label: "Rate", kind: "float",
        min: 0.01, max: 20, default: 4, unit: "hz", log: true
      }, random.rate, "source.random.rate", function (value) {
        random.rate = value;
        refresh();
      });
      sourceControl(card, {
        id: "random.smooth", label: "Smooth", kind: "float",
        min: 0, max: 500, default: 80, unit: "ms", log: true
      }, random.smooth, "source.random.smooth", function (value) {
        random.smooth = value;
        refresh();
      });
    }));

    grid.appendChild(makeSourceCard("Table", "#c8e04a", function (svg) {
      drawTable(svg, table);
    }, function (card, refresh) {
      sourceControl(card, {
        id: "table.rate", label: "Rate", kind: "float",
        min: 0.01, max: 20, default: 1, unit: "hz", log: true
      }, table.rate, "source.table.rate", function (value) {
        table.rate = value;
      });
      sourceControl(card, {
        id: "table.shape", label: "Shape", kind: "int",
        min: 0, max: 3, default: 0, widget: "choice",
        choices: ["Sine", "Triangle", "Saw", "Square"]
      }, table.shape, "source.table.shape", function (value) {
        table.shape = value;
        refresh();
      });
      sourceControl(card, {
        id: "table.morph", label: "Morph", kind: "float",
        min: 0, max: 1, default: 0, unit: "percent"
      }, table.morph, "source.table.morph", function (value) {
        table.morph = value;
        refresh();
      });
    }));

    section.appendChild(grid);

    var list = el("div", "assignments");
    var assignments = data.assignments || [];

    for (var slot = 0; slot < 8; slot++) {
      list.appendChild(makeAssignment(data, slot, assignments[slot] || null));
    }

    section.appendChild(list);
    return section;
  }

  function makeAssignment(data, slot, assignment) {
    var row = el("div", "assign-row");
    var sourceNames = ["LFO", "Envelope", "Random", "Table"];
    var sourceValues = ["lfo", "env", "random", "table"];

    for (var macro = 1; macro <= 8; macro++) {
      sourceNames.push("Macro " + macro);
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

    if (assignment) source.value = sourceValues[assignment.source] || "";

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

  function render(data) {
    current = data;
    rangeLabels = [];
    rack.textContent = "";

    rack.appendChild(makeMacroBar(data));
    rack.appendChild(makeCrossover(data));
    data.bands.forEach(function (band, index) {
      rack.appendChild(makeBand(data, index));
    });
    rack.appendChild(makeModulation(data));

    updateRanges(data);

    var mods = (data.assignments || []).filter(function (assignment) {
      return assignment;
    }).length;

    rate.textContent = (data.rate / 1000).toFixed(1) + " kHz";
    status.textContent = data.modules.length + " modules \u00B7 " + data.bands.length +
      " bands \u00B7 " + mods + " mods";
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

  power.addEventListener("click", function () {
    bypassed = !bypassed;
    power.classList.toggle("off", bypassed);
    power.setAttribute("aria-pressed", bypassed ? "true" : "false");
    if (window.ipc) window.ipc.postMessage("__bypass=" + (bypassed ? 1 : 0));
  });

  load();
})();
