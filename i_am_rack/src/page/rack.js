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
    svg.setAttribute("viewBox", "0 0 560 66");
    svg.setAttribute("role", "img");
    svg.setAttribute("aria-label", "Low, mid and high bands with the crossover points");
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

    function paint() {
      svg.textContent = "";

      var lowX = xFor(current.split_low);
      var highX = xFor(current.split_high);
      var bandTop = 10;
      var bandHeight = 30;

      function band(left, right, color, label) {
        if (right - left < 1) return;

        svg.appendChild(node("rect", {
          x: left.toFixed(1),
          y: bandTop,
          width: (right - left).toFixed(1),
          height: bandHeight,
          fill: color,
          "fill-opacity": ".82"
        }));

        if (right - left < 38) return;

        var text = node("text", {
          x: ((left + right) / 2).toFixed(1),
          y: bandTop + 19,
          "text-anchor": "middle",
          fill: "#111",
          "font-size": "8",
          "font-family": "Segoe UI, system-ui, sans-serif",
          "letter-spacing": "1"
        });
        text.textContent = label;
        svg.appendChild(text);
      }

      band(8, lowX, "#4c8dff", "LOW");
      band(lowX, highX, "#ff9a3d", "MID");
      band(highX, 552, "#35c77a", "HIGH");

      svg.appendChild(node("rect", {
        x: 8,
        y: bandTop,
        width: 544,
        height: bandHeight,
        rx: 3,
        fill: "none",
        stroke: "#444",
        "stroke-width": "1"
      }));

      [lowX, highX].forEach(function (x, index) {
        svg.appendChild(node("line", {
          x1: x.toFixed(1),
          y1: 6,
          x2: x.toFixed(1),
          y2: 44,
          stroke: "#e0e0e0",
          "stroke-opacity": ".55",
          "stroke-width": "1",
          "stroke-dasharray": "2 3"
        }));

        var frequency = index === 0 ? current.split_low : current.split_high;
        var label = node("text", {
          x: Math.max(16, Math.min(544, x)).toFixed(1),
          y: 58,
          "text-anchor": "middle",
          fill: "#9a9a9a",
          "font-size": "9",
          "font-family": "Cascadia Mono, Consolas, monospace"
        });
        label.textContent = formatHz(frequency);
        svg.appendChild(label);
      });
    }

    paintCrossover = paint;
    paint();

    section.appendChild(el("span", "crossover-note", "LR4 \u00B7 3 bands"));
    return section;
  }

  function makeBandHeader(data, bandIndex) {
    var band = data.bands[bandIndex];
    var head = el("div", "band-head");
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
      unit: ""
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
      bipolar: true
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

    section.appendChild(makeBandHeader(data, bandIndex));

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

  function render(data) {
    current = data;
    rangeLabels = [];
    rack.textContent = "";

    rack.appendChild(makeCrossover(data));
    data.bands.forEach(function (band, index) {
      rack.appendChild(makeBand(data, index));
    });

    updateRanges(data);

    rate.textContent = (data.rate / 1000).toFixed(1) + " kHz";
    status.textContent = data.modules.length + " modules \u00B7 " + data.bands.length + " bands";
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
