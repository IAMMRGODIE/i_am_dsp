(function () {
  "use strict";

  var rack = document.getElementById("rack");
  var status = document.getElementById("status");
  var rate = document.getElementById("rate");
  var power = document.getElementById("power");

  var pending = Object.create(null);
  var timer = null;
  var bypassed = false;

  function el(tag, className) {
    var node = document.createElement(tag);
    if (className) node.className = className;
    return node;
  }

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
    return spec.kind === "int" ? String(Math.round(value)) : value.toFixed(2);
  }

  function queue(address, value) {
    pending[address] = address + "=" + value;
    if (timer) clearTimeout(timer);
    timer = setTimeout(flush, 60);
  }

  function flush() {
    if (timer) clearTimeout(timer);
    timer = null;

    var addresses = Object.keys(pending);
    if (!addresses.length) return;

    var body = addresses.map(function (address) {
      return pending[address];
    }).join("\n");

    pending = Object.create(null);

    fetch("/rack-command", { method: "POST", body: body })
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
    flush();
  }

  function control(spec, moduleId, value) {
    var row = el("div", "control");
    var address = moduleId + "." + spec.id;
    var label = el("label", "control-label");
    label.textContent = spec.label;
    label.htmlFor = "p" + moduleId + "_" + spec.id;
    row.appendChild(label);

    var input;

    if (spec.kind === "bool") {
      input = document.createElement("input");
      input.type = "checkbox";
      input.id = label.htmlFor;
      input.checked = !!value;
      var state = el("span", "control-value");
      state.textContent = value ? "On" : "Off";
      input.addEventListener("change", function () {
        state.textContent = input.checked ? "On" : "Off";
        sendNow(address, input.checked ? 1 : 0);
      });
      row.appendChild(input);
      row.appendChild(state);
      return row;
    }

    if (spec.widget === "choice" && spec.choices && spec.choices.length) {
      input = el("select", "");
      input.id = label.htmlFor;
      spec.choices.forEach(function (name, index) {
        var option = document.createElement("option");
        option.value = String(index);
        option.textContent = name;
        input.appendChild(option);
      });
      input.value = String(Math.round(value));
      input.addEventListener("change", function () {
        sendNow(address, input.value);
      });
      row.appendChild(input);
      row.appendChild(el("span", "control-value"));
      return row;
    }

    var steps = 1000;
    input = document.createElement("input");
    input.type = "range";
    input.id = label.htmlFor;
    input.min = "0";
    input.max = String(steps);
    input.step = "1";
    input.value = String(Math.round(norm(spec, value) * steps));

    var readout = el("span", "control-value");
    readout.textContent = format(spec, value);

    input.addEventListener("input", function () {
      var next = at(spec, Number(input.value) / steps);
      if (spec.kind === "int") next = Math.round(next);
      readout.textContent = format(spec, next);
      queue(address, next);
    });

    input.addEventListener("dblclick", function () {
      var next = spec.default;
      input.value = String(Math.round(norm(spec, next) * steps));
      readout.textContent = format(spec, next);
      sendNow(address, next);
    });

    row.appendChild(input);
    row.appendChild(readout);
    return row;
  }

  function card(module) {
    var box = el("section", "card");
    var head = el("header", "card-head");

    head.appendChild(el("span", "card-dot"));

    var name = el("span", "card-name");
    name.textContent = module.kind;
    head.appendChild(name);

    var id = el("span", "card-id");
    id.textContent = "M" + (module.id + 1);
    head.appendChild(id);

    var toggle = el("button", "card-bypass");
    toggle.type = "button";
    toggle.textContent = module.bypass ? "Bypassed" : "Active";
    if (module.bypass) toggle.classList.add("off");
    toggle.addEventListener("click", function () {
      module.bypass = !module.bypass;
      toggle.textContent = module.bypass ? "Bypassed" : "Active";
      toggle.classList.toggle("off", module.bypass);
      sendNow(module.id + ".__bypass", module.bypass ? 1 : 0);
    });
    head.appendChild(toggle);

    box.appendChild(head);

    var body = el("div", "card-body");
    module.controls.forEach(function (spec) {
      var value = typeof module.values[spec.id] === "number"
        ? module.values[spec.id]
        : spec.default;
      body.appendChild(control(spec, module.id, value));
    });
    box.appendChild(body);

    return box;
  }

  function render(data) {
    rack.textContent = "";
    data.modules.forEach(function (module) {
      rack.appendChild(card(module));
    });
    rate.textContent = (data.rate / 1000).toFixed(1) + " kHz";
    status.textContent = data.modules.length + " modules";
  }

  power.addEventListener("click", function () {
    bypassed = !bypassed;
    power.classList.toggle("off", bypassed);
    power.setAttribute("aria-pressed", bypassed ? "true" : "false");
    if (window.ipc) window.ipc.postMessage("__bypass=" + (bypassed ? 1 : 0));
  });

  fetch("/rack", { cache: "no-store" })
    .then(function (response) {
      if (!response.ok) throw new Error("rack " + response.status);
      return response.json();
    })
    .then(render)
    .catch(function (error) {
      status.textContent = "rack unavailable: " + error.message;
    });
})();
