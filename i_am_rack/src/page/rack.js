(function () {
  "use strict";

  var rack = document.getElementById("rack");
  var status = document.getElementById("status");
  var rate = document.getElementById("rate");
  var power = document.getElementById("power");

  var pending = Object.create(null);
  var timer = null;
  var bypassed = false;

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

  function render(data) {
    rack.textContent = "";

    data.modules.forEach(function (module) {
      rack.appendChild(window.IAmCard.create({
        title: module.kind,
        badge: "M" + (module.id + 1),
        accent: module.accent,
        hero: module.hero,
        mode: module.mode,
        header: module.header,
        visible: module.visible,
        bypass: module.bypass,
        controls: module.controls,
        values: module.values,
        onEdit: function (identifier, value) {
          queue(module.id + "." + identifier, value);
        },
        onBypass: function (next) {
          sendNow(module.id + ".__bypass", next ? 1 : 0);
        }
      }));
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
