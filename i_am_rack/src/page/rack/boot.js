// Part twelve: the wiring the page runs once its elements exist.
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
