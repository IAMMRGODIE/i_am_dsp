// Part eleven: the bottom panel and its height.
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
