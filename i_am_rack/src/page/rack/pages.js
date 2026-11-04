// Part eight: the sources, the matrix, the automation and the assignments.
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
