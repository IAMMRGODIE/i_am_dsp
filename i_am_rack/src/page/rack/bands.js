// Part three: the crossover, the bands, the slots and their groups.
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

    module.controls = (module.controls || []).map(function (control) {
      // The server annotates every control with the assignments that drive it;
      // the page narrows that to the selected source, and gives each one the
      // label and colour its arc is drawn in.
      var assignments = (control.modulations || [])
        .filter(function (assignment) {
          return selectedSource === null || assignment.source === selectedSource;
        })
        .map(function (assignment) {
          assignment.label = sourceLabel(assignment.source);
          assignment.color = sourceColor(assignment.source);
          assignment.selected = selectedSource !== null;
          return assignment;
        });

      var floating = control.kind === "float";

      return window.IAmCard.modulate(control, {
        assignments: assignments,
        onDepth: setModulationDepth,
        onMode: floating ? toggleModulationMode : null,
        dropTarget: floating ? "m" + module.id + "." + control.id : null,
        onDrop: floating ? linkModulation : null
      });
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
      history: module.history,
      filterTable: module.filter_table,
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
      history: !!module.history,
      filterTable: !!module.filter_table,
      filterTableState: function () { return entry.context.filterTable; },
      filterTableSet: function (id, value) { queue(module.id + "." + id, value); },
      filterTablePost: function (body) {
        fetch("/module-filter-table?id=" + module.id, { method: "POST", body: body }).catch(function () {});
      },
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
