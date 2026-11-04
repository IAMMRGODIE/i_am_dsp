// Part six: the macros and the source cards.
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
