// Part ten: the card, and the exports the page and the rack share.
  function create(options) {
    var card = document.createElement("section");
    card.className = "mcard";
    card.style.setProperty("--accent", options.accent || "#8a6aff");
    if (options.collapsed) card.classList.add("collapsed");

    var values = {};
    (options.controls || []).forEach(function (spec) {
      values[spec.id] = valueOf(options.values, spec.id, spec.default);
    });

    // The unmodulated values, kept apart: a live update must not become the next
    // tick's base, or the needle would drift back onto the hand.
    var baseValues = {};
    Object.keys(values).forEach(function (id) { baseValues[id] = values[id]; });

    var byId = {};
    (options.controls || []).forEach(function (spec) { byId[spec.id] = spec; });

    var head = document.createElement("header");
    head.className = "mcard-head";

    if (options.dragData !== undefined) {
      var handle = document.createElement("span");
      handle.className = "mcard-drag";
      handle.textContent = "\u28FF";
      handle.draggable = true;
      handle.title = "Drag module";
      handle.setAttribute("aria-label", "Drag module");
      handle.addEventListener("dragstart", function (event) {
        event.dataTransfer.setData("text/plain", String(options.dragData));
        event.dataTransfer.effectAllowed = "move";
        card.classList.add("dragging");
      });
      handle.addEventListener("dragend", function () {
        card.classList.remove("dragging");
      });
      head.appendChild(handle);
    }

    var collapse = document.createElement("button");
    collapse.type = "button";
    collapse.className = "mcard-collapse";
    collapse.textContent = options.collapsed ? "\u25B8" : "\u25BE";
    collapse.title = options.collapsed ? "Expand" : "Collapse";
    collapse.setAttribute("aria-label", collapse.title);
    collapse.addEventListener("click", function () {
      var collapsed = card.classList.toggle("collapsed");
      collapse.textContent = collapsed ? "\u25B8" : "\u25BE";
      collapse.title = collapsed ? "Expand" : "Collapse";
      collapse.setAttribute("aria-label", collapse.title);
      if (options.onCollapse) options.onCollapse(collapsed);
    });
    head.appendChild(collapse);

    // A collapsed card is a strip: clicking it should open it again, not only
    // clicking the arrow does.
    card.addEventListener("click", function (event) {
      if (!card.classList.contains("collapsed")) return;
      if (event.target.closest("button, select, input, textarea, a")) return;

      collapse.click();
    });

    // A scripted product carries its source in a small editor under the head.
    var scriptArea = null;
    var scriptError = null;

    if (options.script) {
      var scriptToggle = document.createElement("button");
      scriptToggle.type = "button";
      scriptToggle.className = "mcard-script-toggle";
      scriptToggle.textContent = "{ }";
      scriptToggle.title = "Edit the script";
      scriptToggle.setAttribute("aria-label", "Edit the script");
      head.appendChild(scriptToggle);

      var panel = document.createElement("div");
      panel.className = "mcard-script";
      panel.style.display = "none";

      scriptArea = document.createElement("textarea");
      scriptArea.className = "mcard-script-text";
      scriptArea.spellcheck = false;
      panel.appendChild(scriptArea);

      scriptError = document.createElement("div");
      scriptError.className = "mcard-script-error";
      scriptError.style.display = "none";
      panel.appendChild(scriptError);

      var actions = document.createElement("div");
      actions.className = "mcard-script-actions";

      var apply = document.createElement("button");
      apply.type = "button";
      apply.className = "mcard-script-apply";
      apply.textContent = "Apply";
      apply.addEventListener("click", function () {
        if (options.applyScript) options.applyScript(scriptArea.value);
      });
      actions.appendChild(apply);
      panel.appendChild(actions);
      card.appendChild(panel);

      scriptToggle.addEventListener("click", function () {
        var open = panel.style.display === "none";
        panel.style.display = open ? "" : "none";
        if (open && options.loadScript) options.loadScript();
      });
    }

    // A card for a module that follows the host's notes shows the note it is
    // hearing; the page pushes it in from /module-note.
    var noteBadge = null;

    if (options.note) {
      noteBadge = document.createElement("span");
      noteBadge.className = "mcard-note-badge";
      noteBadge.style.display = "none";
      head.appendChild(noteBadge);
    }

    var bypassed = !!options.bypass;

    var power = document.createElement("button");
    power.type = "button";
    power.className = "mcard-power";
    power.title = "Bypass";
    power.setAttribute("aria-label", "Bypass");
    power.setAttribute("aria-pressed", bypassed ? "true" : "false");
    if (bypassed) power.classList.add("off");
    var dot = document.createElement("span");
    dot.className = "dot";
    power.appendChild(dot);
    power.addEventListener("click", function () {
      bypassed = !bypassed;
      power.classList.toggle("off", bypassed);
      power.setAttribute("aria-pressed", bypassed ? "true" : "false");
      card.classList.toggle("bypassed", bypassed);
      if (options.onBypass) options.onBypass(bypassed);
    });
    head.appendChild(power);

    var name = document.createElement("span");
    name.className = "mcard-name";
    name.textContent = options.title;
    head.appendChild(name);

    var badge = document.createElement("span");
    badge.className = "mcard-badge";
    badge.textContent = options.badge || "";
    head.appendChild(badge);

    var spacer = document.createElement("span");
    spacer.className = "mcard-spacer";
    head.appendChild(spacer);

    if (options.kinds && options.kinds.length) {
      var kindSelect = document.createElement("select");
      kindSelect.className = "mcard-select mcard-kind";
      kindSelect.title = "Module type";
      kindSelect.setAttribute("aria-label", "Module type");

      var groups = Object.create(null);

      options.kinds.forEach(function (entry) {
        var name = typeof entry === "string" ? entry : entry.name;
        var group = typeof entry === "string" ? "" : entry.group || "";
        var option = document.createElement("option");
        option.value = name;
        option.textContent = name;

        if (!group) {
          kindSelect.appendChild(option);
          return;
        }

        if (!groups[group]) {
          groups[group] = document.createElement("optgroup");
          groups[group].label = group;
          kindSelect.appendChild(groups[group]);
        }

        groups[group].appendChild(option);
      });

      kindSelect.value = options.kind || options.kinds[0];
      kindSelect.addEventListener("change", function () {
        if (options.onKind) options.onKind(kindSelect.value);
      });
      head.appendChild(kindSelect);
    }

    if (options.onDuplicate) {
      card.classList.add("has-duplicate");
      var duplicate = document.createElement("button");
      duplicate.type = "button";
      duplicate.className = "mcard-duplicate";
      duplicate.textContent = "\u29C9";
      duplicate.title = "Duplicate module";
      duplicate.setAttribute("aria-label", "Duplicate module");
      duplicate.addEventListener("click", function () { options.onDuplicate(); });
      head.appendChild(duplicate);
    }

    if (options.onRemove) {
      card.classList.add("has-remove");
      var remove = document.createElement("button");
      remove.type = "button";
      remove.className = "mcard-remove";
      remove.textContent = "\u00D7";
      remove.title = "Remove module";
      remove.setAttribute("aria-label", "Remove module");
      remove.addEventListener("click", function () { options.onRemove(); });
      head.appendChild(remove);
    }

    var heroElement = null;
    function paintHero() {
      if (heroElement) heroElement.innerHTML = hero(options.hero, values, options.context);
    }

    function edited(id, value) {
      values[id] = value;
      baseValues[id] = value;
      paintHero();
      // A switch appears in the header, so the body has to follow it.
      applyVisibility();
      if (options.onEdit) options.onEdit(id, value);
    }

    var modeSpec = options.mode ? byId[options.mode] : null;
    var currentMode = modeSpec ? Math.round(values[modeSpec.id]) : 0;

    if (modeSpec) {
      var modeSelect = makeChoice(modeSpec, values[modeSpec.id], function (_, value) {
        currentMode = Math.round(value);
        applyVisibility();
        edited(modeSpec.id, currentMode);
      }, "mcard-select mcard-mode");
      head.appendChild(setting("Mode", modeSelect));
    }

    (options.header || []).forEach(function (id) {
      var spec = byId[id];
      if (!spec) return;
      var control;

      if (spec.kind === "int" && spec.choices && spec.choices.length) {
        control = makeChoice(spec, values[id], function (_, value) {
          edited(id, value);
        });
        head.appendChild(setting(spec.label, control));
      } else if (spec.kind === "bool" && spec.widget === "pill") {
        control = makePill(spec, values[id], function (_, value) {
          edited(id, value);
        });
        head.appendChild(control);
      } else if (spec.kind === "bool") {
        control = makeHeaderDot(spec, values[id], function (_, value) {
          edited(id, value);
        });
        head.appendChild(settingAfter(control, spec.label));
      } else {
        control = makeToggle(spec, values[id], function (_, value) {
          edited(id, value);
        });
        head.appendChild(setting(spec.label, control));
      }
    });

    card.appendChild(head);

    if (options.hero) {
      heroElement = document.createElement("div");
      heroElement.className = "mcard-hero";
      card.appendChild(heroElement);
      paintHero();

      // A product whose picture is also its file picker (the convolver) makes
      // the hero clickable; the page that owns the card does the asking.
      if (options.ir) {
        heroElement.classList.add("ir-target");
        heroElement.title = "Click to load an impulse response";
        heroElement.addEventListener("click", function () {
          if (options.pickIr) options.pickIr();
        });
      }
    }

    var body = document.createElement("div");
    body.className = "mcard-body";

    var bodyControls = [];
    var liveKnobs = Object.create(null);

    (options.controls || []).forEach(function (spec) {
      if (modeSpec && spec.id === modeSpec.id) return;
      if ((options.header || []).indexOf(spec.id) >= 0) return;
      var el = makeBodyControl(spec, values[spec.id], edited);

      if (el && el.__live) liveKnobs[spec.id] = el.__live;

      body.appendChild(el);
      bodyControls.push({ spec: spec, el: el });
    });

    card.appendChild(body);

    function applyVisibility() {
      if (!bodyControls) return;

      // A switch trades one branch of a pair for the other: the delay's free
      // time gives way to its beats, the chorus's hertz to its beats.
      var hidden = Object.create(null);
      (options.switches || []).forEach(function (row) {
        var on = Number(values[row.control] || 0) !== 0;
        (on ? row.off : row.on).forEach(function (id) {
          hidden[id] = true;
        });
      });

      var allowed = null;
      if (options.visible) {
        for (var i = 0; i < options.visible.length; i++) {
          if (Number(options.visible[i][0]) === currentMode) {
            allowed = options.visible[i][1];
          }
        }
      }

      bodyControls.forEach(function (entry) {
        var show = (!allowed || allowed.indexOf(entry.spec.id) >= 0) && !hidden[entry.spec.id];
        entry.el.style.display = show ? "" : "none";
      });
    }

    applyVisibility();

    // A live hero has to be redrawn when the audio moves, not only when a knob
    // does: the page keeps the card and calls this on every scope frame.
    card.repaintHero = paintHero;

    card.setLiveValues = function (live) {
      var changed = false;

      Object.keys(live).forEach(function (id) {
        var value = live[id];
        var spec = byId[id];

        if (typeof value !== "number") return;

        if (liveKnobs[id]) {
          var offset = spec && typeof baseValues[id] === "number"
            ? norm(spec, value) - norm(spec, baseValues[id])
            : null;

          liveKnobs[id](offset === null ? null : clamp(offset, -1, 1));
        }

        if (values[id] === value) return;

        values[id] = value;
        changed = true;
      });

      // The hero reads the same map the knobs' base does, so it follows the
      // source without the page having to know what the picture means.
      if (changed) paintHero();
    };

    card.setScript = function (source, force) {
      if (!scriptArea || source === undefined || source === null) return;
      // A poll must not overwrite what the person is typing.
      if (!force && document.activeElement === scriptArea) return;

      scriptArea.value = source;
    };

    card.setScriptError = function (message) {
      if (!scriptError) return;

      if (!message) {
        scriptError.style.display = "none";
        scriptError.textContent = "";
        return;
      }

      scriptError.textContent = message;
      scriptError.style.display = "";
    };

    card.setNote = function (notes) {
      if (!noteBadge) return;

      var list = (notes || []).filter(function (note) {
        return typeof note === "number" && note >= 0;
      });

      if (!list.length) {
        noteBadge.style.display = "none";
        return;
      }

      var names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"];
      noteBadge.textContent = list
        .slice()
        .sort(function (a, b) { return a - b; })
        .map(function (note) {
          return names[note % 12] + (Math.floor(note / 12) - 1);
        })
        .join(" ");
      noteBadge.style.display = "";
    };

    return card;
  }

  window.IAmCard = {
    create: create,
    control: control,
    controlHandle: controlHandle,
    switchHandle: makeSwitch,
    dotHandle: makeHeaderDot,
    notePickerHandle: makeNotePickerHandle,
    choiceHandle: makeChoiceHandle,
    sliderHandle: makeSliderHandle,
    hero: hero,
    wire: function (el, kind, values, state) {
      if (kind === "canvas") wireCanvas(el, kind, values, state);
      else if (kind === "spectrum") wireSpectrum(el, kind, values, state);
    },
    modulate: modulate,
    registerHero: registerHero,
    bandsOf: bandsOf,
    smoothBands: smoothBands,
    bandsPath: bandsPath,
    format: format,
    norm: norm,
    at: at
  };
})();
