// Part nine: the controls themselves.
  function makeKnob(spec, value, onEdit) {
    var wrap = document.createElement("div");
    wrap.className = "knob-wrap";

    var dial = document.createElement("div");
    dial.className = "knob";
    dial.tabIndex = 0;
    dial.setAttribute("role", "slider");
    dial.setAttribute("aria-label", spec.label);

    if (spec.dropTarget) dial.setAttribute("data-drop-target", spec.dropTarget);

    dial.setAttribute("aria-valuemin", String(spec.min));
    dial.setAttribute("aria-valuemax", String(spec.max));

    var drawing = svg("svg", { viewBox: "0 0 64 64" });
    drawing.appendChild(svg("circle", { class: "knob-bg", cx: 32, cy: 32, r: 25 }));
    drawing.appendChild(svg("path", {
      class: "knob-track",
      d: arcPath(32, 32, 25, START, START + SWEEP)
    }));

    var modulations = spec.modulations || [];

    var mods = svg("g", { class: "knob-mods" });
    var fill = svg("path", { class: "knob-arc" });
    var hand = svg("line", { class: "knob-hand" });
    var live = svg("line", { class: "knob-live" });
    live.style.display = "none";
    drawing.appendChild(mods);
    drawing.appendChild(fill);
    drawing.appendChild(hand);
    drawing.appendChild(live);
    dial.appendChild(drawing);

    if (modulations.some(function (assignment) { return assignment.selected; })) {
      dial.classList.add("mod-selected");
    }

    // A thin needle for where the modulation is right now, so the knob shows
    // the sound while its main hand keeps the value a person set.
    /// The needle is an offset from the hand, so turning the knob carries it
    /// along and the modulation keeps its distance.
    var liveOffset = null;

    wrap.__live = function (offset) {
      liveOffset = offset;
      paintLive();
    };

    function paintLive() {
      if (liveOffset === null) {
        live.style.display = "none";
        return;
      }

      var position = clamp(norm(spec, current) + liveOffset, 0, 1);
      var angle = START + SWEEP * position;
      var inner = polar(32, 32, 18, angle);
      var tip = polar(32, 32, 29, angle);
      live.setAttribute("x1", inner.x.toFixed(2));
      live.setAttribute("y1", inner.y.toFixed(2));
      live.setAttribute("x2", tip.x.toFixed(2));
      live.setAttribute("y2", tip.y.toFixed(2));
      live.style.display = "";
    }

    // The depth a source is adding while it is dragged with Alt.
    var modHint = document.createElement("span");
    modHint.className = "knob-mod-hint";
    modHint.style.display = "none";
    wrap.appendChild(modHint);

    var label = document.createElement("div");
    label.className = "knob-label";
    label.textContent = spec.label;

    var readout = document.createElement("div");
    readout.className = "knob-value";

    var current = clamp(value, spec.min, spec.max);

    function paint() {
      var position = clamp(norm(spec, current), 0, 1);
      var angle = START + SWEEP * position;
      fill.setAttribute("d", arcPath(32, 32, 25, START, angle));

      // One arc per source, in that source's colour: a knob driven by two of
      // them shows two arcs instead of one merged span nobody can attribute.
      while (mods.firstChild) {
        mods.removeChild(mods.firstChild);
      }

      modulations.forEach(function (assignment) {
        var depth = assignment.depth || 0;
        var mode = assignment.mode || "centered";
        var low = position;
        var high = position;

        if (mode === "frombase" || mode === "from-base" || mode === "start") {
          var end = clamp(position + depth, 0, 1);
          low = Math.min(low, end);
          high = Math.max(high, end);
        } else {
          var span = Math.abs(depth);
          low = Math.min(low, clamp(position - span, 0, 1));
          high = Math.max(high, clamp(position + span, 0, 1));
        }

        if (high - low <= 0.0001) return;

        var arc = svg("path", {
          class: "knob-mod",
          d: arcPath(32, 32, 29, START + SWEEP * low, START + SWEEP * high)
        });

        if (assignment.color) arc.style.stroke = assignment.color;
        mods.appendChild(arc);
      });

      if (modulations.length) {
        dial.title = modulations
          .map(function (assignment) {
            var depth = assignment.depth || 0;
            return (assignment.label || "source") + "  " +
              (depth >= 0 ? "+" : "") + Math.round(depth * 100) + "%";
          })
          .join("\n");
      }
      var inner = polar(32, 32, 9, angle);
      var tip = polar(32, 32, 25, angle);
      hand.setAttribute("x1", inner.x.toFixed(2));
      hand.setAttribute("y1", inner.y.toFixed(2));
      hand.setAttribute("x2", tip.x.toFixed(2));
      hand.setAttribute("y2", tip.y.toFixed(2));
      readout.textContent = display(spec, current);
      dial.setAttribute("aria-valuenow", String(current));
      dial.setAttribute("aria-valuetext", display(spec, current));
      paintLive();
    }

    function set(value, send) {
      current = clamp(value, spec.min, spec.max);
      if (spec.kind === "int") current = Math.round(current);
      // A stepped parameter only takes its own intervals: a kernel length that
      // has to be odd, a semitone, an eighth note.
      if (spec.step > 0) {
        current = clamp(
          spec.min + Math.round((current - spec.min) / spec.step) * spec.step,
          spec.min,
          spec.max
        );
        if (spec.kind === "int") current = Math.round(current);
      }
      paint();
      if (send && onEdit) onEdit(spec.id, current);
    }

    var drag = null;
    var modDrag = null;

    dial.addEventListener("pointerdown", function (event) {
      if (event.button !== 0) return;

      if (event.altKey && modulations.length) {
        var target = modulations[0];
        modDrag = {
          y: event.clientY,
          depth: target.depth || 0,
          name: target.label || "source"
        };
        dial.classList.add("mod-drag");
        modHint.textContent = modDrag.name + "  " + Math.round(modDrag.depth * 100) + "%";
        modHint.style.display = "";
      } else {
        drag = { y: event.clientY, position: norm(spec, current) };
      }

      dial.setPointerCapture(event.pointerId);
      event.preventDefault();
    });

    dial.addEventListener("pointermove", function (event) {
      if (modDrag) {
        var next = clamp(modDrag.depth + (modDrag.y - event.clientY) / 150, -1, 1);
        modulations[0].depth = next;
        modHint.textContent = modDrag.name + "  " +
          (next >= 0 ? "+" : "") + Math.round(next * 100) + "%";
        modHint.style.display = "";

        if (spec.onModulation && modulations[0].slot !== undefined) {
          spec.onModulation(modulations[0].slot, next);
        }

        paint();
        return;
      }

      if (!drag) return;
      // Shift is the fine drag, the same gesture the faceplate used to have.
      var distance = (drag.y - event.clientY) / 150 * (event.shiftKey ? 0.1 : 1);
      var position = clamp(drag.position + distance, 0, 1);
      set(at(spec, position), true);
    });

    function stop(event) {
      if (!drag && !modDrag) return;
      drag = null;
      modDrag = null;
      modHint.style.display = "none";
      dial.classList.remove("mod-drag");
      try {
        dial.releasePointerCapture(event.pointerId);
      } catch (error) {
        // Released outside the window.
      }
    }

    dial.addEventListener("pointerup", stop);
    dial.addEventListener("pointercancel", stop);
    dial.addEventListener("dblclick", function () { set(spec.default, true); });
    dial.addEventListener("click", function (event) {
      if (!event.ctrlKey || !modulations.length) return;
      var next = (modulations[0].mode || "frombase") === "frombase" ? "center" : "start";
      if (spec.onModeToggle) {
        spec.onModeToggle(modulations.map(function (modulation) { return modulation.slot; }), next);
      }
      event.preventDefault();
    });
    // The parameter's own sentence leads: what the range cannot say is what
    // a person needs before the gesture, not after it.
    var gesture = "Drag to change; right-click to type a value";
    dial.title = spec.hint ? spec.hint + "\n\n" + gesture : gesture;
    dial.addEventListener("contextmenu", function (event) {
      // The modifiers belong to the modulation gestures below.
      if (event.altKey || event.ctrlKey || event.metaKey) return;

      event.preventDefault();
      // The page underneath has a menu of its own: a right click on a knob is
      // not a right click on the canvas.
      event.stopPropagation();
      openKnobEditor(spec, current, set, dial);
    });

    if (spec.dropTarget) {
      dial.addEventListener("dragover", function (event) {
        event.preventDefault();
        if (event.dataTransfer) event.dataTransfer.dropEffect = "link";
        dial.classList.add("drop-target");
      });
      dial.addEventListener("dragleave", function () {
        dial.classList.remove("drop-target");
      });
      dial.addEventListener("drop", function (event) {
        event.preventDefault();
        // The knob stops the drop: without this the slot underneath would read
        // the source as a module and try to move it.
        event.stopPropagation();
        dial.classList.remove("drop-target");
        var data = event.dataTransfer ? event.dataTransfer.getData("text/plain") : "";
        if (data && spec.onModulationDrop) spec.onModulationDrop(data, spec.dropTarget);
      });
    }

    // Alt + right click edits how far each source pushes this knob; ctrl +
    // right click takes every source off it.
    dial.addEventListener("contextmenu", function (event) {
      if (!modulations.length) return;

      if (event.ctrlKey || event.metaKey) {
        event.preventDefault();
        event.stopPropagation();

        if (spec.onClearModulations) {
          spec.onClearModulations(modulations.map(function (modulation) {
            return modulation.slot;
          }));
        }

        return;
      }

      if (!event.altKey) return;

      event.preventDefault();
      event.stopPropagation();
      openDepths();
    });

    function openDepths() {
      var box = document.createElement("div");
      box.className = "knob-depths";

      modulations.forEach(function (assignment) {
        var row = document.createElement("label");
        row.className = "knob-depth";

        var name = document.createElement("span");
        name.textContent = assignment.label || "source";

        var range = document.createElement("input");
        range.type = "range";
        range.min = "-100";
        range.max = "100";
        range.step = "1";
        range.value = String(Math.round((assignment.depth || 0) * 100));

        // The number is the same value as the slider: a depth of 0.5 is hard to
        // hit by dragging, and a patch wants the number it was told.
        var value = document.createElement("input");
        value.type = "number";
        value.min = "-100";
        value.max = "100";
        value.step = "1";
        value.className = "knob-depth-value";
        value.value = range.value;

        var apply = function (depth, source) {
          depth = Math.max(-1, Math.min(1, depth));
          assignment.depth = depth;
          range.value = String(Math.round(depth * 100));

          if (source !== value) value.value = range.value;

          paint();

          if (spec.onModulation) spec.onModulation(assignment.slot, depth);
        };

        range.addEventListener("input", function () {
          apply(Number(range.value) / 100, range);
        });
        value.addEventListener("change", function () {
          var percent = Math.max(-100, Math.min(100, Math.round(Number(value.value) || 0)));

          value.value = String(percent);
          apply(percent / 100, value);
        });

        var step = function (delta) {
          var next = Math.max(-100, Math.min(100, Math.round((Number(value.value) || 0) + delta)));

          value.value = String(next);
          apply(next / 100, value);
        };

        // ArrowUp/Down step without waiting for the field to lose focus, so the
        // field and its buttons agree on what a step is.
        value.addEventListener("keydown", function (event) {
          if (event.key !== "ArrowUp" && event.key !== "ArrowDown") return;

          event.preventDefault();
          step(event.key === "ArrowUp" ? 1 : -1);
        });

        var wrap = document.createElement("span");
        wrap.className = "knob-depth-value-wrap";
        wrap.appendChild(value);

        var steps = document.createElement("span");
        steps.className = "knob-depth-steps";

        [["up", 1, "Increase depth"], ["down", -1, "Decrease depth"]].forEach(function (item) {
          var button = document.createElement("button");

          button.type = "button";
          button.className = item[0];
          button.setAttribute("aria-label", item[2]);
          button.addEventListener("click", function () { step(item[1]); });
          steps.appendChild(button);
        });

        wrap.appendChild(steps);

        row.appendChild(name);
        row.appendChild(range);
        row.appendChild(wrap);

        // How the depth is measured is part of the depth: from the knob's own value,
        // or around it. The ring gesture (ctrl + click on the knob) is easy to miss,
        // so the panel that shows the number also shows what the number means.
        if (spec.onModeToggle) {
          var fromBase = ["frombase", "from-base", "start"].indexOf(assignment.mode || "centered") >= 0;
          var modeButton = document.createElement("button");

          modeButton.type = "button";
          modeButton.className = "knob-depth-mode";

          var paintMode = function () {
            modeButton.textContent = fromBase ? "From base" : "Centered";
            modeButton.dataset.mode = fromBase ? "frombase" : "centered";
            modeButton.title = fromBase
              ? "Starts at the knob's value and moves one way; click for centered"
              : "Swings around the knob's value; click for from base";
          };

          modeButton.addEventListener("click", function () {
            fromBase = !fromBase;
            assignment.mode = fromBase ? "frombase" : "centered";
            paintMode();
            paint();
            spec.onModeToggle([assignment.slot], fromBase ? "start" : "center");

            // A page that rebuilds its card on the change (the rack reloads after a
            // matrix edit) leaves this panel hanging over a knob that is gone.
            setTimeout(function () {
              if (!dial.isConnected) box.remove();
            }, 600);
          });

          paintMode();
          row.appendChild(modeButton);
        }

        box.appendChild(row);
      });

      var rect = dial.getBoundingClientRect();
      // The panel has a width of its own (a mode button rides beside the number), so
      // it is measured rather than assumed: 240 was the width before that.
      document.body.appendChild(box);
      var size = box.getBoundingClientRect();
      box.style.left = Math.max(8, Math.min(window.innerWidth - size.width - 8, rect.left)) + "px";
      box.style.top = Math.min(window.innerHeight - size.height - 8, rect.bottom + 8) + "px";

      function dismiss(event) {
        if (box.contains(event.target)) {
          document.addEventListener("pointerdown", dismiss, { once: true, capture: true });
          return;
        }

        box.remove();
      }

      setTimeout(function () {
        document.addEventListener("pointerdown", dismiss, { once: true, capture: true });
      }, 0);
    }

    dial.addEventListener("wheel", function (event) {
      event.preventDefault();
      // A knob on a card that lives on a zooming board: the wheel is the knob's,
      // not the board's.
      event.stopPropagation();
      var direction = event.deltaY < 0 ? 1 : -1;
      set(stepBy(spec, current, direction, event.shiftKey), true);
    }, { passive: false });

    dial.addEventListener("keydown", function (event) {
      var fine = event.shiftKey;
      if (event.key === "ArrowUp" || event.key === "ArrowRight") {
        set(stepBy(spec, current, 1, fine), true);
        event.preventDefault();
      } else if (event.key === "ArrowDown" || event.key === "ArrowLeft") {
        set(stepBy(spec, current, -1, fine), true);
        event.preventDefault();
      } else if (event.key === "Home") {
        set(spec.min, true);
        event.preventDefault();
      } else if (event.key === "End") {
        set(spec.max, true);
        event.preventDefault();
      }
    });

    paint();
    wrap.appendChild(dial);
    wrap.appendChild(label);
    wrap.appendChild(readout);

    return { el: wrap, set: set };
  }

  function makeChoiceHandle(spec, value, onEdit, className) {
    var select = document.createElement("select");
    select.className = className || "mcard-select";
    select.setAttribute("aria-label", spec.label);

    spec.choices.forEach(function (name, index) {
      var option = document.createElement("option");
      option.value = String(index);
      option.textContent = name;
      select.appendChild(option);
    });

    function set(next) {
      select.value = String(Math.round(next));
    }

    set(value);
    select.addEventListener("change", function () {
      if (onEdit) onEdit(spec.id, Number(select.value));
    });

    return { el: select, set: set };
  }

  /// The parameter's own sentence, when it declared one: a tooltip on the
  /// control, which is where a person looks after the range has surprised them.
  function hintOf(el, spec) {
    if (spec.hint) el.title = spec.hint;
    return el;
  }

  function makeChoice(spec, value, onEdit, className) {
    return hintOf(makeChoiceHandle(spec, value, onEdit, className).el, spec);
  }

  function makeSliderHandle(spec, value, onEdit) {
    var wrap = document.createElement("div");
    wrap.className = "mcard-slider";

    var top = document.createElement("div");
    top.className = "mcard-slider-top";
    var label = document.createElement("span");
    label.className = "mcard-row-label";
    label.textContent = spec.label;
    var readout = document.createElement("span");
    readout.className = "mcard-slider-value";
    top.appendChild(label);
    top.appendChild(readout);
    wrap.appendChild(top);

    var track = document.createElement("div");
    track.className = "mcard-slider-track";
    track.tabIndex = 0;
    track.setAttribute("role", "slider");
    track.setAttribute("aria-label", spec.label);
    track.setAttribute("aria-valuemin", String(spec.min));
    track.setAttribute("aria-valuemax", String(spec.max));

    var fill = document.createElement("div");
    fill.className = "mcard-slider-fill";
    var knob = document.createElement("div");
    knob.className = "mcard-slider-handle";
    track.appendChild(fill);
    track.appendChild(knob);
    wrap.appendChild(track);

    var zero = spec.bipolar ? clamp(norm(spec, 0), 0, 1) : 0;
    var current = clamp(value, spec.min, spec.max);

    function paint() {
      var position = clamp(norm(spec, current), 0, 1);
      fill.style.left = (Math.min(zero, position) * 100) + "%";
      fill.style.width = (Math.abs(position - zero) * 100) + "%";
      knob.style.left = (position * 100) + "%";
      readout.textContent = display(spec, current);
      track.setAttribute("aria-valuenow", String(current));
      track.setAttribute("aria-valuetext", display(spec, current));
    }

    function set(next, send) {
      current = clamp(next, spec.min, spec.max);
      if (spec.kind === "int") current = Math.round(current);
      paint();
      if (send && onEdit) onEdit(spec.id, current);
    }

    function apply(event) {
      var box = track.getBoundingClientRect();
      var position = box.width > 0 ? (event.clientX - box.left) / box.width : 0;
      set(at(spec, position), true);
    }

    var drag = false;
    track.addEventListener("pointerdown", function (event) {
      if (event.button !== 0) return;
      drag = true;
      track.setPointerCapture(event.pointerId);
      event.preventDefault();
      apply(event);
    });
    track.addEventListener("pointermove", function (event) {
      if (drag) apply(event);
    });
    function stop(event) {
      if (!drag) return;
      drag = false;
      try { track.releasePointerCapture(event.pointerId); } catch (error) {}
    }
    track.addEventListener("pointerup", stop);
    track.addEventListener("pointercancel", stop);
    track.addEventListener("wheel", function (event) {
      event.preventDefault();
      // As on a knob: the slider owns the wheel while the pointer is on it.
      event.stopPropagation();
      set(stepBy(spec, current, event.deltaY < 0 ? 1 : -1, event.shiftKey), true);
    }, { passive: false });
    track.addEventListener("keydown", function (event) {
      var fine = event.shiftKey;
      if (event.key === "ArrowUp" || event.key === "ArrowRight") {
        set(stepBy(spec, current, 1, fine), true);
        event.preventDefault();
      } else if (event.key === "ArrowDown" || event.key === "ArrowLeft") {
        set(stepBy(spec, current, -1, fine), true);
        event.preventDefault();
      } else if (event.key === "Home") {
        set(spec.min, true);
        event.preventDefault();
      } else if (event.key === "End") {
        set(spec.max, true);
        event.preventDefault();
      }
    });

    paint();
    return { el: wrap, set: set };
  }

  function makeSwitch(spec, value, onEdit) {
    var button = document.createElement("button");
    button.type = "button";
    button.className = "mcard-switch";
    var current = !!value;
    var thumb = document.createElement("i");
    button.appendChild(thumb);
    button.setAttribute("role", "switch");
    button.setAttribute("aria-label", spec.label);
    button.setAttribute("aria-checked", current ? "true" : "false");
    button.classList.toggle("on", current);

    function paint() {
      button.classList.toggle("on", current);
      button.setAttribute("aria-checked", current ? "true" : "false");
    }

    function set(value, send) {
      current = !!value;
      paint();
      if (send && onEdit) onEdit(spec.id, current ? 1 : 0);
    }

    button.addEventListener("click", function () {
      set(!current, true);
    });

    paint();
    return { el: button, set: set };
  }

  function makeToggle(spec, value, onEdit) {
    return hintOf(makeSwitch(spec, value, onEdit).el, spec);
  }

  /// The rack's source cards put a Sync pill in the header: a text button that
  /// lights up when the source follows the host's tempo.
  function makePill(spec, value, onEdit) {
    var button = document.createElement("button");
    button.type = "button";
    button.className = "mcard-pill";
    button.textContent = spec.label;
    button.setAttribute("aria-pressed", "false");

    var on = !!Number(value);

    function paint() {
      button.classList.toggle("on", on);
      button.setAttribute("aria-pressed", on ? "true" : "false");
    }

    button.addEventListener("click", function () {
      on = !on;
      paint();
      if (onEdit) onEdit(spec.id, on ? 1 : 0);
    });

    paint();
    return button;
  }

  /// A boolean in the header is a dot, like the bypass one: the product page
  /// draws it that way and the caption sits to its right.
  function makeHeaderDot(spec, value, onEdit) {
    var button = document.createElement("button");
    button.type = "button";
    button.className = "mcard-dot";
    button.title = spec.label;
    button.setAttribute("aria-label", spec.label);
    var dot = document.createElement("span");
    dot.className = "dot";
    button.appendChild(dot);

    var on = !!Number(value);

    function paint() {
      button.setAttribute("aria-pressed", on ? "true" : "false");
    }

    button.addEventListener("click", function () {
      on = !on;
      paint();
      if (onEdit) onEdit(spec.id, on ? 1 : 0);
    });

    paint();
    return button;
  }

  function makeNotePickerHandle(spec, value, onEdit) {
    var labels = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"];
    var mask = Math.round(Number(value) || 0) & 4095;
    var wrap = document.createElement("div");
    wrap.className = "mcard-notes";
    wrap.setAttribute("role", "group");
    wrap.setAttribute("aria-label", spec.label);
    var buttons = [];

    function paint() {
      buttons.forEach(function (button, index) {
        var on = (mask & (1 << index)) !== 0;
        button.classList.toggle("on", on);
        button.setAttribute("aria-checked", on ? "true" : "false");
      });
    }

    function set(next) {
      mask = Math.round(Number(next) || 0) & 4095;
      paint();
    }

    labels.forEach(function (label, index) {
      var button = document.createElement("button");
      button.type = "button";
      button.className = "mcard-note" + (label.length > 1 ? " sharp" : "");
      button.textContent = label;
      button.setAttribute("role", "checkbox");
      button.setAttribute("aria-label", label);
      button.addEventListener("click", function () {
        mask ^= 1 << index;
        paint();
        if (onEdit) onEdit(spec.id, mask);
      });
      buttons.push(button);
      wrap.appendChild(button);
    });

    paint();
    return { el: wrap, set: set };
  }

  function makeNotePicker(spec, value, onEdit) {
    return makeNotePickerHandle(spec, value, onEdit).el;
  }

  function makeBodyControl(spec, value, onEdit) {
    if (spec.widget === "notes") {
      return makeNotePicker(spec, value, onEdit);
    }

    if (spec.kind === "bool") {
      // The product page draws a toggle as a column: switch, label, value.
      var wrap = document.createElement("div");
      wrap.className = "knob-wrap";

      var label = document.createElement("div");
      label.className = "knob-label";
      label.textContent = spec.label;

      var readout = document.createElement("div");
      readout.className = "knob-value";

      var on = !!Number(value);

      function paintReadout() {
        readout.textContent = on ? "On" : "Off";
      }

      var handle = makeSwitch(spec, value, function (id, next) {
        on = !!Number(next);
        paintReadout();
        if (onEdit) onEdit(id, next);
      });

      paintReadout();
      wrap.appendChild(handle.el);
      wrap.appendChild(label);
      wrap.appendChild(readout);
      return wrap;
    }

    if (spec.widget === "slider") {
      return makeSliderHandle(spec, value, onEdit).el;
    }

    if (spec.widget === "choice" || (spec.kind === "int" && spec.choices && spec.choices.length)) {
      // The product page stacks a choice's label above its select; the card
      // keeps the same column so the rows wrap the same way.
      var choice = document.createElement("div");
      choice.className = "mcard-choice";
      var choiceLabel = document.createElement("div");
      choiceLabel.className = "mcard-choice-label";
      choiceLabel.textContent = spec.label;
      choice.appendChild(choiceLabel);
      choice.appendChild(makeChoice(spec, value, onEdit));
      return choice;
    }

    var knob = makeKnob(spec, value, onEdit);
    knob.el.__set = knob.set;

    return knob.el;
  }

  /// One control with its modulation ring: the assignments it shows and the
  /// gestures that change them. Both pages build their cards from this.
  function modulate(control, options) {
    options = options || {};

    var spec = Object.assign({}, control);
    spec.modulations = options.assignments || [];

    if (options.onDepth) spec.onModulation = options.onDepth;
    if (options.onMode) spec.onModeToggle = options.onMode;
    if (options.onClearModulations) spec.onClearModulations = options.onClearModulations;

    if (options.dropTarget) {
      spec.dropTarget = options.dropTarget;
      spec.onModulationDrop = options.onDrop;
    }

    return spec;
  }

  function control(spec, value, onEdit) {
    return makeBodyControl(spec, value, onEdit);
  }

  /// A bare knob with a setter, for a faceplate that owns its own layout.
  function controlHandle(spec, value, onEdit) {
    return makeKnob(spec, value, onEdit);
  }

  function setting(label, control) {
    var box = document.createElement("div");
    box.className = "mcard-setting";
    var name = document.createElement("span");
    name.className = "mcard-setting-label";
    name.textContent = label.toUpperCase();
    box.appendChild(name);
    box.appendChild(control);
    return box;
  }

  /// The product page's header order: the control first, its caption after.
  function settingAfter(control, label) {
    var box = document.createElement("div");
    box.className = "mcard-setting";
    box.appendChild(control);
    var name = document.createElement("span");
    name.className = "mcard-setting-label";
    name.textContent = label.toUpperCase();
    box.appendChild(name);
    return box;
  }
