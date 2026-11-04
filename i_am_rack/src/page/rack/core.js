// Part two: the shared card context, the command queue and the modulation math.
  /// Every card gets its own rings: a module's card shows what entered and left
  /// that module, not the rack's own input and output.
  function cardContext() {
    return {
      scope: { input: null, output: null, sidechain: null },
      stereo: { inputLeft: null, inputRight: null, outputLeft: null, outputRight: null },
      meter: { input: null, sidechain: null },
      spectra: { input: null, output: null },
      eqSpectrum: { input: null, output: null, rate: 0, inputPath: "", outputPath: "" },
      envHistory: { input: [], output: [] },
      envWindow: 128,
      rate: pageRate
    };
  }

  function slotLetter(index) {
    return index < 26 ? String.fromCharCode(65 + index) : String(index + 1);
  }

  function el(tag, className, text) {
    var node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined) node.textContent = text;
    return node;
  }

  function queue(address, value) {
    pending[address] = address + "=" + value;
    if (timer) cancelAnimationFrame(timer);
    timer = requestAnimationFrame(flush);
  }

  function flush() {
    if (timer) {
      cancelAnimationFrame(timer);
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
        if (!result.ok) {
          status.textContent = "some commands were ignored";
          return;
        }

        // A normal edit is a new undo step and drops the redo branch; a
        // whole-state command refreshes the buttons from /rack right after.
        undoButton.disabled = false;
        redoButton.disabled = true;
      })
      .catch(function (error) {
        status.textContent = error.message;
      });
  }

  function sendNow(address, value) {
    pending[address] = address + "=" + value;
    return flush();
  }

  // A command that replaces the whole state must not share a batch with a
  // slider that is still in the queue: the server treats it as a lone line.
  function runSpecial(address, value) {
    return flush().then(function () {
      return sendNow(address, value);
    });
  }

  function refreshEditSlots(data) {
    undoButton.disabled = !data.history || !data.history.undo;
    redoButton.disabled = !data.history || !data.history.redo;

    var active = data.ab && data.ab.active === "B" ? "B" : "A";
    abButton.textContent = active;
    abButton.dataset.active = active;
    abButton.title = "A/B compare (now " + active + ")";
  }

  function historyStep(direction) {
    return runSpecial("history." + direction, "1").then(load);
  }

  function formatHz(value) {
    if (value >= 1000) {
      return (value / 1000).toFixed(value >= 10000 ? 1 : 2) + " kHz";
    }
    return Math.round(value) + " Hz";
  }

  function bandRange(data, index) {
    if (index > 2) return "20 Hz \u2013 20 kHz";

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

  function setModulationDepth(slot, depth) {
    queue("assign." + slot + ".depth", depth);
  }

  function attachHeightResize(handle, target, minimum, maximum, key) {
    try {
      var saved = Number(localStorage.getItem(key));
      if (saved > 0) target.style.height = saved + "px";
    } catch (ignore) {}

    var drag = null;

    handle.addEventListener("pointerdown", function (event) {
      if (event.button !== 0) return;
      drag = {
        y: event.clientY,
        height: target.getBoundingClientRect().height
      };
      handle.classList.add("dragging");
      handle.setPointerCapture(event.pointerId);
      event.preventDefault();
    });

    handle.addEventListener("pointermove", function (event) {
      if (!drag) return;
      var height = Math.max(minimum, Math.min(maximum, drag.height + event.clientY - drag.y));
      target.style.height = height + "px";
      event.preventDefault();
    });

    function stop(event) {
      if (!drag) return;
      drag = null;
      handle.classList.remove("dragging");
      try { localStorage.setItem(key, String(Math.round(target.getBoundingClientRect().height))); } catch (ignore) {}
      try { handle.releasePointerCapture(event.pointerId); } catch (ignore) {}
    }

    handle.addEventListener("pointerup", stop);
    handle.addEventListener("pointercancel", stop);
    handle.addEventListener("dblclick", function () {
      target.style.height = "";
      try { localStorage.removeItem(key); } catch (ignore) {}
    });
  }

  function toggleModulationMode(slots, mode) {
    slots.forEach(function (slot) {
      queue("assign." + slot + ".mode", mode);
    });
    flush().then(load);
  }

  function linkModulation(data, target) {
    var source = "";

    if (data.indexOf("source:") === 0) {
      source = data.slice(7);
    } else if (data.indexOf("macro:") === 0) {
      source = "macro" + (Number(data.slice(6)) + 1);
    }

    if (!source || !target) return;
    sendNow("assign.add", source + "|" + target).then(load);
  }

  // The card owns the normalisation, including the symmetric-log mapping a
  // bipolar parameter uses; asking it keeps the page and the knobs identical.
  function normalizeValue(spec, value) {
    return window.IAmCard.norm(spec, value);
  }

  function denormalizeValue(spec, position) {
    return window.IAmCard.at(spec, position);
  }

  /// Whether a source only ever runs forward, which decides how its value reads
  /// as a modulation amount.
  function unipolar(source) {
    if (source >= 4) return true;

    var kind = current && current.sources
      ? (current.sources.find(function (entry) { return entry.id === source; }) || {}).kind
      : "";

    return kind === "envelope" || kind === "pitch" || kind === "midi";
  }

  /// The values a card should draw after modulation: the engine applies this
  /// arithmetic on the audio thread, and the card reproduces it so a hero moves
  /// with the source instead of sitting at the knob's base value.
  function liveModulation(entry) {
    var module = entry.module;
    if (!module) return null;

    var prefix = "m" + entry.id + ".";
    var sums = Object.create(null);
    var any = false;

    (current && current.assignments ? current.assignments : []).forEach(function (assignment) {
      if (!assignment || !assignment.target) return;
      if (assignment.target.indexOf(prefix) !== 0) return;

      var value = live.values[String(assignment.source)];
      if (typeof value !== "number") return;

      var offset = assignment.mode === "centered"
        ? value * assignment.depth
        : (unipolar(assignment.source) ? value : (value + 1) * 0.5) * assignment.depth;

      var parameter = assignment.target.slice(prefix.length);
      sums[parameter] = (sums[parameter] || 0) + offset;
      any = true;
    });

    if (!any) return null;

    var out = Object.create(null);

    Object.keys(sums).forEach(function (parameter) {
      var control = (module.controls || []).find(function (entry) {
        return entry.id === parameter;
      });

      if (!control || control.kind !== "float") return;

      var base = typeof module.values[parameter] === "number"
        ? module.values[parameter]
        : control.default;
      var normalized = normalizeValue(control, base) + sums[parameter];

      out[parameter] = denormalizeValue(control, normalized);
    });

    return out;
  }

  /// The colour a source kind uses, in its card and on the knobs it drives.
  function sourceAccent(kind) {
    return kind === "lfo" ? "#4c8dff" :
      kind === "envelope" ? "#35c77a" :
        kind === "random" ? "#ff9a3d" :
          kind === "table" ? "#c8e04a" :
            kind === "pitch" ? "#b08cff" : "#ff6f91";
  }

  function sourceLabel(source) {
    return sourceNames[source] || ("Source " + (source + 1));
  }

  function sourceColor(source) {
    return sourceColors[source] || "var(--accent)";
  }

  function selectSource(id) {
    selectedSource = selectedSource === id ? null : id;
    load();
  }

  function modulationsFor(data, target) {
    var result = [];

    (data.assignments || []).forEach(function (assignment, slot) {
      if (!assignment || assignment.target !== target) return;
      if (selectedSource !== null && assignment.source !== selectedSource) return;

      result.push({
        slot: slot,
        source: assignment.source,
        label: sourceLabel(assignment.source),
        color: sourceColor(assignment.source),
        depth: assignment.depth,
        mode: assignment.mode || "centered",
        selected: selectedSource !== null
      });
    });

    return result;
  }
