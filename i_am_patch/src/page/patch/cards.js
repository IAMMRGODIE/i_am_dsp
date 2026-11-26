// Part four: the plain cards, the files they load and the context their heroes read.
  // A source card's history comes from the engine's tap for that node, and a
  // tap exists only after it is asked for once.
  const tapped = new Set();

  function tapSource(id) {
    if (tapped.has(id)) return;
    tapped.add(id);
    fetch(`/scope?node=${id}`, { cache: "no-store" }).catch(() => {});
  }

  // `values` is the card's own control map, passed when a control moved: the
  // page's copy of the node is only refreshed by the next full redraw, so the
  // hero has to be drawn from what the control just said.
  function drawSourceCard(entry, values) {
    if (!entry || !entry.card || !entry.kind) return;

    tapSource(entry.node.id);

    const svg = entry.card.querySelector(".mcard-hero svg");
    if (!svg) return;

    const dots = entry.card.querySelector(".mcard-hero .source-dots");
    const liveDots = entry.card.querySelector(".mcard-hero .source-live-dots");

    entry.dots = dots;
    entry.liveDots = liveDots;

    if (dots) dots.textContent = "";

    const params = Object.assign({}, (entry.node && entry.node.params) || {}, values || {});
    const midiMode = Math.round(params.mode ?? 0);
    const velocity = entry.kind.id === "midi" && midiMode === 1;
    const active = entry.kind.id === "midi"
      ? ((entry.node && (velocity ? entry.node.velocity_points : entry.node.points)) || [])
      : ((entry.node && entry.node.points) || []);
    const pasted = active.length >= 2;
    // A pasted curve wins; otherwise the shape decides. The derived curve stays
    // local: writing it back into the node would outrank the shape for good and
    // make the page disagree with the server on every poll.
    const points = pasted
      ? active
      : entry.kind.id === "midi" || entry.kind.id === "map"
        ? MIDI_MAPPING
        : IAmCard.sources.preset(Math.round(params.shape ?? 0));

    svg.textContent = "";
    drawSourceStatic(svg, entry.node, entry.kind.id, dots, params, points);

    if (entry.kind.id === "lfo") {
      IAmCard.sources.editLfo(svg, { points: points.slice() }, {
        shape: params.shape,
        dots,
        commit: (next) => {
          op({
            op: "set_points",
            id: entry.node.id,
            points: next.map((point) => [point.time, point.value, point.bend || 0]),
          }, false);
        },
        onStart: () => { editingPoints = true; },
        onEnd: () => { editingPoints = false; },
      });
    } else if (entry.kind.id === "midi") {
      // The mapping is the node's curve: a point is a note (or a velocity) and
      // the value it sends, so dragging one is how the map is edited. The two
      // modes keep separate point lists; the bend edits the segment after it.
      IAmCard.sources.editLfo(svg, { points: points.slice() }, {
        dots,
        paint: (next) => {
          svg.textContent = "";
          drawSourceStatic(svg, entry.node, "midi", dots, params, next);
          entry.live = el("g", { class: "source-live" });
          svg.appendChild(entry.live);
          drawSourceLive(entry);
        },
        commit: (next) => {
          op({
            op: "set_points",
            id: entry.node.id,
            curve: velocity ? "velocity" : "note",
            points: next.map((point) => [point.time, point.value, point.bend || 0]),
          }, false);
        },
        onStart: () => { editingPoints = true; },
        onEnd: () => { editingPoints = false; },
      });
    } else if (entry.kind.id === "map") {
      // The curve is the whole node, so dragging a point is how the mapping is
      // written: the engine reads the same points when it next builds.
      IAmCard.sources.editLfo(svg, { points: points.slice() }, {
        dots,
        paint: (next) => {
          svg.textContent = "";
          drawSourceStatic(svg, entry.node, "map", dots, params, next);
        },
        commit: (next) => {
          op({
            op: "set_points",
            id: entry.node.id,
            points: next.map((point) => [point.time, point.value, point.bend || 0]),
          }, false);
        },
        onStart: () => { editingPoints = true; },
        onEnd: () => { editingPoints = false; },
      });
    }

    // Last, so the editor's repaint of the points does not erase the playhead.
    // A MIDI card's editor paints its own live layer while it draws the
    // mapping; drop that one before adding the layer the poll updates, or the
    // old level line stays frozen under the moving one.
    for (const stale of svg.querySelectorAll(".source-live")) stale.remove();

    entry.live = el("g", { class: "source-live" });
    svg.appendChild(entry.live);
    drawSourceLive(entry);
  }

  // The rack's source cards put these choices in the header instead of a knob.
  const GENERIC_HEADER = {
    lfo: ["shape", "sync", "one_shot"],
    table: ["shape", "sync"],
    random: ["sync"],
    midi: ["mode"],
  };
  const NODE_NOTE = {
    "input:midi": "Host MIDI input",
    "input:audio": "Host audio input",
    "input:sidechain": "Host sidechain input",
    "out:audio": "Host audio output",
    "voice:in": "A note starts the voice group here",
    "voice:out": "The voice group's audio leaves here",
    "split:ms": "Mid and side, each one mono",
    "split:lr": "Left and right, each copied into both channels",
    "mix:ms": "Mid and side back into left and right",
    "mix:lr": "Left and right back into a stereo pair",
    "mix:bands": "Low, mid and high added back together",
  };

  function labelOf(key) {
    return key.replace(/_/g, " ").replace(/\b\w/g, (letter) => letter.toUpperCase());
  }


  function genericCard(node, kind, w, h) {
    let entry = cards[node.id];
    const mods = modsOf(node.id);

    // A macro's value is a host parameter, not a graph one: the card carries a
    // control for it anyway, and an edit of that control goes to /macro, the same
    // atomic a host writes. It is in the rebuild key so a host-driven move shows.
    const macroIndex = kind.id === "macro"
      ? Math.min(64, Math.max(1, Math.round(node.params.index ?? 1)))
      : 0;
    const macroLevel = macroIndex ? (macroValues[macroIndex - 1] || 0) : 0;

    const key = JSON.stringify([
      mods.map((mod) => [mod.slot, mod.to[1], Math.round(mod.depth * 1000), mod.mode]),
      node.points || [],
      node.params,
      macroLevel,
    ]);

    if (entry && (entry.key !== key || entry.kindId !== node.kind)) {
      entry.host.remove();
      delete cards[node.id];
      entry = null;
    }

    if (entry) {
      entry.node = node;
      entry.kind = kind;
      entry.host.style.setProperty(
        "--k",
        String(Math.max(0.05, Math.min(w / baseWidth(node), h / baseHeight(node)))),
      );
      entry.host.classList.toggle("compact", w < 430);
      // A note can change under the card — undo, a redo, an opened patch — so the
      // field follows what the graph now says. The caret keeps its own edits.
      const area = entry.host.querySelector(".note-text");

      if (area && document.activeElement !== area && area.value !== (node.text || "")) {
        area.value = node.text || "";
      }
      // Only a source card's picture is redrawn here. Every other card keeps the
      // hero its kind registered; painting the source picture over it would wipe
      // that hero on the next redraw.
      if (SOURCE_KINDS.includes(kind.id)) drawSourceCard(entry);

      return entry.host;
    }

    const context = contextOf(node);
    const meta = Object.fromEntries((kind.param_meta || []).map((item) => [item.name, item]));
    const specs = (kind.params || []).map(([id, fallback]) => {
      // The catalog carries the range the DSP itself declares; the fallbacks
      // only matter for a kind whose parameters were written by hand.
      const range = meta[id] || {};
      const choices = range.choices || null;
      const span = Math.max(0.001, Math.abs(fallback) || 1);
      const control = {
        id,
        label: labelOf(id),
        kind: range.boolean ? "bool" : choices || range.step === 1 ? "int" : "float",
        min: range.min ?? (range.boolean ? 0 : -span * 2),
        max: range.max ?? (choices ? choices.length - 1 : span * 2),
        default: fallback,
        log: !!range.log,
        step: choices ? 1 : range.step,
        choices,
      };

      // A depth is a share of the parameter's own range, which is the unit the
      // card's ring already works in: a logarithmic control then swings by a
      // ratio, the way its knob moves.
      const assignments = modsOf(node.id)
        .filter((mod) => mod.to[1] === control.id)
        .map((mod) => ({
          slot: mod.slot,
          source: mod.from[0],
          label: sourceName(mod.from[0]),
          depth: mod.depth,
          mode: mod.mode === "base" ? "frombase" : "centered",
        }));

      return IAmCard.modulate(control, {
        assignments,
        onDepth: (slot, next) => {
          touchNode(node.id);
          op(
            {
              op: "mod_depth",
              index: liveModSlot(node.id, id, slot, sourceOf(assignments, slot)),
              depth: next,
            },
            false,
          );
        },
        onMode: (slots, next) => {
          touchNode(node.id);
          op(
            {
              op: "mod_mode",
              index: liveModSlot(node.id, id, slots[0], sourceOf(assignments, slots[0])),
              mode: next === "center" ? "center" : "base",
            },
            false,
          );
        },
        // Ctrl + right click on the knob: every source comes off it.
        onClearModulations: (slots) => {
          // Highest first: removing one shifts every later slot, so the rest of the
          // list would point past the end. The last removal redraws, which is what
          // gives the other cards their slots back.
          const order = [...slots].sort((left, right) => right - left);

          order.forEach((slot, index) => {
            op({ op: "unmodulate", index: slot }, index === order.length - 1);
          });
        },
        dropTarget: node.id + ":" + id,
      });
    });

    // The macro's value rides beside the node's own parameters: it is a host
    // parameter, so it is written through /macro and never through `op param`.
    if (macroIndex) {
      specs.push({
        id: "value",
        label: "Value",
        kind: "float",
        min: 0,
        max: 1,
        default: macroLevel,
        log: false,
        step: 0.01,
        choices: null,
      });
    }

    const host = document.createElement("div");
    host.className = "card-host";
    const scale = document.createElement("div");
    scale.className = "card-scale";
    scale.style.width = `${baseWidth(node)}px`;
    scale.style.height = `${baseHeight(node)}px`;
    scale.style.flexBasis = `${baseWidth(node)}px`;

    const card = window.IAmCard.create({
      title: kind.name,
      accent: kind.accent || "#e07a2f",
      hero: CARD_HEROES[kind.id] || null,
      header: GENERIC_HEADER[kind.id] || [],
      switches: kind.switches || [],
      controls: specs,
      values: {
        ...node.params,
        ...(macroIndex ? { value: macroLevel } : {}),
        points: JSON.stringify((node.points || []).map((point) => (Array.isArray(point) ? point : [point.time, point.value, point.bend || 0]))),
      },
      context,
      onEdit: (identifier, value) => {
        // The macro value is not in the graph: it goes to the host parameter the
        // node names, so a drag lands in the automation lane the host reads.
        if (macroIndex && identifier === "value") {
          macroValues[macroIndex - 1] = value;
          queueMacro(macroIndex, value);
          return;
        }

        op({ op: "param", id: node.id, key: identifier, value }, false);
      },
    });

    // A text node is a note on the canvas: the card's body is the field, and
    // typing in it never redraws the graph — a rebuild would close the field
    // under the caret.
    if (kind.id === "text") {
      const body = card.querySelector(".mcard-body");

      if (body) {
        body.classList.add("card-note");

        const area = document.createElement("textarea");
        area.className = "note-text";
        area.spellcheck = false;
        area.placeholder = "Type a note";
        area.value = node.text || "";
        area.addEventListener("input", () => {
          node.text = area.value;
          queueText(node.id, area.value);
        });
        area.addEventListener("blur", flushTexts);
        body.appendChild(area);
      }
    }

    if (!specs.length) {
      const body = card.querySelector(".mcard-body");
      const note = NODE_NOTE[kind.id];

      if (body && note) {
        const row = document.createElement("div");
        row.className = "card-note";
        row.textContent = note;
        body.appendChild(row);
      }
    }

    scale.appendChild(card);
    host.appendChild(scale);
    entry = { host, key, kindId: node.kind, context, card, node, kind, samples: null, phase: null };
    cards[node.id] = entry;

    // A node that already holds a file shows it as soon as the card is built,
    // not only right after the drop that loaded it.
    if (IMPORT_ROUTES[kind.id]) refreshImport(entry, IMPORT_ROUTES[kind.id].path);

    if (SOURCE_KINDS.includes(kind.id)) {
      const repaint = card.repaintHero;
      card.repaintHero = function (values) {
        repaint(values);
        drawSourceCard(entry, values);
      };
      drawSourceCard(entry);
    }
    entry.host.style.setProperty(
      "--k",
      String(Math.max(0.2, Math.min(w / baseWidth(node), h / baseHeight(node)))),
    );
    entry.host.classList.toggle("compact", w < 430);

    return entry.host;
  }

  // A recorded note dropped or picked on a Modal card: the page posts it to
  // the node's route, which fits it and selects the result.
  let sampleInput = null;

  function pickSampleFile(onFile) {
    if (!sampleInput) {
      sampleInput = document.createElement("input");
      sampleInput.type = "file";
      sampleInput.accept = ".wav,.aif,.aiff";
      sampleInput.style.display = "none";
      document.body.appendChild(sampleInput);
    }

    sampleInput.onchange = () => {
      const file = sampleInput.files && sampleInput.files[0];
      sampleInput.value = "";
      if (file) onFile(file);
    };
    sampleInput.click();
  }

  function postSample(node, file) {
    const reader = new FileReader();

    reader.onload = async () => {
      const response = await fetch(
        "/sample?node=" + node + "&name=" + encodeURIComponent(file.name),
        { method: "POST", body: reader.result },
      );
      const result = response.ok ? await response.json() : null;

      if (result && result.ok) {
        patch = readPatch(await (await fetch("/patch")).json());
        draw();
      } else {
        showStatus((result && result.error) || "the sample could not be fitted");
      }
    };
    reader.readAsArrayBuffer(file);
  }

  // The kinds whose card takes a file, and the route that loads it.
  const IMPORT_ROUTES = {
    "synth:sampler": { path: "sampler", hint: "Load a sample: click or drop a .wav" },
    "synth:shapes": { path: "waves", hint: "Build the table from a .wav: click or drop a file" },
    "table:shapes": { path: "waves", hint: "Build the table from a .wav: click or drop a file" },
    "mod:convolver": { path: "ir", hint: "Load an impulse response: click or drop a file" },
  };

  /// Posts a file to a node's import route, then redraws what the card shows.
  function postImport(node, path, file) {
    const reader = new FileReader();

    reader.onload = async () => {
      const entry = cards[node];
      const mode = entry && entry.context && entry.context.waveMode;
      const response = await fetch(
        "/" + path + "?node=" + node + "&name=" + encodeURIComponent(file.name) +
          (mode ? "&mode=" + encodeURIComponent(mode) : ""),
        { method: "POST", body: reader.result },
      );
      const result = response.ok ? await response.json() : null;

      if (!result || !result.ok) {
        showStatus((result && result.error) || "the file could not be loaded");
        return;
      }

      showStatus("");
      await refreshImport(cards[node], path);
    };
    reader.readAsArrayBuffer(file);
  }

  /// The live state a card's heroes read, kept across the rebuilds of that card.
  const cardContexts = new Map();

  function contextOf(node) {
    let context = cardContexts.get(node.id);

    if (!context) {
      context = {
        rate: 48000,
        scope: { input: null, output: null },
        spectra: {},
        eqSpectrum: {},
        // The spectral dynamics cards draw these before their first fetch.
        spectralDynamics: { threshold: [], ratio: [] },
        sdCurve: { view: "threshold" },
      };

      // How long since this card last painted, for a picture with ballistics:
      // a card also repaints when a knob moves, and a peak must not decay
      // faster because of it. Read once per picture by the heroes that use it.
      let painted = 0;

      Object.defineProperty(context, "dt", {
        get() {
          const now = window.performance && window.performance.now
            ? window.performance.now()
            : Date.now();
          const ms = painted ? Math.max(1, Math.min(200, now - painted)) : 16;

          painted = now;
          return ms;
        },
      });

      // What an envelope card draws in its Shape hero: whatever node feeds its
      // oscillator input, read when the picture is painted so it never lags the
      // source's own parameters behind.
      Object.defineProperty(context, "oscSource", {
        get() {
          const edge = patch.edges.find(
            (candidate) => candidate.to[0] === node.id && candidate.to[1] === 1,
          );
          const upstream = edge ? nodeOf(edge.from[0]) : null;

          return upstream ? { kind: upstream.kind, params: upstream.params || {} } : null;
        },
      });

      // What feeds one of this node's inputs, read when the picture is painted.
      context.source = (port) => {
        const edge = patch.edges.find(
          (candidate) => candidate.to[0] === node.id && candidate.to[1] === port,
        );

        return edge ? nodeOf(edge.from[0]) : null;
      };

      cardContexts.set(node.id, context);
    }

    return context;
  }

  /// The slot a modulation has right now.
  ///
  /// A card keeps the slot number it was built with, and removing a modulation shifts
  /// every later one: a depth or mode edit from a stale card then lands on the wrong
  /// link (or past the end, where the graph refuses it), which is how an alt-drag looked
  /// like it needed a click on the card to take.
  ///
  /// A link is identified by its source and its parameter, never by "the first
  /// modulation on this parameter": with two sources on one knob that match sent every
  /// row's edit to the first one, which is how adjusting the macro moved the LFO.
  function liveModSlot(node, key, slot, source) {
    const link = (mod) =>
      mod.to[0] === node &&
      mod.to[1] === key &&
      (source === undefined || mod.from[0] === source);

    if (patch.mods[slot] && link(patch.mods[slot])) return slot;

    const index = patch.mods.findIndex(link);

    return index >= 0 ? index : slot;
  }

  /// The source a panel row names. The panel hands back the slot the card was
  /// built with, and the assignment it came from is what says which link that
  /// slot was.
  function sourceOf(assignments, slot) {
    const assignment = assignments.find((item) => item.slot === slot);

    return assignment ? assignment.source : undefined;
  }

  /// Watches the node a card edit came from.
  ///
  /// A press on a control inside a card never selects the node, and the live poll only
  /// covers the selected one, so an edit there was made without the data the card
  /// draws from: its picture and the modulation rings only caught up on the click that
  /// finally selected it. Editing a card is a reason to watch it.
  function touchNode(id) {
    if (!selection.has(id)) {
      selection.add(id);
      primary = id;
    }

    startScope(id);
  }

  /// Where a press sits along a hero, as a share of it.
  function heroShare(hero, event) {
    const rect = hero.getBoundingClientRect();

    return Math.min(1, Math.max(0, ((event.clientX - rect.left) / rect.width * 520 - 20) / 480));
  }

  /// What a hero's behaviour acts on for a patch node: the card's own values and
  /// the graph behind them, instead of the product page's parameter map.
  function heroContextFor(node, kind) {
    const route = IMPORT_ROUTES[kind.id];
    const entryOf = () => cards[node.id];
    const cardOf = () => (entryOf() && entryOf().card) || null;
    const state = () => (entryOf() && entryOf().context) || {};
    const heroOf = () => {
      const card = cardOf();

      return card ? card.querySelector(".mcard-hero") : null;
    };

    // The card paints the gesture at once; the graph hears every step, in the
    // order the hand made them, so the sound follows instead of waiting for the
    // pointer to come up.
    const set = (id, value) => {
      node.params[id] = value;

      // A hero gesture is an edit of this node: watch it, or its picture is left
      // with the data it had when the card was built.
      touchNode(node.id);

      const card = cardOf();

      if (card && card.setValue) card.setValue(id, value);

      postParam(node.id, id, value);
    };

    const postFades = () => {
      const ir = state().ir || {};

      return queued(() =>
        fetch(
          "/ir?node=" + node.id + "&fade_in=" + (ir.fade_in || 0) +
            "&fade_out=" + (ir.fade_out || 0),
          { method: "POST", body: "fades" },
        ).catch(() => {}),
      );
    };

    const post = async (ids) => {
      // The gesture is over: wait for its writes, then rebuild the card from the
      // graph, which is now what the hand left it at.
      await writeChain;

      if (kind.id === "mod:convolver") return;

      for (const id of ids) await postParam(node.id, id, node.params[id]);

      await reload();
    };

    const frames = () => (state().waves && state().waves.count) || 0;

    return {
      // The spectral dynamics editor reads its pages on every use, so these are
      // getters: a card that has just been refetched hands the editor the new
      // lists instead of the ones it was wired with.
      get spectralDynamics() {
        return state().spectralDynamics || { threshold: [], ratio: [] };
      },
      get sdCurve() {
        const context = state();

        return context.sdCurve || (context.sdCurve = { view: "threshold" });
      },
      read: () => {
        const current = state();

        if (kind.id === "mod:convolver") return current.ir;
        if (kind.id === "synth:shapes" || tableProduct(kind) === "shapes") return current.waves;

        return current.sample;
      },
      value: (id) => node.params[id] ?? 0,
      set,
      toggle: (id) => set(id, (node.params[id] ?? 0) > 0.5 ? 0 : 1),
      commit: post,
      action: async (name, query) => {
        if (!route) return;

        await postImportAction(node.id, route.path, name, query);
      },
      pick: (file) => {
        if (!route) return;

        if (file) postImport(node.id, route.path, file);
        else pickSampleFile((picked) => postImport(node.id, route.path, picked));
      },
      share: (event) => {
        const hero = heroOf();

        return hero ? heroShare(hero, event) : 0;
      },
      near: (at) => {
        const sample = state().sample;

        return !!(
          sample &&
          sample.slices &&
          sample.slices.some((mark) => Math.abs(mark - at) < 0.02)
        );
      },
      repaint: () => {
        // The fades are the convolver's own state, so the picture and the graph
        // move together.
        if (kind.id === "mod:convolver") postFades();

        const card = cardOf();

        if (card && card.repaintHero) card.repaintHero();
      },
      frame: (step) => stepFrame(node, step),
      frameIndex: () => IAmPage.index(node.params["oscillator.smooth_factor"] ?? 0, frames()),
      cycleMode: () => {
        const entry = entryOf();

        if (!entry) return;

        const modes = ["auto", "fixed", "pitch", "single"];
        const index = modes.indexOf(entry.context.waveMode || "auto");

        entry.context.waveMode = modes[(index + 1) % modes.length];

        const card = cardOf();

        if (card && card.repaintHero) card.repaintHero();
      },
    };
  }

  /// Moves the morph to the next whole frame, the way the shape hero's arrows
  /// do.
  async function stepFrame(node, step) {
    const entry = cards[node.id];
    const count = (entry && entry.context.waves && entry.context.waves.count) || 0;

    if (count <= 1) return;

    const current = node.params["oscillator.smooth_factor"] ?? 0;
    const frame = Math.round(current * (count - 1)) + step;
    const clamped = Math.min(Math.max(frame, 0), count - 1);

    await op(
      {
        op: "param",
        id: node.id,
        key: "oscillator.smooth_factor",
        value: clamped / (count - 1),
      },
      false,
    );
    await reload();
  }

  /// Posts one of the hero's edit actions and redraws the picture.
  async function postImportAction(node, path, action, query) {
    const extra = Object.entries(query || {})
      .map(([key, value]) => "&" + key + "=" + value)
      .join("");
    const response = await fetch(
      "/" + path + "?node=" + node + "&action=" + encodeURIComponent(action) + extra,
      { method: "POST", body: action },
    );
    const result = response.ok ? await response.json() : null;

    if (!result || !result.ok) {
      showStatus((result && result.error) || "the action did not run");
      return;
    }

    showStatus("");
    await refreshImport(cards[node], path);
  }

