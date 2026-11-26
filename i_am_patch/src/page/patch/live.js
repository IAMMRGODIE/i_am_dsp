// Part nine: one poll for every live picture, and the scope strip.
  // The card the page is watching. Its two sides ride along with the sources'
  // rings in one request, so every live picture on the page is the same age.
  let scopeNode = null;
  let scopeCanvas = null;

  function stopScope() {
    scopeNode = null;

    if (scopeCanvas) {
      scopeCanvas.remove();
      scopeCanvas = null;
    }
  }

  function startScope(id) {
    if (scopeNode === id) return;

    stopScope();

    const node = nodeOf(id);
    const kind = node ? kindOf(node.kind) : null;

    // The strip draws a signal, so a node without an audio output has nothing to
    // show: a note, the host's MIDI port and the control sources would only get a
    // flat line that looks like silence.
    if (kind && !kind.outputs.some((port) => port.kind === "audio")) return;

    scopeNode = id;
  }

  // One tick for every live picture: the source cards' histories and the card
  // being watched. A picture read by the eye is enough at twenty a second, and
  // a hidden window does not need that many.
  const LIVE_INTERVAL = 20;
  const HIDDEN_INTERVAL = 150;
  let liveTimer = null;

  // A picture under the hand is left alone: a repaint would take the surface
  // the gesture is drawn on, and the drag would fight what the engine reports.
  let holding = false;

  window.addEventListener("pointerdown", () => { holding = true; }, true);
  window.addEventListener("pointerup", () => { holding = false; }, true);
  window.addEventListener("pointercancel", () => { holding = false; }, true);

  function startScopes() {
    if (liveTimer !== null) return;

    liveTimer = 0;
    pollLive();
  }

  async function pollLive() {
    const started = performance.now();

    try {
      const cursors = [];

      for (const [node, block] of sourceCursors) cursors.push(node + ":" + block);

      let query = "?since=" + cursors.join(",");

      if (scopeNode !== null) query += "&node=" + scopeNode;
      if (envelopeCards.size) query += "&meters=" + [...envelopeCards.keys()].join(",");

      const data = await (await fetch("/scopes" + query, { cache: "no-store" })).json();

      applySourceScopes(data);
      applyMeters(data.meters);
      applyLiveValues(data.values);

      if (data.live) applyLiveScope(data.live, data.rate);
    } catch {
      // The next tick tries again.
    }

    // Keep the cadence: the fetch's own time is not part of the interval, or
    // every point would arrive a little later than the one before it.
    const interval = document.hidden ? HIDDEN_INTERVAL : LIVE_INTERVAL;
    const elapsed = performance.now() - started;

    // A slow fetch or a heavy repaint is not a reason to start the next poll
    // immediately; a short floor keeps the loop from running flat out.
    liveTimer = setTimeout(pollLive, Math.max(4, interval - elapsed));
  }

  /// Draws the control sources' histories from one tick's payload.
  function applySourceScopes(data) {
    // A point drag owns the card until it ends: redrawing the hero here would
    // replace the handlers the drag is using.
    if (editingPoints) return;

    const blockFrames = (typeof data.block_frames === "number" && data.block_frames > 0)
      ? data.block_frames
      : 64;
    const rate = (typeof data.rate === "number" && data.rate > 0) ? data.rate : 44100;
    const blockMs = blockFrames / rate * 1000;

    for (const scope of data.scopes || []) {
      const entry = cards[scope.node];

      if (!entry || !SOURCE_KINDS.includes(entry.kind ? entry.kind.id : "")) continue;

      entry.phase = typeof scope.phase === "number" ? scope.phase : null;
      entry.input = typeof scope.input === "number" ? scope.input : null;
      entry.history = historyOf(scope.node);

      // The LFO editor's repaint clears the hero while dragging, which takes
      // the live layer with it; put it back before drawing into it again.
      if (entry.live && !entry.live.isConnected) {
        const svg = entry.card.querySelector(".mcard-hero svg");
        if (svg) svg.appendChild(entry.live);
      }

      const blocks = scope.blocks || [];
      const total = typeof scope.block === "number" ? scope.block : 0;

      if (blocks.length) {
        const cursor = sourceCursors.get(scope.node) || 0;
        const first = total - blocks.length;

        // A new scope, or a gap longer than the ring, starts a new picture: the
        // old points do not belong on the same line.
        if (total < cursor || first > cursor) entry.history.length = 0;

        // One point per block, timed by the block's own index: the trace is
        // drawn against audio time, so the poll's jitter never reaches the x
        // axis. Three seconds of points keep a little more than the 2.4 s the
        // card draws.
        const keep = Math.ceil(3000 / blockMs);

        for (let i = 0; i < blocks.length; i++) {
          entry.history.push({ t: (first + i) * blockMs, v: blocks[i] });
        }

        if (entry.history.length > keep) {
          entry.history.splice(0, entry.history.length - keep);
        }

        entry.lastValue = blocks[blocks.length - 1];
        sourceCursors.set(scope.node, total);
      }

      drawSourceLive(entry);
    }
  }

  /// The heroes that draw their law beside an envelope window.
  const ENVELOPE_HEROES = ["compressor", "gate", "expander", "limiter"];

  /// How many points that window holds, and how rare a repaint is: the poll runs
  /// at the scope's rate (about fifty a second), and a picture the eye reads does
  /// not need every one of those ticks.
  const ENVELOPE_MAX = 256;
  const ENVELOPE_EVERY = 3;

  /// The dynamics cards, by node id: they ask the poll for their level.
  const envelopeCards = new Map();

  /// Whether a card's hero draws an envelope: the compressor, the gate, the
  /// expander and the limiter do, and the rest do not.
  function showsEnvelope(kind) {
    const heroes = (kind.sections || []).map((section) => section.hero);

    heroes.push(kind.hero);

    return heroes.some((hero) => ENVELOPE_HEROES.includes(hero));
  }

  /// Asks the poll for this card's level, so its hero draws what the product page
  /// draws: the law on the left, the window of levels beside it, the marker on the
  /// law at the level it is at. Without it the hero falls back to one full-width
  /// curve — the layout a page with only parameters gets.
  function watchEnvelope(entry) {
    if (showsEnvelope(entry.kind)) envelopeCards.set(entry.node.id, entry);
  }

  /// One tick of the levels the dynamics cards draw, in dB.
  function applyMeters(meters) {
    if (!Array.isArray(meters)) return;

    const fromDb = (db) => Math.pow(10, (typeof db === "number" ? db : -120) / 20);

    for (const meter of meters) {
      const entry = envelopeCards.get(meter.node);

      if (!entry || !entry.host.isConnected || !entry.context) continue;

      entry.meterTick = (entry.meterTick || 0) + 1;

      const context = entry.context;
      const history = context.envHistory || (context.envHistory = { input: [], output: [] });

      // A window that has just opened starts at the level the card is at, the way
      // a meter does, instead of growing a trace from a single point.
      if (!history.input.length) {
        const input = fromDb(meter.input);
        const output = fromDb(meter.output);

        for (let index = 0; index < 128; index++) {
          history.input.push(input);
          history.output.push(output);
        }
      }

      history.input.push(fromDb(meter.input));
      history.output.push(fromDb(meter.output));

      while (history.input.length > ENVELOPE_MAX) {
        history.input.shift();
        history.output.shift();
      }

      context.meter = {
        input: typeof meter.input === "number" ? meter.input : null,
        output: typeof meter.output === "number" ? meter.output : null,
        sidechain: null,
      };

      if (entry.meterTick % ENVELOPE_EVERY === 0 && entry.card.repaintHero) {
        entry.card.repaintHero();
      }
    }
  }

  /// The heroes that edit their node rather than drawing one of its
  /// parameters: a live repaint would take the picture the hand works in.
  function isEditorHero(kind) {
    return (
      kind.id === "synth:drummer" ||
      kind.id === "mod:spectral_dynamics" ||
      kind.id === "mod:filter_table" ||
      kind.id === "mod:history" ||
      isProductCard(kind, "canvas") ||
      isProductCard(kind, "spectrum")
    );
  }

  /// How many polls the values have been read for.
  let liveTick = 0;

  /// One tick of the values a modulation is moving.
  ///
  /// A modulation leaves a parameter somewhere between its base and the
  /// source, so a card that draws that parameter draws where it actually is
  /// instead of where its knob was left. The picture is painted on every third
  /// tick: it is a picture the eye reads, not a sample-accurate trace.
  function applyLiveValues(values) {
    if (!Array.isArray(values) || !values.length || holding) return;

    liveTick += 1;

    const paint = liveTick % ENVELOPE_EVERY === 0;

    for (const [node, list] of values) {
      const entry = cards[node];

      if (!entry || !entry.card || !entry.card.liveValues) continue;
      if (isEditorHero(entry.kind)) continue;

      // A voice node has no value while no voice plays it, so the card goes
      // back to the base the page posted instead of freezing on the last note.
      const at = list.length ? list : Object.entries(entry.node.params || {});

      if (entry.card.liveValues(at) && paint && entry.card.repaintHero) {
        entry.card.repaintHero();
      }
    }
  }

  /// Draws the card the page is watching from the same tick's payload.
  function applyLiveScope(live, rate) {
    const id = live.node;
    const samples = live.output || [];
    const arriving = live.input || [];

    // A product card's hero draws both sides of the node, the way the product
    // page's does: the filter's spectrum sits on its response curve, with
    // what arrived at it behind what left it.
    const entry = cards[id];

    if (entry && entry.context && entry.card && samples.length > 1) {
      const mono = (frames) => frames.map(([left, right]) => (left + right) * 0.5);

      // The rate the host runs at, so a spectrum lands on the right axis.
      if (rate) entry.context.rate = rate;

      entry.context.scope.input = arriving.length > 1 ? mono(arriving) : null;
      entry.context.scope.output = mono(samples);

      // A goniometer hero reads four channels, not a mono sum: the picture is the
      // difference between left and right, which the sum throws away. The taps carry
      // the stereo frames, so both sides of the card are there to draw.
      const branch = (frames, side) => frames.map((frame) => frame[side] || 0);

      entry.context.stereo = {
        inputLeft: branch(arriving, 0),
        inputRight: branch(arriving, 1),
        outputLeft: branch(samples, 0),
        outputRight: branch(samples, 1),
      };

      entry.card.repaintHero();
    }

    // The strip is drawn last: repainting the hero replaces its markup, which
    // takes the canvas with it, and drawing it first flashed the waveform on
    // every poll.
    drawScope(id, samples);
  }

  function drawScope(id, samples) {
    const node = document.querySelector(`[data-node="${id}"]`);
    const card = node && node.querySelector(".mcard");

    if (!card) return;

    // A card that draws its own picture already shows the signal: a strip on top
    // of the hero would cover the picture and say the same thing twice.
    if (card.querySelector(".mcard-hero")) return;

    if (!scopeCanvas || !scopeCanvas.isConnected || scopeCanvas.parentElement !== card) {
      scopeCanvas = document.createElement("canvas");
      scopeCanvas.className = "scope-canvas";
      scopeCanvas.width = 512;
      scopeCanvas.height = 88;
      card.appendChild(scopeCanvas);
    }

    const context = scopeCanvas.getContext("2d");
    const width = scopeCanvas.width;
    const height = scopeCanvas.height;
    context.clearRect(0, 0, width, height);

    if (samples.length < 2) return;

    let peak = 0.0001;

    for (const [, right] of samples) {
      peak = Math.max(peak, Math.abs(right));
    }

    const scale = Math.min(1, 0.5 / peak);
    const accent = getComputedStyle(card).getPropertyValue("--accent").trim();
    context.beginPath();
    context.strokeStyle = accent || "rgba(224,122,47,.9)";
    context.lineWidth = 1.5;

    const step = samples.length / width;

    for (let x = 0; x < width; x++) {
      const index = Math.floor(x * step);
      const value = samples[index] ? samples[index][1] : 0;
      const y = height / 2 - value * scale * (height / 2 - 4);

      if (x === 0) context.moveTo(x, y);
      else context.lineTo(x, y);
    }

    context.stroke();
  }

