// Part five: the heroes that edit their node, the canvas, the spectrum and the drummer.
  /// Writes one parameter without the page's undo snapshot or a redraw.
  ///
  /// A gesture that is still running posts as it goes, so the sound follows the
  /// hand and the graph ends up with what the card already shows. A parameter
  /// is cheap now: the engine takes the value without rebuilding.
  /// A card that draws another node's state (the envelope's Shape hero) follows
  /// a parameter edit on that node without a full redraw.
  function repaintDependents(nodeId) {
    for (const entry of Object.values(cards)) {
      if (!entry || !entry.card || entry.kindId !== "adsr") continue;

      const edge = patch.edges.find(
        (candidate) => candidate.to[0] === entry.node.id && candidate.to[1] === 1,
      );

      if (edge && edge.from[0] === nodeId && entry.card.repaintHero) {
        entry.card.repaintHero();
      }
    }
  }

  // One cycle of each constructor node, as the engine's own builder renders it.
  const waveCache = new Map();
  const waveKeys = new Map();

  /// Asks for the sampled cycle of every constructor node, so a hero draws the
  /// DSP's output instead of a guess.
  function refreshWaves(force) {
    const live = new Set(patch.nodes.map((node) => node.id));

    for (const id of [...waveCache.keys()]) {
      if (!live.has(id)) {
        waveCache.delete(id);
        waveKeys.delete(id);
      }
    }

    for (const node of patch.nodes) {
      if (
        !(
          node.kind.startsWith("table:") ||
          node.kind.startsWith("osc:") ||
          node.kind.startsWith("warp:")
        )
      ) {
        continue;
      }

      const key = JSON.stringify(node.params);

      if (!force && waveKeys.get(node.id) === key) continue;

      waveKeys.set(node.id, key);
      fetch("/wave?node=" + node.id + "&points=192", { cache: "no-store" })
        .then((response) => (response.ok ? response.json() : null))
        .then((body) => {
          if (!body || !body.values) return;

          waveCache.set(node.id, body.values);
          repaintHeroes();
        })
        .catch(() => {});
    }
  }

  /// Heroes that draw another node (an envelope's source, a warp's table) follow
  /// a structural edit too: rewiring changes what they should show.
  function repaintHeroes() {
    for (const entry of Object.values(cards)) {
      if (entry && entry.card && entry.card.repaintHero) entry.card.repaintHero();
    }
  }

  function postParam(node, key, value) {
    repaintDependents(node);

    return queued(() =>
      fetch("/op", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ op: "param", id: node, key, value }),
      }).catch(() => {}),
    );
  }

  /// Reloads the graph from the server and draws it, so a card that shows a
  /// parameter picks the new value up.
  async function reload() {
    patch = readPatch(await (await fetch("/patch")).json());
    refreshWaves();
    draw();
  }

  /// A hero's editor attaches to the markup a repaint replaces: watch the box
  /// and re-attach to whatever the repaint puts there, or only the first drag
  /// works.
  function watchHero(entry, wire) {
    const hero = entry.card && entry.card.querySelector(".mcard-hero");

    if (!hero || entry.heroWatch) return;

    entry.heroWatch = new MutationObserver(() => {
      if (!entry.host || !entry.host.isConnected) {
        entry.heroWatch.disconnect();
        return;
      }

      wire(entry);
    });

    entry.heroWatch.observe(hero, { childList: true });
  }

  /// The Canvas node's drawing is not a parameter: the product keeps it, and
  /// the card asks for it the way the product page does.
  function refreshCanvas(entry) {
    if (!entry || !isProductCard(entry.kind, "canvas")) return;

    // A drawing under the hand is not replaced: the pointer capture sits on the
    // svg the new picture would take away.
    const hero = entry.card && entry.card.querySelector(".mcard-hero");

    if (hero && hero.dataset.busy === "1") return;

    fetch("/canvas?node=" + entry.node.id, { cache: "no-store" })
      .then((response) => (response.ok ? response.json() : null))
      .then((state) => {
        if (!state || !state.shape) return;

        const same = entry.context.canvas && entry.context.canvas.shape === state.shape;
        entry.context.canvas = state;

        if (same) wireCanvasCard(entry);
        else if (entry.card.repaintHero) entry.card.repaintHero();
      })
      .catch(() => {});
  }

  /// The drawing lives in the node's slot, and it can change under the card: an
  /// undo puts the old text back, and a fetch that was lost at build time would
  /// leave the card on its placeholder for good. The product page polls its own
  /// canvas; a patch card asks again on a slow timer, and only repaints when the
  /// drawing really moved.
  function watchCanvas(entry) {
    if (entry.canvasTimer) return;

    const tick = () => {
      entry.canvasTimer = null;

      if (!entry.host.isConnected) return;

      refreshCanvas(entry);
      entry.canvasTimer = setTimeout(tick, 500);
    };

    entry.canvasTimer = setTimeout(tick, 500);
  }

  /// A canvas hero on a patch card edits this node's drawing and morph, not a
  /// page-wide canvas, so its writes carry the node id.
  function wireCanvasCard(entry) {
    if (!entry || !isProductCard(entry.kind, "canvas") || !entry.context.canvas) return;
    const hero = entry.card && entry.card.querySelector(".mcard-hero");

    if (!hero) return;

    const svg = hero.querySelector("svg");

    if (!svg || svg.__canvasWired) return;

    svg.__canvasWired = true;
    watchHero(entry, wireCanvasCard);

    IAmCard.wire(hero, "canvas", entry.node.params, {
      canvas: entry.context.canvas,
      set: (id, value) => {
        entry.node.params[id] = value;
        postParam(entry.node.id, id, value);

        // The product page repaints through its dirty flag; a patch card has to
        // move its own control and picture, or a page button would leave the old
        // reading up.
        requestAnimationFrame(() => {
          if (!entry.host.isConnected) return;

          if (entry.card.setValue) entry.card.setValue(id, value);
          else if (entry.card.repaintHero) entry.card.repaintHero();
        });
      },
      post: (text) => {
        fetch("/canvas?node=" + entry.node.id, { method: "POST", body: text })
          .then((response) => (response.ok ? response.json() : null))
          .then((result) => {
            if (result && result.error) showStatus(result.error);
          })
          .catch(() => {});
      },
    });
  }

  /// The OTT and the vocoder draw a meter, and a meter is only written by the
  /// effect as it plays: the card asks its own node for the bands, the way the
  /// product's page asks the one product it has.
  ///
  /// Without this the hero is an empty grid that never moves — the effect a node
  /// builds keeps its meters to itself, so nothing on the page had them.
  function watchBands(entry) {
    if (entry.bandsTimer) return;

    const hero = entry.card && entry.card.querySelector(".mcard-hero");

    if (!hero) return;

    const key = entry.kind.hero === "ott" ? "ott" : "vocoder";
    let last = "";

    const tick = () => {
      entry.bandsTimer = null;

      // The card left the page: the timer goes with it, and the next card for
      // this node starts its own.
      if (!entry.host.isConnected) return;

      if (document.hidden) {
        entry.bandsTimer = setTimeout(tick, 250);
        return;
      }

      fetch("/bands?node=" + entry.node.id, { cache: "no-store" })
        .then((response) => (response.ok ? response.json() : null))
        .then((state) => {
          if (!state) return;

          // A meter that has not moved does not need a repaint: the whole hero is
          // rebuilt each time, and a still picture is a wasted frame.
          const text = JSON.stringify(state);

          if (text === last) return;

          last = text;
          entry.context[key] = state;

          if (entry.card.repaintHero) entry.card.repaintHero();
        })
        .catch(() => {})
        .then(() => {
          if (entry.host.isConnected) entry.bandsTimer = setTimeout(tick, 60);
        });
    };

    entry.bandsTimer = setTimeout(tick, 60);
  }

  /// The Spectrum node's pages are outside the parameter map too: the card
  /// asks the product's route and the editor writes them back to the same node.
  function refreshSpectrum(entry) {
    if (!entry || !isProductCard(entry.kind, "spectrum")) return;

    fetch("/spectrum?node=" + entry.node.id, { cache: "no-store" })
      .then((response) => (response.ok ? response.json() : null))
      .then((state) => {
        if (!state || !state.frames) return;

        // The hero's handlers hold this object, so its pages are replaced in
        // place: a fresh object would leave them editing the old snapshot and
        // posting it back over the new one.
        const holder = entry.context.spectrum || { page: 0 };
        holder.generation = state.generation;
        holder.gains = state.gains;
        holder.frames = state.frames;
        entry.context.spectrum = holder;

        if (entry.card.repaintHero) entry.card.repaintHero();
      })
      .catch(() => {});
  }

  /// A spectrum hero on a patch card edits this node's pages and morph, not a
  /// page-wide spectrum, so its writes carry the node id.
  function wireSpectrumCard(entry) {
    if (!entry || !isProductCard(entry.kind, "spectrum") || !entry.context.spectrum) return;
    const hero = entry.card && entry.card.querySelector(".mcard-hero");

    if (!hero) return;

    const svg = hero.querySelector("svg");

    if (!svg || svg.__spectrumWired) return;

    svg.__spectrumWired = true;
    watchHero(entry, wireSpectrumCard);

    IAmCard.wire(hero, "spectrum", entry.node.params, {
      spectrum: entry.context.spectrum,
      set: (id, value) => {
        entry.node.params[id] = value;
        postParam(entry.node.id, id, value);

        requestAnimationFrame(() => {
          if (!entry.host.isConnected) return;

          if (entry.card.setValue) entry.card.setValue(id, value);
          else if (entry.card.repaintHero) entry.card.repaintHero();
        });
      },
      post: (body) => {
        fetch("/spectrum?node=" + entry.node.id, { method: "POST", body })
          .then((response) => (response.ok ? response.json() : null))
          .then((result) => {
            if (result && result.error) showStatus(result.error);
          })
          .catch(() => {});
      },
    });
  }

  /// The Drummer's click and six slots are the product's own state, so the
  /// card asks for both and its buttons post back through the node.
  function refreshDrummer(entry) {
    if (!entry || entry.kind.id !== "synth:drummer") return;

    Promise.all([
      fetch("/click?node=" + entry.node.id, { cache: "no-store" })
        .then((response) => (response.ok ? response.json() : null))
        .catch(() => null),
      fetch("/samples?node=" + entry.node.id, { cache: "no-store" })
        .then((response) => (response.ok ? response.json() : null))
        .catch(() => null),
      fetch("/pitch?node=" + entry.node.id, { cache: "no-store" })
        .then((response) => (response.ok ? response.json() : null))
        .catch(() => null),
      fetch("/amp?node=" + entry.node.id, { cache: "no-store" })
        .then((response) => (response.ok ? response.json() : null))
        .catch(() => null),
    ]).then(([click, samples, pitch, amp]) => {
      if (click) entry.context.click = click;
      if (samples) entry.context.samples = samples;
      if (pitch && pitch.points) entry.context.fall = pitch;
      if (amp && amp.curves) entry.context.ampCurves = amp.curves;

      if (entry.card.repaintHero) entry.card.repaintHero();
    });
  }

  /// Where a hero's 520x160 picture sits after its meet scaling.
  function drummerMap(hero) {
    const svg = hero.querySelector("svg");

    if (!svg) return null;

    const rect = svg.getBoundingClientRect();

    if (!rect.width || !rect.height) return null;

    const scale = Math.min(rect.width / 520, rect.height / 160);

    return {
      scale,
      left: rect.left + (rect.width - 520 * scale) / 2,
      top: rect.top + (rect.height - 160 * scale) / 2,
    };
  }

  /// The amp axis: a cymbal rings for ten seconds, the snare's pitch is short,
  /// and every other kit piece gets at least one.
  function drummerAmpSpan(points, drum) {
    const last = points && points.length ? points[points.length - 1][0] : 1;

    if (drum === 8) return Math.max(0.03, last * 1.1);
    if (drum === 5) return Math.max(10, last * 1.1);

    return Math.max(1, last * 1.1);
  }

  function drummerCurveBody(points) {
    return points
      .map((point) => point[0].toFixed(4) + ":" + point[1].toFixed(4) + ":" + (point[2] || 0).toFixed(3))
      .join(";");
  }

  function clamp(value, low, high) {
    return Math.min(high, Math.max(low, value));
  }

  /// A Drummer card loads a click, its slots and its curves through its own
  /// node's routes. The buttons, rows and curve handles are one delegated
  /// gesture on the hero, so every page the card can show is covered.
  function wireDrummerCard(entry) {
    if (entry.drummerWired) return;

    const hero = entry.card && entry.card.querySelector(".mcard-hero");

    if (!hero) return;

    entry.drummerWired = true;

    function finish() {
      const drag = entry.drumDrag;

      if (!drag) return;

      entry.drumDrag = null;
      window.removeEventListener("pointermove", onMove);
      window.removeEventListener("pointerup", onUp);
      window.removeEventListener("pointercancel", onUp);

      const body = drummerCurveBody(drag.points);
      const url = drag.isFall
        ? "/pitch?node=" + entry.node.id
        : "/amp?node=" + entry.node.id + "&drum=" + drag.drum;

      fetch(url, { method: "POST", body })
        .then((response) => (response.ok ? response.json() : null))
        .then((result) => {
          if (result && result.error) showStatus(String(result.error));
          refreshDrummer(entry);
        })
        .catch(() => {});
    }

    function onUp(event) {
      finish();
      event.preventDefault();
    }

    function onMove(event) {
      const drag = entry.drumDrag;

      if (!drag) return;

      const map = drummerMap(drag.hero);

      if (!map) return;

      const x = (event.clientX - map.left) / map.scale;
      const y = (event.clientY - map.top) / map.scale;
      const points = drag.points;

      if (drag.kind === "point" && drag.isFall) {
        const time = clamp((x - 26) / 468, 0, 1) * 0.25;
        const ratio = Math.pow(2, clamp((132 - y) / 112, 0, 1) * 10);
        const lower = drag.index > 0 ? points[drag.index - 1][0] + 0.0005 : 0;
        const upper = drag.index < points.length - 1 ? points[drag.index + 1][0] - 0.0005 : 0.25;

        points[drag.index][0] = drag.index === 0 ? 0 : clamp(time, lower, Math.max(lower, upper));
        points[drag.index][1] = ratio;
      } else if (drag.kind === "point") {
        const span = drummerAmpSpan(points, drag.drum);
        const time = clamp((x - 26) / 468, 0, 1) * span;
        const value = clamp((132 - y) / 112, 0, 1) * (drag.drum === 8 ? 10 : 1);
        const lower = drag.index === 0 ? 0 : points[drag.index - 1][0] + 0.001;
        const upper = drag.index === points.length - 1
          ? span
          : points[drag.index + 1][0] - 0.001;

        points[drag.index][0] = clamp(time, lower, Math.max(lower, upper));
        points[drag.index][1] = value;
      } else {
        const travel = (drag.startY - event.clientY) / Math.max(map.scale, 1e-6);

        points[drag.index][2] = clamp(
          drag.startBend + travel * 0.09 * drag.direction,
          -10,
          10,
        );
      }

      entry.card.repaintHero();
      event.preventDefault();
    }

    function pageCurve() {
      const page = entry.pageName || "";
      const isFall = page === "Kick" && (entry.context.kickMode || "pitch") !== "amp";
      const drum = page === "Kick"
        ? 0
        : page === "Snare"
          ? ({ base: 6, noise: 7, pitch: 8 }[entry.context.snareMode || "base"] ?? 6)
          : ({ Hat: 2, Clap: 3, Tom: 4, Cymbal: 5 }[page] ?? -1);
      const points = isFall
        ? (entry.context.fall && entry.context.fall.points)
        : (drum >= 0 && entry.context.ampCurves ? entry.context.ampCurves[drum] : null);

      return { isFall, drum, points };
    }

    function pointOf(event, isFall, points, drum) {
      const map = drummerMap(hero);

      if (!map) return null;

      const x = (event.clientX - map.left) / map.scale;
      const y = (event.clientY - map.top) / map.scale;

      if (isFall) {
        return {
          time: clamp((x - 26) / 468, 0, 1) * 0.25,
          value: Math.pow(2, clamp((132 - y) / 112, 0, 1) * 10),
        };
      }

      const span = drummerAmpSpan(points, drum);

      return {
        time: clamp((x - 26) / 468, 0, 1) * span,
        value: clamp((132 - y) / 112, 0, 1) * (drum === 8 ? 10 : 1),
      };
    }

    function postCurve(isFall, drum, points) {
      const body = drummerCurveBody(points);
      const url = isFall
        ? "/pitch?node=" + entry.node.id
        : "/amp?node=" + entry.node.id + "&drum=" + drum;

      fetch(url, { method: "POST", body })
        .then((response) => (response.ok ? response.json() : null))
        .then((result) => {
          if (result && result.error) showStatus(String(result.error));
          refreshDrummer(entry);
        })
        .catch(() => {});
    }

    hero.addEventListener("pointerdown", (event) => {
      if (event.button !== 0) return;

      const action = event.target.closest("[data-action]");
      const load = event.target.closest("[data-load]");
      const clear = event.target.closest("[data-clear]");
      const kick = event.target.closest("[data-kick-mode]");
      const snare = event.target.closest("[data-snare-mode]");

      if (action && action.getAttribute("data-action") === "load") {
        event.preventDefault();
        pickSampleFile((file) => postDrummer(entry, "click", "", file));
        return;
      }

      if (load) {
        event.preventDefault();
        const slot = load.getAttribute("data-load");

        pickSampleFile((file) => postDrummer(entry, "samples", slot, file));
        return;
      }

      if (clear) {
        event.preventDefault();
        const slot = clear.getAttribute("data-clear");

        fetch("/samples?node=" + entry.node.id + "&slot=" + encodeURIComponent(slot) + "&clear=1", {
          method: "POST",
          body: "clear",
        })
          .then((response) => (response.ok ? response.json() : null))
          .then(() => refreshDrummer(entry))
          .catch(() => {});
        return;
      }

      if (kick) {
        entry.context.kickMode = kick.getAttribute("data-kick-mode");
        event.preventDefault();
        entry.card.repaintHero();
        return;
      }

      if (snare) {
        entry.context.snareMode = snare.getAttribute("data-snare-mode");
        event.preventDefault();
        entry.card.repaintHero();
        return;
      }

      const page = entry.pageName || "";
      const isFall = page === "Kick" && (entry.context.kickMode || "pitch") !== "amp";
      const ampDrum = page === "Kick"
        ? 0
        : page === "Snare"
          ? ({ base: 6, noise: 7, pitch: 8 }[entry.context.snareMode || "base"] ?? 6)
          : ({ Hat: 2, Clap: 3, Tom: 4, Cymbal: 5 }[page] ?? -1);

      if (!isFall && ampDrum < 0) return;

      const point = event.target.closest(isFall ? "[data-point]" : "[data-amp-point]");
      const bend = event.target.closest(isFall ? "[data-bend]" : "[data-amp-bend]");

      if (!point && !bend) return;

      const map = drummerMap(hero);

      if (!map) return;

      const points = isFall
        ? (entry.context.fall && entry.context.fall.points)
        : (entry.context.ampCurves && entry.context.ampCurves[ampDrum]);

      if (!points) return;

      const join = (point || bend).getAttribute(
        isFall ? (point ? "data-point" : "data-bend") : (point ? "data-amp-point" : "data-amp-bend"),
      );
      const index = Number(join);

      if (!Number.isFinite(index) || !points[index]) return;

      entry.drumDrag = {
        kind: point ? "point" : "bend",
        isFall,
        drum: ampDrum,
        points,
        index,
        hero,
        startY: event.clientY,
        startBend: points[index][2] || 0,
        direction: points[index + 1] && points[index + 1][1] > points[index][1] ? -1 : 1,
      };

      try {
        hero.setPointerCapture(event.pointerId);
      } catch (error) {
        // A synthetic pointer has nothing to capture.
      }

      window.addEventListener("pointermove", onMove);
      window.addEventListener("pointerup", onUp);
      window.addEventListener("pointercancel", onUp);
      event.preventDefault();
      event.stopPropagation();
    });

    hero.addEventListener("dblclick", (event) => {
      const curve = pageCurve();

      if (!curve.points || curve.points.length >= 10) return;
      if (event.target.closest("[data-point],[data-bend],[data-amp-point],[data-amp-bend]")) return;

      const position = pointOf(event, curve.isFall, curve.points, curve.drum);

      if (!position) return;

      curve.points.push([position.time, position.value, 0]);
      curve.points.sort((a, b) => a[0] - b[0]);
      entry.card.repaintHero();
      postCurve(curve.isFall, curve.drum, curve.points);
      event.preventDefault();
      event.stopPropagation();
    });

    hero.addEventListener("contextmenu", (event) => {
      const curve = pageCurve();

      if (!curve.points) return;

      const target = event.target.closest(curve.isFall ? "[data-point]" : "[data-amp-point]");

      if (!target) return;

      const index = Number(target.getAttribute(curve.isFall ? "data-point" : "data-amp-point"));

      if (!Number.isFinite(index) || curve.points.length <= 2) return;

      curve.points.splice(index, 1);
      entry.card.repaintHero();
      postCurve(curve.isFall, curve.drum, curve.points);
      event.preventDefault();
      event.stopPropagation();
    });
  }

  /// Posts one file to the click or a sample slot, then redraws the card.
  function postDrummer(entry, route, slot, file) {
    const reader = new FileReader();

    reader.onload = () => {
      const query = "?node=" + entry.node.id + (slot ? "&slot=" + encodeURIComponent(slot) : "");

      fetch("/" + route + query, { method: "POST", body: reader.result })
        .then((response) => (response.ok ? response.json() : null))
        .then((result) => {
          if (result && result.error) showStatus(String(result.error));
          else showStatus("");
          refreshDrummer(entry);
        })
        .catch(() => {});
    };

    reader.readAsArrayBuffer(file);
  }

  /// The script's source, error and picture are the product's own state. The
