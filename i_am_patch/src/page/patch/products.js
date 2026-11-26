// Part six: the script, spectral, filter table and history editors, and the product card.
  /// card asks for it through the node's route and shows it in the panel and the
  /// hero.
  ///
  /// Every scripted product carries the panel (the catalog's `script` flag), not
  /// only the two synth ids: the mapper and the distortion are scripted too, and
  /// their heroes draw the curve the script produces.
  function refreshScript(entry) {
    if (!entry || !entry.kind.script) return;

    fetch("/script?node=" + entry.node.id, { cache: "no-store" })
      .then((response) => (response.ok ? response.json() : null))
      .then((state) => {
        if (!state) return;

        entry.context.script = state;
        // The mapper and the distortion draw points; the generator draws its
        // spectrum, which the same state carries under another name.
        if (state.curve) entry.context.curve = { points: state.curve };
        if (entry.card.setScript) entry.card.setScript(state.source || "");
        if (entry.card.setScriptError) entry.card.setScriptError(state.error || "");
        if (entry.card.repaintHero) entry.card.repaintHero();
      })
      .catch(() => {});
  }

  /// A script compiles on the worker's own thread, so its picture lands a moment
  /// after its source: the card keeps asking while it is on the page, at the
  /// cadence the product page uses.
  function watchScript(entry) {
    if (entry.scriptTimer) return;

    const tick = () => {
      entry.scriptTimer = null;

      if (!entry.host.isConnected) return;

      refreshScript(entry);
      entry.scriptTimer = setTimeout(tick, 400);
    };

    entry.scriptTimer = setTimeout(tick, 400);
  }

  /// The spectral dynamics curves: product state, not parameters, so the card
  /// asks its own node for them the way the product page asks the one product.
  function refreshSpectralDynamics(entry) {
    if (!entry || entry.kind.id !== "mod:spectral_dynamics") return;

    const hero = entry.card && entry.card.querySelector(".mcard-hero");

    // A curve under the hand is not replaced: the repaint a refetch causes takes
    // the picture the drag is drawn on.
    if (hero && hero.dataset.busy === "1") return;

    fetch("/spectral-dynamics?node=" + entry.node.id, { cache: "no-store" })
      .then((response) => (response.ok ? response.json() : null))
      .then((state) => {
        if (!state || !state.threshold) return;

        const toPoints = (pages) =>
          (pages || []).map((page) =>
            page.map((point) => ({ time: point[0], value: point[1], bend: point[2] || 0 })),
          );
        const next = { threshold: toPoints(state.threshold), ratio: toPoints(state.ratio) };

        if (JSON.stringify(entry.context.spectralDynamics) === JSON.stringify(next)) return;

        entry.context.spectralDynamics = next;
        if (entry.card.repaintHero) entry.card.repaintHero();
      })
      .catch(() => {});
  }

  /// The curves compile in the audio thread and the page's own edits come back
  /// from the server, so the card asks again on the product page's cadence.
  function watchSpectral(entry) {
    if (entry.spectralTimer) return;

    const tick = () => {
      entry.spectralTimer = null;

      if (!entry.host.isConnected) return;

      refreshSpectralDynamics(entry);
      entry.spectralTimer = setTimeout(tick, 1200);
    };

    entry.spectralTimer = setTimeout(tick, 1200);
  }

  /// A spectral dynamics hero edits this node's two curves.
  function wireSpectralDynamicsCard(entry) {
    if (!entry || entry.kind.id !== "mod:spectral_dynamics" || !entry.context.spectralDynamics) {
      return;
    }

    // The product has two heroes and the card shows one page at a time: only the
    // curve's page takes the point editor, or the spectrum picture would end up
    // with one.
    const page = (entry.kind.pages || [])[pagesOf[entry.node.id] ?? 0];

    if (page && page.hero !== "spectraldynamics-curve") return;

    const hero = entry.card && entry.card.querySelector(".mcard-hero");

    if (!hero) return;

    const svg = hero.querySelector("svg");

    if (!svg || svg.__sdWired) return;

    svg.__sdWired = true;
    watchHero(entry, wireSpectralDynamicsCard);

    IAmCard.wire(hero, "spectraldynamics-curve", entry.node.params, {
      get spectralDynamics() {
        return entry.context.spectralDynamics || { threshold: [], ratio: [] };
      },
      get sdCurve() {
        return entry.context.sdCurve || (entry.context.sdCurve = { view: "threshold" });
      },
      repaint: () => {
        if (entry.card.repaintHero) entry.card.repaintHero();
      },
      set: (id, value) => {
        entry.node.params[id] = value;
        postParam(entry.node.id, id, value);
      },
      post: (body) => {
        fetch("/spectral-dynamics?node=" + entry.node.id, { method: "POST", body })
          .then((response) => (response.ok ? response.json() : null))
          .catch(() => {});
      },
    });
  }

  /// The Filter Table frame, fetched the same way the product page does.
  function refreshFilterTable(entry) {
    if (!entry || entry.kind.id !== "mod:filter_table") return;

    fetch("/filter-table?node=" + entry.node.id, { cache: "no-store" })
      .then((response) => (response.ok ? response.json() : null))
      .then((state) => {
        if (!state || !state.pages) return;

        state.view = entry.context.filterTable && entry.context.filterTable.view;
        state.windowSamples = entry.context.filterTable && entry.context.filterTable.windowSamples;
        state.frozen = entry.context.filterTable && entry.context.filterTable.frozen;
        state.importMode = entry.context.filterTable && entry.context.filterTable.importMode;
        state.sliceMode = entry.context.filterTable && entry.context.filterTable.sliceMode;
        entry.context.filterTable = state;
        if (entry.card.repaintHero) entry.card.repaintHero();
      })
      .catch(() => {});
  }

  /// A Filter Table hero edits this node's frame.
  function wireFilterTableCard(entry) {
    if (!entry || entry.kind.id !== "mod:filter_table" || !entry.context.filterTable) return;
    const hero = entry.card && entry.card.querySelector(".mcard-hero");

    if (!hero) return;

    const svg = hero.querySelector("svg");

    if (!svg || svg.__filterTableWired) return;

    svg.__filterTableWired = true;
    watchHero(entry, wireFilterTableCard);

    IAmCard.wire(hero, "filtertable", entry.node.params, {
      filterTable: entry.context.filterTable,
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
        fetch("/filter-table?node=" + entry.node.id, { method: "POST", body })
          .then((response) => (response.ok ? response.json() : null))
          .catch(() => {});
      },
      import: (body, window, page, mode, slice) =>
        fetch(
          "/filter-table/import?node=" + entry.node.id + "&window=" + window + "&page=" + page + "&mode=" + mode + "&slice=" + slice,
          { method: "POST", body }
        ).then((response) => (response.ok ? response.json() : null)),
      importProgress: () =>
        fetch("/filter-table/import/progress?node=" + entry.node.id, { cache: "no-store" })
          .then((response) => (response.ok ? response.json() : null)),
      reload: () =>
        fetch("/filter-table?node=" + entry.node.id, { cache: "no-store" })
          .then((response) => (response.ok ? response.json() : null)),
      edit: (action, page) =>
        fetch(
          "/filter-table/edit?node=" + entry.node.id + "&action=" + action + "&page=" + page,
          { method: "POST" }
        ).then((response) => (response.ok ? response.json() : null)),
    });
  }

  /// The History pages are the product's state, so the card asks its node for
  /// them the way the product's own page does.
  function refreshHistory(entry) {
    if (!entry || entry.kind.id !== "mod:history") return;

    fetch("/history?node=" + entry.node.id, { cache: "no-store" })
      .then((response) => (response.ok ? response.json() : null))
      .then((state) => {
        if (!state || !state.pages) return;

        entry.context.history = state;
        if (entry.card.repaintHero) entry.card.repaintHero();
      })
      .catch(() => {});
  }

  /// A history hero edits this node's pages, not a page-wide history.
  function wireHistoryCard(entry) {
    if (!entry || entry.kind.id !== "mod:history" || !entry.context.history) return;
    const hero = entry.card && entry.card.querySelector(".mcard-hero");

    if (!hero) return;

    const svg = hero.querySelector("svg");

    if (!svg || svg.__historyWired) return;

    svg.__historyWired = true;
    watchHero(entry, wireHistoryCard);

    IAmCard.wire(hero, "history", entry.node.params, {
      history: entry.context.history,
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
        fetch("/history?node=" + entry.node.id, { method: "POST", body })
          .then((response) => (response.ok ? response.json() : null))
          .then((result) => {
            if (result && result.error) showStatus(result.error);
          })
          .catch(() => {});
      },
    });

    if (!entry.historyTimer) {
      entry.historyTimer = setInterval(() => {
        fetch("/history-phase?node=" + entry.node.id, { cache: "no-store" })
          .then((response) => (response.ok ? response.json() : null))
          .then((state) => {
            if (!state || typeof state.phase !== "number") return;

            entry.context.history.phase = state.phase;
            const line = entry.card.querySelector(".history-playhead");
            if (line) line.style.left = (state.phase * 100).toFixed(2) + "%";
          })
          .catch(() => {});
      }, 100);
    }
  }

  /// Fetches what a node's slot now holds and repaints the card's picture.
  async function refreshImport(entry, path) {
    if (!entry || !entry.card || !entry.context) return;

    const response = await fetch("/" + path + "?node=" + entry.node.id, {
      cache: "no-store",
    });

    if (!response.ok) return;

    const state = await response.json();
    const key = path === "ir" ? "ir" : path === "waves" ? "waves" : "sample";
    entry.context[key] = state;

    if (entry.card.repaintHero) entry.card.repaintHero();
  }

  // A product card is the product's own face: the shared builder draws its
  // sections, pages, heroes and controls, so the patch cannot drift from the
  // product page. The patch adds the modulation rings around the controls.
  function productCard(node, kind, w, h) {
    let entry = cards[node.id];
    const mods = modsOf(node.id);
    const pages = kind.pages || [];
    const activePage = pages.length
      ? Math.min(pagesOf[node.id] ?? 0, pages.length - 1)
      : 0;
    const key = JSON.stringify([
      mods.map((mod) => [mod.slot, mod.to[1], Math.round(mod.depth * 1000), mod.mode]),
      activePage,
      node.params,
    ]);

    if (entry && (entry.key !== key || entry.kindId !== node.kind)) {
      entry.host.remove();
      delete cards[node.id];
      entry = null;
    }

    if (entry) {
      entry.host.style.setProperty(
        "--k",
        String(Math.max(0.2, Math.min(w / baseWidth(node), h / baseHeight(node)))),
      );
      entry.host.classList.toggle("compact", w < 430);

      // The pages live outside the parameter map, so they can change without a
      // graph change: keep this card's copy current or its next edit would post
      // a stale snapshot back over them.
      if (isProductCard(kind, "spectrum")) refreshSpectrum(entry);

      return entry.host;
    }

    // The heroes that draw live data read it from here. The context outlives the
    // card: a rebuild happens on every graph change, and a fresh context would
    // flash the hero's fallback picture until the next poll filled the live one
    // back in.
    const context = contextOf(node);
    const importRoute = IMPORT_ROUTES[kind.id];

    const card = IAmCard.faceCard({
      face: kind,
      values: node.params,
      context,
      clear: !!kind.clear,
      onClear: () => {
        fetch("/clear?node=" + node.id, { method: "POST" }).catch(() => {});
      },
      script: !!kind.script,
      // The fitted instrument fits a recorded note; the sampler, the shapes and
      // the convolver load the file straight into the slot their voice reads.
      sample: kind.id === "synth:fitted" || !!importRoute,
      sampleHint: importRoute
        ? importRoute.hint
        : "Load a recorded note (.wav) to fit: click or drop a file",
      heroContext: heroContextFor(node, kind),
      pickSample: (file) => {
        if (kind.id === "synth:fitted") {
          if (file) postSample(node.id, file);
          else pickSampleFile((picked) => postSample(node.id, picked));
          return;
        }

        if (!importRoute) return;

        if (file) postImport(node.id, importRoute.path, file);
        else pickSampleFile((picked) => postImport(node.id, importRoute.path, picked));
      },
      loadScript: () => {
        fetch("/script?node=" + node.id, { cache: "no-store" })
          .then((response) => (response.ok ? response.json() : null))
          .then((state) => {
            if (!state) return;

            card.setScript(state.source || "");
            card.setScriptError(state.error || "");

            if (entry) {
              entry.context.script = state;
              if (card.repaintHero) card.repaintHero();
            }
          })
          .catch(() => {});
      },
      applyScript: (source) => {
        fetch("/script?node=" + node.id, { method: "POST", body: source })
          .then((response) => (response.ok ? response.json() : null))
          .then((state) => {
            if (!state) return;

            card.setScriptError(state.error || "");

            if (state.error) {
              if (entry) {
                entry.context.script = Object.assign({}, entry.context.script, { error: state.error, spectrum: [] });
                if (card.repaintHero) card.repaintHero();
              }
            } else if (entry) {
              refreshScript(entry);
            }
          })
          .catch(() => {});
      },
      activePage,
      onPage: (index) => {
        pagesOf[node.id] = index;
        delete cards[node.id];
        draw();
      },
      onEdit: (identifier, value) => {
        // The card already shows the value; rebuilding the graph here would
        // take the focus away from the control being dragged.
        touchNode(node.id);
        op({ op: "param", id: node.id, key: identifier, value }, false);

        if (kind.script && entry) {
          clearTimeout(entry.scriptTimer);
          entry.scriptTimer = setTimeout(() => refreshScript(entry), 150);
        }
      },
      wrapControl: (control) => {
        const assignments = mods
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
                index: liveModSlot(node.id, control.id, slot, sourceOf(assignments, slot)),
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
                index: liveModSlot(
                  node.id,
                  control.id,
                  slots[0],
                  sourceOf(assignments, slots[0]),
                ),
                mode: next === "center" ? "center" : "base",
              },
              false,
            );
          },
          // Ctrl + right click on the knob: every source comes off it.
          onClearModulations: (slots) => {
            const order = [...slots].sort((left, right) => right - left);

            order.forEach((slot, index) => {
              op({ op: "unmodulate", index: slot }, index === order.length - 1);
            });
          },
          dropTarget: node.id + ":" + control.id,
        });
      },
    });

    const host = document.createElement("div");
    host.className = "card-host";
    const scale = document.createElement("div");
    scale.className = "card-scale";
    scale.style.width = `${baseWidth(node)}px`;
    scale.style.height = `${baseHeight(node)}px`;
    scale.style.flexBasis = `${baseWidth(node)}px`;
    scale.appendChild(card);
    host.appendChild(scale);
    entry = { host, key, kindId: node.kind, context, card, node, kind, samples: null, phase: null };

    if (isProductCard(kind, "canvas")) {
      const repaint = card.repaintHero;

      card.repaintHero = function (values) {
        repaint(values);
        wireCanvasCard(entry);
      };

      refreshCanvas(entry);
      wireCanvasCard(entry);
      watchCanvas(entry);
    }

    cards[node.id] = entry;

    // A node that already holds a file shows it as soon as the card is built,
    // not only right after the drop that loaded it.
    if (IMPORT_ROUTES[kind.id]) refreshImport(entry, IMPORT_ROUTES[kind.id].path);
    // The two metering products: their hero keeps moving only while the card
    // asks the node for the bands the effect is writing.
    if (kind.hero === "ott" || kind.hero === "vocodex") watchBands(entry);
    if (isProductCard(kind, "spectrum")) {
      const repaint = card.repaintHero;

      card.repaintHero = function (values) {
        repaint(values);
        wireSpectrumCard(entry);
      };

      refreshSpectrum(entry);
    }

    if (kind.id === "synth:drummer") {
      entry.page = activePage;
      entry.pageName = ((kind.pages || [])[activePage] || {}).name || "";
      wireDrummerCard(entry);
      refreshDrummer(entry);
    }

    if (kind.script) {
      refreshScript(entry);
      watchScript(entry);
    }

    // The dynamics heroes keep their window moving from the poll.
    watchEnvelope(entry);

    if (kind.id === "mod:history") {
      const repaint = card.repaintHero;

      card.repaintHero = function (values) {
        repaint(values);
        wireHistoryCard(entry);
      };

      refreshHistory(entry);
      wireHistoryCard(entry);
    }

    if (kind.id === "mod:spectral_dynamics") {
      const repaint = card.repaintHero;

      card.repaintHero = function (values) {
        repaint(values);
        wireSpectralDynamicsCard(entry);
      };

      // The fetch only repaints when the curves moved, and a card that was just
      // built already shows them: the editor is attached here, not only from the
      // repaint, or a card rebuilt from cached pages would have no editor.
      refreshSpectralDynamics(entry);
      wireSpectralDynamicsCard(entry);
      watchSpectral(entry);
    }

    if (kind.id === "mod:filter_table") {
      const repaint = card.repaintHero;

      card.repaintHero = function (values) {
        repaint(values);
        wireFilterTableCard(entry);
      };

      refreshFilterTable(entry);
    }

    entry.host.style.setProperty(
      "--k",
      String(Math.max(0.2, Math.min(w / baseWidth(node), h / baseHeight(node)))),
    );
    entry.host.classList.toggle("compact", w < 430);
    return entry.host;
  }

