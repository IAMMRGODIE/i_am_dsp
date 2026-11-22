// Part five: the pictures and the level history behind them.
  function heroValues() {
    var values = {};

    Object.keys(controls).forEach(function (id) {
      values[id] = controls[id].value;
    });

    return values;
  }

  /// The hero kinds whose picture is drawn from the scope.
  var SCOPE_HEROES = [
    "spectrum", "spectra", "waveform", "unison", "transfer",
    "compressor", "gate", "expander", "limiter", "eq", "spectraldynamics-spectrum", "ring"
  ];

  /// Whether a picture that reads the scope is on screen.
  ///
  /// A scope frame is only worth repainting the whole face for when one of these
  /// is visible: a live hero on another tab would otherwise rebuild every
  /// control twenty times a second and swallow their clicks.
  function scopeHeroVisible() {
    var list = document.querySelectorAll(".hero[data-hero]");

    for (var i = 0; i < list.length; i++) {
      if (list[i].clientWidth > 0 && SCOPE_HEROES.indexOf(list[i].getAttribute("data-hero")) >= 0) {
        return true;
      }
    }

    return false;
  }

  /// Repaints every hero. `force` is for a change that is the button's own:
  /// the pointer is resting on it, and it is exactly the markup that has to be
  /// replaced, so the hover guard below must not keep the old label.
  /// Milliseconds since the last repaint, so a picture with ballistics runs
  /// on the wall clock rather than on the frame rate: a face that is drawing
  /// because a knob is moving must not make a peak decay faster.
  var lastPaint = 0;
  function paintMilliseconds() {
    var now = window.performance && window.performance.now
      ? window.performance.now()
      : Date.now();
    var ms = lastPaint ? Math.max(1, Math.min(200, now - lastPaint)) : 16;

    lastPaint = now;
    return ms;
  }

  function updateHeroes(force) {
    if (!window.IAmCard || !IAmCard.hero) return;

    var values = heroValues();

    heroes.forEach(function (hero) {
      // A canvas being dragged owns its own markup until the pointer is up.
      if (hero.el.dataset.busy === "1") return;
      // A pointer resting on the hero's buttons owns them too: replacing the
      // markup would drop the hover and the click that is about to land.
      // A change made while it rests there is not lost: the click that made
      // it asked for a forced paint, and leaving the buttons asks for one.
      if (!force && hero.el.dataset.hoverActions === "1") return;

      var content = IAmCard.hero(hero.kind, values, {
        active: activeControl,
        dt: paintMilliseconds(),
        ir: irState, curve: curveState, fall: fallState, stereo: stereoScope, scope: scope,
        rate: (face && face.rate) || 48000, spectra: spectra,
        meter: meter, envHistory: envHistory, envWindow: envWindow,
        ott: ottState, vocoder: vocoderState, eqSpectrum: eqSpectrum,
        autotune: autotuneState, canvas: canvasState, spectrum: spectrumState, history: historyState, filterTable: filterTableState, sidechain: filterTableSidechain,
        waves: wavesState, waveMode: waveMode, sample: sampleState, sampleView: sampleView, script: scriptState, click: clickState, samples: samplesState, ampCurves: ampCurves, kickMode: kickMode, snareMode: snareMode,
        spectralDynamics: spectralDynamicsState, sdCurve: spectralDynamicsCurve
      });
      if (content) {
        // A live hero elsewhere on the face repaints every scope frame; when
        // this hero's markup is the same, leaving the nodes alone is what keeps
        // its buttons and its editor under the pointer.
        if (hero.el.innerHTML !== content) hero.el.innerHTML = content;
        if (IAmCard.wire) {
          IAmCard.wire(hero.el, hero.kind, values, {
            canvas: canvasState, spectrum: spectrumState, history: historyState, filterTable: filterTableState, sidechain: filterTableSidechain,
            spectralDynamics: spectralDynamicsState, sdCurve: spectralDynamicsCurve,
            repaint: markDirty,
            set: setControl,
            post: function (body) {
              // Each hero edits its own product state, and each product answers
              // on its own route: sending the spectrum's pages to /history both
              // dropped the edit and left the sound on the old tables.
              var kind = hero.kind || "";
              var path = kind.indexOf("filter") !== -1 ? "/filter-table"
                : kind.indexOf("spectrum") !== -1 ? "/spectrum"
                : kind.indexOf("canvas") !== -1 ? "/canvas"
                : "/history";
              fetch(path, { method: "POST", body: body }).catch(function () {});
            },
            import: function (body, window, page, mode, slice) {
              return fetch("/filter-table/import?window=" + window + "&page=" + page + "&mode=" + mode + "&slice=" + slice, {
                method: "POST",
                body: body
              }).then(function (response) {
                return response.ok ? response.json() : null;
              });
            },
            importProgress: function () {
              return fetch("/filter-table/import/progress", { cache: "no-store" })
                .then(function (response) { return response.ok ? response.json() : null; });
            },
            reload: function () {
              return fetch("/filter-table", { cache: "no-store" })
                .then(function (response) { return response.ok ? response.json() : null; });
            },
            edit: function (action, page) {
              return fetch("/filter-table/edit?action=" + action + "&page=" + page, {
                method: "POST"
              }).then(function (response) {
                return response.ok ? response.json() : null;
              });
            }
          });
        }
      }
    });
  }

  /// Lets an editor hero move a control it is coupled to, like the morph that
  /// decides which page is being drawn.
  function setControl(id, value) {
    var control = controls[id];

    // Nothing moved, nothing to repaint: an editor that selects the page the
    // knob already sits on must not rebuild the hero under the pointer.
    if (!control || control.value === value) return;

    setValue(control, value, true);
  }

  /// Remembers the control being held (or focused), so a hero can draw the part
  /// of itself that control edits.
  function setActiveControl(id) {
    if (activeControl === id) return;

    activeControl = id;
    markDirty();
  }

  function markDirty(force) {
    forced = forced || !!force;

    if (dirty) return;
    dirty = true;
    requestAnimationFrame(function () {
      var now = forced;

      dirty = false;
      forced = false;
      updateHeroes(now);
    });
  }

  // -------------------------------------------------------------------- poll
  function poll() {
    if (polling) return;
    polling = true;

    if (document.hidden) {
      polling = false;
      setTimeout(poll, 250);
      return;
    }

    fetch("/state", { cache: "no-store" })
      .then(function (response) {
        if (!response.ok) throw new Error("state " + response.status);
        return response.json();
      })
      .then(function (state) {
        var changed = false;

        if (typeof state.__bypass !== "undefined" && state.__bypass !== bypassed) {
          setBypassed(state.__bypass);
          changed = true;
        }

        Object.keys(state).forEach(function (id) {
          if (id === "__bypass") return;

          var control = controls[id];
          if (!control) return;

          var next = control.spec.kind === "bool"
            ? state[id]
            : snap(control.spec, state[id]);

          if (control.value === next) return;

          control.value = next;
          control.paint();
          changed = true;
        });

        // A repaint rebuilds every hero, which replaces the node under a press:
        // only do it when a value really moved.
        if (changed) {
          applySwitches();
          applyEqType();
          markDirty();
        }
      })
      .catch(function (error) {
        console.warn("i_am_web_gui: " + error);
      })
      .then(function () {
        polling = false;
        setTimeout(poll, 60);
      });
  }

  function levelOf(samples, start, count) {
    var energy = 0;

    for (var i = 0; i < count; i++) {
      var sample = samples[start + i];
      energy += sample * sample;
    }

    return Math.sqrt(energy / Math.max(count, 1));
  }

  // Fill the history from the scope the first time, so the envelope has a
  // complete picture before the first few live frames arrive.
  function seedEnvelope(input, output) {
    if (envHistory.input.length) return;

    var segments = 128;
    var step = Math.floor(input.length / segments);

    if (step < 1) step = 1;

    for (var s = 0; s < segments; s++) {
      var start = Math.min(s * step, input.length - step);
      envHistory.input.push(levelOf(input, start, step));
      envHistory.output.push(levelOf(output, start, step));
    }
  }

  function pushEnvelope(input, output) {
    var step = Math.min(1024, input.length);
    var start = input.length - step;

    if (envHistory.input.length >= ENV_MAX) {
      envHistory.input.shift();
      envHistory.output.shift();
    }

    envHistory.input.push(levelOf(input, start, step));
    envHistory.output.push(levelOf(output, start, step));
  }

  // The scope is its own poll: it is a binary document, and a host with no audio
  // running still wants the rest of the page to keep working.
