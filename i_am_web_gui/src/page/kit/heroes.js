// Part five: the pictures and the level history behind them.
  function heroValues() {
    var values = {};

    Object.keys(controls).forEach(function (id) {
      values[id] = controls[id].value;
    });

    return values;
  }

  function updateHeroes() {
    if (!window.IAmCard || !IAmCard.hero) return;

    var values = heroValues();

    heroes.forEach(function (hero) {
      // A canvas being dragged owns its own markup until the pointer is up.
      if (hero.el.dataset.busy === "1") return;

      var content = IAmCard.hero(hero.kind, values, {
        active: activeControl,
        ir: irState, curve: curveState, fall: fallState, stereo: stereoScope, scope: scope,
        rate: (face && face.rate) || 48000, spectra: spectra,
        meter: meter, envHistory: envHistory, envWindow: envWindow,
        ott: ottState, vocoder: vocoderState, eqSpectrum: eqSpectrum,
        autotune: autotuneState, canvas: canvasState, spectrum: spectrumState, history: historyState, filterTable: filterTableState,
        waves: wavesState, waveMode: waveMode, sample: sampleState, sampleView: sampleView, script: scriptState, click: clickState, samples: samplesState, ampCurves: ampCurves, kickMode: kickMode, snareMode: snareMode
      });
      if (content) {
        hero.el.innerHTML = content;
        if (IAmCard.wire) {
          IAmCard.wire(hero.el, hero.kind, values, {
            canvas: canvasState, spectrum: spectrumState, history: historyState, filterTable: filterTableState,
            set: setControl,
            post: function (body) {
              var path = hero.kind === "filtertable" ? "/filter-table" : "/history";
              fetch(path, { method: "POST", body: body }).catch(function () {});
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

  function markDirty() {
    if (dirty) return;
    dirty = true;
    requestAnimationFrame(function () {
      dirty = false;
      updateHeroes();
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
