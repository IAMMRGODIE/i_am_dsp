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
        ir: irState, curve: curveState, fall: fallState, stereo: stereoScope, scope: scope,
        rate: (face && face.rate) || 48000, spectra: spectra,
        meter: meter, envHistory: envHistory, envWindow: envWindow,
        ott: ottState, vocoder: vocoderState, eqSpectrum: eqSpectrum,
        autotune: autotuneState, canvas: canvasState
      });
      if (content) {
        hero.el.innerHTML = content;
        if (IAmCard.wire) IAmCard.wire(hero.el, hero.kind, values, canvasState);
      }
    });
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
        if (typeof state.__bypass !== "undefined") setBypassed(state.__bypass);

        Object.keys(state).forEach(function (id) {
          if (id === "__bypass") return;

          var control = controls[id];
          if (control) {
            control.value = control.spec.kind === "bool"
              ? state[id]
              : snap(control.spec, state[id]);
            control.paint();
          }
        });
        applySwitches();
        applyEqType();
        markDirty();
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
