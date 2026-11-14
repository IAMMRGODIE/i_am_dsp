// Part ten: the polling that keeps the pictures live.
  /// The level of a scope ring in dBFS, which is the unit the dynamics heroes
  /// mark their traces in.
  function levelOf(samples) {
    if (!samples || !samples.length) return null;

    var energy = 0;
    for (var i = 0; i < samples.length; i++) energy += samples[i] * samples[i];

    return 20 * Math.log10(Math.max(Math.sqrt(energy / samples.length), 1e-6));
  }

  function repaint(entry) {
    if (entry.card.repaintHero) entry.card.repaintHero();
  }

  /// One card's own input and output: the document is the same three rings the
  /// module scope keeps, so a card never shows another module's audio.
  function fetchScope(entry) {
    entry.fetching = true;

    fetch("/module-scope?id=" + entry.id, { cache: "no-store" })
      .then(function (response) {
        return response.ok ? response.arrayBuffer() : null;
      })
      .then(function (buffer) {
        if (!buffer) return;

        var data = new Float32Array(buffer);
        var third = data.length / 3;
        if (third <= 64) return;

        entry.context.scope.input = data.subarray(0, third);
        entry.context.scope.output = data.subarray(third, third * 2);
        entry.context.scope.sidechain = data.subarray(third * 2);
        entry.context.meter.input = levelOf(entry.context.scope.input);
        entry.context.meter.sidechain = levelOf(entry.context.scope.sidechain);
        pushEnvelope(entry.context);
        entry.context.rate = pageRate;
        entry.context.eqSpectrum.rate = pageRate;
        repaint(entry);
      })
      .catch(function () {})
      .then(function () {
        entry.fetching = false;
      });
  }

  /// The goniometer wants both channels before and after the module; the mono
  /// document cannot tell it anything about the stereo image.
  function fetchStereo(entry) {
    fetch("/module-stereo?id=" + entry.id, { cache: "no-store" })
      .then(function (response) {
        return response.ok ? response.arrayBuffer() : null;
      })
      .then(function (buffer) {
        if (!buffer) return;

        var data = new Float32Array(buffer);
        var points = data.length / 4;
        if (points <= 64) return;

        entry.context.stereo.inputLeft = data.subarray(0, points);
        entry.context.stereo.inputRight = data.subarray(points, points * 2);
        entry.context.stereo.outputLeft = data.subarray(points * 2, points * 3);
        entry.context.stereo.outputRight = data.subarray(points * 3);
        repaint(entry);
      })
      .catch(function () {});
  }

  /// Asks the person for an impulse response file and posts its bytes to the
  /// module. The page reads the file itself, so the rack never opens a native
  /// dialog and the host window cannot end up behind the DAW.
  function pickImpulse(entry) {
    var input = document.createElement("input");
    input.type = "file";
    input.accept = ".wav,.aif,.aiff,.flac,audio/*";

    input.addEventListener("change", function () {
      var file = input.files && input.files[0];
      if (!file) return;

      file
        .arrayBuffer()
        .then(function (bytes) {
          return fetch(
            "/module-ir?id=" + entry.id + "&name=" + encodeURIComponent(file.name),
            { method: "POST", body: bytes }
          );
        })
        .then(function () { return fetchIr(entry); })
        .catch(function () {});
    });

    input.click();
  }

  /// The impulse response a convolver module holds, with its picture.
  function fetchIr(entry) {
    fetch("/module-ir?id=" + entry.id, { cache: "no-store" })
      .then(function (response) {
        return response.ok ? response.json() : null;
      })
      .then(function (value) {
        if (!value) return;

        var changed = JSON.stringify(entry.context.ir) !== JSON.stringify(value);
        entry.context.ir = value;

        if (changed) repaint(entry);
      })
      .catch(function () {});
  }

  /// The History pages a module holds, with their page count and morph.
  function fetchHistory(entry) {
    fetch("/module-history?id=" + entry.id, { cache: "no-store" })
      .then(function (response) {
        return response.ok ? response.json() : null;
      })
      .then(function (value) {
        if (!value || !value.pages) return;

        var changed = JSON.stringify(entry.context.history) !== JSON.stringify(value);
        entry.context.history = value;

        if (changed) repaint(entry);
      })
      .catch(function () {});
  }

  /// The source a scripted module runs, for its editor. The poll only refreshes
  /// the error line, so it cannot overwrite what a person is typing.
  function loadScript(entry, force) {
    fetch("/module-script?id=" + entry.id, { cache: "no-store" })
      .then(function (response) {
        return response.ok ? response.json() : null;
      })
      .then(function (value) {
        if (!value) return;

        entry.context.script = value;
        entry.card.setScript(value.source, !!force);

        if (entry.card.setScriptError) entry.card.setScriptError(value.error);
      })
      .catch(function () {});
  }

  /// Posts a new source; the worker renders it on its own thread.
  function applyScript(entry, source) {
    fetch("/module-script?id=" + entry.id, {
      method: "POST",
      body: source
    })
      .then(function () { loadScript(entry); })
      .catch(function () {});
  }

  /// The curve a script renders right now, which the card's picture draws.
  function fetchCurve(entry) {
    fetch("/module-curve?id=" + entry.id, { cache: "no-store" })
      .then(function (response) {
        return response.ok ? response.json() : null;
      })
      .then(function (value) {
        if (!value || !value.points) return;

        var changed = JSON.stringify(entry.context.curve) !== JSON.stringify(value);
        entry.context.curve = value;
        entry.context.mapping = value;

        if (changed) repaint(entry);
      })
      .catch(function () {});
  }

  function pollImpulses() {
    if (!document.hidden) {
      liveCards.forEach(function (entry) {
        if (entry.ir) fetchIr(entry);
        if (entry.script) {
          fetchCurve(entry);
          loadScript(entry, false);
        }
      });
    }

    irTimer = setTimeout(pollImpulses, 250);
  }

  /// The band meter a module draws: the OTT's three gain bars, or the vocoder's
  /// input and output level per band.
  function fetchBands(entry) {
    fetch("/module-bands?id=" + entry.id, { cache: "no-store" })
      .then(function (response) {
        return response.ok ? response.json() : null;
      })
      .then(function (value) {
        if (!value) return;

        var changed;

        if (value.gains) {
          changed = JSON.stringify(entry.context.ott) !== JSON.stringify({ bands: value.input });
          entry.context.ott = { bands: value.input };
        } else {
          var next = { bands: value.bands, input: value.input, output: value.output };
          changed = JSON.stringify(entry.context.vocoder) !== JSON.stringify(next);
          entry.context.vocoder = next;
        }

        if (changed) repaint(entry);
      })
      .catch(function () {});
  }

  /// The RMS of a block of samples, in the 0..1 the heroes plot.
  function levelOf(samples) {
    if (!samples || !samples.length) return 0;

    var energy = 0;
    for (var i = 0; i < samples.length; i++) energy += samples[i] * samples[i];

    return Math.sqrt(energy / samples.length);
  }

  /// Fills a card's envelope history: the first scope frame seeds a whole window
  /// so the picture starts complete, and every later frame appends one step.
  function pushEnvelope(context) {
    var scope = context.scope;
    var history = context.envHistory;

    if (!scope || !scope.input || !scope.output || !history) return;

    if (!history.input.length) {
      var segments = 128;
      var step = Math.max(1, Math.floor(scope.input.length / segments));

      for (var s = 0; s < segments; s++) {
        var start = Math.min(s * step, scope.input.length - step);
        history.input.push(levelOf(scope.input.subarray(start, start + step)));
        history.output.push(levelOf(scope.output.subarray(start, start + step)));
      }

      context.envWindow = segments;
      return;
    }

    var size = Math.min(1024, scope.input.length);
    var begin = scope.input.length - size;

    if (history.input.length >= 512) {
      history.input.shift();
      history.output.shift();
    }

    history.input.push(levelOf(scope.input.subarray(begin, begin + size)));
    history.output.push(levelOf(scope.output.subarray(begin, begin + size)));
  }

  /// The note a pitch-tracking module is hearing, and the mask it tunes to: the
  /// card shows the note in its header, the autotune hero lights the mask.
  function fetchNote(entry) {
    fetch("/module-note?id=" + entry.id, { cache: "no-store" })
      .then(function (response) {
        return response.ok ? response.json() : null;
      })
      .then(function (value) {
        if (!value) return;

        if (entry.card.setNote) {
          // A chord shows every note it is playing, not just the last one.
          var held = value.notes && value.notes.length
            ? value.notes
            : (value.note >= 0 ? [value.note] : []);

          entry.card.setNote(held);
        }

        if (typeof value.allowed === "number") {
          var changed =
            !entry.context.autotune || entry.context.autotune.notes !== value.allowed;
          entry.context.autotune = { notes: value.allowed };
          if (changed) repaint(entry);
        }
      })
      .catch(function () {});
  }

  /// Only the cards on screen and drawing audio are polled, one document each,
  /// so a rack of twenty-seven modules costs what the visible few cost.
  function pollCards() {
    if (!document.hidden) {
      liveCards.forEach(function (entry) {
        if (entry.fetching) return;
        if (!entry.card.isConnected || entry.card.offsetParent === null) return;

        // The audio polls are for the cards that draw audio; the note poll is
        // for the cards that follow the host's notes, and they do not overlap:
        // an autotune has nothing to draw but its mask.
        if (entry.channels) {
          fetchScope(entry);
          if (entry.channels === "stereo") fetchStereo(entry);
        }

        if (entry.note) fetchNote(entry);
        if (entry.meter) fetchBands(entry);
      });
    }

    cardTimer = setTimeout(pollCards, 60);
  }

  function startLive() {
    if (liveTimer) clearInterval(liveTimer);

    liveTimer = setInterval(function () {
      fetch("/modulation", { cache: "no-store" })
        .then(function (response) {
          if (!response.ok) throw new Error("modulation " + response.status);
          return response.json();
        })
        .then(function (data) {
          (data.sources || []).forEach(function (source) {
            var id = String(source.id);
            live.values[id] = source.value;
            live.phases[id] = source.phase;

            if (!live.history[id]) live.history[id] = [];
            live.history[id].push(source.value);
            if (live.history[id].length > 160) live.history[id].shift();
          });

          applyHostValues(data);

          liveCards.forEach(function (entry) {
            if (!entry.card || !entry.card.isConnected) return;

            var values = liveModulation(entry);
            if (values && entry.card.setLiveValues) entry.card.setLiveValues(values);
          });

          livePainters.forEach(function (paint) { paint(); });
        })
        .catch(function () {
          // The live route is absent in a static preview; the static heroes stay.
        });
    }, 50);
  }

  /// Pull the host's macro and automation values into what the page shows. A
  /// pointer that is down owns the control it grabbed, so a drag is never
  /// fought by a value arriving from the host.
  function applyHostValues(data) {
    if (pointerDown) return;

    (data.macros || []).forEach(function (value, index) {
      var handle = macroHandles[index];

      if (!handle || typeof value !== "number") return;
      if (Math.abs((macroValues[index] || 0) - value) < 0.0005) return;

      macroValues[index] = value;
      handle.set(value);
    });

    (data.automation || []).forEach(function (value, index) {
      var readout = automationSpans[index];

      if (!readout || typeof value !== "number") return;

      var text = Math.round(value * 100) + "%";
      if (readout.textContent !== text) readout.textContent = text;
    });
  }
