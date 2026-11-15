// Part six: the polling that keeps the scope, the panels and the meters live.
  function pollScope() {
    if (document.hidden || !wantsScope) {
      setTimeout(pollScope, 150);
      return;
    }

    fetch("/scope", { cache: "no-store" })
      .then(function (response) {
        if (!response.ok) throw new Error("scope " + response.status);
        return response.arrayBuffer();
      })
      .then(function (buffer) {
        var data = new Float32Array(buffer);
        var third = data.length / 3;

        if (third > 64) {
          scope.input = data.subarray(0, third);
          scope.output = data.subarray(third, third * 2);
          scope.sidechain = data.subarray(third * 2);
          seedEnvelope(scope.input, scope.output);
          pushEnvelope(scope.input, scope.output);
          markDirty();
        }
      })
      .catch(function () {})
      .then(function () {
        setTimeout(pollScope, 50);
      });
  }

  /// The note the host last sent, shown beside the title. A product without
  /// The autotune publishes the mask it is actually using: MIDI held notes win
  /// over the faceplate mask. The preview server has no /allowed and the hero
  /// falls back to the note control.
  function pollAllowed() {
    if (!wantsAutotune) return;
    if (isPreview || document.hidden) {
      setTimeout(pollAllowed, 100);
      return;
    }

    fetch("/allowed", { cache: "no-store" })
      .then(function (response) {
        return response.ok ? response.json() : null;
      })
      .then(function (data) {
        if (data && typeof data.notes === "number") {
          autotuneState.notes = data.notes;
          updateHeroes();
        }
      })
      .catch(function () {
        autotuneState.notes = -1;
      })
      .then(function () {
        setTimeout(pollAllowed, 100);
      });
  }

  /// note input answers with -1 and the badge stays hidden.
  function pollNote() {
    if (document.hidden || !noteBadge) {
      setTimeout(pollNote, 250);
      return;
    }

    fetch("/note", { cache: "no-store" })
      .then(function (response) {
        return response.ok ? response.json() : null;
      })
      .then(function (data) {
        if (data && data.note >= 0) {
          var names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"];
          var index = ((Math.round(data.note) % 12) + 12) % 12;
          var octave = Math.floor(Math.round(data.note) / 12) - 1;
          noteBadge.textContent = names[index] + octave;
          noteBadge.style.display = "";
        } else {
          noteBadge.style.display = "none";
        }
      })
      .catch(function () {
        noteBadge.style.display = "none";
      })
      .then(function () {
        setTimeout(pollNote, 100);
      });
  }

  /// The loaded impulse response, with its picture. The product owns /ir; the
  /// preview server does not, so a failed fetch just leaves the empty hero.
  function pollIr() {
    if (document.hidden || !wantsIr || irDragging) {
      setTimeout(pollIr, 500);
      return;
    }

    fetch("/ir", { cache: "no-store" })
      .then(function (response) {
        return response.ok ? response.json() : null;
      })
      .then(function (value) {
        if (!value) return;
        var changed = JSON.stringify(value) !== JSON.stringify(irState);
        irState = value;
        if (changed) markDirty();
      })
      .catch(function () {})
      .then(function () {
        setTimeout(pollIr, 500);
      });
  }

  /// Hands an audio file to the product's upload route as it is. The Rust side
  /// decodes and resamples it, so the page never has to parse an audio format.
  function postIr(file) {
    if (!file) return;

    var reader = new FileReader();
    reader.onload = function () {
      fetch("/ir?name=" + encodeURIComponent(file.name), {
        method: "POST",
        body: reader.result
      })
        .then(function (response) {
          return response.ok ? response.json() : null;
        })
        .then(function (result) {
          if (result && result.error) {
            irState = { name: null, rate: 0, samples: 0, peaks: [], error: result.error };
            markDirty();
          }
        })
        .catch(function () {});
    };
    reader.readAsArrayBuffer(file);
  }

  /// The picker's change handler: the same upload, and a cleared input so the
  /// same file can be chosen twice.
  function uploadIr() {
    postIr(irInput && irInput.files && irInput.files[0]);

    if (irInput) irInput.value = "";
  }

  /// What the shapes face has imported: the route decodes the WAV into frames,
  /// so the page asks it for the frames instead of parsing the file itself.
  function pollWaves() {
    if (document.hidden || !wantsWaves) {
      setTimeout(pollWaves, 1000);
      return;
    }

    var since = wavesState && wavesState.version ? "?since=" + wavesState.version : "";

    fetch("/waves" + since, { cache: "no-store" })
      .then(function (response) {
        return response.ok ? response.json() : null;
      })
      .then(function (value) {
        if (!value || value.unchanged) return;
        var changed = JSON.stringify(value) !== JSON.stringify(wavesState);
        wavesState = value;
        if (changed) markDirty();
      })
      .catch(function () {})
      .then(function () {
        setTimeout(pollWaves, 1000);
      });
  }

  /// Hands a dropped or picked WAV to the table route.
  function postWave(file) {
    if (!file) return;

    var reader = new FileReader();
    reader.onload = function () {
      fetch("/waves?name=" + encodeURIComponent(file.name) + "&mode=" + waveMode, {
        method: "POST",
        body: reader.result
      })
        .then(function (response) {
          return response.ok ? response.json() : null;
        })
        .then(function (result) {
          if (result && result.error) {
            wavesState = { name: null, count: 0, previews: [], error: result.error };
            markDirty();
          } else if (result && result.version) {
            wavesState = result;
            markDirty();
          }
        })
        .catch(function () {});
    };
    reader.readAsArrayBuffer(file);
  }

  /// What the sampler face has imported: the route decodes the WAV and returns
  /// the picture, so the page never parses the file itself.
  function pollSample() {
    if (document.hidden || !wantsSample || sampleDragging) {
      setTimeout(pollSample, 1000);
      return;
    }

    fetch("/sample", { cache: "no-store" })
      .then(function (response) {
        return response.ok ? response.json() : null;
      })
      .then(function (value) {
        if (!value) return;
        var changed = JSON.stringify(value) !== JSON.stringify(sampleState);
        sampleState = value;
        if (changed) markDirty();
      })
      .catch(function () {})
      .then(function () {
        setTimeout(pollSample, 1000);
      });
  }

  /// Hands a dropped or picked audio file to the sample route.
  function postSample(file) {
    if (!file) return;

    var reader = new FileReader();
    reader.onload = function () {
      fetch("/sample?name=" + encodeURIComponent(file.name), {
        method: "POST",
        body: reader.result
      })
        .then(function (response) {
          return response.ok ? response.json() : null;
        })
        .then(function (result) {
          if (result && result.error) {
            sampleState = { name: null, peaks: [], error: result.error };
            markDirty();
          } else if (result && result.version) {
            sampleState = result;
            sampleView.from = 0;
            sampleView.to = 1;
            markDirty();
          }
        })
        .catch(function () {});
    };
    reader.readAsArrayBuffer(file);
  }

  /// Moves one slice mark, once a drag on it ends.
  /// Asks the route for a fresh set of slice marks, from the grid settings or
  /// from the transients.
  function generateSlices() {
    var mode = controls["oscillator.slice_mode"] ? Math.round(controls["oscillator.slice_mode"].value) : 0;
    var start = controls["oscillator.slice_start"] ? controls["oscillator.slice_start"].value : 0;
    var step = controls["oscillator.slice_step"] ? controls["oscillator.slice_step"].value : 0.125;

    fetch("/sample?action=slices&mode=" + (mode === 1 ? "transient" : "grid") +
      "&start=" + start + "&step=" + step, { method: "POST", body: "slices" })
      .then(function (response) {
        return response.ok ? response.json() : null;
      })
      .then(function (result) {
        if (result && result.version) {
          sampleState = result;
          markDirty();
        }
      })
      .catch(function () {});
  }

  /// Adds one mark where the picture was double-clicked.
  /// Removes the mark under a right-click.
  /// Sets where the sample starts playing, from a drag on the picture.
  /// Sets the loop region from a right-drag; the ends are ordered here.
  /// Flips one of the sampler hero's switches.
  /// The picker's change handler.
  function uploadSample() {
    postSample(sampleInput && sampleInput.files && sampleInput.files[0]);

    if (sampleInput) sampleInput.value = "";
  }

  /// Runs one of the shape hero's buttons.
  /// The table's frame count and the frame the morph sits on.
  function waveCount() {
    return wavesState && wavesState.previews ? wavesState.previews.length : 0;
  }

  function waveFrame() {
    var count = waveCount();
    var control = controls["oscillator.smooth_factor"];
    var morph = control ? Math.min(Math.max(control.value, 0), 1) : 0;

    return IAmPage.index(morph, count);
  }

  /// Moves the morph to a whole frame, which is how a frame is picked.
  function setWaveFrame(index) {
    var count = waveCount();

    if (count <= 1) {
      setControl("oscillator.smooth_factor", 0);
      return;
    }

    var frame = Math.min(Math.max(index, 0), count - 1);

    setControl("oscillator.smooth_factor", frame / (count - 1));
  }

  function stepWaveFrame(step) {
    setWaveFrame(waveFrame() + step);
  }

  /// Removes the frame the morph sits on, then parks the morph on the neighbour.
  function deleteWaveFrame() {
    var count = waveCount();

    if (count <= 1) return;

    var frame = waveFrame();

    fetch("/waves?action=delete&frame=" + frame, { method: "POST", body: "delete" })
      .then(function (response) {
        return response.ok ? response.json() : null;
      })
      .then(function (result) {
        if (!result || !result.version) return;
        wavesState = result;
        setWaveFrame(Math.min(frame, waveCount() - 1));
        markDirty();
      })
      .catch(function () {});
  }

  /// Cycles how the next import cuts the file.
  function cycleWaveMode() {
    var modes = ["auto", "fixed", "pitch", "single"];

    waveMode = modes[(modes.indexOf(waveMode) + 1) % modes.length];
    markDirty();
  }

  /// Flips the imported table; the route rebuilds it and describes it back.
  /// The picker's change handler.
  function uploadWave() {
    postWave(waveInput && waveInput.files && waveInput.files[0]);

    if (waveInput) waveInput.value = "";
  }

  /// Hands a picked table or sample to the product's route. A `.bin` is a
  /// table the fitter already wrote; a `.wav` is one recorded note, which the
  /// product fits itself before it stores it.
  function postTableFile(file) {
    if (!file) return;

    var sample = /\.wav$/i.test(file.name);
    var reader = new FileReader();
    reader.onload = function () {
      fetch(sample ? "/sample?name=" + encodeURIComponent(file.name) : "/table", {
        method: "POST",
        body: reader.result
      })
        .then(function (response) {
          return response.ok ? response.json() : null;
        })
        .then(function (result) {
          if (result && result.ok) markDirty();
        })
        .catch(function () {});
    };
    reader.readAsArrayBuffer(file);
  }

  /// The picker's change handler: the same upload, and a cleared input so the
  /// same file can be chosen twice.
  function uploadTable() {
    postTableFile(tableInput && tableInput.files && tableInput.files[0]);

    if (tableInput) tableInput.value = "";
  }

  /// The vocoder bank: the modulator bands above the centre line and the output
  /// bands below it, one pair per band.
  function pollVocoder() {
    if (!wantsVocoder || document.hidden) {
      setTimeout(pollVocoder, 60);
      return;
    }

    fetch("/bands", { cache: "no-store" })
      .then(function (response) {
        return response.ok ? response.json() : null;
      })
      .then(function (state) {
        if (!state) return;
        var changed = JSON.stringify(state) !== JSON.stringify(vocoderState);
        vocoderState = state;
        if (changed) markDirty();
      })
      .catch(function () {})
      .then(function () {
        setTimeout(pollVocoder, 60);
      });
  }

  /// The canvas product's drawn curves. It is not a parameter, so the page has
  /// to ask the product for it.
  function pollCanvas() {
    fetch("/canvas", { cache: "no-store" })
      .then(function (response) {
        return response.ok ? response.json() : null;
      })
      .then(function (state) {
        if (!state || !state.shape) return;
        if (document.querySelector(".hero[data-busy='1']")) return;
        if (canvasState && canvasState.shape === state.shape) return;

        canvasState = state;
        markDirty();
      })
      .catch(function () {})
      .then(function () {
        setTimeout(pollCanvas, 1500);
      });
  }

  /// The Spectrum product's drawn pages live outside the parameter map, so the
  /// page asks the product for them, and posts new ones back on an edit.
  function pollSpectrum() {
    fetch("/spectrum", { cache: "no-store" })
      .then(function (response) {
        return response.ok ? response.json() : null;
      })
      .then(function (state) {
        if (!state || !state.frames) return;
        if (document.querySelector(".hero[data-busy='1']")) return;

        if (!spectrumState) spectrumState = { generation: 0, page: 0, frames: null };

        spectrumState.generation = state.generation;
        spectrumState.frames = state.frames;
        spectrumState.page = Math.min(spectrumState.page || 0, state.frames.length - 1);
        markDirty();
      })
      .catch(function () {})
      .then(function () {
        setTimeout(pollSpectrum, 1500);
      });
  }

  /// The History product's pages, like the Spectrum's: its own route, not a
  /// parameter.
  function pollHistory() {
    if (!document.querySelector(".hero[data-hero='history']")) {
      setTimeout(pollHistory, 500);
      return;
    }

    fetch("/history", { cache: "no-store" })
      .then(function (response) {
        return response.ok ? response.json() : null;
      })
      .then(function (state) {
        if (!state || !state.pages) return;

        var changed = !historyState ||
          JSON.stringify(historyState.pages) !== JSON.stringify(state.pages);
        historyState = state;

        var line = document.querySelector(".hero .history-playhead");
        if (line && typeof state.phase === "number") {
          line.style.left = (state.phase * 100).toFixed(2) + "%";
        }

        if (changed && !document.querySelector(".hero[data-busy='1']")) markDirty();
      })
      .catch(function () {})
      .then(function () {
        setTimeout(pollHistory, 100);
      });
  }

  /// The playhead is one number; poll it quickly without re-reading all pages.
  function pollHistoryPhase() {
    if (!document.querySelector(".hero[data-hero='history']")) {
      setTimeout(pollHistoryPhase, 250);
      return;
    }

    fetch("/history-phase", { cache: "no-store" })
      .then(function (response) {
        return response.ok ? response.json() : null;
      })
      .then(function (state) {
        if (!state || typeof state.phase !== "number") return;
        var line = document.querySelector(".hero .history-playhead");
        if (line) line.style.left = (state.phase * 100).toFixed(2) + "%";
      })
      .catch(function () {})
      .then(function () {
        setTimeout(pollHistoryPhase, 50);
      });
  }

  /// The Filter Table frame lives outside the parameter map, like the canvas.
  function pollFilterTable() {
    fetch("/filter-table", { cache: "no-store" })
      .then(function (response) {
        return response.ok ? response.json() : null;
      })
      .then(function (state) {
        if (!state || !state.pages) return;
        if (document.querySelector(".hero[data-busy='1']")) return;
        state.view = filterTableState && filterTableState.view;
        state.windowMs = filterTableState && filterTableState.windowMs;
        filterTableState = state;
        markDirty();
      })
      .catch(function () {})
      .then(function () {
        setTimeout(pollFilterTable, 1500);
      });
  }

  /// The OTT band meters, in dB of net gain: positive is attenuation, negative
  /// is the upward compressor's boost.
  function pollOtt() {
    if (!wantsOtt || document.hidden) {
      setTimeout(pollOtt, 60);
      return;
    }

    fetch("/bands", { cache: "no-store" })
      .then(function (response) {
        return response.ok ? response.json() : null;
      })
      .then(function (state) {
        if (!state) return;
        var changed = JSON.stringify(state) !== JSON.stringify(ottState);
        ottState = state;
        if (changed) markDirty();
      })
      .catch(function () {})
      .then(function () {
        setTimeout(pollOtt, 60);
      });
  }

  /// The script panel's text and status. The worker owns the script; the page
  /// only shows what it reports.
  function pollScript() {
    if (!wantsScript || document.hidden) {
      setTimeout(pollScript, 500);
      return;
    }

    var epoch = scriptEpoch;

    fetch("/script", { cache: "no-store" })
      .then(function (response) {
        return response.ok ? response.json() : null;
      })
      .then(function (state) {
        if (!state) return;

        // A poll that started before an edit or an apply is stale: writing its
        // source back would undo what the person just typed. The next poll reads
        // the applied source.
        if (epoch !== scriptEpoch) return;

        scriptState = state;

        var fingerprint = String(state.applied) + "|" + (state.error || "") + "|" +
          JSON.stringify(state.spectrum || []);

        if (fingerprint !== scriptFingerprint) {
          scriptFingerprint = fingerprint;
          markDirty();
        }

        if (scriptArea && !scriptDirty && document.activeElement !== scriptArea) {
          scriptArea.value = state.source || "";
        }

        if (scriptError) scriptError.textContent = state.error || "";

        if (scriptStatus) {
          scriptStatus.textContent = state.error
            ? "error"
            : "applied " + (state.applied || 0);
        }
      })
      .catch(function () {})
      .then(function () {
        setTimeout(pollScript, 500);
      });
  }

  function applyScript() {
    if (!scriptArea) return;

    if (scriptStatus) scriptStatus.textContent = "applying...";
    if (scriptError) scriptError.textContent = "";

    var source = scriptArea.value;
    scriptDirty = false;
    scriptEpoch += 1;

    fetch("/script", { method: "POST", body: source })
      .then(function (response) {
        return response.ok ? response.json() : null;
      })
      .then(function (result) {
        if (result && result.error && scriptError) scriptError.textContent = result.error;
        pollScript();
      })
      .catch(function () {});
  }

  function buildScriptPanel() {
    var panel = element("details", "script-panel");
    var summary = element("summary", "");
    summary.textContent = "Script (Rhai)";

    scriptArea = document.createElement("textarea");
    scriptArea.className = "script-area";
    scriptArea.spellcheck = false;
    scriptArea.rows = 7;
    scriptArea.addEventListener("input", function () {
      scriptDirty = true;
      scriptEpoch += 1;
    });

    var actions = element("div", "script-actions");
    var apply = element("button", "script-apply");
    apply.type = "button";
    apply.textContent = "Apply";
    apply.addEventListener("click", applyScript);

    scriptStatus = element("span", "script-status");
    scriptError = element("div", "script-error");

    actions.appendChild(apply);
    actions.appendChild(scriptStatus);
    panel.appendChild(summary);
    panel.appendChild(scriptArea);
    panel.appendChild(actions);
    panel.appendChild(scriptError);

    return panel;
  }

  /// The curve the worker rendered, downsampled by it for the picture.
  function pollCurve() {
    if (!wantsCurve || document.hidden) {
      setTimeout(pollCurve, 400);
      return;
    }

    fetch("/curve", { cache: "no-store" })
      .then(function (response) {
        return response.ok ? response.json() : null;
      })
      .then(function (state) {
        if (!state) return;
        var changed = JSON.stringify(state) !== JSON.stringify(curveState);
        curveState = state;
        if (changed) markDirty();
      })
      .catch(function () {})
      .then(function () {
        setTimeout(pollCurve, 400);
      });
  }

  /// The pitch fall the editor keeps, so a reopened window shows the curve the
  /// voice is actually playing.
  /// The six amplitude envelopes.
  function pollAmp() {
    if (document.hidden || !wantsAmp) {
      setTimeout(pollAmp, 700);
      return;
    }

    fetch("/amp", { cache: "no-store" })
      .then(function (response) {
        return response.ok ? response.json() : null;
      })
      .then(function (state) {
        if (ampDrag) return;
        if (!state || !state.curves) return;
        if (JSON.stringify(state.curves) === JSON.stringify(ampCurves)) return;

        ampCurves = state.curves;
        markDirty();
      })
      .catch(function () {})
      .then(function () {
        setTimeout(pollAmp, 700);
      });
  }

  /// The six one-shot slots: what each drum is playing.
  function pollSamples() {
    if (document.hidden || !wantsSamples) {
      setTimeout(pollSamples, 800);
      return;
    }

    fetch("/samples", { cache: "no-store" })
      .then(function (response) {
        return response.ok ? response.json() : null;
      })
      .then(function (state) {
        if (!state) return;
        if (JSON.stringify(state) === JSON.stringify(samplesState)) return;
        samplesState = state;
        markDirty();
      })
      .catch(function () {})
      .then(function () {
        setTimeout(pollSamples, 800);
      });
  }

  function uploadSampleSlot(slot, file) {
    if (!file) return;

    var reader = new FileReader();
    reader.onload = function () {
      fetch("/samples?slot=" + encodeURIComponent(slot), {
        method: "POST",
        body: reader.result
      })
        .then(function (response) {
          return response.ok ? response.json() : null;
        })
        .then(function (result) {
          if (result && result.ok) pollSamples();
        })
        .catch(function () {});
    };
    reader.readAsArrayBuffer(file);
  }

  function clearSampleSlot(slot) {
    fetch("/samples?slot=" + encodeURIComponent(slot) + "&clear=1", { method: "POST", body: "clear" })
      .then(function (response) {
        return response.ok ? response.json() : null;
      })
      .then(function (result) {
        if (result && result.ok) pollSamples();
      })
      .catch(function () {});
  }

  /// The kit's click: which one the voices are using, and loading a new one.
  function pollClick() {
    if (document.hidden || !wantsClick) {
      setTimeout(pollClick, 800);
      return;
    }

    fetch("/click", { cache: "no-store" })
      .then(function (response) {
        return response.ok ? response.json() : null;
      })
      .then(function (state) {
        if (!state) return;
        if (JSON.stringify(state) === JSON.stringify(clickState)) return;
        clickState = state;
        markDirty();
      })
      .catch(function () {})
      .then(function () {
        setTimeout(pollClick, 800);
      });
  }

  function uploadClick(file) {
    if (!file) return;

    var reader = new FileReader();
    reader.onload = function () {
      fetch("/click", { method: "POST", body: reader.result })
        .then(function (response) {
          return response.ok ? response.json() : null;
        })
        .then(function (result) {
          if (result && result.ok) pollClick();
        })
        .catch(function () {});
    };
    reader.readAsArrayBuffer(file);
  }

  function pollFall() {
    if (!wantsFall || document.hidden) {
      setTimeout(pollFall, 400);
      return;
    }

    fetch("/pitch", { cache: "no-store" })
      .then(function (response) {
        return response.ok ? response.json() : null;
      })
      .then(function (state) {
        if (!state || !state.points || fallDrag) return;
        if (JSON.stringify(state) === JSON.stringify(fallState)) return;
        fallState = state;
        markDirty();
      })
      .catch(function () {})
      .then(function () {
        setTimeout(pollFall, 400);
      });
  }

  // The goniometer needs L/R before and after the effect; the mono /scope is not
  // enough for it, so it has its own small poll.
  function pollStereo() {
    if (document.hidden || !wantsStereo) {
      setTimeout(pollStereo, 150);
      return;
    }

    fetch("/stereo", { cache: "no-store" })
      .then(function (response) {
        if (!response.ok) throw new Error("stereo " + response.status);
        return response.arrayBuffer();
      })
      .then(function (buffer) {
        var data = new Float32Array(buffer);
        var points = data.length / 4;

        if (points > 64) {
          stereoScope.inputLeft = data.subarray(0, points);
          stereoScope.inputRight = data.subarray(points, points * 2);
          stereoScope.outputLeft = data.subarray(points * 2, points * 3);
          stereoScope.outputRight = data.subarray(points * 3);
          markDirty();
        }
      })
      .catch(function () {})
      .then(function () {
        setTimeout(pollStereo, 50);
      });
  }

  // The level meter is tiny JSON, polled alongside the scope for the transfer
  // marker. A plug-in without one just answers 404.
  function pollMeter() {
    if (document.hidden || !wantsMeter) return;

    fetch("/meter", { cache: "no-store" })
      .then(function (response) {
        if (!response.ok) throw new Error("meter " + response.status);
        return response.json();
      })
      .then(function (value) {
        meter.input = typeof value.input === "number" ? value.input : null;
        meter.sidechain = typeof value.sidechain === "number" ? value.sidechain : null;
        markDirty();
      })
      .catch(function () {});
  }

  // ------------------------------------------------------------------- build
  function setBypassed(value) {
    bypassed = !!value;
    app.classList.toggle("bypassed", bypassed);
    if (powerButton) powerButton.setAttribute("aria-pressed", bypassed ? "true" : "false");
  }

  function toggleBypass() {
    setBypassed(!bypassed);
    if (window.ipc) window.ipc.postMessage("__bypass=" + (bypassed ? 1 : 0));
  }

  /// Show the controls the current mode has. A mode with no entry shows all.