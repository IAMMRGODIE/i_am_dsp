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
    if (document.hidden || !wantsIr) {
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

  /// Hands the picked file to the product's upload route as it is. The Rust side
  /// decodes and resamples it, so the page never has to parse an audio format.
  function uploadIr() {
    var file = irInput && irInput.files && irInput.files[0];
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
    irInput.value = "";
  }

  /// Hands a picked fitted table to the product's route, which parses it and
  /// stores it in the plug-in state.
  function uploadTable() {
    var file = tableInput && tableInput.files && tableInput.files[0];
    if (!file) return;

    var reader = new FileReader();
    reader.onload = function () {
      fetch("/table", {
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
    tableInput.value = "";
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
        canvasState = state;
        markDirty();
      })
      .catch(function () {})
      .then(function () {
        setTimeout(pollCanvas, 1500);
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