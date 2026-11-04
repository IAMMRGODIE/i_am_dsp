// Part seven: the modulation matrix.
  function numberOr(value, fallback) {
    return typeof value === "number" ? value : fallback;
  }

  function makeModulation(data) {
    var sources = data.sources || [];
    var section = el("section", "modulation");
    var head = el("div", "mod-head");

    head.appendChild(el("span", "mod-title", "MODULATION"));
    head.appendChild(el("span", "mod-hint", "source slots \u00B7 assignments \u00B7 macros"));
    section.appendChild(head);

    var grid = el("div", "mod-sources");
    grid.addEventListener("scroll", function () { sourceScroll = grid.scrollLeft; });
    grid.addEventListener("dragover", function (event) {
      var card = event.target.closest(".mod-card");
      if (!card) return;
      event.preventDefault();
      if (event.dataTransfer) event.dataTransfer.dropEffect = "move";
      grid.querySelectorAll(".mod-card.drop").forEach(function (candidate) {
        candidate.classList.remove("drop");
      });
      card.classList.add("drop");
    });
    grid.addEventListener("dragleave", function (event) {
      var card = event.target.closest(".mod-card");
      if (card) card.classList.remove("drop");
    });
    grid.addEventListener("drop", function (event) {
      event.preventDefault();
      var card = event.target.closest(".mod-card");
      if (!card) return;
      var data = event.dataTransfer ? event.dataTransfer.getData("text/plain") : "";
      if (!data || data.indexOf("source:") !== 0) return;
      var from = data.slice(7);
      var to = card.getAttribute("data-source-id");
      card.classList.remove("drop");
      if (!to || from === to) return;
      sendNow("source." + from + ".move", to).then(load);
    });

    sources.forEach(function (source) {
      var id = source.id;
      var settings = source.settings || {};
      var kind = source.kind;
      var same = sources.filter(function (candidate) { return candidate.kind === kind; });
      var title = source.name + (same.length > 1 ? " " + (same.indexOf(source) + 1) : "");
      var accent = kind === "lfo" ? "#4c8dff" :
        kind === "envelope" ? "#35c77a" :
          kind === "random" ? "#ff9a3d" :
            kind === "table" ? "#c8e04a" :
              kind === "pitch" ? "#b08cff" : "#ff6f91";
      var paint = function () {};
      var build = function () {};
      var livePaint = null;
      var chrome;

      var onBypass = function (next) {
        sendNow("source." + id + ".bypass", next ? 1 : 0).then(load);
      };
      var onSync = function (next) {
        sendNow("source." + id + ".sync", next ? 1 : 0).then(load);
      };
      var onRemove = function () {
        sendNow("source." + id + ".remove", 1).then(load);
      };

      if (kind === "lfo") {
        var lfo = {
          rate: numberOr(settings.rate, 1),
          shape: numberOr(settings.shape, 0),
          points: lfoPointsFromJson(settings.points, numberOr(settings.shape, 0)),
          sync: !!settings.sync,
          beats: numberOr(settings.beats, 1)
        };

        paint = function (svg) { drawLfo(svg, lfo, id); };
        livePaint = function (svg, live) { drawLiveLfo(svg, live, id); };
        build = function (card, head, refresh) {
          sourceChoice(head, "Shape", ["Sine", "Triangle", "Saw", "Square", "Custom"],
            lfo.shape, function (value) {
              lfo.shape = value;
              if (value <= 3) lfo.points = lfoPreset(value);
              refresh();
              sendNow("source." + id + ".shape", value).then(load);
            });

          if (lfo.sync) {
            sourceControl(card, {
              id: "lfo." + id + ".beats", label: "Time", kind: "float",
              min: 0.03125, max: 32, default: 1, unit: "beats", log: true
            }, lfo.beats, "source." + id + ".beats", function (value) {
              lfo.beats = value;
              refresh();
            });
          } else {
            sourceControl(card, {
              id: "lfo." + id + ".rate", label: "Rate", kind: "float",
              min: 0.01, max: 20, default: 1, unit: "hz", log: true
            }, lfo.rate, "source." + id + ".rate", function (value) {
              lfo.rate = value;
              refresh();
            });
          }
        };

        chrome = {
          key: "source-" + id,
          sync: lfo.sync,
          bypass: !!source.bypass,
          onSync: onSync,
          onBypass: onBypass,
          onRemove: onRemove
        };
      } else if (kind === "envelope") {
        var env = {
          attack: numberOr(settings.attack, 10),
          release: numberOr(settings.release, 100)
        };

        paint = function (svg) { drawEnvelope(svg, env); };
        livePaint = function (svg, live) { drawLiveEnvelope(svg, live, id); };
        build = function (card) {
          sourceControl(card, {
            id: "env." + id + ".attack", label: "Attack", kind: "float",
            min: 1, max: 500, default: 10, unit: "ms", log: true
          }, env.attack, "source." + id + ".attack", function (value) {
            env.attack = value;
          });
          sourceControl(card, {
            id: "env." + id + ".release", label: "Release", kind: "float",
            min: 1, max: 500, default: 100, unit: "ms", log: true
          }, env.release, "source." + id + ".release", function (value) {
            env.release = value;
          });
        };

        chrome = {
          key: "source-" + id,
          bypass: !!source.bypass,
          onBypass: onBypass,
          onRemove: onRemove
        };
      } else if (kind === "random") {
        var random = {
          rate: numberOr(settings.rate, 4),
          smooth: numberOr(settings.smooth, 80),
          sync: !!settings.sync,
          beats: numberOr(settings.beats, 1)
        };

        paint = function (svg) { drawRandom(svg, random); };
        livePaint = function (svg, live) { drawLiveRandom(svg, live, id); };
        build = function (card, head, refresh) {
          if (random.sync) {
            sourceControl(card, {
              id: "random." + id + ".beats", label: "Time", kind: "float",
              min: 0.03125, max: 32, default: 1, unit: "beats", log: true
            }, random.beats, "source." + id + ".beats", function (value) {
              random.beats = value;
              refresh();
            });
          } else {
            sourceControl(card, {
              id: "random." + id + ".rate", label: "Rate", kind: "float",
              min: 0.01, max: 20, default: 4, unit: "hz", log: true
            }, random.rate, "source." + id + ".rate", function (value) {
              random.rate = value;
              refresh();
            });
          }

          sourceControl(card, {
            id: "random." + id + ".smooth", label: "Smooth", kind: "float",
            min: 0, max: 500, default: 80, unit: "ms", log: true
          }, random.smooth, "source." + id + ".smooth", function (value) {
            random.smooth = value;
            refresh();
          });
        };

        chrome = {
          key: "source-" + id,
          sync: random.sync,
          bypass: !!source.bypass,
          onSync: onSync,
          onBypass: onBypass,
          onRemove: onRemove
        };
      } else if (kind === "table") {
        var table = {
          rate: numberOr(settings.rate, 1),
          shape: numberOr(settings.shape, 0),
          morph: numberOr(settings.morph, 0),
          sync: !!settings.sync,
          beats: numberOr(settings.beats, 1)
        };

        paint = function (svg) { drawTable(svg, table); };
        livePaint = function (svg, live) { drawLiveTable(svg, live, id); };
        build = function (card, head, refresh) {
          sourceChoice(head, "Shape", ["Sine", "Triangle", "Saw", "Square"],
            table.shape, function (value) {
              table.shape = value;
              refresh();
              sendNow("source." + id + ".shape", value).then(load);
            });

          if (table.sync) {
            sourceControl(card, {
              id: "table." + id + ".beats", label: "Time", kind: "float",
              min: 0.03125, max: 32, default: 1, unit: "beats", log: true
            }, table.beats, "source." + id + ".beats", function (value) {
              table.beats = value;
              refresh();
            });
          } else {
            sourceControl(card, {
              id: "table." + id + ".rate", label: "Rate", kind: "float",
              min: 0.01, max: 20, default: 1, unit: "hz", log: true
            }, table.rate, "source." + id + ".rate", function (value) {
              table.rate = value;
              refresh();
            });
          }

          sourceControl(card, {
            id: "table." + id + ".morph", label: "Morph", kind: "float",
            min: 0, max: 1, default: 0, unit: "percent"
          }, table.morph, "source." + id + ".morph", function (value) {
            table.morph = value;
            refresh();
          });
        };

        chrome = {
          key: "source-" + id,
          sync: table.sync,
          bypass: !!source.bypass,
          onSync: onSync,
          onBypass: onBypass,
          onRemove: onRemove
        };
      } else if (kind === "pitch") {
        var pitch = {
          clarity: numberOr(settings.clarity, 0.5),
          glide: numberOr(settings.glide, 40)
        };

        paint = function (svg) { drawEnvelope(svg); };
        livePaint = function (svg, live) { drawLivePitch(svg, live, id); };
        build = function (card) {
          sourceControl(card, {
            id: "pitch." + id + ".clarity", label: "Clarity", kind: "float",
            min: 0, max: 1, default: 0.5, unit: "percent"
          }, pitch.clarity, "source." + id + ".clarity", function (value) {
            pitch.clarity = value;
          });
          sourceControl(card, {
            id: "pitch." + id + ".glide", label: "Glide", kind: "float",
            min: 0, max: 200, default: 40, unit: "ms"
          }, pitch.glide, "source." + id + ".glide", function (value) {
            pitch.glide = value;
          });
        };

        chrome = {
          key: "source-" + id,
          bypass: !!source.bypass,
          onBypass: onBypass,
          onRemove: onRemove
        };
      } else if (kind === "midi") {
        var midi = {
          mode: numberOr(settings.mode, 0),
          transpose: numberOr(settings.transpose, 0),
          fine: numberOr(settings.fine, 0),
          note: curvePointsFromJson(settings.note_curve),
          velocity: curvePointsFromJson(settings.velocity_curve)
        };
        var curve = midi.mode === 1 ? midi.velocity : midi.note;
        var curveName = midi.mode === 1 ? "velocity_curve" : "note_curve";

        paint = function (svg) {
          drawMidiCurve(svg, curve, function (points) {
            sendNow("source." + id + "." + curveName, serializeCurve(points)).then(load);
          });
        };
        livePaint = function (svg, live) { drawLiveMidi(svg, live, id); };
        build = function (card, head) {
          sourceChoice(head, "Value", ["Note", "Velocity"], midi.mode, function (value) {
            sendNow("source." + id + ".mode", value).then(load);
          });

          sourceControl(card, {
            id: "midi." + id + ".transpose", label: "Transpose", kind: "int",
            min: -24, max: 24, default: 0, unit: "st"
          }, midi.transpose, "source." + id + ".transpose", function (value) {
            midi.transpose = Math.round(value);
          });

          sourceControl(card, {
            id: "midi." + id + ".fine", label: "Fine", kind: "float",
            min: -100, max: 100, default: 0, unit: "ct"
          }, midi.fine, "source." + id + ".fine", function (value) {
            midi.fine = value;
          });
        };

        chrome = {
          key: "source-" + id,
          bypass: !!source.bypass,
          onBypass: onBypass,
          onRemove: onRemove
        };
      }

      chrome.dragId = id;
      var card = makeSourceCard(title, accent, paint, build, livePaint, chrome);
      card.setAttribute("data-source-id", String(id));
      grid.appendChild(card);
    });

    var addCard = el("div", "slot-empty mod-add");
    var addSelect = document.createElement("select");
    addSelect.className = "mcard-select";
    addSelect.setAttribute("aria-label", "Add modulation source");

    var addPlaceholder = document.createElement("option");
    addPlaceholder.value = "";
    addPlaceholder.textContent = "+ Add source";
    addSelect.appendChild(addPlaceholder);

    [["lfo", "LFO"], ["envelope", "Envelope"], ["random", "Random"], ["table", "Table"],
      ["pitch", "Pitch"], ["midi", "MIDI"]]
      .forEach(function (entry) {
        var option = document.createElement("option");
        option.value = entry[0];
        option.textContent = entry[1];
        addSelect.appendChild(option);
      });

    addSelect.addEventListener("change", function () {
      if (!addSelect.value) return;
      sendNow("source.add", addSelect.value).then(load);
    });

    addCard.appendChild(addSelect);
    grid.appendChild(addCard);

    section.appendChild(grid);

    var resizer = el("div", "mod-resizer");
    resizer.title = "Drag to resize modulation sources";
    section.appendChild(resizer);
    attachHeightResize(resizer, grid, 220, 800, "iamrack.modHeight");

    return section;
  }
