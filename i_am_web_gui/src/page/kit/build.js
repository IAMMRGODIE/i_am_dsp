// Part eight: building the face itself.
  /// Lets a hero take a file by drop as well as by its picker: the browsers a
  /// plug-in runs in all hand drag-and-drop over to the page, while their file
  /// pickers are the ones that can end up behind the DAW window.
  function acceptDrop(el, onFile) {
    el.addEventListener("dragover", function (event) {
      event.preventDefault();
      if (event.dataTransfer) event.dataTransfer.dropEffect = "copy";
      el.classList.add("drop");
    });

    el.addEventListener("dragleave", function () {
      el.classList.remove("drop");
    });

    el.addEventListener("drop", function (event) {
      event.preventDefault();
      el.classList.remove("drop");

      var file = event.dataTransfer && event.dataTransfer.files && event.dataTransfer.files[0];
      if (file) onFile(file);
    });
  }

  function buildFace() {
    isPreview = !face.hosted;
    app.classList.toggle("hosted", !isPreview);
    accent = face.accent || accent;
    negative = mixHex(accent, "#9a9a9a", 0.5);

    var root = document.documentElement.style;
    root.setProperty("--accent", accent);
    root.setProperty("--negative", negative);
    document.title = face.title;

    if (isPreview) {
      document.body.classList.add("preview");
      app.style.width = face.size[0] + "px";
      app.style.height = face.size[1] + "px";
    }

    // A mode switch and the header's extra choices are parameters drawn in the
    // header, not knobs in the body.
    var headerIds = face.header || [];
    var headerSpecs = [];
    // Parameters the face is told not to draw: the host still has them.
    var hiddenIds = face.hidden || [];
    // Hidden parameters still need a value the page can read: a hero can draw
    // and switch what its section no longer shows (the sampler's loop, which the
    // picture owns).
    var hiddenSpecs = [];

    if (face.mode || headerIds.length || hiddenIds.length) {
      face.sections.forEach(function (section) {
        section.params = section.params.filter(function (spec) {
          if (face.mode && spec.id === face.mode) {
            modeSpec = spec;
            return false;
          }

          if (headerIds.indexOf(spec.id) >= 0) {
            headerSpecs.push(spec);
            return false;
          }

          if (hiddenIds.indexOf(spec.id) >= 0) {
            hiddenSpecs.push(spec);
            return false;
          }

          return true;
        });
      });

      face.sections = face.sections.filter(function (section) {
        return section.params.length > 0 || section.hero;
      });

      hiddenSpecs.forEach(function (spec) {
        controls[spec.id] = {
          spec: spec,
          value: spec.default,
          el: null,
          paint: function () {}
        };
      });
    }

    // The downsampler cannot hold samples faster than the host is running; the
    // DSP clamps it, so the knob must not offer a range the effect cannot use.
    if (face.rate) {
      face.sections.forEach(function (section) {
        section.params.forEach(function (spec) {
          if (spec.id === "target_sample_rate" && spec.max > face.rate) {
            spec.max = face.rate;
          }
        });
      });
    }

    var header = element("header", "header");
    powerButton = element("button", "power");
    powerButton.type = "button";
    powerButton.title = "Bypass";
    powerButton.setAttribute("aria-label", "Bypass");
    powerButton.setAttribute("aria-pressed", "false");
    powerButton.appendChild(element("span", "dot"));
    powerButton.addEventListener("click", toggleBypass);

    var title = element("span", "title");
    title.textContent = face.title;

    header.appendChild(powerButton);
    header.appendChild(title);

    noteBadge = element("span", "note-badge");
    noteBadge.style.display = "none";
    header.appendChild(noteBadge);

    header.appendChild(element("span", "spacer"));

    var tools = element("div", "header-tools");
    var modeEl = modeSpec ? buildMode(modeSpec) : null;

    if (modeEl) tools.appendChild(modeEl);

    headerSpecs.forEach(function (spec) {
      if (spec.widget === "choice") {
        tools.appendChild(buildHeaderChoice(spec));
      } else if (spec.kind === "bool") {
        var dot = makeControl(spec, true);
        var wrap = element("div", "header-switch");
        var caption = element("span", "header-caption");
        caption.textContent = spec.label;
        wrap.appendChild(dot);
        wrap.appendChild(caption);
        // A mode that hides this switch has to hide its caption too, so the
        // control the visibility rules move is the whole item, not just the dot.
        if (controls[spec.id]) controls[spec.id].el = wrap;
        tools.appendChild(wrap);
      } else {
        tools.appendChild(makeControl(spec, true));
      }
    });

    if (face.clear) {
      var clearTail = element("button", "clear-tail");
      clearTail.type = "button";
      clearTail.textContent = "CLEAR";
      clearTail.title = "Stop the tail immediately";
      clearTail.addEventListener("click", function () {
        fetch("/clear", { method: "POST" }).catch(function () {});
        clearTail.classList.add("cleared");
        setTimeout(function () {
          clearTail.classList.remove("cleared");
        }, 160);
      });
      tools.appendChild(clearTail);
    }

    if (modeEl || headerSpecs.length || face.clear) header.appendChild(tools);

    // The shell exists before the body so the tab strip can sit between them.
    var shell = element("div", "face");

    // A paged faceplate packs its rows: the tabs name the page, so the room the
    // wide spacing used to take is not needed.
    if (face.pages && face.pages.length) shell.classList.add("compact");

    var body = element("div", "body");
    var many = face.sections.length > 1;
    // When every page holds a single section, the tab already names it: drawing
    // the section caption as well would say the same thing twice.
    var oneSectionPerPage =
      face.pages && face.pages.length > 0 &&
      face.pages.every(function (page) { return (page.sections || []).length === 1; });
    var usedHeroes = {};
    var sectionEls = [];
    // Which page a section is on, so a hero shared by two pages (the OTT's meter
    // is on both) can be drawn twice: only one page is visible at a time.
    var heroPageOf = {};

    if (face.pages) {
      face.pages.forEach(function (page, index) {
        (page.sections || []).forEach(function (name) { heroPageOf[name] = index; });
      });
    }

    face.sections.forEach(function (section) {
      var el = element("section", "section");

      if (many && section.name && !oneSectionPerPage) {
        var caption = element("div", "caption");
        caption.textContent = section.name;
        el.appendChild(caption);
      }

      // One hero per page is enough: a nested effect can ask for the same hero
      // in a second section on the same page (the crusher's gain stage embeds
      // Gain), and drawing the same live trace twice only costs space. Two
      // different pages may share it, because only one of them is ever shown.
      var heroPage = Object.prototype.hasOwnProperty.call(heroPageOf, section.name)
        ? heroPageOf[section.name]
        : -1;
      var heroUsed = usedHeroes[section.hero];
      var heroAllowed = section.hero && (heroUsed === undefined ||
        (heroPage >= 0 && heroUsed >= 0 && heroPage !== heroUsed));

      if (heroAllowed) {
        usedHeroes[section.hero] = heroPage;
        var hero = element("div", "hero");
        hero.dataset.hero = section.hero;

        if (section.hero === "sampler") {
          hero.title = "Drop a .wav. Drag: start. Right-drag: loop. Wheel: zoom. Shift-drag: pan.";
          hero.style.cursor = "pointer";
          hero.style.touchAction = "none";
          acceptDrop(hero, postSample);

          // The picture leaves a margin at each end of its 520 unit box, so the
          // sample itself runs from 20 to 500 inside it; the view says which
          // part of the sample those 480 units show.
          var sampleView = { from: 0, to: 1 };

          var sampleX = function (event) {
            var rect = hero.getBoundingClientRect();
            var across = (event.clientX - rect.left) / rect.width;
            var unit = clamp((across * 520 - 20) / 480, 0, 1);

            return sampleView.from + unit * (sampleView.to - sampleView.from);
          };

          // Six pixels of the picture, measured in sample units, so the hit test
          // stays the same size however far the picture is zoomed in.
          var nearMark = function (x, marks) {
            var reach = (sampleView.to - sampleView.from) * 0.0125;

            return marks.some(function (mark) { return Math.abs(mark - x) < reach; });
          };

          // What a click or a drag means is the hero's own behaviour, shared
          // with a patcher node; this page says where the edit goes.
          wireHero(hero, "sampler", {
            read: function () { return sampleState; },
            value: function (id) { return controls[id] ? controls[id].value : 0; },
            set: function (id, value) {
              sampleDragging = true;
              setControl(id, value);
            },
            toggle: function (id) {
              var control = controls[id];

              if (!control) return;

              setControl(id, control.value > 0.5 ? 0 : 1);
            },
            commit: function () {
              sampleDragging = false;
              markDirty();
            },
            action: function (name, query) {
              var extra = Object.keys(query || {})
                .map(function (key) { return "&" + key + "=" + query[key]; })
                .join("");

              fetch("/sample?action=" + name + extra, { method: "POST", body: name })
                .then(function (response) { return response.ok ? response.json() : null; })
                .then(function (result) {
                  if (!result || !result.version) return;
                  sampleState = result;
                  markDirty();
                })
                .catch(function () {});
            },
            pick: function (file) {
              if (file) {
                postSample(file);
                return;
              }

              if (sampleInput) sampleInput.click();
            },
            share: function (event) { return sampleX(event); },
            near: function (at) {
              return nearMark(at, (sampleState && sampleState.slices) || []);
            },
            repaint: function () { markDirty(); }
          });

          // The view is the page's own: panning and zooming change what the
          // picture shows, not what the sample is.
          hero.addEventListener("pointerdown", function (event) {
            if (!event.shiftKey || !sampleState || !sampleState.peaks) return;

            var rect = hero.getBoundingClientRect();
            var pan = {
              startX: event.clientX,
              width: rect.width,
              from: sampleView.from,
              to: sampleView.to
            };
            var move = function (moved) {
              var span = pan.to - pan.from;
              var delta = (moved.clientX - pan.startX) / pan.width * span;
              var from = clamp(pan.from - delta, 0, 1 - span);

              sampleView.from = from;
              sampleView.to = from + span;
              markDirty();
            };
            var up = function () {
              hero.removeEventListener("pointermove", move);
              hero.removeEventListener("pointerup", up);
            };

            hero.addEventListener("pointermove", move);
            hero.addEventListener("pointerup", up);
            event.preventDefault();
          });

          // The wheel narrows the window around the pointer, so a drag lands on
          // the exact place it points at.
          hero.addEventListener("wheel", function (event) {
            if (!sampleState || !sampleState.peaks || !sampleState.peaks.length) return;

            event.preventDefault();

            var rect = hero.getBoundingClientRect();
            var across = clamp((event.clientX - rect.left) / rect.width * 520 - 20, 0, 480) / 480;
            var span = sampleView.to - sampleView.from;
            var focus = sampleView.from + across * span;
            var next = clamp(span * (event.deltaY > 0 ? 1.25 : 0.8), 1 / 32, 1);
            var from = clamp(focus - across * next, 0, 1 - next);

            sampleView.from = from;
            sampleView.to = from + next;
            markDirty();
          }, { passive: false });
        }
        if (section.hero === "shape") {
          hero.title = "Load a wavetable: click, or drop a .wav";
          hero.style.cursor = "pointer";
          acceptDrop(hero, postWave);

          wireHero(hero, "shape", {
            read: function () { return wavesState; },
            value: function (id) { return controls[id] ? controls[id].value : 0; },
            set: setControl,
            toggle: function (id) {
              var control = controls[id];

              if (!control) return;

              setControl(id, control.value > 0.5 ? 0 : 1);
            },
            commit: markDirty,
            action: function (name, query) {
              var extra = Object.keys(query || {})
                .map(function (key) { return "&" + key + "=" + query[key]; })
                .join("");

              fetch("/waves?action=" + name + extra, { method: "POST", body: name })
                .then(function (response) { return response.ok ? response.json() : null; })
                .then(function (result) {
                  if (!result || !result.version) return;
                  wavesState = result;
                  markDirty();
                })
                .catch(function () {});
            },
            pick: function (file) {
              if (file) {
                postWave(file);
                return;
              }

              if (waveInput) waveInput.click();
            },
            frame: function (step) { stepWaveFrame(step); },
            frameIndex: waveFrame,
            cycleMode: cycleWaveMode,
            repaint: markDirty
          });
        }
        if (section.hero === "modal") {
          hero.title = "Load a fitted table (.bin), or a recorded note (.wav) to fit: click or drop a file";
          hero.style.cursor = "pointer";
          acceptDrop(hero, postTableFile);
          hero.addEventListener("click", function () {
            if (tableInput) tableInput.click();
          });
        }

        if (section.hero === "fall") {
          hero.title = "Drag a point: time and pitch. Drag the ring: segment bend. Right-click a point removes it.";
          hero.style.touchAction = "none";
          hero.addEventListener("pointerdown", fallPointerDown);
          hero.addEventListener("pointermove", fallPointerMove);
          hero.addEventListener("pointerup", fallPointerUp);
          hero.addEventListener("pointercancel", fallPointerUp);
          hero.addEventListener("contextmenu", fallContextMenu);
          hero.addEventListener("dblclick", fallDoubleClick);
        }

        if (section.hero === "kick") {
          var switchCurve = function (event) {
            var button = event.target.closest ? event.target.closest("[data-kick-mode]") : null;
            if (!button) return false;

            kickMode = button.getAttribute("data-kick-mode");
            markDirty();
            event.preventDefault();
            return true;
          };

          hero.dataset.ampDrum = "0";
          hero.style.touchAction = "none";
          hero.addEventListener("pointerdown", function (event) {
            if (event.target.closest && event.target.closest("[data-kick-mode]")) return;
            if (kickMode === "amp") ampPointerDown(hero, event); else fallPointerDown(event);
          });
          hero.addEventListener("pointermove", function (event) {
            if (kickMode === "amp") ampPointerMove(hero, event); else fallPointerMove(event);
          });
          hero.addEventListener("pointerup", function (event) {
            if (switchCurve(event)) return;
            if (kickMode === "amp") ampPointerUp(hero, event); else fallPointerUp(event);
          });
          hero.addEventListener("dblclick", function (event) {
            if (kickMode === "amp") ampDoubleClick(hero, event); else fallDoubleClick(event);
          });
          hero.addEventListener("contextmenu", function (event) {
            if (kickMode === "amp") ampContextMenu(hero, event); else fallContextMenu(event);
          });
        } else if (section.hero === "snare") {
          var snareSlots = { base: 6, noise: 7, pitch: 8 };
          var snareSetMode = function () {
            hero.dataset.ampDrum = String(snareSlots[snareMode] || 6);
          };

          snareSetMode();
          hero.title = "Drag a dot, a ring for the bend; double click to add, right click to remove. The button switches shell, wires and pitch.";
          hero.addEventListener("pointerdown", function (event) {
            if (event.target.closest && event.target.closest("[data-snare-mode]")) return;
            ampPointerDown(hero, event);
          });
          hero.addEventListener("pointermove", function (event) { ampPointerMove(hero, event); });
          hero.addEventListener("pointerup", function (event) {
            var button = event.target.closest ? event.target.closest("[data-snare-mode]") : null;

            if (button) {
              snareMode = button.getAttribute("data-snare-mode");
              snareSetMode();
              markDirty();
              event.preventDefault();
              return;
            }

            ampPointerUp(hero, event);
          });
          hero.addEventListener("dblclick", function (event) { ampDoubleClick(hero, event); });
          hero.addEventListener("contextmenu", function (event) { ampContextMenu(hero, event); });
        } else if (section.hero && section.hero.indexOf("amp_") === 0) {
          var drums = { amp_snare: 1, amp_hat: 2, amp_clap: 3, amp_tom: 4, amp_cymbal: 5 };
          hero.dataset.ampDrum = String(drums[section.hero] || 0);
          hero.title = "Drag a dot for the level, a ring for the bend; double click to add, right click to remove.";
          hero.addEventListener("pointerdown", function (event) { ampPointerDown(hero, event); });
          hero.addEventListener("pointermove", function (event) { ampPointerMove(hero, event); });
          hero.addEventListener("pointerup", function (event) { ampPointerUp(hero, event); });
          hero.addEventListener("dblclick", function (event) { ampDoubleClick(hero, event); });
          hero.addEventListener("contextmenu", function (event) { ampContextMenu(hero, event); });
        }

        if (section.hero === "samples") {
          hero.title = "Load a one-shot into a slot: it replaces that drum.";
          var sampleInputs = {};
          var slots = ["kick", "snare", "hat", "clap", "tom", "cymbal"];

          slots.forEach(function (slot) {
            var input = element("input", "");

            input.type = "file";
            input.accept = ".wav,audio/*";
            input.style.display = "none";
            input.addEventListener("change", function () {
              if (input.files && input.files[0]) uploadSampleSlot(slot, input.files[0]);
              input.value = "";
            });
            document.body.appendChild(input);
            sampleInputs[slot] = input;
          });

          hero.addEventListener("pointerup", function (event) {
            var clear = event.target.closest("[data-clear]");

            if (clear) {
              clearSampleSlot(clear.getAttribute("data-clear"));
              event.preventDefault();
              return;
            }

            var target = event.target.closest("[data-load]");
            if (!target) return;

            var slot = target.getAttribute("data-load");
            if (sampleInputs[slot]) {
              sampleInputs[slot].click();
              event.preventDefault();
            }
          });
        }

        if (section.hero === "kit") {
          hero.title = "Load a one-shot click: it plays under every hit.";
          hero.addEventListener("pointerup", function (event) {
            var button = event.target.closest(".hero-action");

            if (button && button.getAttribute("data-action") === "load") {
              if (clickInput) clickInput.click();
              event.preventDefault();
            }
          });
        }

        if (section.hero === "impulse") {
          hero.title = "Load an impulse response: click or drop a file; drag the dots to fade its ends";
          acceptDrop(hero, postIr);

          var irX = function (event) {
            var rect = hero.getBoundingClientRect();
            var across = (event.clientX - rect.left) / rect.width;

            return clamp((across * 520 - 20) / 480, 0, 1);
          };

          wireHero(hero, "impulse", {
            read: function () { return irState; },
            value: function (id) { return controls[id] ? controls[id].value : 0; },
            set: setControl,
            toggle: function (id) {
              var control = controls[id];

              if (!control) return;

              setControl(id, control.value > 0.5 ? 0 : 1);
            },
            commit: function () {
              irDragging = false;

              fetch("/ir?fade_in=" + irState.fade_in + "&fade_out=" + irState.fade_out, {
                method: "POST",
                body: "fades"
              }).catch(function () {});
            },
            pick: function (file) {
              if (file) {
                postIr(file);
                return;
              }

              // In a host, ask Rust for a native dialog owned by the DAW. The
              // browser's own file input has no owner there, which is why its
              // picker can open behind the window or appear not to open at all.
              if (window.ipc && window.ipc.postMessage) {
                window.ipc.postMessage("__load_ir");
                return;
              }

              if (irInput) irInput.click();
            },
            share: function (event) { return irX(event); },
            repaint: function () {
              irDragging = true;
              markDirty();
            }
          });
        }
        if (section.hero === "eq") {
          hero.title = "Drag a band: horizontal frequency, vertical gain; wheel changes Q";
          hero.style.touchAction = "none";
          hero.addEventListener("pointerdown", eqPointerDown);
          hero.addEventListener("pointermove", eqPointerMove);
          hero.addEventListener("pointerup", eqPointerUp);
          hero.addEventListener("pointercancel", eqPointerUp);
          hero.addEventListener("wheel", eqWheel, { passive: false });
          hero.addEventListener("dblclick", eqDoubleClick);
        }

        if (section.hero === "compressor") {
          hero.addEventListener("wheel", function (event) {
            if (!event.target.closest(".env-box")) return;
            event.preventDefault();
            envWindow = clamp(
              Math.round(envWindow * (event.deltaY > 0 ? 1.25 : 0.8)),
              32,
              ENV_MAX
            );
            markDirty();
          }, { passive: false });
        }

        heroes.push({ el: hero, kind: section.hero });
        el.appendChild(hero);
      } else {
        el.classList.add("plain");
      }

      var row = element("div", "controls");
      section.params.forEach(function (spec) {
        row.appendChild(makeControl(spec));
      });
      el.appendChild(row);
      body.appendChild(el);
      sectionEls.push({ name: section.name, el: el });
    });

    // A faceplate with more controls than one window shows comfortably can hand
    // its sections to tabs; the hero and the switches stay in the sections they
    // belong to.
    var pagesStrip = face.pages && face.pages.length ? buildPages(sectionEls) : null;

    if (modeSpec) {
      var initialMode = Math.round(controls[modeSpec.id].value);
      applyVisibility(initialMode);

      // A header control is mode-specific only when some mode lists it; the rest
      // (an envelope's sync, say) belong to the face and stay put.
      if (face.visible) {
        var initialIds = null;
        var listed = {};

        face.visible.forEach(function (row) {
          (row[1] || []).forEach(function (id) { listed[id] = true; });
          if (row[0] === initialMode) initialIds = row[1];
        });

        headerSpecs.forEach(function (spec) {
          var control = controls[spec.id];

          if (!listed[spec.id]) return;

          if (control && control.el && initialIds && initialIds.indexOf(spec.id) < 0) {
            control.el.style.display = "none";
          }
        });
      }
    }
    applySwitches();
    setTimeout(function () {
      if (modeSpec) applyVisibility(Math.round(controls[modeSpec.id].value));
    }, 0);
    setTimeout(pollNote, 100);

    // Every hero the page draws is live: a waveform, a transfer with its RMS
    // bands, or a response with the input and output spectra on it.
    wantsScope = heroes.some(function (hero) {
      return hero.kind === "waveform" || hero.kind === "unison" || hero.kind === "transfer" ||
        hero.kind === "spectrum" || hero.kind === "spectra" ||
        hero.kind === "compressor" || hero.kind === "gate" ||
        hero.kind === "expander" || hero.kind === "limiter" || hero.kind === "eq";
    });
    wantsMeter = heroes.some(function (hero) {
      return hero.kind === "compressor" || hero.kind === "gate" ||
        hero.kind === "expander" || hero.kind === "limiter";
    });
    wantsStereo = heroes.some(function (hero) {
      return hero.kind === "stereo";
    });
    wantsParticles = heroes.some(function (hero) {
      return hero.kind === "particles";
    });
    wantsIr = heroes.some(function (hero) {
      return hero.kind === "impulse";
    });
    if (wantsIr) {
      irInput = document.createElement("input");
      irInput.type = "file";
      irInput.accept = ".wav,.flac,.mp3,.ogg,audio/*";
      irInput.style.display = "none";
      irInput.addEventListener("change", uploadIr);
      document.body.appendChild(irInput);
      setTimeout(pollIr, 50);
    }
    wantsSample = heroes.some(function (hero) {
      return hero.kind === "sampler";
    });
    if (wantsSample) {
      sampleInput = document.createElement("input");
      sampleInput.type = "file";
      sampleInput.accept = ".wav,audio/*";
      sampleInput.style.display = "none";
      sampleInput.addEventListener("change", uploadSample);
      document.body.appendChild(sampleInput);
      setTimeout(pollSample, 50);
    }
    wantsWaves = heroes.some(function (hero) {
      return hero.kind === "shape";
    });
    if (wantsWaves) {
      waveInput = document.createElement("input");
      waveInput.type = "file";
      waveInput.accept = ".wav";
      waveInput.style.display = "none";
      waveInput.addEventListener("change", uploadWave);
      document.body.appendChild(waveInput);
      setTimeout(pollWaves, 50);
    }
    wantsTable = heroes.some(function (hero) {
      return hero.kind === "modal";
    });
    if (wantsTable) {
      tableInput = document.createElement("input");
      tableInput.type = "file";
      tableInput.accept = ".bin,.wav";
      tableInput.style.display = "none";
      tableInput.addEventListener("change", uploadTable);
      document.body.appendChild(tableInput);
    }
    wantsCurve = heroes.some(function (hero) {
      return hero.kind === "curve" || hero.kind === "mapping";
    });
    if (wantsCurve) setTimeout(pollCurve, 50);
    wantsFall = heroes.some(function (hero) {
      return hero.kind === "fall" || hero.kind === "kick";
    });
    if (wantsFall) setTimeout(pollFall, 50);
    wantsClick = heroes.some(function (hero) {
      return hero.kind === "kit";
    });
    wantsSamples = heroes.some(function (hero) {
      return hero.kind === "samples";
    });
    if (wantsSamples) setTimeout(pollSamples, 50);
    wantsAmp = heroes.some(function (hero) {
      return hero.kind === "kick" || hero.kind === "snare" || hero.kind.indexOf("amp") === 0;
    });
    if (wantsAmp) setTimeout(pollAmp, 50);
    if (wantsClick) {
      clickInput = document.createElement("input");
      clickInput.type = "file";
      clickInput.accept = ".wav,audio/*";
      clickInput.style.display = "none";
      clickInput.addEventListener("change", uploadClick);
      document.body.appendChild(clickInput);
      setTimeout(pollClick, 50);
    }
    wantsOtt = heroes.some(function (hero) {
      return hero.kind === "ott";
    });
    if (wantsOtt) setTimeout(pollOtt, 50);
    wantsVocoder = heroes.some(function (hero) {
      return hero.kind === "vocodex";
    });
    if (wantsVocoder) setTimeout(pollVocoder, 50);
    if (face.script) {
      body.appendChild(buildScriptPanel());
      wantsScript = true;
      setTimeout(pollScript, 50);
    }
    if (wantsMeter) setInterval(pollMeter, 100);
    if (wantsStereo) setTimeout(pollStereo, 50);
    if (wantsParticles) {
      setInterval(function () {
        if (!document.hidden && wantsParticles) markDirty();
      }, 40);
    }

    shell.appendChild(header);
    if (pagesStrip) shell.appendChild(pagesStrip);
    shell.appendChild(body);
    app.appendChild(shell);

    applyScale();
    markDirty();
    poll();
    pollCanvas();
    pollSpectrum();
    if (wantsScope) pollScope();
  }

  /// Size the faceplate to the window: a bigger host window gets bigger knobs,
  /// labels and margins, not just a bigger picture area.
