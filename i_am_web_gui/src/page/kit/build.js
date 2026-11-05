// Part eight: building the face itself.
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

          if (hiddenIds.indexOf(spec.id) >= 0) return false;

          return true;
        });
      });

      face.sections = face.sections.filter(function (section) {
        return section.params.length > 0 || section.hero;
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

        if (section.hero === "modal") {
          hero.title = "Load a fitted instrument table (.bin)";
          hero.style.cursor = "pointer";
          hero.addEventListener("click", function () {
            if (tableInput) tableInput.click();
          });
        }

        if (section.hero === "impulse") {
          hero.title = "Load an impulse response";
          hero.addEventListener("click", function () {
            // In a host, ask Rust for a native dialog owned by the DAW. The
            // browser's own file input has no owner there, which is why its
            // picker can open behind the window or appear not to open at all.
            // The preview server has no IPC, so it uses the input instead.
            if (window.ipc && window.ipc.postMessage) {
              window.ipc.postMessage("__load_ir");
              return;
            }

            if (irInput) irInput.click();
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

      // applyVisibility also covers the header, but re-apply it directly here so
      // a header choice built just before the body is hidden on the first paint.
      if (face.visible) {
        var initialIds = null;
        face.visible.forEach(function (row) {
          if (row[0] === initialMode) initialIds = row[1];
        });
        headerSpecs.forEach(function (spec) {
          var control = controls[spec.id];
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
      return hero.kind === "waveform" || hero.kind === "transfer" ||
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
    wantsTable = heroes.some(function (hero) {
      return hero.kind === "modal";
    });
    if (wantsTable) {
      tableInput = document.createElement("input");
      tableInput.type = "file";
      tableInput.accept = ".bin";
      tableInput.style.display = "none";
      tableInput.addEventListener("change", uploadTable);
      document.body.appendChild(tableInput);
    }
    wantsCurve = heroes.some(function (hero) {
      return hero.kind === "curve" || hero.kind === "mapping";
    });
    if (wantsCurve) setTimeout(pollCurve, 50);
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
    if (wantsScope) pollScope();
  }

  /// Size the faceplate to the window: a bigger host window gets bigger knobs,
  /// labels and margins, not just a bigger picture area.
