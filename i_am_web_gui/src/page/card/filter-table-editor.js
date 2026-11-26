// The filter table's editor: the pages, the drag and the wheel.
  function wireFilterTable(el, kind, values, state) {
    var table = state && state.filterTable;

    if (kind !== "filtertable" || !table || !table.pages) return;

    // Sidechain mode shows the input, not the frame: no editor, no page buttons.
    if (valueOf(values, "mode", 0) >= 0.5) return;

    var svg = el.querySelector("svg");
    if (!svg) return;

    var pages = table.pages;
    var selected = filterTablePage(values, pages);
    var view = table.view === "time" ? "time" : "freq";
    var accent = (getComputedStyle(el).getPropertyValue("--accent") || "#4c8dff").trim();
    var attached = false;
    var frozen = !!table.frozen;
    var response = view === "freq" ? filterResponsePoints(pages[selected]) : null;

    var post = function () {
      if (state.post) state.post(JSON.stringify(pages));
    };

    var drawFrequency = function () {
      IAmCard.sources.drawPreview(svg, "lfo", {
        points: response,
        color: accent,
        segments: 36,
        handles: !frozen,
      }, null);

      // Harmonic 24 is Cutoff; on the Nyquist-relative axis it sits at 24/512.
      var marker = document.createElementNS("http://www.w3.org/2000/svg", "line");
      marker.setAttribute("x1", "21.3"); marker.setAttribute("y1", "0");
      marker.setAttribute("x2", "21.3"); marker.setAttribute("y2", "64");
      marker.setAttribute("stroke", "#666");
      marker.setAttribute("stroke-dasharray", "2 4");
      marker.setAttribute("stroke-width", "1");
      marker.setAttribute("pointer-events", "none");
      svg.appendChild(marker);
    };

    var drawTime = function () {
      IAmCard.sources.drawPreview(svg, "lfo", {
        points: pages[selected],
        color: accent,
        segments: 36,
        handles: !frozen,
      }, null);
    };

    var redraw = function () {
      if (view === "freq") drawFrequency();
      else drawTime();
    };

    var setLabel = function () {
      var label = el.querySelector(".hero-label");
      if (!label) return;

      label.innerHTML = "FILTER TABLE \u00b7 PAGE " + (selected + 1) + " / " +
        pages.length + (view === "freq" ? " \u00b7 FREQ" : " \u00b7 TIME") +
        (frozen ? " \u00b7 FROZEN" : "") +
        ' \u00b7 <span data-action="filter-window" style="pointer-events:auto;cursor:pointer">WIN ' +
        (table.windowSamples || 2048) + '</span>';
    };

    var editWindow = function (anchor) {
      IAmCard.openKnobEditor(
        { label: "Window", unit: "samples" },
        table.windowSamples || 2048,
        function (value) {
          table.windowSamples = Math.max(1, Math.round(value));
          setLabel();
        },
        anchor
      );
    };

    var attachEditor = function () {
      if (attached || frozen) return;

      var holder = view === "freq" ? { points: response } : { points: pages[selected] };

      attached = true;
      IAmCard.sources.editLfo(svg, holder, {
        paint: function (points) {
          if (view === "freq") {
            response = points;
            drawFrequency();
          } else {
            pages[selected] = points;
            drawTime();
          }
        },
        commit: function (points) {
          if (view === "freq") {
            response = points;
            pages[selected] = filterTimeFromResponse(points);

            // A kernel can only realize a smoothed version of a drawn magnitude:
            // settle the picture on the response the frame really has, or the
            // next read would jump to it and the drawn curve would have shown a
            // shape the sound never had.
            response = filterResponsePoints(pages[selected]);
            drawFrequency();
          }

          post();
        },
        onStart: function () { el.dataset.busy = "1"; },
        onEnd: function () { delete el.dataset.busy; },
      });
    };

    var detachEditor = function () {
      if (!attached) return;
      attached = false;
      svg.onpointerdown = null;
      svg.ondblclick = null;
      svg.oncontextmenu = null;
    };

    var setView = function (next) {
      detachEditor();
      view = next === "time" ? "time" : "freq";
      table.view = view;

      if (view === "freq") response = filterResponsePoints(pages[selected]);

      redraw();
      attachEditor();

      setLabel();

      var button = el.querySelector('[data-action="filter-view"]');
      if (button) button.textContent = view === "freq" ? "TIME" : "FREQ";
    };

    var setFrozen = function (next) {
      frozen = next;
      table.frozen = frozen;
      el.classList.toggle("frozen", frozen);

      if (frozen) detachEditor();
      else attachEditor();

      redraw();
      setLabel();

      var button = el.querySelector('[data-action="filter-freeze"]');
      if (button) button.textContent = frozen ? "FROZEN" : "FREEZE";
    };

    var replacePages = function (result) {
      if (!result || !result.pages) return;

      table.pages = result.pages;
      pages = result.pages;
      selected = Math.min(selected, pages.length - 1);
      response = view === "freq" ? filterResponsePoints(pages[selected]) : null;
      detachEditor();
      redraw();
      attachEditor();
    };

    var progressEl = el.querySelector('[data-role="filter-progress"]');
    var importing = false;

    var showProgress = function (text) {
      if (!progressEl) return;
      progressEl.textContent = text;
      progressEl.style.display = "block";
    };

    var hideProgress = function () {
      if (progressEl) progressEl.style.display = "none";
    };

    var importBuffer = function (file) {
      if (!file || frozen || importing || table.importing) return;

      importing = true;
      table.importing = true;
      el.dataset.busy = "1";

      var windowSamples = Math.max(1, parseInt(table.windowSamples, 10) || 2048);

      file.arrayBuffer().then(function (body) {
        if (!state.import) throw new Error("no import route");
        return state.import(body, windowSamples, selected, table.importMode || "time", table.sliceMode || "auto");
      }).then(function (progress) {
        var watch = function (value) {
          if (!value) throw new Error("no progress");
          if (value.error) throw new Error(value.error);

          if (value.active) {
            var percent = value.total ? Math.round(value.done / value.total * 100) : 0;
            showProgress(value.stage === "decode" ? "IMPORT \u00b7 DECODING" :
              "IMPORT \u00b7 " + percent + "%");
            setTimeout(function () {
              state.importProgress().then(watch).catch(function (error) { fail(error); });
            }, 140);
            return;
          }

          importing = false;
          table.importing = false;
          delete el.dataset.busy;
          showProgress("IMPORT \u00b7 DONE");
          state.reload().then(function (result) {
            replacePages(result);
            setTimeout(hideProgress, 450);
          }).catch(function (error) { fail(error); });
        };

        var fail = function (error) {
          importing = false;
          table.importing = false;
          delete el.dataset.busy;
          showProgress("IMPORT \u00b7 FAILED");
          setTimeout(hideProgress, 900);
          console.warn(error);
        };

        watch(progress);
      }).catch(function (error) {
        importing = false;
        table.importing = false;
        delete el.dataset.busy;
        showProgress("IMPORT \u00b7 FAILED");
        setTimeout(hideProgress, 900);
        console.warn(error);
      });
    };

    var importFile = function () {
      var input = document.createElement("input");
      input.type = "file";
      input.accept = ".wav,audio/wav";

      input.onchange = function () {
        var file = input.files && input.files[0];

        if (file) importBuffer(file);
      };

      input.click();
    };

    // These are properties, not listeners: the product page rewires the hero
    // on every poll, and addEventListener would stack a new drop handler each
    // time. A drop then ran hundreds of handlers and took the host with it.
    el.ondragover = function (event) {
      event.preventDefault();
      if (event.dataTransfer) event.dataTransfer.dropEffect = "copy";
      el.classList.add("drop");
    };

    el.ondragleave = function () {
      el.classList.remove("drop");
    };

    el.ondrop = function (event) {
      event.preventDefault();
      el.classList.remove("drop");

      var file = event.dataTransfer && event.dataTransfer.files && event.dataTransfer.files[0];
      if (file) importBuffer(file);
    };

    svg.addEventListener("pointerdown", function (event) {
      var buttons = el.querySelectorAll(".hero-action");

      for (var index = 0; index < buttons.length; index++) {
        var rect = buttons[index].getBoundingClientRect();

        if (event.clientX >= rect.left && event.clientX <= rect.right &&
            event.clientY >= rect.top && event.clientY <= rect.bottom) {
          buttons[index].dispatchEvent(new PointerEvent("pointerdown", {
            bubbles: true,
            cancelable: true,
            clientX: event.clientX,
            clientY: event.clientY,
            pointerId: event.pointerId,
          }));
          event.preventDefault();
          return;
        }
      }
    });

    el.onpointerdown = function (event) {
      var button = event.target.closest && event.target.closest("[data-action]");

      if (!button) return;

      var action = button.getAttribute("data-action");
      var last = pages.length - 1;

      if (action === "filter-freeze") {
        setFrozen(!frozen);
      } else if (action === "filter-view") {
        setView(view === "freq" ? "time" : "freq");
      } else if (action === "filter-window") {
        if (!frozen) editWindow(button);
      } else if (action === "filter-mode") {
        table.importMode = table.importMode === "freq" ? "time" : "freq";
        button.textContent = table.importMode === "freq" ? "IMP FREQ" : "IMP TIME";
      } else if (action === "filter-slice") {
        var modes = ["auto", "fixed", "pitch", "single"];
        var next = (modes.indexOf(table.sliceMode) + 1) % modes.length;
        table.sliceMode = modes[next];
        button.textContent = "SLICE " + table.sliceMode.toUpperCase();
      } else if (action === "filter-rev" || action === "filter-inv" || action === "filter-dc") {
        var edit = action.replace("filter-", "");
        if (!frozen && state.edit) state.edit(edit, selected).then(replacePages);
      } else if (action === "filter-import") {
        if (!frozen) importFile();
      } else if (action === "filter-prev") {
        state.set("morph", last > 0 ? Math.max(selected - 1, 0) / last : 0);
      } else if (action === "filter-next") {
        state.set("morph", last > 0 ? Math.min(selected + 1, last) / last : 0);
      } else if (action === "filter-add" && !frozen) {
        pages.splice(selected + 1, 0, pages[selected].map(function (point) { return Object.assign({}, point); }));
        post();
        state.set("morph", (selected + 1) / Math.max(pages.length - 1, 1));
      } else if (action === "filter-remove" && !frozen && pages.length > 1) {
        pages.splice(selected, 1);
        post();
        state.set("morph", Math.min(selected, pages.length - 1) / Math.max(pages.length - 1, 1));
      }

      event.preventDefault();
    };

    el.classList.toggle("frozen", frozen);
    setLabel();
    redraw();
    attachEditor();
  }


