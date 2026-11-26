// Part twelve: save and open, and the page starting up.
  // The patch saves as the text the host state stores, and opens from one.
  document.getElementById("undo").onclick = undo;
  document.getElementById("redo").onclick = redo;
  document.getElementById("zoom-in").onclick = () => zoomBy(1.2);
  document.getElementById("zoom-out").onclick = () => zoomBy(1 / 1.2);
  document.getElementById("zoom-fit").onclick = fitView;
  updateHistory();

  /// The patch file's name: the moment in the local zone, so two saves are two files.
  function patchName() {
    const now = new Date();
    const pad = (value) => String(value).padStart(2, "0");

    return (
      "patch-" + now.getFullYear() + "-" + pad(now.getMonth() + 1) + "-" + pad(now.getDate()) +
      "-" + pad(now.getHours()) + pad(now.getMinutes()) + ".txt"
    );
  }

  /// Says something in the status bar and lets it go by itself.
  function note(text) {
    showStatus(text, true);
    setTimeout(() => {
      if (opError === text) showStatus("");
    }, 6_000);
  }

  /// Saves the patch as the text file the Open button reads back.
  ///
  /// Beside a host the file is written where the native save-as dialog says; a page
  /// served on its own has no dialog to ask, so it falls back to a download (the one
  /// file route a browser and a webview both have). The host's own state saves itself,
  /// so this button is the export.
  async function savePatch() {
    await flushTexts();

    const text = await (await fetch("/text")).json();

    try {
      const asked = await (await fetch("/save?name=" + encodeURIComponent(patchName()), { method: "POST" })).json();

      if (asked && asked.ok && asked.dialog) {
        await watchSave();
        return;
      }
    } catch (error) {
      // No route: this page is served by something that cannot save.
    }

    downloadPatch(text);
  }

  /// Waits for the dialog to close and says where the file went.
  async function watchSave() {
    for (let tries = 0; tries < 480; tries++) {
      await new Promise((done) => setTimeout(done, 250));

      let state;

      try {
        state = await (await fetch("/save", { cache: "no-store" })).json();
      } catch (error) {
        showStatus("the patch did not save: " + error);
        return;
      }

      if (state.state === "pending") continue;

      if (state.state === "saved") note("saved " + state.path);
      else if (state.state === "failed") showStatus("the patch did not save: " + state.error);

      // Cancelled says nothing: the answer was no file.
      return;
    }
  }

  /// The download, for a page with no host dialog to ask.
  function downloadPatch(text) {
    const name = patchName();
    const link = document.createElement("a");

    link.href = URL.createObjectURL(new Blob([text], { type: "text/plain" }));
    link.download = name;
    document.body.appendChild(link);
    link.click();
    link.remove();

    note("saved " + name + " to your downloads");
    // The blob is only needed until the download has taken its name.
    setTimeout(() => URL.revokeObjectURL(link.href), 3_000);
  }

  /// The Save button is one button and two files: the graph as text a person can
  /// read and edit, or one bundle carrying the samples, tables and responses
  /// beside it. The format is a choice under the button, not a second button.
  function openSaveMenu(button) {
    const rect = button.getBoundingClientRect();

    menu.innerHTML = "";
    menu.appendChild(menuButton("Save the graph as text", () => {
      closeMenu();
      saving(savePatch);
    }));
    menu.appendChild(menuButton("Save the graph and what it plays", () => {
      closeMenu();
      saving(saveAll);
    }));
    placeMenu(rect.left, rect.bottom + 4);
  }

  /// A save that fails says so where a save that works does.
  function saving(save) {
    save().catch((error) => showStatus("the patch did not save: " + error));
  }

  document.getElementById("save").onclick = (event) => openSaveMenu(event.currentTarget);

  document.getElementById("open").onclick = () => document.getElementById("file").click();

  /// Saves everything the patch plays, not only its graph: one file that opens
  /// somewhere else with its samples, tables and responses still inside it.
  async function saveAll() {
    await flushTexts();

    const name = patchName().replace(/\.txt$/, ".iamp");

    try {
      const asked = await (await fetch(
        "/save?bundle=1&name=" + encodeURIComponent(name),
        { method: "POST" },
      )).json();

      if (asked && asked.ok && asked.dialog) {
        await watchSave();
        return;
      }
    } catch (error) {
      // No route: this page is served by something that cannot save.
    }

    try {
      const bytes = await (await fetch("/bundle", { cache: "no-store" })).arrayBuffer();
      const link = document.createElement("a");

      link.href = URL.createObjectURL(new Blob([bytes], { type: "application/octet-stream" }));
      link.download = name;
      document.body.appendChild(link);
      link.click();
      link.remove();

      note("saved " + name + " to your downloads");
      setTimeout(() => URL.revokeObjectURL(link.href), 3_000);
    } catch (error) {
      showStatus("the patch did not save: " + error);
    }
  }

  document.getElementById("file").onchange = async (event) => {
    const file = event.target.files && event.target.files[0];

    if (!file) return;

    // A saved bundle starts with its magic; anything else is the text a person saved
    // with the text button, or wrote by hand.
    const bytes = new Uint8Array(await file.arrayBuffer());
    const bundled = bytes.length >= 4
      && bytes[0] === 0x49 && bytes[1] === 0x41 && bytes[2] === 0x4d && bytes[3] === 0x50;

    const result = bundled
      ? await (await fetch("/bundle", { method: "POST", body: bytes })).json()
      : await (await fetch("/text", { method: "POST", body: await file.text() })).json();

    if (result.error) {
      showStatus(String(result.error));
    }

    event.target.value = "";

    if (bundled && result.ok === false) return;

    patch = readPatch(await (await fetch("/patch")).json());
    refreshWaves(true);
    draw();
  };

  startScopes();

  load();
  readMacros().then(draw);

  // A host moving a macro does not touch the graph, so the revision poll below
  // never sees it. The values are small enough to read on their own faster timer,
  // and a card is redrawn only when one of them actually moved.
  setInterval(async () => {
    if (busy() || drag) return;

    const before = macroValues.join(",");
    await readMacros();

    if (macroValues.join(",") !== before) draw();
  }, 120);

  // The graph changes from outside the page too: the host restores its saved
  // state, another client posts text, the fitting route writes parameters back.
  // One small request says whether anything moved, and the graph itself is only
  // read when it did.
  let patchRevision = -1;
  let spectrumTick = 0;

  setInterval(async () => {
    if (busy()) return;

    try {
      const state = await (await fetch("/state", { cache: "no-store" })).json();

      serverError = state.error || "";
      paintStatus();

      // The drawn pages are not part of the graph, so a page edited in another
      // window does not move the revision: ask for them on a slow tick.
      if (++spectrumTick >= 8) {
        spectrumTick = 0;

        for (const entry of Object.values(cards)) {
          if (entry && entry.kind && isProductCard(entry.kind, "spectrum")) refreshSpectrum(entry);
        }

      }

      if (state.revision === patchRevision) return;

      patchRevision = state.revision;

      const next = readPatch(await (await fetch("/patch", { cache: "no-store" })).json());

      if (JSON.stringify(next) === JSON.stringify(patch)) return;

      patch = next;
      refreshWaves();
      draw();
    } catch {
      // The next tick tries again.
    }
  }, 250);
})();
