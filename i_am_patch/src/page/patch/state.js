// Part one: the page's own state, the header, the selection, the macros and the cards.
// The graph editor: every node is its own card on the canvas.
//
// A card is drawn at a base size and scaled by itself, so zooming a node scales
// its face the way the product's faceplate scales to its window. The page talks
// to /catalog, /patch and /op; the graph stays the truth.
(() => {
  const svg = document.getElementById("graph");
  const menu = document.getElementById("menu");
  const NS = "http://www.w3.org/2000/svg";

  // The header's hint folds away before the buttons do: a narrow window keeps the
  // toolbar whole, and the text moves behind the ? button instead of wrapping the
  // header into three lines and pushing the stage down with it.
  const header = document.querySelector("header");
  const hint = document.getElementById("hint");
  const hintToggle = document.getElementById("hint-toggle");
  const hintPanel = document.createElement("div");

  hintPanel.className = "hint-panel";
  hintPanel.hidden = true;
  hintPanel.textContent = hint.textContent;
  header.appendChild(hintPanel);

  function hideHint() {
    hintPanel.hidden = true;
    hintToggle.setAttribute("aria-expanded", "false");
  }

  /// Folds the hint when it no longer fits. The measurement is taken unfolded:
  /// a hidden element has no width of its own to compare against.
  function foldHint() {
    hideHint();
    header.classList.remove("folded");
    header.classList.toggle("folded", hint.scrollWidth > hint.clientWidth + 1);
  }

  hintToggle.addEventListener("click", () => {
    hintPanel.hidden = !hintPanel.hidden;
    hintToggle.setAttribute("aria-expanded", hintPanel.hidden ? "false" : "true");
  });

  window.addEventListener("resize", foldHint);

  // Anywhere else is a dismissal, the way a menu closes.
  window.addEventListener("pointerdown", (event) => {
    if (hintPanel.hidden) return;
    if (event.target.closest("#hint-toggle, .hint-panel")) return;

    hideHint();
  });

  foldHint();

  const BASE_W = 280;
  const BASE_H = 300;

  let catalog = [];
  let patch = { nodes: [], edges: [], mods: [] };
  // What is selected: a set, plus the one clicked last. The scope strip and
  // the scale handle follow the primary; Delete and copy take the set.
  const selection = new Set();
  let primary = null;
  let selectedEdge = null;
  let selectedMod = null;
  let view = { x: 30, y: 30, zoom: 1 };
  let spaceHeld = false;
  let clipboard = null;
  let suspendSnapshot = 0;
  let drag = null;
  // Each card scales by itself: x and y are independent, so a wide filter and
  // a narrow one can sit side by side at the sizes they need.
  const scales = {};

  const cards = {};
  const pagesOf = {};

  // A note a text card carries. It is page text the server keeps in the graph, so
  // it posts on a timer of its own and is flushed before a save or a redraw: a
  // note typed a moment before Ctrl+S still has to be in the file.
  const textTimers = new Map();
  const textPending = new Map();

  function queueText(id, text) {
    textPending.set(id, text);
    clearTimeout(textTimers.get(id));
    textTimers.set(id, setTimeout(flushTexts, 300));
  }

  /// Writes every note still waiting on its timer, oldest first.
  async function flushTexts() {
    const pending = [...textPending];

    textPending.clear();

    for (const [id, text] of pending) {
      clearTimeout(textTimers.get(id));
      textTimers.delete(id);
      await op({ op: "text", id, text }, false);
    }
  }

  // The host's macro values, read from /macros. A macro card draws the one its
  // index names; dragging it writes the atomics a host writes, so a drag is
  // automation rather than a graph edit.
  let macroValues = new Array(64).fill(0);
  let macroPending = null;
  let macroTimer = null;

  function queueMacro(index, value) {
    macroPending = { index, value };

    if (macroTimer) return;
    macroTimer = setTimeout(flushMacro, 30);
  }

  function flushMacro() {
    macroTimer = null;
    const pending = macroPending;
    macroPending = null;

    if (!pending) return;

    fetch("/macro?index=" + pending.index + "&value=" + pending.value, { method: "POST" })
      .catch(() => {});
  }

  async function readMacros() {
    try {
      const answer = await (await fetch("/macros", { cache: "no-store" })).json();
      if (answer && Array.isArray(answer.values)) macroValues = answer.values.slice(0, 64);
    } catch {
      // The next tick tries again.
    }
  }

  function sourceName(id) {
    const source = nodeOf(id);
    const kind = source ? kindOf(source.kind) : null;
    return kind ? kind.name : "source";
  }

  function modsOf(id) {
    return patch.mods
      .map((mod, index) => ({
        slot: index,
        from: mod.from,
        to: mod.to,
        depth: mod.depth,
        mode: mod.mode || "center",
      }))
      .filter((mod) => mod.to[0] === id);
  }

