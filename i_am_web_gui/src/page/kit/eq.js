// Part four: the EQ picture, whose bands can be dragged.
  function eqBandIndex(target) {
    if (!target || !target.closest) return -1;
    var node = target.closest("[data-eq-band]");
    return node ? parseInt(node.getAttribute("data-eq-band"), 10) : -1;
  }

  function eqSelectBand(index) {
    var control = controls.band;
    if (control) setValue(control, index, true);
  }

  function eqSet(index, name, value) {
    var control = controls["b" + (index + 1) + "_" + name];
    if (control) setValue(control, value, true);
  }

  function eqPointerDown(event) {
    var index = eqBandIndex(event.target);
    if (index < 0) return;

    var hero = event.currentTarget;
    if (!hero.querySelector("svg")) return;

    eqDrag = { index: index, hero: hero, pointerId: event.pointerId };
    eqSelectBand(index);
    if (hero.setPointerCapture) hero.setPointerCapture(event.pointerId);
    event.preventDefault();
  }

  function eqPointerMove(event) {
    if (!eqDrag) return;

    var svg = eqDrag.hero.querySelector("svg");
    if (!svg) return;

    var rect = svg.getBoundingClientRect();
    if (!rect.width || !rect.height) return;

    var u = clamp((event.clientX - rect.left) / rect.width, 0, 1);
    var v = clamp((event.clientY - rect.top) / rect.height, 0, 1);
    var type = Math.round(valueOf("b" + (eqDrag.index + 1) + "_type", 0));
    eqSet(eqDrag.index, "freq", 20 * Math.pow(1000, u));
    if (type <= 2) eqSet(eqDrag.index, "gain", 24 - v * 48);
    event.preventDefault();
  }

  function eqPointerUp(event) {
    if (!eqDrag) return;

    var hero = event.currentTarget;
    if (hero.releasePointerCapture && eqDrag.pointerId !== undefined) {
      try {
        hero.releasePointerCapture(eqDrag.pointerId);
      } catch (error) {}
    }

    eqDrag = null;
  }

  function eqWheel(event) {
    var index = eqBandIndex(event.target);
    if (index < 0) return;

    event.preventDefault();
    eqSelectBand(index);
    var q = valueOf("b" + (index + 1) + "_q", 1);
    eqSet(index, "q", q * (event.deltaY < 0 ? 1.12 : 0.89));
  }

  function eqDoubleClick(event) {
    var index = eqBandIndex(event.target);
    if (index < 0) return;

    eqSelectBand(index);
    eqSet(index, "gain", 0);
    event.preventDefault();
  }

  /// The current control values, in the shape the shared card heroes expect.
  /// Passing this map is what keeps a hero tied to its parameters after the
  /// renderers moved out of this page and into card.js.
