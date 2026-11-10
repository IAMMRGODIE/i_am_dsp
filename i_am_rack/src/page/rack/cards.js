// Part four: the small drawings a card is built from.
  /// The listeners sit on the card, not on the picture: a repaint replaces the
  /// picture's contents while the pointer is down.
  function attachEqBands(card, module) {
    var hero = card.querySelector(".mcard-hero");
    if (!hero) return;

    var drag = null;

    function bandOf(target) {
      var node = target && target.closest ? target.closest("[data-eq-band]") : null;
      return node ? Number(node.getAttribute("data-eq-band")) : -1;
    }

    /// The node closest to a horizontal position in the picture. Grabbing the
    /// picture anywhere takes the nearest one, which is more forgiving than
    /// insisting on its six-pixel dot.
    function nearestBand(x) {
      var dots = hero.querySelectorAll(".hero-dot");
      var best = -1;
      var bestDistance = 48;

      for (var i = 0; i < dots.length; i++) {
        var rect = dots[i].getBoundingClientRect();
        var distance = Math.abs(rect.left + rect.width / 2 - x);

        if (distance < bestDistance) {
          bestDistance = distance;
          best = Number(dots[i].getAttribute("data-eq-band"));
        }
      }

      return best;
    }

    function insideHero(event) {
      var rect = hero.getBoundingClientRect();
      return event.clientX >= rect.left && event.clientX <= rect.right
        && event.clientY >= rect.top && event.clientY <= rect.bottom;
    }

    function value(name, fallback) {
      var stored = module.values[name];
      return typeof stored === "number" ? stored : fallback;
    }

    function begin(band, pointerId) {
      if (!hero.querySelector("svg")) return;

      // The picture's contents are replaced while a card is live (the scope
      // arrives every few frames), so nothing inside it is held on to: the
      // element measured below is the one on screen at that moment.
      drag = { band: band, pointerId: pointerId };
      queue(module.id + ".band", band);
    }

    function move(x, y) {
      if (!drag) return;

      var svg = hero.querySelector("svg");
      if (!svg) return;

      var rect = svg.getBoundingClientRect();
      if (!rect.width || !rect.height) return;

      var u = Math.min(1, Math.max(0, (x - rect.left) / rect.width));
      var v = Math.min(1, Math.max(0, (y - rect.top) / rect.height));
      var index = drag.band + 1;
      var frequency = 20 * Math.pow(1000, u);
      var type = Math.round(value("b" + index + "_type", 0));

      queue(module.id + ".b" + index + "_freq", frequency);
      module.values["b" + index + "_freq"] = frequency;

      if (type <= 2) {
        var gain = 24 - v * 48;
        queue(module.id + ".b" + index + "_gain", gain);
        module.values["b" + index + "_gain"] = gain;
      }

      // Show the move while the pointer is down; the next reload settles it.
      if (card.repaintHero) card.repaintHero();
    }

    function release() {
      drag = null;
    }

    card.addEventListener("pointerdown", function (event) {
      if (!hero.querySelector("svg")) return;

      var band = bandOf(event.target);
      if (band < 0 && insideHero(event)) band = nearestBand(event.clientX);
      if (band < 0) return;

      begin(band, event.pointerId);
      try { card.setPointerCapture(event.pointerId); } catch (ignore) {}
      event.preventDefault();
    });

    card.addEventListener("pointermove", function (event) {
      if (!drag) return;
      move(event.clientX, event.clientY);
      event.preventDefault();
    });

    // The picture redraws with every scope frame; while a node is held, those
    // redraws would fight the drag, so the card pauses them.
    card.addEventListener("pointerdown", function (event) {
      if (drag) card.classList.add("dragging-band");
    }, true);
    card.addEventListener("pointerup", function () {
      card.classList.remove("dragging-band");
    }, true);

    card.addEventListener("pointerup", release);
    card.addEventListener("pointercancel", release);

    // A backend that does not deliver pointer capture still sends mouse events.
    card.addEventListener("mousedown", function (event) {
      if (event.button !== 0) return;
      if (drag || !hero.querySelector("svg")) return;

      var band = bandOf(event.target);
      if (band < 0 && insideHero(event)) band = nearestBand(event.clientX);
      if (band < 0) return;

      begin(band, null);
      event.preventDefault();
    });

    card.addEventListener("mousemove", function (event) {
      if (!drag || drag.pointerId !== null) return;
      move(event.clientX, event.clientY);
    });

    card.addEventListener("mouseup", function () {
      if (drag && drag.pointerId === null) release();
    });
  }

  function makeBand(data, bandIndex) {
    var band = data.bands[bandIndex];
    var section = el("section", "band");
    var anySolo = data.bands.some(function (candidate) { return candidate.solo; });

    if (band.mute || (anySolo && !band.solo)) section.classList.add("muted");
    if (band.solo) section.classList.add("solo");
    if (collapsedBands[bandIndex]) section.classList.add("collapsed");

    section.appendChild(makeBandHeader(data, bandIndex, section));

    var slots = el("div", "slots");
    var modules = band.slots.map(function (_, slot) {
      return data.modules.find(function (candidate) {
        return candidate.band === bandIndex && candidate.slot === slot;
      }) || null;
    });
    var groups = groupIndex(modules);
    var collapsed = collapsedGroups[bandIndex] || Object.create(null);

    for (var slot = 0; slot < modules.length; slot++) {
      var module = modules[slot];

      if (module && module.group !== null && module.group !== undefined) {
        var info = groups[String(module.group)];

        if (info && collapsed[String(module.group)] && info.first !== slot) {
          continue;
        }
      }

      var slotElement = makeSlot(data, bandIndex, slot, module);

      if (module) decorateGroup(slotElement, bandIndex, slot, module, groups, collapsed);

      slots.appendChild(slotElement);
    }

    section.appendChild(slots);
    slots.scrollLeft = slotScroll[bandIndex] || 0;
    slots.addEventListener("scroll", function () {
      slotScroll[bandIndex] = slots.scrollLeft;
    });

    var resizer = el("div", "band-resizer");
    resizer.title = "Drag to resize this band";
    section.appendChild(resizer);
    attachHeightResize(resizer, slots, 260, 900, "iamrack.bandHeight." + bandIndex);

    return section;
  }

  function svgNode(tag, attributes) {
    var node = document.createElementNS("http://www.w3.org/2000/svg", tag);
    for (var key in attributes) node.setAttribute(key, attributes[key]);
    return node;
  }

  

  

  

  function lfoPreset(shape) {
    var raw;

    switch (((shape % 4) + 4) % 4) {
      case 1: raw = [[0, 0], [0.5, 1], [1, 0]]; break;
      case 2: raw = [[0, 0], [1, 1]]; break;
      case 3: raw = [[0, 1], [0.5, 1], [0.5001, 0], [1, 0]]; break;
      default: raw = [[0, 0.5], [0.25, 1], [0.5, 0.5], [0.75, 0], [1, 0.5]];
    }

    return raw.map(function (point) {
      return { time: point[0], value: point[1], bend: 0 };
    });
  }

  function lfoPointsFromJson(points, shape) {
    if (Array.isArray(points) && points.length >= 2) {
      return points.map(function (point) {
        return {
          time: Number(point[0]),
          value: Number(point[1]),
          bend: Number(point[2] === undefined ? 0 : point[2])
        };
      });
    }

    return lfoPreset(shape);
  }
