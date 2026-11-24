// Part ten: what a hero's buttons and gestures mean.
//
// A product page and a patcher node draw the same picture from the same
// metadata; the only difference is where an edit goes. So the meaning lives
// here, once per hero, and the page supplies a context:
//
//   read()          what the picture shows now
//   value(id)       one control's current value
//   set(id, value)  move one control (the page decides whether that posts)
//   toggle(id)      flip a switch
//   action(name, query)  run one of the product's own actions
//   commit(ids)     the gesture ended: write what it moved
//   pick()          ask the page for a file
//   share(event)    where a press sits along the picture, 0..1
//   near(at)        whether a slice mark is that close
//
// A hero without a behaviour keeps whatever its page wires itself.
  var HERO_BEHAVIOURS = {};

  function registerHeroBehaviour(hero, behaviour) {
    HERO_BEHAVIOURS[hero] = behaviour;
  }

  function heroBehaviour(hero) {
    return HERO_BEHAVIOURS[hero] || null;
  }

  /// Wires one hero to a page's context: its buttons, its drags, its clicks.
  ///
  /// The listeners are attached here rather than per page, so a hero behaves
  /// the same wherever it is drawn. A press that travels is a drag, not a
  /// click, and neither runs the other.
  function wireHero(hero, name, context) {
    var behaviour = heroBehaviour(name);

    if (!behaviour) return;

    var drag = null;
    var pressed = null;
    var moved = false;

    hero.addEventListener("pointerdown", function (event) {
      pressed = { x: event.clientX, y: event.clientY };
      moved = false;

      if (event.target.closest(".hero-action")) return;
      if (!behaviour.down) return;

      drag = behaviour.down(context, event) || null;

      if (drag) {
        try {
          hero.setPointerCapture(event.pointerId);
        } catch (ignore) {}
        event.preventDefault();
      }
    });

    hero.addEventListener("pointermove", function (event) {
      if (pressed) {
        moved = moved ||
          Math.abs(event.clientX - pressed.x) + Math.abs(event.clientY - pressed.y) > 4;
      }

      if (!drag || !behaviour.move) return;

      behaviour.move(context, drag, event);
    });

    hero.addEventListener("pointerup", function () {
      var finished = drag;

      drag = null;

      if (finished && behaviour.up) behaviour.up(context, finished);
    });

    hero.addEventListener("pointercancel", function () {
      drag = null;
    });

    hero.addEventListener("wheel", function (event) {
      if (!behaviour.wheel) return;

      behaviour.wheel(context, event);

      // The card can sit inside a board that zooms on the wheel (the patcher does):
      // a hero that took the wheel for its own control keeps it to itself. A hero
      // that had nothing to do with it leaves the event alone, so the board still
      // zooms over a card's empty picture.
      if (event.defaultPrevented) event.stopPropagation();
    }, { passive: false });

    hero.addEventListener("click", function (event) {
      var button = event.target.closest(".hero-action");
      var action = button && button.getAttribute("data-action");

      if (button && action) {
        var handler = behaviour.actions && behaviour.actions[action];

        if (handler) handler(context);

        return;
      }

      // A drag ends with a click of its own; it is not a press on the picture.
      if (moved) {
        moved = false;
        return;
      }

      if (behaviour.click) behaviour.click(context, event);
    });

    hero.addEventListener("dblclick", function (event) {
      if (!behaviour.double || event.target.closest(".hero-action")) return;

      behaviour.double(context, event);
    });

    hero.addEventListener("contextmenu", function (event) {
      if (behaviour.menu && behaviour.menu(context, event)) event.preventDefault();
    });
  }

  /// Flips a switch: a button is a whole edit, so it is written at once rather
  /// than left for a gesture to finish.
  function flip(ctx, id) {
    ctx.set(id, ctx.value(id) > 0.5 ? 0 : 1);
    ctx.commit([id]);
  }

  /// One EQ band, from the element under the press.
  function eqBandAt(event) {
    var node = event.target.closest && event.target.closest("[data-eq-band]");

    return node ? Number(node.getAttribute("data-eq-band")) : -1;
  }

  registerHeroBehaviour("eq", {
    down: function (ctx, event) {
      var band = eqBandAt(event);

      if (band < 0) return null;

      // Selecting the band rewrites the card's controls and repaints the hero;
      // doing that on the press would move the picture out from under the
      // pointer. Keep selection for the release and only start moving after a
      // real drag, so a click stays a click.
      return {
        band: band,
        startX: event.clientX,
        startY: event.clientY,
        moved: false,
        // Where the band was when the press came: a fine drag moves from here
        // instead of putting the band under the pointer.
        freq: ctx.value("b" + (band + 1) + "_freq"),
        gain: ctx.value("b" + (band + 1) + "_gain")
      };
    },
    move: function (ctx, drag, event) {
      if (!drag.moved) {
        if (Math.abs(event.clientX - drag.startX) +
            Math.abs(event.clientY - drag.startY) < 3) {
          return;
        }

        drag.moved = true;
      }

      var hero = event.currentTarget;
      var svg = hero.querySelector("svg");

      if (!svg) return;

      var rect = svg.getBoundingClientRect();

      if (!rect.width || !rect.height) return;

      var u = Math.min(1, Math.max(0, (event.clientX - rect.left) / rect.width));
      var v = Math.min(1, Math.max(0, (event.clientY - rect.top) / rect.height));
      var type = Math.round(ctx.value("b" + (drag.band + 1) + "_type") || 0);

      // Shift is the fine drag: the band moves a tenth of the pointer's travel from
      // where it started, so a narrow band can be placed without overshooting.
      if (event.shiftKey) {
        var du = (event.clientX - drag.startX) / rect.width * 0.1;
        var dv = (event.clientY - drag.startY) / rect.height * 0.1;
        var fine = Math.min(20000, Math.max(20, (drag.freq || 1000) * Math.pow(1000, du)));

        ctx.set("b" + (drag.band + 1) + "_freq", fine);

        if (type <= 2) {
          ctx.set(
            "b" + (drag.band + 1) + "_gain",
            Math.min(24, Math.max(-24, (drag.gain || 0) - dv * 48))
          );
        }

        event.preventDefault();
        return;
      }

      ctx.set("b" + (drag.band + 1) + "_freq", 20 * Math.pow(1000, u));

      // Only the bell and shelf bands have a gain; a high/low pass has none.
      if (type <= 2) ctx.set("b" + (drag.band + 1) + "_gain", 24 - v * 48);
      event.preventDefault();
    },
    up: function (ctx, drag) {
      ctx.set("band", drag.band);
    },
    wheel: function (ctx, event) {
      var band = eqBandAt(event);

      if (band < 0) return;

      event.preventDefault();
      ctx.set("band", band);

      var q = ctx.value("b" + (band + 1) + "_q") || 1;

      ctx.set("b" + (band + 1) + "_q", q * (event.deltaY < 0 ? 1.12 : 0.89));
    },
    double: function (ctx, event) {
      var band = eqBandAt(event);

      if (band < 0) return;

      ctx.set("band", band);
      ctx.set("b" + (band + 1) + "_gain", 0);
      event.preventDefault();
    }
  });

  registerHeroBehaviour("sampler", {
    actions: {
      load: function (ctx) { ctx.pick(); },
      slice: function (ctx) { ctx.action("slices"); },
      loop: function (ctx) { flip(ctx, "oscillator.loop"); },
      pingpong: function (ctx) { flip(ctx, "oscillator.pingpong"); },
      reverse: function (ctx) { flip(ctx, "oscillator.reverse"); },
      polarity: function (ctx) { flip(ctx, "oscillator.polarity"); }
    },
    down: function (ctx, event) {
      var sample = ctx.read();

      if (!sample || !sample.peaks || !sample.peaks.length) return null;

      var at = ctx.share(event);

      if (event.button === 2) {
        // The right button takes off the mark it lands on, and elsewhere draws
        // the loop region from press to release.
        if (ctx.near(at)) {
          ctx.action("slice-remove", { at: at });
          return null;
        }

        ctx.set("oscillator.loop_start", at);
        ctx.set("oscillator.loop_end", at);
        ctx.set("oscillator.loop", 1);
        return { kind: "loop", at: at };
      }

      if (event.button !== 0 || event.shiftKey) return null;

      if (Math.round(ctx.value("oscillator.mode") || 0) === 1 && ctx.near(at)) {
        return { kind: "mark", from: at, to: at };
      }

      // The left button sets where playback starts and drops the loop, the way
      // the sampler's own editor does.
      ctx.set("oscillator.start", at);
      ctx.set("oscillator.loop", 0);
      return { kind: "start" };
    },
    move: function (ctx, drag, event) {
      var at = ctx.share(event);

      if (drag.kind === "loop") {
        ctx.set("oscillator.loop_start", Math.min(drag.at, at));
        ctx.set("oscillator.loop_end", Math.max(drag.at, at));
      } else if (drag.kind === "start") {
        ctx.set("oscillator.start", at);
      } else if (drag.kind === "mark") {
        drag.to = at;
      }
    },
    up: function (ctx, drag) {
      if (drag.kind === "mark") {
        if (Math.abs(drag.to - drag.from) > 1e-4) {
          ctx.action("slice-move", { from: drag.from, to: drag.to });
        }

        return;
      }

      ctx.commit(drag.kind === "loop"
        ? ["oscillator.loop_start", "oscillator.loop_end", "oscillator.loop"]
        : ["oscillator.start", "oscillator.loop"]);
    },
    double: function (ctx, event) {
      var at = ctx.share(event);

      if (ctx.near(at)) return;

      ctx.action("slice-add", { at: at });
    },
    click: function (ctx) {
      var sample = ctx.read();

      // With nothing loaded, a click anywhere asks for a file; with a sample
      // up, the picture belongs to the gestures.
      if (!sample || !sample.peaks || !sample.peaks.length) ctx.pick();
    },
    menu: function (ctx) {
      var sample = ctx.read();

      return !!(sample && sample.peaks && sample.peaks.length);
    }
  });

  registerHeroBehaviour("shape", {
    actions: {
      prev: function (ctx) { ctx.frame(-1); },
      next: function (ctx) { ctx.frame(1); },
      delete: function (ctx) { ctx.action("delete", { frame: ctx.frameIndex() }); },
      mode: function (ctx) { ctx.cycleMode(); },
      reverse: function (ctx) { ctx.action("reverse"); },
      invert: function (ctx) { ctx.action("invert"); },
      dc: function (ctx) { ctx.action("dc"); },
      normalize: function (ctx) { ctx.action("normalize"); }
    },
    click: function (ctx) { ctx.pick(); }
  });

  registerHeroBehaviour("impulse", {
    down: function (ctx, event) {
      var ir = ctx.read();

      if (!ir || !ir.peaks || !ir.peaks.length) return null;

      var rect = event.currentTarget.getBoundingClientRect();

      // The dots ride the envelope's top corners, so only a press up there
      // starts a fade; a press on the body still asks for the picker.
      if ((event.clientY - rect.top) / rect.height > 0.4) return null;

      var at = ctx.share(event);
      var fadeIn = Math.min(1, Math.max(0, ir.fade_in || 0));
      var fadeOut = Math.min(1, Math.max(0, ir.fade_out || 0));

      if (Math.abs(at - fadeIn) < 0.03) return { kind: "in" };
      if (Math.abs(at - (1 - fadeOut)) < 0.03) return { kind: "out" };

      return null;
    },
    move: function (ctx, drag, event) {
      var at = ctx.share(event);
      var ir = ctx.read();

      if (drag.kind === "in") ir.fade_in = at;
      else ir.fade_out = 1 - at;

      ctx.repaint();
    },
    up: function (ctx) {
      ctx.commit(["fade_in", "fade_out"]);
    },
    click: function (ctx) { ctx.pick(); }
  });
