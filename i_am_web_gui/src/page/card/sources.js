// Part nine: the control-source pictures.
//
// The patch page's control nodes and the rack's modulation cards draw the same
// six sources: an LFO (points and presets), an envelope, a random walk and a
// table. The rack's drawing is the reference, so it lives here now and both
// pages call it; a change lands on the two pages at once.
(function () {
  "use strict";

  function node(tag, attributes) {
    var element = document.createElementNS("http://www.w3.org/2000/svg", tag);
    for (var key in attributes) element.setAttribute(key, attributes[key]);
    return element;
  }

  // A dot drawn as HTML: the picture is stretched over the card, and a circle
  // inside it would follow the card's aspect instead of staying round. The
  // viewBox is 300x70, so the percentages are x/3 and y/0.7.
  function pip(dots, x, y, radius, fill, ring) {
    var dot = document.createElement("div");

    dot.className = "hero-pip";
    dot.style.left = (x / 3).toFixed(2) + "%";
    dot.style.top = (y / 0.7).toFixed(2) + "%";
    dot.style.width = (radius * 2).toFixed(1) + "px";
    dot.style.height = (radius * 2).toFixed(1) + "px";

    if (fill) dot.style.background = fill;
    if (ring) dot.style.border = "1.4px solid " + ring;

    dots.appendChild(dot);
  }

  function bend(t, factor) {
    if (Math.abs(factor) < 0.0001) return t;
    return (Math.exp(t * factor) - 1) / (Math.exp(factor) - 1);
  }

  // The four LFO shapes, in -1..1.
  function shape(kind, phase) {
    switch (((kind % 4) + 4) % 4) {
      case 1: return 1 - 4 * Math.abs(phase - 0.5);
      case 2: return phase * 2 - 1;
      case 3: return phase < 0.5 ? 1 : -1;
      default: return Math.sin(phase * Math.PI * 2);
    }
  }

  function tableValue(kind, morph, phase) {
    var first = shape(kind, phase);
    var second = shape(kind + 1, phase);
    return first + (second - first) * Math.max(0, Math.min(1, morph));
  }

  // The points a shape preset starts from, the same table the rack's page uses.
  function preset(kind) {
    var raw;

    switch (((kind % 4) + 4) % 4) {
      case 1: raw = [[0, 0], [0.5, 1], [1, 0]]; break;
      case 2: raw = [[0, 0], [1, 1]]; break;
      case 3: raw = [[0, 1], [0.5, 1], [0.5001, 0], [1, 0]]; break;
      default: raw = [[0, 0.5], [0.25, 1], [0.5, 0.5], [0.75, 0], [1, 0.5]];
    }

    return raw.map(function (point) {
      return { time: point[0], value: point[1], bend: 0 };
    });
  }

  function baseline(svg, y) {
    svg.appendChild(node("line", {
      x1: 8, y1: y, x2: 292, y2: y,
      stroke: "#333", "stroke-width": "1", "stroke-dasharray": "2 4"
    }));
  }

  function lfoPreview(svg, data, dots) {
    var color = data.color || "#4c8dff";
    var handleColor = data.color || "#8fb8ff";
    var points = (data.points || []).slice().sort(function (a, b) { return a.time - b.time; });

    if (points.length < 2) {
      // A node that has no points yet still says which wave it plays.
      if (typeof data.shape === "number") {
        var wave = "";

        for (var index = 0; index <= 120; index++) {
          var t = index / 120;
          var value = shape(Math.round(data.shape), (t * 2) % 1);
          wave += (index ? " L " : "M ") + (8 + t * 284).toFixed(1) + " " +
            (60 - (value * 0.5 + 0.5) * 52).toFixed(1);
        }

        svg.appendChild(node("path", {
          d: wave, fill: "none", stroke: color, "stroke-width": 2,
          "stroke-linecap": "round", "vector-effect": "non-scaling-stroke"
        }));
      }

      return;
    }

    var path = "M " + (8 + points[0].time * 284).toFixed(1) + " " +
      (60 - points[0].value * 52).toFixed(1);
    var segments = data.segments || 14;

    for (var i = 1; i < points.length; i++) {
      var previous = points[i - 1];
      var point = points[i];

      for (var step = 1; step <= segments; step++) {
        var unit = step / segments;
        var time = previous.time + (point.time - previous.time) * unit;
        var between = previous.value +
          (point.value - previous.value) * bend(unit, previous.bend);
        path += " L " + (8 + time * 284).toFixed(1) + " " +
          (60 - between * 52).toFixed(1);
      }
    }

    svg.appendChild(node("path", {
      d: path, fill: "none", stroke: color, "stroke-width": 2,
      "stroke-linecap": "round", "vector-effect": "non-scaling-stroke"
    }));

    points.forEach(function (item) {
      var x = 8 + item.time * 284;
      var y = 60 - item.value * 52;

      if (dots) {
        pip(dots, x, y, 3.5, color, "#1a1a1a");
        return;
      }

      var circle = node("circle", {
        cx: x.toFixed(1),
        cy: y.toFixed(1),
        r: "3.5", fill: color, stroke: "#1a1a1a", "stroke-width": "1.5"
      });
      circle.style.pointerEvents = "none";
      svg.appendChild(circle);
    });

    for (var index = 0; index < points.length - 1; index++) {
      var start = points[index];
      var end = points[index + 1];
      var bendTime = (start.time + end.time) / 2;
      var bendValue = start.value + (end.value - start.value) * bend(0.5, start.bend);
      var bx = 8 + bendTime * 284;
      var by = 60 - bendValue * 52;

      if (dots) {
        pip(dots, bx, by, 2.6, null, handleColor);
        continue;
      }

      var handle = node("circle", {
        cx: bx.toFixed(1),
        cy: by.toFixed(1),
        r: "2.6", fill: "none", stroke: handleColor, "stroke-width": "1.4"
      });
      handle.style.pointerEvents = "none";
      svg.appendChild(handle);
    }
  }

  function tablePreview(svg, data) {
    var values = [];

    for (var index = 0; index < 120; index++) {
      values.push(tableValue(data.shape || 0, data.morph || 0, (index / 120 * 2) % 1));
    }

    baseline(svg, 32);

    var path = "";

    for (var i = 0; i < values.length; i++) {
      var x = 8 + i / (values.length - 1) * 284;
      var y = 32 - Math.max(-1, Math.min(1, values[i])) * 22;
      path += (i ? " L " : "M ") + x.toFixed(1) + " " + y.toFixed(1);
    }

    svg.appendChild(node("path", {
      d: path, fill: "none", stroke: "#c8e04a", "stroke-width": "2",
      "stroke-linecap": "round", "vector-effect": "non-scaling-stroke"
    }));
  }

  function drawPreview(svg, kind, data, dots) {
    svg.textContent = "";
    // The HTML dots are a second layer, so they have to be cleared with it.
    if (dots) dots.textContent = "";
    data = data || {};

    if (kind === "lfo") {
      baseline(svg, 60);
      lfoPreview(svg, data, dots);
    } else if (kind === "table") {
      tablePreview(svg, data);
    } else if (kind === "envelope") {
      baseline(svg, 58);
    } else if (kind === "random") {
      baseline(svg, 32);
    }
  }

  // How much of a time-carrying history the trace shows, in milliseconds. The
  // page used to take the last 120 points, which made the same span as this
  // only while every poll was exactly 20 ms apart.
  var TRACE_WINDOW = 2400;

  // A history point is a value, or {t, v} once the page records when the value
  // arrived. A timed history is drawn against its own clock, so a slow poll
  // makes a wider step instead of squeezing two samples into one pixel. The
  // line is straight between samples: a spline would round the corners the
  // samples have, and would overshoot when the intervals are uneven.
  function trace(svg, values, color, base, height) {
    if (!values || values.length < 2) return;

    var timed = typeof values[0] === "object" && values[0] !== null;
    var newest = timed ? values[values.length - 1].t : 0;
    var path = "";
    var at = function (index) {
      var i = Math.max(0, Math.min(values.length - 1, index));
      var point = values[i];
      var x = timed
        ? 8 + (1 - (newest - point.t) / TRACE_WINDOW) * 284
        : 8 + i / (values.length - 1) * 284;

      return {
        x: Math.max(8, Math.min(292, x)),
        y: base - (timed ? point.v : point) * height
      };
    };
    var first = at(0);
    path = "M " + first.x.toFixed(1) + " " + first.y.toFixed(2);

    for (var index = 0; index < values.length - 1; index++) {
      var next = at(index + 1);

      path += " L " + next.x.toFixed(1) + " " + next.y.toFixed(2);
    }

    svg.appendChild(node("path", {
      d: path, fill: "none", stroke: color, "stroke-width": "2",
      "stroke-linecap": "round", "vector-effect": "non-scaling-stroke"
    }));
  }

  function playhead(svg, x, y, color, fill, dots) {
    svg.appendChild(node("line", {
      x1: x, y1: 6, x2: x, y2: 60,
      stroke: color, "stroke-opacity": ".35", "stroke-width": "1"
    }));

    if (dots) {
      pip(dots, Number(x), Number(y), 3.5, fill, color);
      return;
    }

    svg.appendChild(node("circle", {
      cx: x, cy: y, r: "3.5", fill: fill, stroke: color, "stroke-width": "1.5"
    }));
  }

  // The live overlay: a playhead for the LFO and the table, a trace for the
  // others. clear defaults to true; pass false to draw over a static preview.
  function drawLive(svg, kind, live, id, clear, dots) {
    if (clear !== false) {
      svg.textContent = "";
      if (dots) dots.textContent = "";
    }

    var phase = live.phases[id] || 0;
    var value = live.values[id] || 0;

    if (kind === "lfo") {
      playhead(svg, (8 + phase * 284).toFixed(1),
        (60 - ((value + 1) / 2) * 52).toFixed(1), "#4c8dff", "#dce9ff", dots);
    } else if (kind === "table") {
      playhead(svg, (8 + phase * 284).toFixed(1),
        (32 - value * 22).toFixed(1), "#c8e04a", "#f4ffd0", dots);
    } else {
      var history = live.history[id] || [];

      if (history.length && history[0] && typeof history[0] === "object") {
        // Timed points: keep only the part of the history inside the window,
        // so an old point cannot land left of the card.
        var newest = history[history.length - 1].t;
        var start = 0;

        while (start < history.length && history[start].t < newest - TRACE_WINDOW) start++;
        history = history.slice(start);
      } else {
        history = history.slice(-120);
      }

      if (kind === "envelope") trace(svg, history, "#35c77a", 58, 48);
      else if (kind === "random") trace(svg, history, "#ff9a3d", 32, 22);
      else if (kind === "pitch") trace(svg, history, "#b08cff", 58, 48);
      else if (kind === "midi") trace(svg, history, "#ff6f91", 58, 48);
    }
  }

  // Editing the LFO on the card itself: drag a point or its bend ring,
  // double-click to add one, right-click to remove one. The rack's modulation
  // card and the patch page's LFO card share this editor.
  function editLfo(svg, holder, options) {
    options = options || {};

    if (!holder.points || holder.points.length < 2) {
      holder.points = preset(typeof options.shape === "number" ? Math.round(options.shape) : 0);
    }

    function eventPoint(event) {
      var x;
      var y;
      var matrix = svg.getScreenCTM();

      if (matrix) {
        var pointer = svg.createSVGPoint();
        pointer.x = event.clientX;
        pointer.y = event.clientY;
        var local = pointer.matrixTransform(matrix.inverse());
        x = local.x;
        y = local.y;
      } else {
        var rect = svg.getBoundingClientRect();
        x = (event.clientX - rect.left) / rect.width * 300;
        y = (event.clientY - rect.top) / rect.height * 64;
      }

      return {
        x: x,
        y: y,
        time: Math.max(0, Math.min(1, (x - 8) / 284)),
        value: Math.max(0, Math.min(1, (60 - y) / 52))
      };
    }

    function pointAt(event) {
      var position = eventPoint(event);

      for (var i = 0; i < holder.points.length; i++) {
        var point = holder.points[i];
        var x = 8 + point.time * 284;
        var y = 60 - point.value * 52;

        if (Math.abs(x - position.x) < 9 && Math.abs(y - position.y) < 9) {
          return point;
        }
      }

      return null;
    }

    function bendAt(event) {
      var position = eventPoint(event);
      var points = holder.points.slice().sort(function (a, b) {
        return a.time - b.time;
      });

      for (var i = 0; i < points.length - 1; i++) {
        var point = points[i];
        var next = points[i + 1];
        var time = (point.time + next.time) / 2;
        var value = point.value + (next.value - point.value) * bend(0.5, point.bend);
        var x = 8 + time * 284;
        var y = 60 - value * 52;

        if (Math.abs(x - position.x) < 8 && Math.abs(y - position.y) < 8) {
          return point;
        }
      }

      return null;
    }

    function paint() {
      if (options.paint) options.paint(holder.points);
      else drawPreview(svg, "lfo", { points: holder.points }, options.dots);
    }

    var drag = null;

    function commit() {
      if (options.commit) options.commit(holder.points);
    }

    function removePoint(point) {
      if (holder.points.length <= 2) return;

      var index = holder.points.indexOf(point);
      if (index >= 0) holder.points.splice(index, 1);
      paint();
      commit();
    }

    svg.onpointerdown = function (event) {
      if (event.button !== 0) return;

      // The card sits in the patch canvas: a drag on a point is not a canvas
      // gesture, even if the canvas ever sees the event.
      event.stopPropagation();

      var point = pointAt(event);

      if (point) {
        drag = { kind: "point", point: point };
      } else if (options.bends !== false) {
        // A straight mapping has no bend handles: only its points move.
        var bendPoint = bendAt(event);

        if (bendPoint) {
          var sorted = holder.points.slice().sort(function (a, b) {
            return a.time - b.time;
          });
          var index = sorted.indexOf(bendPoint);
          var next = sorted[index + 1];
          var position = eventPoint(event);

          drag = {
            kind: "bend",
            point: bendPoint,
            startValue: position.value,
            startBend: bendPoint.bend,
            direction: next && next.value > bendPoint.value ? -1 : 1
          };
        }
      }

      if (!drag) return;

      try {
        svg.setPointerCapture(event.pointerId);
      } catch (error) {
        // A synthetic or already-lost pointer has nothing to capture.
      }

      // The window owns the gesture: a capture can be refused inside a
      // foreignObject, and a lost pointerup would leave the points uncommitted.
      window.addEventListener("pointermove", onMove);
      window.addEventListener("pointerup", finish);
      window.addEventListener("pointercancel", finish);

      if (options.onStart) options.onStart();
      paint();
      event.preventDefault();
    };

    function onMove(event) {
      if (!drag) return;

      var position = eventPoint(event);

      if (drag.kind === "point") {
        drag.point.time = position.time;
        drag.point.value = position.value;
        holder.points.sort(function (a, b) { return a.time - b.time; });
      } else {
        var delta = position.value - drag.startValue;
        drag.point.bend = Math.max(-10, Math.min(10,
          drag.startBend + delta * 20 * drag.direction));
      }

      paint();
      event.preventDefault();
    }

    function finish(event) {
      if (!drag) return;
      drag = null;
      window.removeEventListener("pointermove", onMove);
      window.removeEventListener("pointerup", finish);
      window.removeEventListener("pointercancel", finish);
      commit();
      if (options.onEnd) options.onEnd();
      event.preventDefault();
    }

    svg.ondblclick = function (event) {
      // Adding is deliberate: a single click starts a drag on a point or a
      // bend ring, while an empty double-click appends a new point.
      if (pointAt(event)) return;

      var position = eventPoint(event);
      if (position.x < 8 || position.x > 292 || position.y < 8 || position.y > 60) return;
      holder.points.push({ time: position.time, value: position.value, bend: 0 });
      holder.points.sort(function (a, b) { return a.time - b.time; });
      paint();
      commit();
      event.preventDefault();
    };
    svg.oncontextmenu = function (event) {
      var point = pointAt(event);

      if (!point) return;

      // The card sits on a canvas with a menu of its own: taking a point off
      // must not also open that menu.
      event.preventDefault();
      event.stopPropagation();
      removePoint(point);
    };

    paint();
  }
  window.IAmCard = window.IAmCard || {};
  window.IAmCard.sources = {
    shape: shape,
    preset: preset,
    tableValue: tableValue,
    bend: bend,
    drawPreview: drawPreview,
    editLfo: editLfo,
    drawLive: drawLive
  };
})();
