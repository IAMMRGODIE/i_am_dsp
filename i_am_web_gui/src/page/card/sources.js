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

    if (data.handles === false) return;

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

  // The shared curve editor: pointer hit-testing, bend rings, add/remove and
  // gesture capture. The axis and the point constraints are supplied by the
  // caller, so an LFO, an envelope and a pitch curve share one editor.
  function attachCurveEditor(svg, holder, options) {
    options = options || {};

    var plot = options.plot || {};
    var left = typeof plot.left === "number" ? plot.left : 8;
    var right = typeof plot.right === "number" ? plot.right : 292;
    var top = typeof plot.top === "number" ? plot.top : 8;
    var base = typeof plot.base === "number" ? plot.base : 60;
    var width = right - left;
    var height = base - top;
    var minPoints = typeof options.minPoints === "number" ? options.minPoints : 2;
    var bends = options.bends !== false;
    var hit = options.hit || 9;
    var bendHit = options.bendHit || 8;
    var bendScale = typeof options.bendScale === "number" ? options.bendScale : 20;
    var bendLimit = typeof options.bendLimit === "number" ? options.bendLimit : 10;

    if (!holder.points || holder.points.length < 2) {
      holder.points = options.defaultPoints
        ? options.defaultPoints()
        : preset(typeof options.shape === "number" ? Math.round(options.shape) : 0);
    }

    function toPoint(unit) {
      if (options.fromLocal) return options.fromLocal(unit);
      return {
        time: Math.max(0, Math.min(1, (unit.x - left) / width)),
        value: Math.max(0, Math.min(1, (base - unit.y) / height)),
      };
    }

    function pointPosition(point) {
      if (options.pointPosition) return options.pointPosition(point);
      return { x: left + point.time * width, y: base - point.value * height };
    }

    function bendPosition(point, next) {
      if (options.bendPosition) return options.bendPosition(point, next);
      var time = (point.time + next.time) / 2;
      var value = point.value + (next.value - point.value) * bend(0.5, point.bend);
      return { x: left + time * width, y: base - value * height };
    }

    function sortPoints(points) {
      if (options.sort) options.sort(points);
      else points.sort(function (a, b) { return a.time - b.time; });
      return points;
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

      var point = toPoint({ x: x, y: y });
      point.x = x;
      point.y = y;
      return point;
    }

    function pointAt(event) {
      var position = eventPoint(event);

      for (var i = 0; i < holder.points.length; i++) {
        var at = pointPosition(holder.points[i]);

        if (Math.abs(at.x - position.x) < hit && Math.abs(at.y - position.y) < hit) {
          return holder.points[i];
        }
      }

      return null;
    }

    function bendAt(event) {
      if (!bends) return null;

      var position = eventPoint(event);
      var points = sortPoints(holder.points.slice());

      for (var i = 0; i < points.length - 1; i++) {
        var at = bendPosition(points[i], points[i + 1]);

        if (Math.abs(at.x - position.x) < bendHit && Math.abs(at.y - position.y) < bendHit) {
          return points[i];
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
      if (holder.points.length <= minPoints) return;

      var index = holder.points.indexOf(point);
      if (index >= 0) holder.points.splice(index, 1);
      paint();
      commit();
    }

    svg.onpointerdown = function (event) {
      if (event.button !== 0) return;

      if (options.stopPropagation !== false) event.stopPropagation();

      var point = pointAt(event);

      if (point) {
        drag = { kind: "point", point: point };
      } else if (bends) {
        var bendPoint = bendAt(event);

        if (bendPoint) {
          var sorted = sortPoints(holder.points.slice());
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
        if (options.movePoint) options.movePoint(drag.point, position);
        else {
          drag.point.time = position.time;
          drag.point.value = position.value;
        }
        sortPoints(holder.points);
      } else {
        var delta = position.value - drag.startValue;
        drag.point.bend = Math.max(-bendLimit, Math.min(bendLimit,
          drag.startBend + delta * bendScale * drag.direction));
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
      if (pointAt(event)) return;

      var position = eventPoint(event);
      if (position.x < left || position.x > right || position.y < top || position.y > base) return;

      if (options.addPoint) options.addPoint(position, holder.points);
      else holder.points.push({ time: position.time, value: position.value, bend: 0 });

      sortPoints(holder.points);
      paint();
      commit();
      event.preventDefault();
    };

    svg.oncontextmenu = function (event) {
      var point = pointAt(event);

      if (!point) return;
      if (options.removePoint && options.removePoint(point, holder.points) === false) return;

      event.preventDefault();
      event.stopPropagation();
      removePoint(point);
    };

    paint();
    return { paint: paint, pointAt: pointAt, eventPoint: eventPoint };
  }

  // The 0..1 LFO editor used by the cards, the patch page and the rack.
  function editLfo(svg, holder, options) {
    return attachCurveEditor(svg, holder, options);
  }

  window.IAmCard = window.IAmCard || {};
  window.IAmCard.sources = {
    shape: shape,
    preset: preset,
    tableValue: tableValue,
    bend: bend,
    drawPreview: drawPreview,
    editLfo: editLfo,
    curveEditor: attachCurveEditor,
    drawLive: drawLive
  };
})();
