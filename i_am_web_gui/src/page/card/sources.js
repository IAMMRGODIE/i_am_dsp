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

  function lfoPreview(svg, data) {
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
          d: wave, fill: "none", stroke: "#4c8dff", "stroke-width": 2,
          "stroke-linecap": "round", "vector-effect": "non-scaling-stroke"
        }));
      }

      return;
    }

    var path = "M " + (8 + points[0].time * 284).toFixed(1) + " " +
      (60 - points[0].value * 52).toFixed(1);

    for (var i = 1; i < points.length; i++) {
      var previous = points[i - 1];
      var point = points[i];

      for (var step = 1; step <= 14; step++) {
        var unit = step / 14;
        var time = previous.time + (point.time - previous.time) * unit;
        var between = previous.value +
          (point.value - previous.value) * bend(unit, previous.bend);
        path += " L " + (8 + time * 284).toFixed(1) + " " +
          (60 - between * 52).toFixed(1);
      }
    }

    svg.appendChild(node("path", {
      d: path, fill: "none", stroke: "#4c8dff", "stroke-width": 2,
      "stroke-linecap": "round", "vector-effect": "non-scaling-stroke"
    }));

    points.forEach(function (item) {
      var circle = node("circle", {
        cx: (8 + item.time * 284).toFixed(1),
        cy: (60 - item.value * 52).toFixed(1),
        r: "3.5", fill: "#4c8dff", stroke: "#1a1a1a", "stroke-width": "1.5"
      });
      circle.style.pointerEvents = "none";
      svg.appendChild(circle);
    });

    for (var index = 0; index < points.length - 1; index++) {
      var start = points[index];
      var end = points[index + 1];
      var bendTime = (start.time + end.time) / 2;
      var bendValue = start.value + (end.value - start.value) * bend(0.5, start.bend);
      var handle = node("circle", {
        cx: (8 + bendTime * 284).toFixed(1),
        cy: (60 - bendValue * 52).toFixed(1),
        r: "2.6", fill: "none", stroke: "#8fb8ff", "stroke-width": "1.4"
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

  function drawPreview(svg, kind, data) {
    svg.textContent = "";
    data = data || {};

    if (kind === "lfo") {
      baseline(svg, 60);
      lfoPreview(svg, data);
    } else if (kind === "table") {
      tablePreview(svg, data);
    } else if (kind === "envelope") {
      baseline(svg, 58);
    } else if (kind === "random") {
      baseline(svg, 32);
    }
  }

  function trace(svg, values, color, base, height) {
    if (!values || values.length < 2) return;

    var path = "";

    for (var index = 0; index < values.length; index++) {
      var x = 8 + index / (values.length - 1) * 284;
      var y = base - values[index] * height;
      path += (index ? " L " : "M ") + x.toFixed(1) + " " + y.toFixed(1);
    }

    svg.appendChild(node("path", {
      d: path, fill: "none", stroke: color, "stroke-width": "2",
      "stroke-linecap": "round", "vector-effect": "non-scaling-stroke"
    }));
  }

  function playhead(svg, x, y, color, fill) {
    svg.appendChild(node("line", {
      x1: x, y1: 6, x2: x, y2: 60,
      stroke: color, "stroke-opacity": ".35", "stroke-width": "1"
    }));
    svg.appendChild(node("circle", {
      cx: x, cy: y, r: "3.5", fill: fill, stroke: color, "stroke-width": "1.5"
    }));
  }

  // The live overlay: a playhead for the LFO and the table, a trace for the
  // others. clear defaults to true; pass false to draw over a static preview.
  function drawLive(svg, kind, live, id, clear) {
    if (clear !== false) svg.textContent = "";

    var phase = live.phases[id] || 0;
    var value = live.values[id] || 0;

    if (kind === "lfo") {
      playhead(svg, (8 + phase * 284).toFixed(1),
        (60 - ((value + 1) / 2) * 52).toFixed(1), "#4c8dff", "#dce9ff");
    } else if (kind === "table") {
      playhead(svg, (8 + phase * 284).toFixed(1),
        (32 - value * 22).toFixed(1), "#c8e04a", "#f4ffd0");
    } else {
      var history = (live.history[id] || []).slice(-120);

      if (kind === "envelope") trace(svg, history, "#35c77a", 58, 48);
      else if (kind === "random") trace(svg, history, "#ff9a3d", 32, 22);
      else if (kind === "pitch") trace(svg, history, "#b08cff", 58, 48);
      else if (kind === "midi") trace(svg, history, "#ff6f91", 58, 48);
    }
  }

  window.IAmCard = window.IAmCard || {};
  window.IAmCard.sources = {
    shape: shape,
    preset: preset,
    tableValue: tableValue,
    bend: bend,
    drawPreview: drawPreview,
    drawLive: drawLive
  };
})();
