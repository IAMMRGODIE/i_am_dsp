// Part five: the curve, LFO and table editors and their live traces.
  function curvePointsFromJson(points) {
    if (Array.isArray(points) && points.length >= 2) {
      return points.map(function (point) {
        return { x: Number(point[0]), y: Number(point[1]) };
      });
    }

    return [{ x: 0, y: 0 }, { x: 1, y: 1 }];
  }

  function serializeCurve(points) {
    return points.map(function (point) {
      return point.x.toFixed(4) + ":" + point.y.toFixed(4);
    }).join(";");
  }

  function drawMidiCurve(svg, points, save) {
    function eventPoint(event) {
      var rect = svg.getBoundingClientRect();
      return {
        x: Math.max(0, Math.min(1, (event.clientX - rect.left) / rect.width)),
        y: Math.max(0, Math.min(1, 1 - (event.clientY - rect.top) / rect.height))
      };
    }

    function at(event) {
      var position = eventPoint(event);

      for (var index = 0; index < points.length; index++) {
        if (Math.abs(points[index].x - position.x) < 0.035 &&
            Math.abs(points[index].y - position.y) < 0.12) {
          return points[index];
        }
      }

      return null;
    }

    function paint() {
      svg.textContent = "";

      for (var line = 0; line <= 4; line++) {
        svg.appendChild(svgNode("line", {
          x1: 8, y1: 10 + line * 11, x2: 292, y2: 10 + line * 11,
          stroke: "#2a2a2a", "stroke-width": "1"
        }));
      }

      var sorted = points.slice().sort(function (a, b) { return a.x - b.x; });
      var path = "";

      sorted.forEach(function (point, index) {
        path += (index ? " L " : "M ") + (8 + point.x * 284).toFixed(1) +
          " " + (54 - point.y * 44).toFixed(1);
      });

      svg.appendChild(svgNode("path", {
        d: path, fill: "none", stroke: "#ff6f91", "stroke-width": "2",
        "stroke-linecap": "round", "vector-effect": "non-scaling-stroke"
      }));

      sorted.forEach(function (point) {
        var circle = svgNode("circle", {
          cx: (8 + point.x * 284).toFixed(1),
          cy: (54 - point.y * 44).toFixed(1),
          r: "3.2", fill: "#ff6f91", stroke: "#1a1a1a", "stroke-width": "1.4"
        });
        circle.style.pointerEvents = "none";
        svg.appendChild(circle);
      });
    }

    var drag = null;

    function commit() {
      save(points);
    }

    svg.onpointerdown = function (event) {
      if (event.button !== 0) return;

      var point = at(event);

      if (!point) {
        var position = eventPoint(event);
        point = { x: position.x, y: position.y };
        points.push(point);
        points.sort(function (a, b) { return a.x - b.x; });
      }

      drag = { point: point };
      svg.setPointerCapture(event.pointerId);
      paint();
      event.preventDefault();
    };

    svg.onpointermove = function (event) {
      if (!drag) return;
      var position = eventPoint(event);
      drag.point.x = position.x;
      drag.point.y = position.y;
      points.sort(function (a, b) { return a.x - b.x; });
      paint();
      event.preventDefault();
    };

    function finish(event) {
      if (!drag) return;
      drag = null;
      commit();
      event.preventDefault();
    }

    svg.onpointerup = finish;
    svg.onpointercancel = finish;

    function remove(event) {
      var point = at(event);
      if (!point || points.length <= 2) return;
      var index = points.indexOf(point);
      if (index >= 0) points.splice(index, 1);
      paint();
      commit();
      event.preventDefault();
    }

    svg.ondblclick = remove;
    svg.oncontextmenu = remove;

    paint();
  }

  function serializeLfo(points) {
    return points.map(function (point) {
      return point.time.toFixed(4) + ":" + point.value.toFixed(4) +
        ":" + point.bend.toFixed(4);
    }).join(";");
  }

  function lfoBend(t, bend) {
    if (Math.abs(bend) < 0.0001) return t;
    return (Math.exp(t * bend) - 1) / (Math.exp(bend) - 1);
  }

  function drawLfo(svg, lfo, id) {
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

      for (var i = 0; i < lfo.points.length; i++) {
        var point = lfo.points[i];
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
      var points = lfo.points.slice().sort(function (a, b) {
        return a.time - b.time;
      });

      for (var i = 0; i < points.length - 1; i++) {
        var point = points[i];
        var next = points[i + 1];
        var time = (point.time + next.time) / 2;
        var value = point.value + (next.value - point.value) * lfoBend(0.5, point.bend);
        var x = 8 + time * 284;
        var y = 60 - value * 52;

        if (Math.abs(x - position.x) < 8 && Math.abs(y - position.y) < 8) {
          return point;
        }
      }

      return null;
    }

    function paint() {
      svg.textContent = "";

      svg.appendChild(svgNode("line", {
        x1: 8, y1: 60, x2: 292, y2: 60,
        stroke: "#333", "stroke-width": "1", "stroke-dasharray": "2 4"
      }));

      var points = lfo.points.slice().sort(function (a, b) {
        return a.time - b.time;
      });
      var path = "";

      for (var i = 0; i < points.length; i++) {
        if (i === 0) {
          path = "M " + (8 + points[i].time * 284).toFixed(1) + " " +
            (60 - points[i].value * 52).toFixed(1);
          continue;
        }

        var previous = points[i - 1];
        var point = points[i];

        for (var step = 1; step <= 14; step++) {
          var unit = step / 14;
          var time = previous.time + (point.time - previous.time) * unit;
          var value = previous.value +
            (point.value - previous.value) * lfoBend(unit, previous.bend);
          path += " L " + (8 + time * 284).toFixed(1) + " " +
            (60 - value * 52).toFixed(1);
        }
      }

      if (points.length > 1) {
        svg.appendChild(svgNode("path", {
          d: path, fill: "none", stroke: "#4c8dff", "stroke-width": "2",
          "stroke-linecap": "round", "vector-effect": "non-scaling-stroke"
        }));
      }

      points.forEach(function (point) {
        var circle = svgNode("circle", {
          cx: (8 + point.time * 284).toFixed(1),
          cy: (60 - point.value * 52).toFixed(1),
          r: "3.5", fill: "#4c8dff", stroke: "#1a1a1a", "stroke-width": "1.5"
        });
        circle.style.pointerEvents = "none";
        svg.appendChild(circle);
      });

      for (var bendIndex = 0; bendIndex < points.length - 1; bendIndex++) {
        var start = points[bendIndex];
        var end = points[bendIndex + 1];
        var bendTime = (start.time + end.time) / 2;
        var bendValue = start.value +
          (end.value - start.value) * lfoBend(0.5, start.bend);
        var handle = svgNode("circle", {
          cx: (8 + bendTime * 284).toFixed(1),
          cy: (60 - bendValue * 52).toFixed(1),
          r: "2.6", fill: "none", stroke: "#8fb8ff", "stroke-width": "1.4"
        });
        handle.style.pointerEvents = "none";
        svg.appendChild(handle);
      }
    }

    var drag = null;

    function commit() {
      sendNow("source." + id + ".points", serializeLfo(lfo.points)).then(load);
    }

    function removePoint(point) {
      if (lfo.points.length <= 2) return;

      var index = lfo.points.indexOf(point);
      if (index >= 0) lfo.points.splice(index, 1);
      paint();
      commit();
    }

    svg.onpointerdown = function (event) {
      if (event.button !== 0) return;

      var point = pointAt(event);

      if (point) {
        drag = { kind: "point", point: point };
      } else {
        var bend = bendAt(event);

        if (bend) {
          var sorted = lfo.points.slice().sort(function (a, b) {
            return a.time - b.time;
          });
          var index = sorted.indexOf(bend);
          var next = sorted[index + 1];
          var position = eventPoint(event);

          drag = {
            kind: "bend",
            point: bend,
            startValue: position.value,
            startBend: bend.bend,
            direction: next && next.value > bend.value ? -1 : 1
          };
        }
      }

      if (!drag) return;

      svg.setPointerCapture(event.pointerId);
      paint();
      event.preventDefault();
    };

    svg.onpointermove = function (event) {
      if (!drag) return;

      var position = eventPoint(event);

      if (drag.kind === "point") {
        drag.point.time = position.time;
        drag.point.value = position.value;
        lfo.points.sort(function (a, b) { return a.time - b.time; });
      } else {
        var delta = position.value - drag.startValue;
        drag.point.bend = Math.max(-10, Math.min(10,
          drag.startBend + delta * 20 * drag.direction));
      }

      paint();
      event.preventDefault();
    };

    function finish(event) {
      if (!drag) return;
      drag = null;
      commit();
      event.preventDefault();
    }

    svg.onpointerup = finish;
    svg.onpointercancel = finish;
    svg.ondblclick = function (event) {
      // Adding is deliberate: a single click starts a drag on a point or bend,
      // while an empty double-click appends a new point. Right-click still
      // removes an existing point.
      if (pointAt(event)) return;

      var position = eventPoint(event);
      if (position.x < 8 || position.x > 292 || position.y < 8 || position.y > 60) return;
      var point = { time: position.time, value: position.value, bend: 0 };
      lfo.points.push(point);
      lfo.points.sort(function (a, b) { return a.time - b.time; });
      paint();
      commit();
      event.preventDefault();
    };
    svg.oncontextmenu = function (event) {
      var point = pointAt(event);
      if (point) {
        removePoint(point);
        event.preventDefault();
      }
    };

    paint();
  }

  function drawEnvelope(svg) {
    svg.textContent = "";
    svg.appendChild(svgNode("line", {
      x1: 8, y1: 58, x2: 292, y2: 58,
      stroke: "#333", "stroke-width": "1", "stroke-dasharray": "2 4"
    }));
  }

  function drawRandom(svg) {
    svg.textContent = "";
    svg.appendChild(svgNode("line", {
      x1: 8, y1: 32, x2: 292, y2: 32,
      stroke: "#333", "stroke-width": "1", "stroke-dasharray": "2 4"
    }));
  }

  function drawTable(svg, table) {
    var values = [];

    for (var i = 0; i < 120; i++) {
      values.push(tableValue(table.shape, table.morph, (i / 120 * 2) % 1));
    }

    paintCurve(svg, values, "#c8e04a");
  }

  function liveTrace(svg, values, color, base, height) {
    svg.textContent = "";
    if (values.length < 2) return;

    var path = "";

    for (var i = 0; i < values.length; i++) {
      var x = 8 + i / (values.length - 1) * 284;
      var y = base - values[i] * height;
      path += (i ? " L " : "M ") + x.toFixed(1) + " " + y.toFixed(1);
    }

    svg.appendChild(svgNode("path", {
      d: path, fill: "none", stroke: color, "stroke-width": "2",
      "stroke-linecap": "round", "vector-effect": "non-scaling-stroke"
    }));
  }

  function drawLiveEnvelope(svg, live, id) {
    liveTrace(svg, (live.history[id] || []).slice(-120), "#35c77a", 58, 48);
  }

  function drawLiveRandom(svg, live, id) {
    liveTrace(svg, (live.history[id] || []).slice(-120), "#ff9a3d", 32, 22);
  }

  function drawLivePitch(svg, live, id) {
    liveTrace(svg, (live.history[id] || []).slice(-120), "#b08cff", 58, 48);
  }

  function drawLiveMidi(svg, live, id) {
    liveTrace(svg, (live.history[id] || []).slice(-120), "#ff6f91", 58, 48);
  }

  function drawLiveLfo(svg, live, id) {
    svg.textContent = "";

    var phase = live.phases[id] || 0;
    var value = live.values[id] || 0;
    var x = 8 + phase * 284;
    var y = 60 - ((value + 1) / 2) * 52;

    svg.appendChild(svgNode("line", {
      x1: x.toFixed(1), y1: 6, x2: x.toFixed(1), y2: 60,
      stroke: "#4c8dff", "stroke-opacity": ".35", "stroke-width": "1"
    }));
    svg.appendChild(svgNode("circle", {
      cx: x.toFixed(1), cy: y.toFixed(1), r: "3.5",
      fill: "#dce9ff", stroke: "#4c8dff", "stroke-width": "1.5"
    }));
  }

  function drawLiveTable(svg, live, id) {
    svg.textContent = "";

    var phase = live.phases[id] || 0;
    var value = live.values[id] || 0;
    var x = 8 + phase * 284;
    var y = 32 - value * 22;

    svg.appendChild(svgNode("line", {
      x1: x.toFixed(1), y1: 6, x2: x.toFixed(1), y2: 60,
      stroke: "#c8e04a", "stroke-opacity": ".35", "stroke-width": "1"
    }));
    svg.appendChild(svgNode("circle", {
      cx: x.toFixed(1), cy: y.toFixed(1), r: "3.5",
      fill: "#f4ffd0", stroke: "#c8e04a", "stroke-width": "1.5"
    }));
  }
