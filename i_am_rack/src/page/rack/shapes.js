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
        var value = point.value + (next.value - point.value) * IAmCard.sources.bend(0.5, point.bend);
        var x = 8 + time * 284;
        var y = 60 - value * 52;

        if (Math.abs(x - position.x) < 8 && Math.abs(y - position.y) < 8) {
          return point;
        }
      }

      return null;
    }

    function paint() {
      IAmCard.sources.drawPreview(svg, "lfo", { points: lfo.points });
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
    IAmCard.sources.drawPreview(svg, "envelope", {});
  }

  function drawRandom(svg) {
    IAmCard.sources.drawPreview(svg, "random", {});
  }

  function drawTable(svg, table) {
    IAmCard.sources.drawPreview(svg, "table", table);
  }

  

  function drawLiveEnvelope(svg, live, id) {
    IAmCard.sources.drawLive(svg, "envelope", live, id);
  }

  function drawLiveRandom(svg, live, id) {
    IAmCard.sources.drawLive(svg, "random", live, id);
  }

  function drawLivePitch(svg, live, id) {
    IAmCard.sources.drawLive(svg, "pitch", live, id);
  }

  function drawLiveMidi(svg, live, id) {
    IAmCard.sources.drawLive(svg, "midi", live, id);
  }

  function drawLiveLfo(svg, live, id) {
    IAmCard.sources.drawLive(svg, "lfo", live, id);
  }

  function drawLiveTable(svg, live, id) {
    IAmCard.sources.drawLive(svg, "table", live, id);
  }
