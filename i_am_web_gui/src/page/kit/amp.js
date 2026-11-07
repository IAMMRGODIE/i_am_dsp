  // Part five-and-a-half: the amplitude envelope editor.
  //
  // Each drum's page owns one curve. The x axis is the share of the drum's
  // decay and the y axis is the level. The gestures match the fall editor: drag
  // a dot, drag a ring for the bend, double click to add, right click to remove.

  function ampDrumOf(hero) {
    return Number(hero.dataset.ampDrum || 0);
  }

  function ampCurveOf(hero) {
    return ampCurves[ampDrumOf(hero)] || ampCurves[0];
  }

  function ampMap(hero) {
    var svg = hero.querySelector("svg");
    if (!svg) return null;

    var rect = svg.getBoundingClientRect();
    if (!rect.width || !rect.height) return null;

    var scale = Math.min(rect.width / 520, rect.height / 160);

    return {
      scale: scale,
      left: rect.left + (rect.width - 520 * scale) / 2,
      top: rect.top + (rect.height - 160 * scale) / 2
    };
  }

  function ampX(time) {
    return 26 + clamp(time, 0, 1) * 468;
  }

  function ampLevel(value) {
    return 132 - clamp(value, 0, 1) * 112;
  }

  function ampSpanOf(hero) {
    var points = ampCurveOf(hero);
    var last = points.length ? points[points.length - 1][0] : 1;

    // A short snare wires envelope still gets a real second of room to grow
    // into, the cymbal gets ten (a real crash rings that long), and the pitch
    // keeps its own, much shorter, axis.
    var drum = ampDrumOf(hero);

    if (drum === 8) return Math.max(0.03, last * 1.1);
    if (drum === 5) return Math.max(10.0, last * 1.1);

    return Math.max(1.0, last * 1.1);
  }

  function ampPointAt(hero, event) {
    var map = ampMap(hero);
    if (!map) return null;

    var span = ampSpanOf(hero);

    return {
      time: clamp(((event.clientX - map.left) / map.scale - 26) / 468, 0, 1) * span,
      value: clamp((132 - (event.clientY - map.top) / map.scale) / 112, 0, 1)
    };
  }

  var ampSentAt = 0;

  function sendAmp(hero) {
    var points = ampCurveOf(hero);
    var body = points
      .map(function (point) {
        return point[0].toFixed(4) + ":" + point[1].toFixed(4) + ":" + (point[2] || 0).toFixed(3);
      })
      .join(";");

    ampSentAt = Date.now();

    fetch("/amp?drum=" + ampDrumOf(hero), { method: "POST", body: body })
      .then(function (response) {
        return response.ok ? response.json() : null;
      })
      .catch(function () {});
  }

  function ampPointerDown(hero, event) {
    if (event.button !== 0) return;

    var points = ampCurveOf(hero);
    if (!points.length) return;

    var map = ampMap(hero);
    if (!map) return;

    var bend = event.target.closest ? event.target.closest("[data-amp-bend]") : null;
    var point = event.target.closest ? event.target.closest("[data-amp-point]") : null;

    if (bend) {
      var index = Number(bend.getAttribute("data-amp-bend"));
      var left = points[index];
      var right = points[index + 1];
      if (!left || !right) return;

      ampDrag = {
        kind: "bend",
        hero: hero,
        index: index,
        pointerId: event.pointerId,
        startY: event.clientY,
        scale: map.scale || 1,
        startBend: left[2] || 0,
        direction: right[1] > left[1] ? -1 : 1
      };
      try { hero.setPointerCapture(event.pointerId); } catch (error) {}
      event.preventDefault();
      return;
    }

    if (!point) return;

    ampDrag = {
      kind: "point",
      hero: hero,
      index: Number(point.getAttribute("data-amp-point")),
      pointerId: event.pointerId
    };
    try { hero.setPointerCapture(event.pointerId); } catch (error) {}
    event.preventDefault();
  }

  function ampPointerMove(hero, event) {
    if (!ampDrag || event.pointerId !== ampDrag.pointerId) return;

    var points = ampCurveOf(ampDrag.hero);

    if (ampDrag.kind === "point") {
      var position = ampPointAt(ampDrag.hero, event);
      if (!position) return;

      var point = points[ampDrag.index];
      if (!point) return;

      var lower = ampDrag.index === 0 ? 0 : points[ampDrag.index - 1][0] + 0.001;
      var upper = ampDrag.index === points.length - 1 ? 1 : points[ampDrag.index + 1][0] - 0.001;

      point[0] = clamp(position.time, lower, Math.max(lower, upper));
      point[1] = clamp(position.value, 0, 1) * (ampDrumOf(hero) === 8 ? 10 : 1);
      markDirty();
      event.preventDefault();
      return;
    }

    var travel = (ampDrag.startY - event.clientY) / Math.max(ampDrag.scale, 1e-6);
    points[ampDrag.index][2] = clamp(ampDrag.startBend + travel * 0.09 * ampDrag.direction, -10, 10);
    markDirty();
    event.preventDefault();
  }

  function ampPointerUp(hero, event) {
    if (!ampDrag || event.pointerId !== ampDrag.pointerId) return;

    ampDrag = null;
    try { hero.releasePointerCapture(event.pointerId); } catch (error) {}
    sendAmp(hero);
  }

  function ampDoubleClick(hero, event) {
    var position = ampPointAt(hero, event);
    if (!position) return;

    var points = ampCurveOf(hero);
    var span = ampSpanOf(hero);
    if (points.some(function (point) { return Math.abs(point[0] - position.time) < span * 0.03; })) return;

    points.push([position.time, position.value, 0]);
    points.sort(function (a, b) { return a[0] - b[0]; });
    markDirty();
    sendAmp(hero);
    event.preventDefault();
  }

  function ampContextMenu(hero, event) {
    var node = event.target.closest ? event.target.closest("[data-amp-point]") : null;
    if (!node) return;

    var index = Number(node.getAttribute("data-amp-point"));
    var points = ampCurveOf(hero);
    if (points.length <= 2) return;

    points.splice(index, 1);
    markDirty();
    sendAmp(hero);
    event.preventDefault();
  }
