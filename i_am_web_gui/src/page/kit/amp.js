  // Part five-and-a-half: the amplitude envelope editor.
  //
  // The x axis is the share of the drum's decay and the y axis is the level, so
  // the picture is the shape and the Decay knob is how long it lasts. The
  // gestures match the fall editor: drag a dot, drag a ring for the bend, double
  // click to add, right click to remove.

  function ampCurve() {
    return ampPoints;
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

  function ampY(value) {
    return 132 - clamp(value, 0, 1) * 112;
  }

  function ampPointAt(hero, event) {
    var map = ampMap(hero);
    if (!map) return null;

    return {
      time: clamp(((event.clientX - map.left) / map.scale - 26) / 468, 0, 1),
      value: clamp((132 - (event.clientY - map.top) / map.scale) / 112, 0, 1)
    };
  }

  function ampDrumNow() {
    return controls.amp_drum ? Math.round(controls.amp_drum.value) : 0;
  }

  function sendAmp() {
    var body = ampPoints
      .map(function (point) {
        return point[0].toFixed(4) + ":" + point[1].toFixed(4) + ":" + (point[2] || 0).toFixed(3);
      })
      .join(";");

    fetch("/amp?drum=" + ampDrumNow(), { method: "POST", body: body })
      .then(function (response) {
        return response.ok ? response.json() : null;
      })
      .catch(function () {});
  }

  function ampPointerDown(hero, event) {
    if (event.button !== 0 || !ampPoints.length) return;

    var map = ampMap(hero);
    if (!map) return;

    var bend = event.target.closest ? event.target.closest("[data-amp-bend]") : null;
    var point = event.target.closest ? event.target.closest("[data-amp-point]") : null;

    if (bend) {
      var index = Number(bend.getAttribute("data-amp-bend"));
      var left = ampPoints[index];
      var right = ampPoints[index + 1];
      if (!left || !right) return;

      ampDrag = {
        kind: "bend",
        index: index,
        pointerId: event.pointerId,
        startY: event.clientY,
        scale: map.scale || 1,
        startBend: left[2] || 0,
        direction: right[1] > left[1] ? 1 : -1
      };
      try { hero.setPointerCapture(event.pointerId); } catch (error) {}
      event.preventDefault();
      return;
    }

    if (!point) return;

    var pointIndex = Number(point.getAttribute("data-amp-point"));
    ampDrag = {
      kind: "point",
      index: pointIndex,
      pointerId: event.pointerId
    };
    try { hero.setPointerCapture(event.pointerId); } catch (error) {}
    event.preventDefault();
  }

  function ampPointerMove(hero, event) {
    if (!ampDrag || event.pointerId !== ampDrag.pointerId) return;

    if (ampDrag.kind === "point") {
      var position = ampPointAt(hero, event);
      if (!position) return;

      var point = ampPoints[ampDrag.index];
      if (!point) return;

      point[0] = ampDrag.index === 0 ? 0 : Math.max(point[0], position.time);
      point[1] = Math.max(0, Math.min(1, position.value));
      ampPoints.sort(function (a, b) { return a[0] - b[0]; });
      markDirty();
      event.preventDefault();
      return;
    }

    var travel = (ampDrag.startY - event.clientY) / Math.max(ampDrag.scale, 1e-6);
    ampPoints[ampDrag.index][2] = clamp(ampDrag.startBend + travel * 0.09 * ampDrag.direction, -10, 10);
    markDirty();
    event.preventDefault();
  }

  function ampPointerUp(hero, event) {
    if (!ampDrag || event.pointerId !== ampDrag.pointerId) return;

    ampDrag = null;
    try { hero.releasePointerCapture(event.pointerId); } catch (error) {}
    sendAmp();
  }

  function ampDoubleClick(hero, event) {
    var position = ampPointAt(hero, event);
    if (!position) return;

    if (ampPoints.some(function (point) { return Math.abs(point[0] - position.time) < 0.02; })) return;

    ampPoints.push([position.time, position.value, 0]);
    ampPoints.sort(function (a, b) { return a[0] - b[0]; });
    markDirty();
    sendAmp();
    event.preventDefault();
  }

  function ampContextMenu(hero, event) {
    var node = event.target.closest ? event.target.closest("[data-amp-point]") : null;
    if (!node) return;

    var index = Number(node.getAttribute("data-amp-point"));
    if (ampPoints.length <= 2 || index === 0) return;

    ampPoints.splice(index, 1);
    markDirty();
    sendAmp();
    event.preventDefault();
  }
