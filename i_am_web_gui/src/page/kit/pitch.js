// Part: the pitch fall editor. A point is [seconds, ratio, bend], the same
// breakpoint curve the voice evaluates, so the picture and the sound agree.
var fallDrag = null;

function fallTimeOf(pixelX, width) {
  var x = clamp(pixelX / Math.max(width, 1), 0, 1) * 520;
  return clamp((x - 26) / 468, 0, 1) * 0.25;
}

function fallRatioOf(pixelY, height) {
  var y = clamp(pixelY / Math.max(height, 1), 0, 1) * 160;
  return Math.pow(2, clamp((132 - y) / 112, 0, 1) * 3);
}

function fallPosition(hero, event) {
  var svg = hero.querySelector("svg");
  if (!svg) return null;

  var rect = svg.getBoundingClientRect();
  if (!rect.width || !rect.height) return null;

  return {
    time: fallTimeOf(event.clientX - rect.left, rect.width),
    ratio: fallRatioOf(event.clientY - rect.top, rect.height)
  };
}

function fallPointIndex(event) {
  var node = event.target && event.target.closest ? event.target.closest("[data-point]") : null;
  return node ? parseInt(node.getAttribute("data-point"), 10) : -1;
}

function fallBendIndex(event) {
  var node = event.target && event.target.closest ? event.target.closest("[data-bend]") : null;
  return node ? parseInt(node.getAttribute("data-bend"), 10) : -1;
}

function fallRender(hero) {
  if (!window.IAmCard || !IAmCard.hero) return;

  var html = IAmCard.hero("fall", heroValues(), {
    fall: fallState,
    rate: (face && face.rate) || 48000
  });
  if (html) hero.innerHTML = html;
}

function fallPointerDown(event) {
  var hero = event.currentTarget;
  if (event.button !== 0 || !fallState || !fallState.points) return;

  var svg = hero.querySelector("svg");
  if (!svg) return;

  var position = fallPosition(hero, event);
  if (!position) return;

  var pointIndex = fallPointIndex(event);
  var bendIndex = fallBendIndex(event);

  if (pointIndex >= 0) {
    fallDrag = { kind: "point", hero: hero, index: pointIndex, pointerId: event.pointerId };
  } else if (bendIndex >= 0) {
    var rect = svg.getBoundingClientRect();
    fallDrag = {
      kind: "bend",
      hero: hero,
      index: bendIndex,
      pointerId: event.pointerId,
      startY: event.clientY,
      height: rect.height || 160,
      startBend: fallState.points[bendIndex][2] || 0
    };
  } else {
    // Empty space does nothing on a press: adding a point is a double click,
    // so a press that misses a handle never edits the curve.
    return;
  }

  hero.dataset.busy = "1";

  // A host whose pointer id is not active (or a synthetic event) makes this
  // throw; the drag still works without the capture, so it is not fatal.
  if (hero.setPointerCapture) {
    try {
      hero.setPointerCapture(event.pointerId);
    } catch (error) {}
  }

  fallRender(hero);
  event.preventDefault();
}

function fallPointerMove(event) {
  if (!fallDrag) return;

  var hero = fallDrag.hero;

  // A bend handle moves the curve, not the point: up spends the segment late,
  // down spends it early, exactly the sign of the value the voice reads.
  if (fallDrag.kind === "bend") {
    var travel = (fallDrag.startY - event.clientY) / Math.max(fallDrag.height, 1) * 160;
    // The sign follows the segment's slope, so the handle always follows the
    // pointer: up bows the line up, whichever way the segment runs.
    var left = fallDrag.index;
    var points = fallState.points;
    var direction = left + 1 < points.length && points[left + 1][1] > points[left][1] ? -1 : 1;

    points[left][2] = clamp(fallDrag.startBend + travel * 0.09 * direction, -10, 10);
    fallRender(hero);
    event.preventDefault();
    return;
  }

  var position = fallPosition(hero, event);
  if (!position) return;

  var points = fallState.points;
  var index = fallDrag.index;
  var low = index > 0 ? points[index - 1][0] + 0.0005 : 0;
  var high = index < points.length - 1 ? points[index + 1][0] - 0.0005 : 0.25;
  var time = index === 0 ? 0 : clamp(position.time, low, Math.max(low, high));

  points[index][0] = time;
  points[index][1] = position.ratio;
  fallRender(hero);
  event.preventDefault();
}

function fallPointerUp(event) {
  if (!fallDrag) return;

  var hero = fallDrag.hero;
  hero.dataset.busy = "0";

  if (hero.releasePointerCapture && fallDrag.pointerId !== undefined) {
    try {
      hero.releasePointerCapture(fallDrag.pointerId);
    } catch (error) {}
  }

  fallDrag = null;
  sendFall();
  markDirty();
  event.preventDefault();
}

/// Add a point where the picture was double clicked.
function fallDoubleClick(event) {
  if (!fallState || !fallState.points) return;
  if (fallPointIndex(event) >= 0 || fallBendIndex(event) >= 0) return;
  if (fallState.points.length >= 10) return;

  var position = fallPosition(event.currentTarget, event);

  if (!position) return;

  fallState.points.push([position.time, position.ratio, 0]);
  fallState.points.sort(function (a, b) { return a[0] - b[0]; });
  fallRender(event.currentTarget);
  sendFall();
  markDirty();
  event.preventDefault();
}

function fallContextMenu(event) {
  var index = fallPointIndex(event);
  if (index < 0 || !fallState || fallState.points.length <= 2) return;

  fallState.points.splice(index, 1);
  fallRender(event.currentTarget);
  sendFall();
  event.preventDefault();
}

function sendFall() {
  if (!fallState || !fallState.points) return;

  var body = fallState.points.map(function (point) {
    return point[0].toFixed(4) + ":" + point[1].toFixed(4) + ":" + (point[2] || 0).toFixed(3);
  }).join(";");

  fetch("/pitch", { method: "POST", body: body }).catch(function () {});
}
