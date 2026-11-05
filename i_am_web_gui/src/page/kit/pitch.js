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

  var position = fallPosition(hero, event);
  if (!position) return;

  var index = fallPointIndex(event);

  if (index < 0) {
    if (fallState.points.length >= 10) return;

    var point = [position.time, position.ratio, 0];
    fallState.points.push(point);
    fallState.points.sort(function (a, b) { return a[0] - b[0]; });
    index = fallState.points.indexOf(point);
  }

  fallDrag = { hero: hero, index: index, pointerId: event.pointerId };
  hero.dataset.busy = "1";
  if (hero.setPointerCapture) hero.setPointerCapture(event.pointerId);
  fallRender(hero);
  event.preventDefault();
}

function fallPointerMove(event) {
  if (!fallDrag) return;

  var hero = fallDrag.hero;
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
