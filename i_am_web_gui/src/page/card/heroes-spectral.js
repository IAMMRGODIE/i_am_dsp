// The Spectral Dynamics curves: the threshold tilt and the ratio every bin
// scales by. Each curve is its own hero, so one point editor per picture and
// the pointer geometry stays the one every other curve already uses.

function spectralCurvePages(state, which) {
  var pages = state && state[which];

  if (!pages || !pages.length) {
    var flat = which === "ratio" ? 1 : 0.5;
    return [[{ time: 0, value: flat, bend: 0 }, { time: 1, value: flat, bend: 0 }]];
  }

  return pages;
}

/// The page's value at a position across the frequency axis.
function spectralCurveAt(points, x) {
  var sorted = points.slice().sort(function (a, b) { return a.time - b.time; });

  if (sorted.length < 2) return 0.5;
  if (x <= sorted[0].time) return sorted[0].value;

  for (var i = 1; i < sorted.length; i++) {
    var left = sorted[i - 1];
    var right = sorted[i];

    if (x <= right.time) {
      var span = Math.max(right.time - left.time, 1e-6);
      var unit = Math.max(0, Math.min(1, (x - left.time) / span));
      return left.value + (right.value - left.value) * unit;
    }
  }

  return sorted[sorted.length - 1].value;
}

function spectralCurvePath(points) {
  var path = "";

  for (var i = 0; i < 72; i++) {
    var x = i / 71;
    var value = Math.max(0, Math.min(1, spectralCurveAt(points, x)));
    var px = 8 + x * 284;
    var py = 56 - value * 46;

    path += (i ? " L " : "M ") + px.toFixed(1) + " " + py.toFixed(1);
  }

  return path;
}

/// The parameter id of the morph: the DSP nests it, so the id carries a prefix.
function spectralMorphId(values) {
  var keys = Object.keys(values || {});

  for (var i = 0; i < keys.length; i++) {
    if (keys[i] === "morph" || keys[i].slice(-6) === ".morph") return keys[i];
  }

  return null;
}

/// The morph's current value, whichever prefix its id has.
function spectralMorph(values) {
  var id = spectralMorphId(values);

  return id ? valueOf(values, id, 0) : 0;
}

/// Which of the two curves the one hero is showing: threshold or ratio.
function spectralCurveView(context) {
  return context && context.sdCurve && context.sdCurve.view === "ratio" ? "ratio" : "threshold";
}

/// The curve hero: one picture, a mode button, and the page strip.
function spectralDynamicsCurveHero(values, context) {
  var view = spectralCurveView(context);
  var pages = spectralCurvePages(context && context.spectralDynamics, view);
  var selected = IAmPage.index(spectralMorph(values), pages.length);
  var points = pages[selected] || [];
  var label = view === "ratio" ? "RATIO" : "THRESHOLD";
  var unit = view === "ratio" ? "1:1 to the master ratio" : "-36 to +36 dB";

  return '<svg class="mcard-graph" viewBox="0 0 300 64" preserveAspectRatio="xMidYMid meet">' +
    '<line x1="8" y1="56" x2="292" y2="56" stroke="#333" stroke-width="1" stroke-dasharray="2 4"/>' +
    '<line x1="8" y1="33" x2="292" y2="33" stroke="#262626" stroke-width="1"/>' +
    '<path d="' + spectralCurvePath(points) + '" fill="none" stroke="var(--accent)" ' +
    'stroke-width="2" vector-effect="non-scaling-stroke"/>' +
    '</svg>' +
    '<div class="hero-actions">' +
      '<button type="button" class="hero-action" data-action="sd-prev">\u2039</button>' +
      '<button type="button" class="hero-action" data-action="sd-next">\u203a</button>' +
      '<button type="button" class="hero-action" data-action="sd-add">+ PAGE</button>' +
      '<button type="button" class="hero-action" data-action="sd-remove">- PAGE</button>' +
      '<button type="button" class="hero-action" data-action="sd-view">' +
        (view === "ratio" ? "THRESH" : "RATIO") + '</button>' +
    '</div>' +
    '<div class="hero-label" style="left:2%;top:2%">' + label + ' \u00b7 PAGE ' +
    (selected + 1) + ' / ' + pages.length + ' \u00b7 ' + unit + '</div>';
}

/// The two page lists as arrays, which is what the product route reads.
function spectralPosts(pages) {
  if (!pages) return [];

  return pages.map(function (page) {
    return page.map(function (point) { return [point.time, point.value, point.bend || 0]; });
  });
}

/// Attaches the point editor and the hero's buttons to one curve.
///
/// The state object is replaced by the poll, so every callback resolves it
/// again instead of holding the array it started with: an edit must land in the
/// page the server will read back.
function wireSpectralCurve(el, state, values) {
  if (!state || !state.spectralDynamics) return;

  var svg = el.querySelector("svg");
  if (!svg) return;

  var curves = function () { return state.spectralDynamics; };
  var view = function () { return spectralCurveView(state); };
  var pagesOf = function () {
    var live = curves();
    var which = view();

    // A page the poll has not filled yet still has to be a real array: an edit
    // written into a default the getter rebuilt would be gone on the next paint.
    if (!live[which] || !live[which].length) {
      live[which] = spectralCurvePages(live, which);
    }

    return live[which];
  };
  var selectedOf = function () {
    return IAmPage.index(spectralMorph(values), pagesOf().length);
  };

  var pages = pagesOf();
  var selected = selectedOf();
  var accent = (getComputedStyle(el).getPropertyValue("--accent") || "#4c8dff").trim();
  var points = pages[selected];

  // The shared editor interpolates with the segment bend, so every point the
  // canvas sees has to carry one; a page the poll has not filled yet may not.
  points = (points || []).map(function (point) {
    return { time: point.time, value: point.value, bend: point.bend || 0 };
  });

  if (points.length < 2) {
    var flat = view() === "ratio" ? 1 : 0.5;
    points = [{ time: 0, value: flat, bend: 0 }, { time: 1, value: flat, bend: 0 }];
    pages[selected] = points;
  }

  var repaint = function () {
    if (state.repaint) state.repaint();
  };

  // The page's shared post helper only knows the filter table and the history
  // curve, so this product posts to its own route.
  var post = function () {
    var live = curves();

    fetch("/spectral-dynamics", {
      method: "POST",
      body: JSON.stringify({
        threshold: spectralPosts(live.threshold),
        ratio: spectralPosts(live.ratio)
      })
    }).catch(function () {});
  };

  var copyPage = function () {
    return points.map(function (point) {
      return { time: point.time, value: point.value, bend: point.bend || 0 };
    });
  };

  // The morph is the page selector, so stepping a page moves the same control
  // the host automates. The control's id carries the mapper prefix the DSP has,
  // so it is found by suffix rather than guessed.
  var morphId = function () {
    return spectralMorphId(values) || "morph";
  };

  var step = function (delta) {
    var list = pagesOf();
    var index = Math.max(0, Math.min(list.length - 1, selectedOf() + delta));

    if (list.length > 1 && state.set) state.set(morphId(), index / (list.length - 1));
  };

  // Both curves read the same morph, so they have to keep the same page count:
  // otherwise the same knob lands on page 2 of one curve and page 1 of the
  // other.
  var clonePage = function (page) {
    return page.map(function (point) {
      return { time: point.time, value: point.value, bend: point.bend || 0 };
    });
  };

  var eachCurve = function (handler) {
    var live = curves();

    ["threshold", "ratio"].forEach(function (which) {
      if (!live[which] || !live[which].length) live[which] = spectralCurvePages(live, which);
      handler(live[which], which);
    });
  };

  var actions = {
    "sd-prev": function () { step(-1); },
    "sd-next": function () { step(1); },
    "sd-add": function () {
      var active = view();

      eachCurve(function (list, which) {
        list.push(which === active ? copyPage() : clonePage(list[list.length - 1]));
      });

      post();
      repaint();
    },
    "sd-remove": function () {
      var index = selectedOf();
      var removable = true;

      eachCurve(function (list) {
        if (list.length <= 1) removable = false;
      });

      if (!removable) return;

      eachCurve(function (list) {
        list.splice(index, 1);
      });

      post();
      repaint();
    },
    "sd-view": function () {
      if (!state.sdCurve) return;
      state.sdCurve.view = view() === "ratio" ? "threshold" : "ratio";
      repaint();
    }
  };

  // The hero element survives a repaint while its buttons do not, so the click
  // is heard once here and dispatched to whatever buttons the paint produced.
  if (!el.__sdClick) {
    el.__sdClick = true;
    el.addEventListener("click", function (event) {
      var button = event.target.closest("[data-action]");
      var handler = button && el.__sdActions && el.__sdActions[button.getAttribute("data-action")];

      if (!handler) return;

      event.stopPropagation();
      handler();
    });
  }

  el.__sdActions = actions;

  // A double click has to survive the gap between its two presses, so the hero
  // stays busy until the gesture is over rather than only while a drag runs.
  svg.addEventListener("pointerdown", function () {
    el.dataset.busy = "1";
    clearTimeout(el.__sdRelease);
  });
  svg.addEventListener("pointerup", function () {
    clearTimeout(el.__sdRelease);
    el.__sdRelease = setTimeout(function () { delete el.dataset.busy; }, 400);
  });

  IAmCard.sources.editLfo(svg, { points: points }, {
    paint: function (next) {
      IAmCard.sources.drawPreview(svg, "lfo", {
        points: next,
        color: accent,
        segments: 36,
        handles: true
      }, null);
    },
    commit: function (next) {
      pagesOf()[selectedOf()] = next;
      post();
    }
  });
}

/// The before/after spectrum: what the bin gains do to the signal, as bands.
function spectralDynamicsSpectrumHero(values, context) {
  var scope = context && context.scope;
  var state = context && context.spectra;
  var rate = (context && context.rate) || 48000;
  var empty = '<line x1="20" y1="140" x2="500" y2="140" stroke="#2c2c2c" stroke-width="1"/>';

  if (!scope || !scope.input || !scope.output || !state) {
    return heroSvg(empty) + '<div class="hero-label" style="left:2%;top:2%">BEFORE / AFTER</div>';
  }

  var inputBands = bandsOf(scope.input, rate);
  var outputBands = bandsOf(scope.output, rate);

  state.input = smoothBands(state.input, inputBands);
  state.output = smoothBands(state.output, outputBands);

  var before = bandsPath(state.input);
  var after = bandsPath(state.output);

  return heroSvg(
    '<path d="' + before + '" fill="none" stroke="#8a8a8a" stroke-width="1.2" ' +
      'opacity=".7" vector-effect="non-scaling-stroke"/>' +
    '<path d="' + after + ' L 500 150 L 20 150 Z" fill="var(--accent)" fill-opacity=".06"/>' +
    '<path d="' + after + '" fill="none" stroke="var(--accent)" stroke-width="1.5" ' +
      'opacity=".9" vector-effect="non-scaling-stroke"/>'
  ) +
    '<div class="hero-label" style="left:2%;top:2%">SPECTRUM · BEFORE (GREY) / AFTER</div>';
}