// The voice pictures: the partial ladder, the string, the modal, the vibrato and the shape.
// Part seven: the pictures for the voices and the canvas editor.
  var HEROES = Object.create(null);

  /// The string's partial ladder: where every partial sits, stretched by the
  /// string's inharmonicity, against how loud it is, shaped by the excitation
  /// point and the body. Unison strings are drawn as a cluster around each bar,
  /// which is what the detune control buys.
  function partialsHero(values) {
    var position = clamp(valueOf(values, "position", 0.125), 0.01, 0.5);
    var inharmonicity = clamp(valueOf(values, "inharmonicity", 1), 0, 4);
    var damping = clamp(valueOf(values, "damping", 1), 0.1, 4);
    var brightness = clamp(valueOf(values, "brightness", 1), 0.1, 4);
    var hardness = clamp(valueOf(values, "hardness", 0.5), 0, 1);
    var excitation = Math.round(valueOf(values, "excitation", 0));
    var strings = Math.max(1, Math.round(valueOf(values, "strings", 1)));
    var detune = clamp(valueOf(values, "detune_cents", 0), 0, 20) / 1200;

    var count = 32;
    var stretch = 0.0007 * inharmonicity;
    var tilt = 0.55 * damping + 0.35 * brightness;
    var partials = [];
    var highest = 0;

    for (var n = 1; n <= count; n++) {
      var ratio = n * Math.sqrt(1 + stretch * n * n);
      // The excitation point is a comb of nodes: a partial with a node where the
      // string is touched is weak, which is what makes a pluck near the bridge
      // bright and one in the middle hollow.
      var amp = Math.abs(Math.sin(Math.PI * n * position));
      amp *= Math.exp(-(n - 1) * tilt / 26);

      if (excitation === 1) amp *= 0.55 + 0.9 * hardness;
      else if (excitation === 2) amp *= 1.1 - 0.35 * Math.exp(-(n - 1) / 5);

      partials.push({ ratio: ratio, amp: amp });
      if (ratio > highest) highest = ratio;
    }

    var max = 0;
    partials.forEach(function (partial) { if (partial.amp > max) max = partial.amp; });

    var left = 20, width = 480, base = 148, full = 112;
    var gap = width / count;
    var bars = "";

    for (var n = 0; n < count; n++) {
      var partial = partials[n];
      var weight = partial.amp / (max || 1);
      var height = Math.max(2, weight * full);
      var opacity = (0.18 + 0.7 * weight).toFixed(2);

      for (var s = 0; s < strings; s++) {
        var offset = strings === 1 ? 0 : (s - (strings - 1) / 2) * detune;
        var x = left + clamp(partial.ratio * (1 + offset) / highest, 0, 1) * width;
        var w = Math.max(2, gap * (strings > 1 ? 0.28 : 0.5));
        bars += '<rect x="' + (x - w / 2).toFixed(1) + '" y="' + (base - height).toFixed(1) +
          '" width="' + w.toFixed(1) + '" height="' + height.toFixed(1) +
          '" rx="1" fill="var(--accent)" fill-opacity="' + opacity + '"/>';
      }
    }

    var names = ["PLUCK", "HAMMER", "BOW"];
    var excitationName = names[excitation] || names[0];
    var stringName = strings === 1 ? "1 STRING" : strings + " STRINGS";

    return heroSvg(
      '<line x1="20" y1="148" x2="500" y2="148" stroke="#2f2f2f"/>' + bars
    ) +
      '<div class="hero-label" style="top:4%">PARTIALS</div>' +
      '<div class="hero-label" style="top:4%; left:auto; right:14px">' +
        excitationName + ' · ' + stringName + '</div>';
  }

  /// The string itself: the triangle a pluck leaves, the point it is touched at,
  /// and a ghost pair for a course of strings, so the unison count reads at a
  /// glance. The dashed curve is the mode the string settles into, whose antinode
  /// sits in the middle rather than at the excitation point.
  function stringHero(values) {
    var position = clamp(valueOf(values, "position", 0.125), 0.01, 0.5);
    var strings = Math.max(1, Math.round(valueOf(values, "strings", 1)));
    var excitation = Math.round(valueOf(values, "excitation", 0));

    var left = 20, right = 500, middle = 92, peak = 58;
    var x = left + position * (right - left);
    var pluck = "M " + left + " " + middle +
      " L " + x.toFixed(1) + " " + (middle - peak) +
      " L " + right + " " + middle;
    var mode = "";

    for (var i = 0; i <= 120; i++) {
      var t = i / 120;
      var px = left + t * (right - left);
      var py = middle - Math.sin(Math.PI * t) * peak * 0.82;
      mode += (i ? " L " : "M ") + px.toFixed(1) + " " + py.toFixed(1);
    }

    var ghosts = "";

    if (strings > 1) {
      ghosts =
        '<path d="' + pluck + '" fill="none" stroke="var(--accent)" stroke-opacity=".22" ' +
          'stroke-width="1.4" transform="translate(0 -7)" vector-effect="non-scaling-stroke"/>' +
        '<path d="' + pluck + '" fill="none" stroke="var(--accent)" stroke-opacity=".22" ' +
          'stroke-width="1.4" transform="translate(0 7)" vector-effect="non-scaling-stroke"/>';
    }

    var names = ["PLUCK", "HAMMER", "BOW"];

    return heroSvg(
      '<line x1="20" y1="' + middle + '" x2="500" y2="' + middle + '" stroke="#2f2f2f"/>' +
      ghosts +
      '<path d="' + mode + '" fill="none" stroke="var(--accent)" stroke-opacity=".32" ' +
        'stroke-dasharray="4 5" stroke-width="1.2" vector-effect="non-scaling-stroke"/>' +
      '<path d="' + pluck + '" fill="none" stroke="var(--accent)" stroke-width="1.9" ' +
        'vector-effect="non-scaling-stroke"/>' +
      '<line x1="' + x.toFixed(1) + '" y1="14" x2="' + x.toFixed(1) + '" y2="118" ' +
        'stroke="var(--accent)" stroke-opacity=".22" stroke-dasharray="2 4"/>'
    ) +
      '<div class="hero-pip" style="left:' + (x / 5.2).toFixed(2) + '%;top:' +
        ((middle - peak) / 1.6).toFixed(2) +
        '%;width:6px;height:6px;background:var(--accent)"></div>' +
      '<div class="hero-label" style="top:4%">STRING</div>' +
      '<div class="hero-label" style="top:4%; left:auto; right:14px">' +
        (names[excitation] || names[0]) + ' AT ' + Math.round(position * 100) + '%</div>';
  }

  /// The hand's vibrato: the pitch stays where it is for the delay, then swings
  /// at the rate and depth the controls ask for.

  /// The fitted instrument: the velocity curve on the left, and the loaded mode
  /// table on the right drawn in oblique projection - depth goes up and to the
  /// right, so the modes read as a bank of decaying partials rather than a flat
  /// trace. Height is the measured level, length the measured decay.
  function modalHero(values) {
    var curl = clamp(valueOf(values, "velocity_curve", 0), -10, 10);
    var attack = clamp(valueOf(values, "attack", 0.004), 0, 0.08);
    var width = clamp(valueOf(values, "width", 0.4), 0, 1);
    var preset = Math.round(valueOf(values, "instrument", 0));

    var curve = "";
    for (var i = 0; i <= 48; i++) {
      var t = i / 48;
      var y = curl === 0 ? t : (Math.exp(curl * t) - 1) / (Math.exp(curl) - 1);
      var px = 24 + t * 184;
      var py = 132 - y * 96;
      curve += (i ? " L " : "M ") + px.toFixed(1) + " " + py.toFixed(1);
    }

    // One shape per fitted table: height is where the partials sit, length is
    // how long they ring. The bass hangs low and long, the celeste is a short
    // bright bell, the guitars sit in between.
    var names = ["GENTLEMAN", "PEEL", "BASS", "CELESTE", "HARP", "CUSTOM"];
    var shapes = [
      { levels: [0.62, 0.46, 0.34, 0.26, 0.19, 0.14, 0.1, 0.07],
        decays: [0.95, 0.6, 0.4, 0.52, 0.34, 0.28, 0.22, 0.18] },
      { levels: [0.5, 0.43, 0.37, 0.31, 0.25, 0.2, 0.16, 0.12],
        decays: [0.55, 0.46, 0.38, 0.31, 0.26, 0.21, 0.17, 0.13] },
      { levels: [0.7, 0.52, 0.4, 0.3, 0.22, 0.16, 0.12, 0.09],
        decays: [1.1, 0.8, 0.6, 0.45, 0.34, 0.26, 0.2, 0.16] },
      { levels: [0.5, 0.4, 0.32, 0.26, 0.2, 0.15, 0.11, 0.08],
        decays: [0.45, 0.38, 0.3, 0.24, 0.19, 0.15, 0.12, 0.1] },
      { levels: [0.6, 0.48, 0.38, 0.3, 0.23, 0.17, 0.12, 0.09],
        decays: [0.75, 0.58, 0.46, 0.38, 0.3, 0.24, 0.19, 0.15] }
    ];
    var shape = shapes[preset] || shapes[0];
    var levels = shape.levels;
    var decays = shape.decays;
    var count = levels.length;
    var depthX = 12, depthY = 8;
    var originX = 250, originY = 134;
    var length = 138;

    var back = "M " + (originX + count * depthX) + " " + (originY - count * depthY) +
      " L " + (originX + count * depthX + length) + " " + (originY - count * depthY);
    var floor = "M " + originX + " " + originY +
      " L " + (originX + length) + " " + originY +
      " L " + (originX + length + count * depthX) + " " + (originY - count * depthY) +
      " L " + (originX + count * depthX) + " " + (originY - count * depthY) + " Z";

    var bases = "";
    var modes = "";
    for (var m = count - 1; m >= 0; m--) {
      var ox = originX + m * depthX;
      var oy = originY - m * depthY;
      bases += '<line x1="' + ox + '" y1="' + oy + '" x2="' + (ox + length) +
        '" y2="' + oy + '" stroke="#2f2f2f" stroke-opacity="' +
        (0.55 - 0.04 * m).toFixed(2) + '"/>';
      var amp = levels[m] * 62;
      var path = "";
      for (var j = 0; j <= 72; j++) {
        var u = j / 72;
        var x = ox + u * length;
        var envelope = Math.exp(-u * 4.2);
        var y = oy - Math.sin(u * Math.PI * 2 * (m + 1) * 1.7) * amp * envelope;
        path += (j ? " L " : "M ") + x.toFixed(1) + " " + y.toFixed(1);
      }
      var opacity = (0.14 + 0.42 * levels[m] + 0.12 * (1 - m / count)).toFixed(2);
      modes += '<path d="' + path + '" fill="none" stroke="var(--accent)" ' +
        'stroke-opacity="' + opacity + '" stroke-width="1.2" ' +
        'vector-effect="non-scaling-stroke"/>';
    }

    return heroSvg(
      '<line x1="24" y1="132" x2="208" y2="132" stroke="#2f2f2f"/>' +
      '<line x1="24" y1="36" x2="24" y2="132" stroke="#2f2f2f"/>' +
      '<path d="' + curve + '" fill="none" stroke="var(--accent)" ' +
        'stroke-width="1.9" vector-effect="non-scaling-stroke"/>' +
      '<path d="' + floor + '" fill="none" stroke="#2f2f2f"/>' +
      '<path d="' + back + '" stroke="#2f2f2f" stroke-opacity=".5"/>' +
      bases + modes
    ) +
      '<div class="hero-label" style="top:4%">VELOCITY</div>' +
      '<div class="hero-label" style="top:4%; left:auto; right:14px">' +
        names[preset === 1 ? 1 : 0] + ' · ' + Math.round(attack * 1000) +
        ' ms · width ' + Math.round(width * 100) + '%</div>';
  }
  function vibratoHero(values) {
    var rate = clamp(valueOf(values, "vibrato_rate", 5.4), 0, 12);
    var depth = clamp(valueOf(values, "vibrato_depth", 0), 0, 120);
    var delay = clamp(valueOf(values, "vibrato_delay", 0), 0, 1);

    var left = 20, right = 500, base = 88;
    // The hero is about the shape: a subtle depth still shows its motion, and the
    // label carries the exact cents.
    var swing = depth <= 0 ? 0 : 12 + 44 * clamp(depth / 120, 0, 1);
    var onset = left + delay * (right - left);
    var path = "M " + left + " " + base + " L " + onset.toFixed(1) + " " + base;

    for (var i = 0; i <= 240; i++) {
      var t = i / 240;
      var px = onset + (right - onset) * t;
      var py = base + Math.sin(t * rate * Math.PI * 2) * swing;
      path += " L " + px.toFixed(1) + " " + py.toFixed(1);
    }

    return heroSvg(
      '<line x1="20" y1="' + base + '" x2="500" y2="' + base + '" stroke="#2f2f2f" ' +
        'stroke-dasharray="3 5"/>' +
      '<path d="' + path + '" fill="none" stroke="var(--accent)" stroke-width="2" ' +
        'vector-effect="non-scaling-stroke"/>' +
      '<line x1="' + onset.toFixed(1) + '" y1="18" x2="' + onset.toFixed(1) + '" y2="126" ' +
        'stroke="var(--accent)" stroke-opacity=".22" stroke-dasharray="2 4"/>'
    ) +
      '<div class="hero-label" style="top:4%">VIBRATO</div>' +
      '<div class="hero-label" style="top:4%; left:auto; right:14px">' +
        rate.toFixed(1) + ' Hz · ' + Math.round(depth) + ' ¢</div>';
  }

  /// The wavetable morph: the four classic tables, and the one the Shape control
  /// sits between. The caption says which pair and how far into it the blend is.
  function shapeHero(values, context) {
    var waves = context && context.waves;
    var morph = clamp(valueOf(values, "oscillator.smooth_factor", 0), 0, 1);
    var names = ["SINE", "TRIANGLE", "SAW", "SQUARE"];

    // A table the page imported: the curve is the two frames Morph sits between,
    // blended the way the voice blends them, so the picture follows the knob.
    if (waves && waves.previews && waves.previews.length > 0) {
      var count = waves.previews.length;
      var position = morph * (count - 1);
      var frameFrom = Math.min(Math.floor(position), count - 1);
      var frameTo = Math.min(frameFrom + 1, count - 1);
      var blend = position - frameFrom;
      var first = waves.previews[frameFrom];
      var second = waves.previews[frameTo];

      // The table's value at one phase, blended between the two frames the morph
      // sits on, which is what the voice plays.
      var curve = function (time) {
        var at = clamp(time, 0, 1) * (first.length - 1);
        var left = Math.floor(at);
        var part = at - left;
        var next = Math.min(left + 1, first.length - 1);
        var a = first[left] + (first[next] - first[left]) * part;
        var b = second[left] + (second[next] - second[left]) * part;

        return a + (b - a) * blend;
      };

      // The X-Fade knob pulls the end of each frame towards its beginning, so the
      // picture shows the same loop the voice plays.
      var width = clamp(valueOf(values, "oscillator.x_fade", 0), 0, 1) * 0.25;
      var imported = "";

      for (var index = 0; index <= 240; index++) {
        var phase = index / 240;
        var value = curve(phase);

        if (width > 0 && phase >= 1 - width) {
          var over = (phase - (1 - width)) / width;

          value = value * (1 - over) + curve(phase - (1 - width)) * over;
        }

        var x = 20 + phase * 480;
        var y = 80 - clamp(value, -1, 1) * 54;
        imported += (index ? " L " : "M ") + x.toFixed(1) + " " + y.toFixed(1);
      }

      var name = String(waves.name || "TABLE").toUpperCase();

      if (name.length > 26) name = name.slice(0, 25) + "\u2026";

      var caption = waves.error
        ? String(waves.error).slice(0, 40)
        : name + " \u00b7 " + count + " FRAMES \u00b7 " + Math.round(morph * 100) + "%";

      return heroSvg(
        '<line x1="20" y1="80" x2="500" y2="80" stroke="#2c2c2c" stroke-dasharray="3 5"/>' +
        '<path d="' + imported + '" fill="none" stroke="var(--accent)" stroke-width="2" ' +
          'vector-effect="non-scaling-stroke"/>'
      ) +
        '<div class="hero-label" style="top:4%">SHAPE</div>' +
        '<div class="hero-label" style="top:4%; left:auto; right:14px">' + caption + '</div>' +
        '<div class="hero-actions">' +
          '<button type="button" class="hero-action" data-action="prev">\u2039</button>' +
          '<button type="button" class="hero-action" data-action="next">\u203a</button>' +
          '<button type="button" class="hero-action" data-action="delete">DEL FRAME</button>' +
          '<button type="button" class="hero-action" data-action="mode">MODE: ' +
            String(context && context.waveMode || "auto").toUpperCase() + '</button>' +
          '<button type="button" class="hero-action" data-action="reverse">REVERSE</button>' +
          '<button type="button" class="hero-action" data-action="invert">INVERT</button>' +
          '<button type="button" class="hero-action" data-action="dc">REMOVE DC</button>' +
          '<button type="button" class="hero-action" data-action="normalize">NORMALIZE</button>' +
        '</div>';
    }

    var tables = [
      function (x) { return Math.sin(2 * Math.PI * x); },
      function (x) { return 1 - 4 * Math.abs(x - 0.5); },
      function (x) { return 2 * x - 1; },
      function (x) { return x < 0.5 ? 1 : -1; }
    ];
    var smooth = morph * (tables.length - 1);
    var from = Math.min(Math.floor(smooth), tables.length - 1);
    var to = Math.min(from + 1, tables.length - 1);
    var part = smooth - from;
    var classic = "";

    for (var step = 0; step <= 240; step++) {
      var time = step / 240;
      var px = 20 + time * 480;
      var phase = (time * 2) % 1;
      var classicValue = tables[from](phase) + (tables[to](phase) - tables[from](phase)) * part;
      var py = 80 - classicValue * 54;
      classic += (step ? " L " : "M ") + px.toFixed(1) + " " + py.toFixed(1);
    }

    var classicCaption = from === to
      ? names[from]
      : names[from] + " \u2192 " + names[to] + " " + Math.round(part * 100) + "%";

    return heroSvg(
      '<line x1="20" y1="80" x2="500" y2="80" stroke="#2c2c2c" stroke-dasharray="3 5"/>' +
      '<path d="' + classic + '" fill="none" stroke="var(--accent)" stroke-width="2" ' +
        'vector-effect="non-scaling-stroke"/>'
    ) +
      '<div class="hero-label" style="top:4%">SHAPE</div>' +
      '<div class="hero-label" style="top:4%; left:auto; right:14px">' + classicCaption + '</div>';
  }

  /// The sample: its picture, the loop region, and the slices the mode plays.
