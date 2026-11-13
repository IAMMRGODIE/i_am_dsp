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
  function samplerHero(values, context) {
    var sample = context && context.sample;
    var mode = Math.round(valueOf(values, "oscillator.mode", 0));
    var modeName = ["SAMPLE", "SLICE", "GRAIN"][Math.min(Math.max(mode, 0), 2)];

    if (!sample || !sample.peaks || !sample.peaks.length) {
      // Labels are HTML: text inside the stretched SVG changes shape with the
      // window.
      return heroSvg(
        '<line x1="20" y1="80" x2="500" y2="80" stroke="#2c2c2c"/>'
      ) +
        '<div class="hero-empty" style="color:' +
        (sample && sample.error ? "#ff7a6b" : "#8a8a8a") + '">' +
        escapeXml(sample && sample.error ? String(sample.error).slice(0, 60) : "drop a sample") +
        '</div>' +
        '<div class="hero-label" style="top:4%">SAMPLE</div>';
    }

    // The window of the sample the picture shows: the wheel narrows it, so a
    // drag lands on the exact place it points at.
    var view = (context && context.sampleView) || { from: 0, to: 1 };
    var span = Math.max(1e-6, view.to - view.from);
    var mapX = function (position) { return 20 + (position - view.from) / span * 480; };
    var peaks = sample.peaks;
    var seconds = sample.rate ? sample.samples / sample.rate : 0;
    var loop = valueOf(values, "oscillator.loop", 0) > 0.5;
    var start = clamp(valueOf(values, "oscillator.loop_start", 0), 0, 1);
    var end = clamp(valueOf(values, "oscillator.loop_end", 1), 0, 1);
    var marks = sample.slices || [];
    var region = "";
    var markers = "";
    var bars = "";

    if (loop && end > start) {
      var regionFrom = clamp((start - view.from) / span, 0, 1);
      var regionTo = clamp((end - view.from) / span, 0, 1);

      if (regionTo > regionFrom) {
        region = '<rect x="' + (20 + regionFrom * 480).toFixed(1) + '" y="14" width="' +
          ((regionTo - regionFrom) * 480).toFixed(1) +
          '" height="132" fill="var(--accent)" fill-opacity="0.08"/>';
      }
    }

    if (mode === 1) {
      for (var index = 0; index < marks.length; index++) {
        var cut = mapX(clamp(marks[index], 0, 1));

        if (cut < 20 || cut > 500) continue;

        markers += '<line x1="' + cut.toFixed(1) + '" y1="14" x2="' + cut.toFixed(1) +
          '" y2="146" stroke="#4a4a4a" vector-effect="non-scaling-stroke"/>';
      }
    }

    // One bar per column, read from the peaks the route sent: it stays smooth at
    // every zoom because the columns interpolate between them.
    var columns = 240;

    for (var column = 0; column < columns; column++) {
      var at = view.from + (column / (columns - 1)) * span;
      var index = clamp(at, 0, 1) * (peaks.length - 1);
      var left = Math.floor(index);
      var part = index - left;
      var next = Math.min(left + 1, peaks.length - 1);
      var value = peaks[left] + (peaks[next] - peaks[left]) * part;
      var height = Math.abs(clamp(value, -1, 1)) * 54;
      var x = 20 + column / (columns - 1) * 480;

      bars += '<line x1="' + x.toFixed(1) + '" y1="' + (80 - height).toFixed(1) +
        '" x2="' + x.toFixed(1) + '" y2="' + (80 + height).toFixed(1) +
        '" stroke="var(--accent)" stroke-width="2" vector-effect="non-scaling-stroke"/>';
    }

    var name = String(sample.name || "SAMPLE").toUpperCase();

    if (name.length > 22) name = name.slice(0, 21) + "\u2026";

    var caption = name + " \u00b7 " + seconds.toFixed(2) + " s \u00b7 " +
      (Math.round(sample.rate / 100) / 10) + " kHz \u00b7 " + modeName;

    if (span < 0.999) caption += " \u00b7 " + (1 / span).toFixed(1) + "\u00d7";

    var reverse = valueOf(values, "oscillator.reverse", 0) > 0.5;
    var polarity = valueOf(values, "oscillator.polarity", 0) > 0.5;
    var pingpong = valueOf(values, "oscillator.pingpong", 0) > 0.5;
    var on = ' style="color:var(--accent)"';

    // The grain mode's picture: the spray the grains are thrown over, and a few
    // grain windows inside it.
    if (mode === 2) {
      var position = clamp(valueOf(values, "oscillator.position", 0), 0, 1);
      var spray = clamp(valueOf(values, "oscillator.spray", 0), 0, 1);
      var grainSize = clamp(valueOf(values, "oscillator.grain_size", 0.06), 0.001, 1);
      var total = sample.rate ? sample.samples / sample.rate : 0;
      var half = clamp(grainSize / Math.max(total, 0.001), 0.002, 1) * 240 / span;
      var bandFrom = clamp(position - spray / 2, 0, 1);
      var bandTo = clamp(position + spray / 2, 0, 1);
      var offsets = [-0.42, 0.06, 0.38, -0.18, 0.22];

      markers += '<rect x="' + mapX(bandFrom).toFixed(1) + '" y="14" width="' +
        ((bandTo - bandFrom) * 480 / span).toFixed(1) +
        '" height="132" fill="var(--accent)" fill-opacity="0.06"/>';

      for (var grain = 0; grain < offsets.length; grain++) {
        var x = mapX(clamp(position + offsets[grain] * spray, 0, 1));

        markers += '<path d="M ' + (x - half).toFixed(1) + ' 146 Q ' + x.toFixed(1) + ' 46 ' +
          (x + half).toFixed(1) + ' 146" fill="none" stroke="var(--accent)" ' +
          'stroke-opacity="0.45" vector-effect="non-scaling-stroke"/>';
      }
    }

    var startX = mapX(clamp(valueOf(values, "oscillator.start", 0), 0, 1));
    var startMark = "";

    if (startX >= 20 && startX <= 500) {
      startMark = '<line x1="' + startX.toFixed(1) + '" y1="14" x2="' + startX.toFixed(1) +
        '" y2="146" stroke="var(--accent)" stroke-opacity="0.6" vector-effect="non-scaling-stroke"/>';
    }

    return heroSvg(
      region + markers + startMark +
      '<line x1="20" y1="80" x2="500" y2="80" stroke="#2c2c2c" stroke-dasharray="3 5"/>' +
      bars
    ) +
      '<div class="hero-label" style="top:4%">SAMPLE</div>' +
      '<div class="hero-label" style="top:4%; left:auto; right:14px">' + caption + '</div>' +
      '<div class="hero-actions">' +
        '<button type="button" class="hero-action" data-action="load">LOAD</button>' +
        (mode === 1 ? '<button type="button" class="hero-action" data-action="slice">SLICE</button>' : '') +
        '<button type="button" class="hero-action" data-action="loop"' + (loop ? on : '') + '>LOOP</button>' +
        '<button type="button" class="hero-action" data-action="pingpong"' + (pingpong ? on : '') + '>PINGPONG</button>' +
        '<button type="button" class="hero-action" data-action="reverse"' + (reverse ? on : '') + '>REVERSE</button>' +
        '<button type="button" class="hero-action" data-action="polarity"' + (polarity ? on : '') + '>INVERT</button>' +
      '</div>';
  }

  /// The envelope the controls describe: delay, attack, hold, decay, the level it
  /// sustains at, and the release, with the bends as the curves the DSP uses.
  function envelopeHero(values) {
    // With the envelope's Sync switch on, the stage lengths are the musical ones.
    // The page is not told the host's tempo, so the picture reads them at 120 bpm:
    // every stage scales together, which is the shape the picture is about.
    var synced = !!valueOf(values, "sync", 0);
    var beats = function (id) {
      return Math.max(valueOf(values, id, 1), 0.03125) * 500;
    };
    var free = function (id, fallback) {
      return Math.max(valueOf(values, id, fallback), 0);
    };

    var delay = synced ? beats("delay.beats") : free("delay_time", 0);
    var attack = synced ? beats("attack.beats") : free("attack_time", 10);
    var hold = synced ? beats("hold.beats") : free("hold_time", 100);
    var decay = synced ? beats("decay.beats") : free("decay_time", 100);
    var sustain = clamp(valueOf(values, "sustain_level", 1), 0, 1);
    var release = synced ? beats("release.beats") : free("release_time", 100);
    var attackBend = valueOf(values, "attack_bend", 0);
    var decayBend = valueOf(values, "decay_bend", 0);
    var releaseBend = valueOf(values, "release_bend", 0);

    var weight = function (time) { return Math.sqrt(Math.max(time, 0.001)); };
    var sustainWidth =
      (weight(delay) + weight(attack) + weight(hold) + weight(decay) + weight(release)) * 0.35;
    var total = weight(delay) + weight(attack) + weight(hold) + weight(decay) +
      weight(release) + sustainWidth;

    var left = 20, width = 480, base = 140, top = 18;
    var height = base - top;
    var x = left;
    var points = [];

    var bend = function (t, factor) {
      if (Math.abs(factor) < 1e-4) return t;
      return (Math.exp(factor * t) - 1) / (Math.exp(factor) - 1);
    };

    var stage = function (time, from, to, factor) {
      var stageWidth = weight(time) / total * width;
      var steps = Math.max(2, Math.round(stageWidth / 6));

      for (var i = 1; i <= steps; i++) {
        var t = bend(i / steps, factor);
        points.push([
          x + stageWidth * (i / steps),
          base - (from + (to - from) * t) * height
        ]);
      }

      x += stageWidth;
    };

    stage(delay, 0, 0, 0);
    stage(attack, 0, 1, attackBend);
    stage(hold, 1, 1, 0);
    stage(decay, 1, sustain, decayBend);
    stage(sustainWidth, sustain, sustain, 0);
    stage(release, sustain, 0, releaseBend);

    var line = "M " + left + " " + base;

    points.forEach(function (point) {
      line += " L " + point[0].toFixed(1) + " " + point[1].toFixed(1);
    });

    return heroSvg(
      '<line x1="20" y1="' + base + '" x2="500" y2="' + base + '" stroke="#2c2c2c"/>' +
      '<path d="' + line + '" fill="none" stroke="var(--accent)" stroke-width="2" ' +
        'vector-effect="non-scaling-stroke"/>' +
      '<path d="' + line + ' L ' + (left + width) + ' ' + base + ' Z" ' +
        'fill="var(--accent)" fill-opacity=".10"/>'
    ) +
      '<div class="hero-label" style="top:4%">ENVELOPE</div>' +
      '<div class="hero-label" style="top:calc(4% + 14px); left:auto; right:14px">' +
        'S ' + Math.round(sustain * 100) + '%</div>' +
      (synced ? '<div class="hero-badge">SYNC</div>' : '');
  }

  /// A spectrum-first additive voice: the bars are the partial amplitudes the
  /// tilt, odd/even and formant controls make. Unlike the bent saw, every ratio
  /// is an integer, so the oscillator plays this whole sum from one table.
  function sinesHero(values) {
    var count = Math.round(clamp(valueOf(values, "oscillator.num_sines", 64), 1, 128));
    var tilt = valueOf(values, "oscillator.freq_gen.tilt", 1);
    var odd = clamp(valueOf(values, "oscillator.freq_gen.odd", 0), 0, 1);
    var formant = valueOf(values, "oscillator.freq_gen.formant", 0);
    var centre = Math.max(valueOf(values, "oscillator.freq_gen.formant_center", 4), 1);

    var amplitudes = [];
    var max = 0;

    for (var i = 1; i <= count; i++) {
      var amplitude = Math.pow(i, -tilt);
      if (i % 2 === 0) amplitude *= 1 - odd;
      if (formant > 0) {
        var distance = Math.log(i / centre) / 0.5;
        amplitude *= 1 + formant * Math.exp(-distance * distance);
      }
      amplitudes.push(amplitude);
      max = Math.max(max, amplitude);
    }

    var base = 132;
    var height = base - 24;
    var step = 480 / count;
    var barWidth = Math.max(1.5, step * 0.7);
    var bars = "";

    amplitudes.forEach(function (amplitude, index) {
      var x = 20 + (index + 0.5) * step;
      var y = base - (amplitude / Math.max(max, 1e-6)) * height * 0.9;
      bars += '<rect x="' + (x - barWidth / 2).toFixed(1) + '" y="' + y.toFixed(1) +
        '" width="' + barWidth.toFixed(1) + '" height="' + (base - y).toFixed(1) +
        '" fill="var(--accent)" opacity="0.85"/>';
    });

    return heroSvg(
      '<line x1="20" y1="' + base + '" x2="500" y2="' + base + '" stroke="#2c2c2c"/>' + bars
    ) +
      '<div class="hero-label" style="top:4%">SPECTRUM</div>' +
      '<div class="hero-label" style="top:4%; left:auto; right:14px">' +
        count + ' PARTIALS</div>';
  }

  /// The additive voice's partials: the saw the generator draws, bent by the
  /// controls. Bars are placed by partial ratio and sized by amplitude.
  function additiveHero(values, context) {
    if (Object.prototype.hasOwnProperty.call(values, "oscillator.freq_gen.tilt")) {
      return sinesHero(values);
    }

    var script = context && context.script;

    if (script && script.error) {
      return heroSvg("") +
        '<div class="hero-empty" style="color:#ff7a6b">' +
        escapeXml(String(script.error).slice(0, 120)) + '</div>';
    }

    if (script && script.spectrum && script.spectrum.length) {
      var spectrum = script.spectrum;
      var lastLog = 1;
      var loudest = 0;

      spectrum.forEach(function (partial) {
        lastLog = Math.max(lastLog, Math.log(Math.max(partial[0], 1.0001)));
        loudest = Math.max(loudest, Math.abs(partial[1]));
      });

      var floor = 132;
      var high = floor - 24;
      var bars = "";

      spectrum.forEach(function (partial) {
        var x = 20 + (Math.log(Math.max(partial[0], 1.0001)) / lastLog) * 480;
        var y = floor - (Math.abs(partial[1]) / Math.max(loudest, 1e-6)) * high * 0.9;

        bars += '<line x1="' + x.toFixed(1) + '" y1="' + floor + '" x2="' + x.toFixed(1) +
          '" y2="' + y.toFixed(1) + '" stroke="var(--accent)" stroke-width="2" ' +
          'vector-effect="non-scaling-stroke"/>';
      });

      return heroSvg(
        '<line x1="20" y1="' + floor + '" x2="500" y2="' + floor + '" stroke="#2c2c2c"/>' + bars
      ) + '<div class="hero-label" style="top:2%">SCRIPT \u00b7 ' + spectrum.length +
        ' partials</div>';
    }

    var count = Math.round(clamp(valueOf(values, "oscillator.num_sines", 32), 1, 64));
    var bend = valueOf(values, "oscillator.freq_gen.bend_amount", 0);
    var center = Math.max(valueOf(values, "oscillator.freq_gen.center_space", 1), 0.125);
    var overall = valueOf(values, "oscillator.freq_gen.total_bend_amount", 0);
    var scale = Math.max(valueOf(values, "oscillator.freq_gen.scale_amount", 1), 0.01);
    var total = Math.log(64);

    var bendCurve = function (t, amount) {
      if (Math.abs(amount) < 1e-4) return t;
      return (Math.exp(amount * t) - 1) / (Math.exp(amount) - 1);
    };

    var partials = [];
    var maxLog = 0.0;
    var maxAmp = 0.0;

    for (var i = 1; i <= count; i++) {
      var ratio = i * scale;
      var ratioLn = Math.log(ratio);
      var trunc = Math.trunc(ratioLn / center);
      var fract = ratioLn / center - trunc;
      var bentRatio = trunc * center + bendCurve(fract, bend);
      var ratioOut = Math.exp(bendCurve(bentRatio / total, overall) * total);
      var amplitude = (2 / Math.PI) / (i + 1);
      partials.push([ratioOut, amplitude]);
      maxLog = Math.max(maxLog, Math.log(Math.max(ratioOut, 1.0001)));
      maxAmp = Math.max(maxAmp, amplitude);
    }

    var base = 132;
    var height = base - 24;
    var bars = "";

    partials.forEach(function (partial) {
      var x = 20 + (Math.log(Math.max(partial[0], 1.0001)) / Math.max(maxLog, 1e-6)) * 480;
      var y = base - (partial[1] / Math.max(maxAmp, 1e-6)) * height * 0.9;
      bars += '<line x1="' + x.toFixed(1) + '" y1="' + base + '" x2="' + x.toFixed(1) +
        '" y2="' + y.toFixed(1) + '" stroke="var(--accent)" stroke-width="2" ' +
        'vector-effect="non-scaling-stroke"/>';
    });

    return heroSvg(
      '<line x1="20" y1="' + base + '" x2="500" y2="' + base + '" stroke="#2c2c2c"/>' + bars
    ) +
      '<div class="hero-label" style="top:4%">PARTIALS</div>' +
      '<div class="hero-label" style="top:4%; left:auto; right:14px">' +
        count + ' SINES</div>';
  }

  /// The stereo canvas is a list of token numbers: a page count, then per page
  /// a start point, a node count, and six numbers per node (two controls and the
  /// end point).
  function parseCanvas(shape) {
    if (!shape) return null;

    var tokens = String(shape).trim().split(/\s+/).map(Number);
    var at = 0;
    var next = function () { return tokens[at++]; };
    var pageCount = next();

    if (!(pageCount >= 1)) return null;

    var pages = [];

    for (var page = 0; page < pageCount; page++) {
      var start = [next(), next()];
      var count = next();

      if (!(count >= 1) || at + count * 6 > tokens.length) return null;

      var nodes = [];
      for (var index = 0; index < count; index++) {
        nodes.push([next(), next(), next(), next(), next(), next()]);
      }
      pages.push({ start: start, nodes: nodes });
    }

    return { pages: pages };
  }

  function serializeCanvas(model) {
    var parts = [String(model.pages.length)];
    model.pages.forEach(function (page) {
      parts.push(page.start[0] + " " + page.start[1] + " " + page.nodes.length);
      page.nodes.forEach(function (node) { parts.push(node.join(" ")); });
    });
    return parts.join(" ");
  }

  /// The canvas' [-1, 1] field inside a square view box: 160 units with a 16
  /// unit margin, so a circle stays a circle at any window shape.
  function canvasPoint(point) {
    return [80 + point[0] * 64, 80 - point[1] * 64];
  }

  // The last click on the canvas, for counting double clicks across rebuilds.
  var canvasClick = { time: 0, x: 0, y: 0 };

  function canvasPath(page) {
    var first = canvasPoint(page.start);
    var d = "M " + first[0].toFixed(1) + " " + first[1].toFixed(1);
    var previous = page.start;

    page.nodes.forEach(function (node) {
      // The six numbers are the two handle offsets and the end point: the
      // handles are measured from the ends they pull, so they follow them.
      var c1 = canvasPoint([previous[0] + node[0], previous[1] + node[1]]);
      var c2 = canvasPoint([node[4] + node[2], node[5] + node[3]]);
      var to = canvasPoint([node[4], node[5]]);

      d += " C " + c1[0].toFixed(1) + " " + c1[1].toFixed(1) + " " +
        c2[0].toFixed(1) + " " + c2[1].toFixed(1) + " " +
        to[0].toFixed(1) + " " + to[1].toFixed(1);

      previous = [node[4], node[5]];
    });

    return d;
  }

  /// The handles of one page and the lines that say which end each one pulls.
  ///
  /// Kept out of the hero so a drag can redraw this layer in place: the handles
  /// belong to the ends they pull, and dragging an end has to move them with it.
  function canvasHandles(page, index) {
    var handle = function (node, part, point) {
      var screen = canvasPoint(point);

      return '<circle class="canvas-handle" data-page="' + index + '" data-node="' + node +
        '" data-part="' + part + '" cx="' + screen[0].toFixed(1) + '" cy="' +
        screen[1].toFixed(1) + '" r="4" fill="var(--accent)"/>';
    };

    var guide = function (from, to) {
      var a = canvasPoint(from);
      var b = canvasPoint(to);

      return '<line x1="' + a[0].toFixed(1) + '" y1="' + a[1].toFixed(1) + '" x2="' +
        b[0].toFixed(1) + '" y2="' + b[1].toFixed(1) + '" stroke="#777" ' +
        'stroke-width="1" vector-effect="non-scaling-stroke"/>';
    };

    var markup = handle(-1, 0, page.start);
    var previous = page.start;

    page.nodes.forEach(function (node, nodeIndex) {
      var control1 = [previous[0] + node[0], previous[1] + node[1]];
      var control2 = [node[4] + node[2], node[5] + node[3]];
      var end = [node[4], node[5]];

      markup += handle(nodeIndex, 0, control1) + guide(previous, control1);
      markup += handle(nodeIndex, 1, control2) + guide(end, control2);
      markup += handle(nodeIndex, 2, end);
      previous = end;
    });

    return markup;
  }

  /// The drawn pages. The left and right channels are the curve's x and y, so a
  /// circle is a quadrature pair and a line is mono; the selected page is the one
  /// whose handles are shown.
  function canvasHero(values, context) {
    var state = context && context.canvas;
    var model = parseCanvas(state && state.shape);

    if (!model) {
      return heroSvg("") + '<div class="hero-empty">no canvas</div>';
    }

    // The page being drawn is the one the morph sits on, the same way the
    // spectrum page works: the knob is the position, the arrows are exact.
    var morph = Math.max(0, Math.min(1, valueOf(values, "oscillator.smooth_factor", 0)));
    var position = morph * (model.pages.length - 1);
    var from = Math.floor(position);
    var to = Math.min(from + 1, model.pages.length - 1);
    var blend = position - from;
    var selected = Math.min(Math.max(Math.round(position), 0), model.pages.length - 1);
    var paths = "";

    model.pages.forEach(function (page, index) {
      var active = index === selected;
      var blended = index === from || index === to;
      paths += '<path class="canvas-path" data-page="' + index + '" d="' + canvasPath(page) +
        '" fill="none" stroke="' +
        (active ? "var(--accent)" : blended ? "#9a9a9a" : "#555") + '" stroke-width="' +
        (active ? 2 : 1.25) + '" vector-effect="non-scaling-stroke" style="pointer-events:stroke"/>';

    });

    var pageLabel = blend > 0.001
      ? "PAGE " + (from + 1) + "\u2192" + (to + 1) + " / " + model.pages.length
      : "PAGE " + (selected + 1) + " / " + model.pages.length;

    return '<div class="canvas-hero"><div class="canvas-box">' +
        '<svg class="mcard-graph" viewBox="0 0 160 160" preserveAspectRatio="xMidYMid meet">' +
          '<rect x="16" y="16" width="128" height="128" fill="none" stroke="#2b2b2b"/>' +
          '<line x1="80" y1="16" x2="80" y2="144" stroke="#242424"/>' +
          '<line x1="16" y1="80" x2="144" y2="80" stroke="#242424"/>' +
          paths + '<g class="canvas-edit">' + canvasHandles(model.pages[selected], selected) +
          '</g>' +
        '</svg>' +
      '</div>' +
      '<div class="hero-label" style="top:4%">STEREO CANVAS \u00b7 ' + pageLabel + '</div>' +
      '<div class="hero-label" style="top:4%; left:auto; right:14px">DRAG · DBL-CLICK ADD · RIGHT-CLICK REMOVE</div>' +
      '<div class="hero-actions">' +
        '<button type="button" class="hero-action" data-action="page-prev">\u2039</button>' +
        '<button type="button" class="hero-action" data-action="page-next">\u203a</button>' +
        '<button type="button" class="hero-action" data-action="page-add">+ PAGE</button>' +
        '<button type="button" class="hero-action" data-action="page-remove">- PAGE</button>' +
      '</div>' +
      '<div class="hero-label" style="bottom:4%; left:auto; right:14px">MORPH ' +
        Math.round(morph * 100) + '%</div>' +
    '</div>';
  }

  /// The canvas editor: drag a handle, double-click to add a node to the selected
  /// page, right-click an end node to remove it, or use the page buttons.
  function wireCanvas(el, kind, values, state) {
    // The wrapper carries the drawn pages and the control setter, the same shape
    // the spectrum editor takes: the page follows the morph, the arrows move it.
    var canvasState = state && state.canvas;
    if (kind !== "canvas" || !canvasState) return;

    var model = parseCanvas(canvasState.shape);
    var svg = el.querySelector("svg");
    if (!model || !svg) return;

    var drag = null;

    var selected = function () {
      var morph = Math.max(0, Math.min(1, valueOf(values, "oscillator.smooth_factor", 0)));
      var last = model.pages.length - 1;

      return Math.min(Math.max(Math.round(morph * last), 0), last);
    };

    var setMorph = function (value) {
      value = Math.max(0, Math.min(1, value));

      if (state.set) state.set("oscillator.smooth_factor", value);
      else refresh();
    };

    var pointer = function (event) {
      var rect = svg.getBoundingClientRect();
      var x = ((event.clientX - rect.left) / Math.max(rect.width, 1)) * 160;
      var y = ((event.clientY - rect.top) / Math.max(rect.height, 1)) * 160;

      return [
        Math.max(-1, Math.min(1, (x - 80) / 64)),
        Math.max(-1, Math.min(1, (80 - y) / 64))
      ];
    };

    // Double clicks are counted here rather than left to the dblclick event: the
    // hero is rebuilt between clicks, which replaces the node under the pointer
    // and costs the browser its pair of clicks.
    var addNode = function (point) {
      var page = model.pages[selected()];
      var last = page.nodes.length
        ? [page.nodes[page.nodes.length - 1][4], page.nodes[page.nodes.length - 1][5]]
        : page.start;

      var pull = 0.5 * (point[0] - last[0]);

      page.nodes.push([
        pull, 0.0,
        -pull, 0.0,
        point[0], point[1]
      ]);

      canvasClick.time = 0;
      post();
      refresh();
    };

    var redraw = function (pageIndex) {
      var path = svg.querySelector('.canvas-path[data-page="' + pageIndex + '"]');
      if (path) path.setAttribute("d", canvasPath(model.pages[pageIndex]));

      // The handles belong to the ends they pull: moving an end moves them, and
      // the layer has to be rebuilt, not just the dragged dot.
      var editing = svg.querySelector(".canvas-edit");
      if (editing) editing.innerHTML = canvasHandles(model.pages[pageIndex], pageIndex);
    };

    var post = function () {
      var text = serializeCanvas(model);
      canvasState.shape = text;

      // The product page owns one canvas, so its POST has no node. A patch card
      // shares the page with many, and its state carries the route to use.
      if (state && state.post) {
        state.post(text);
        return;
      }

      fetch("/canvas", { method: "POST", body: text })
        .then(function (response) { return response.ok ? response.json() : null; })
        .then(function (result) {
          if (result && result.error) console.warn(result.error);
        })
        .catch(function () {});
    };

    var refresh = function () {
      el.innerHTML = canvasHero(values, { canvas: canvasState });
      wireCanvas(el, "canvas", values, state);
    };

    var select = function (page) {
      var last = model.pages.length - 1;

      setMorph(last > 0 ? page / last : 0);
    };

    // The page buttons are HTML and handled on pointerdown: the hero is rebuilt
    // on every repaint, so a click would lose its node before the release.
    el.onpointerdown = function (event) {
      var action = event.target.closest && event.target.closest("[data-action]");
      if (!action) return;

      var what = action.getAttribute("data-action");
      var last = model.pages.length - 1;
      var step = last > 0 ? 1 / last : 0;
      var page = selected();

      event.preventDefault();

      if (what === "page-prev") setMorph(Math.max(page - 1, 0) * step);
      else if (what === "page-next") setMorph(Math.min(page + 1, last) * step);
      else if (what === "page-add") {
        model.pages.push(JSON.parse(JSON.stringify(model.pages[model.pages.length - 1])));
        var added = model.pages.length - 1;
        post();
        setMorph(added / Math.max(added, 1));
      } else if (what === "page-remove" && model.pages.length > 1) {
        model.pages.splice(page, 1);
        var left = model.pages.length - 1;
        post();
        setMorph(left > 0 ? Math.min(page, left) / left : 0);
      }
    };

    svg.addEventListener("pointerdown", function (event) {
      var handle = event.target.closest && event.target.closest("[data-node]");
      var point = pointer(event);

      if (!handle) {
        var now = performance.now();
        var again = now - canvasClick.time < 400 &&
          Math.abs(point[0] - canvasClick.x) < 0.12 &&
          Math.abs(point[1] - canvasClick.y) < 0.12;

        canvasClick.time = now;
        canvasClick.x = point[0];
        canvasClick.y = point[1];

        if (again) {
          addNode(point);
          return;
        }
      }

      if (!handle) {
        var path = event.target.closest && event.target.closest("[data-page]");
        if (path) select(Number(path.getAttribute("data-page")));
        return;
      }

      drag = {
        page: Number(handle.getAttribute("data-page")),
        node: Number(handle.getAttribute("data-node")),
        part: Number(handle.getAttribute("data-part"))
      };

      el.dataset.busy = "1";
      event.preventDefault();

      if (svg.setPointerCapture) {
        try { svg.setPointerCapture(event.pointerId); } catch (error) { }
      }
    });

    svg.addEventListener("pointermove", function (event) {
      if (!drag) return;

      var point = pointer(event);
      var page = model.pages[drag.page];

      if (drag.node < 0) {
        page.start = point;
      } else {
        var node = page.nodes[drag.node];
        var previous = drag.node > 0
          ? [page.nodes[drag.node - 1][4], page.nodes[drag.node - 1][5]]
          : page.start;

        if (drag.part === 0) {
          node[0] = point[0] - previous[0];
          node[1] = point[1] - previous[1];
        } else if (drag.part === 1) {
          node[2] = point[0] - node[4];
          node[3] = point[1] - node[5];
        } else {
          node[4] = point[0];
          node[5] = point[1];
        }
      }

      redraw(drag.page);
    });

    svg.addEventListener("pointerup", function () {
      if (!drag) return;
      drag = null;
      delete el.dataset.busy;
      post();
    });

    svg.addEventListener("pointercancel", function () {
      if (!drag) return;
      drag = null;
      delete el.dataset.busy;
      post();
    });

    svg.addEventListener("contextmenu", function (event) {
      var handle = event.target.closest && event.target.closest("[data-node]");
      if (!handle) return;

      event.preventDefault();

      var pageIndex = Number(handle.getAttribute("data-page"));
      var node = Number(handle.getAttribute("data-node"));
      var part = Number(handle.getAttribute("data-part"));

      if (part !== 2 || node < 0 || model.pages[pageIndex].nodes.length <= 1) return;

      var page = model.pages[pageIndex];
      var removed = page.nodes[node];
      var previous = node > 0
        ? [page.nodes[node - 1][4], page.nodes[node - 1][5]]
        : page.start;
      var next = page.nodes[node + 1];

      if (next) {
        // The next handle was measured from the removed point; move the offset
        // so the curve stays where it was drawn.
        next[0] += removed[4] - previous[0];
        next[1] += removed[5] - previous[1];
      }

      page.nodes.splice(node, 1);
      post();
      refresh();
    });
  }

  /// The voice's unison: every copy as the same wave played at its detuned pitch,
  /// so the traces fan apart across the three periods drawn here, and the wave the
  /// copies add up to on the right. Random Phase is not drawn on the copies
  /// themselves - it would just look like noise - but as the window a copy may
  /// start in, and only while its knob is being held. The window is one band that
  /// runs on into the later periods instead of wrapping back to the left edge.
  function unisonHero(values, context) {
    var count = clamp(Math.round(valueOf(values, "unisons", 1)), 1, 32);
    var detune = clamp(valueOf(values, "unison_detune", 0), 0, 2);
    var bend = clamp(valueOf(values, "unison_bend", 0), -10, 10);
    var blend = clamp(valueOf(values, "unison_blend", 0), 0, 1);
    var randomPhase = clamp(valueOf(values, "random_phase", 0), 0, 1);
    var startPhase = clamp(valueOf(values, "phase_start", 0), 0, 1);
    var active = context && context.active;

    /// The DSP's bend: zero is untouched, positive bunches the copies towards
    /// the centre, negative towards the edges.
    function bendAt(t, amount) {
      if (Math.abs(amount) < 1e-6) return t;
      return (Math.exp(amount * t) - 1) / (Math.exp(amount) - 1);
    }

    var voices = [];
    var middle = (count - 1) / 2;

    for (var i = 0; i < count; i++) {
      var index = middle === 0 ? 0 : (i - middle) / middle;
      var bent = index >= 0 ? bendAt(index, bend) : -bendAt(-index, bend);

      voices.push({
        pitch: detune * bent,
        weight: 1 - Math.abs(index) * blend
      });
    }

    // The copies: the same three periods each, played at the pitch its detune asks
    // for. They start together, so the fan opens towards the right; Blend makes the
    // quiet copies fainter.
    var plotLeft = 12, plotRight = 128, plotMiddle = 70;
    var cycles = 3;
    // One period is also the width of the random-phase window, which is the only
    // thing Start Phase moves: the wave keeps its three-period phase drawing.
    var period = (plotRight - plotLeft) / cycles;
    var windowStart = plotLeft + startPhase * period;
    var traces = "";
    // The square only has room for a dozen readable copies; a larger unison is
    // sampled across the fan and the count is in the label.
    var drawn = Math.min(count, 12);

    for (var v = 0; v < drawn; v++) {
      var picked = drawn === 1 ? 0 : Math.round(v * (count - 1) / (drawn - 1));
      var voice = voices[picked];
      var ratio = Math.pow(2, voice.pitch / 12);
      var opacity = (0.12 + 0.5 * voice.weight).toFixed(2);
      var path = "";

      for (var p = 0; p <= 96; p++) {
        var t = p / 96;
        var px = plotLeft + t * (plotRight - plotLeft);
        var py = plotMiddle - Math.sin(2 * Math.PI * (cycles * t * ratio + startPhase)) * 30 *
          voice.weight;
        path += (p ? " L " : "M ") + px.toFixed(1) + " " + py.toFixed(1);
      }

      traces += '<path d="' + path + '" fill="none" stroke="var(--accent)" stroke-width="1" ' +
        'stroke-opacity="' + opacity + '" vector-effect="non-scaling-stroke"/>';
    }

    // Random Phase is a window on the period, not an offset on the trace: while
    // its knob is held the region a copy may start in is filled in. The first
    // copy starts at Start Phase, so the window opens there.
    var phaseWindow = "";

    if (active === "random_phase" && randomPhase > 0.001) {
      // One band from the chosen Start Phase, running into the periods after it:
      // the window is what a wrap would cut in two, not the wave.
      phaseWindow =
        '<rect x="' + windowStart.toFixed(1) + '" y="24" width="' +
        Math.min(randomPhase * period, plotRight - windowStart).toFixed(1) +
        '" height="92" fill="var(--accent)" fill-opacity=".12"/>';
    }

    // The wave the copies make together, as the host reports it. A silent
    // instrument is a flat line: this pane shows the output and nothing else.
    var scope = context && context.scope;
    var samples = scope && scope.output;
    var tracePath = "";
    var left = 8, right = 392, waveMiddle = 70, height = 52;

    var peak = 0;

    if (samples && samples.length > 64) {
      for (var m = 0; m < samples.length; m++) peak = Math.max(peak, Math.abs(samples[m]));
    }

    if (samples && samples.length > 64 && peak > 1e-4) {
      var last = samples.length - 1;

      for (var s = 0; s <= last; s++) {
        var sx = left + s / last * (right - left);
        var sy = waveMiddle - clamp(samples[s], -1.2, 1.2) / 1.2 * height;
        tracePath += (s ? " L " : "M ") + sx.toFixed(1) + " " + sy.toFixed(1);
      }
    }

    return '<div class="unison-hero">' +
      '<div class="unison-voices">' +
        '<svg viewBox="0 0 140 140" preserveAspectRatio="xMidYMid meet">' +
          '<line x1="' + plotLeft + '" y1="' + plotMiddle + '" x2="' + plotRight + '" y2="' +
            plotMiddle + '" stroke="#242424"/>' +
          phaseWindow +
          traces +
        '</svg>' +
        '<div class="hero-label" style="top:4%">UNISON</div>' +
        '<div class="hero-label" style="top:4%; left:auto; right:8px">\u00d7' + count + '</div>' +
      '</div>' +
      '<div class="unison-wave">' +
        '<svg viewBox="0 0 400 140" preserveAspectRatio="none">' +
          '<line x1="' + left + '" y1="' + waveMiddle + '" x2="' + right + '" y2="' +
            waveMiddle + '" stroke="#242424"/>' +
          (tracePath
            ? '<path d="' + tracePath + '" fill="none" stroke="var(--accent)" ' +
              'stroke-width="1.8" vector-effect="non-scaling-stroke"/>'
            : "") +
        '</svg>' +
        '<div class="hero-label" style="top:4%">OUT</div>' +
        '<div class="hero-badge">\u00b1' + detune.toFixed(1) + ' st</div>' +
      '</div>' +
    '</div>';
  }

  var spectrumWaveCache = { generation: -1, pages: {} };

  /// One period of a page, normalized to its own peak, for the preview.
  function spectrumPagePoints(frame) {
    var points = 256;
    var partials = Math.min(frame.levels.length, 512);
    var values = [];
    var peak = 1e-6;

    for (var p = 0; p < points; p++) {
      var t = p / points;
      var sum = 0;

      for (var n = 0; n < partials; n++) {
        var level = frame.levels[n] || 0;
        if (level <= 0) continue;
        sum += level * Math.sin(2 * Math.PI * ((n + 1) * t + (frame.phases[n] || 0)));
      }

      values.push(sum);
      if (Math.abs(sum) > peak) peak = Math.abs(sum);
    }

    for (var q = 0; q < points; q++) values[q] /= peak;

    return values;
  }

  function spectrumLevelsPath(frame) {
    var count = frame.levels.length;
    var parts = [];

    for (var i = 0; i < count; i++) {
      var x = 20 + (count === 1 ? 0.5 : i / (count - 1)) * 480;
      var height = Math.max(0, Math.min(1, frame.levels[i] || 0)) * 40;

      parts.push("M " + x.toFixed(1) + " " + (116 - height).toFixed(1) + " L " +
        x.toFixed(1) + " 116");
    }

    return parts.join(" ");
  }

  function spectrumPhasesPath(frame) {
    var count = frame.phases.length;
    var parts = [];

    for (var i = 0; i < count; i++) {
      var x = 20 + (count === 1 ? 0.5 : i / (count - 1)) * 480;
      var height = Math.max(0, Math.min(1, frame.phases[i] || 0)) * 16;

      parts.push("M " + x.toFixed(1) + " " + (136 - height).toFixed(1) + " L " +
        x.toFixed(1) + " 136");
    }

    return parts.join(" ");
  }

  /// The drawn spectrum: the wave Morph makes right now, the levels of the page
  /// it sits on, and that page's start phases. Pages are what Morph fades
  /// between, so the page being edited follows the knob.
  function morphSpectrumHero(values, context) {
    var state = context && context.spectrum;
    var frames = state && state.frames;

    if (!frames || !frames.length) {
      return heroSvg("") + '<div class="hero-empty">no spectrum</div>';
    }

    var morph = clamp(valueOf(values, "oscillator.morph", 0), 0, 1);
    var position = morph * (frames.length - 1);
    var from = Math.floor(position);
    var to = Math.min(from + 1, frames.length - 1);
    var blend = position - from;
    var selected = Math.min(Math.max(Math.round(position), 0), frames.length - 1);
    var frame = frames[selected];

    var cache = spectrumWaveCache;

    if (cache.generation !== state.generation) {
      cache.generation = state.generation;
      cache.pages = {};
    }

    var pointsOf = function (page) {
      if (!cache.pages[page]) cache.pages[page] = spectrumPagePoints(frames[page]);
      return cache.pages[page];
    };

    var first = pointsOf(from);
    var second = pointsOf(to);
    var path = "";

    for (var q = 0; q <= 256; q++) {
      var index = q % 256;
      var value = first[index] * (1 - blend) + second[index] * blend;

      path += (q ? " L " : "M ") + (20 + q / 256 * 480).toFixed(1) + " " +
        (39 - value * 27).toFixed(1);
    }

    var page = blend > 0.001
      ? "PAGE " + (from + 1) + "\u2192" + (to + 1) + " / " + frames.length
      : "PAGE " + (selected + 1) + " / " + frames.length;

    return heroSvg(
      '<line x1="20" y1="39" x2="500" y2="39" stroke="#1f1f1f"/>' +
      '<path d="' + path + '" fill="none" stroke="var(--accent)" stroke-width="1.8" ' +
        'vector-effect="non-scaling-stroke"/>' +
      '<line x1="20" y1="76" x2="500" y2="76" stroke="#242424"/>' +
      '<path class="spectrum-levels" d="' + spectrumLevelsPath(frame) + '" stroke="var(--accent)" ' +
        'stroke-opacity=".75" stroke-width="1" vector-effect="non-scaling-stroke"/>' +
      '<line x1="20" y1="116" x2="500" y2="116" stroke="#2c2c2c"/>' +
      '<path class="spectrum-phases" d="' + spectrumPhasesPath(frame) + '" stroke="#8a8a8a" ' +
        'stroke-width="1" vector-effect="non-scaling-stroke"/>' +
      '<line x1="20" y1="136" x2="500" y2="136" stroke="#2c2c2c"/>'
    ) +
      '<div class="hero-label" style="top:2%">SPECTRUM \u00b7 ' + frame.levels.length +
        ' PARTIALS</div>' +
      '<div class="hero-label" style="top:2%; left:auto; right:10px">' + page +
        ' \u00b7 MORPH ' + Math.round(morph * 100) + '%</div>' +
      '<div class="hero-actions">' +
        '<button type="button" class="hero-action" data-action="spectrum-prev">\u2039</button>' +
        '<button type="button" class="hero-action" data-action="spectrum-next">\u203a</button>' +
        '<button type="button" class="hero-action" data-action="spectrum-add">+ PAGE</button>' +
        '<button type="button" class="hero-action" data-action="spectrum-remove">- PAGE</button>' +
      '</div>' +
      '<div class="hero-label" style="bottom:2%; left:auto; right:10px">' +
        'DRAG LEVELS \u00b7 SHIFT DRAG PHASES</div>';
  }

  /// The spectrum editor: drag across the levels or the phase strip, add and
  /// remove pages, and post the pages back to the product.
  function wireSpectrum(el, kind, values, state) {
    // The wrapper carries the pages and the control setter: the editor edits one
    // and moves the morph through the other.
    var spectrum = state && state.spectrum;

    if (kind !== "spectrum" || !spectrum || !spectrum.frames || !spectrum.frames.length) return;

    var svg = el.querySelector("svg");
    if (!svg) return;

    var frames = spectrum.frames;
    var morph = clamp(valueOf(values, "oscillator.morph", 0), 0, 1);
    var selected = Math.min(
      Math.max(Math.round(morph * (frames.length - 1)), 0),
      frames.length - 1
    );
    var frame = frames[selected];
    var drag = null;

    var setMorph = function (value) {
      value = Math.max(0, Math.min(1, value));

      if (state.set) state.set("oscillator.morph", value);
      else {
        spectrum.generation = (spectrum.generation || 0) + 1;
        el.innerHTML = morphSpectrumHero(values, { spectrum: spectrum });
        wireSpectrum(el, "spectrum", values, state);
      }
    };

    var point = function (event) {
      var rect = svg.getBoundingClientRect();
      return {
        x: ((event.clientX - rect.left) / Math.max(rect.width, 1)) * 520,
        y: ((event.clientY - rect.top) / Math.max(rect.height, 1)) * 160
      };
    };

    var paint = function () {
      var levels = svg.querySelector(".spectrum-levels");
      var phases = svg.querySelector(".spectrum-phases");

      if (levels) levels.setAttribute("d", spectrumLevelsPath(frame));
      if (phases) phases.setAttribute("d", spectrumPhasesPath(frame));
    };

    var redraw = function () {
      spectrum.generation = (spectrum.generation || 0) + 1;
      el.innerHTML = morphSpectrumHero(values, { spectrum: spectrum });
      wireSpectrum(el, "spectrum", values, state);
    };

    var post = function () {
      fetch("/spectrum", { method: "POST", body: JSON.stringify(frames) })
        .then(function (response) { return response.ok ? response.json() : null; })
        .then(function (result) { if (result && result.error) console.warn(result.error); })
        .catch(function () {});
    };

    /// Paints the bars the pointer passed over since the last event, not just
    /// the one under it: a fast drag draws a line instead of a picket fence.
    var edit = function (event) {
      var at = point(event);
      var count = frame.levels.length;
      var index = Math.max(0, Math.min(count - 1,
        Math.round((at.x - 20) / 480 * (count - 1))));
      var phase = event.shiftKey || at.y >= 120;

      if (!phase && at.y < 84) return;

      var value = phase
        ? Math.max(0, Math.min(1, (136 - at.y) / 16))
        : Math.max(0, Math.min(1, (116 - at.y) / 40));

      if (!drag || drag.phase !== phase) {
        drag = { phase: phase, index: index, value: value };
      }

      var steps = Math.abs(index - drag.index);
      var direction = index >= drag.index ? 1 : -1;

      for (var step = 0; step <= steps; step++) {
        var partial = drag.index + direction * step;
        var t = steps === 0 ? 1 : step / steps;
        var filled = drag.value + (value - drag.value) * t;

        if (phase) frame.phases[partial] = filled;
        else frame.levels[partial] = filled;
      }

      drag.index = index;
      drag.value = value;
      paint();
    };

    // One handler, not one per repaint, and on pointerdown rather than click: a
    // repaint replaces the button between press and release, so a click would
    // never land. The arrows set the exact morph of the page they move to.
    el.onpointerdown = function (event) {
      var action = event.target.closest && event.target.closest("[data-action]");
      if (!action) return;

      var what = action.getAttribute("data-action");
      var last = frames.length - 1;
      var step = last > 0 ? 1 / last : 0;

      event.preventDefault();

      if (what === "spectrum-prev") setMorph(Math.max(selected - 1, 0) * step);
      else if (what === "spectrum-next") setMorph(Math.min(selected + 1, last) * step);
      else if (what === "spectrum-add") {
        frames.splice(selected + 1, 0, {
          levels: frame.levels.slice(),
          phases: frame.phases.slice()
        });
        spectrum.generation = (spectrum.generation || 0) + 1;
        setMorph((selected + 1) / Math.max(frames.length - 1, 1));
        post();
      } else if (what === "spectrum-remove" && frames.length > 1) {
        frames.splice(selected, 1);
        spectrum.generation = (spectrum.generation || 0) + 1;
        setMorph(Math.min(selected, frames.length - 1) / Math.max(frames.length - 1, 1));
        post();
      }
    };

    svg.addEventListener("pointerdown", function (event) {
      event.preventDefault();
      el.dataset.busy = "1";
      drag = null;
      edit(event);

      if (svg.setPointerCapture) {
        try { svg.setPointerCapture(event.pointerId); } catch (error) {}
      }
    });

    svg.addEventListener("pointermove", function (event) {
      if (el.dataset.busy !== "1") return;
      event.preventDefault();
      edit(event);
    });

    var finish = function () {
      if (el.dataset.busy !== "1") return;
      el.dataset.busy = "";
      drag = null;
      post();
      redraw();
    };

    svg.addEventListener("pointerup", finish);
    svg.addEventListener("pointercancel", finish);
  }
