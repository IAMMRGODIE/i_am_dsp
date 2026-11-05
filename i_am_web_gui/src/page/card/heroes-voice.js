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
        'stroke="var(--accent)" stroke-opacity=".22" stroke-dasharray="2 4"/>' +
      '<circle cx="' + x.toFixed(1) + '" cy="' + (middle - peak) + '" r="3" ' +
        'fill="var(--accent)"/>'
    ) +
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
    var preset = Math.round(valueOf(values, "preset", 0));

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
    var names = ["GENTLEMAN", "PEEL", "BASS", "CELESTE", "CLEAN GTR", "HARP"];
    var shapes = [
      { levels: [0.62, 0.46, 0.34, 0.26, 0.19, 0.14, 0.1, 0.07],
        decays: [0.95, 0.6, 0.4, 0.52, 0.34, 0.28, 0.22, 0.18] },
      { levels: [0.5, 0.43, 0.37, 0.31, 0.25, 0.2, 0.16, 0.12],
        decays: [0.55, 0.46, 0.38, 0.31, 0.26, 0.21, 0.17, 0.13] },
      { levels: [0.7, 0.52, 0.4, 0.3, 0.22, 0.16, 0.12, 0.09],
        decays: [1.1, 0.8, 0.6, 0.45, 0.34, 0.26, 0.2, 0.16] },
      { levels: [0.5, 0.4, 0.32, 0.26, 0.2, 0.15, 0.11, 0.08],
        decays: [0.45, 0.38, 0.3, 0.24, 0.19, 0.15, 0.12, 0.1] },
      { levels: [0.55, 0.46, 0.38, 0.3, 0.23, 0.18, 0.14, 0.1],
        decays: [0.5, 0.42, 0.34, 0.28, 0.22, 0.18, 0.14, 0.11] },
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
  function shapeHero(values) {
    var morph = clamp(valueOf(values, "oscillator.smooth_factor", 0), 0, 1);
    var names = ["SINE", "TRIANGLE", "SAW", "SQUARE"];
    var tables = [
      function (x) { return Math.sin(2 * Math.PI * x); },
      function (x) { return 1 - 4 * Math.abs(x - 0.5); },
      function (x) { return 2 * x - 1; },
      function (x) { return x < 0.5 ? 1 : -1; }
    ];
    var smooth = morph * (tables.length - 1);
    var from = Math.min(Math.floor(smooth), tables.length - 1);
    var to = Math.min(from + 1, tables.length - 1);
    var blend = smooth - from;
    var path = "";

    for (var i = 0; i <= 240; i++) {
      var t = i / 240;
      var x = 20 + t * 480;
      var phase = (t * 2) % 1;
      var value = tables[from](phase) + (tables[to](phase) - tables[from](phase)) * blend;
      var y = 80 - value * 54;
      path += (i ? " L " : "M ") + x.toFixed(1) + " " + y.toFixed(1);
    }

    var caption = from === to
      ? names[from]
      : names[from] + " → " + names[to] + " " + Math.round(blend * 100) + "%";

    return heroSvg(
      '<line x1="20" y1="80" x2="500" y2="80" stroke="#2c2c2c" stroke-dasharray="3 5"/>' +
      '<path d="' + path + '" fill="none" stroke="var(--accent)" stroke-width="2" ' +
        'vector-effect="non-scaling-stroke"/>'
    ) +
      '<div class="hero-label" style="top:4%">SHAPE</div>' +
      '<div class="hero-label" style="top:4%; left:auto; right:14px">' + caption + '</div>';
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
  function spectrumHero(values) {
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
  function additiveHero(values) {
    if (Object.prototype.hasOwnProperty.call(values, "oscillator.freq_gen.tilt")) {
      return spectrumHero(values);
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

  /// The canvas' [-1, 1] square in the hero's view box.
  function canvasPoint(point) {
    return [20 + (point[0] + 1) * 240, 80 - point[1] * 60];
  }

  function canvasPath(page) {
    var first = canvasPoint(page.start);
    var d = "M " + first[0].toFixed(1) + " " + first[1].toFixed(1);

    page.nodes.forEach(function (node) {
      var c1 = canvasPoint([node[0], node[1]]);
      var c2 = canvasPoint([node[2], node[3]]);
      var to = canvasPoint([node[4], node[5]]);
      d += " C " + c1[0].toFixed(1) + " " + c1[1].toFixed(1) + " " +
        c2[0].toFixed(1) + " " + c2[1].toFixed(1) + " " +
        to[0].toFixed(1) + " " + to[1].toFixed(1);
    });

    return d;
  }

  /// The drawn pages. The left and right channels are the curve's x and y, so a
  /// circle is a quadrature pair and a line is mono; the selected page is the one
  /// whose handles are shown.
  function canvasHero(values, context) {
    var state = context && context.canvas;
    var model = parseCanvas(state && state.shape);

    if (!model) {
      return heroSvg('<text x="260" y="84" fill="#6a6a6a" font-size="12" text-anchor="middle">no canvas</text>');
    }

    var selected = Math.min(Math.max((state && state.page) || 0, 0), model.pages.length - 1);
    var handle = function (page, node, part, point) {
      var screen = canvasPoint(point);
      return '<circle class="canvas-handle" data-page="' + page + '" data-node="' + node +
        '" data-part="' + part + '" cx="' + screen[0].toFixed(1) + '" cy="' +
        screen[1].toFixed(1) + '" r="4" fill="var(--accent)"/>';
    };

    var paths = "";
    var handles = "";

    model.pages.forEach(function (page, index) {
      var active = index === selected;
      paths += '<path class="canvas-path" data-page="' + index + '" d="' + canvasPath(page) +
        '" fill="none" stroke="' + (active ? "var(--accent)" : "#666") + '" stroke-width="' +
        (active ? 2 : 1.25) + '" vector-effect="non-scaling-stroke" style="pointer-events:stroke"/>';

      if (!active) return;

      handles += handle(index, -1, 0, page.start);
      page.nodes.forEach(function (node, nodeIndex) {
        handles += handle(index, nodeIndex, 0, [node[0], node[1]]);
        handles += handle(index, nodeIndex, 1, [node[2], node[3]]);
        handles += handle(index, nodeIndex, 2, [node[4], node[5]]);
      });
    });

    return heroSvg(
      '<rect x="20" y="18" width="480" height="124" fill="none" stroke="#2b2b2b"/>' +
      '<line x1="260" y1="18" x2="260" y2="142" stroke="#242424"/>' +
      '<line x1="20" y1="80" x2="500" y2="80" stroke="#242424"/>' +
      paths + handles +
      '<g data-action="page-add" style="cursor:pointer;pointer-events:all">' +
        '<text x="20" y="156" fill="#8a8a8a" font-size="10" font-family="monospace">+ PAGE</text></g>' +
      '<g data-action="page-remove" style="cursor:pointer;pointer-events:all">' +
        '<text x="500" y="156" fill="#8a8a8a" font-size="10" text-anchor="end" font-family="monospace">- PAGE</text></g>'
    ) +
      '<div class="hero-label" style="top:4%">STEREO CANVAS</div>' +
      '<div class="hero-label" style="top:4%; left:auto; right:14px">DRAG · DBL-CLICK ADD · RIGHT-CLICK REMOVE</div>';
  }

  /// The canvas editor: drag a handle, double-click to add a node to the selected
  /// page, right-click an end node to remove it, or use the page buttons.
  function wireCanvas(el, kind, values, canvasState) {
    if (kind !== "canvas" || !canvasState) return;

    var model = parseCanvas(canvasState.shape);
    var svg = el.querySelector("svg");
    if (!model || !svg) return;

    var drag = null;
    var selected = function () {
      return Math.min(Math.max(canvasState.page || 0, 0), model.pages.length - 1);
    };

    var pointer = function (event) {
      var rect = svg.getBoundingClientRect();
      var x = ((event.clientX - rect.left) / Math.max(rect.width, 1)) * 520;
      var y = ((event.clientY - rect.top) / Math.max(rect.height, 1)) * 160;
      return [
        Math.max(-1, Math.min(1, (x - 20) / 240 - 1)),
        Math.max(-1, Math.min(1, (80 - y) / 60))
      ];
    };

    var redraw = function (pageIndex) {
      var path = svg.querySelector('.canvas-path[data-page="' + pageIndex + '"]');
      if (path) path.setAttribute("d", canvasPath(model.pages[pageIndex]));
    };

    var post = function () {
      var text = serializeCanvas(model);
      canvasState.shape = text;

      fetch("/canvas", { method: "POST", body: text })
        .then(function (response) { return response.ok ? response.json() : null; })
        .then(function (result) {
          if (result && result.error) console.warn(result.error);
        })
        .catch(function () {});
    };

    var refresh = function () {
      el.innerHTML = canvasHero(values, { canvas: canvasState });
      wireCanvas(el, "canvas", values, canvasState);
    };

    var select = function (page) {
      canvasState.page = page;
      refresh();
    };

    svg.addEventListener("pointerdown", function (event) {
      var action = event.target.closest && event.target.closest("[data-action]");

      if (action) {
        var what = action.getAttribute("data-action");

        if (what === "page-add") {
          model.pages.push(JSON.parse(JSON.stringify(model.pages[model.pages.length - 1])));
          canvasState.page = model.pages.length - 1;
          post();
          refresh();
        } else if (what === "page-remove" && model.pages.length > 1) {
          var removing = selected();
          model.pages.splice(removing, 1);
          canvasState.page = Math.min(removing, model.pages.length - 1);
          post();
          refresh();
        }

        event.preventDefault();
        return;
      }

      var handle = event.target.closest && event.target.closest("[data-node]");
      if (!handle) {
        var path = event.target.closest && event.target.closest("[data-page]");
        if (path) select(Number(path.getAttribute("data-page")));
        return;
      }

      drag = {
        page: Number(handle.getAttribute("data-page")),
        node: Number(handle.getAttribute("data-node")),
        part: Number(handle.getAttribute("data-part")),
        element: handle
      };

      canvasState.page = drag.page;
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
        node[drag.part * 2] = point[0];
        node[drag.part * 2 + 1] = point[1];
      }

      var screen = canvasPoint(point);
      drag.element.setAttribute("cx", screen[0].toFixed(1));
      drag.element.setAttribute("cy", screen[1].toFixed(1));
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

    svg.addEventListener("dblclick", function (event) {
      var handle = event.target.closest && event.target.closest("[data-node]");
      if (handle) return;

      var page = model.pages[selected()];
      var point = pointer(event);
      var last = page.nodes.length
        ? [page.nodes[page.nodes.length - 1][4], page.nodes[page.nodes.length - 1][5]]
        : page.start;

      page.nodes.push([
        0.5 * (last[0] + point[0]), last[1],
        0.5 * (last[0] + point[0]), point[1],
        point[0], point[1]
      ]);

      post();
      refresh();
    });

    svg.addEventListener("contextmenu", function (event) {
      var handle = event.target.closest && event.target.closest("[data-node]");
      if (!handle) return;

      event.preventDefault();

      var pageIndex = Number(handle.getAttribute("data-page"));
      var node = Number(handle.getAttribute("data-node"));
      var part = Number(handle.getAttribute("data-part"));

      if (part !== 2 || node < 0 || model.pages[pageIndex].nodes.length <= 1) return;

      model.pages[pageIndex].nodes.splice(node, 1);
      post();
      refresh();
    });
  }