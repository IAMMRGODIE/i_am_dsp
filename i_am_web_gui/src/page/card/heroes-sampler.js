// The sampler, the envelope and the sines pictures.
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
