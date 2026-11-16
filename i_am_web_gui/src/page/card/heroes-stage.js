// Part six: the pictures for the delay, the modulation, the resonator and the autotune.
  function stereoHero(values, context) {
    var scope = (context && context.stereo) || {};
    var cx = 160;
    var cy = 72;
    var radius = 62;
    var grid = "";

    for (var r = 20; r <= 60; r += 20) {
      grid += '<circle cx="' + cx + '" cy="' + cy + '" r="' + r +
        '" fill="none" stroke="#3a3a3a" stroke-width="1"/>';
    }

    grid += '<line x1="' + (cx - 62) + '" y1="' + cy + '" x2="' + (cx + 62) + '" y2="' +
      cy + '" stroke="#2f2f2f"/>' +
      '<line x1="' + cx + '" y1="' + (cy - 62) + '" x2="' + cx + '" y2="' + (cy + 62) +
      '" stroke="#2f2f2f"/>';

    var diagonal = 62 / Math.SQRT2;
    grid += '<line x1="' + (cx - diagonal).toFixed(1) + '" y1="' + (cy - diagonal).toFixed(1) +
      '" x2="' + (cx + diagonal).toFixed(1) + '" y2="' + (cy + diagonal).toFixed(1) +
      '" stroke="#2a2a2a"/>' +
      '<line x1="' + (cx - diagonal).toFixed(1) + '" y1="' + (cy + diagonal).toFixed(1) +
      '" x2="' + (cx + diagonal).toFixed(1) + '" y2="' + (cy - diagonal).toFixed(1) +
      '" stroke="#2a2a2a"/>';

    var input = "";
    var output = "";
    var labels = "";
    var points = scope.inputLeft ? scope.inputLeft.length : 0;

    if (points > 1) {
      var peak = 0.4;

      for (var p = 0; p < points; p += 2) {
        peak = Math.max(
          peak,
          Math.abs(scope.inputLeft[p]),
          Math.abs(scope.inputRight[p]),
          Math.abs(scope.outputLeft[p]),
          Math.abs(scope.outputRight[p])
        );
      }

      var scale = radius / peak;
      var step = Math.max(1, Math.floor(points / 512));

      for (var i = 0; i < points; i += step) {
        var il = scope.inputLeft[i];
        var ir = scope.inputRight[i];
        var ol = scope.outputLeft[i];
        var or = scope.outputRight[i];

        input += '<circle cx="' + (cx + (il - ir) * 0.5 * scale).toFixed(1) +
          '" cy="' + (cy - (il + ir) * 0.5 * scale).toFixed(1) +
          '" r="1.1" fill="#a8a8a8" fill-opacity=".42"/>';
        output += '<circle cx="' + (cx + (ol - or) * 0.5 * scale).toFixed(1) +
          '" cy="' + (cy - (ol + or) * 0.5 * scale).toFixed(1) +
          '" r="1.2" fill="var(--accent)" fill-opacity=".52"/>';
      }

      labels =
        '<text x="238" y="24" fill="#8a8a8a" font-size="8">IN</text>' +
        '<text x="278" y="24" fill="var(--accent)" font-size="8">OUT</text>';
    }

    var corrInput = correlation(scope.inputLeft, scope.inputRight);
    var corrOutput = correlation(scope.outputLeft, scope.outputRight);
    var barX = 60;
    var barY = 164;
    var barWidth = 200;
    var barHeight = 8;
    var bar =
      '<text x="' + barX + '" y="' + (barY - 6) +
        '" fill="#8a8a8a" font-size="7">CORR</text>' +
      '<text x="' + (barX + 42) + '" y="' + (barY - 6) +
        '" fill="#a8a8a8" font-size="7">IN ' + corrInput.toFixed(2) + '</text>' +
      '<text x="' + (barX + 126) + '" y="' + (barY - 6) +
        '" fill="var(--accent)" font-size="7">OUT ' + corrOutput.toFixed(2) + '</text>' +
      '<rect x="' + barX + '" y="' + barY + '" width="' + barWidth + '" height="' + barHeight +
        '" rx="4" fill="#0b0b0b" stroke="#3a3a3a"/>' +
      '<rect x="' + (barX + 1) + '" y="' + (barY + 1) + '" width="' + (barWidth - 2) +
        '" height="' + (barHeight - 2) + '" rx="3" fill="#050505"/>' +
      '<line x1="' + (barX + barWidth / 2) + '" y1="' + barY + '" x2="' + (barX + barWidth / 2) +
        '" y2="' + (barY + barHeight) + '" stroke="#3a3a3a"/>';

    bar +=
      '<circle cx="' + (barX + (corrInput + 1) * 0.5 * barWidth).toFixed(1) + '" cy="' +
        (barY + barHeight / 2) + '" r="2.2" fill="#a8a8a8"/>' +
      '<circle cx="' + (barX + (corrOutput + 1) * 0.5 * barWidth).toFixed(1) + '" cy="' +
        (barY + barHeight / 2) + '" r="2.2" fill="var(--accent)"/>' +
      '<text x="' + barX + '" y="' + (barY + 20) +
        '" fill="#6a6a6a" font-size="7">-1</text>' +
      '<text x="' + (barX + barWidth / 2 - 3) + '" y="' + (barY + 20) +
        '" fill="#6a6a6a" font-size="7">0</text>' +
      '<text x="' + (barX + barWidth - 8) + '" y="' + (barY + 20) +
        '" fill="#6a6a6a" font-size="7">+1</text>';

    return heroSvgMeter(grid + input + output + labels + bar);
  }

  function tailHero(values) {
    var decay = clamp(valueOf(values, "decay_time", 2200), 100, 10000);
    var pre = clamp(valueOf(values, "pre_delay", 20), 0, 200);
    var size = clamp(valueOf(values, "size", 10), 1, 50);
    var damping = clamp(valueOf(values, "damping", 8000), 200, 20000);
    var diffusion = clamp(valueOf(values, "diffusion", 0.7), 0, 1);
    var total = Math.max(pre + decay * 1.15, 100) * 1.02;
    var xAt = function (ms) { return 20 + ms / total * 480; };
    var yAt = function (level) { return 150 - level * 140; };
    var preX = xAt(pre);
    var ticks = "";
    var tail = "";
    var high = "";

    for (var k = 0; k < 5; k++) {
      var at = pre + size * (0.35 + k * 0.55);
      if (at >= total) break;
      var height = (0.55 - k * 0.08) * (0.65 + diffusion * 0.35);
      ticks += '<line x1="' + xAt(at).toFixed(1) + '" y1="150" x2="' + xAt(at).toFixed(1) +
        '" y2="' + yAt(height).toFixed(1) + '" stroke="var(--accent)" ' +
        'stroke-width="2" opacity="' + (0.35 + k * 0.1).toFixed(2) + '"/>';
    }

    for (var i = 0; i <= 180; i++) {
      var t = pre + (total - pre) * i / 180;
      var level = Math.exp(-6.9 * (t - pre) / decay);
      tail += (i ? " L " : "M ") + xAt(t).toFixed(1) + " " + yAt(level).toFixed(1);
    }

    var hfDecay = decay * (0.35 + 0.65 * clamp(damping / 20000, 0.1, 1));

    for (var j = 0; j <= 180; j++) {
      var ht = pre + (total - pre) * j / 180;
      var hlevel = Math.exp(-6.9 * (ht - pre) / hfDecay);
      high += (j ? " L " : "M ") + xAt(ht).toFixed(1) + " " + yAt(hlevel).toFixed(1);
    }

    return heroSvg(
      '<line x1="20" y1="150" x2="500" y2="150" stroke="#3a3a3a"/>' +
      (pre > 0
        ? '<line x1="20" y1="150" x2="' + preX.toFixed(1) + '" y2="150" stroke="#343434"/>'
        : "") +
      '<path d="' + tail + ' L ' + xAt(total).toFixed(1) + ' 150 L ' + xAt(pre).toFixed(1) +
        ' 150 Z" fill="var(--accent)" fill-opacity=".07"/>' +
      '<path d="' + tail + '" fill="none" stroke="var(--accent)" stroke-width="2" ' +
        'vector-effect="non-scaling-stroke"/>' +
      '<path d="' + high + '" fill="none" stroke="#ffffff" stroke-opacity=".22" ' +
        'stroke-width="1" vector-effect="non-scaling-stroke"/>' +
      ticks
    ) + (pre > 0 ? '<div class="hero-label" style="left:4.6%;top:88%">pre</div>' : "") +
      '<div class="hero-badge">T60 ' + (decay / 1000).toFixed(1) + ' s</div>';
  }

  function tapsHero(values) {
    var sync = !!valueOf(values, "tempo.sync", 0);
    var beats = Math.max(valueOf(values, "tempo.beats", 1), 0.03125);
    var free = Math.max(valueOf(values, "delay_time", 250), 0);
    var time = sync ? beats * 500 : free;
    var feedback = clamp(valueOf(values, "decay_factor", 0.8), 0, 0.95);
    var mix = clamp(valueOf(values, "wet_gain", 1), 0, 2);
    var base = 140;
    var taps = "";
    var dots = "";

    for (var tap = 0; tap < 8; tap++) {
      var at = time * (tap + 1);
      if (at <= 0 || at > 2000) break;

      var x = 20 + at / 2000 * 480;
      var level = Math.pow(feedback, tap) * Math.min(mix, 1);
      var height = 16 + 100 * level;

      taps += '<line x1="' + x.toFixed(1) + '" y1="' + base + '" x2="' + x.toFixed(1) +
        '" y2="' + (base - height).toFixed(1) + '" stroke="var(--accent)" stroke-width="2" ' +
        'opacity="' + (0.3 + 0.7 * level).toFixed(2) + '"/>';

      dots += '<div class="hero-pip" style="left:' + (x / 5.2).toFixed(2) + '%;top:' +
        ((base - height) / 1.6).toFixed(2) + '%;width:4px;height:4px;background:var(--accent);' +
        'opacity:' + (0.3 + 0.7 * level).toFixed(2) + '"></div>';
    }

    return heroSvg(
      '<line x1="20" y1="' + base + '" x2="500" y2="' + base + '" stroke="#3a3a3a"/>' +
      taps
    ) + dots + (sync ? '<div class="hero-badge">SYNC</div>' : "");
  }

  /// The LFO as it is set now: the free rate in hertz, or the beat period converted
  /// to hertz at 120 BPM for the picture.
  function lfoHero(values) {
    var sync = !!valueOf(values, "tempo.sync", 0);
    var beats = Math.max(valueOf(values, "tempo.beats", 1), 0.03125);
    var free = Math.max(valueOf(values, "lfo_frequency", valueOf(values, "lfo_freq", 1)), 0.01);
    var rate = clamp(sync ? 1 / (beats * 0.5) : free, 0.01, 20);
    var amplitude = valueOf(values, "lfo_amplitude", 0);
    var depth = amplitude > 0 ? clamp(amplitude / 10, 0.1, 1) : 0.6;
    var swing = 45 * depth;
    var path = "";

    for (var i = 0; i <= 520; i++) {
      var t = i / 520 * 2.0;
      var x = 20 + i / 520 * 480;
      var y = 80 - Math.sin(t * rate * Math.PI * 2) * swing;
      path += (i ? " L " : "M ") + x.toFixed(1) + " " + y.toFixed(1);
    }

    return heroSvg(
      '<line x1="20" y1="80" x2="500" y2="80" stroke="#2f2f2f" stroke-dasharray="3 5"/>' +
      '<path d="' + path + '" fill="none" stroke="var(--accent)" stroke-width="2" ' +
      'vector-effect="non-scaling-stroke"/>'
    ) + (sync ? '<div class="hero-badge">SYNC</div>' : "");
  }

  function phaseHero(values) {
    var cutoff = Math.max(valueOf(values, "cutoff", 440), 10);
    var bandwidth = Math.max(valueOf(values, "bandwidth", 1000), 10);
    var amount = clamp(valueOf(values, "amount", 0.5), 0, 1);
    var width = Math.max(0.15, Math.abs(Math.log2(bandwidth / cutoff)) * 0.35);
    var strength = 0.35 + 0.65 * amount;
    var path = "";

    for (var i = 0; i <= 120; i++) {
      var frequency = 20 * Math.pow(1000, i / 120);
      var octaves = Math.log2(frequency / cutoff);
      var delay = 1 / (1 + Math.pow(octaves / width, 2)) * strength;
      var x = 20 + i / 120 * 480;
      var y = 150 - delay * 120;
      path += (i ? " L " : "M ") + x.toFixed(1) + " " + y.toFixed(1);
    }

    return heroSvg(
      '<line x1="20" y1="150" x2="500" y2="150" stroke="#2a2a2a"/>' +
      '<path d="' + path + ' L 500 150 L 20 150 Z" fill="var(--accent)" fill-opacity=".08"/>' +
      '<path d="' + path + '" fill="none" stroke="var(--accent)" stroke-width="2" ' +
      'vector-effect="non-scaling-stroke"/>'
    );
  }

  /// Two channel gains as meters; balance is independent left and right levels.
  function balanceHero(values) {
    var left = clamp(valueOf(values, "left_gain", 1), 0, 4);
    var right = clamp(valueOf(values, "right_gain", 1), 0, 4);
    var base = 140;
    var top = 20;
    var full = base - top;
    var height = function (gain) { return Math.max(4, clamp(gain / 2, 0, 1) * full); };
    var lh = height(left);
    var rh = height(right);
    var unity = base - height(1);

    var meter = function (x, value) {
      return '<rect x="' + x + '" y="' + top + '" width="50" height="' + full +
        '" rx="3" fill="#242424"/>' +
        '<rect x="' + x + '" y="' + (base - value).toFixed(1) + '" width="50" height="' +
        value.toFixed(1) + '" rx="3" fill="var(--accent)" fill-opacity=".22" stroke="' +
        'var(--accent)" stroke-width="1.4"/>';
    };

    return heroSvg(
      '<line x1="20" y1="' + unity.toFixed(1) + '" x2="500" y2="' + unity.toFixed(1) +
        '" stroke="#3a3a3a" stroke-dasharray="3 5"/>' +
      '<line x1="20" y1="' + base + '" x2="500" y2="' + base + '" stroke="#3d3d3d"/>' +
      meter(190, lh) +
      meter(280, rh)
    );
  }

  function waveformHero(values, context) {
    var scope = context && context.scope;

    if (scope && scope.input && scope.output) {
      return heroSvg(
        '<line x1="0" y1="80" x2="520" y2="80" stroke="#2c2c2c" stroke-width="1"/>' +
        '<path d="' + trace(scope.input, 80, 58, 8) + '" fill="none" stroke="#6a6a6a" ' +
          'stroke-width="1.5" vector-effect="non-scaling-stroke"/>' +
        '<path d="' + trace(scope.output, 80, 58, 8) + '" fill="none" stroke="var(--accent)" ' +
          'stroke-width="2" vector-effect="non-scaling-stroke"/>'
      );
    }

    var gain = valueOf(values, "gain", 1);
    var db = 20 * Math.log10(Math.max(gain, 1e-4));
    var output = clamp(20 * Math.pow(10, db / 20), 3, 58);
    var input = wavePath(16, 80, 3);

    return heroSvg(
      '<line x1="0" y1="80" x2="520" y2="80" stroke="#2c2c2c" stroke-width="1"/>' +
      '<path d="' + input + ' L 500 80 L 20 80 Z" fill="#8a8a8a" fill-opacity=".14"/>' +
      '<path d="' + input + '" fill="none" stroke="#6a6a6a" stroke-width="1.5" ' +
        'vector-effect="non-scaling-stroke"/>' +
      '<path d="' + wavePath(output, 80, 3) + '" fill="none" stroke="var(--accent)" ' +
        'stroke-width="2" vector-effect="non-scaling-stroke"/>'
    );
  }

  function resonatorHero(values) {
    var note = clamp(valueOf(values, "note", 60), 0, 127);
    var decay = clamp(valueOf(values, "decay", 1.5), 0.05, 10);
    var damping = clamp(valueOf(values, "damping", 8000), 200, 18000);
    var saw = clamp(valueOf(values, "saw", 0.25), 0, 1);
    var frequency = 440 * Math.pow(2, (note - 69) / 12);
    var names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"];
    var noteName = names[Math.round(note) % 12] + (Math.floor(note / 12) - 1);
    var brightness = clamp(damping / 18000, 0.08, 1);
    var ring = clamp(decay / 4, 0.15, 1);
    var bars = "";

    for (var harmonic = 1; harmonic <= 64; harmonic++) {
      var partial = frequency * harmonic;
      if (partial > 20000) break;
      var x = 20 + Math.log(partial / 20) / Math.log(1000) * 480;
      var height = 90 * Math.pow(1 / harmonic, 0.55) * (0.35 + 0.65 * brightness) *
        (0.45 + 0.55 * ring);
      bars += '<rect x="' + (x - 1.2).toFixed(1) + '" y="' + (146 - height).toFixed(1) +
        '" width="2.4" height="' + height.toFixed(1) + '" fill="var(--accent)" fill-opacity="' +
        (0.25 + 0.75 / harmonic).toFixed(3) + '"/>';
    }

    return heroSvg(
      '<line x1="20" y1="146" x2="500" y2="146" stroke="#3a3a3a"/>' +
      '<line x1="20" y1="14" x2="20" y2="146" stroke="#2b2b2b"/>' +
      bars +
      '<line x1="20" y1="' + (146 - 72 * saw).toFixed(1) + '" x2="500" y2="' +
        (146 - 72 * saw).toFixed(1) + '" stroke="#8a8a8a" stroke-dasharray="3 5"/>'
    ) + '<div class="hero-label" style="left:2%;top:2%">SAW EXCITE</div>' +
      '<div class="hero-badge">' + noteName + '</div>';
  }

  function formantHero(values) {
    var vowels = ["A", "E", "I", "O", "U"];
    var formants = [
      [800, 1150, 2900],
      [400, 1700, 2600],
      [350, 1700, 2700],
      [450, 800, 2830],
      [325, 700, 2530]
    ];
    var vowel = clamp(Math.round(valueOf(values, "vowel", 0)), 0, 4);
    var shift = clamp(valueOf(values, "shift", 0), -24, 24);
    var resonance = clamp(valueOf(values, "resonance", 4), 0.5, 12);
    var factor = Math.pow(2, shift / 12);
    var centers = formants[vowel].map(function (frequency) { return frequency * factor; });
    var path = "";
    var markers = "";

    for (var index = 0; index <= 240; index++) {
      var frequency = 20 * Math.pow(1000, index / 240);
      var amplitude = 0;

      centers.forEach(function (center, band) {
        var width = 0.28 + 0.16 * band;
        var distance = Math.log(frequency / center) / width;
        amplitude += 1 / (1 + distance * distance) * (1.0 - 0.14 * band);
      });

      amplitude = Math.min(amplitude, 1.6);
      var x = 20 + index / 240 * 480;
      var y = 146 - amplitude * 82;
      path += (index ? " L " : "M ") + x.toFixed(1) + " " + y.toFixed(1);
    }

    var labelsHtml = "";

    centers.forEach(function (center, band) {
      var x = 20 + Math.log(clamp(center, 20, 20000) / 20) / Math.log(1000) * 480;
      markers += '<line x1="' + x.toFixed(1) + '" y1="12" x2="' + x.toFixed(1) +
        '" y2="146" stroke="var(--accent)" stroke-opacity=".25" stroke-dasharray="2 4"/>';
      labelsHtml += heroLabel(x, 14 + band * 12, "F" + (band + 1),
        "color:var(--accent);font-size:10px;transform:translateX(-50%)");
    });

    return heroSvg(
      '<line x1="20" y1="146" x2="500" y2="146" stroke="#3a3a3a"/>' +
      '<path d="' + path + '" fill="none" stroke="var(--accent)" stroke-width="2" ' +
        'vector-effect="non-scaling-stroke"/>' +
      markers
    ) + labelsHtml +
      '<div class="hero-label" style="left:2%;top:2%">' + vowels[vowel] + '</div>' +
      '<div class="hero-badge">' + (shift > 0 ? "+" : "") + shift.toFixed(0) + ' st</div>';
  }

  function autotuneHero(values, context) {
    var labels = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"];
    var labelsHtml = "";
    var live = context && context.autotune && context.autotune.notes;
    var mask = Math.round(live >= 0 ? live : valueOf(values, "notes", 4095)) & 4095;
    var strength = clamp(valueOf(values, "strength", 1), 0, 1);
    var cells = "";

    labels.forEach(function (label, index) {
      var on = (mask & (1 << index)) !== 0;
      var sharp = label.length > 1;
      var x = 20 + index * 40;
      var fill = on ? "var(--accent)" : (sharp ? "#202020" : "#2a2a2a");
      var opacity = on && sharp ? ' fill-opacity=".5"' : "";
      var text = on ? (sharp ? "var(--text1)" : "#111") : "var(--text3)";
      labelsHtml += heroLabel(x + 18, 143, label,
        "color:" + text + ";transform:translateX(-50%)");

      cells += '<rect x="' + x + '" y="26" width="36" height="104" rx="4" ' +
        'fill="' + fill + '"' + opacity +
        ' stroke="' + (on ? "var(--accent)" : "#414141") + '"/>' +
        '';
    });

    return heroSvg(cells) + labelsHtml +
      '<div class="hero-label" style="left:2%;top:2%">ALLOWED NOTES</div>' +
      '<div class="hero-badge">' + Math.round(strength * 100) + '%</div>';
  }

  /// The history time map: one page at a time, with the same 300x64 editor the
  /// LFO uses.
  function historySelected(values, pages) {
    return IAmPage.index(valueOf(values, "morph", 0), pages.length);
  }

  function historyPath(points) {
    if (!points || !points.length) return '';

    var sorted = points.slice().sort(function (a, b) { return a.time - b.time; });
    var path = "";
    var samples = 36;

    for (var index = 0; index < sorted.length - 1; index++) {
      var left = sorted[index];
      var right = sorted[index + 1];

      for (var step = 0; step <= samples; step++) {
        var unit = step / samples;
        var time = left.time + (right.time - left.time) * unit;
        var value = left.value + (right.value - left.value) *
          IAmCard.sources.bend(unit, left.bend);
        var x = 8 + Math.max(0, Math.min(1, time)) * 284;
        var y = 60 - Math.max(0, Math.min(1, value)) * 52;
        path += (path ? " L " : "M ") + x.toFixed(1) + " " + y.toFixed(1);
      }
    }

    return path;
  }

  function historyHero(values, context) {
    var history = context && context.history;
    var pages = (history && history.pages) || [{ points: [] }];
    var selected = historySelected(values, pages);
    var points = (pages[selected] && pages[selected].points) || pages[selected] || [];
    var path = historyPath(points);
    var phase = history && typeof history.phase === "number" ? history.phase : 0;

    return '<svg class="mcard-graph" viewBox="0 0 300 64" preserveAspectRatio="xMidYMid meet">' +
      '<line x1="8" y1="60" x2="292" y2="60" stroke="#333" stroke-width="1" stroke-dasharray="2 4"/>' +
      (path ? '<path d="' + path + '" fill="none" stroke="var(--accent)" stroke-width="2" vector-effect="non-scaling-stroke"/>' : '') +
      '</svg>' +
      '<div class="hero-actions">' +
        '<button type="button" class="hero-action" data-action="history-prev">‹</button>' +
        '<button type="button" class="hero-action" data-action="history-next">›</button>' +
        '<button type="button" class="hero-action" data-action="history-add">+ PAGE</button>' +
        '<button type="button" class="hero-action" data-action="history-remove">- PAGE</button>' +
      '</div>' +
      '<div class="history-playhead" style="left:' + (phase * 100).toFixed(2) + '%"></div>' +
      '<div class="hero-label" style="left:2%;top:2%">HISTORY · PAGE ' +
        (selected + 1) + ' / ' + pages.length + '</div>';
  }

  /// The page editor: editLfo moves the points of one page; the page buttons
  /// move the morph and add or remove whole pages.
  function wireHistory(el, kind, values, state) {
    var history = state && state.history;

    if (kind !== "history" || !history || !history.pages || !history.pages.length) return;

    var pages = history.pages;
    var selected = historySelected(values, pages);
    var svg = el.querySelector("svg");

    if (!svg) return;

    var page = pages[selected];
    var holder = { points: page.points || page };
    var accent = (getComputedStyle(el).getPropertyValue("--accent") || "#4c8dff").trim();
    var paint = function (points) {
      if (page.points) page.points = points;
      else pages[selected] = points;
      IAmCard.sources.drawPreview(svg, "lfo", { points: points, color: accent, segments: 36 }, null);
    };

    var post = function () {
      if (state.post) state.post(JSON.stringify(pages));
    };

    // The action row can sit under the SVG inside a scaled foreignObject, so a
    // press that misses the button is matched to its rectangle here.
    svg.addEventListener("pointerdown", function (event) {
      var buttons = el.querySelectorAll(".hero-action");

      for (var index = 0; index < buttons.length; index++) {
        var rect = buttons[index].getBoundingClientRect();

        if (event.clientX >= rect.left && event.clientX <= rect.right &&
            event.clientY >= rect.top && event.clientY <= rect.bottom) {
          buttons[index].dispatchEvent(new PointerEvent("pointerdown", {
            bubbles: true,
            cancelable: true,
            clientX: event.clientX,
            clientY: event.clientY,
            pointerId: event.pointerId,
          }));
          event.preventDefault();
          event.stopPropagation();
          return;
        }
      }
    });

    IAmCard.sources.editLfo(svg, holder, {
      paint: paint,
      commit: post,
      dots: true,
      onStart: function () { el.dataset.busy = "1"; },
      onEnd: function () { delete el.dataset.busy; },
    });

    el.onpointerdown = function (event) {
      var button = event.target.closest && event.target.closest("[data-action]");

      if (!button) return;

      var action = button.getAttribute("data-action");
      var last = pages.length - 1;

      if (action === "history-prev") {
        state.set("morph", last > 0 ? Math.max(selected - 1, 0) / last : 0);
      } else if (action === "history-next") {
        state.set("morph", last > 0 ? Math.min(selected + 1, last) / last : 0);
      } else if (action === "history-add") {
        var copy = { points: (page.points || page).map(function (point) { return Object.assign({}, point); }) };
        pages.splice(selected + 1, 0, copy);
        post();
        state.set("morph", (selected + 1) / Math.max(pages.length - 1, 1));
      } else if (action === "history-remove" && pages.length > 1) {
        pages.splice(selected, 1);
        post();
        state.set("morph", Math.min(selected, pages.length - 1) / Math.max(pages.length - 1, 1));
      }

      event.preventDefault();
    };

    paint(holder.points);
  }

  /// The frame a Filter Table uses as its convolution kernel.
  function filterTablePage(values, pages) {
    return IAmPage.index(valueOf(values, "morph", 0), pages.length);
  }

  /// The continuous kernel value at `x` in [0, 1].
  function filterCurve(points, x) {
    var sorted = points.slice().sort(function (a, b) { return a.time - b.time; });

    if (sorted.length < 2) return 0.5;
    if (x <= sorted[0].time) return sorted[0].value;

    for (var i = 1; i < sorted.length; i++) {
      var left = sorted[i - 1];
      var right = sorted[i];

      if (x <= right.time) {
        var span = Math.max(right.time - left.time, 1e-6);
        var unit = Math.max(0, Math.min(1, (x - left.time) / span));
        return left.value + (right.value - left.value) * IAmCard.sources.bend(unit, left.bend);
      }
    }

    return sorted[sorted.length - 1].value;
  }

  /// The 0..1 response points of the current kernel, relative to cutoff.
  ///
  /// x = 0 is DC, x = 1 is Nyquist; cutoff sits at harmonic 24, so its marker is
  /// at 24 / 512 of the axis.
  function filterResponsePoints(points) {
    var size = 1024;
    var frame = [];
    var mean = 0;

    for (var index = 0; index < size; index++) {
      var value = filterCurve(points, index / (size - 1)) * 2 - 1;
      frame.push(value);
      mean += value;
    }

    mean /= size;
    for (var i = 0; i < size; i++) frame[i] -= mean;

    var re = frame.slice();
    var im = new Array(size).fill(0);
    fft(re, im, false);

    var magnitude = [];
    var peak = 1e-9;

    for (var bin = 0; bin <= size / 2; bin++) {
      var value = Math.sqrt(re[bin] * re[bin] + im[bin] * im[bin]);
      magnitude.push(value);
      peak = Math.max(peak, value);
    }

    var bins = 64;
    var response = [];

    for (var band = 0; band < bins; band++) {
      var at = band / (bins - 1) * (size / 2);
      var index = Math.floor(at);
      var fraction = at - index;
      var first = magnitude[index] || 0;
      var second = magnitude[Math.min(index + 1, magnitude.length - 1)] || 0;
      var level = first + (second - first) * fraction;
      var db = 20 * Math.log10(Math.max(level / peak, 1e-4));
      var value = Math.max(0, Math.min(1, (db + 48) / 48));

      response.push({ time: band / (bins - 1), value: value, bend: 0 });
    }

    return response;
  }

  /// An in-place radix-2 FFT, enough for the response editor.
  function fft(re, im, inverse) {
    var size = re.length;
    var j = 0;

    for (var i = 1; i < size; i++) {
      var bit = size >> 1;

      while (j & bit) {
        j ^= bit;
        bit >>= 1;
      }

      j ^= bit;

      if (i < j) {
        var swapRe = re[i]; re[i] = re[j]; re[j] = swapRe;
        var swapIm = im[i]; im[i] = im[j]; im[j] = swapIm;
      }
    }

    for (var length = 2; length <= size; length <<= 1) {
      var angle = 2 * Math.PI / length * (inverse ? 1 : -1);
      var stepRe = Math.cos(angle);
      var stepIm = Math.sin(angle);

      for (var offset = 0; offset < size; offset += length) {
        var wRe = 1;
        var wIm = 0;

        for (var k = 0; k < length / 2; k++) {
          var uRe = re[offset + k];
          var uIm = im[offset + k];
          var vRe = re[offset + k + length / 2] * wRe - im[offset + k + length / 2] * wIm;
          var vIm = re[offset + k + length / 2] * wIm + im[offset + k + length / 2] * wRe;

          re[offset + k] = uRe + vRe;
          im[offset + k] = uIm + vIm;
          re[offset + k + length / 2] = uRe - vRe;
          im[offset + k + length / 2] = uIm - vIm;

          var nextRe = wRe * stepRe - wIm * stepIm;
          wIm = wRe * stepIm + wIm * stepRe;
          wRe = nextRe;
        }
      }
    }

    if (inverse) {
      for (var index = 0; index < size; index++) {
        re[index] /= size;
        im[index] /= size;
      }
    }
  }

  /// Interpolate a response point list at x in [0, 1].
  function responseValue(response, x) {
    var sorted = response.slice().sort(function (a, b) { return a.time - b.time; });
    if (!sorted.length) return 0;
    if (x <= sorted[0].time) return sorted[0].value;

    for (var index = 1; index < sorted.length; index++) {
      var left = sorted[index - 1];
      var right = sorted[index];

      if (x <= right.time) {
        var span = Math.max(right.time - left.time, 1e-6);
        var unit = Math.max(0, Math.min(1, (x - left.time) / span));
        return left.value + (right.value - left.value) * unit;
      }
    }

    return sorted[sorted.length - 1].value;
  }

  /// Turn the relative magnitude curve into a zero-phase time kernel.
  function filterTimeFromResponse(response) {
    var size = 1024;
    var re = new Array(size).fill(0);
    var im = new Array(size).fill(0);

    for (var bin = 0; bin <= size / 2; bin++) {
      var x = bin / (size / 2);
      var db = responseValue(response, x) * 48 - 48;
      var magnitude = Math.pow(10, db / 20);
      re[bin] = magnitude;
      if (bin > 0 && bin < size / 2) re[size - bin] = magnitude;
    }

    fft(re, im, true);

    var shifted = new Array(size);
    for (var index = 0; index < size; index++) {
      shifted[(index + size / 2) % size] = re[index];
    }

    var peak = 1e-9;
    for (var index = 0; index < size; index++) peak = Math.max(peak, Math.abs(shifted[index]));

    var points = [];
    var count = 256;

    for (var index = 0; index < count; index++) {
      var value = shifted[Math.floor(index / count * size)] / peak;
      points.push({
        time: index / (count - 1),
        value: Math.max(0, Math.min(1, value * 0.5 + 0.5)),
        bend: 0,
      });
    }

    return points;
  }

  /// The mono samples of a window of an AudioBuffer.

  /// A sample as a time kernel, resampled to 256 page points.

  /// A sample as a relative frequency curve, then IFFT to a time kernel.

  /// Decode a file with Web Audio, so a sample can be imported as a kernel.

  /// A raw sample window as an SVG path on the hero canvas.

  /// The carrier wave the ring modulator multiplies the signal with.
  function ringCarrier(shape, t) {
    var phase = t - Math.floor(t);

    if (shape === 1) {
      return phase < 0.25 ? 4 * phase : phase < 0.75 ? 4 * (0.5 - phase) : 4 * (phase - 1);
    }
    if (shape === 2) return phase < 0.5 ? 2 * phase : 2 * phase - 2;
    if (shape === 3) return phase < 0.5 ? 1 : -1;
    if (shape === 4) {
      // A stable pseudo noise, so the picture does not flicker between paints.
      var hash = Math.sin(t * 12.9898 + 78.233) * 43758.5453;
      return (hash - Math.floor(hash)) * 2 - 1;
    }
    return Math.sin(phase * Math.PI * 2);
  }

  function ringWavePath(shape, offset, cycles) {
    var path = "";
    var samples = 128;

    for (var index = 0; index < samples; index++) {
      var unit = index / (samples - 1);
      var value = ringCarrier(shape, unit * cycles + offset);
      var x = 20 + unit * 480;
      var y = 80 - value * 60;
      path += (index ? "L" : "M") + x.toFixed(2) + " " + y.toFixed(2);
    }

    return path;
  }

  /// The ring modulator: the carrier it multiplies with, or, in the sidechain
  /// mode, what that carrier does to the signal.
  function ringHero(values, context) {
    var shape = Math.round(valueOf(values, "shape", 0));
    var mode = Math.round(valueOf(values, "mode", 0));
    var stereo = valueOf(values, "stereo", 0);
    var shapes = ["SINE", "TRIANGLE", "SAW", "SQUARE", "NOISE"];
    var modes = ["RING", "AM", "SIDECHAIN"];

    if (mode === 2) {
      var scope = context && context.scope;
      var spectra = context && context.spectra;
      var rate = (context && context.rate) || 48000;
      var inputPath = "";
      var outputPath = "";

      if (scope && scope.input && scope.output && spectra) {
        var inputBands = bandsOf(scope.input, rate);
        var outputBands = bandsOf(scope.output, rate);

        if (inputBands && outputBands) {
          spectra.input = smoothBands(spectra.input, inputBands);
          spectra.output = smoothBands(spectra.output, outputBands);
          inputPath = bandsPath(spectra.input);
          outputPath = bandsPath(spectra.output);
        }
      }

      return '<svg class="mcard-graph" viewBox="0 0 520 160" preserveAspectRatio="xMidYMid meet">' +
        (inputPath ? '<path d="' + inputPath + '" fill="none" stroke="#666" stroke-width="1.5" vector-effect="non-scaling-stroke"/>' : '') +
        (outputPath ? '<path d="' + outputPath + '" fill="none" stroke="var(--accent)" stroke-width="2" vector-effect="non-scaling-stroke"/>' : '') +
        '</svg>' +
        (outputPath ? '' : '<div class="hero-empty">Waiting for the sidechain...</div>') +
        '<div class="hero-label" style="top:2%">SIDECHAIN \u00b7 BEFORE / AFTER</div>';
    }

    var offset = stereo / 360;

    return '<svg class="mcard-graph" viewBox="0 0 520 160" preserveAspectRatio="xMidYMid meet">' +
      '<line x1="20" y1="80" x2="500" y2="80" stroke="#333" stroke-width="1" stroke-dasharray="2 4"/>' +
      '<path d="' + ringWavePath(shape, offset, 2) + '" fill="none" stroke="var(--accent)" stroke-width="2" opacity="0.4" vector-effect="non-scaling-stroke"/>' +
      '<path d="' + ringWavePath(shape, 0, 2) + '" fill="none" stroke="var(--accent)" stroke-width="2" vector-effect="non-scaling-stroke"/>' +
      '</svg>' +
      '<div class="hero-label" style="top:2%">' + modes[mode] + ' \u00b7 ' + shapes[shape] + ' \u00b7 R+' + Math.round(stereo) + '\u00b0</div>';
  }

  function filterWaveformPath(samples) {
    if (!samples || !samples.length) return "";

    var path = "";

    for (var index = 0; index < samples.length; index++) {
      var x = 8 + (index / (samples.length - 1)) * 284;
      var y = 32 - Math.max(-1, Math.min(1, samples[index] * 4)) * 22;
      path += (index === 0 ? "M" : "L") + x.toFixed(2) + " " + y.toFixed(2);
    }

    return path;
  }

  function filterTableHero(values, context) {
    var table = context && context.filterTable;
    var sidechain = context && context.sidechain;

    // The sidechain mode has no page to edit: the hero is the input it follows.
    if (valueOf(values, "mode", 0) >= 0.5) {
      if (sidechain) sidechain.wanted = true;

      var samples = sidechain && sidechain.input;
      var path = filterWaveformPath(samples);

      return '<svg class="mcard-graph" viewBox="0 0 300 64" preserveAspectRatio="xMidYMid meet">' +
        '<line x1="8" y1="32" x2="292" y2="32" stroke="#333" stroke-width="1" stroke-dasharray="2 4"/>' +
        (path ? '<path d="' + path + '" fill="none" stroke="var(--accent)" stroke-width="2" vector-effect="non-scaling-stroke"/>' : '') +
        '</svg>' +
        (samples ? '' : '<div class="hero-empty">Waiting for the sidechain...</div>') +
        '<div class="hero-label" style="left:2%;top:2%">SIDECHAIN \u00b7 ' +
        (Math.round(valueOf(values, "mode", 0) || 0) === 2 ? "KERNEL" : "RESPONSE") +
        '</div>';
    }

    if (sidechain) sidechain.wanted = false;

    var pages = (table && table.pages) || [[]];
    var selected = filterTablePage(values, pages);
    var points = pages[selected] || [];
    var view = table && table.view === "time" ? "time" : "freq";
    var path = view === "freq"
      ? historyPath(filterResponsePoints(points))
      : historyPath(points);

    return '<svg class="mcard-graph" viewBox="0 0 300 64" preserveAspectRatio="xMidYMid meet">' +
      '<line x1="8" y1="60" x2="292" y2="60" stroke="#333" stroke-width="1" stroke-dasharray="2 4"/>' +
      (view === "freq" ? '<line x1="21.3" y1="0" x2="21.3" y2="64" stroke="#666" stroke-dasharray="2 4" stroke-width="1"/>' : '') +
      (path ? '<path d="' + path + '" fill="none" stroke="var(--accent)" stroke-width="2" vector-effect="non-scaling-stroke"/>' : '') +
      '</svg>' +
      '<div class="hero-actions">' +
        '<button type="button" class="hero-action" data-action="filter-prev">\u2039</button>' +
        '<button type="button" class="hero-action" data-action="filter-next">\u203a</button>' +
        '<button type="button" class="hero-action" data-action="filter-add">+ PAGE</button>' +
        '<button type="button" class="hero-action" data-action="filter-remove">- PAGE</button>' +
        '<button type="button" class="hero-action" data-action="filter-view">' +
          (view === "freq" ? "TIME" : "FREQ") + '</button>' +
        '<button type="button" class="hero-action" data-action="filter-import">IMPORT</button>' +
      '</div>' +
      '<div class="hero-actions" style="bottom:20px">' +
        '<button type="button" class="hero-action" data-action="filter-mode">' +
          (table && table.importMode === "freq" ? "IMP FREQ" : "IMP TIME") + '</button>' +
        '<button type="button" class="hero-action" data-action="filter-slice">SLICE ' +
          String(table && table.sliceMode || "auto").toUpperCase() + '</button>' +
        '<button type="button" class="hero-action" data-action="filter-rev">REVERSE</button>' +
        '<button type="button" class="hero-action" data-action="filter-inv">INVERT</button>' +
        '<button type="button" class="hero-action" data-action="filter-dc">REMOVE DC</button>' +
        '<button type="button" class="hero-action" data-action="filter-freeze">' +
          (table && table.frozen ? "FROZEN" : "FREEZE") + '</button>' +
      '</div>' +
      '<div data-role="filter-progress" style="position:absolute;left:2%;bottom:36px;font:9px/1.2 var(--mono);color:var(--text2);display:none"></div>' +
      '<div class="hero-label" style="left:2%;top:2%">FILTER TABLE \u00b7 PAGE ' +
        (selected + 1) + ' / ' + pages.length +
        (view === "freq" ? ' \u00b7 FREQ' : ' \u00b7 TIME') + '</div>';
  }

  /// The page editor, on the same 300x64 LFO canvas.
  function wireFilterTable(el, kind, values, state) {
    var table = state && state.filterTable;

    if (kind !== "filtertable" || !table || !table.pages) return;

    // Sidechain mode shows the input, not the frame: no editor, no page buttons.
    if (valueOf(values, "mode", 0) >= 0.5) return;

    var svg = el.querySelector("svg");
    if (!svg) return;

    var pages = table.pages;
    var selected = filterTablePage(values, pages);
    var view = table.view === "time" ? "time" : "freq";
    var accent = (getComputedStyle(el).getPropertyValue("--accent") || "#4c8dff").trim();
    var attached = false;
    var frozen = !!table.frozen;
    var response = view === "freq" ? filterResponsePoints(pages[selected]) : null;

    var post = function () {
      if (state.post) state.post(JSON.stringify(pages));
    };

    var drawFrequency = function () {
      IAmCard.sources.drawPreview(svg, "lfo", {
        points: response,
        color: accent,
        segments: 36,
        handles: !frozen,
      }, null);

      // Harmonic 24 is Cutoff; on the Nyquist-relative axis it sits at 24/512.
      var marker = document.createElementNS("http://www.w3.org/2000/svg", "line");
      marker.setAttribute("x1", "21.3"); marker.setAttribute("y1", "0");
      marker.setAttribute("x2", "21.3"); marker.setAttribute("y2", "64");
      marker.setAttribute("stroke", "#666");
      marker.setAttribute("stroke-dasharray", "2 4");
      marker.setAttribute("stroke-width", "1");
      marker.setAttribute("pointer-events", "none");
      svg.appendChild(marker);
    };

    var drawTime = function () {
      IAmCard.sources.drawPreview(svg, "lfo", {
        points: pages[selected],
        color: accent,
        segments: 36,
        handles: !frozen,
      }, null);
    };

    var redraw = function () {
      if (view === "freq") drawFrequency();
      else drawTime();
    };

    var setLabel = function () {
      var label = el.querySelector(".hero-label");
      if (!label) return;

      label.innerHTML = "FILTER TABLE \u00b7 PAGE " + (selected + 1) + " / " +
        pages.length + (view === "freq" ? " \u00b7 FREQ" : " \u00b7 TIME") +
        (frozen ? " \u00b7 FROZEN" : "") +
        ' \u00b7 <span data-action="filter-window" style="pointer-events:auto;cursor:pointer">WIN ' +
        (table.windowSamples || 2048) + '</span>';
    };

    var editWindow = function (anchor) {
      IAmCard.openKnobEditor(
        { label: "Window", unit: "samples" },
        table.windowSamples || 2048,
        function (value) {
          table.windowSamples = Math.max(1, Math.round(value));
          setLabel();
        },
        anchor
      );
    };

    var attachEditor = function () {
      if (attached || frozen) return;

      var holder = view === "freq" ? { points: response } : { points: pages[selected] };

      attached = true;
      IAmCard.sources.editLfo(svg, holder, {
        paint: function (points) {
          if (view === "freq") {
            response = points;
            drawFrequency();
          } else {
            pages[selected] = points;
            drawTime();
          }
        },
        commit: function (points) {
          if (view === "freq") {
            response = points;
            pages[selected] = filterTimeFromResponse(points);
          }

          post();
        },
        onStart: function () { el.dataset.busy = "1"; },
        onEnd: function () { delete el.dataset.busy; },
      });
    };

    var detachEditor = function () {
      if (!attached) return;
      attached = false;
      svg.onpointerdown = null;
      svg.ondblclick = null;
      svg.oncontextmenu = null;
    };

    var setView = function (next) {
      detachEditor();
      view = next === "time" ? "time" : "freq";
      table.view = view;

      if (view === "freq") response = filterResponsePoints(pages[selected]);

      redraw();
      attachEditor();

      setLabel();

      var button = el.querySelector('[data-action="filter-view"]');
      if (button) button.textContent = view === "freq" ? "TIME" : "FREQ";
    };

    var setFrozen = function (next) {
      frozen = next;
      table.frozen = frozen;
      el.classList.toggle("frozen", frozen);

      if (frozen) detachEditor();
      else attachEditor();

      redraw();
      setLabel();

      var button = el.querySelector('[data-action="filter-freeze"]');
      if (button) button.textContent = frozen ? "FROZEN" : "FREEZE";
    };

    var replacePages = function (result) {
      if (!result || !result.pages) return;

      table.pages = result.pages;
      pages = result.pages;
      selected = Math.min(selected, pages.length - 1);
      response = view === "freq" ? filterResponsePoints(pages[selected]) : null;
      detachEditor();
      redraw();
      attachEditor();
    };

    var progressEl = el.querySelector('[data-role="filter-progress"]');
    var importing = false;

    var showProgress = function (text) {
      if (!progressEl) return;
      progressEl.textContent = text;
      progressEl.style.display = "block";
    };

    var hideProgress = function () {
      if (progressEl) progressEl.style.display = "none";
    };

    var importBuffer = function (file) {
      if (!file || frozen || importing || table.importing) return;

      importing = true;
      table.importing = true;
      el.dataset.busy = "1";

      var windowSamples = Math.max(1, parseInt(table.windowSamples, 10) || 2048);

      file.arrayBuffer().then(function (body) {
        if (!state.import) throw new Error("no import route");
        return state.import(body, windowSamples, selected, table.importMode || "time", table.sliceMode || "auto");
      }).then(function (progress) {
        var watch = function (value) {
          if (!value) throw new Error("no progress");
          if (value.error) throw new Error(value.error);

          if (value.active) {
            var percent = value.total ? Math.round(value.done / value.total * 100) : 0;
            showProgress(value.stage === "decode" ? "IMPORT \u00b7 DECODING" :
              "IMPORT \u00b7 " + percent + "%");
            setTimeout(function () {
              state.importProgress().then(watch).catch(function (error) { fail(error); });
            }, 140);
            return;
          }

          importing = false;
          table.importing = false;
          delete el.dataset.busy;
          showProgress("IMPORT \u00b7 DONE");
          state.reload().then(function (result) {
            replacePages(result);
            setTimeout(hideProgress, 450);
          }).catch(function (error) { fail(error); });
        };

        var fail = function (error) {
          importing = false;
          table.importing = false;
          delete el.dataset.busy;
          showProgress("IMPORT \u00b7 FAILED");
          setTimeout(hideProgress, 900);
          console.warn(error);
        };

        watch(progress);
      }).catch(function (error) {
        importing = false;
        table.importing = false;
        delete el.dataset.busy;
        showProgress("IMPORT \u00b7 FAILED");
        setTimeout(hideProgress, 900);
        console.warn(error);
      });
    };

    var importFile = function () {
      var input = document.createElement("input");
      input.type = "file";
      input.accept = ".wav,audio/wav";

      input.onchange = function () {
        var file = input.files && input.files[0];

        if (file) importBuffer(file);
      };

      input.click();
    };

    // These are properties, not listeners: the product page rewires the hero
    // on every poll, and addEventListener would stack a new drop handler each
    // time. A drop then ran hundreds of handlers and took the host with it.
    el.ondragover = function (event) {
      event.preventDefault();
      if (event.dataTransfer) event.dataTransfer.dropEffect = "copy";
      el.classList.add("drop");
    };

    el.ondragleave = function () {
      el.classList.remove("drop");
    };

    el.ondrop = function (event) {
      event.preventDefault();
      el.classList.remove("drop");

      var file = event.dataTransfer && event.dataTransfer.files && event.dataTransfer.files[0];
      if (file) importBuffer(file);
    };

    svg.addEventListener("pointerdown", function (event) {
      var buttons = el.querySelectorAll(".hero-action");

      for (var index = 0; index < buttons.length; index++) {
        var rect = buttons[index].getBoundingClientRect();

        if (event.clientX >= rect.left && event.clientX <= rect.right &&
            event.clientY >= rect.top && event.clientY <= rect.bottom) {
          buttons[index].dispatchEvent(new PointerEvent("pointerdown", {
            bubbles: true,
            cancelable: true,
            clientX: event.clientX,
            clientY: event.clientY,
            pointerId: event.pointerId,
          }));
          event.preventDefault();
          return;
        }
      }
    });

    el.onpointerdown = function (event) {
      var button = event.target.closest && event.target.closest("[data-action]");

      if (!button) return;

      var action = button.getAttribute("data-action");
      var last = pages.length - 1;

      if (action === "filter-freeze") {
        setFrozen(!frozen);
      } else if (action === "filter-view") {
        setView(view === "freq" ? "time" : "freq");
      } else if (action === "filter-window") {
        if (!frozen) editWindow(button);
      } else if (action === "filter-mode") {
        table.importMode = table.importMode === "freq" ? "time" : "freq";
        button.textContent = table.importMode === "freq" ? "IMP FREQ" : "IMP TIME";
      } else if (action === "filter-slice") {
        var modes = ["auto", "fixed", "pitch", "single"];
        var next = (modes.indexOf(table.sliceMode) + 1) % modes.length;
        table.sliceMode = modes[next];
        button.textContent = "SLICE " + table.sliceMode.toUpperCase();
      } else if (action === "filter-rev" || action === "filter-inv" || action === "filter-dc") {
        var edit = action.replace("filter-", "");
        if (!frozen && state.edit) state.edit(edit, selected).then(replacePages);
      } else if (action === "filter-import") {
        if (!frozen) importFile();
      } else if (action === "filter-prev") {
        state.set("morph", last > 0 ? Math.max(selected - 1, 0) / last : 0);
      } else if (action === "filter-next") {
        state.set("morph", last > 0 ? Math.min(selected + 1, last) / last : 0);
      } else if (action === "filter-add" && !frozen) {
        pages.splice(selected + 1, 0, pages[selected].map(function (point) { return Object.assign({}, point); }));
        post();
        state.set("morph", (selected + 1) / Math.max(pages.length - 1, 1));
      } else if (action === "filter-remove" && !frozen && pages.length > 1) {
        pages.splice(selected, 1);
        post();
        state.set("morph", Math.min(selected, pages.length - 1) / Math.max(pages.length - 1, 1));
      }

      event.preventDefault();
    };

    el.classList.toggle("frozen", frozen);
    setLabel();
    redraw();
    attachEditor();
  }


