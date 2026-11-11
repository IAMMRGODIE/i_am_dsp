// Part four: the pictures that draw the scope, the transfer curve and the note.
  function trace(samples, middle, amplitude, stride) {
    stride = stride || 1;

    var last = samples.length - 1;
    var path = "";
    var point = function (index) {
      var x = 20 + index / last * 480;
      var y = middle - clamp(samples[index], -1.2, 1.2) / 1.2 * amplitude;
      return (path ? " L " : "M ") + x.toFixed(1) + " " + y.toFixed(1);
    };

    for (var i = 0; i <= last; i += stride) {
      path += point(i);
    }

    if (last % stride !== 0) path += point(last);

    return path;
  }

  function rmsOf(samples) {
    if (!samples || samples.length === 0) return 0;

    var sum = 0;
    for (var i = 0; i < samples.length; i++) sum += samples[i] * samples[i];

    return Math.sqrt(sum / samples.length);
  }

  function dbText(rms) {
    if (!(rms > 1e-6)) return "-inf";
    return (20 * Math.log10(rms)).toFixed(1);
  }

  function rmsBadge(scope) {
    if (!scope || !scope.input || !scope.output) {
      return '<div class="hero-badge">RMS --</div>';
    }

    return '<div class="hero-badge">RMS ' + dbText(rmsOf(scope.input)) +
      ' \u2192 ' + dbText(rmsOf(scope.output)) + ' dB</div>';
  }

  function transferHero(values, context) {
    var mode = Math.round(valueOf(values, "mode", 0));
    var threshold = clamp(valueOf(values, "threshold", 0.8), 0.01, 4);
    var gain = valueOf(values, "gain", 1);
    var path = "";
    var bands = "";
    var scope = context && context.scope;

    if (scope && scope.input && scope.output) {
      var inputRms = Math.min(rmsOf(scope.input), 2);
      var outputRms = Math.min(rmsOf(scope.output), 2);

      if (inputRms > 0.0005) {
        var left = 20 + (-inputRms + 2) / 4 * 480;
        var right = 20 + (inputRms + 2) / 4 * 480;
        bands += '<rect x="' + left.toFixed(1) + '" y="0" width="' +
          (right - left).toFixed(1) + '" height="160" fill="#9a9a9a" fill-opacity=".10"/>';
      }

      if (outputRms > 0.0005) {
        var top = 80 - outputRms / 2 * 66;
        var bottom = 80 + outputRms / 2 * 66;
        bands += '<rect x="0" y="' + top.toFixed(1) + '" width="520" height="' +
          (bottom - top).toFixed(1) + '" fill="var(--accent)" fill-opacity=".08"/>';
      }
    }

    var alpha = Math.max(valueOf(values, "alpha", 0.5), 0.01);
    var k = valueOf(values, "k", 2);
    var a = Math.max(valueOf(values, "a", 1), 0.01);
    var p = valueOf(values, "p", 2);
    var bits = Math.max(1, valueOf(values, "bits", 8));

    for (var i = 0; i <= 140; i++) {
      var x = -2 + i / 140 * 4;
      var shaped;

      if (mode === 1) shaped = Math.tanh(x / alpha);
      else if (mode === 2) shaped = 2 / (1 + Math.exp(-x * k)) - 1;
      else if (mode === 3) {
        var factor = 1 + 1 / Math.pow(a, p);
        var norm = Math.pow(Math.abs(x) / a, p);
        shaped = (x / (1 + norm)) * factor;
      } else if (mode === 4) shaped = Math.round(bits * x) / bits;
      else if (mode === 5) shaped = x;
      else shaped = clamp(x, -threshold, threshold);

      var y = clamp(shaped * gain, -2, 2);
      path += (i ? " L " : "M ") + (20 + (x + 2) / 4 * 480).toFixed(1) +
        " " + (80 - y / 2 * 66).toFixed(1);
    }

    return heroSvg(
      bands +
      '<path d="M 20 146 L 500 14" stroke="#333" stroke-width="1" stroke-dasharray="3 5" ' +
        'vector-effect="non-scaling-stroke"/>' +
      '<line x1="260" y1="12" x2="260" y2="148" stroke="#2b2b2b"/>' +
      '<line x1="20" y1="80" x2="500" y2="80" stroke="#2b2b2b"/>' +
      '<path d="' + path + '" fill="none" stroke="var(--accent)" stroke-width="2.4" ' +
        'vector-effect="non-scaling-stroke"/>'
    ) + (context && context.scope ? rmsBadge(context.scope) : "");
  }

  function compressorHero(values, context) {
    var meterState = context && context.meter;
    var env = context && context.envHistory;

    if (!meterState || !env) return compressorSimpleHero(values);

    var threshold = clamp(valueOf(values, "threshold", -24), -60, 0);
    var ratio = Math.max(valueOf(values, "ratio", 4), 1);
    var makeup = valueOf(values, "makeup_db", 0);
    var sidechainOn = !!valueOf(values, "sidechain", 0);
    var level = sidechainOn ? meterState.sidechain : meterState.input;
    var window = (context && context.envWindow) || 128;
    var path = "";

    for (var i = 0; i <= 120; i++) {
      var input = -60 + i / 120 * 60;
      var output = input <= threshold ? input : threshold + (input - threshold) / ratio;
      output = clamp(output + makeup, -60, 0);
      var x = 8 + i / 120 * 124;
      var y = 132 - (output + 60) / 60 * 124;
      path += (i ? " L " : "M ") + x.toFixed(1) + " " + y.toFixed(1);
    }

    var knee = 8 + (threshold + 60) / 60 * 124;
    var marker = "";

    if (level !== null && level !== undefined && isFinite(level)) {
      var db = clamp(level, -60, 0);
      var mx = 8 + (db + 60) / 60 * 124;
      var out = db <= threshold ? db : threshold + (db - threshold) / ratio;
      out = clamp(out + makeup, -60, 0);
      var my = 132 - (out + 60) / 60 * 124;
      var tag = sidechainOn ? "SC " : "IN ";
      var textX = mx > 92 ? mx - 4 : mx + 4;
      var anchor = mx > 92 ? "end" : "start";

      marker =
        '<line x1="' + mx.toFixed(1) + '" y1="8" x2="' + mx.toFixed(1) + '" y2="132" stroke="var(--accent)" stroke-opacity=".22" stroke-dasharray="2 4"/>' +
        '<line x1="8" y1="' + my.toFixed(1) + '" x2="132" y2="' + my.toFixed(1) + '" stroke="var(--accent)" stroke-opacity=".12" stroke-dasharray="2 4"/>' +
        '<circle cx="' + mx.toFixed(1) + '" cy="' + my.toFixed(1) + '" r="2.4" fill="var(--accent)"/>' +
        '<text x="' + textX.toFixed(1) + '" y="17" fill="var(--accent)" font-size="7" text-anchor="' + anchor + '">' + tag + db.toFixed(1) + ' dB</text>';
    }

    var left =
      '<svg viewBox="0 0 140 140" preserveAspectRatio="xMidYMid meet">' +
        '<rect x="8" y="8" width="124" height="124" fill="none" stroke="#2a2a2a"/>' +
        '<line x1="8" y1="132" x2="132" y2="8" stroke="#2f2f2f" stroke-dasharray="3 5"/>' +
        '<line x1="' + knee.toFixed(1) + '" y1="8" x2="' + knee.toFixed(1) + '" y2="132" stroke="#343434" stroke-dasharray="3 5"/>' +
        '<line x1="8" y1="132" x2="132" y2="132" stroke="#3a3a3a"/>' +
        '<path d="' + path + '" fill="none" stroke="var(--accent)" stroke-width="2" vector-effect="non-scaling-stroke"/>' +
        marker +
      '</svg>';

    var right = "";
    var count = Math.min(window, env.input.length);

    if (count > 1) {
      var loudest = 1e-6;

      for (var k = 0; k < count; k++) {
        var index = env.input.length - count + k;
        loudest = Math.max(loudest, env.input[index], env.output[index]);
      }

      var inPath = "";
      var outPath = "";

      for (var s = 0; s < count; s++) {
        var idx = env.input.length - count + s;
        var px = 8 + s / (count - 1) * 384;
        var iy = 132 - env.input[idx] / loudest * 124;
        var oy = 132 - env.output[idx] / loudest * 124;
        inPath += (s ? " L " : "M ") + px.toFixed(1) + " " + iy.toFixed(1);
        outPath += (s ? " L " : "M ") + px.toFixed(1) + " " + oy.toFixed(1);
      }

      right =
        '<path d="' + outPath + ' L 392 132 L 8 132 Z" fill="var(--accent)" fill-opacity=".06"/>' +
        '<path d="' + inPath + '" fill="none" stroke="#8a8a8a" stroke-width="1.1" opacity=".65" vector-effect="non-scaling-stroke"/>' +
        '<path d="' + outPath + '" fill="none" stroke="var(--accent)" stroke-width="1.6" vector-effect="non-scaling-stroke"/>';
    } else {
      right = '<line x1="8" y1="132" x2="392" y2="132" stroke="#3a3a3a"/>';
    }

    var seconds = (window * 0.05).toFixed(1);

    return '<div class="compressor-hero">' +
      '<div class="transfer-box">' + left + '</div>' +
      '<div class="env-box" title="Scroll to change the envelope time window">' +
        '<svg viewBox="0 0 400 140" preserveAspectRatio="none">' + right + '</svg>' +
        (count > 1
          ? '<div class="hero-label" style="left:4%;top:2%">IN</div>' +
            '<div class="hero-label" style="left:13%;top:2%;color:var(--accent)">OUT</div>'
          : "") +
        '<div class="hero-badge">' + seconds + ' s</div></div>' +
      '</div>';
  }

  function compressorSimpleHero(values) {
    var threshold = clamp(valueOf(values, "threshold", -24), -60, 0);
    var ratio = clamp(valueOf(values, "ratio", 4), 1, 20);
    var makeup = clamp(valueOf(values, "gain_linear", 1), 0.01, 4);
    var makeupDb = 20 * Math.log10(makeup);
    var points = [];

    function xFor(db) { return 20 + (db + 60) / 60 * 480; }
    function yFor(db) { return 148 - (db + 60) / 60 * 136; }

    for (var i = 0; i <= 120; i++) {
      var inputDb = -60 + i / 120 * 60;
      var outputDb = inputDb < threshold
        ? inputDb
        : threshold + (inputDb - threshold) / ratio;
      outputDb = clamp(outputDb + makeupDb, -60, 0);
      points.push((i ? "L " : "M ") + xFor(inputDb).toFixed(1) +
        " " + yFor(outputDb).toFixed(1));
    }

    var thresholdX = xFor(threshold);

    return heroSvg(
      '<line x1="20" y1="12" x2="20" y2="148" stroke="#2b2b2b"/>' +
      '<line x1="20" y1="148" x2="500" y2="148" stroke="#2b2b2b"/>' +
      '<line x1="20" y1="148" x2="500" y2="12" stroke="#333" stroke-width="1" ' +
      'stroke-dasharray="3 5" vector-effect="non-scaling-stroke"/>' +
      '<line x1="' + thresholdX.toFixed(1) + '" y1="12" x2="' + thresholdX.toFixed(1) +
      '" y2="148" stroke="#3a2a2a"/>' +
      '<path d="' + points.join(" ") + '" fill="none" stroke="var(--accent)" ' +
      'stroke-width="2.2" vector-effect="non-scaling-stroke"/>'
    ) +
      heroLabel(thresholdX + 4, 22, threshold.toFixed(0) + " dB", "color:#777;font-size:9px") +
      heroLabel(462, 140, ratio.toFixed(1) + ":1", "color:#777;font-size:9px");
  }

  /// The FDN tail: a pre-delay gap, a few early reflections, then a T60 decay
  /// with a lighter high-frequency trace.
  function escapeXml(text) {
    return String(text)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&apos;");
  }

  function impulseHero(values, context) {
    var ir = context && context.ir;

    if (!ir || !ir.peaks || !ir.peaks.length) {
      var failed = ir && ir.error;
      var pending = ir && ir.pending;
      var message = failed
        ? ir.error
        : (pending ? "Choosing an impulse response..." : "Click to load an impulse response");
      var color = failed ? "#ff7a6b" : (pending ? "var(--accent)" : "#8a8a8a");

      return heroSvg(
        '<line x1="20" y1="80" x2="500" y2="80" stroke="#2c2c2c"/>'
      ) + '<div class="hero-empty" style="color:' + color + '">' +
        escapeXml(message) + '</div>';
    }

    var peaks = ir.peaks;
    var last = Math.max(peaks.length - 1, 1);
    var top = "M 20 80";
    var bottom = "";

    for (var i = 0; i < peaks.length; i++) {
      var x = 20 + i / last * 480;
      var low = clamp(peaks[i][0], -1, 1);
      var high = clamp(peaks[i][1], -1, 1);
      top += " L " + x.toFixed(1) + " " + (80 - high * 62).toFixed(1);
      bottom = " L " + x.toFixed(1) + " " + (80 - low * 62).toFixed(1) + bottom;
    }

    var seconds = ir.rate ? ir.samples / ir.rate : 0;
    var label = (ir.name || "impulse") + "  -  " + seconds.toFixed(2) + " s";
    var fadeIn = clamp(ir.fade_in || 0, 0, 1);
    var fadeOut = clamp(ir.fade_out || 0, 0, 1);
    var ceiling = 22;
    var floor = 146;

    // The gain the two fades leave at a position: they multiply, so a fade that
    // runs into the other one lifts the curve instead of crossing it.
    var envelope = function (position) {
      var gain = 1.0;

      if (fadeIn > 0.0001 && position < fadeIn) {
        gain *= 0.5 - 0.5 * Math.cos(Math.PI * position / fadeIn);
      }

      if (fadeOut > 0.0001) {
        var fromEnd = 1 - position;

        if (fromEnd < fadeOut) {
          gain *= 0.5 - 0.5 * Math.cos(Math.PI * fromEnd / fadeOut);
        }
      }

      return gain;
    };

    var curve = "";
    var steps = 96;

    for (var step = 0; step <= steps; step++) {
      var position = step / steps;
      var x = 20 + position * 480;
      var y = floor - (floor - ceiling) * envelope(position);

      curve += (step ? " L " : "M ") + x.toFixed(1) + " " + y.toFixed(1);
    }

    var inX = 20 + fadeIn * 480;
    var outX = 20 + (1 - fadeOut) * 480;
    var inY = floor - (floor - ceiling) * envelope(fadeIn);
    var outY = floor - (floor - ceiling) * envelope(1 - fadeOut);

    // The fill closes along the floor; the outline stops at the two ends, so the
    // bottom edge has no stroke of its own.
    var fades =
      '<path d="' + curve + ' L 500 ' + floor + ' L 20 ' + floor + ' Z" fill="var(--accent)" fill-opacity="0.10"/>' +
      '<path d="' + curve + '" fill="none" stroke="var(--accent)" stroke-width="1.25" ' +
        'vector-effect="non-scaling-stroke"/>';

    var handles =
      '<div class="hero-pip" style="left:' + (inX / 5.2).toFixed(2) + '%;top:' +
        (inY / 1.6).toFixed(2) + '%;width:8px;height:8px;background:var(--accent);' +
        'border:1px solid #0e0e0e"></div>' +
      '<div class="hero-pip" style="left:' + (outX / 5.2).toFixed(2) + '%;top:' +
        (outY / 1.6).toFixed(2) + '%;width:8px;height:8px;background:var(--accent);' +
        'border:1px solid #0e0e0e"></div>';

    if (fadeIn > 0.001 || fadeOut > 0.001) {
      label += "  -  fade " + Math.round(fadeIn * 100) + "% / " + Math.round(fadeOut * 100) + "%";
    }

    return heroSvg(
      '<line x1="20" y1="80" x2="500" y2="80" stroke="#2f2f2f"/>' +
      '<path d="' + top + bottom + ' Z" fill="var(--accent)" fill-opacity=".14" stroke="' +
        'var(--accent)" stroke-width="1" vector-effect="non-scaling-stroke"/>' +
      fades
    ) + handles + '<div class="hero-label" style="top:2%">' + escapeXml(label) + '</div>';
  }

  function pitchHero(values) {
    var factor = clamp(valueOf(values, "pitch_shift_factor", 1), 0.25, 4);
    var semitones = 12 * Math.log2(factor);
    var marker = 40 + (semitones + 24) / 48 * 440;
    var ticks = "";

    for (var step = -24; step <= 24; step += 3) {
      var x = 40 + (step + 24) / 48 * 440;
      var major = step % 12 === 0;
      ticks += '<line x1="' + x.toFixed(1) + '" y1="' + (major ? 58 : 66) + '" x2="' +
        x.toFixed(1) + '" y2="' + (major ? 102 : 94) + '" stroke="' +
        (major ? "#4a4a4a" : "#333") + '" stroke-width="1"/>';
    }

    var sign = semitones > 0.05 ? "+" : "";

    return heroSvg(
      '<line x1="40" y1="80" x2="480" y2="80" stroke="#3a3a3a" stroke-width="1"/>' +
      ticks +
      '<line x1="' + marker.toFixed(1) + '" y1="30" x2="' + marker.toFixed(1) +
        '" y2="130" stroke="var(--accent)" stroke-width="2"/>'
    ) +
      '<div class="hero-pip" style="left:' + (marker / 5.2).toFixed(2) +
        '%;top:50%;width:8px;height:8px;background:var(--accent)"></div>' +
      '<div class="hero-badge">' + sign + semitones.toFixed(1) + ' st</div>';
  }

  function particlesHero(values) {
    var depth = clamp(valueOf(values, "depth", 0.5), 0, 1);
    var low = clamp(valueOf(values, "low_frequency", 200), 20, 20000);
    var high = clamp(valueOf(values, "high_frequency", 8000), 20, 20000);
    var side = clamp(valueOf(values, "side_gain", 1), 0, 2);
    var allowNegative = !!valueOf(values, "allow_negative", false);
    var fftIndex = clamp(Math.round(valueOf(values, "fft_size", 2)), 0, 3);
    var windowSize = [512, 1024, 2048, 4096][fftIndex];
    var count = Math.max(16, Math.min(128, windowSize / 32));
    var left = "";
    var right = "";
    var lower = Math.min(low, high);
    var upper = Math.max(low, high);
    var frequencyToX = function (frequency) {
      return 30 + Math.log(frequency / 20) / Math.log(1000) * 460;
    };
    var bandLeft = frequencyToX(lower);
    var bandRight = frequencyToX(upper);
    var grid =
      '<line x1="18" y1="80" x2="502" y2="80" stroke="#343434"/>' +
      '<line x1="260" y1="14" x2="260" y2="146" stroke="#2a2a2a"/>' +
      '<rect x="' + bandLeft.toFixed(1) + '" y="22" width="' +
        (bandRight - bandLeft).toFixed(1) + '" height="116" fill="var(--accent)" fill-opacity=".05"/>' +
      '<line x1="' + bandLeft.toFixed(1) + '" y1="22" x2="' + bandLeft.toFixed(1) +
        '" y2="138" stroke="var(--accent)" stroke-opacity=".25"/>' +
      '<line x1="' + bandRight.toFixed(1) + '" y1="22" x2="' + bandRight.toFixed(1) +
        '" y2="138" stroke="var(--accent)" stroke-opacity=".25"/>';

    for (var index = 0; index < count; index++) {
      var x = 30 + index / (count - 1) * 460;
      var frequency = 20 * Math.pow(1000, index / (count - 1));
      var active = frequency >= lower && frequency <= upper;
      var range = (0.12 + 0.88 * depth) * (0.25 + 0.75 * Math.min(side, 2) / 2);
      var signLeft = allowNegative ? Math.random() * 2 - 1 : Math.random();
      var signRight = allowNegative ? Math.random() * 2 - 1 : Math.random();
      var jumpLeft = active ? signLeft * range * 60 : 0;
      var jumpRight = active ? signRight * range * 60 : 0;

      left += '<line x1="' + x.toFixed(1) + '" y1="80" x2="' + x.toFixed(1) +
        '" y2="' + (80 - jumpLeft).toFixed(1) + '" stroke="var(--accent)" ' +
        'stroke-width="1.4" opacity="' + (active ? 0.7 : 0.18) + '"/>';
      right += '<line x1="' + (x + 1.5).toFixed(1) + '" y1="80" x2="' +
        (x + 1.5).toFixed(1) + '" y2="' + (80 - jumpRight).toFixed(1) +
        '" stroke="#b7f0e2" stroke-width="1.1" opacity="' + (active ? 0.5 : 0.12) +
        '"/>';
    }

    return heroSvgMeet(grid + left + right);
  }
