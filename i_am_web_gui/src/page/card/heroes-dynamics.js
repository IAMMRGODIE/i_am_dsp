// Part five: the pictures for the curve, the mapping, the OTT, the vocoder, the EQ and the dynamics.
  function curveHero(values, context) {
    var grid =
      '<line x1="20" y1="80" x2="500" y2="80" stroke="#2f2f2f"/>' +
      '<line x1="260" y1="14" x2="260" y2="146" stroke="#2a2a2a"/>' +
      '<line x1="20" y1="142" x2="500" y2="18" stroke="#3a3a3a" stroke-dasharray="3 5"/>';
    var points = context && context.curve && context.curve.points;

    if (!points || points.length < 2) {
      return heroSvg(grid) +
        '<div class="hero-empty">Waiting for the script...</div>';
    }

    var path = "";

    for (var i = 0; i < points.length; i++) {
      var x = clamp(points[i][0], -1, 1);
      var y = clamp(points[i][1], -2, 2);
      var px = 20 + (x + 1) * 0.5 * 480;
      var py = 80 - y / 2 * 62;
      path += (i ? " L " : "M ") + px.toFixed(1) + " " + py.toFixed(1);
    }

    return heroSvg(
      grid +
      '<path d="' + path + '" fill="none" stroke="var(--accent)" stroke-width="1.8" ' +
      'vector-effect="non-scaling-stroke"/>'
    ) + '<div class="hero-label" style="top:2%">in -1..1 / out -2..2</div>';
  }

  function mappingHero(values, context) {
    var grid =
      '<rect x="20" y="12" width="480" height="82" fill="none" stroke="#2a2a2a"/>' +
      '<line x1="20" y1="94" x2="500" y2="12" stroke="#3a3a3a" stroke-dasharray="3 5"/>' +
      '<line x1="20" y1="148" x2="500" y2="148" stroke="#2f2f2f"/>' +
      '<line x1="20" y1="116" x2="500" y2="116" stroke="#242424"/>';
    var points = context && context.curve && context.curve.points;

    if (!points || points.length < 2) {
      return heroSvg(grid) +
        '<div class="hero-empty">Waiting for the script...</div>';
    }

    var span = Math.log(1000);
    var frequencyPath = "";
    var amplitudePath = "";

    for (var i = 0; i < points.length; i++) {
      var input = clamp(points[i][0], 20, 20000);
      var mapped = clamp(points[i][1], 20, 20000);
      var amplitude = clamp(points[i][2], 0, 2);
      var x = 20 + Math.log(input / 20) / span * 480;
      var y = 94 - Math.log(mapped / 20) / span * 82;
      var amp = 148 - amplitude / 2 * 32;

      frequencyPath += (i ? " L " : "M ") + x.toFixed(1) + " " + y.toFixed(1);
      amplitudePath += (i ? " L " : "M ") + x.toFixed(1) + " " + amp.toFixed(1);
    }

    return heroSvg(
      grid +
      '<path d="' + frequencyPath + '" fill="none" stroke="var(--accent)" stroke-width="1.6" ' +
      'vector-effect="non-scaling-stroke"/>' +
      '<path d="' + amplitudePath + '" fill="none" stroke="#9a9a9a" stroke-width="1.2" ' +
      'vector-effect="non-scaling-stroke"/>'
    ) +
      '<div class="hero-label" style="top:3%">frequency in / out</div>' +
      '<div class="hero-label" style="top:68%">amplitude</div>';
  }

  function ottHero(values, context) {
    var bands = context && context.ott && context.ott.bands ? context.ott.bands : [0, 0, 0];
    var grid = '<line x1="260" y1="10" x2="260" y2="150" stroke="#2f2f2f"/>';
    var bars = "";

    for (var i = 0; i < 3; i++) {
      var y = 30 + i * 45;
      var db = bands[i] || 0;
      var x = db >= 0
        ? 260 + Math.min(db, 60) / 60 * 230
        : 260 + Math.max(db, -24) / 24 * 100;
      var left = Math.min(260, x);
      var width = Math.abs(x - 260);
      var color = db >= 0 ? "var(--accent)" : "var(--negative, #9a8fbf)";

      bars += '<rect x="' + left.toFixed(1) + '" y="' + (y - 7) +
        '" width="' + width.toFixed(1) + '" height="14" fill="' + color +
        '" fill-opacity=".55" rx="2"/>';
    }

    return heroSvg(grid + bars) +
      '<div class="hero-label" style="left:2%;top:12%">LOW</div>' +
      '<div class="hero-label" style="left:2%;top:40%">MID</div>' +
      '<div class="hero-label" style="left:2%;top:68%">HIGH</div>';
  }

  function vocoderHero(values, context) {
    var state = context && context.vocoder;
    var bands = state && state.bands ? Math.min(state.bands, 128) : 24;
    var input = state && state.input ? state.input : [];
    var output = state && state.output ? state.output : [];
    var grid = '<line x1="20" y1="80" x2="500" y2="80" stroke="#2f2f2f"/>';
    var bars = "";
    var slot = 480 / bands;
    var bandwidth = clamp(valueOf(values, "bandwidth", 1.0), 0.25, 4.0);
    var bar_width = Math.max(1, slot * bandwidth - 1);

    for (var i = 0; i < bands; i++) {
      var x = 20 + (i + 0.5) * slot - bar_width / 2;
      // The bands are log-spaced from 20 Hz to 20 kHz, the same range the spectrum
      // pictures cover, so the same +3 dB per octave applies: without it the low bars
      // are pinned and the high ones have nothing to show.
      var tilt = Math.pow(10, tiltDb(bandHz(i, bands)) / 20);
      var in_height = Math.sqrt(clamp((input[i] || 0) * tilt, 0, 1)) * 60;
      var out_height = Math.sqrt(clamp((output[i] || 0) * tilt, 0, 1)) * 60;

      bars += '<rect x="' + x.toFixed(1) + '" y="' + (80 - in_height).toFixed(1) +
        '" width="' + bar_width.toFixed(1) + '" height="' + in_height.toFixed(1) +
        '" fill="#8a8a8a" fill-opacity=".5"/>';
      bars += '<rect x="' + x.toFixed(1) + '" y="80" width="' + bar_width.toFixed(1) +
        '" height="' + out_height.toFixed(1) + '" fill="var(--accent)" ' +
        'fill-opacity=".65"/>';
    }

    var source = valueOf(values, "sidechain", 1) ? "SIDECHAIN" : "SELF";

    return heroSvg(grid + bars) +
      '<div class="hero-label" style="left:2%;top:3%">IN ' + source + '</div>' +
      '<div class="hero-label" style="left:2%;top:55%">OUTPUT</div>';
  }

  function eqCoefficients(kind, frequency, gainDb, q, rate) {
    var cutoff = clamp(frequency, 10, rate * 0.49);
    var w = 2 * Math.PI * cutoff / rate;
    var cos = Math.cos(w);
    var sin = Math.sin(w);
    var alpha = sin / (2 * clamp(q, 0.05, 20));
    var a = Math.pow(10, gainDb / 40);
    var b0, b1, b2, a0, a1, a2;

    if (kind === 1 || kind === 2) {
      var two = 2 * Math.sqrt(a) * alpha;

      if (kind === 1) {
        b0 = a * ((a + 1) - (a - 1) * cos + two);
        b1 = 2 * a * ((a - 1) - (a + 1) * cos);
        b2 = a * ((a + 1) - (a - 1) * cos - two);
        a0 = (a + 1) + (a - 1) * cos + two;
        a1 = -2 * ((a - 1) + (a + 1) * cos);
        a2 = (a + 1) + (a - 1) * cos - two;
      } else {
        b0 = a * ((a + 1) + (a - 1) * cos + two);
        b1 = -2 * a * ((a - 1) + (a + 1) * cos);
        b2 = a * ((a + 1) + (a - 1) * cos - two);
        a0 = (a + 1) - (a - 1) * cos + two;
        a1 = 2 * ((a - 1) - (a + 1) * cos);
        a2 = (a + 1) - (a - 1) * cos - two;
      }
    } else if (kind === 3) {
      b0 = (1 + cos) * 0.5;
      b1 = -(1 + cos);
      b2 = b0;
      a0 = 1 + alpha;
      a1 = -2 * cos;
      a2 = 1 - alpha;
    } else if (kind === 4) {
      b0 = (1 - cos) * 0.5;
      b1 = 1 - cos;
      b2 = b0;
      a0 = 1 + alpha;
      a1 = -2 * cos;
      a2 = 1 - alpha;
    } else if (kind === 5) {
      b0 = 1;
      b1 = -2 * cos;
      b2 = 1;
      a0 = 1 + alpha;
      a1 = -2 * cos;
      a2 = 1 - alpha;
    } else {
      b0 = 1 + alpha * a;
      b1 = -2 * cos;
      b2 = 1 - alpha * a;
      a0 = 1 + alpha / a;
      a1 = -2 * cos;
      a2 = 1 - alpha / a;
    }

    return { b: [b0 / a0, b1 / a0, b2 / a0], a: [a1 / a0, a2 / a0] };
  }

  function eqResponseDb(frequency, bands, rate) {
    var magnitude = 1;

    for (var i = 0; i < bands.length; i++) {
      var band = bands[i];
      if (!band.on) continue;

      var c = eqCoefficients(band.type, band.freq, band.gain, band.q, rate);
      var w = 2 * Math.PI * clamp(frequency, 10, rate * 0.49) / rate;
      var cos = Math.cos(w);
      var sin = Math.sin(w);
      var cos2 = Math.cos(2 * w);
      var sin2 = Math.sin(2 * w);
      var nr = c.b[0] + c.b[1] * cos + c.b[2] * cos2;
      var ni = -c.b[1] * sin - c.b[2] * sin2;
      var dr = 1 + c.a[0] * cos + c.a[1] * cos2;
      var di = -c.a[0] * sin - c.a[1] * sin2;
      var bandMagnitude = Math.sqrt(nr * nr + ni * ni) / Math.sqrt(dr * dr + di * di);

      if (band.type === 3 || band.type === 4) {
        bandMagnitude = Math.pow(bandMagnitude, clamp(band.slope, 0, 3) + 1);
      }

      magnitude *= bandMagnitude;
    }

    return 20 * Math.log10(Math.max(magnitude, 1e-6));
  }

  function eqHero(values, context) {
    var rate = (context && context.rate) || 48000;
    var selected = Math.round(valueOf(values, "band", 0));
    var output = valueOf(values, "output", 0);
    var bands = [];

    for (var n = 1; n <= 6; n++) {
      bands.push({
        on: !!valueOf(values, "b" + n + "_on", 0),
        type: Math.round(valueOf(values, "b" + n + "_type", 0)),
        freq: valueOf(values, "b" + n + "_freq", 1000),
        gain: valueOf(values, "b" + n + "_gain", 0),
        q: valueOf(values, "b" + n + "_q", 1),
        slope: Math.round(valueOf(values, "b" + n + "_slope", 0))
      });
    }

    var grid = "";
    var labelsHtml = "";
    var dotsHtml = "";
    var ticks = [20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000];
    var labels = { 100: "100", 1000: "1k", 10000: "10k" };

    for (var t = 0; t < ticks.length; t++) {
      var tx = EQ_PLOT.left + Math.log(ticks[t] / 20) / Math.log(1000) * EQ_PLOT.across;
      grid += '<line x1="' + tx.toFixed(1) + '" y1="' + EQ_PLOT.top + '" x2="' + tx.toFixed(1) +
        '" y2="' + EQ_PLOT.bottom + '" stroke="#252525"/>';

      if (labels[ticks[t]]) {
        labelsHtml += heroLabel(tx, 157, labels[ticks[t]],
          "transform:translate(-50%,-100%)");
      }
    }

    for (var db = -24; db <= 24; db += 12) {
      var gy = EQ_PLOT.middle - db / 24 * EQ_PLOT.half;
      grid += '<line x1="' + EQ_PLOT.left + '" y1="' + gy.toFixed(1) + '" x2="' + EQ_PLOT.right + '" y2="' + gy.toFixed(1) +
        '" stroke="' + (db === 0 ? "#363636" : "#252525") + '"/>';
      labelsHtml += heroLabel(22, gy - 3, (db > 0 ? "+" : "") + db);
    }

    grid += '<rect x="' + EQ_PLOT.left + '" y="' + EQ_PLOT.top + '" width="' + EQ_PLOT.across +
      '" height="' + EQ_PLOT.down + '" fill="none" stroke="#2a2a2a"/>';

    var curve = "";

    for (var i = 0; i <= 240; i++) {
      var frequency = 20 * Math.pow(1000, i / 240);
      var decibels = eqResponseDb(frequency, bands, rate) + output;
      var x = EQ_PLOT.left + i / 240 * EQ_PLOT.across;
      var y = clamp(EQ_PLOT.middle - decibels / 24 * EQ_PLOT.half, EQ_PLOT.top, EQ_PLOT.bottom);
      curve += (i ? " L " : "M ") + x.toFixed(1) + " " + y.toFixed(1);
    }

    var spectrum = "";
    var dt = (context && context.dt) || 16;
    var scope = context && context.scope;
    var spectra = context && context.spectra;
    var eqSpectrum = context && context.eqSpectrum;

    if (scope && scope.input && scope.output && spectra && eqSpectrum) {
      if (eqSpectrum.input !== scope.input || eqSpectrum.output !== scope.output ||
          eqSpectrum.rate !== rate) {
        var inputBands = bandsOf(scope.input, rate);
        var outputBands = bandsOf(scope.output, rate);

        if (inputBands && outputBands) {
          spectra.input = smoothBands(spectra.input, inputBands, dt);
          spectra.output = smoothBands(spectra.output, outputBands, dt);
          eqSpectrum.input = scope.input;
          eqSpectrum.output = scope.output;
          eqSpectrum.rate = rate;
          eqSpectrum.inputPath = bandsPath(spectra.input);
          eqSpectrum.outputPath = bandsPath(spectra.output);
        }
      }

      if (eqSpectrum.inputPath) {
        spectrum =
          '<path d="' + eqSpectrum.inputPath + '" fill="none" stroke="#8a8a8a" ' +
            'stroke-width="1" opacity=".4" vector-effect="non-scaling-stroke"/>' +
          '<path d="' + eqSpectrum.outputPath + '" fill="none" stroke="var(--accent)" ' +
            'stroke-width="1.2" opacity=".45" vector-effect="non-scaling-stroke"/>';
      }
    }

    var nodes = "";

    for (var b = 0; b < bands.length; b++) {
      var band = bands[b];
      var bx = EQ_PLOT.left +
        Math.log(clamp(band.freq, 20, 20000) / 20) / Math.log(1000) * EQ_PLOT.across;
      var by = EQ_PLOT.middle - (band.type <= 2 ? band.gain : 0) / 24 * EQ_PLOT.half;
      var active = b === selected;
      var color = active ? "var(--accent)" : "#777";
      var opacity = band.on ? 1 : 0.35;
      var radius = active ? 4.5 : 3.2;

      labelsHtml += heroLabel(bx + 7, by - 7, b + 1,
        "color:" + color + ";opacity:" + opacity + ";font-size:9px");

      // The dot is HTML: the picture is stretched with the card, and a circle
      // drawn in it would stretch into an ellipse. The transparent circle
      // behind it stays in the picture's coordinates, where the drag reads
      // the pointer from.
      var dotSize = Math.round(radius * 2);

      dotsHtml += '<div class="hero-dot" data-eq-band="' + b + '" style="left:' +
        (bx / 5.2).toFixed(2) + '%;top:' + (by / 1.6).toFixed(2) + '%;width:' +
        dotSize + 'px;height:' + dotSize + 'px;background:' + color +
        ';opacity:' + opacity + '"></div>';

      nodes +=
        '<circle cx="' + bx.toFixed(1) + '" cy="' + by.toFixed(1) +
          '" r="18" fill="#000" fill-opacity="0" pointer-events="all"' +
          ' data-eq-band="' + b + '" style="cursor:move"/>';
    }

    return heroSvg(
      grid +
      spectrum +
      '<path d="' + curve + '" fill="none" stroke="var(--accent)" stroke-width="2" ' +
        'vector-effect="non-scaling-stroke"/>' +
      nodes
    ) + labelsHtml + dotsHtml;
  }

  /// A gate, expander or limiter law drawn on its own, across the whole picture.
  /// The meter version needs envelope history to plot beside it; a page that
  /// only has parameters (the rack card) gets this instead.
  function dynamicsSimpleHero(values, mode) {
    var threshold = clamp(valueOf(values, "threshold", -40), -80, 0);
    var range = clamp(valueOf(values, "range", 60), 0, 80);
    var ratio = Math.max(valueOf(values, "ratio", 2), 1);
    var ceiling = clamp(valueOf(values, "ceiling", -0.1), -30, 0);
    var makeup = clamp(valueOf(values, "output", 0), -24, 24);
    var points = [];

    function xFor(db) { return 20 + (db + 60) / 60 * 480; }
    function yFor(db) { return 148 - (db + 60) / 60 * 136; }

    for (var i = 0; i <= 120; i++) {
      var input = -60 + i / 120 * 60;
      var output;

      if (mode === "limiter") output = Math.min(input, ceiling);
      else if (mode === "gate") output = input >= threshold ? input : input - range;
      else output = input >= threshold ? input : threshold + (input - threshold) * ratio;

      output = clamp(output + makeup, -60, 0);
      points.push((i ? "L " : "M ") + xFor(input).toFixed(1) + " " + yFor(output).toFixed(1));
    }

    var law = mode === "limiter" ? ceiling : threshold;
    var lawX = xFor(law);

    return heroSvg(
      '<line x1="20" y1="12" x2="20" y2="148" stroke="#2b2b2b"/>' +
      '<line x1="20" y1="148" x2="500" y2="148" stroke="#2b2b2b"/>' +
      '<line x1="20" y1="148" x2="500" y2="12" stroke="#333" stroke-width="1" ' +
      'stroke-dasharray="3 5" vector-effect="non-scaling-stroke"/>' +
      '<line x1="' + lawX.toFixed(1) + '" y1="12" x2="' + lawX.toFixed(1) +
      '" y2="148" stroke="#3a2a2a"/>' +
      '<path d="' + points.join(" ") + '" fill="none" stroke="var(--accent)" ' +
      'stroke-width="2.2" vector-effect="non-scaling-stroke"/>'
    ) +
      heroLabel(lawX + 4, 22, law.toFixed(0) + " dB", "color:#777;font-size:9px") +
      heroLabel(462, 140, mode === "limiter" ? "0 dBFS" : ratio.toFixed(1) + ":1",
        "color:#777;font-size:9px");
  }

  function dynamicsHero(values, mode, context) {
    if (!context || !context.envHistory || !context.envHistory.input.length) {
      return dynamicsSimpleHero(values, mode);
    }

    var threshold = clamp(valueOf(values, "threshold", -40), -80, 0);
    var range = clamp(valueOf(values, "range", 60), 0, 80);
    var ratio = Math.max(valueOf(values, "ratio", 2), 1);
    var ceiling = clamp(valueOf(values, "ceiling", -0.1), -30, 0);
    var makeup = clamp(valueOf(values, "output", 0), -24, 24);
    var sidechainOn = !!valueOf(values, "sidechain", 0);
    var meterState = (context && context.meter) || {};
    var level = sidechainOn ? meterState.sidechain : meterState.input;
    var env = (context && context.envHistory) || { input: [], output: [] };
    var window = (context && context.envWindow) || 128;
    var path = "";

    for (var i = 0; i <= 120; i++) {
      var input = -60 + i / 120 * 60;
      var output;

      if (mode === "limiter") output = Math.min(input, ceiling);
      else if (mode === "gate") output = input >= threshold ? input : input - range;
      else output = input >= threshold ? input : threshold + (input - threshold) * ratio;

      output = clamp(output + makeup, -60, 0);
      var x = 8 + i / 120 * 124;
      var y = 132 - (output + 60) / 60 * 124;
      path += (i ? " L " : "M ") + x.toFixed(1) + " " + y.toFixed(1);
    }

    var law = mode === "limiter" ? ceiling : threshold;
    var lawX = 8 + (law + 60) / 60 * 124;
    var marker = "";

    if (level !== null && level !== undefined && isFinite(level)) {
      var db = clamp(level, -60, 0);
      var mx = 8 + (db + 60) / 60 * 124;
      var out;

      if (mode === "limiter") out = Math.min(db, ceiling);
      else if (mode === "gate") out = db >= threshold ? db : db - range;
      else out = db >= threshold ? db : threshold + (db - threshold) * ratio;
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
        '<line x1="' + lawX.toFixed(1) + '" y1="8" x2="' + lawX.toFixed(1) + '" y2="132" stroke="#343434" stroke-dasharray="3 5"/>' +
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
