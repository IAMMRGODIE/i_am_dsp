// The card a faceplate and a rack module agree on.
//
// A product page is one card: a header with the mode and header choices, an
// optional picture, and the controls the effect's metadata declares. A rack
// module is the same card without the product title. This file keeps no global
// parameter map, so several cards on one page cannot overwrite each other.
(function () {
  "use strict";

  var START = -225;
  var SWEEP = 270;
  var knobEditor = null;

  function clamp(value, low, high) {
    return Math.min(high, Math.max(low, value));
  }

  function curveOf(spec) {
    return spec.curve || null;
  }

  function atanKnee(spec) {
    var limit = Math.max(Math.abs(spec.min), Math.abs(spec.max), 1);
    return { limit: limit, knee: Math.max(limit * 0.1, 1e-3) };
  }

  function norm(spec, value) {
    var low = spec.min;
    var high = spec.max;

    if (curveOf(spec) === "atan") {
      var k = atanKnee(spec);
      return 0.5 + Math.atan(value / k.knee) / (2 * Math.atan(k.limit / k.knee));
    }

    if (spec.log && low > 0 && high > low) {
      return clamp(Math.log(value / low) / Math.log(high / low), 0, 1);
    }

    return high > low ? clamp((value - low) / (high - low), 0, 1) : 0;
  }

  function at(spec, position) {
    var low = spec.min;
    var high = spec.max;
    position = clamp(position, 0, 1);

    if (curveOf(spec) === "atan") {
      var k = atanKnee(spec);
      return k.knee * Math.tan((position - 0.5) * 2 * Math.atan(k.limit / k.knee));
    }

    var value = spec.log && low > 0 && high > low
      ? low * Math.pow(high / low, position)
      : low + (high - low) * position;
    return clamp(value, low, high);
  }

  function format(spec, value) {
    if (spec.kind === "bool") return value ? "On" : "Off";
    if (spec.kind === "int" && spec.choices && spec.choices.length) {
      return spec.choices[clamp(Math.round(value), 0, spec.choices.length - 1)];
    }
    if (spec.unit === "hz") {
      if (value >= 1000) return (value / 1000).toFixed(2) + " kHz";
      if (value < 100) return value.toFixed(2) + " Hz";
      return value.toFixed(0) + " Hz";
    }
    if (spec.unit === "percent") return (value * 100).toFixed(0) + " %";
    if (spec.unit === "db") return (value > 0 ? "+" : "") + value.toFixed(1) + " dB";
    if (spec.unit === "ms") return value.toFixed(1) + " ms";
    if (spec.unit === "s") return value.toFixed(3) + " s";
    if (spec.unit === "beats") return value.toFixed(2) + " beats";
    if (spec.unit === "samples") return Math.round(value) + " smp";
    if (spec.unit === "st") return (value > 0 ? "+" : "") + Math.round(value) + " st";
    if (spec.unit === "ct") return (value > 0 ? "+" : "") + Math.round(value) + " ct";
    return spec.kind === "int" ? String(Math.round(value)) : value.toFixed(2);
  }

  function closeKnobEditor() {
    if (knobEditor) {
      knobEditor.remove();
      knobEditor = null;
    }
  }

  /// A small text field over a knob, for entering an exact value.
  function openKnobEditor(spec, current, set, anchor) {
    closeKnobEditor();

    var box = document.createElement("div");
    box.className = "knob-edit";

    var input = document.createElement("input");
    input.type = "text";
    input.value = String(Number(current.toFixed(4)));
    input.spellcheck = false;
    input.setAttribute("aria-label", spec.label + " value");

    var unit = document.createElement("span");
    unit.className = "unit";
    unit.textContent = spec.unit || "";

    box.appendChild(input);
    box.appendChild(unit);
    document.body.appendChild(box);

    var rect = anchor.getBoundingClientRect();
    var width = 118;
    var top = rect.bottom + 4;

    if (top + 30 > window.innerHeight) top = Math.max(4, rect.top - 30);

    box.style.left = Math.min(window.innerWidth - width, Math.max(4, rect.left)) + "px";
    box.style.top = top + "px";

    input.focus();
    input.select();

    input.addEventListener("keydown", function (event) {
      if (event.key === "Enter") {
        var parsed = Number.parseFloat(input.value);
        if (Number.isFinite(parsed)) set(parsed, true);
        closeKnobEditor();
      } else if (event.key === "Escape") {
        closeKnobEditor();
      }
    });

    input.addEventListener("blur", function () {
      setTimeout(closeKnobEditor, 0);
    });

    knobEditor = box;
  }

  function svg(tag, attributes) {
    var node = document.createElementNS("http://www.w3.org/2000/svg", tag);
    for (var key in attributes) {
      node.setAttribute(key, attributes[key]);
    }
    return node;
  }

  function polar(cx, cy, radius, degrees) {
    var radians = degrees * Math.PI / 180;
    return {
      x: cx + radius * Math.cos(radians),
      y: cy + radius * Math.sin(radians)
    };
  }

  function arcPath(cx, cy, radius, from, to) {
    if (Math.abs(to - from) < 0.01) return "";
    var start = polar(cx, cy, radius, from);
    var end = polar(cx, cy, radius, to);
    var large = Math.abs(to - from) > 180 ? 1 : 0;
    var sweep = to > from ? 1 : 0;
    return "M " + start.x.toFixed(2) + " " + start.y.toFixed(2) +
      " A " + radius + " " + radius + " 0 " + large + " " + sweep + " " +
      end.x.toFixed(2) + " " + end.y.toFixed(2);
  }

  function valueOf(values, id, fallback) {
    var value = values ? values[id] : undefined;
    return typeof value === "number" ? value : fallback;
  }

  function heroSvg(inner) {
    return '<svg class="mcard-graph" viewBox="0 0 520 160" preserveAspectRatio="none">' +
      inner +
      '</svg>';
  }

  function heroSvgMeet(inner) {
    return '<svg class="mcard-graph" viewBox="0 0 520 160" preserveAspectRatio="xMidYMid meet">' +
      inner +
      '</svg>';
  }

  function wavePath(amplitude, middle, cycles) {
    var path = "";
    for (var i = 0; i <= 180; i++) {
      var x = 20 + i / 180 * 480;
      var y = middle - Math.sin(i / 180 * Math.PI * 2 * cycles) * amplitude;
      path += (i ? " L " : "M ") + x.toFixed(1) + " " + y.toFixed(1);
    }
    return path;
  }

  function spectrumHero(values) {
    var cutoff = Math.max(valueOf(values, "cutoff", 1000), 10);
    var q = Math.max(valueOf(values, "q", 0.707), 0.05);
    var mode = Math.round(valueOf(values, "mode", 0));
    var gainDb = valueOf(values, "gain_db", 0);
    var slope = Math.max(valueOf(values, "slope", 1), 0.5);
    var sections = mode < 4
      ? clamp(Math.round(valueOf(values, "stages", 0)) + 1, 1, 4)
      : 1;
    var magnitudes = [];

    for (var i = 0; i < 80; i++) {
      var frequency = 20 * Math.pow(1000, i / 79);
      var ratio = frequency / cutoff;
      var denominator = Math.sqrt(
        Math.pow(1 - ratio * ratio, 2) + Math.pow(ratio / q, 2)
      );
      var magnitude;

      if (mode === 1) magnitude = (ratio * ratio) / denominator;
      else if (mode === 2) magnitude = (ratio / q) / denominator;
      else if (mode === 3) magnitude = Math.abs(1 - ratio * ratio) / denominator;
      else if (mode === 4) {
        var peakGain = Math.pow(10, gainDb / 20);
        var octaves = Math.max(0.08, Math.log2(1 + 1 / q));
        var bump = Math.exp(-0.5 * Math.pow(Math.log2(ratio) / octaves, 2));
        magnitude = 1 + (peakGain - 1) * bump;
      } else if (mode === 5 || mode === 6) {
        var shelfGain = Math.pow(10, gainDb / 20);
        var transition = 1 / (1 + Math.pow(ratio, 4 * slope));
        magnitude = mode === 5
          ? 1 + (shelfGain - 1) * transition
          : 1 + (shelfGain - 1) * (1 - transition);
      } else magnitude = 1 / denominator;

      if (mode < 4) magnitude = Math.pow(magnitude, sections);

      magnitudes.push(magnitude);
    }

    var loudest = Math.max.apply(null, magnitudes);
    var points = [];

    for (var j = 0; j < magnitudes.length; j++) {
      var decibels = 20 * Math.log10(Math.max(magnitudes[j] / loudest, 1e-6));
      var x = 20 + j / 79 * 480;
      var y = clamp(10 + (1 - (1 + decibels / 40)) * 140, 10, 150);
      points.push((j ? "L " : "M ") + x.toFixed(1) + " " + y.toFixed(1));
    }

    return heroSvg(
      '<path d="' + points.join(" ") + '" fill="none" stroke="var(--accent)" ' +
      'stroke-width="2" vector-effect="non-scaling-stroke"/>'
    );
  }

  function transferHero(values) {
    var mode = Math.round(valueOf(values, "mode", 0));
    var threshold = clamp(valueOf(values, "threshold", 0.8), 0.01, 4);
    var gain = valueOf(values, "gain", 1);
    var alpha = Math.max(valueOf(values, "alpha", 0.5), 0.01);
    var k = valueOf(values, "k", 2);
    var a = Math.max(valueOf(values, "a", 1), 0.01);
    var p = valueOf(values, "p", 2);
    var bits = Math.max(1, valueOf(values, "bits", 8));
    var path = "";

    // The domain is wider than the clip point on purpose: a hard clipper's curve
    // only shows its knees when the flat part has somewhere to run.
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
      '<path d="M 20 146 L 500 14" stroke="#333" stroke-width="1" stroke-dasharray="3 5" ' +
      'vector-effect="non-scaling-stroke"/>' +
      '<line x1="260" y1="12" x2="260" y2="148" stroke="#2b2b2b"/>' +
      '<line x1="20" y1="80" x2="500" y2="80" stroke="#2b2b2b"/>' +
      '<path d="' + path + '" fill="none" stroke="var(--accent)" stroke-width="2.4" ' +
      'vector-effect="non-scaling-stroke"/>'
    );
  }

  function compressorHero(values) {
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
      'stroke-width="2.2" vector-effect="non-scaling-stroke"/>' +
      '<text x="' + (thresholdX + 4).toFixed(1) + '" y="22" fill="#777" font-size="9">' +
      threshold.toFixed(0) + ' dB</text>' +
      '<text x="462" y="140" fill="#777" font-size="9">' + ratio.toFixed(1) + ':1</text>'
    );
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

    return heroSvg(
      '<line x1="20" y1="80" x2="500" y2="80" stroke="#2f2f2f"/>' +
      '<path d="' + top + bottom + ' Z" fill="var(--accent)" fill-opacity=".14" stroke="' +
        'var(--accent)" stroke-width="1" vector-effect="non-scaling-stroke"/>'
    ) + '<div class="hero-label" style="top:2%">' + escapeXml(label) + '</div>';
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
        '" y2="130" stroke="var(--accent)" stroke-width="2"/>' +
      '<circle cx="' + marker.toFixed(1) + '" cy="80" r="4" fill="var(--accent)"/>'
    ) + '<div class="hero-badge">' + sign + semitones.toFixed(1) + ' st</div>';
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

    for (var tap = 0; tap < 8; tap++) {
      var at = time * (tap + 1);
      if (at <= 0 || at > 2000) break;

      var x = 20 + at / 2000 * 480;
      var level = Math.pow(feedback, tap) * Math.min(mix, 1);
      var height = 16 + 100 * level;

      taps += '<line x1="' + x.toFixed(1) + '" y1="' + base + '" x2="' + x.toFixed(1) +
        '" y2="' + (base - height).toFixed(1) + '" stroke="var(--accent)" stroke-width="2" ' +
        'opacity="' + (0.3 + 0.7 * level).toFixed(2) + '"/>' +
        '<circle cx="' + x.toFixed(1) + '" cy="' + (base - height).toFixed(1) +
        '" r="2" fill="var(--accent)"/>';
    }

    return heroSvg(
      '<line x1="20" y1="' + base + '" x2="500" y2="' + base + '" stroke="#3a3a3a"/>' +
      taps
    ) + (sync ? '<div class="hero-badge">SYNC</div>' : "");
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

  function waveformHero(values) {
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

  var HEROES = Object.create(null);

  function registerHero(kind, renderer) {
    HEROES[kind] = renderer;
  }

  function hero(kind, values, context) {
    var renderer = HEROES[kind];
    return renderer ? renderer(values, context) : "";
  }

  registerHero("spectrum", spectrumHero);
  registerHero("transfer", transferHero);
  registerHero("compressor", compressorHero);
  registerHero("impulse", impulseHero);
  registerHero("pitch", pitchHero);
  registerHero("particles", particlesHero);
  registerHero("curve", curveHero);
  registerHero("mapping", mappingHero);
  registerHero("tail", tailHero);
  registerHero("taps", tapsHero);
  registerHero("lfo", lfoHero);
  registerHero("phase", phaseHero);
  registerHero("balance", balanceHero);
  registerHero("waveform", waveformHero);

  function makeKnob(spec, value, onEdit) {
    var wrap = document.createElement("div");
    wrap.className = "knob-wrap";

    var dial = document.createElement("div");
    dial.className = "knob";
    dial.tabIndex = 0;
    dial.setAttribute("role", "slider");
    dial.setAttribute("aria-label", spec.label);
    dial.setAttribute("aria-valuemin", String(spec.min));
    dial.setAttribute("aria-valuemax", String(spec.max));

    var drawing = svg("svg", { viewBox: "0 0 64 64" });
    drawing.appendChild(svg("circle", { class: "knob-bg", cx: 32, cy: 32, r: 25 }));
    drawing.appendChild(svg("path", {
      class: "knob-track",
      d: arcPath(32, 32, 25, START, START + SWEEP)
    }));

    var modulations = spec.modulations || [];

    var mod = svg("path", { class: "knob-mod" });
    var fill = svg("path", { class: "knob-arc" });
    var hand = svg("line", { class: "knob-hand" });
    drawing.appendChild(mod);
    drawing.appendChild(fill);
    drawing.appendChild(hand);
    dial.appendChild(drawing);

    var label = document.createElement("div");
    label.className = "knob-label";
    label.textContent = spec.label;

    var readout = document.createElement("div");
    readout.className = "knob-value";

    var current = clamp(value, spec.min, spec.max);

    function paint() {
      var position = clamp(norm(spec, current), 0, 1);
      var angle = START + SWEEP * position;
      fill.setAttribute("d", arcPath(32, 32, 25, START, angle));

      var low = position;
      var high = position;

      modulations.forEach(function (assignment) {
        var depth = assignment.depth || 0;
        var mode = assignment.mode || "centered";

        if (mode === "frombase" || mode === "from-base" || mode === "start") {
          var end = clamp(position + depth, 0, 1);
          low = Math.min(low, end);
          high = Math.max(high, end);
        } else {
          var span = Math.abs(depth);
          low = Math.min(low, clamp(position - span, 0, 1));
          high = Math.max(high, clamp(position + span, 0, 1));
        }
      });

      if (high - low > 0.0001) {
        mod.setAttribute("d", arcPath(32, 32, 29, START + SWEEP * low, START + SWEEP * high));
        mod.style.display = "";
      } else {
        mod.style.display = "none";
      }
      var inner = polar(32, 32, 9, angle);
      var tip = polar(32, 32, 25, angle);
      hand.setAttribute("x1", inner.x.toFixed(2));
      hand.setAttribute("y1", inner.y.toFixed(2));
      hand.setAttribute("x2", tip.x.toFixed(2));
      hand.setAttribute("y2", tip.y.toFixed(2));
      readout.textContent = format(spec, current);
      dial.setAttribute("aria-valuenow", String(current));
      dial.setAttribute("aria-valuetext", format(spec, current));
    }

    function set(value, send) {
      current = clamp(value, spec.min, spec.max);
      if (spec.kind === "int") current = Math.round(current);
      paint();
      if (send && onEdit) onEdit(spec.id, current);
    }

    var drag = null;
    var modDrag = null;

    dial.addEventListener("pointerdown", function (event) {
      if (event.button !== 0) return;

      if (event.altKey && modulations.length) {
        modDrag = { y: event.clientY, depth: modulations[0].depth || 0 };
        dial.classList.add("mod-drag");
      } else {
        drag = { y: event.clientY, position: norm(spec, current) };
      }

      dial.setPointerCapture(event.pointerId);
      event.preventDefault();
    });

    dial.addEventListener("pointermove", function (event) {
      if (modDrag) {
        var next = clamp(modDrag.depth + (modDrag.y - event.clientY) / 150, -1, 1);
        modulations[0].depth = next;

        if (spec.onModulation && modulations[0].slot !== undefined) {
          spec.onModulation(modulations[0].slot, next);
        }

        paint();
        return;
      }

      if (!drag) return;
      // Shift is the fine drag, the same gesture the faceplate used to have.
      var distance = (drag.y - event.clientY) / 150 * (event.shiftKey ? 0.1 : 1);
      var position = clamp(drag.position + distance, 0, 1);
      set(at(spec, position), true);
    });

    function stop(event) {
      if (!drag && !modDrag) return;
      drag = null;
      modDrag = null;
      dial.classList.remove("mod-drag");
      try {
        dial.releasePointerCapture(event.pointerId);
      } catch (error) {
        // Released outside the window.
      }
    }

    dial.addEventListener("pointerup", stop);
    dial.addEventListener("pointercancel", stop);
    dial.addEventListener("dblclick", function () { set(spec.default, true); });
    dial.addEventListener("click", function (event) {
      if (!event.ctrlKey || !modulations.length) return;
      var next = (modulations[0].mode || "frombase") === "frombase" ? "center" : "start";
      if (spec.onModeToggle) {
        spec.onModeToggle(modulations.map(function (modulation) { return modulation.slot; }), next);
      }
      event.preventDefault();
    });
    dial.title = "Drag to change; right-click to type a value";
    dial.addEventListener("contextmenu", function (event) {
      event.preventDefault();
      openKnobEditor(spec, current, set, dial);
    });

    if (spec.dropTarget) {
      dial.addEventListener("dragover", function (event) {
        event.preventDefault();
        if (event.dataTransfer) event.dataTransfer.dropEffect = "link";
        dial.classList.add("drop-target");
      });
      dial.addEventListener("dragleave", function () {
        dial.classList.remove("drop-target");
      });
      dial.addEventListener("drop", function (event) {
        event.preventDefault();
        dial.classList.remove("drop-target");
        var data = event.dataTransfer ? event.dataTransfer.getData("text/plain") : "";
        if (data && spec.onModulationDrop) spec.onModulationDrop(data, spec.dropTarget);
      });
    }

    dial.addEventListener("wheel", function (event) {
      event.preventDefault();
      var step = spec.step && spec.step > 0 ? spec.step : (spec.max - spec.min) / 100;
      var direction = event.deltaY < 0 ? 1 : -1;
      set(current + direction * step * (event.shiftKey ? 0.1 : 1), true);
    }, { passive: false });

    dial.addEventListener("keydown", function (event) {
      var step = (spec.max - spec.min) / 100 * (event.shiftKey ? 0.1 : 1);
      if (event.key === "ArrowUp" || event.key === "ArrowRight") {
        set(current + step, true);
        event.preventDefault();
      } else if (event.key === "ArrowDown" || event.key === "ArrowLeft") {
        set(current - step, true);
        event.preventDefault();
      } else if (event.key === "Home") {
        set(spec.min, true);
        event.preventDefault();
      } else if (event.key === "End") {
        set(spec.max, true);
        event.preventDefault();
      }
    });

    paint();
    wrap.appendChild(dial);
    wrap.appendChild(label);
    wrap.appendChild(readout);

    return { el: wrap, set: set };
  }

  function makeChoiceHandle(spec, value, onEdit, className) {
    var select = document.createElement("select");
    select.className = className || "mcard-select";
    select.setAttribute("aria-label", spec.label);

    spec.choices.forEach(function (name, index) {
      var option = document.createElement("option");
      option.value = String(index);
      option.textContent = name;
      select.appendChild(option);
    });

    function set(next) {
      select.value = String(Math.round(next));
    }

    set(value);
    select.addEventListener("change", function () {
      if (onEdit) onEdit(spec.id, Number(select.value));
    });

    return { el: select, set: set };
  }

  function makeChoice(spec, value, onEdit, className) {
    return makeChoiceHandle(spec, value, onEdit, className).el;
  }

  function makeSliderHandle(spec, value, onEdit) {
    var wrap = document.createElement("div");
    wrap.className = "mcard-slider";

    var top = document.createElement("div");
    top.className = "mcard-slider-top";
    var label = document.createElement("span");
    label.className = "mcard-row-label";
    label.textContent = spec.label;
    var readout = document.createElement("span");
    readout.className = "mcard-slider-value";
    top.appendChild(label);
    top.appendChild(readout);
    wrap.appendChild(top);

    var track = document.createElement("div");
    track.className = "mcard-slider-track";
    track.tabIndex = 0;
    track.setAttribute("role", "slider");
    track.setAttribute("aria-label", spec.label);
    track.setAttribute("aria-valuemin", String(spec.min));
    track.setAttribute("aria-valuemax", String(spec.max));

    var fill = document.createElement("div");
    fill.className = "mcard-slider-fill";
    var knob = document.createElement("div");
    knob.className = "mcard-slider-handle";
    track.appendChild(fill);
    track.appendChild(knob);
    wrap.appendChild(track);

    var zero = spec.bipolar ? clamp(norm(spec, 0), 0, 1) : 0;
    var current = clamp(value, spec.min, spec.max);

    function paint() {
      var position = clamp(norm(spec, current), 0, 1);
      fill.style.left = (Math.min(zero, position) * 100) + "%";
      fill.style.width = (Math.abs(position - zero) * 100) + "%";
      knob.style.left = (position * 100) + "%";
      readout.textContent = format(spec, current);
      track.setAttribute("aria-valuenow", String(current));
      track.setAttribute("aria-valuetext", format(spec, current));
    }

    function set(next, send) {
      current = clamp(next, spec.min, spec.max);
      if (spec.kind === "int") current = Math.round(current);
      paint();
      if (send && onEdit) onEdit(spec.id, current);
    }

    function apply(event) {
      var box = track.getBoundingClientRect();
      var position = box.width > 0 ? (event.clientX - box.left) / box.width : 0;
      set(at(spec, position), true);
    }

    var drag = false;
    track.addEventListener("pointerdown", function (event) {
      if (event.button !== 0) return;
      drag = true;
      track.setPointerCapture(event.pointerId);
      event.preventDefault();
      apply(event);
    });
    track.addEventListener("pointermove", function (event) {
      if (drag) apply(event);
    });
    function stop(event) {
      if (!drag) return;
      drag = false;
      try { track.releasePointerCapture(event.pointerId); } catch (error) {}
    }
    track.addEventListener("pointerup", stop);
    track.addEventListener("pointercancel", stop);
    track.addEventListener("wheel", function (event) {
      event.preventDefault();
      var step = spec.step && spec.step > 0 ? spec.step : (spec.max - spec.min) / 100;
      set(current + (event.deltaY < 0 ? step : -step) * (event.shiftKey ? 0.1 : 1), true);
    }, { passive: false });
    track.addEventListener("keydown", function (event) {
      var step = spec.step && spec.step > 0 ? spec.step : (spec.max - spec.min) / 100;
      if (event.key === "ArrowUp" || event.key === "ArrowRight") {
        set(current + step, true);
        event.preventDefault();
      } else if (event.key === "ArrowDown" || event.key === "ArrowLeft") {
        set(current - step, true);
        event.preventDefault();
      } else if (event.key === "Home") {
        set(spec.min, true);
        event.preventDefault();
      } else if (event.key === "End") {
        set(spec.max, true);
        event.preventDefault();
      }
    });

    paint();
    return { el: wrap, set: set };
  }

  function makeSwitch(spec, value, onEdit) {
    var button = document.createElement("button");
    button.type = "button";
    button.className = "mcard-switch";
    var current = !!value;
    var thumb = document.createElement("i");
    button.appendChild(thumb);
    button.setAttribute("role", "switch");
    button.setAttribute("aria-label", spec.label);
    button.setAttribute("aria-checked", current ? "true" : "false");
    button.classList.toggle("on", current);

    function paint() {
      button.classList.toggle("on", current);
      button.setAttribute("aria-checked", current ? "true" : "false");
    }

    function set(value, send) {
      current = !!value;
      paint();
      if (send && onEdit) onEdit(spec.id, current ? 1 : 0);
    }

    button.addEventListener("click", function () {
      set(!current, true);
    });

    paint();
    return { el: button, set: set };
  }

  function makeToggle(spec, value, onEdit) {
    return makeSwitch(spec, value, onEdit).el;
  }

  function makeBodyControl(spec, value, onEdit) {
    if (spec.kind === "bool") {
      var row = document.createElement("div");
      row.className = "mcard-row";
      var toggleLabel = document.createElement("span");
      toggleLabel.className = "mcard-row-label";
      toggleLabel.textContent = spec.label;
      row.appendChild(toggleLabel);
      row.appendChild(makeSwitch(spec, value, onEdit).el);
      return row;
    }

    if (spec.widget === "slider") {
      return makeSliderHandle(spec, value, onEdit).el;
    }

    if (spec.widget === "choice" || (spec.kind === "int" && spec.choices && spec.choices.length)) {
      var choiceRow = document.createElement("div");
      choiceRow.className = "mcard-row";
      var choiceLabel = document.createElement("span");
      choiceLabel.className = "mcard-row-label";
      choiceLabel.textContent = spec.label;
      choiceRow.appendChild(choiceLabel);
      choiceRow.appendChild(makeChoice(spec, value, onEdit));
      return choiceRow;
    }

    return makeKnob(spec, value, onEdit).el;
  }

  function control(spec, value, onEdit) {
    return makeBodyControl(spec, value, onEdit);
  }

  /// A bare knob with a setter, for a faceplate that owns its own layout.
  function controlHandle(spec, value, onEdit) {
    return makeKnob(spec, value, onEdit);
  }

  function setting(label, control) {
    var box = document.createElement("div");
    box.className = "mcard-setting";
    var name = document.createElement("span");
    name.className = "mcard-setting-label";
    name.textContent = label.toUpperCase();
    box.appendChild(name);
    box.appendChild(control);
    return box;
  }

  function create(options) {
    var card = document.createElement("section");
    card.className = "mcard";
    card.style.setProperty("--accent", options.accent || "#8a6aff");
    if (options.collapsed) card.classList.add("collapsed");

    var values = {};
    (options.controls || []).forEach(function (spec) {
      values[spec.id] = valueOf(options.values, spec.id, spec.default);
    });

    var byId = {};
    (options.controls || []).forEach(function (spec) { byId[spec.id] = spec; });

    var head = document.createElement("header");
    head.className = "mcard-head";

    if (options.dragData !== undefined) {
      var handle = document.createElement("span");
      handle.className = "mcard-drag";
      handle.textContent = "\u28FF";
      handle.draggable = true;
      handle.title = "Drag module";
      handle.setAttribute("aria-label", "Drag module");
      handle.addEventListener("dragstart", function (event) {
        event.dataTransfer.setData("text/plain", String(options.dragData));
        event.dataTransfer.effectAllowed = "move";
        card.classList.add("dragging");
      });
      handle.addEventListener("dragend", function () {
        card.classList.remove("dragging");
      });
      head.appendChild(handle);
    }

    var collapse = document.createElement("button");
    collapse.type = "button";
    collapse.className = "mcard-collapse";
    collapse.textContent = options.collapsed ? "\u25B8" : "\u25BE";
    collapse.title = options.collapsed ? "Expand" : "Collapse";
    collapse.setAttribute("aria-label", collapse.title);
    collapse.addEventListener("click", function () {
      var collapsed = card.classList.toggle("collapsed");
      collapse.textContent = collapsed ? "\u25B8" : "\u25BE";
      collapse.title = collapsed ? "Expand" : "Collapse";
      collapse.setAttribute("aria-label", collapse.title);
      if (options.onCollapse) options.onCollapse(collapsed);
    });
    head.appendChild(collapse);

    var bypassed = !!options.bypass;

    var power = document.createElement("button");
    power.type = "button";
    power.className = "mcard-power";
    power.title = "Bypass";
    power.setAttribute("aria-label", "Bypass");
    power.setAttribute("aria-pressed", bypassed ? "true" : "false");
    if (bypassed) power.classList.add("off");
    var dot = document.createElement("span");
    dot.className = "dot";
    power.appendChild(dot);
    power.addEventListener("click", function () {
      bypassed = !bypassed;
      power.classList.toggle("off", bypassed);
      power.setAttribute("aria-pressed", bypassed ? "true" : "false");
      card.classList.toggle("bypassed", bypassed);
      if (options.onBypass) options.onBypass(bypassed);
    });
    head.appendChild(power);

    var name = document.createElement("span");
    name.className = "mcard-name";
    name.textContent = options.title;
    head.appendChild(name);

    var badge = document.createElement("span");
    badge.className = "mcard-badge";
    badge.textContent = options.badge || "";
    head.appendChild(badge);

    var spacer = document.createElement("span");
    spacer.className = "mcard-spacer";
    head.appendChild(spacer);

    if (options.kinds && options.kinds.length) {
      var kindSelect = document.createElement("select");
      kindSelect.className = "mcard-select mcard-kind";
      kindSelect.title = "Module type";
      kindSelect.setAttribute("aria-label", "Module type");

      options.kinds.forEach(function (kind) {
        var option = document.createElement("option");
        option.value = kind;
        option.textContent = kind;
        kindSelect.appendChild(option);
      });

      kindSelect.value = options.kind || options.kinds[0];
      kindSelect.addEventListener("change", function () {
        if (options.onKind) options.onKind(kindSelect.value);
      });
      head.appendChild(kindSelect);
    }

    if (options.onDuplicate) {
      card.classList.add("has-duplicate");
      var duplicate = document.createElement("button");
      duplicate.type = "button";
      duplicate.className = "mcard-duplicate";
      duplicate.textContent = "\u29C9";
      duplicate.title = "Duplicate module";
      duplicate.setAttribute("aria-label", "Duplicate module");
      duplicate.addEventListener("click", function () { options.onDuplicate(); });
      head.appendChild(duplicate);
    }

    if (options.onRemove) {
      card.classList.add("has-remove");
      var remove = document.createElement("button");
      remove.type = "button";
      remove.className = "mcard-remove";
      remove.textContent = "\u00D7";
      remove.title = "Remove module";
      remove.setAttribute("aria-label", "Remove module");
      remove.addEventListener("click", function () { options.onRemove(); });
      head.appendChild(remove);
    }

    var heroElement = null;
    function paintHero() {
      if (heroElement) heroElement.innerHTML = hero(options.hero, values);
    }

    function edited(id, value) {
      values[id] = value;
      paintHero();
      if (options.onEdit) options.onEdit(id, value);
    }

    var modeSpec = options.mode ? byId[options.mode] : null;
    var currentMode = modeSpec ? Math.round(values[modeSpec.id]) : 0;

    if (modeSpec) {
      var modeSelect = makeChoice(modeSpec, values[modeSpec.id], function (_, value) {
        currentMode = Math.round(value);
        applyVisibility();
        edited(modeSpec.id, currentMode);
      }, "mcard-select mcard-mode");
      head.appendChild(setting("Mode", modeSelect));
    }

    (options.header || []).forEach(function (id) {
      var spec = byId[id];
      if (!spec) return;
      var control;
      if (spec.kind === "int" && spec.choices && spec.choices.length) {
        control = makeChoice(spec, values[id], function (_, value) {
          edited(id, value);
        });
      } else {
        control = makeToggle(spec, values[id], function (_, value) {
          edited(id, value);
        });
      }
      head.appendChild(setting(spec.label, control));
    });

    card.appendChild(head);

    if (options.hero) {
      heroElement = document.createElement("div");
      heroElement.className = "mcard-hero";
      card.appendChild(heroElement);
      paintHero();
    }

    var body = document.createElement("div");
    body.className = "mcard-body";

    var bodyControls = [];
    (options.controls || []).forEach(function (spec) {
      if (modeSpec && spec.id === modeSpec.id) return;
      if ((options.header || []).indexOf(spec.id) >= 0) return;
      var el = makeBodyControl(spec, values[spec.id], edited);
      body.appendChild(el);
      bodyControls.push({ spec: spec, el: el });
    });

    card.appendChild(body);

    function applyVisibility() {
      if (!options.visible) return;
      var allowed = null;
      for (var i = 0; i < options.visible.length; i++) {
        if (Number(options.visible[i][0]) === currentMode) {
          allowed = options.visible[i][1];
        }
      }
      bodyControls.forEach(function (entry) {
        var show = !allowed || allowed.indexOf(entry.spec.id) >= 0;
        entry.el.style.display = show ? "" : "none";
      });
    }

    applyVisibility();

    return card;
  }

  window.IAmCard = {
    create: create,
    control: control,
    controlHandle: controlHandle,
    switchHandle: makeSwitch,
    choiceHandle: makeChoiceHandle,
    sliderHandle: makeSliderHandle,
    hero: hero,
    registerHero: registerHero,
    format: format,
    norm: norm,
    at: at
  };
})();
