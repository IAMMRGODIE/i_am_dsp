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

  /// The scale of the smooth logarithmic map used when a logarithmic range
  /// reaches zero or goes negative. It is 1% of the larger endpoint, so the
  /// middle of a bipolar knob spends more travel near zero without making the
  /// ends feel linear.
  function symlogScale(spec) {
    return Math.max(Math.abs(spec.min), Math.abs(spec.max), 1e-6) / 100;
  }

  function symlogNorm(spec, value) {
    var scale = symlogScale(spec);
    var low = Math.asinh(spec.min / scale);
    var high = Math.asinh(spec.max / scale);
    return high > low ? clamp((Math.asinh(value / scale) - low) / (high - low), 0, 1) : 0;
  }

  function symlogAt(spec, position) {
    var scale = symlogScale(spec);
    var low = Math.asinh(spec.min / scale);
    var high = Math.asinh(spec.max / scale);
    return high > low ? Math.sinh(low + clamp(position, 0, 1) * (high - low)) * scale : spec.min;
  }

  function norm(spec, value) {
    var low = spec.min;
    var high = spec.max;

    if (spec.log && high > low) {
      return low > 0
        ? clamp(Math.log(value / low) / Math.log(high / low), 0, 1)
        : symlogNorm(spec, value);
    }

    return high > low ? clamp((value - low) / (high - low), 0, 1) : 0;
  }

  /// One notch of a control. An explicit step keeps its own units (semitones,
  /// kernel lengths), an integer counts by one, and everything else moves a
  /// fraction of its *normalised* range: a log or symmetric-log knob then takes
  /// a constant ratio per notch instead of a constant number of hertz.
  function stepBy(spec, value, direction, fine) {
    var scale = fine ? 0.1 : 1;

    if (spec.step && spec.step > 0) {
      return value + direction * spec.step * scale;
    }

    if (spec.kind === "int") {
      return value + direction;
    }

    // A log knob spans a much wider range than its pixel width suggests, so it
    // takes a finer share of that range per notch.
    var amount = spec.log ? 0.005 : 0.0125;
    var position = norm(spec, value) + direction * amount * scale;

    return at(spec, clamp(position, 0, 1));
  }

  function at(spec, position) {
    var low = spec.min;
    var high = spec.max;
    position = clamp(position, 0, 1);

    if (spec.log && high > low) {
      return low > 0
        ? low * Math.pow(high / low, position)
        : symlogAt(spec, position);
    }

    return low + (high - low) * position;
  }

  /// A control can override its readout without changing its unit metadata.
  function display(spec, value) {
    return typeof spec.format === "function" ? spec.format(value) : format(spec, value);
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

    // Switches are booleans in the faceplate's control map, while hero code
    // reads them as 0/1. Accepting both also fixes the rack's own heroes.
    if (typeof value === "boolean") return value ? 1 : 0;
    return typeof value === "number" ? value : fallback;
  }

  function heroSvg(inner) {
    return '<svg class="mcard-graph" viewBox="0 0 520 160" preserveAspectRatio="none">' +
      inner +
      '</svg>';
  }

  function heroSvgMeter(inner) {
    return '<svg class="mcard-graph" viewBox="0 0 320 200" preserveAspectRatio="xMidYMid meet">' +
      inner +
      '</svg>';
  }

  var BANDS = 96;

  function fft(real, imag) {
    var n = real.length;

    for (var i = 1, j = 0; i < n; i++) {
      var bit = n >> 1;
      for (; j & bit; bit >>= 1) j ^= bit;
      j ^= bit;

      if (i < j) {
        var tr = real[i]; real[i] = real[j]; real[j] = tr;
        var ti = imag[i]; imag[i] = imag[j]; imag[j] = ti;
      }
    }

    for (var length = 2; length <= n; length <<= 1) {
      var angle = -2 * Math.PI / length;
      var stepR = Math.cos(angle);
      var stepI = Math.sin(angle);
      var half = length >> 1;

      for (var start = 0; start < n; start += length) {
        var wr = 1, wi = 0;

        for (var k = 0; k < half; k++) {
          var ar = real[start + k], ai = imag[start + k];
          var br = real[start + k + half], bi = imag[start + k + half];
          var vr = br * wr - bi * wi;
          var vi = br * wi + bi * wr;

          real[start + k] = ar + vr;
          imag[start + k] = ai + vi;
          real[start + k + half] = ar - vr;
          imag[start + k + half] = ai - vi;

          var nextR = wr * stepR - wi * stepI;
          wi = wr * stepI + wi * stepR;
          wr = nextR;
        }
      }
    }
  }

  /// Log-spaced bands in dBFS, so a spectrum can share the response's x axis.
  function bandsOf(samples, rate) {
    var n = samples.length;
    if (!n || (n & (n - 1)) !== 0) return null;

    var fftLength = n * 4;
    var real = new Float32Array(fftLength);
    var imag = new Float32Array(fftLength);

    for (var i = 0; i < n; i++) {
      var window = 0.5 - 0.5 * Math.cos(2 * Math.PI * i / (n - 1));
      real[i] = samples[i] * window;
    }

    fft(real, imag);

    var raw = new Float32Array(BANDS);
    var filled = new Uint8Array(BANDS);

    for (var k = 1; k < fftLength / 2; k++) {
      var frequency = k * rate / fftLength;
      if (frequency < 20 || frequency > 20000) continue;

      var magnitude = 4 * Math.sqrt(real[k] * real[k] + imag[k] * imag[k]) / n;
      var band = Math.floor(Math.log(frequency / 20) / Math.log(1000) * BANDS);

      if (band < 0) band = 0;
      if (band >= BANDS) band = BANDS - 1;
      if (magnitude > raw[band]) {
        raw[band] = magnitude;
        filled[band] = 1;
      }
    }

    var first = 0;
    while (first < BANDS && !filled[first]) first++;
    if (first < BANDS) {
      for (var b = 0; b < first; b++) raw[b] = raw[first];
      var carry = raw[first];
      for (var b = first; b < BANDS; b++) {
        if (filled[b]) carry = raw[b];
        else raw[b] = carry;
      }
    }

    for (var b = 0; b < BANDS; b++) {
      raw[b] = 20 * Math.log10(Math.max(raw[b], 1e-7));
    }

    var smooth = new Float32Array(BANDS);
    var weights = [1, 2, 1];

    for (var c = 0; c < BANDS; c++) {
      var sum = 0;
      var weight = 0;

      for (var w = -1; w <= 1; w++) {
        var index = clamp(c + w, 0, BANDS - 1);
        var factor = weights[w + 1];
        sum += raw[index] * factor;
        weight += factor;
      }

      smooth[c] = Math.max(raw[c], sum / weight);
    }

    return smooth;
  }

  function smoothBands(previous, next) {
    if (!previous || previous.length !== next.length) return next;

    for (var i = 0; i < next.length; i++) {
      previous[i] += (next[i] - previous[i]) * 0.28;
    }

    return previous;
  }

  function bandsPath(bands) {
    var path = "";

    for (var i = 0; i < bands.length; i++) {
      var x = 20 + i / (BANDS - 1) * 480;
      var y = clamp(10 + clamp(-bands[i] / 60, 0, 1) * 140, 10, 150);
      path += (i ? " L " : "M ") + x.toFixed(1) + " " + y.toFixed(1);
    }

    return path;
  }

  function correlation(left, right) {
    if (!left || !right || left.length < 2) return 0;

    var n = Math.min(left.length, right.length);
    var sumLeft = 0;
    var sumRight = 0;

    for (var i = 0; i < n; i++) {
      sumLeft += left[i];
      sumRight += right[i];
    }

    var meanLeft = sumLeft / n;
    var meanRight = sumRight / n;
    var covariance = 0;
    var leftEnergy = 0;
    var rightEnergy = 0;

    for (var j = 0; j < n; j++) {
      var dl = left[j] - meanLeft;
      var dr = right[j] - meanRight;
      covariance += dl * dr;
      leftEnergy += dl * dl;
      rightEnergy += dr * dr;
    }

    var denominator = Math.sqrt(leftEnergy * rightEnergy);
    return denominator > 1e-12
      ? Math.max(-1, Math.min(1, covariance / denominator))
      : 0;
  }

  /// A label that lives in HTML instead of inside the picture. The pictures are
  /// stretched to the card (preserveAspectRatio="none"), and text drawn inside
  /// one stretches with it; this keeps its own proportions.
  function heroLabel(x, y, text, style) {
    return '<div class="hero-label" style="left:' + (x / 5.2).toFixed(2) +
      '%;top:' + (y / 1.6).toFixed(2) + '%;' + (style || "") + '">' + text + '</div>';
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

  function spectrumHero(values, context) {
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

    var inputPath = "";
    var outputPath = "";
    var scope = context && context.scope;
    var spectraState = context && context.spectra;
    var rate = (context && context.rate) || 48000;

    if (scope && scope.input && scope.output && spectraState) {
      var inputBands = bandsOf(scope.input, rate);
      var outputBands = bandsOf(scope.output, rate);

      if (inputBands && outputBands) {
        spectraState.input = smoothBands(spectraState.input, inputBands);
        spectraState.output = smoothBands(spectraState.output, outputBands);
        inputPath = bandsPath(spectraState.input);
        outputPath = bandsPath(spectraState.output);
      }
    }

    return heroSvg(
      (inputPath
        ? '<path d="' + inputPath + '" fill="none" stroke="#8a8a8a" stroke-width="1.2" ' +
          'opacity=".7" vector-effect="non-scaling-stroke"/>'
        : "") +
      (outputPath
        ? '<path d="' + outputPath + ' L 500 150 L 20 150 Z" fill="var(--accent)" ' +
            'fill-opacity=".06"/>' +
          '<path d="' + outputPath + '" fill="none" stroke="var(--accent)" stroke-width="1.5" ' +
            'opacity=".9" vector-effect="non-scaling-stroke"/>'
        : "") +
      '<path d="' + points.join(" ") + '" fill="none" stroke="var(--accent)" ' +
        'stroke-width="2" opacity=".95" vector-effect="non-scaling-stroke"/>'
    );
  }

  function spectraHero(values, context) {
    var scope = context && context.scope;
    var spectraState = context && context.spectra;

    if (!scope || !scope.input || !scope.output || !spectraState) {
      return heroSvg(
        '<line x1="20" y1="140" x2="500" y2="140" stroke="#2c2c2c" stroke-width="1"/>'
      );
    }

    var rate = (context && context.rate) || 48000;
    var before = valueOf(values, "sidechain", 0) && scope.sidechain ? scope.sidechain : scope.input;
    var inputBands = bandsOf(before, rate);
    var outputBands = bandsOf(scope.output, rate);

    if (!inputBands || !outputBands) return "";

    spectraState.input = smoothBands(spectraState.input, inputBands);
    spectraState.output = smoothBands(spectraState.output, outputBands);

    return heroSvg(
      '<path d="' + bandsPath(spectraState.input) + '" fill="none" stroke="#8a8a8a" ' +
        'stroke-width="1.2" opacity=".7" vector-effect="non-scaling-stroke"/>' +
      '<path d="' + bandsPath(spectraState.output) + '" fill="none" stroke="var(--accent)" ' +
        'stroke-width="1.8" vector-effect="non-scaling-stroke"/>'
    );
  }

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
      var in_height = Math.sqrt(clamp(input[i] || 0, 0, 1)) * 60;
      var out_height = Math.sqrt(clamp(output[i] || 0, 0, 1)) * 60;

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
      var tx = 20 + Math.log(ticks[t] / 20) / Math.log(1000) * 480;
      grid += '<line x1="' + tx.toFixed(1) + '" y1="10" x2="' + tx.toFixed(1) +
        '" y2="150" stroke="#252525"/>';

      if (labels[ticks[t]]) {
        labelsHtml += heroLabel(tx, 157, labels[ticks[t]],
          "transform:translate(-50%,-100%)");
      }
    }

    for (var db = -24; db <= 24; db += 12) {
      var gy = 80 - db / 24 * 70;
      grid += '<line x1="20" y1="' + gy.toFixed(1) + '" x2="500" y2="' + gy.toFixed(1) +
        '" stroke="' + (db === 0 ? "#363636" : "#252525") + '"/>';
      labelsHtml += heroLabel(22, gy - 3, (db > 0 ? "+" : "") + db);
    }

    grid += '<rect x="20" y="10" width="480" height="140" fill="none" stroke="#2a2a2a"/>';

    var curve = "";

    for (var i = 0; i <= 240; i++) {
      var frequency = 20 * Math.pow(1000, i / 240);
      var decibels = eqResponseDb(frequency, bands, rate) + output;
      var x = 20 + i / 240 * 480;
      var y = clamp(80 - decibels / 24 * 70, 10, 150);
      curve += (i ? " L " : "M ") + x.toFixed(1) + " " + y.toFixed(1);
    }

    var spectrum = "";
    var scope = context && context.scope;
    var spectra = context && context.spectra;
    var eqSpectrum = context && context.eqSpectrum;

    if (scope && scope.input && scope.output && spectra && eqSpectrum) {
      if (eqSpectrum.input !== scope.input || eqSpectrum.output !== scope.output ||
          eqSpectrum.rate !== rate) {
        var inputBands = bandsOf(scope.input, rate);
        var outputBands = bandsOf(scope.output, rate);

        if (inputBands && outputBands) {
          spectra.input = smoothBands(spectra.input, inputBands);
          spectra.output = smoothBands(spectra.output, outputBands);
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
      var bx = 20 + Math.log(clamp(band.freq, 20, 20000) / 20) / Math.log(1000) * 480;
      var by = 80 - (band.type <= 2 ? band.gain : 0) / 24 * 70;
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
      'stroke-width="2.2" vector-effect="non-scaling-stroke"/>' +
      '<text x="' + (lawX + 4).toFixed(1) + '" y="22" fill="#777" font-size="9">' +
      law.toFixed(0) + ' dB</text>' +
      '<text x="462" y="140" fill="#777" font-size="9">' +
      (mode === "limiter" ? "0 dBFS" : ratio.toFixed(1) + ":1") + '</text>'
    );
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

    centers.forEach(function (center, band) {
      var x = 20 + Math.log(clamp(center, 20, 20000) / 20) / Math.log(1000) * 480;
      markers += '<line x1="' + x.toFixed(1) + '" y1="12" x2="' + x.toFixed(1) +
        '" y2="146" stroke="var(--accent)" stroke-opacity=".25" stroke-dasharray="2 4"/>' +
        '<text x="' + x.toFixed(1) + '" y="' + (14 + band * 12) +
        '" fill="var(--accent)" font-size="10" text-anchor="middle">F' + (band + 1) + '</text>';
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

  function registerHero(kind, renderer) {
    HEROES[kind] = renderer;
  }

  function hero(kind, values, context) {
    var renderer = HEROES[kind];
    return renderer ? renderer(values, context) : "";
  }

  registerHero("spectrum", spectrumHero);
  registerHero("spectra", spectraHero);
  registerHero("transfer", transferHero);
  registerHero("compressor", compressorHero);
  registerHero("impulse", impulseHero);
  registerHero("pitch", pitchHero);
  registerHero("particles", particlesHero);
  registerHero("curve", curveHero);
  registerHero("mapping", mappingHero);
  registerHero("ott", ottHero);
  registerHero("vocodex", vocoderHero);
  registerHero("eq", eqHero);
  registerHero("gate", function (values, context) { return dynamicsHero(values, "gate", context); });
  registerHero("expander", function (values, context) { return dynamicsHero(values, "expander", context); });
  registerHero("limiter", function (values, context) { return dynamicsHero(values, "limiter", context); });
  registerHero("stereo", stereoHero);
  registerHero("tail", tailHero);
  registerHero("taps", tapsHero);
  registerHero("lfo", lfoHero);
  registerHero("phase", phaseHero);
  registerHero("balance", balanceHero);
  registerHero("waveform", waveformHero);
  registerHero("resonator", resonatorHero);
  registerHero("formant", formantHero);
  registerHero("autotune", autotuneHero);
  registerHero("partials", partialsHero);
  registerHero("string", stringHero);
  registerHero("vibrato", vibratoHero);
  registerHero("shape", shapeHero);
  registerHero("envelope", envelopeHero);
  registerHero("additive", additiveHero);

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

    var mods = svg("g", { class: "knob-mods" });
    var fill = svg("path", { class: "knob-arc" });
    var hand = svg("line", { class: "knob-hand" });
    var live = svg("line", { class: "knob-live" });
    live.style.display = "none";
    drawing.appendChild(mods);
    drawing.appendChild(fill);
    drawing.appendChild(hand);
    drawing.appendChild(live);
    dial.appendChild(drawing);

    if (modulations.some(function (assignment) { return assignment.selected; })) {
      dial.classList.add("mod-selected");
    }

    // A thin needle for where the modulation is right now, so the knob shows
    // the sound while its main hand keeps the value a person set.
    /// The needle is an offset from the hand, so turning the knob carries it
    /// along and the modulation keeps its distance.
    var liveOffset = null;

    wrap.__live = function (offset) {
      liveOffset = offset;
      paintLive();
    };

    function paintLive() {
      if (liveOffset === null) {
        live.style.display = "none";
        return;
      }

      var position = clamp(norm(spec, current) + liveOffset, 0, 1);
      var angle = START + SWEEP * position;
      var inner = polar(32, 32, 18, angle);
      var tip = polar(32, 32, 29, angle);
      live.setAttribute("x1", inner.x.toFixed(2));
      live.setAttribute("y1", inner.y.toFixed(2));
      live.setAttribute("x2", tip.x.toFixed(2));
      live.setAttribute("y2", tip.y.toFixed(2));
      live.style.display = "";
    }

    // The depth a source is adding while it is dragged with Alt.
    var modHint = document.createElement("span");
    modHint.className = "knob-mod-hint";
    modHint.style.display = "none";
    wrap.appendChild(modHint);

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

      // One arc per source, in that source's colour: a knob driven by two of
      // them shows two arcs instead of one merged span nobody can attribute.
      while (mods.firstChild) {
        mods.removeChild(mods.firstChild);
      }

      modulations.forEach(function (assignment) {
        var depth = assignment.depth || 0;
        var mode = assignment.mode || "centered";
        var low = position;
        var high = position;

        if (mode === "frombase" || mode === "from-base" || mode === "start") {
          var end = clamp(position + depth, 0, 1);
          low = Math.min(low, end);
          high = Math.max(high, end);
        } else {
          var span = Math.abs(depth);
          low = Math.min(low, clamp(position - span, 0, 1));
          high = Math.max(high, clamp(position + span, 0, 1));
        }

        if (high - low <= 0.0001) return;

        var arc = svg("path", {
          class: "knob-mod",
          d: arcPath(32, 32, 29, START + SWEEP * low, START + SWEEP * high)
        });

        if (assignment.color) arc.style.stroke = assignment.color;
        mods.appendChild(arc);
      });

      if (modulations.length) {
        dial.title = modulations
          .map(function (assignment) {
            var depth = assignment.depth || 0;
            return (assignment.label || "source") + "  " +
              (depth >= 0 ? "+" : "") + Math.round(depth * 100) + "%";
          })
          .join("\n");
      }
      var inner = polar(32, 32, 9, angle);
      var tip = polar(32, 32, 25, angle);
      hand.setAttribute("x1", inner.x.toFixed(2));
      hand.setAttribute("y1", inner.y.toFixed(2));
      hand.setAttribute("x2", tip.x.toFixed(2));
      hand.setAttribute("y2", tip.y.toFixed(2));
      readout.textContent = display(spec, current);
      dial.setAttribute("aria-valuenow", String(current));
      dial.setAttribute("aria-valuetext", display(spec, current));
      paintLive();
    }

    function set(value, send) {
      current = clamp(value, spec.min, spec.max);
      if (spec.kind === "int") current = Math.round(current);
      // A stepped parameter only takes its own intervals: a kernel length that
      // has to be odd, a semitone, an eighth note.
      if (spec.step > 0) {
        current = clamp(
          spec.min + Math.round((current - spec.min) / spec.step) * spec.step,
          spec.min,
          spec.max
        );
        if (spec.kind === "int") current = Math.round(current);
      }
      paint();
      if (send && onEdit) onEdit(spec.id, current);
    }

    var drag = null;
    var modDrag = null;

    dial.addEventListener("pointerdown", function (event) {
      if (event.button !== 0) return;

      if (event.altKey && modulations.length) {
        var target = modulations[0];
        modDrag = {
          y: event.clientY,
          depth: target.depth || 0,
          name: target.label || "source"
        };
        dial.classList.add("mod-drag");
        modHint.textContent = modDrag.name + "  " + Math.round(modDrag.depth * 100) + "%";
        modHint.style.display = "";
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
        modHint.textContent = modDrag.name + "  " +
          (next >= 0 ? "+" : "") + Math.round(next * 100) + "%";
        modHint.style.display = "";

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
      modHint.style.display = "none";
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
        // The knob stops the drop: without this the slot underneath would read
        // the source as a module and try to move it.
        event.stopPropagation();
        dial.classList.remove("drop-target");
        var data = event.dataTransfer ? event.dataTransfer.getData("text/plain") : "";
        if (data && spec.onModulationDrop) spec.onModulationDrop(data, spec.dropTarget);
      });
    }

    // Alt + right click edits how far each source pushes this knob.
    dial.addEventListener("contextmenu", function (event) {
      if (!event.altKey || !modulations.length) return;

      event.preventDefault();
      openDepths();
    });

    function openDepths() {
      var box = document.createElement("div");
      box.className = "knob-depths";

      modulations.forEach(function (assignment) {
        var row = document.createElement("label");
        row.className = "knob-depth";

        var name = document.createElement("span");
        name.textContent = assignment.label || "source";

        var range = document.createElement("input");
        range.type = "range";
        range.min = "-100";
        range.max = "100";
        range.step = "1";
        range.value = String(Math.round((assignment.depth || 0) * 100));

        var value = document.createElement("span");
        value.className = "knob-depth-value";
        value.textContent = range.value + "%";

        range.addEventListener("input", function () {
          var depth = Number(range.value) / 100;
          assignment.depth = depth;
          value.textContent = range.value + "%";
          paint();

          if (spec.onModulation) spec.onModulation(assignment.slot, depth);
        });

        row.appendChild(name);
        row.appendChild(range);
        row.appendChild(value);
        box.appendChild(row);
      });

      var rect = dial.getBoundingClientRect();
      box.style.left = Math.max(8, Math.min(window.innerWidth - 240, rect.left)) + "px";
      box.style.top = Math.min(window.innerHeight - 120, rect.bottom + 8) + "px";
      document.body.appendChild(box);

      function dismiss(event) {
        if (box.contains(event.target)) {
          document.addEventListener("pointerdown", dismiss, { once: true, capture: true });
          return;
        }

        box.remove();
      }

      setTimeout(function () {
        document.addEventListener("pointerdown", dismiss, { once: true, capture: true });
      }, 0);
    }

    dial.addEventListener("wheel", function (event) {
      event.preventDefault();
      var direction = event.deltaY < 0 ? 1 : -1;
      set(stepBy(spec, current, direction, event.shiftKey), true);
    }, { passive: false });

    dial.addEventListener("keydown", function (event) {
      var fine = event.shiftKey;
      if (event.key === "ArrowUp" || event.key === "ArrowRight") {
        set(stepBy(spec, current, 1, fine), true);
        event.preventDefault();
      } else if (event.key === "ArrowDown" || event.key === "ArrowLeft") {
        set(stepBy(spec, current, -1, fine), true);
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
      readout.textContent = display(spec, current);
      track.setAttribute("aria-valuenow", String(current));
      track.setAttribute("aria-valuetext", display(spec, current));
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
      set(stepBy(spec, current, event.deltaY < 0 ? 1 : -1, event.shiftKey), true);
    }, { passive: false });
    track.addEventListener("keydown", function (event) {
      var fine = event.shiftKey;
      if (event.key === "ArrowUp" || event.key === "ArrowRight") {
        set(stepBy(spec, current, 1, fine), true);
        event.preventDefault();
      } else if (event.key === "ArrowDown" || event.key === "ArrowLeft") {
        set(stepBy(spec, current, -1, fine), true);
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

  function makeNotePickerHandle(spec, value, onEdit) {
    var labels = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"];
    var mask = Math.round(Number(value) || 0) & 4095;
    var wrap = document.createElement("div");
    wrap.className = "mcard-notes";
    wrap.setAttribute("role", "group");
    wrap.setAttribute("aria-label", spec.label);
    var buttons = [];

    function paint() {
      buttons.forEach(function (button, index) {
        var on = (mask & (1 << index)) !== 0;
        button.classList.toggle("on", on);
        button.setAttribute("aria-checked", on ? "true" : "false");
      });
    }

    function set(next) {
      mask = Math.round(Number(next) || 0) & 4095;
      paint();
    }

    labels.forEach(function (label, index) {
      var button = document.createElement("button");
      button.type = "button";
      button.className = "mcard-note" + (label.length > 1 ? " sharp" : "");
      button.textContent = label;
      button.setAttribute("role", "checkbox");
      button.setAttribute("aria-label", label);
      button.addEventListener("click", function () {
        mask ^= 1 << index;
        paint();
        if (onEdit) onEdit(spec.id, mask);
      });
      buttons.push(button);
      wrap.appendChild(button);
    });

    paint();
    return { el: wrap, set: set };
  }

  function makeNotePicker(spec, value, onEdit) {
    return makeNotePickerHandle(spec, value, onEdit).el;
  }

  function makeBodyControl(spec, value, onEdit) {
    if (spec.widget === "notes") {
      return makeNotePicker(spec, value, onEdit);
    }

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

    // The unmodulated values, kept apart: a live update must not become the next
    // tick's base, or the needle would drift back onto the hand.
    var baseValues = {};
    Object.keys(values).forEach(function (id) { baseValues[id] = values[id]; });

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

    // A collapsed card is a strip: clicking it should open it again, not only
    // clicking the arrow does.
    card.addEventListener("click", function (event) {
      if (!card.classList.contains("collapsed")) return;
      if (event.target.closest("button, select, input, textarea, a")) return;

      collapse.click();
    });

    // A scripted product carries its source in a small editor under the head.
    var scriptArea = null;
    var scriptError = null;

    if (options.script) {
      var scriptToggle = document.createElement("button");
      scriptToggle.type = "button";
      scriptToggle.className = "mcard-script-toggle";
      scriptToggle.textContent = "{ }";
      scriptToggle.title = "Edit the script";
      scriptToggle.setAttribute("aria-label", "Edit the script");
      head.appendChild(scriptToggle);

      var panel = document.createElement("div");
      panel.className = "mcard-script";
      panel.style.display = "none";

      scriptArea = document.createElement("textarea");
      scriptArea.className = "mcard-script-text";
      scriptArea.spellcheck = false;
      panel.appendChild(scriptArea);

      scriptError = document.createElement("div");
      scriptError.className = "mcard-script-error";
      scriptError.style.display = "none";
      panel.appendChild(scriptError);

      var actions = document.createElement("div");
      actions.className = "mcard-script-actions";

      var apply = document.createElement("button");
      apply.type = "button";
      apply.className = "mcard-script-apply";
      apply.textContent = "Apply";
      apply.addEventListener("click", function () {
        if (options.applyScript) options.applyScript(scriptArea.value);
      });
      actions.appendChild(apply);
      panel.appendChild(actions);
      card.appendChild(panel);

      scriptToggle.addEventListener("click", function () {
        var open = panel.style.display === "none";
        panel.style.display = open ? "" : "none";
        if (open && options.loadScript) options.loadScript();
      });
    }

    // A card for a module that follows the host's notes shows the note it is
    // hearing; the page pushes it in from /module-note.
    var noteBadge = null;

    if (options.note) {
      noteBadge = document.createElement("span");
      noteBadge.className = "mcard-note-badge";
      noteBadge.style.display = "none";
      head.appendChild(noteBadge);
    }

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

      var groups = Object.create(null);

      options.kinds.forEach(function (entry) {
        var name = typeof entry === "string" ? entry : entry.name;
        var group = typeof entry === "string" ? "" : entry.group || "";
        var option = document.createElement("option");
        option.value = name;
        option.textContent = name;

        if (!group) {
          kindSelect.appendChild(option);
          return;
        }

        if (!groups[group]) {
          groups[group] = document.createElement("optgroup");
          groups[group].label = group;
          kindSelect.appendChild(groups[group]);
        }

        groups[group].appendChild(option);
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
      if (heroElement) heroElement.innerHTML = hero(options.hero, values, options.context);
    }

    function edited(id, value) {
      values[id] = value;
      baseValues[id] = value;
      paintHero();
      // A switch appears in the header, so the body has to follow it.
      applyVisibility();
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

      // A product whose picture is also its file picker (the convolver) makes
      // the hero clickable; the page that owns the card does the asking.
      if (options.ir) {
        heroElement.classList.add("ir-target");
        heroElement.title = "Click to load an impulse response";
        heroElement.addEventListener("click", function () {
          if (options.pickIr) options.pickIr();
        });
      }
    }

    var body = document.createElement("div");
    body.className = "mcard-body";

    var bodyControls = [];
    var liveKnobs = Object.create(null);

    (options.controls || []).forEach(function (spec) {
      if (modeSpec && spec.id === modeSpec.id) return;
      if ((options.header || []).indexOf(spec.id) >= 0) return;
      var el = makeBodyControl(spec, values[spec.id], edited);

      if (el && el.__live) liveKnobs[spec.id] = el.__live;

      body.appendChild(el);
      bodyControls.push({ spec: spec, el: el });
    });

    card.appendChild(body);

    function applyVisibility() {
      if (!bodyControls) return;

      // A switch trades one branch of a pair for the other: the delay's free
      // time gives way to its beats, the chorus's hertz to its beats.
      var hidden = Object.create(null);
      (options.switches || []).forEach(function (row) {
        var on = Number(values[row.control] || 0) !== 0;
        (on ? row.off : row.on).forEach(function (id) {
          hidden[id] = true;
        });
      });

      var allowed = null;
      if (options.visible) {
        for (var i = 0; i < options.visible.length; i++) {
          if (Number(options.visible[i][0]) === currentMode) {
            allowed = options.visible[i][1];
          }
        }
      }

      bodyControls.forEach(function (entry) {
        var show = (!allowed || allowed.indexOf(entry.spec.id) >= 0) && !hidden[entry.spec.id];
        entry.el.style.display = show ? "" : "none";
      });
    }

    applyVisibility();

    // A live hero has to be redrawn when the audio moves, not only when a knob
    // does: the page keeps the card and calls this on every scope frame.
    card.repaintHero = paintHero;

    card.setLiveValues = function (live) {
      var changed = false;

      Object.keys(live).forEach(function (id) {
        var value = live[id];
        var spec = byId[id];

        if (typeof value !== "number") return;

        if (liveKnobs[id]) {
          var offset = spec && typeof baseValues[id] === "number"
            ? norm(spec, value) - norm(spec, baseValues[id])
            : null;

          liveKnobs[id](offset === null ? null : clamp(offset, -1, 1));
        }

        if (values[id] === value) return;

        values[id] = value;
        changed = true;
      });

      // The hero reads the same map the knobs' base does, so it follows the
      // source without the page having to know what the picture means.
      if (changed) paintHero();
    };

    card.setScript = function (source, force) {
      if (!scriptArea || source === undefined || source === null) return;
      // A poll must not overwrite what the person is typing.
      if (!force && document.activeElement === scriptArea) return;

      scriptArea.value = source;
    };

    card.setScriptError = function (message) {
      if (!scriptError) return;

      if (!message) {
        scriptError.style.display = "none";
        scriptError.textContent = "";
        return;
      }

      scriptError.textContent = message;
      scriptError.style.display = "";
    };

    card.setNote = function (notes) {
      if (!noteBadge) return;

      var list = (notes || []).filter(function (note) {
        return typeof note === "number" && note >= 0;
      });

      if (!list.length) {
        noteBadge.style.display = "none";
        return;
      }

      var names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"];
      noteBadge.textContent = list
        .slice()
        .sort(function (a, b) { return a - b; })
        .map(function (note) {
          return names[note % 12] + (Math.floor(note / 12) - 1);
        })
        .join(" ");
      noteBadge.style.display = "";
    };

    return card;
  }

  window.IAmCard = {
    create: create,
    control: control,
    controlHandle: controlHandle,
    switchHandle: makeSwitch,
    notePickerHandle: makeNotePickerHandle,
    choiceHandle: makeChoiceHandle,
    sliderHandle: makeSliderHandle,
    hero: hero,
    registerHero: registerHero,
    bandsOf: bandsOf,
    smoothBands: smoothBands,
    bandsPath: bandsPath,
    format: format,
    norm: norm,
    at: at
  };
})();
