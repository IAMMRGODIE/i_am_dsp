// The filter table's math: the curves, the FFT and the response, and the picture they draw.
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
    // The frame's own resolution. An FFT at a longer size means upsampling the
    // kernel first, which rolls the top of the band off: the curve derived here
    // then disagrees with the one that was drawn, and toggling the view moved it.
    var size = Math.max(64, points.length);
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
  ///
  /// The kernel is built at the size the page holds. A longer transform decimated
  /// back to the page aliases: the high bands folded onto the low ones, so the
  /// curve drawn here came back different the next time the page was read.
  function filterTimeFromResponse(response) {
    var count = 256;
    var size = count;
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

  /// One trace of a scope buffer, as a path in the 520 by 160 hero space.
  function ringTracePath(samples, middle, height) {
    if (!samples || samples.length < 64) return "";

    var last = samples.length - 1;
    var stride = Math.max(1, Math.floor(last / 400));
    var path = "";

    for (var index = 0; index <= last; index += stride) {
      var x = 20 + index / last * 480;
      var y = middle - clamp(samples[index], -1.2, 1.2) / 1.2 * height;
      path += (path ? "L" : "M") + x.toFixed(1) + " " + y.toFixed(1);
    }

    return path;
  }

  /// The ring modulator: the carrier it multiplies with, or, in the sidechain
  /// mode, the signal before and after that multiplication.
  function ringHero(values, context) {
    var shape = Math.round(valueOf(values, "shape", 0));
    var mode = Math.round(valueOf(values, "mode", 0));
    var stereo = valueOf(values, "stereo", 0);
    var shapes = ["SINE", "TRIANGLE", "SAW", "SQUARE", "NOISE"];
    var modes = ["RING", "AM", "SIDECHAIN"];

    if (mode === 2) {
      var scope = context && context.scope;
      var inputPath = ringTracePath(scope && scope.input, 80, 60);
      var outputPath = ringTracePath(scope && scope.output, 80, 60);

      return '<svg class="mcard-graph" viewBox="0 0 520 160" preserveAspectRatio="xMidYMid meet">' +
        '<line x1="20" y1="80" x2="500" y2="80" stroke="#333" stroke-width="1" stroke-dasharray="2 4"/>' +
        (inputPath ? '<path d="' + inputPath + '" fill="none" stroke="#666" stroke-width="1.5" opacity="0.7" vector-effect="non-scaling-stroke"/>' : '') +
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
