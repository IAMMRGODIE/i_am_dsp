// Part two: the FFT and the band helpers the scope pictures draw from.
  var BANDS = 96;

  /// The band below which the picture is read from the whole frame instead.
  ///
  /// Fifty hertz is two cycles of a forty-three millisecond window, and a
  /// kick drum lives there: the low bands are read from one transform of the
  /// whole frame, which is four times the resolution, while everything above
  /// keeps the short windows a transient needs.
  var LOW_KNEE = 150;

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
  ///
  /// The frame is analysed in halves and the louder half wins each band. A
  /// drum hit lasts twenty milliseconds and the frame holds eighty-five, so
  /// one transform over the whole of it buries the hit in the silence around
  /// it — which is why a percussion track seemed to have no spectrum at all.
  /// Splitting the frame costs one more transform and keeps the hit whole.
  function bandsOf(samples, rate) {
    var n = samples.length;
    if (!n || (n & (n - 1)) !== 0) return null;

    var raw = new Float32Array(BANDS);
    var filled = new Uint8Array(BANDS);

    /// One window of the frame, contributing its bands up to `top`.
    ///
    /// `replace` makes this window the answer for those bands rather than a
    /// candidate: the low end is read from the long window alone, because a
    /// short one does not have the resolution to keep a thirty hertz tone out
    /// of the band an octave above it.
    function add(offset, length, padding, top, replace) {
      var fftLength = length * padding;
      // The transform reads its length off the array, so every window needs
      // its own of exactly the size it asks for.
      var real = new Float32Array(fftLength);
      var imag = new Float32Array(fftLength);

      for (var i = 0; i < length; i++) {
        var weight = 0.5 - 0.5 * Math.cos(2 * Math.PI * i / (length - 1));
        real[i] = samples[offset + i] * weight;
      }

      fft(real, imag);

      for (var k = 1; k < fftLength / 2; k++) {
        var frequency = k * rate / fftLength;
        if (frequency < 20 || frequency > top) continue;

        var magnitude = 4 * Math.sqrt(real[k] * real[k] + imag[k] * imag[k]) / length;
        var band = Math.floor(Math.log(frequency / 20) / Math.log(1000) * BANDS);

        if (band < 0) band = 0;
        if (band >= BANDS) band = BANDS - 1;
        if (replace || magnitude > raw[band]) {
          raw[band] = magnitude;
          filled[band] = 1;
        }
      }
    }

    // Halves that overlap by half, so a hit that lands on a join is still
    // whole inside one of them. Three transforms of a frame the page reads
    // twenty times a second is nothing next to the audio thread's work.
    var window = n / 2;
    var hop = window / 2;
    for (var start = 0; start + window <= n; start += hop) {
      add(start, window, 4, 20000);
    }

    // Nothing a short window can say about the bottom of the band, so the
    // whole frame says it instead, and says it alone: the low bands the short
    // windows filled are cleared first, or their leakage would survive the
    // carry below. Padded by two, so a tone that falls between bins is not
    // read low.
    var knee = Math.floor(Math.log(LOW_KNEE / 20) / Math.log(1000) * BANDS);

    for (var low = 0; low < knee; low++) {
      raw[low] = 0;
      filled[low] = 0;
    }

    add(0, n, 2, LOW_KNEE, true);

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

  /// What the picture shows: rising at once, falling at a fixed rate.
  ///
  /// A symmetric one-pole can show a pad and cannot show a drum. The page
  /// draws twenty frames a second, so a hit is one frame of signal followed
  /// by the silence after it, and a pole averages the two: the peak arrives
  /// at a quarter of its height, several frames late, and the next hit lands
  /// on top of it. An analyser rises on the hit and lets go slowly instead,
  /// which is the whole reason a percussion track is readable on one.
  function smoothBands(previous, next, ms) {
    if (!previous || previous.length !== next.length) return next;

    // A sixtieth of a second of rise: instant at the rate the page draws.
    var rise = Math.min(1, (ms || 16) / 16);
    // Sixty decibels a second, so the full height lets go in a second.
    var fall = (ms || 16) * 0.06;

    for (var i = 0; i < next.length; i++) {
      var value = next[i];

      if (value > previous[i]) {
        previous[i] += (value - previous[i]) * rise;
      } else {
        previous[i] = Math.max(value, previous[i] - fall);
      }
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
