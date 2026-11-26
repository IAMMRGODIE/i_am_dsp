// The unison picture: the detuned stack and the spread it buys.
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
