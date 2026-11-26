// The additive picture: the bars the voice sums.
  function additiveHero(values, context) {
    if (Object.prototype.hasOwnProperty.call(values, "oscillator.freq_gen.tilt")) {
      return sinesHero(values);
    }

    var script = context && context.script;

    if (script && script.error) {
      return heroSvg("") +
        '<div class="hero-empty" style="color:#ff7a6b">' +
        escapeXml(String(script.error).slice(0, 120)) + '</div>';
    }

    if (script && script.spectrum && script.spectrum.length) {
      var spectrum = script.spectrum;
      var lastLog = 1;
      var loudest = 0;

      spectrum.forEach(function (partial) {
        lastLog = Math.max(lastLog, Math.log(Math.max(partial[0], 1.0001)));
        loudest = Math.max(loudest, Math.abs(partial[1]));
      });

      var floor = 132;
      var high = floor - 24;
      var bars = "";

      spectrum.forEach(function (partial) {
        var x = 20 + (Math.log(Math.max(partial[0], 1.0001)) / lastLog) * 480;
        var y = floor - (Math.abs(partial[1]) / Math.max(loudest, 1e-6)) * high * 0.9;

        bars += '<line x1="' + x.toFixed(1) + '" y1="' + floor + '" x2="' + x.toFixed(1) +
          '" y2="' + y.toFixed(1) + '" stroke="var(--accent)" stroke-width="2" ' +
          'vector-effect="non-scaling-stroke"/>';
      });

      return heroSvg(
        '<line x1="20" y1="' + floor + '" x2="500" y2="' + floor + '" stroke="#2c2c2c"/>' + bars
      ) + '<div class="hero-label" style="top:2%">SCRIPT \u00b7 ' + spectrum.length +
        ' partials</div>';
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
