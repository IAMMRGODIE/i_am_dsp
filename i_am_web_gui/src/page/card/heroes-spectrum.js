// Part three: the spectrum picture.
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
