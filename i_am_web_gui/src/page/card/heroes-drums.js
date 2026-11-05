// The drum pictures: the pitch fall the editor drags, drawn with the same
// bend the voice evaluates.

  function fallX(time) {
    return 26 + Math.min(Math.max(time, 0), 0.25) / 0.25 * 468;
  }

  function fallY(ratio) {
    var octaves = Math.log(Math.max(ratio, 1)) / Math.LN2;
    return 132 - Math.min(octaves, 3) / 3 * 112;
  }

  function fallBend(unit, bend) {
    if (Math.abs(bend) < 0.0001) return unit;
    return (Math.exp(unit * bend) - 1) / (Math.exp(bend) - 1);
  }

  function fallHero(values, context) {
    var points = context && context.fall && context.fall.points;
    var grid = '<rect x="26" y="20" width="468" height="112" fill="none" stroke="#2a2a2a"/>';

    for (var octave = 0; octave <= 3; octave++) {
      var y = 132 - octave * (112 / 3);
      grid += '<line x1="26" y1="' + y.toFixed(1) + '" x2="494" y2="' + y.toFixed(1) +
        '" stroke="#232323"/>' +
        '<text x="6" y="' + (y + 3).toFixed(1) + '" fill="#5f5f5f" font-size="8">' +
        Math.pow(2, octave).toFixed(0) + 'x</text>';
    }

    for (var step = 0; step <= 5; step++) {
      var x = 26 + step * (468 / 5);
      grid += '<line x1="' + x.toFixed(1) + '" y1="20" x2="' + x.toFixed(1) +
        '" y2="132" stroke="#202020"/>';
    }

    grid += '<text x="26" y="148" fill="#5f5f5f" font-size="8">0</text>' +
      '<text x="462" y="148" fill="#5f5f5f" font-size="8">250 ms</text>';

    if (!points || points.length < 2) {
      return heroSvg(grid) + '<div class="hero-empty">Waiting for the curve...</div>';
    }

    var sorted = points.slice().sort(function (a, b) { return a[0] - b[0]; });
    var path = "M " + fallX(sorted[0][0]).toFixed(1) + " " + fallY(sorted[0][1]).toFixed(1);

    for (var index = 1; index < sorted.length; index++) {
      var start = sorted[index - 1];
      var end = sorted[index];

      for (var sample = 1; sample <= 18; sample++) {
        var unit = sample / 18;
        var time = start[0] + (end[0] - start[0]) * unit;
        var ratio = start[1] + (end[1] - start[1]) * fallBend(unit, start[2] || 0);
        path += " L " + fallX(time).toFixed(1) + " " + fallY(ratio).toFixed(1);
      }
    }

    var dots = "";
    sorted.forEach(function (point, dotIndex) {
      dots += '<circle data-point="' + dotIndex + '" cx="' + fallX(point[0]).toFixed(1) +
        '" cy="' + fallY(point[1]).toFixed(1) + '" r="4.5" fill="var(--accent)" ' +
        'stroke="#141414" stroke-width="1.4"/>';
    });

    var label = sorted[0][1].toFixed(1) + "x to " + sorted[sorted.length - 1][1].toFixed(1) +
      "x in " + Math.round(sorted[sorted.length - 1][0] * 1000) + " ms";

    return heroSvg(
      grid +
      '<path d="' + path + '" fill="none" stroke="var(--accent)" stroke-width="2" ' +
      'vector-effect="non-scaling-stroke"/>' +
      dots +
      '<text x="30" y="13" fill="#8a8a8a" font-size="9">' + label + '</text>'
    );
  }
