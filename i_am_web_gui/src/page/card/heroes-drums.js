// The drum pictures: the pitch fall the editor drags, drawn with the same
// bend the voice evaluates.

  function fallX(time) {
    return 26 + Math.min(Math.max(time, 0), 0.25) / 0.25 * 468;
  }

  function fallY(ratio) {
    var octaves = Math.log(Math.max(ratio, 1)) / Math.LN2;
    return 132 - Math.min(octaves, 10) / 10 * 112;
  }

  function fallBend(unit, bend) {
    if (Math.abs(bend) < 0.0001) return unit;
    return (Math.exp(unit * bend) - 1) / (Math.exp(bend) - 1);
  }

  function fallHero(values, context) {
    var points = context && context.fall && context.fall.points;
    var grid = '<rect x="26" y="20" width="468" height="112" fill="none" stroke="#2a2a2a"/>';

    for (var octave = 0; octave <= 10; octave += 2) {
      var y = 132 - octave * (112 / 10);
      grid += '<line x1="26" y1="' + y.toFixed(1) + '" x2="494" y2="' + y.toFixed(1) +
        '" stroke="#232323"/>' +
        '<text x="2" y="' + (y + 3).toFixed(1) + '" fill="#5f5f5f" font-size="7">' +
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

      // Pitch is interpolated in octaves, the same way the voice reads it.
      var from = Math.log2(Math.max(start[1], 1e-4));
      var to = Math.log2(Math.max(end[1], 1e-4));

      for (var sample = 1; sample <= 18; sample++) {
        var unit = sample / 18;
        var time = start[0] + (end[0] - start[0]) * unit;
        var ratio = Math.pow(2, from + (to - from) * fallBend(unit, start[2] || 0));
        path += " L " + fallX(time).toFixed(1) + " " + fallY(ratio).toFixed(1);
      }
    }

    var dots = "";
    sorted.forEach(function (point, dotIndex) {
      dots += '<circle data-point="' + dotIndex + '" cx="' + fallX(point[0]).toFixed(1) +
        '" cy="' + fallY(point[1]).toFixed(1) + '" r="4.5" fill="var(--accent)" ' +
        'stroke="#141414" stroke-width="1.4"/>';
    });

    // A ring on each segment: vertical drag bends the curve without moving the
    // points. Its centre is the midpoint the bend actually produces, so the
    // handle sits on the line it controls.
    var bends = "";
    for (var segment = 0; segment < sorted.length - 1; segment++) {
      var left = sorted[segment];
      var right = sorted[segment + 1];
      if (right[0] - left[0] < 0.002) continue;

      var middle = (left[0] + right[0]) / 2;
      var leftOctave = Math.log2(Math.max(left[1], 1e-4));
      var rightOctave = Math.log2(Math.max(right[1], 1e-4));
      var value = Math.pow(2, leftOctave + (rightOctave - leftOctave) * fallBend(0.5, left[2] || 0));
      bends += '<circle data-bend="' + segment + '" cx="' + fallX(middle).toFixed(1) +
        '" cy="' + fallY(value).toFixed(1) + '" r="5" fill="transparent" ' +
        'stroke="var(--accent)" stroke-opacity="0.7" stroke-width="1.4" pointer-events="all"/>';
    }

    var label = sorted[0][1].toFixed(1) + "x to " + sorted[sorted.length - 1][1].toFixed(1) +
      "x in " + Math.round(sorted[sorted.length - 1][0] * 1000) + " ms";

    return heroSvgMeet(
      grid +
      '<path d="' + path + '" fill="none" stroke="var(--accent)" stroke-width="2" ' +
      'vector-effect="non-scaling-stroke"/>' +
      dots +
      bends +
      '<text x="30" y="13" fill="#8a8a8a" font-size="9">' + label + '</text>'
    );
  }


  /// The kit's click: which one is loaded, and a way to load another.
  function kitHero(values, context) {
    var click = context && context.click;
    var name = click && click.name ? String(click.name) : "no click";
    var detail = click && click.name
      ? (click.samples / Math.max(click.rate, 1)).toFixed(3) + " s \u00b7 " +
        (Math.round(click.rate / 100) / 10) + " kHz"
      : "the synthetic click";

    return heroSvg("") +
      '<div class="hero-label" style="top:6%">CLICK</div>' +
      '<div class="hero-label" style="top:36%; left:0; right:0; text-align:center">' +
      escapeXml(name) + '</div>' +
      '<div class="hero-label" style="top:52%; left:0; right:0; text-align:center">' +
      escapeXml(detail) + '</div>' +
      '<div class="hero-actions">' +
        '<button type="button" class="hero-action" data-action="load">LOAD CLICK</button>' +
      '</div>';
  }


  /// The six one-shot slots: what each drum plays, and a way to load one.
  function samplesHero(values, context) {
    var slots = (context && context.samples) || [];
    var names = ["Kick", "Snare", "Hat", "Clap", "Tom", "Cymbal"];
    var keys = ["kick", "snare", "hat", "clap", "tom", "cymbal"];
    var rows = "";

    for (var index = 0; index < keys.length; index++) {
      var slot = slots[index];
      var y = 22 + index * 22;
      var detail = slot
        ? slot.name + "  " + (slot.samples / Math.max(slot.rate, 1)).toFixed(2) + " s"
        : "synthesised";

      rows += '<text x="26" y="' + y + '" fill="#8a8a8a" font-size="9">' + names[index] + '</text>';
      rows += '<text x="88" y="' + y + '" fill="' + (slot ? "var(--accent)" : "#5f5f5f") +
        '" font-size="9">' + escapeXml(detail) + '</text>';
      if (slot && slot.peaks && slot.peaks.length) {
        var bars = slot.peaks.length;
        var span = 170;

        for (var bar = 0; bar < bars; bar++) {
          var height = Math.abs(slot.peaks[bar]) * 6;
          var x = 232 + (bar / (bars - 1)) * span;

          rows += '<line x1="' + x.toFixed(1) + '" y1="' + (y - 2 - height).toFixed(1) +
            '" x2="' + x.toFixed(1) + '" y2="' + (y - 2 + height).toFixed(1) +
            '" stroke="var(--accent)" stroke-opacity=".55" stroke-width="1.5"/>';
        }
      }

      rows += '<rect data-load="' + keys[index] + '" x="414" y="' + (y - 11) +
        '" width="52" height="15" rx="3" fill="#262626" stroke="#3a3a3a" pointer-events="all" style="cursor:pointer"/>';
      rows += '<text x="440" y="' + (y - 1) + '" fill="#d8d8d8" font-size="8" text-anchor="middle" pointer-events="none">LOAD</text>';
      rows += '<rect data-clear="' + keys[index] + '" x="472" y="' + (y - 11) +
        '" width="18" height="15" rx="3" fill="#262626" stroke="#3a3a3a" pointer-events="all" style="cursor:pointer"/>';
      rows += '<text x="481" y="' + (y - 1) + '" fill="#d8d8d8" font-size="9" text-anchor="middle" pointer-events="none">\u00d7</text>';
    }

    return heroSvgMeet(rows);
  }


  function ampX(time) {
    return 26 + Math.max(0, Math.min(1, time)) * 468;
  }

  function ampLevel(value) {
    return 132 - Math.max(0, Math.min(1, value)) * 112;
  }

  /// The amplitude envelope of the selected drum: level against the share of
  /// its decay. The Decay knob is the length, so the picture is only the shape.
  function ampHero(values, context) {
    var amp = context && context.amp;
    var points = amp && amp.points && amp.points.length ? amp.points : [[0, 1, -2.5], [1, 0, 0]];
    var drum = Math.round(values.amp_drum || 0);
    var knobs = ["decay", "snare_decay", "hat_decay", "clap_decay", "tom_decay", "cymbal_decay"];
    var length = Number(values[knobs[drum]] || 0.3) * 1000;
    var grid = "";
    var path = "";
    var bends = "";
    var dots = "";

    for (var column = 0; column <= 4; column++) {
      var x = 26 + column * 117;
      grid += '<line x1="' + x + '" y1="20" x2="' + x + '" y2="132" stroke="#2a2a2a" stroke-width="1"/>';
    }

    for (var row = 0; row <= 4; row++) {
      var y = 132 - row * 28;
      grid += '<line x1="26" y1="' + y + '" x2="494" y2="' + y + '" stroke="#2a2a2a" stroke-width="1"/>';
    }

    grid += '<text x="26" y="146" fill="#5f5f5f" font-size="7">0</text>';
    grid += '<text x="452" y="146" fill="#5f5f5f" font-size="7">' + Math.round(length) + " ms</text>";
    grid += '<text x="26" y="16" fill="#5f5f5f" font-size="7">LEVEL</text>';

    var move = "M " + ampX(points[0][0]).toFixed(1) + " " + ampLevel(points[0][1]).toFixed(1);
    path += move;

    for (var index = 0; index < points.length - 1; index++) {
      var start = points[index];
      var end = points[index + 1];
      var from = start[1];
      var to = end[1];

      for (var sample = 1; sample <= 18; sample++) {
        var unit = sample / 18;
        var time = start[0] + (end[0] - start[0]) * unit;
        var value = from + (to - from) * fallBend(unit, start[2] || 0);
        path += " L " + ampX(time).toFixed(1) + " " + ampLevel(value).toFixed(1);
      }

      var middle = (start[0] + end[0]) / 2;
      var middleValue = from + (to - from) * fallBend(0.5, start[2] || 0);
      bends += '<circle data-amp-bend="' + index + '" cx="' + ampX(middle).toFixed(1) +
        '" cy="' + ampLevel(middleValue).toFixed(1) + '" r="5" fill="transparent" ' +
        'stroke="var(--accent)" stroke-opacity="0.7" stroke-width="1.4" pointer-events="all"/>';
    }

    points.forEach(function (point, index) {
      dots += '<circle data-amp-point="' + index + '" cx="' + ampX(point[0]).toFixed(1) +
        '" cy="' + ampLevel(point[1]).toFixed(1) + '" r="4.4" fill="var(--accent)" ' +
        'stroke="#0f0f0f" stroke-width="1" pointer-events="all"/>';
    });

    return heroSvgMeet(
      grid +
      '<path d="' + path + '" fill="none" stroke="var(--accent)" stroke-width="2" ' +
      'vector-effect="non-scaling-stroke"/>' +
      bends +
      dots
    );
  }
