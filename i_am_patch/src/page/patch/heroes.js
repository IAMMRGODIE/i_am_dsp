// Part three: the pictures the constructor nodes draw.
  // Heroes for the constructor nodes. The card kit draws product pictures from
  // their own metadata; these nodes have no product, so the page draws what the
  // node is set to: the table, the phase map, the modulation, the envelope.
  function patchSvg(content) {
    // `mcard-graph` makes the svg take the hero's full box; without it the
    // waveform keeps its own aspect and the box's lower half stays empty.
    return '<svg class="mcard-graph" viewBox="0 0 520 160" preserveAspectRatio="none">' + content + '</svg>';
  }

  function patchValue(values, name, fallback) {
    var value = values && values[name];
    return typeof value === "number" ? value : fallback;
  }

  function patchTrace(points) {
    var path = "";

    for (var i = 0; i < points.length; i++) {
      path += (i ? " L " : "M ") + points[i][0].toFixed(1) + " " + points[i][1].toFixed(1);
    }

    return path;
  }

  /// A sampler over one cycle of samples, linearly interpolated.
  function patchSamples(values) {
    var length = values.length;

    return function (t) {
      var x = patchFrac(t) * length;
      var index = Math.floor(x) % length;
      var next = (index + 1) % length;
      var fraction = x - Math.floor(x);

      return values[index] + (values[next] - values[index]) * fraction;
    };
  }

  /// The sample function of a node: the engine's own rendering when it has been
  /// fetched, a plain table's shape otherwise.
  function patchSourceShape(source) {
    if (!source) return null;

    const samples = source.id !== undefined ? waveCache.get(source.id) : null;

    if (samples) return patchSamples(samples);

    const kind = source.kind;
    const shape = Math.round((source.params && source.params.shape) || 0);

    if (kind === "table:shape" || kind === "osc:shape") {
      return function (t) {
        return patchShapeValue(shape, t);
      };
    }

    return null;
  }

  function patchShapeValue(shape, t) {
    var cycle = t - Math.floor(t);

    if (shape === 1) return 2 * Math.abs(2 * cycle - 1) - 1;
    if (shape === 2) return 2 * cycle - 1;
    if (shape === 3) return cycle < 0.5 ? 1 : -1;
    if (shape === 4) {
      var noise = Math.sin(cycle * 12345.678) * 43758.5453;
      return (noise - Math.floor(noise)) * 2 - 1;
    }

    return Math.sin(cycle * Math.PI * 2);
  }

  function patchTableHero(values) {
    var shapes = ["SINE", "TRIANGLE", "SAW", "SQUARE", "NOISE"];
    var shape = Math.round(patchValue(values, "shape", 0));
    var points = [];

    for (var i = 0; i <= 240; i++) {
      var t = i / 240;
      points.push([20 + t * 480, 80 - patchShapeValue(shape, t) * 52]);
    }

    return patchSvg(
      '<line x1="20" y1="80" x2="500" y2="80" stroke="#2f2f2f" stroke-dasharray="3 5"/>' +
      '<path d="' + patchTrace(points) + '" fill="none" stroke="var(--accent)" stroke-width="2" vector-effect="non-scaling-stroke"/>'
    ) + '<div class="hero-label" style="left:2%;top:2%">' + shapes[shape] + '</div>';
  }

  /// The table a warp reads, sampled through the warp: the actual waveform, not
  /// the phase map that produced it.
  function patchWarpHero(mode, values, context) {
    var amount = patchValue(values, "amount", 0);
    var power = Math.max(patchValue(values, "power", 1), 0.25);
    var ratio = Math.max(patchValue(values, "ratio", 1), 1);
    var steps = Math.max(patchValue(values, "steps", 1), 1);
    var shape = patchSourceShape(context && context.source(0)) ||
      function (t) { return patchShapeValue(0, t); };
    var points = [];

    for (var i = 0; i <= 240; i++) {
      var t = i / 240;
      var folded = 1 - Math.abs(2 * t - 1);
      var warped = mode === "bend" ? Math.pow(t, power)
        : mode === "mirror" ? t + (folded - t) * amount
        : mode === "sync" ? t * ratio - Math.floor(t * ratio)
        : mode === "quantize" ? Math.floor(t * steps) / steps
        : t + (0.5 + 0.5 * Math.sin(t * Math.PI * 2) - t) * amount;

      points.push([20 + t * 480, 80 - shape(warped) * 56]);
    }

    var names = { bend: "BEND", mirror: "MIRROR", sync: "SYNC", quantize: "QUANTIZE", remap: "REMAP" };

    return patchSvg(
      '<line x1="20" y1="80" x2="500" y2="80" stroke="#2f2f2f" stroke-dasharray="3 5"/>' +
      '<path d="' + patchTrace(points) + '" fill="none" stroke="var(--accent)" stroke-width="2" vector-effect="non-scaling-stroke"/>'
    ) + '<div class="hero-label" style="left:2%;top:2%">' + names[mode] + '</div>';
  }

  function patchFrac(t) {
    return t - Math.floor(t);
  }

  /// The two tables a modulation reads, combined the way the node combines them.
  function patchModHero(mode, values, context) {
    var depth = patchValue(values, "depth", 0);
    var ratio = Math.max(patchValue(values, "ratio", 1), 0.0625);
    // A modulator slower than the carrier moves almost not at all over two
    // cycles, so the window grows until one modulator cycle fits.
    var cycles = Math.min(Math.max(2, Math.ceil(1 / ratio)), 32);
    var sine = function (t) { return Math.sin(t * Math.PI * 2); };
    var carrier = patchSourceShape(context && context.source(0)) || sine;
    var modulator = patchSourceShape(context && context.source(1)) || sine;
    var points = [];
    var steps = 480;
    var integral = 0;

    for (var i = 0; i <= steps; i++) {
      var t = i / steps * cycles;
      var m = modulator(patchFrac(t * ratio));
      var value;

      if (mode === "am") value = carrier(patchFrac(t)) * (1 + depth * m);
      else if (mode === "rm") value = carrier(patchFrac(t)) * (1 - depth + depth * m);
      else if (mode === "pm") value = carrier(patchFrac(t + depth * m));
      else {
        integral += m * (cycles / steps);
        value = carrier(patchFrac(t + depth * integral));
      }

      points.push([20 + i / steps * 480, 80 - value * 56]);
    }

    var names = { am: "AM", rm: "RM", pm: "PM", fm: "FM" };

    return patchSvg(
      '<line x1="20" y1="80" x2="500" y2="80" stroke="#2f2f2f" stroke-dasharray="3 5"/>' +
      '<path d="' + patchTrace(points) + '" fill="none" stroke="var(--accent)" stroke-width="2" vector-effect="non-scaling-stroke"/>'
    ) + '<div class="hero-label" style="left:2%;top:2%">' +
        names[mode] +
        " · " +
        depth.toFixed(2) +
        (Math.abs(ratio - 1) > 0.001 ? " × " + ratio.toFixed(2) : "") +
        "</div>";
  }

  function patchEnvelopeHero(values) {
    var attack = Math.min(Math.max(patchValue(values, "attack_time", 0.01), 0.001), 0.9);
    var decay = Math.min(Math.max(patchValue(values, "decay_time", 0.2), 0.001), 0.9);
    var sustain = Math.min(Math.max(patchValue(values, "sustain_level", 0.7), 0), 1);
    var release = Math.min(Math.max(patchValue(values, "release_time", 0.3), 0.001), 1);
    var x = function (time) { return 20 + time * 300; };
    var y = function (level) { return 148 - level * 120; };
    var points = [
      [x(0), y(0)],
      [x(attack), y(1)],
      [x(Math.min(attack + decay, 0.98)), y(sustain)],
      [x(1), y(sustain)],
      [x(1 + release), y(0)]
    ];

    return patchSvg(
      '<line x1="20" y1="148" x2="500" y2="148" stroke="#2a2a2a"/>' +
      '<path d="' + patchTrace(points) + ' L 20 148 Z" fill="var(--accent)" fill-opacity=".08"/>' +
      '<path d="' + patchTrace(points) + '" fill="none" stroke="var(--accent)" stroke-width="2" vector-effect="non-scaling-stroke"/>'
    ) + '<div class="hero-label" style="left:2%;top:2%">ADSR</div>';
  }

  function patchProductHero(name, shape) {
    return function () {
      var points = [];

      for (var i = 0; i <= 240; i++) {
        var t = i / 240;
        points.push([20 + t * 480, 80 - patchShapeValue(shape, t) * 50]);
      }

      return patchSvg(
        '<line x1="20" y1="80" x2="500" y2="80" stroke="#2f2f2f" stroke-dasharray="3 5"/>' +
        '<path d="' + patchTrace(points) + '" fill="none" stroke="var(--accent)" stroke-width="2" vector-effect="non-scaling-stroke"/>'
      ) + '<div class="hero-label" style="left:2%;top:2%">' + name + '</div>';
    };
  }

  if (window.IAmCard) {
    IAmCard.registerHero("patch-wavetable", patchTableHero);
    IAmCard.registerHero("patch-warp-bend", function (values, context) { return patchWarpHero("bend", values, context); });
    IAmCard.registerHero("patch-warp-mirror", function (values, context) { return patchWarpHero("mirror", values, context); });
    IAmCard.registerHero("patch-warp-sync", function (values, context) { return patchWarpHero("sync", values, context); });
    IAmCard.registerHero("patch-warp-quantize", function (values, context) { return patchWarpHero("quantize", values, context); });
    IAmCard.registerHero("patch-warp-remap", function (values, context) { return patchWarpHero("remap", values, context); });
    IAmCard.registerHero("patch-mod-am", function (values, context) { return patchModHero("am", values, context); });
    IAmCard.registerHero("patch-mod-rm", function (values, context) { return patchModHero("rm", values, context); });
    IAmCard.registerHero("patch-mod-pm", function (values, context) { return patchModHero("pm", values, context); });
    IAmCard.registerHero("patch-mod-fm", function (values, context) { return patchModHero("fm", values, context); });
    IAmCard.registerHero("patch-envelope", patchEnvelopeHero);
    // The envelope's shape page draws the oscillator it is actually fed, not a
    // fixed sine: the source's kind and parameters decide the picture.
    IAmCard.registerHero("patch-osc", function (values, context) {
      const source = context && context.oscSource;

      if (!source) return patchProductHero("OSC", 0)(values, context);

      const kind = source.kind;
      const params = source.params || {};
      const sampled = waveCache.get(source.id);

      // The engine's own cycle beats any hand-drawn guess.
      if (sampled) {
        const sample = patchSamples(sampled);
        const points = [];

        for (let index = 0; index <= 240; index++) {
          const t = index / 240;
          points.push([20 + t * 480, 80 - sample(t) * 56]);
        }

        return patchSvg(
          '<line x1="20" y1="80" x2="500" y2="80" stroke="#2f2f2f" stroke-dasharray="3 5"/>' +
          '<path d="' + patchTrace(points) + '" fill="none" stroke="var(--accent)" stroke-width="2" vector-effect="non-scaling-stroke"/>'
        ) + '<div class="hero-label" style="left:2%;top:2%">' +
          String(kind.split(":")[1] || kind).toUpperCase() + '</div>';
      }

      if (kind === "table:shape" || kind === "osc:shape") return patchTableHero(params);

      const mode = kind.indexOf("osc:") === 0 ? kind.slice(4) : "";

      if (["am", "rm", "pm", "fm"].includes(mode)) return patchModHero(mode, params, context);

      if (kind.indexOf("table:") === 0) {
        const product = kind.slice(6);
        const heroes = { shapes: "shape", additive: "partials", spectrum: "spectra" };
        const hero = heroes[product];

        if (hero) {
          const html = IAmCard.hero(hero, params, context);

          if (html) return html;
        }
      }

      return patchProductHero("OSC", 0)(values, context);
    });
    IAmCard.registerHero("patch-osc-shapes", patchProductHero("SHAPES", 0));
    IAmCard.registerHero("patch-osc-additive", patchProductHero("ADDITIVE", 2));
    IAmCard.registerHero("patch-osc-spectrum", patchProductHero("SPECTRUM", 1));
    IAmCard.registerHero("patch-osc-script", patchProductHero("SCRIPT", 3));
    IAmCard.registerHero("patch-osc-canvas", patchProductHero("CANVAS", 4));
    IAmCard.registerHero("patch-canvas-table", patchProductHero("CANVAS", 4));
  }

  CARD_HEROES["table:shape"] = "patch-wavetable";
  CARD_HEROES["table:canvas"] = "patch-canvas-table";
  CARD_HEROES["osc:shape"] = "patch-wavetable";
  CARD_HEROES["osc:am"] = "patch-mod-am";
  CARD_HEROES["osc:rm"] = "patch-mod-rm";
  CARD_HEROES["osc:pm"] = "patch-mod-pm";
  CARD_HEROES["osc:fm"] = "patch-mod-fm";
  CARD_HEROES["warp:bend"] = "patch-warp-bend";
  CARD_HEROES["warp:mirror"] = "patch-warp-mirror";
  CARD_HEROES["warp:sync"] = "patch-warp-sync";
  CARD_HEROES["warp:quantize"] = "patch-warp-quantize";
  CARD_HEROES["warp:remap"] = "patch-warp-remap";
  CARD_HEROES["adsr"] = "patch-envelope";
  CARD_HEROES["table:shapes"] = "patch-osc-shapes";
  CARD_HEROES["table:additive"] = "patch-osc-additive";
  CARD_HEROES["table:spectrum"] = "patch-osc-spectrum";
  CARD_HEROES["table:script"] = "patch-osc-script";

