// Part two: the number formatting and the value mapping.
  function clamp(value, low, high) {
    return Math.min(high, Math.max(low, value));
  }

  function element(tag, className) {
    var node = document.createElement(tag);
    if (className) node.className = className;
    return node;
  }

  function mixHex(from, to, amount) {
    var a = [1, 3, 5].map(function (i) { return parseInt(from.slice(i, i + 2), 16); });
    var b = [1, 3, 5].map(function (i) { return parseInt(to.slice(i, i + 2), 16); });
    return "#" + a.map(function (value, i) {
      return Math.round(value + (b[i] - value) * amount).toString(16).padStart(2, "0");
    }).join("");
  }

  // ------------------------------------------------------------- formatting
  function beats(value) {
    var forms = [[1, ""], [1.5, " dotted"], [2 / 3, " triplet"]];
    for (var i = 0; i < forms.length; i++) {
      var note = value / forms[i][0];
      if (note <= 0) continue;
      var denominator = Math.round(4 / note);
      if (denominator >= 1 && denominator <= 128 &&
          (denominator & (denominator - 1)) === 0 &&
          Math.abs(4 / denominator - note) < 1e-4) {
        return "1/" + denominator + forms[i][1];
      }
    }
    return value.toFixed(2) + " beats";
  }

  function format(spec, value) {
    if (spec.kind === "bool") return value ? "on" : "off";
    if (spec.kind === "int" && spec.choices && spec.choices.length) {
      return spec.choices[clamp(Math.round(value), 0, spec.choices.length - 1)] || String(value);
    }
    switch (spec.unit) {
      case "db": return (value > 0 ? "+" : "") + value.toFixed(2) + " dB";
      case "hz":
        if (value >= 1000) return (value / 1000).toFixed(2) + " kHz";
        if (value >= 10) return value.toFixed(0) + " Hz";
        return value.toFixed(2) + " Hz";
      case "percent": return (value * 100).toFixed(1) + " %";
      case "ms": return value.toFixed(1) + " ms";
      case "s": return value.toFixed(3) + " s";
      case "samples": return value.toFixed(0);
      case "beats": return beats(value);
      default: return spec.kind === "int" ? String(Math.round(value)) : value.toFixed(2);
    }
  }

  /// The number shown in a knob's editor, in the unit the readout uses.
  // ------------------------------------------------------------ value space
  /// The scale of the smooth logarithmic map used when a logarithmic range
  /// reaches zero or goes negative.
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
    if (spec.max === spec.min) return 0;

    if (spec.log && spec.max > spec.min) {
      return spec.min > 0
        ? Math.log(value / spec.min) / Math.log(spec.max / spec.min)
        : symlogNorm(spec, value);
    }

    return (value - spec.min) / (spec.max - spec.min);
  }

  function at(spec, position) {
    position = clamp(position, 0, 1);

    if (spec.log && spec.max > spec.min) {
      return spec.min > 0
        ? spec.min * Math.pow(spec.max / spec.min, position)
        : symlogAt(spec, position);
    }

    return spec.min + position * (spec.max - spec.min);
  }

  function snap(spec, value) {
    if (spec.step && spec.step > 0) value = Math.round(value / spec.step) * spec.step;
    return clamp(value, spec.min, spec.max);
  }

  function setValue(control, value, send) {
    value = snap(control.spec, value);

    // An integer parameter has to go back as an integer: the host parses it with
    // the parameter's own kind, and "8.37" is not an integer.
    if (control.spec.kind === "int") value = Math.round(value);

    control.value = value;
    control.paint();
    markDirty();
    applySwitches();

    if (control.spec.id.indexOf("_type") >= 0) applyEqType();

    if (send && window.ipc) {
      var wire = control.spec.kind === "int" ? Math.round(value) : value;
      window.ipc.postMessage(control.spec.id + "=" + wire);
    }
  }

  // --------------------------------------------------------------- controls
