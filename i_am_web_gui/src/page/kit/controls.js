// Part three: the controls and the small SVG helpers they are built with.
  function makeControl(spec, header) {
    var control = {
      spec: spec,
      value: spec.default,
      el: null,
      paint: function () {}
    };

    controls[spec.id] = control;

    var el;

    if (header && spec.kind === "bool") el = buildHeaderDot(control);
    else if (spec.widget === "notes") el = buildNotePicker(control);
    else if (spec.widget === "toggle" || spec.kind === "bool") el = buildToggle(control);
    else if (spec.widget === "choice") el = buildChoice(control);
    else if (spec.widget === "slider") el = buildSlider(control);
    else el = buildKnob(control);

    control.el = el;

    // A held control owns the picture it edits: the unison's random phase, for
    // one, is only worth drawing while the knob is being moved.
    el.addEventListener("pointerdown", function () { setActiveControl(spec.id); });
    el.addEventListener("focusin", function () { setActiveControl(spec.id); });
    el.addEventListener("focusout", function () { setActiveControl(""); });
    return el;
  }

  /// A boolean in the header is a dot, like the bypass one: no label and no row
  /// taken from the faceplate.
  function buildHeaderDot(control) {
    var spec = control.spec;
    var el = element("button", "header-dot");
    el.type = "button";
    el.title = spec.label;
    el.setAttribute("aria-label", spec.label);
    el.appendChild(element("span", "dot"));

    control.paint = function () {
      var on = !!control.value;
      el.classList.toggle("on", on);
      el.setAttribute("aria-pressed", on ? "true" : "false");
    };

    el.addEventListener("click", function () {
      setValue(control, control.value ? 0 : 1, true);
    });

    control.paint();
    return el;
  }

  function buildKnob(control) {
    var spec = control.spec;
    var handle = IAmCard.controlHandle(spec, control.value, function (_, value) {
      setValue(control, value, true);
    });

    handle.el.classList.add("face-knob");
    control.paint = function () { handle.set(control.value, false); };
    control.paint();
    return handle.el;
  }

  function buildNotePicker(control) {
    var handle = IAmCard.notePickerHandle(control.spec, control.value, function (_, value) {
      setValue(control, value, true);
    });
    handle.el.classList.add("face-notes");
    control.paint = function () { handle.set(control.value); };
    control.paint();
    return handle.el;
  }

  function buildToggle(control) {
    var spec = control.spec;
    var handle = IAmCard.switchHandle(spec, control.value, function (_, value) {
      setValue(control, value, true);
    });
    var el = element("div", "toggle");
    var label = element("div", "label");
    label.textContent = spec.label;
    var readout = element("div", "readout");
    handle.el.classList.add("face-switch");

    control.paint = function () {
      handle.set(control.value, false);
      readout.textContent = format(spec, control.value);
    };

    el.appendChild(handle.el);
    el.appendChild(label);
    el.appendChild(readout);
    control.paint();
    return el;
  }

  function buildChoice(control) {
    var spec = control.spec;
    var handle = IAmCard.choiceHandle(spec, control.value, function (_, value) {
      setValue(control, value, true);
    });
    var el = element("div", "choice");
    var label = element("div", "label");
    label.textContent = spec.label;
    handle.el.classList.add("face-choice");

    control.paint = function () { handle.set(control.value); };

    el.appendChild(label);
    el.appendChild(handle.el);
    control.paint();
    return el;
  }

  function buildSlider(control) {
    var spec = control.spec;
    var handle = IAmCard.sliderHandle(spec, control.value, function (_, value) {
      setValue(control, value, true);
    });
    var el = element("div", "slider");
    handle.el.classList.add("face-slider");
    control.paint = function () { handle.set(control.value, false); };
    el.appendChild(handle.el);
    control.paint();
    return el;
  }

  function valueOf(id, fallback) {
    var control = controls[id];
    return control ? control.value : fallback;
  }

  function heroSvg(content) {
    return '<svg viewBox="0 0 520 160" preserveAspectRatio="none">' + content + '</svg>';
  }

  /// A hero that keeps its proportions: a circle stays a circle whatever shape
  /// the hero box is.
  function heroSvgSquare(content) {
    return '<svg viewBox="0 0 520 160" preserveAspectRatio="xMidYMid meet">' + content + '</svg>';
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

  function escapeXml(value) {
    return String(value === null || value === undefined ? "" : value)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  /// The scripted transfer function: a shaped curve over the input range, with
  /// the identity line for reference.
  /// The EQ's RBJ coefficients, mirrored from the DSP so the curve is the
  /// filter and not an approximation of it.
  /// The EQ curve, its frequency/dB grid and the six draggable band nodes.
