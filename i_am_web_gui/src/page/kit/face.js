// Part seven: visibility, switches, the mode choice and the pages.
  function applyVisibility(mode) {
    if (!face || !face.visible) return;

    var ids = null;

    for (var i = 0; i < face.visible.length; i++) {
      if (face.visible[i][0] === mode) {
        ids = face.visible[i][1];
        break;
      }
    }

    // A mode list names the controls of the mode's own section; the envelope and
    // the voice are not part of it and must not be swept away.
    var touched = [];

    if (ids) {
      ids.forEach(function (id) {
        var control = controls[id];

        if (!control || !control.el) return;

        var section = control.el.closest(".section");

        if (section && touched.indexOf(section) < 0) touched.push(section);
      });
    }

    Object.keys(controls).forEach(function (id) {
      var control = controls[id];
      if (!control.el || id === face.mode) return;

      if (touched.length) {
        var section = control.el.closest(".section");

        if (!section || touched.indexOf(section) < 0) return;
      }

      var show = !ids || ids.indexOf(id) >= 0;
      control.el.style.display = show ? "" : "none";
    });

    applyEqType();
  }

  /// A cut or a notch has no gain, so hide that knob for the selected band.
  function applyEqType() {
    if (!face || face.mode !== "band") return;

    var index = Math.round(valueOf("band", 0));
    var type = Math.round(valueOf("b" + (index + 1) + "_type", 0));
    var gain = controls["b" + (index + 1) + "_gain"];
    var slope = controls["b" + (index + 1) + "_slope"];

    if (gain && gain.el) gain.el.style.display = type <= 2 ? "" : "none";
    if (slope && slope.el) {
      slope.el.style.display = type === 3 || type === 4 ? "" : "none";
    }
  }

  /// Show the branch the current switch selects. Only the listed controls move;
  /// everything else is left to the mode's visibility.
  function applySwitches() {
    if (!face || !face.switches) return;

    face.switches.forEach(function (row) {
      var control = controls[row.control];
      if (!control) return;

      var on = !!control.value;
      var show = on ? row.on : row.off;
      var hide = on ? row.off : row.on;

      hide.forEach(function (id) {
        var target = controls[id];
        if (target && target.el) target.el.style.display = "none";
      });

      show.forEach(function (id) {
        var target = controls[id];
        if (target && target.el) target.el.style.display = "";
      });
    });
  }


  /// A choice in the header: a label and a drop-down, without taking a row
  /// from the faceplate.
  function buildHeaderChoice(spec) {
    var el = element("div", "choice");
    var label = element("span", "label");
    label.textContent = spec.label || "";

    var control = { spec: spec, value: spec.default, el: el, paint: function () {} };
    controls[spec.id] = control;

    if (window.IAmCard && IAmCard.choiceHandle) {
      var handle = IAmCard.choiceHandle(spec, control.value, function (_, value) {
        setValue(control, value, true);
      });
      handle.el.className = "mcard-select mode-select";
      handle.el.setAttribute("aria-label", spec.label || "Choice");
      control.paint = function () { handle.set(control.value); };

      el.appendChild(label);
      el.appendChild(handle.el);
      control.paint();
      return el;
    }

    var select = document.createElement("select");
    select.className = "mode-select";
    select.setAttribute("aria-label", spec.label || "Choice");

    (spec.choices || []).forEach(function (name, index) {
      var option = document.createElement("option");
      option.value = String(index);
      option.textContent = name;
      select.appendChild(option);
    });

    select.addEventListener("change", function () {
      control.value = parseInt(select.value, 10);
      control.paint();
      markDirty();
      if (window.ipc) window.ipc.postMessage(spec.id + "=" + control.value);
    });

    control.paint = function () {
      select.value = String(Math.round(control.value));
    };

    el.appendChild(label);
    el.appendChild(select);
    control.paint();
    return el;
  }

  function buildMode(spec) {
    var el = element("div", "mode mode-drop");

    var control = { spec: spec, value: spec.default, el: null, paint: function () {} };
    controls[spec.id] = control;

    var caption = element("span", "header-caption");
    caption.textContent = spec.label || "";
    el.appendChild(caption);

    var handle = IAmCard.choiceHandle(spec, control.value, function (_, value) {
      setValue(control, value, true);
    });
    handle.el.className = "mcard-select mode-select";
    handle.el.setAttribute("aria-label", spec.label || "Mode");
    control.el = el;
    control.paint = function () {
      var active = Math.round(control.value);
      handle.set(active);
      if (spec.id === face.mode) applyVisibility(active);
    };

    el.appendChild(handle.el);
    control.paint();
    return el;
  }

  /// One strip of tabs, and only the sections the active tab names.
  ///
  /// A section that no page names stays visible on every page: the page list
  /// hides what it mentions and never a control nobody thought about.
  function buildPages(sectionEls) {
    var pages = face.pages;
    var mentioned = {};

    pages.forEach(function (page) {
      (page.sections || []).forEach(function (name) { mentioned[name] = true; });
    });

    var strip = element("div", "face-tabs");
    var buttons = [];

    function show(index) {
      var page = pages[index] || pages[0];
      var shown = {};

      (page.sections || []).forEach(function (name) { shown[name] = true; });

      buttons.forEach(function (button, position) {
        button.setAttribute("aria-pressed", position === index ? "true" : "false");
      });

      sectionEls.forEach(function (section) {
        var visible = !section.name || !mentioned[section.name] || !!shown[section.name];
        section.el.style.display = visible ? "" : "none";
      });
    }

    pages.forEach(function (page, index) {
      var button = element("button", "face-tab");
      button.type = "button";
      button.textContent = page.name;
      button.setAttribute("aria-pressed", "false");
      button.addEventListener("click", function () { show(index); });
      buttons.push(button);
      strip.appendChild(button);
    });

    show(0);
    return strip;
  }
