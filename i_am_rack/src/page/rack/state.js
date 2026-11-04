// The rack page, part one: the elements the page drives and the state it keeps.
(function () {
  "use strict";

  var rack = document.getElementById("rack");
  var status = document.getElementById("status");

  // A page that dies on load leaves an empty rack and no clue; the footer is
  // the only place a person can see why.
  window.addEventListener("error", function (event) {
    if (status && event.message) status.textContent = "error: " + event.message;
  });
  var rate = document.getElementById("rate");
  var power = document.getElementById("power");
  var panelBody = document.getElementById("panel-body");
  var panelTabs = document.querySelectorAll(".panel-tab");
  var bottomPanel = document.getElementById("bottom");
  var splitter = document.getElementById("splitter");
  var panelToggle = document.getElementById("panel-toggle");
  /// The macro knobs and the automation readouts, refreshed from the host.
  var macroHandles = [];
  var macroValues = [];
  var automationSpans = [];
  var pointerDown = false;
  var panelCollapsed = false;
  var panelRestore = 0;
  var activePanel = "sources";
  var panelPages = null;
  var undoButton = document.getElementById("undo");
  var redoButton = document.getElementById("redo");
  var abButton = document.getElementById("ab");
  var presetInit = document.getElementById("preset-init");
  var presetSave = document.getElementById("preset-save");
  var presetLoad = document.getElementById("preset-load");
  var presetFile = document.getElementById("preset-file");

  var pending = Object.create(null);
  var timer = null;
  var bypassed = false;
  var current = null;
  var rangeLabels = [];
  var paintCrossover = null;
  var collapsedBands = Object.create(null);
  var collapsedGroups = Object.create(null);
  var collapsedSources = Object.create(null);
  var collapsedModules = Object.create(null);
  /// The source a person is looking at, when they picked one: every knob then
  /// shows only that source's modulation.
  var selectedSource = null;
  var sourceNames = Object.create(null);
  var sourceColors = Object.create(null);
  var slotScroll = Object.create(null);
  var sourceScroll = 0;
  var livePainters = [];
  var live = {
    values: Object.create(null),
    phases: Object.create(null),
    history: Object.create(null)
  };
  var liveTimer = null;
  var scopeTimer = null;
  var stereoTimer = null;
  var meterTimer = null;
  // The audio the pages draw: the same three documents the product page polls,
  // kept in one context object so a card can be redrawn whenever one arrives.
  var pageRate = 48000;
  var cardTimer = null;
  var irTimer = null;
  var liveCards = [];
