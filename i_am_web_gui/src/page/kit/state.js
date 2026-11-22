// The faceplate page, part one: the elements and the state the page keeps.
(function () {
  "use strict";

  var app = document.getElementById("app");
  // Whether the page is a browser preview or lives in a host. The custom
  // protocol WebView2 actually exposes is an http:// origin, so the protocol
  // cannot tell the difference; the server says so in /face instead.
  var isPreview = false;
  var face = null;
  var accent = "#8a6aff";
  var negative = "#9a8fbf";
  var controls = {};
  // The control the pointer is on, so a hero can show what that control edits.
  var activeControl = "";
  var heroes = [];
  var dirty = false;
  /// Whether the queued repaint is one the page asked for over its own hero
  /// buttons, which the hover guard must let through.
  var forced = false;
  var polling = false;
  var bypassed = false;
  var powerButton = null;
  var modeSpec = null;
  var wantsScope = false;
var wantsAutotune = false;
var autotuneState = { notes: -1 };
var wantsMeter = false;
var wantsStereo = false;
var wantsParticles = false;
var wantsIr = false;
var wantsWaves = false;
var waveInput = null;
var wantsSample = false;
var sampleInput = null;
// The table the shapes face imported: its name, frame count and picture.
var wavesState = null;
// How the next import cuts the file: auto, fixed, pitch or single.
var waveMode = "auto";
// The sample the sampler face imported: its name, rate and picture.
var sampleState = null;
// The part of the sample the picture shows: the whole thing until zoomed.
var sampleView = { from: 0, to: 1 };
// True while a slice mark is being dragged, so the poll leaves the picture alone.
var sampleDragging = false;
// True while an IR fade dot is being dragged, for the same reason.
var irDragging = false;
var wantsTable = false;
var tableInput = null;
var irState = null;
var irInput = null;
var wantsScript = false;
var scriptState = null;
var scriptArea = null;
var scriptStatus = null;
var scriptError = null;
var scriptDirty = false;
var scriptEpoch = 0;
// The last spectrum and error the panel saw, so the picture only redraws on a change.
var scriptFingerprint = "";
var wantsCurve = false;
var curveState = null;
var wantsFall = false;
// The drummer's click slot, and its file input.
var wantsClick = false;
var clickState = null;
var clickInput = null;
// The drummer's six one-shot slots.
var wantsSamples = false;
var samplesState = [];
// The amplitude envelope of the drum the Amp page selects.
var wantsAmp = false;
var kickMode = "pitch";
var snareMode = "base";
var ampCurves = [[], [], [], [], [], []];
var ampDrag = null;
var fallState = null;
var canvasState = null;
  // The Spectrum product's drawn pages, fetched from its own route.
  var spectrumState = null;
  // The History product's curve pages.
  var historyState = {
    generation: 0,
    pages: [{ points: [{ time: 0, value: 1, bend: 0 }, { time: 1, value: 1, bend: 0 }] }]
  };
  // The Spectral Dynamics product's threshold and ratio pages, and which of the
  // two the one curve hero is showing.
  var spectralDynamicsState = { threshold: [], ratio: [] };
  var spectralDynamicsCurve = { view: "threshold" };
  // The Filter Table product's kernel frame.
  var filterTableState = {
    generation: 0,
    pages: [[{ time: 0, value: 0, bend: 0 }, { time: 1, value: 1, bend: 0 }]]
  };
  // The sidechain trace the Filter Table kernel follows, and whether the page is
  // currently showing it.
  var filterTableSidechain = { input: null, wanted: false };
var wantsOtt = false;
var ottState = null;
var wantsVocoder = false;
var vocoderState = null;
var noteBadge = null;
var stereoScope = {
  inputLeft: null,
  inputRight: null,
  outputLeft: null,
  outputRight: null
};
var meter = { input: null, sidechain: null };
// A short level history so the right-hand envelope can scroll; the wheel
// changes how much of it is visible, which changes how fast it appears to move.
var envHistory = { input: [], output: [] };
var envWindow = 128;
var eqDrag = null;
var eqSpectrum = { input: null, output: null, rate: 0, inputPath: "", outputPath: "" };
var ENV_MAX = 512;
  var scope = { input: null, output: null, sidechain: null };
  var scale = 1;
  var spectra = { input: null, output: null };
