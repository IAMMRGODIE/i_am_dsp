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
var wantsCurve = false;
var curveState = null;
var wantsFall = false;
var fallState = null;
var canvasState = null;
  // The Spectrum product's drawn pages, fetched from its own route.
  var spectrumState = null;
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
