// Part eight: the hero registry.
  function registerHero(kind, renderer) {
    HEROES[kind] = renderer;
  }

  function hero(kind, values, context) {
    var renderer = HEROES[kind];
    return renderer ? renderer(values, context) : "";
  }

  registerHero("spectrum", spectrumHero);
  registerHero("canvas", canvasHero);
  registerHero("spectra", spectraHero);
  registerHero("transfer", transferHero);
  registerHero("compressor", compressorHero);
  registerHero("impulse", impulseHero);
  registerHero("pitch", pitchHero);
  registerHero("particles", particlesHero);
  registerHero("curve", curveHero);
  registerHero("mapping", mappingHero);
  registerHero("ott", ottHero);
  registerHero("vocodex", vocoderHero);
  registerHero("eq", eqHero);
  registerHero("gate", function (values, context) { return dynamicsHero(values, "gate", context); });
  registerHero("expander", function (values, context) { return dynamicsHero(values, "expander", context); });
  registerHero("limiter", function (values, context) { return dynamicsHero(values, "limiter", context); });
  registerHero("stereo", stereoHero);
  registerHero("tail", tailHero);
  registerHero("taps", tapsHero);
  registerHero("lfo", lfoHero);
  registerHero("phase", phaseHero);
  registerHero("balance", balanceHero);
  registerHero("waveform", waveformHero);
  registerHero("unison", unisonHero);
  registerHero("resonator", resonatorHero);
  registerHero("formant", formantHero);
  registerHero("autotune", autotuneHero);
  registerHero("partials", partialsHero);
  registerHero("string", stringHero);
  registerHero("vibrato", vibratoHero);
  registerHero("shape", shapeHero);
  registerHero("modal", modalHero);
  registerHero("envelope", envelopeHero);
  registerHero("additive", additiveHero);
  registerHero("fall", fallHero);
