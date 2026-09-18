# i_am_string

A physical model of plucked, struck and bowed strings, built on top of
[`i_am_dsp`](../i_am_dsp).

The crate is built around one observation: a plucked, a struck and a bowed
string are **the same vibrating object**. They differ only in the constitutive
law of the contact at the excitation point.

| excitation | contact law |
|---|---|
| plucked | unilateral displacement constraint (a penalty spring) |
| struck | unilateral non-linear felt force, `F = K d^p` |
| bowed | velocity dependent friction, `F = N mu(v_bow - v)` |

Everything else - the dispersive lossy string, the terminations, the
soundboard - is shared.

## How it works

### The string

The string is two delay lines of `N` samples each, carrying the right- and
left-going **force** waves. Force waves rather than velocity waves pay off
twice: the initial condition of a pluck is just `F = -/+ (T/2) dy0/dx`, with
the string tension as the only material constant, and the bridge output is
already the force that drives the soundboard, so nothing has to be
differentiated.

Stiffness is a cascade of first order allpass sections. The loss is a scalar
gain plus a one pole lowpass, fitted so that every partial decays at
`sigma(f) = sigma0 + sigma1 f^2`, which is the standard measured decay law for
a real string.

### The body

A string on its own sounds like a string. What makes it sound like a piano or a
violin is that everything it does is filtered by a body with resonances of its
own: a violin has its air resonance near 280 Hz, the two main wood resonances at
about 470 and 590 Hz, a forest of higher modes, and the bridge hill around 2 to
3 kHz. Those peaks, and just as importantly the valleys between them, are what
the ear uses to name the instrument.

An earlier version of this used a lowpass plus one peaking filter. It sounded
like a tone control, which is exactly the criticism a physical model gets when
it is called synthetic. [`body`](src/body.rs) now carries a per-instrument
table of bandpass modes.

### The attack

Every real instrument makes a noise at the moment of excitation that is not part
of the string at all: the hammer and the mechanism, the plectrum snapping off
the string. A string excited by a perfectly smooth force is a thing that does
not exist, and leaving the noise out is one of the clearest reasons a model is
heard as synthetic. The engine adds a short band-passed noise burst whose level
grows with the dynamics, but less steeply than the note does.

### The port

An excitation is a point where an external force is applied. Writing `A` for
the incident force wave from the nut side and `B` for the incident force wave
from the bridge side, a massless junction gives

```text
v = (A - B + F/2) / Z0
S = A + F/2      departing towards the bridge
Q = B - F/2      departing towards the nut
```

Combined with a contact law this is one scalar implicit equation in `v`, which
`Port::solve` brackets and iterates. The bracket is exact - with no contact
force the root is `(A - B) / Z0`, and with the largest possible force it is at
most `F_max / (2 Z0)` above that - so bisection always converges and Newton only
accelerates it. A negative differential impedance, which is exactly what a bow
presents, cannot make the solver diverge.

### The three excitations

* **Pluck** releases the string from a triangular displacement. The shape is
  split into the two travelling waves exactly.
* **Hammer** is a mass with a Hunt-Crossley felt law, integrated with
  sub-steps. It leaves the string on its own once the felt comes off, and it
  stops integrating at that moment: a free hammer whose position keeps growing
  would drag the felt law to infinity.
* **Bow** is a Stribeck friction curve. The negative slope around zero relative
  velocity subtracts from the port impedance and turns the string into a
  negative resistance oscillator, which is where Helmholtz motion comes from.
  The friction force is odd in the relative velocity; a force that never
  reverses pumps energy in on every sample and the model runs away. The bow
  also bites: the force overshoots by 1.8x at the start of a stroke and settles
  over about forty milliseconds, which is a large part of what makes a bowed
  note sound *started* rather than switched on.

### The damper, the pedal and sympathetic resonance

Every string on a real instrument is driven through the bridge by every other
string. That is why a piano blooms when the sustain pedal is down, and why the
undamped top octave answers a note played two octaves below it.

The model is a bank of high-Q resonators, one per note and per low partial, fed
by the mixed string output. It is one-way on purpose: the resonators are driven
by the strings but do not load them, so the bank can only add a tail and can
never become unstable.

The same bank carries the damper. Each note has an engagement between zero
(felt lifted) and one (felt resting on the string), and it is set from the key
and the pedal together:

```text
engagement = (key released) * (1 - pedal)
```

so holding a key opens the damper, the pedal opens every damper, and a half
pedal leaves them lightly touching. The engagement maps to a decay rate rather
than to a volume, and it rises faster than the pedal falls, because on a real
instrument the dampers lift over the first part of the pedal travel and the
useful half pedal lives near the bottom of it.

```rust
engine.set_pedal(0.5); // half pedal
engine.handle_event(Event::Sustain { position: 1.0 });
```

On the `i_am_dsp` side this follows MIDI CC64 automatically.

### Stereo

`StringEngine::next_frame` returns a stereo pair, and the width is built from
where the sound actually comes from rather than from a widener:

* each note is placed by its register, low to the left and high to the right,
  the way a player hears their own instrument;
* the unison strings of a note are spread around that position, which is where
  most of the natural width of a piano comes from;
* the body is built twice, with the resonances shifted very slightly in
  opposite directions, so the two channels agree on the broad shape but not on
  the fine detail;
* the sympathetic strings, which are spread all over the instrument, are given
  a few milliseconds of delay on one side.

A chord comes out at about 0.92 correlation between the channels and a single
note at about 0.99, which is what a real instrument does: one note is a point
source, a chord fills the space.

## Usage

```rust
use i_am_string::prelude::*;

let mut engine = StringEngine::new(48_000.0);
engine.set_preset(Preset::Violin);
engine.note_on(69, 0.9); // standard MIDI: 69 is A4
for _ in 0..48_000 {
    let sample = engine.next_sample();
}
engine.note_off(69);
```

`StringEngine` speaks standard MIDI note numbers. The `i_am_dsp` adapter
(`i_am_dsp::generators::string_engine::StringInstrument`) translates, because
that library puts A4 at 57.

## Playing it

```sh
cargo run -p i_am_dsp --release
```

Then add **Modelled Strings** to a track and play with the computer keyboard.
Use `--release`: the dispersion filter takes about ten milliseconds to fit the
first time a given pitch is played, and a debug build is far slower than that.

## Honest limitations

* **The dispersion filter is the hard limit of this architecture.** A cascade
  of first order allpass sections can only realise a stiff-string delay curve
  over a limited stretch, and the failure is not graceful: push past it and the
  upper partials land hundreds of cents out, which does not sound like a stiff
  string, it sounds like a detuned saw. Two things follow from that. Every
  modelled partial is now a target of the fit rather than one of sixteen
  samples, the delay lines are no longer asked to carry so much of the round
  trip that the cascade has to achieve an impossible curve, and the model is
  deliberately made only as inharmonic as it can represent *accurately*. A real
  modal string bank would have none of these problems and is the right next
  step.
* The number of partials per string is a few dozen rather than the hundreds a
  bass string really has, because that is what the dispersion filter can keep in
  tune. Bass notes are therefore less rich than they should be.
* Tension modulation (pitch glide and phantom partials) is not modelled yet.
  The string is linear apart from the contact.
* The bodies are tables of a dozen or so modes per instrument, not measured
  responses. A real violin body has several dozen modes in the same range, so
  the model has deeper valleys between its peaks than the instrument does.
* The sympathetic strings are a resonator bank, not a second set of waveguide
  strings. They reproduce the bloom and the octave answers, but they do not
  load the played strings back, and they carry only the two lowest partials of
  each note.
* The bowed strings are built from the four open strings' worth of one scaling
  law, not from a real string set: the tension and impedance do not vary from
  string to string the way gut and steel actually do.
* The una corda and the sostenuto pedal are not modelled.
* Fitting a dispersion filter costs about ten milliseconds, on the audio
  thread, the first time a pitch is played. It is cached per note afterwards.
* Eight voices with the sympathetic bank run at around fifteen times realtime on
  one core, which is the number the test suite guards.

## Three bugs worth remembering

Most of the sound of this model came from fixing three things that were not
typos but genuine misunderstandings, and each of them has a test now.

1. **The end reflections had the wrong sign.** The delay lines carry *force*
   waves, and at a rigid termination it is the velocity that inverts while the
   force does not. Reflecting with minus one gave the model a DC path: a
   constant force produced a constant velocity instead of a static deflection.
   The visible symptom was a bow that held the string rigidly and never
   slipped, so a violin did not sing at all.
2. **The contact solver assumed the force was never negative.** True for a
   hammer, false for friction, which reverses sign. The bracket therefore
   excluded the entire slip phase.
3. **Some derivatives did not match their forces.** `HammerLaw`'s derivative
   ignored the `max(0)` clamp on the felt force, and the friction derivative
   ignored the stiction band. The solver is a Newton iteration, so a derivative
   that disagrees with the function sends it marching off; one note of the piano
   went silent because the hammer was solved to 2500 m/s and never touched the
   string.

## Tests

```sh
cargo test -p i_am_string --release
```

They check physics rather than plumbing: an ideal string's partials land within
a few cents of the harmonic series, a stiff one's are stretched monotonically,
the decay rate matches the requested `sigma0`, a released string really does
stop, a bow limits its own amplitude and locks into a periodic Helmholtz cycle,
a constant force settles into a static deflection, the pedal holds a released
note and a half pedal lands between the two extremes, the undamped strings ring
on after the note is gone, every note of every preset makes a sound, and every
preset stays finite and bounded over several seconds of polyphony.
