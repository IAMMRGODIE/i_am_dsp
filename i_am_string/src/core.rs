//! The pluggable string core.
//!
//! Everything above this file - the polyphony, the dampers, the sustain pedal,
//! the body, the sympathetic strings - is written against the whole instrument.
//! Everything below it is one vibrating string, and there are two honest ways
//! to model one:
//!
//! * a **waveguide**, two delay lines and a filter that has to realise the
//!   string's dispersion and loss, and
//! * a **modal bank**, one damped oscillator per partial.
//!
//! They are not interchangeable in character. The waveguide is cheap and its
//! low partials are exact, but the dispersion filter is a fit, and the fit is
//! only good while the string is nearly ideal; push the inharmonicity up and
//! the partials slide out of tune, which the ear hears as a detuned saw rather
//! than as a stiff string. The modal bank is exact in tuning at any stiffness,
//! costs a great deal more, and models a bounded number of partials. Which of
//! the two suits a given instrument is a musical question, so both are kept and
//! the choice is a control rather than a constant.

use crate::contact::ContactLaw;
use crate::modal::{ModalString, StringConfig as ModalConfig};
use crate::string::{StringConfig, WaveguideString};

/// Which way the string itself is modelled.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Core {
    /// Two delay lines with a dispersion allpass cascade and a loss filter.
    ///
    /// Cheap, and the partials below the fit's knee are as exact as the modal
    /// bank's.
    Waveguide,
    /// One damped oscillator per partial, so the tuning is exact by
    /// construction at any stiffness.
    #[default]
    Modal,
}

impl Core {
    /// Both cores, in menu order.
    pub const ALL: [Core; 2] = [Core::Modal, Core::Waveguide];

    /// A short human readable name.
    pub fn name(&self) -> &'static str {
        match self {
            Core::Waveguide => "Waveguide",
            Core::Modal => "Modal",
        }
    }

}

// There is deliberately no per-core gain trim here. There was one for a while,
// measured at up to six decibels on the bowed instruments, and it was covering
// for a fault in the mode bank's port equation rather than for any real
// difference between the models. With that fixed the two cores put the same
// force on the bridge at the same settings to within a decibel across the
// keyboard, and a compensation table would now only hide the next such mistake.

/// One string, modelled whichever way the engine asked for.
///
/// The two models have deliberately identical signatures, because the contact
/// solvers do not care which of them they are talking to: both present the
/// excitation point as a port whose velocity is an affine function of the force
/// applied there.
pub enum StringCore {
    /// The delay line model.
    Waveguide(Box<WaveguideString>),
    /// The mode bank model.
    Modal(Box<ModalString>),
}

impl StringCore {
    /// Build one string for the given core.
    pub fn new(core: Core, sample_rate: f32, config: &StringConfig) -> Self {
        match core {
            Core::Waveguide => {
                StringCore::Waveguide(Box::new(WaveguideString::new(sample_rate, config.clone())))
            },
            Core::Modal => StringCore::Modal(Box::new(ModalString::new(
                sample_rate,
                ModalConfig {
                    frequency: config.frequency,
                    inharmonicity: config.inharmonicity,
                    sigma0: config.sigma0,
                    sigma1: config.sigma1,
                    // The tension and the impedance are what turn the abstract
                    // string into one with a length and a mass, which is what the
                    // modal model needs and the waveguide does not: a waveguide
                    // only ever sees the round trip time.
                    tension: config.tension,
                    impedance: config.impedance,
                },
            ))),
        }
    }

    /// Which model this is.
    pub fn core(&self) -> Core {
        match self {
            StringCore::Waveguide(_) => Core::Waveguide,
            StringCore::Modal(_) => Core::Modal,
        }
    }

    /// Advance one sample and return the force the string puts on the bridge.
    #[inline]
    pub fn step(&mut self) -> f32 {
        match self {
            StringCore::Waveguide(string) => string.step(),
            StringCore::Modal(string) => string.step(),
        }
    }

    /// Set how hard a damper felt is resting on the string.
    #[inline]
    pub fn set_damper(&mut self, gain: f32) {
        match self {
            StringCore::Waveguide(string) => string.set_damper(gain),
            StringCore::Modal(string) => string.set_damper(gain),
        }
    }

    /// Bend the whole string, as a player's hand does.
    pub fn set_pitch_ratio(&mut self, ratio: f32) {
        match self {
            StringCore::Waveguide(string) => string.set_pitch_ratio(ratio),
            StringCore::Modal(string) => string.set_pitch_ratio(ratio),
        }
    }

    /// How far the pitch is currently bent.
    pub fn pitch_ratio(&self) -> f32 {
        match self {
            StringCore::Waveguide(string) => string.pitch_ratio(),
            StringCore::Modal(string) => string.pitch_ratio(),
        }
    }

    /// A running estimate of the output magnitude.
    #[inline]
    pub fn envelope(&self) -> f32 {
        match self {
            StringCore::Waveguide(string) => string.envelope(),
            StringCore::Modal(string) => string.envelope(),
        }
    }

    /// Whether the string has decayed into inaudibility.
    #[inline]
    pub fn is_quiet(&self) -> bool {
        match self {
            StringCore::Waveguide(string) => string.is_quiet(),
            StringCore::Modal(string) => string.is_quiet(),
        }
    }

    /// Release the string from a triangular displacement.
    pub fn pluck(&mut self, position: f32, amplitude: f32) {
        match self {
            StringCore::Waveguide(string) => string.pluck(position, amplitude),
            StringCore::Modal(string) => string.pluck(position, amplitude),
        }
    }

    /// Add an excitation point with its own contact law.
    pub fn add_port(&mut self, position: f32, law: Box<dyn ContactLaw>, substeps: u32) {
        match self {
            StringCore::Waveguide(string) => string.add_port(position, law, substeps),
            StringCore::Modal(string) => string.add_port(position, law, substeps),
        }
    }

    /// Tell every excitation point to let go of the string.
    pub fn release_ports(&mut self, dt: f32) {
        match self {
            StringCore::Waveguide(string) => string.release_ports(dt),
            StringCore::Modal(string) => string.release_ports(dt),
        }
    }

    /// The displacement and velocity at each excitation point.
    ///
    /// A bowed string is supposed to alternate between sticking to the bow and
    /// slipping past it, and these are how you tell whether it actually is.
    pub fn port_states(&self) -> Vec<(f32, f32)> {
        match self {
            StringCore::Waveguide(string) => string.port_states(),
            StringCore::Modal(string) => string.port_states(),
        }
    }

    /// The frequencies this string resonates at.
    pub fn partial_frequencies(&self) -> Vec<f32> {
        match self {
            StringCore::Waveguide(string) => string.partial_frequencies(),
            StringCore::Modal(string) => string.partial_frequencies(),
        }
    }

    /// The fitted dispersion coefficients, for the waveguide only.
    pub fn dispersion_coefficients(&self) -> Option<Vec<f32>> {
        match self {
            StringCore::Waveguide(string) => Some(string.dispersion_coefficients()),
            StringCore::Modal(_) => None,
        }
    }
}
