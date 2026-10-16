//! The small effects, as plug-ins.
//!
//! One library, several plug-ins: a host that scans this file finds each of them as a
//! plug-in of its own, and the bundle they ship in is one binary. Nothing here draws
//! a control or places one -- each effect says what its parameters are and where they
//! belong, and the faceplate builds the rest.
//!
//! ```text
//! cargo build -p i_am_effects
//! cargo bundle -p i_am_effects --features vst3
//! ```

use i_am_dsp::prelude::{Crusher as CrusherDsp, HardClipper as DriveDsp, Lowpass as FilterDsp};
use i_am_plugin::export_clap;

pub mod processor;

// The three plug-ins this library exports.
//
// Each line is the whole of what the product knows about itself: which effect, how
// to build it for a sample rate, what the host should call it, and how big its
// window wants to be.
effect_plugin!(
	Crusher,
	CrusherDsp<2>,
	CrusherDsp::<2>::new,
	"iamdsp.effects.crusher",
	"I Am Crusher",
	(620.0, 380.0)
);

effect_plugin!(
	Drive,
	DriveDsp,
	|_rate| DriveDsp::new(0.8, 1.0),
	"iamdsp.effects.drive",
	"I Am Drive",
	(420.0, 300.0)
);

effect_plugin!(
	Filter,
	FilterDsp<2>,
	|rate| FilterDsp::<2>::new(rate, 1_000.0, 0.707),
	"iamdsp.effects.filter",
	"I Am Filter",
	(460.0, 320.0)
);

// One call, one library, three plug-ins.
export_clap!(Crusher, Drive, Filter);
