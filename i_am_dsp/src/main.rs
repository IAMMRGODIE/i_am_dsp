#[cfg(feature = "standalone")]
use i_am_dsp::real_time_demo::DspDemo;

fn main() {
	#[cfg(feature = "standalone")]
	DspDemo::new(None).unwrap().run(Default::default()).unwrap();
	#[cfg(not(feature = "standalone"))]
	println!("`Standalone` feature disabled, nothing to run");
}