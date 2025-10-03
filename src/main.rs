use i_am_dsp::real_time_demo::DspDemo;

fn main() {
	DspDemo::new().unwrap().run(Default::default()).unwrap();
}