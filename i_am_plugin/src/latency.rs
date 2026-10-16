//! latency

use crate::shared::*;

/// A struct to hold a [`Processor`] for the plugin audio thread.
/// 
/// Public mainly for `export_clap` macro. At most case, you don't need to use it directly.
/// 
/// # Threading
/// 
/// Watches the plugin's latency, which the audio thread knows and only the main thread may
/// report.
///
/// A synced delay takes its length from the tempo, so its latency moves with the transport.
/// CLAP wants the host told when it does, and the host may only be called from the main
/// thread, so the audio thread publishes what it is running at and the main thread asks
/// whether there is news.
#[derive(Debug)]
pub(crate) struct LatencyWatch {
	shared: Arc<AtomicU32>,
	/// What the main thread last told the host, so it is only told when it changes.
	reported: u32,
}

/// The audio thread's half of a [`LatencyWatch`].
#[derive(Debug, Clone)]
pub(crate) struct LatencyPublisher {
	shared: Arc<AtomicU32>,
}

impl LatencyWatch {
	pub(crate) fn new(latency: u32) -> Self {
		Self {
			shared: Arc::new(AtomicU32::new(latency)),
			reported: latency,
		}
	}

	pub(crate) fn publisher(&self) -> LatencyPublisher {
		LatencyPublisher {
			shared: self.shared.clone(),
		}
	}

	/// The latency the plugin is running at, as last published.
	pub(crate) fn current(&self) -> u32 {
		self.shared.load(Ordering::Relaxed)
	}

	/// The new latency, if the audio thread has moved since the host was last told.
	pub(crate) fn take_change(&mut self) -> Option<u32> {
		let current = self.current();

		if current == self.reported {
			None
		}else {
			self.reported = current;
			Some(current)
		}
	}
}

impl LatencyPublisher {
	/// Called by the audio thread with the latency it just ran at.
	pub(crate) fn publish(&self, latency: u32) {
		self.shared.store(latency, Ordering::Relaxed);
	}
}
