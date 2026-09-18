//! Delay lines used by the waveguide string.

/// A circular delay line with fractional reads.
///
/// The line is written with [`DelayLine::push`] and read with
/// [`DelayLine::tap`], where the tap delay is measured backwards from the most
/// recent push. All reads are linear interpolations, so fractional delays (a
/// port at an arbitrary position, or a fractional propagation delay) are exact
/// to first order.
#[derive(Clone, Debug)]
pub(crate) struct DelayLine {
    buffer: Vec<f32>,
    mask: usize,
    length: f32,
    write: usize,
}

impl DelayLine {
    /// Create a line whose nominal length is `length` samples.
    pub(crate) fn new(length: f32) -> Self {
        let mut line = Self {
            buffer: Vec::new(),
            mask: 0,
            length: 0.0,
            write: 0,
        };
        line.set_length(length);
        line
    }

    /// Change the nominal length, reallocating only when it no longer fits.
    pub(crate) fn set_length(&mut self, length: f32) {
        let length = length.max(0.0);
        let capacity = (length.ceil() as usize + 2).next_power_of_two();
        if capacity != self.buffer.len() {
            self.buffer = vec![0.0; capacity];
            self.mask = capacity - 1;
            self.write = 0;
        }
        self.length = length;
    }

    /// Zero the contents.
    pub(crate) fn clear(&mut self) {
        self.buffer.fill(0.0);
        self.write = 0;
    }

    /// Push one sample into the line.
    #[inline]
    pub(crate) fn push(&mut self, value: f32) {
        self.buffer[self.write & self.mask] = value;
        self.write += 1;
    }

    /// Read `delay` samples behind the most recent push.
    #[inline]
    pub(crate) fn tap(&self, delay: f32) -> f32 {
        let delay = delay.max(0.0);
        let index = delay as usize;
        let frac = delay - index as f32;
        let a = self.buffer[self.write.wrapping_sub(1).wrapping_sub(index) & self.mask];
        if frac <= 0.0 {
            return a;
        }
        let b = self.buffer[self.write.wrapping_sub(2).wrapping_sub(index) & self.mask];
        a + (b - a) * frac
    }

    /// Read the wave that has travelled exactly `travelled` samples.
    ///
    /// [DelayLine::tap] measures backwards from the last push, and the waveguide
    /// loop reads its delay lines *before* it writes this sample back into them.
    /// A wave pushed at time `s` is therefore `tap(d)` at time `s + d + 1`, and
    /// forgetting that off-by-one detunes the whole string by a few cents.
    #[inline]
    pub(crate) fn arrival(&self, travelled: f32) -> f32 {
        self.tap(travelled - 1.0)
    }

    /// Fill the line with a travelling wave shape.
    ///
    /// `shape(d)` receives the delay in samples behind the write head, which is
    /// exactly the distance the wave has travelled from the end it was launched
    /// at. The line is left ready for the next [`DelayLine::push`].
    pub(crate) fn load(&mut self, shape: impl Fn(f32) -> f32) {
        let steps = self.length.ceil() as usize + 1;
        self.write = steps + 1;
        for i in 0..=steps {
            let value = shape(i as f32);
            self.buffer[self.write.wrapping_sub(1).wrapping_sub(i) & self.mask] = value;
        }
    }
}
