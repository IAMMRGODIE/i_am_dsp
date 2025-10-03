//! Ring buffer implementation.

use std::ops::IndexMut;
use std::ops::Index;
use std::ops::Range;

/// A ring buffer implementation.
pub struct RingBuffer<T: Default> {
	capacity: usize,
	current_pos: usize,
	buffer: Vec<T>
}

impl<T: Default + Clone> RingBuffer<T> {
	/// Extends the buffer with the given length by cloning the default value.
	/// 
	/// Returns `false` if the buffer's capacity is less than the given length.
	pub fn extend_defaults(&mut self, len: usize) -> bool {
		if self.capacity() < len {
			return false
		}

		for _ in 0..len {
			self.push(T::default());
		}

		true
	}
	
	/// Resizes the buffer to the given capacity.
	pub fn resize(&mut self, new_capacity: usize) {
		self.capacity = new_capacity;
		self.buffer.resize(new_capacity, T::default());
		self.current_pos = 0;
	}
}

impl<T: Default> RingBuffer<T> {
	/// Creates a new ring buffer with the given capacity.
	pub fn new(capacity: usize) -> Self {
		Self {
			capacity,
			current_pos: 0,
			buffer: (0..capacity).map(|_| T::default()).collect()
		}
	}

	/// Returns the capacity of the buffer.
	pub fn capacity(&self) -> usize {
		self.capacity
	}

	/// Pushes a new value to the end of the buffer.
	pub fn push(&mut self, value: T) {
		self.buffer[self.current_pos] = value;
		self.current_pos = (self.current_pos + 1) % self.capacity;
	}

	/// Clears the buffer by setting all values to their default value.
	pub fn clear(&mut self) {
		for i in 0..self.capacity {
			self.buffer[i] = T::default();
		}
		self.current_pos = 0;
	}

	/// Returns the current position of the buffer.
	pub fn current_pos(&self) -> usize {
		self.current_pos
	}

	/// Returns a reference to the underlying buffer.
	/// 
	/// Note: This buffer's start position may not be same with [`Self::current_pos`]
	/// unless [`Self::current_pos`] returns 0.
	pub fn underlying_buffer(&self) -> &[T] {
		&self.buffer
	}

	/// Returns an iterator over the buffer's values in the given range.
	pub fn range(&self, range: Range<usize>) -> RangendRingBufferIterator<T> {
		RangendRingBufferIterator {
			buffer: &self.buffer,
			current_pos: self.current_pos,
			start: range.start,
			end: range.end
		}
	}
}

impl<T: Default> Index<usize> for RingBuffer<T> {
	type Output = T;

	fn index(&self, idx: usize) -> &T {
		&self.buffer[(idx + self.current_pos) % self.capacity]
	}
}

// impl<T: Default> Index<usize> for &RingBuffer<T> {
// 	type Output = T;

// 	fn index(&self, idx: usize) -> &T {
// 		&self.buffer[(idx + self.current_pos) % self.capacity]
// 	}
// }

impl<T: Default> Index<isize> for RingBuffer<T> {
	type Output = T;

	fn index(&self, idx: isize) -> &T {
		let mut idx = idx % self.capacity as isize;
		if idx < 0 {
			idx += self.capacity as isize;
		}
		&self[idx as usize]
	}
}

impl<T: Default> IndexMut<usize> for RingBuffer<T> {
	fn index_mut(&mut self, idx: usize) -> &mut T {
		&mut self.buffer[(idx + self.current_pos) % self.capacity]
	}
}

impl<T: Default> IndexMut<isize> for RingBuffer<T> {
	fn index_mut(&mut self, idx: isize) -> &mut T {
		let mut idx = idx % self.capacity as isize;
		if idx < 0 {
			idx += self.capacity as isize;
		}
		&mut self[idx as usize]
	}
}

impl<T: Default> Index<i32> for RingBuffer<T> {
	type Output = T;

	fn index(&self, idx: i32) -> &T {
		let mut idx = idx % self.capacity as i32;
		if idx < 0 {
			idx += self.capacity as i32;
		}
		&self[idx as usize]
	}
}
impl<T: Default> IndexMut<i32> for RingBuffer<T> {
	fn index_mut(&mut self, idx: i32) -> &mut T {
		let mut idx = idx % self.capacity as i32;
		if idx < 0 {
			idx += self.capacity as i32;
		}
		&mut self[idx as usize]
	}
}

impl<T: Default> Default for RingBuffer<T> {
	fn default() -> Self {
		Self::new(0)
	}
}

impl<T: Default + Clone> Clone for RingBuffer<T> {
	fn clone(&self) -> Self {
		Self {
			capacity: self.capacity,
			current_pos: self.current_pos,
			buffer: self.buffer.clone()
		}
	}
}

/// An iterator over the values in a ring buffer.
pub struct RangendRingBufferIterator<'a, T: Default> {
	buffer: &'a [T],
	current_pos: usize,
	start: usize,
	end: usize
}

impl<'a, T: Default> Iterator for RangendRingBufferIterator<'a, T> {
	type Item = &'a T;

	fn next(&mut self) -> Option<Self::Item> {
		if self.start == self.end {
			None
		}else {
			let idx = (self.start + self.current_pos) % self.buffer.len();
			let result = &self.buffer[idx];
			self.start += 1;
			Some(result)
		}
	}
}