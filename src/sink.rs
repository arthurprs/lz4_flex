use alloc::vec::Vec;
use core::{mem::MaybeUninit, ptr::NonNull};

/// Sink is used as target to de/compress data into a preallocated but possibly uninitialized memory space.
/// Sink can be created from a `Vec` or a `Slice`. The new pos on the data after the operation
/// can be retrieved via `sink.pos()`.
///
/// # Safety invariants
///   - `self.output[.. self.pos]` is always initialized.
pub trait Sink {
    /// The entire buffer available (len == capacity), including uninitialized bytes.
    fn output(&mut self) -> &mut [MaybeUninit<u8>];
    /// The bytes that are considered filled.
    fn filled_slice(&self) -> &[u8];
    /// The current position (aka. len) of the the Sink.
    fn pos(&self) -> usize;
    /// The total capacity of the Sink.
    fn capacity(&self) -> usize;
    /// Forces the length of the vector to `new_pos`.
    /// The caller is responsible for ensuring all bytes up to `new_pos` are properly initialized.
    fn set_pos(&mut self, new_pos: usize);

    #[inline]
    fn advance(&mut self, by: usize) {
        self.set_pos(self.pos() + by);
    }

    /// Returns a raw ptr to the first byte of the Sink. Analogous to `[0..].as_ptr()`.
    #[inline]
    unsafe fn base_mut_ptr(&mut self) -> *mut u8 {
        self.output().as_mut_ptr() as *mut u8
    }

    /// Returns a raw ptr to the first unfilled byte of the Sink. Analogous to `[pos..].as_ptr()`.
    #[inline]
    unsafe fn pos_mut_ptr(&mut self) -> *mut u8 {
        self.output().as_mut_ptr().add(self.pos()) as *mut u8
    }

    /// Pushes a byte to the end of the Sink.
    #[inline]
    fn push(&mut self, byte: u8) {
        let pos = self.pos();
        self.output()[pos] = MaybeUninit::new(byte);
        self.advance(1);
    }

    /// Extends the Sink with `data`.
    #[inline]
    fn extend_from_slice(&mut self, data: &[u8]) {
        let pos = self.pos();
        self.output()[pos..pos + data.len()].copy_from_slice(slice_as_uninit_ref(data));
        self.advance(data.len());
    }
}

pub struct VecSink<'a> {
    /// The working slice, which may contain uninitialized bytes
    output: &'a mut [MaybeUninit<u8>],
    /// Number of bytes in start of `output` guaranteed to be initialized
    pos: usize,
    // / The `Vec` backing `output`.
    // / On Drop the sink will adjust the vec len to cover all bytes until `pos`,
    // / this includes truncating the vec!
    vec_ptr: NonNull<Vec<u8>>,
}

impl<'a> VecSink<'a> {
    /// Creates a `Sink` backed by the vec bytes at `vec[offset..vec.capacity()]`.
    /// Note that the bytes at `vec[output.len()..]` are actually uninitialized and will
    /// not be readable until written.
    /// When the `Sink` is dropped the Vec len will be adjusted to `offset` + `pos`.
    #[inline]
    pub fn new(output_vec: &'a mut Vec<u8>, offset: usize, pos: usize) -> VecSink<'a> {
        // SAFETY: Only the first `output.len` are `output` are initialized.
        // Assert that the range output[offset + pos] is all initialized data.
        let _ = &output_vec[..offset + pos];

        // For Stacked-Borrows reasons, let's derive the raw pointer first.
        let vec_ptr = NonNull::new(output_vec).unwrap();

        // SAFETY: `Vec` guarantees that `capacity` elements are available from `as_mut_ptr`.
        // Only the first `output.len` elements are actually initialized but we use a slice of `MaybeUninit` for the entire range.
        // `MaybeUninit<T>` is guaranteed to have the same layout as `T`.
        let vec_with_spare = unsafe {
            core::slice::from_raw_parts_mut(
                output_vec.as_mut_ptr() as *mut MaybeUninit<u8>,
                output_vec.capacity(),
            )
        };
        VecSink {
            output: &mut vec_with_spare[offset..],
            pos,
            vec_ptr,
        }
    }
}

pub struct SliceSink<'a> {
    /// The working slice, which may contain uninitialized bytes
    output: &'a mut [u8],
    /// Number of bytes in start of `output` guaranteed to be initialized
    pos: usize,
}

impl<'a> SliceSink<'a> {
    /// Creates a `Sink` backed by the given byte slice.
    #[inline]
    pub fn new(output: &'a mut [u8], pos: usize) -> Self {
        // SAFETY: Caller guarantees that all elements of `output` are initialized.
        let _ = &mut output[..pos]; // bounds check pos
        SliceSink { output, pos }
    }
}

impl<'a> Sink for SliceSink<'a> {
    #[inline]
    fn output(&mut self) -> &mut [MaybeUninit<u8>] {
        slice_as_uninit_mut(self.output)
    }

    #[inline]
    fn filled_slice(&self) -> &[u8] {
        &self.output[..self.pos]
    }

    #[inline]
    fn pos(&self) -> usize {
        self.pos
    }

    #[inline]
    fn capacity(&self) -> usize {
        self.output.len()
    }

    #[inline]
    fn set_pos(&mut self, new_pos: usize) {
        debug_assert!(new_pos <= self.capacity());
        self.pos = new_pos;
    }
}

impl<'a> Sink for VecSink<'a> {
    #[inline]
    fn output(&mut self) -> &mut [MaybeUninit<u8>] {
        self.output
    }

    #[inline]
    fn filled_slice(&self) -> &[u8] {
        unsafe { slice_assume_init_ref(&self.output[..self.pos]) }
    }

    #[inline]
    fn pos(&self) -> usize {
        self.pos
    }

    #[inline]
    fn capacity(&self) -> usize {
        self.output.len()
    }

    #[inline]
    fn set_pos(&mut self, new_pos: usize) {
        debug_assert!(new_pos <= self.capacity());
        self.pos = new_pos;
    }
}

impl<'a> Drop for VecSink<'a> {
    #[inline]
    fn drop(&mut self) {
        unsafe {
            // SAFETY: From from_vec_with_spare_capacity constructor we know:
            // - self.vec is guaranteed to be alive for the Sink 'a lifetime.
            // - self.output is guaranteed to be within the vec allocated memory.
            // - All bytes between vec[0..] and output[0..], if any, are initialized.
            let output_offset = (self.output.as_ptr() as *const u8)
                .offset_from(self.vec_ptr.as_ref().as_ptr())
                as usize;
            debug_assert!(output_offset + self.pos <= self.vec_ptr.as_ref().capacity());
            // SAFETY: Sink guarantees that bytes in `output[..self.pos]` are initialized.
            // Which in turn means that all bytes in `vec[..offset + self.pos]` are initialized.
            // The vec elements (u8) are Copy and Non-Drop so no other adjustments are required.
            self.vec_ptr.as_mut().set_len(output_offset + self.pos);
        }
    }
}

#[inline]
pub fn slice_as_uninit_ref(slice: &[u8]) -> &[MaybeUninit<u8>] {
    // SAFETY: `&[T]` is guaranteed to have the same layout as `&[MaybeUninit<T>]`
    unsafe { core::slice::from_raw_parts(slice.as_ptr() as *const MaybeUninit<u8>, slice.len()) }
}

#[inline]
fn slice_as_uninit_mut(slice: &mut [u8]) -> &mut [MaybeUninit<u8>] {
    // SAFETY: `&mut [T]` is guaranteed to have the same layout as `&mut [MaybeUninit<T>]`
    unsafe {
        core::slice::from_raw_parts_mut(slice.as_mut_ptr() as *mut MaybeUninit<u8>, slice.len())
    }
}

/// Assuming all the elements are initialized, get a slice to them.
/// # Safety
/// It is up to the caller to guarantee that the MaybeUninit<T> elements really are in an initialized state.
/// Calling this when the content is not yet fully initialized causes undefined behavior.
#[inline]
unsafe fn slice_assume_init_ref(slice: &[MaybeUninit<u8>]) -> &[u8] {
    core::slice::from_raw_parts(slice.as_ptr() as *const u8, slice.len())
}

#[cfg(test)]
mod tests {
    use crate::sink::{SliceSink, VecSink};

    use super::Sink;

    #[test]
    fn test_sink_slice() {
        let mut data = Vec::new();
        data.resize(5, 0);
        let mut sink = SliceSink::new(&mut data, 1);
        assert_eq!(sink.pos(), 1);
        assert_eq!(sink.capacity(), 5);
        assert_eq!(sink.filled_slice(), &[0]);
        sink.extend_from_slice(&[1, 2, 3]);
        assert_eq!(sink.pos(), 4);
        assert_eq!(sink.filled_slice(), &[0, 1, 2, 3]);
    }

    #[test]
    fn test_sink_vec() {
        let mut data = Vec::with_capacity(5);
        data.push(255); // not visible to the sink
        data.push(0);
        let mut sink = VecSink::new(&mut data, 1, 1);
        assert_eq!(sink.pos(), 1);
        assert_eq!(sink.capacity(), 4);
        assert_eq!(sink.filled_slice(), &[0]);
        sink.extend_from_slice(&[1, 2, 3]);
        assert_eq!(sink.pos(), 4);
        assert_eq!(sink.filled_slice(), &[0, 1, 2, 3]);
        drop(sink);
        assert_eq!(data.as_slice(), &[255, 0, 1, 2, 3]);
    }
}
