//! Element loops over large inputs, split across threads.

use rayon::prelude::*;

/// The elements one thread takes at a time; an input of at most this many
/// runs on the calling thread.
pub const GRAIN: usize = 1 << 16;

/// Calls `f` on matching pieces of `input` and `output`, `per_in` and
/// `per_out` items per element, in parallel when the input is large.
pub fn zip<I: Sync, O: Send>(
    input: &[I],
    per_in: usize,
    output: &mut [O],
    per_out: usize,
    f: impl Fn(&[I], &mut [O]) + Sync,
) {
    if input.len() <= GRAIN * per_in {
        return f(input, output);
    }
    input
        .par_chunks(GRAIN * per_in)
        .zip(output.par_chunks_mut(GRAIN * per_out))
        .for_each(|(i, o)| f(i, o));
}
