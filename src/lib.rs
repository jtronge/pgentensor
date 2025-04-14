use std::os::raw::c_int;

#[unsafe(no_mangle)]
pub extern "C" fn gentensor(
    dims: *const usize,
    seed: usize,
    size: c_int,
    rank: c_int,
) {
    // TODO
}
