//! Synthetic tensor generator.
//!
//! Work based on Torun et al. A Sparse Tensor Generator with Efficient Feature
//! Extraction. 2025.
use std::path::Path;
use std::io::prelude::*;
use std::io::BufWriter;
use std::collections::HashSet;
use clap::Parser;
use rand::prelude::*;
use rand_distr::{Normal, Uniform};

struct TensorOptions {
    /// Tensor dimensions (3-way tensors only for now)
    dims: Vec<usize>,
    /// Density of the tensor
    nnz_density: f64,
    /// Fiber density
    fiber_density: f64,
    /// Coefficient of variation of fibers per slice
    cv_fibers_per_slice: f64,
    /// Coefficient of variation of nonzeros per fiber
    cv_nonzeros_per_fiber: f64,
    // TODO: Deal with imbalance
    // /// Imablance of fibers per slice
    // imbal_fiber_per_slice: f64,
    // /// Imbalance of nonzeros per slice
    // imbal_nonzeros_per_fiber: f64,
}

/// Return n indices each uniformly distributed from [0, limit)
fn randinds<R: Rng>(n: usize, limit: usize, rng: &mut R) -> Vec<usize> {
    assert!(limit > 0);
    let distr = Uniform::new(0, limit - 1).expect("failed to create uniform distribution");
    (0..n).map(|_| distr.sample(rng)).collect()
}

/// Implementation of the distribute function from the "A Sparse Tensor Generator
/// with Efficient Feature Extraction".
///
/// n: total number of counts to generate
/// mean: the mean of the count expected
/// std_dev: standard deviation of the count
/// max: the maximum count value possible
/// index_limit: for each count, generates count indices uniformly in range [0, limit-1]
/// rng: random number generator
fn distribute<R: Rng>(
    n: usize,
    mean: f64,
    std_dev: f64,
    max: usize,
    index_limit: usize,
    rng: &mut R,
) -> (Vec<usize>, Vec<Vec<usize>>) {
    // Check whether the normal distribution could generate many negative values
    let use_normal = mean > (3.0 * std_dev);
    let distr = if use_normal {
        Normal::new(mean, std_dev)
    } else {
        // Use log-normal only if there is potential for a lot of negative values
        let mean_log_norm = (mean * mean / (mean * mean + std_dev * std_dev).sqrt()).ln();
        let std_dev_log_norm = (1.0 + std_dev * std_dev / (mean * mean)).ln().sqrt();
        Normal::new(mean_log_norm, std_dev_log_norm)
    };
    let distr = distr.expect("failed to create distribution");

    // Generate the counts
    let mut counts = vec![];
    for _ in 0..n {
        if use_normal {
            // Use a normal distribution
            counts.push(distr.sample(rng) as usize);
        } else {
            // Use a log-normal distribution
            counts.push(distr.sample(rng).exp() as usize);
        }
    }

    // Compare the computed mean with desired mean and scale if it doesn't match exactly
    let total: usize = counts.iter().sum();
    let ratio = total as f64 / n as f64;
    if ratio < 0.95 || ratio > 1.05 {
        for count in &mut counts {
            *count = ((*count as f64) * ratio) as usize;
        }
    }

    // Now generate random indices
    let mut inds = vec![];
    for count in counts.iter_mut() {
        *count = std::cmp::min(*count, max);
        *count = std::cmp::max(*count, 1);
        // Create an array of size counts[i] all in the range [1, index_limit] ---
        // this is done with a uniform distribution here
        inds.push(randinds(*count, index_limit, rng));
    }

    (counts, inds)
}

/// Generate a tensor based on the input metrics.
///
/// Based on the following paper:
///
/// Torun et al. A Sparse Tensor Generator with Efficient Feature Extraction. 2025.
fn gentensor<P: AsRef<Path>>(tensor_fname: P, tensor_opts: TensorOptions) {
    let nnz = (tensor_opts.nnz_density * (tensor_opts.dims[0] * tensor_opts.dims[1]
                                          * tensor_opts.dims[2]) as f64) as usize;
    // Focus on generating mode-(2, 3) slices, that is slices where i is fixed
    // and the other modes vary (indexing of X(i, :, :)).
    let slice_count = tensor_opts.dims[0];
    assert!(nnz > slice_count);

    // This focuses on the mode-2 fibers, that is fibers X(i, :, k), where i and
    // k are fixed, but the middle index j can vary. Each of these fibers can
    // therefore have most tensor_opts.dims[1] nonzeros and there are
    // (tensor_opts.dims[0] * tensor_opts.dims[2]) possible tensors total.
    let nonzero_fiber_count = (tensor_opts.fiber_density
                               * (slice_count * tensor_opts.dims[2]) as f64) as usize;
    let mean_fibers_per_slice = nonzero_fiber_count as f64 / slice_count as f64;
    let std_dev_fibers_per_slice = tensor_opts.cv_fibers_per_slice * mean_fibers_per_slice;
    // We can have a maximum of tensor_opts.dims[2] fibers per slice (can be
    // thought of as columns of the tensor_opts.dims[1] x tensor_opts.dims[2] matrix X(i, :, :)).
    let max_fibers_per_slice = tensor_opts.dims[2];
    let max_nonzeros_per_fiber = tensor_opts.dims[1];

    // Choose random indices for the slices
    // TODO: We need a deterministic and portable RNG here (see
    // https://rust-random.github.io/book/crate-reprod.html#crate-versions).
    // It looks like ChaCha20Rng could be useful (see
    // https://rust-random.github.io/book/guide-seeding.html#the-seed-type)
    let mut rng = rand::rng();
    // Now we need to carefully choose the number of nonzero fibers per slices.
    // This is slightly different from the original GenTensor paper, since here
    // I'm assuming that we never have non-empty slices (which is in practice
    // what SPLATT expects).
    let (count_fibers_per_slice, fiber_indices) = distribute(
        slice_count,
        mean_fibers_per_slice,
        std_dev_fibers_per_slice,
        max_fibers_per_slice,
        max_nonzeros_per_fiber,
        &mut rng
    );
    let true_nonzero_fiber_count: usize = count_fibers_per_slice.iter().sum();

    // Compute nonzeros per fiber
    let mean_nonzeros_per_fiber = nnz as f64 / nonzero_fiber_count as f64;
    let std_dev_nonzeros_per_fiber = tensor_opts.cv_nonzeros_per_fiber * mean_nonzeros_per_fiber;
    // What this is actually generating here: count_nonzeros_per_fiber should be
    // a an array of length true_nonzero_fiber_count with counts distributed
    // according to mean_nonzeros_per_fiber and std_dev_nonzeros_per_fiber
    // (with a max of max_nonzeros_per_fiber). For each count generated, that is
    // for each nonzero fiber, nonzero_indices generates uniform indices on
    // [0, max_fibers_per_slice-1]; NOTE: this doesn't correspond exactly as
    // you would think to each fiber.
    let (count_nonzeros_per_fiber, nonzero_indices) = distribute(
        true_nonzero_fiber_count,
        mean_nonzeros_per_fiber,
        std_dev_nonzeros_per_fiber,
        max_nonzeros_per_fiber,
        max_fibers_per_slice,
        &mut rng,
    );

    let f = std::fs::File::create(tensor_fname).expect("failed to create file");
    let mut tensor_file = BufWriter::new(f);
    let value_distr = Uniform::new(0.0, 1.0).expect("failed to create uniform distribution for tensor values");
    // Iterate over all slices
    let mut fiber_idx = 0;
    let mut total_nnz = 0;
    for i in 0..slice_count {
        let mut slice_coords = HashSet::new();
        // Iterate over all fibers of this slice
        for j in 0..count_fibers_per_slice[i] {
            for k in 0..count_nonzeros_per_fiber[fiber_idx] {
                let value: f64 = value_distr.sample(&mut rng);
                let co = (fiber_indices[i][j], nonzero_indices[fiber_idx][k]);
                // Skip duplicate coordinates
                if slice_coords.contains(&co) {
                    continue;
                }
                writeln!(&mut tensor_file, "{} {} {} {}", i + 1, co.0 + 1, co.1 + 1, value)
                    .expect("failed to write tensor entry");
                slice_coords.insert(co);
            }
            fiber_idx += 1;
        }
        total_nnz += slice_coords.len();
    }
    println!("requested nonzero count: {}", nnz);
    println!("generated nonzero count: {}", total_nnz);
}

/// Synthetic tensor generator tool
#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    /// File name or path to output tensor
    #[arg(short, long, required = true)]
    fname: String,

    /// Dimensions in form {num}x{num}x...
    #[arg(short, long, required = true)]
    dims: String,

    /// Density of the tensor
    #[arg(long, required = true)]
    density: f64,

    /// Density of fibers in the tensor
    #[arg(long, required = true)]
    fiber_density: f64,

    /// Coefficient of variation of number of fibers per slice
    #[arg(long, required = true)]
    cv_fiber_slice: f64,

    /// Coefficient of variation of number of nonzeros per fiber
    #[arg(long, required = true)]
    cv_nonzero_fiber: f64,
}

fn main() {
    let args = Args::parse();

    let dims: Vec<usize> = args.dims
        .split('x')
        .map(|v| usize::from_str_radix(v, 10).expect("invalid dimensions specified"))
        .collect();
    // Only supporting 3-way tensors for right now
    assert_eq!(dims.len(), 3);
    gentensor(args.fname, TensorOptions {
        dims,
        nnz_density: args.density,
        fiber_density: args.fiber_density,
        cv_fibers_per_slice: args.cv_fiber_slice,
        cv_nonzeros_per_fiber: args.cv_nonzero_fiber,
    });
}
