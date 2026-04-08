use std::cmp::max;
use std::cmp::min;
use rand::prelude::*;
use crate::args;

// finds x^+
pub fn pos(x: i64) -> i64 {
    return max(x, 0);
}

// finds x^-
pub fn neg(x: i64) -> i64 {
    return min(x, 0);
}

pub fn get_initial(n: i64) -> Vec<i64> {
    let mut a_part: Vec<i64> = (0..(n-1)).map(|_| 0).collect();
    let b_part: Vec<i64> = (0..(n-1)).map(|_| -1).collect();

    a_part.extend(b_part.iter().cloned());

    return a_part;
}

/// Generate a random word.
pub fn get_random_word(args: &args::Args) -> Vec<i64> {
    let mut rng = rand::rng();
    let upper_bound = args.braid_count_to_scale_to;

    return (
        0..args.max_word_length
    ).map(
        |_| rng.random_range((-upper_bound+1)..upper_bound)
    ).collect();
}

pub fn check_inputs(args: &args::Args) {
    assert!(
        args.dataset_size % args.threads == 0, 
        "\nThreads must divide dataset size\n"
    );

    assert!(
        !(args.braid_count > args.braid_count_to_scale_to),
        "\nbraid_count can't be larger than braid_count_to_scale_to\n"
    );

    assert!(
        !(args.braid_count_to_scale_to <= 0),
        "\nbraid_count_to_scale_to must be at least 0\n"
    );
}

pub fn wasnt_defined(x: i64) -> bool {
    return x <= -1;
}