use clap::Parser;

/// Program to generate data
#[derive(Parser, Debug, Clone)]
#[command(about, long_about = None)]
pub struct Args {
    /// Maximum length of a sequence.
    #[arg(short, long, default_value_t = 10)]
    pub max_word_length: i64,

    /// Braid count to use.
    #[arg(short, long, default_value_t = 3)]
    pub braid_count: i64,

    /// Dataset size.
    #[arg(short, long, default_value_t = 100_000)]
    pub dataset_size: i64,

    /// Filename to write to.
    /// Don't include file extensions.
    #[arg(short, long, default_value_t = String::from("data"))]
    pub filename: String,

    /// Number of threads to use.
    /// Make sure that threads number divides dataset size.
    #[arg(short, long, default_value_t = 1)]
    pub threads: i64,

    /// Maximum hypothetical braid count that can be scaled to later.
    #[arg(short='M', long, default_value_t = -1)]
    pub braid_count_to_scale_to: i64,

    /// Number of files to generate.
    #[arg(short='N', long, default_value_t = 1)]
    pub number_of_files_to_gen: i64,

    /// Starting number.
    /// If this is greater than -1, files will be numbered starting from this number.
    #[arg(short='s', long, default_value_t = -1)]
    pub start_index: i64,
}