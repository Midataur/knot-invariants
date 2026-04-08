use std::error::Error;
use kdam::tqdm;
use clap::Parser;
use std::fs;
use std::env::args;
use std::thread;
use std::sync::mpsc;
use csv::WriterBuilder;

mod utilities;
mod dynnikov;
mod args;

use crate::utilities::get_random_word;

fn main() -> Result<(), Box<dyn Error>> {
    let mut args = args::Args::parse();

    // make sure braid_count_to_scale_to is defined appropriately
    if utilities::wasnt_defined(args.braid_count_to_scale_to) {
        args.braid_count_to_scale_to = args.braid_count;
    }

    // check if inputs are good; if not, raise an error
    utilities::check_inputs(&args);

    // work out how many files to gen
    let number_of_files_needed = std::cmp::max(1, args.number_of_files_to_gen);

    // generate the files
    for file_index in tqdm!(0..number_of_files_needed, desc="Files generated") {
        let mut words: Vec<Vec<i64>> = Vec::new();
        let mut coords: Vec<Vec<i64>> = Vec::new();

        println!("Generating data...");

        let (sender, receiver) = mpsc::channel();

        for _ in 0..args.threads {
            let sender_clone = sender.clone();
            let args_clone: args::Args = args.clone();

            // get the initial configuration template
            let initial = utilities::get_initial(args.braid_count_to_scale_to);

            // Spawn a thread
            thread::spawn(move || {
                for _ in 0..(args.dataset_size/args.threads) {
                    // create sequence
                    let word = get_random_word(&args_clone);
                    let coord = dynnikov::word_action(&initial, &word);
                    
                    sender_clone.send((word, coord)).unwrap();
                }
            });
        }

        // main thread receives results from worker threads
        for _ in tqdm!(
            0..args.dataset_size,
            leave=false,
            position=1
        ) {
            // add a new datapoint
            let (word, coord) = receiver.recv().unwrap();
            words.push(word);
            coords.push(coord);
        }

        let mut filename = args.filename.to_string();

        if args.start_index > -1 {
            filename += &((args.start_index + file_index).to_string())
        }

        // extracts the directory from the filename
        let directory_parts: Vec<&str> = filename.split('/').collect();
        let directory = directory_parts.split_last().unwrap().1.join("/");

        // create the directory if needed
        let _ = fs::create_dir_all(directory);

        // write the data to the file
        if args.number_of_files_to_gen == -1 {
            println!("Writing data to file...");
        }

        let mut writer = WriterBuilder::new().from_path(
            filename.clone() + "_inputs.csv"
        )?;

        for row in tqdm!(words.iter()) {
            let string_row: Vec<String> = row.into_iter().map(|value| value.to_string()).collect();
            let _ = writer.write_record(string_row);
        }

        // flush the writer to ensure all data is written to the file
        writer.flush()?;

        // write the coordinate data
        let mut writer = WriterBuilder::new().from_path(
            filename.clone() + "_targets.csv"
        )?;

        for row in tqdm!(coords.iter()) {
            let string_row: Vec<String> = row.into_iter().map(|value| value.to_string()).collect();
            let _ = writer.write_record(string_row);
        }
    
        // Flush the writer to ensure all data is written to the file
        writer.flush()?;
        
    }

    Ok(())
}
