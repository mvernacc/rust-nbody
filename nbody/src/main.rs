use std::{fs::File, io::BufReader};

use clap::Parser;
use serde::{Deserialize, Serialize};

#[derive(Parser, Debug)]
#[clap(author, version, about, long_about = None)]
struct Args {
    /// Path to the input file giving the bodies' initial positions and velocities.
    bodies_path: std::path::PathBuf,
}

#[derive(Serialize, Deserialize, Debug)]
struct Body {
    name: String,
    mass_kg: f64,
    position_init_m: [f64; 3],
    velocity_init_m_per_s: [f64; 3],
}

#[derive(Serialize, Deserialize, Debug)]
struct NBodyInput {
    bodies: Vec<Body>,
}


fn main() {
    let args = Args::parse();

    let bodies_file = File::open(args.bodies_path).expect("Failed to open bodies_path.");
    let reader = BufReader::new(bodies_file);
    let input: NBodyInput = serde_json::from_reader(reader).expect("Failed to parse bodies file.");
    println!("{:?}", &input);
}
