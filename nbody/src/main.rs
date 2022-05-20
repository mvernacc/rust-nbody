use std::{fs::File, io::BufReader};

use clap::Parser;
use serde::{Deserialize, Serialize};

mod gravity;
mod integrator;
use crate::integrator::LeapfrogIntegrator;

#[derive(Parser, Debug)]
#[clap(author, version, about, long_about = None)]
struct Args {
    /// Path to the input file giving the bodies' initial positions and velocities.
    bodies_path: std::path::PathBuf,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Body {
    pub name: String,
    pub mass_kg: f64,
    pub position_init_m: [f64; 3],
    pub velocity_init_m_per_s: [f64; 3],
}

#[derive(Serialize, Deserialize, Debug)]
pub struct NBodyInput {
    pub bodies: Vec<Body>,
    pub timestep_s: f64,
    pub n_steps: usize,
}


fn main() {
    let args = Args::parse();

    let bodies_file = File::open(args.bodies_path).expect("Failed to open bodies_path.");
    let reader = BufReader::new(bodies_file);
    let input: NBodyInput = serde_json::from_reader(reader).expect("Failed to parse bodies file.");
    println!("{:?}", &input);

    let mut intgr = LeapfrogIntegrator::new(&input);

    intgr.integrate();

    println!("{:?}", intgr.v);

}
