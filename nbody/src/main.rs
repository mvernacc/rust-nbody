use std::{fs::File, io::BufReader};

use clap::Parser;
use ndarray::{Array, Array1};
use serde::{Deserialize, Serialize};

mod integrator;
use crate::integrator::LeapfrogIntegrator;

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
    dt_s: f64,
    n_steps: usize,
}


fn main() {
    let args = Args::parse();

    let bodies_file = File::open(args.bodies_path).expect("Failed to open bodies_path.");
    let reader = BufReader::new(bodies_file);
    let input: NBodyInput = serde_json::from_reader(reader).expect("Failed to parse bodies file.");
    println!("{:?}", &input);

    let n_bodies = input.bodies.len();

    let mut masses: Array1<f64> = Array::zeros(n_bodies);
    let mut x_init: Array1<f64> = Array::zeros(3 * n_bodies);
    let mut v_init: Array1<f64> = Array::zeros(3 * n_bodies);

    for (i, b) in input.bodies.iter().enumerate() {
        masses[i] = b.mass_kg;
        for dim in 0..3 {
            x_init[3 * i + dim] = b.position_init_m[dim];
            v_init[3 * i + dim] = b.velocity_init_m_per_s[dim];
        }
    }

    let mut intgr = LeapfrogIntegrator::new(
        input.dt_s,
        input.n_steps,
        &masses,
        &x_init,
        &v_init,
    );

    intgr.integrate();

    println!("{:?}", intgr.v);

}

// fn calc_grav_force(mass_1: f64, mass_2: f64, r_1: [f64; 3], r_2: [f64; 3]) -> [f64; 3] {

// }
