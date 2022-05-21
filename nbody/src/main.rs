use std::{fs::File, io::BufReader, error::Error};

use clap::Parser;
use serde::{Deserialize, Serialize};
use ndarray::{Array, Array1, Array2};

mod gravity;
mod integrator;
use crate::integrator::LeapfrogIntegrator;

#[derive(Parser, Debug)]
#[clap(author, version, about, long_about = None)]
struct Args {
    /// Path to the input file giving the bodies' initial positions and velocities.
    bodies_path: std::path::PathBuf,
    /// Path to write output file.
    output_path: std::path::PathBuf,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Body {
    pub name: String,
    pub mass_kg: f64,
    pub position_init_m: [f64; 3],
    pub velocity_init_m_per_s: [f64; 3],
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct NBodyInput {
    pub bodies: Vec<Body>,
    pub timestep_s: f64,
    pub n_steps: usize,
}

/// Write the simulation results to a comma-separated value file
fn write_results(
    wtr: &mut csv::Writer<File>,
    input: &NBodyInput,
    t: &Array1<f64>,
    r: &Array2<f64>,
    v: &Array2<f64>,
    a: &Array2<f64>,
) -> Result<(), Box<dyn Error>> {
    let n_steps = t.len();
    let n_bodies = input.bodies.len();

    // Check dimensions of arguments
    assert_eq!(r.shape(), [n_steps, 3 * n_bodies]);
    assert_eq!(v.shape(), [n_steps, 3 * n_bodies]);
    assert_eq!(a.shape(), [n_steps, 3 * n_bodies]);

    // Write headers
    let mut headers = vec!["Time [s]".to_string()];
    for b in &input.bodies {
        for k in 0..3 {
            headers.push(format!("{} position {} [m]", b.name, k));
        }
        for k in 0..3 {
            headers.push(format!("{} velocity {} [m s^-1]", b.name, k));
        }
        for k in 0..3 {
            headers.push(format!("{} acceleration {} [m s^-2]", b.name, k));
        }
    }
    wtr.write_record(headers)?;

    // Write data
    const NUM_COLUMNS_PER_BODY: usize = 9;
    let mut row: Array1<f64> = Array::zeros(1 + NUM_COLUMNS_PER_BODY * n_bodies);
    for i in 0..n_steps {
        row[0] = t[i];
        for j in 0..n_bodies {
            // Write position to the row
            for k in 0..3 {
                row[1 + NUM_COLUMNS_PER_BODY * j + k] = r[[i, 3 * j + k]];
            }
            // Write velocity to the row
            for k in 0..3 {
                row[1 + NUM_COLUMNS_PER_BODY * j + 3 + k] = v[[i, 3 * j + k]];
            }
            // Write acceleration to the row
            for k in 0..3 {
                row[1 + NUM_COLUMNS_PER_BODY * j + 6 + k] = a[[i, 3 * j + k]];
            }
        }
        wtr.serialize(row.to_vec())?;
    }

    Ok(())
}

fn main() {
    let args = Args::parse();

    let bodies_file = File::open(args.bodies_path).expect("Failed to open bodies_path.");
    let reader = BufReader::new(bodies_file);
    let input: NBodyInput = serde_json::from_reader(reader).expect("Failed to parse bodies file.");
    
    let mut wtr = csv::WriterBuilder::new()
        .from_path(args.output_path)
        .expect("Failed to open output_file for writing.");

    let mut intgr = LeapfrogIntegrator::new(&input);

    intgr.integrate();

    write_results(&mut wtr, &input, &intgr.t, &intgr.r, &intgr.v, &intgr.a)
        .expect("Failed to write results.");
}
