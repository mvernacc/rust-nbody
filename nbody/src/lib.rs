use serde::{Deserialize, Serialize};

pub mod gravity;
pub mod integrator;

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
