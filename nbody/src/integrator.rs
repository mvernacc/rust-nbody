use ndarray::prelude::*;
use ndarray::{Array, Array1, Array2};

use super::NBodyInput;
use super::gravity;

pub struct LeapfrogIntegrator {
    pub dt: f64,
    pub n_steps: usize,
    pub n_states: usize,
    pub masses: Array1<f64>,
    pub t: Array1<f64>,
    pub r: Array2<f64>,
    pub v: Array2<f64>,
    pub a: Array2<f64>,
    /// `forces[i, j, k]` is the gravitational force between bodies `i` and `j` along direction `k`.
    /// Used internally to avoid repeated calculations; overwritten at each time step.
    /// [units: N]
    forces: Array3<f64>,
}

impl LeapfrogIntegrator {
    pub fn new(input: &NBodyInput) -> Self {
        let n_bodies = input.bodies.len();
        let n_states = 3 * n_bodies;
        let dt = input.timestep_s;
        let n_steps = input.n_steps;

        let t: Array1<f64> = Array::range(0.0, dt * (n_steps as f64), dt);
        let mut masses: Array1<f64> = Array::zeros(n_bodies);
        let mut r: Array2<f64> = Array::zeros((input.n_steps, n_states));
        let mut v: Array2<f64> = Array::zeros((input.n_steps, n_states));
        let a: Array2<f64> = Array::zeros((input.n_steps, n_states));
        let forces: Array3<f64> = Array::zeros((masses.len(), masses.len(), 3));

        for (i, b) in input.bodies.iter().enumerate() {
            masses[i] = b.mass_kg;
            for dim in 0..3 {
                r[[0, 3 * i + dim]] = b.position_init_m[dim];
                v[[0, 3 * i + dim]] = b.velocity_init_m_per_s[dim];
            }
        }

        Self {
            dt,
            n_steps,
            n_states,
            masses,
            t,
            r,
            v,
            a,
            forces,
        }
    }

    /// Integrate the n-body gravitation dynamics through time, using the Leapfrog algorithm.
    /// The "velocity at integer timesteps" formulation of Leapfrog is used, as given in [Hut2004],
    /// Ch 4.1, http://www.artcompsci.org/vol_1/v1_web/node34.html, equations 4.4 and 4.5.
    ///
    /// References:
    ///     [Hut2004] P. Hut and J. Makino, "Moving Stars Around,"" The Art of Computational Science, 2004.
    ///         http://www.artcompsci.org/vol_1/v1_web/v1_web.html (accessed May 18, 2022).
    pub fn integrate(&mut self) {
        let dt2 = self.dt * self.dt;
        let mut r_current: Array1<f64> = Array::zeros(self.n_states);
        let mut v_current: Array1<f64> = Array::zeros(self.n_states);

        // Compute accelerations at the initial conditions.
        self.update_accelerations(0);

        for i in 0..self.n_steps - 1 {
            // Compute positions at the next time step i + 1.
            r_current.assign(&self.r.slice(s![i, ..]));
            self.r.slice_mut(s![i + 1, ..]).assign(
                &(&r_current
                    + self.dt * &self.v.slice(s![i, ..])
                    + 0.5 * dt2 * &self.a.slice(s![i, ..])),
            );

            // Compute accelerations at the next time step i + 1.
            self.update_accelerations(i + 1);

            // Compute velocities at the next time step i + 1.
            v_current.assign(&self.v.slice(s![i, ..]));
            self.v.slice_mut(s![i + 1, ..]).assign(
                &(&v_current
                    + 0.5 * self.dt * (&self.a.slice(s![i, ..]) + &self.a.slice(s![i + 1, ..]))),
            );
        }
    }

    fn update_accelerations(&mut self, step_index: usize) {
        self.update_grav_forces_all_bodies(step_index);

        let n_bodies = self.masses.len();
        for i in 0..n_bodies {
            // Compute the net gravitational force on this body.
            let mut force_net: [f64; 3] = [0.0, 0.0, 0.0];
            for j in 0..i {
                #[allow(clippy::needless_range_loop)]
                for k in 0..3 {
                    force_net[k] += self.forces[[i, j, k]];
                }
            }
            for j in i + 1..n_bodies {
                #[allow(clippy::needless_range_loop)]
                for k in 0..3 {
                    force_net[k] -= self.forces[[j, i, k]];
                }
            }

            // Update the acceleration of this body using F = m a.
            #[allow(clippy::needless_range_loop)]
            for k in 0..3 {
                self.a[[step_index, 3 * i + k]] = force_net[k] / self.masses[i];
            }
        }
    }

    fn update_grav_forces_all_bodies(&mut self, step_index: usize) {
        let n_bodies = self.masses.len();
        assert_eq!(self.r.shape()[1], 3 * n_bodies);
        assert_eq!(self.forces.shape()[0], n_bodies);
        assert_eq!(self.forces.shape()[1], n_bodies);
        assert_eq!(self.forces.shape()[2], 3);

        let mut r_i: [f64; 3] = Default::default();
        let mut r_j: [f64; 3] = Default::default();

        for i in 0..n_bodies {
            #[allow(clippy::needless_range_loop)]
            for k in 0..3 {
                r_i[k] = self.r[[step_index, 3 * i + k]];
            }
            for j in 0..i {
                #[allow(clippy::needless_range_loop)]
                for k in 0..3 {
                    r_j[k] = self.r[[step_index, 3 * j + k]];
                }
                let f_ij = gravity::calc_grav_force_two_bodies(self.masses[i], self.masses[j], r_i, r_j);
                #[allow(clippy::needless_range_loop)]
                for k in 0..3 {
                    self.forces[[i, j, k]] = f_ij[k];
                }
            }
        }
    }
}


#[cfg(test)]
mod tests {
    use crate::{NBodyInput, Body};

    use super::LeapfrogIntegrator;
    use ndarray::prelude::*;
    use approx::assert_relative_eq;

    /// Check that a lone test mass moves in a straight line at constant velocity.
    #[test]
    fn one_body_straight_line() {
        // Setup
        let test_mass = Body {
            name: "Test Mass".to_string(),
            mass_kg: 1.0,
            position_init_m: [0.0, 0.0, 0.0],
            velocity_init_m_per_s: [1.0, 0.0, 0.0],
        };
        let input = NBodyInput {
            bodies: vec![test_mass],
            timestep_s: 0.1,
            n_steps: 11,
        };
        let mut integrator = LeapfrogIntegrator::new(&input);

        // Action
        integrator.integrate();

        // Verification
        // final velocity == [1, 0, 0]
        assert_eq!(integrator.v[[input.n_steps - 1, 0]], 1.0);
        assert_eq!(integrator.v[[input.n_steps - 1, 1]], 0.0);
        assert_eq!(integrator.v[[input.n_steps - 1, 2]], 0.0);
        // final position == [1, 0, 0]
        assert_relative_eq!(integrator.r[[input.n_steps - 1, 0]], 1.0, epsilon = 1e-9);
        assert_eq!(integrator.r[[input.n_steps - 1, 1]], 0.0);
        assert_eq!(integrator.r[[input.n_steps - 1, 2]], 0.0);
    }

    /// Check that a light test mass, tossed sideways near the surface of the Earth, follows a
    /// parabolic trajectory, and does not move the Earth.
    #[test]
    fn test_mass_earth_parabolic_trajectory() {
        // Setup
        // -----
        let mass_earth = 5.9722e24; // [kg]
        let radius_earth = 6371e3; // [m]
        let g = 9.807; // [m s^-2]
        let test_mass = Body {
            name: "Test Mass".to_string(),
            mass_kg: 1.0,
            position_init_m: [0.0, 0.0, radius_earth],
            velocity_init_m_per_s: [1.0, 0.0, 0.0],
        };
        let earth = Body {
            name: "Earth".to_string(),
            mass_kg: mass_earth,
            position_init_m: [0.0, 0.0, 0.0],
            velocity_init_m_per_s: [0.0, 0.0, 0.0],
        };
        let input = NBodyInput {
            bodies: vec![earth, test_mass.clone()],
            timestep_s: 0.1,
            n_steps: 101,
        };
        let mut integrator = LeapfrogIntegrator::new(&input);

        // Action
        // ------
        integrator.integrate();

        // Verification
        // ------------
        // Calculate the horizontal and vertical displacements for a parabolic trajectory.
        let duration: f64 = input.timestep_s * (input.n_steps - 1) as f64; // [s]
        let vertical_displacement = -0.5 * g * duration * duration; // [m]
        let horizontal_displacement = test_mass.velocity_init_m_per_s[0] * duration; // [m]
        // Earth final velocity should be nil.
        let v_earth_final = integrator.v.slice(s![input.n_steps - 1, 0usize..3usize]);
        assert_relative_eq!(v_earth_final[0], 0.0);
        assert_relative_eq!(v_earth_final[1], 0.0);
        assert_relative_eq!(v_earth_final[2], 0.0);
        // Earth final position should be [0, 0, 0].
        let r_earth_final = integrator.r.slice(s![input.n_steps - 1, 0usize..3usize]);
        assert_relative_eq!(r_earth_final[0], 0.0);
        assert_relative_eq!(r_earth_final[1], 0.0);
        assert_relative_eq!(r_earth_final[2], 0.0);
        // Test mass final velocity should be increased by [0, 0, -duration * g]
        let v_test_final = integrator.v.slice(s![input.n_steps - 1, 3usize..6usize]);
        println!("{:?}", v_test_final);
        let v_test_final_correct = [
            test_mass.velocity_init_m_per_s[0],
            test_mass.velocity_init_m_per_s[1],
            test_mass.velocity_init_m_per_s[2] - duration * g,
        ];
        assert_relative_eq!(v_test_final[0], v_test_final_correct[0], max_relative = 1e-4);
        assert_relative_eq!(v_test_final[1], v_test_final_correct[1]);
        assert_relative_eq!(v_test_final[2], v_test_final_correct[2], max_relative = 5e-3);
        // Test mass final position should be moved by the horizontal and vertical displacements.
        let r_test_final = integrator.r.slice(s![input.n_steps - 1, 3usize..6usize]);
        let r_test_final_correct = [
            horizontal_displacement,
            0.0,
            radius_earth + vertical_displacement,
        ];
        assert_relative_eq!(r_test_final[0], r_test_final_correct[0], epsilon = 1e-3);
        assert_relative_eq!(r_test_final[1], r_test_final_correct[1]);
        assert_relative_eq!(r_test_final[2], r_test_final_correct[2], epsilon = 1.0);
    }
}