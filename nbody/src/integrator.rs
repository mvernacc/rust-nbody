use ndarray::prelude::*;
use ndarray::{Array, Array1, Array2};

pub struct LeapfrogIntegrator {
    pub dt: f64,
    pub n_steps: usize,
    pub n_states: usize,
    pub masses: Array1<f64>,
    pub x: Array2<f64>,
    pub v: Array2<f64>,
    pub a: Array2<f64>,
    /// `forces[i, j, k]` is the gravitational force between bodies `i` and `j` along direction `k`.
    /// Used internally to avoid repeated calculations; overwritten at each time step.
    /// [units: N] 
    forces: Array3<f64>,
}

impl LeapfrogIntegrator {
    pub fn new(
        dt: f64,
        n_steps: usize,
        masses: &Array1<f64>,
        x_init: &Array1<f64>,
        v_init: &Array1<f64>,
    ) -> Self {
        let n_states = 3 * masses.len();
        assert_eq!(n_states, x_init.len());
        assert_eq!(n_states, v_init.len());

        let mut x: Array2<f64> = Array::zeros((n_steps, n_states));
        let mut v: Array2<f64> = Array::zeros((n_steps, n_states));
        let a: Array2<f64> = Array::zeros((n_steps, n_states));
        let forces: Array3<f64> = Array::zeros((masses.len(), masses.len(), 3));

        x.slice_mut(s![0usize, ..]).assign(x_init);
        v.slice_mut(s![0usize, ..]).assign(v_init);

        Self {
            dt,
            n_steps,
            n_states,
            masses: masses.clone(),
            x,
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
        let mut x_current: Array1<f64> = Array::zeros(self.n_states);
        let mut v_current: Array1<f64> = Array::zeros(self.n_states);

        // Compute accelerations at the initial conditions.
        // TODO
        self.update_accelerations(0);

        for i in 0..self.n_steps - 1 {
            // Compute positions at the next time step i + 1.
            x_current.assign(&self.x.slice(s![i, ..]));
            self.x.slice_mut(s![i + 1, ..]).assign(
                &(&x_current
                    + self.dt * &self.v.slice(s![i, ..])
                    + 0.5 * dt2 * &self.a.slice(s![i, ..])),
            );

            // Compute accelerations at the next time step i + 1.
            // TODO
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
            for j in i+1..n_bodies {
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
        assert_eq!(self.x.shape()[1], 3 * n_bodies);
        assert_eq!(self.forces.shape()[0], n_bodies);
        assert_eq!(self.forces.shape()[1], n_bodies);
        assert_eq!(self.forces.shape()[2], 3);

        let mut r_i: [f64; 3] = Default::default();
        let mut r_j: [f64; 3] = Default::default();
    
        for i in 0..n_bodies {
            #[allow(clippy::needless_range_loop)]
            for k in 0..3 {
                r_i[k] = self.x[[step_index, 3 * i + k]];
            }
            for j in 0..i {
                #[allow(clippy::needless_range_loop)]
                for k in 0..3 {
                    r_j[k] = self.x[[step_index, 3 * j + k]];
                }
                let f_ij = calc_grav_force_two_bodies(self.masses[i], self.masses[j], r_i, r_j);
                #[allow(clippy::needless_range_loop)]
                for k in 0..3 {
                    self.forces[[i, j, k]] = f_ij[k];
                }
            }
        }
    }
}

/// Gravitational constant [units: N m^2 kg^-1].
const G: f64 = 6.6743015e-11;


fn calc_grav_force_two_bodies(mass_1: f64, mass_2: f64, r_1: [f64; 3], r_2: [f64; 3]) -> [f64; 3] {
    // Position vector from body 1 center to body 2 center [units: m].
    let r_1to2 = [r_1[0] - r_2[0], r_1[1] - r_2[1], r_1[2] - r_2[2]];
    // Distance between body 1 and 2, to the third power [units: m^3]
    let distance_cubed = (r_1to2[0] * r_1to2[0] + r_1to2[1] * r_1to2[1] + r_1to2[2] * r_1to2[2]).powf(3.0 / 2.0);
    let gmm_over_dist_cubed = G * mass_1 * mass_2 / distance_cubed;
    // Gravitational force of body 2 on body 1 [units: N].
    [
        -gmm_over_dist_cubed * r_1to2[0],
        -gmm_over_dist_cubed * r_1to2[1],
        -gmm_over_dist_cubed * r_1to2[2],
    ]
}
