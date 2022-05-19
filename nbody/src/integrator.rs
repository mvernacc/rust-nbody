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
        let mut a: Array2<f64> = Array::zeros((n_steps, n_states));

        x.slice_mut(s![0, ..]).assign(x_init);
        v.slice_mut(s![0, ..]).assign(v_init);

        Self {
            dt,
            n_steps,
            n_states,
            masses: masses.clone(),
            x,
            v,
            a,
        }
    }

    pub fn integrate(&mut self) {
        let dt2 = self.dt * self.dt;
        let mut x_current: Array1<f64> = Array::zeros(self.n_states);
        let mut v_current: Array1<f64> = Array::zeros(self.n_states);
        // `forces[i, j, k]` is the gravitational force between bodies `i` and `j` along direction `k`.
        // [units: N] 
        let mut forces: Array3<f64> = Array::zeros((self.n_states / 3, self.n_states / 3, 3));

        // Compute accelerations at the initial conditions.
        // TODO
        self.a
            .slice_mut(s![0, ..])
            .assign(&Array::ones(self.n_states));

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
            self.a
                .slice_mut(s![i + 1, ..])
                .assign(&Array::ones(self.n_states));

            // Compute velocities at the next time step i + 1.
            v_current.assign(&self.v.slice(s![i, ..]));
            self.v.slice_mut(s![i + 1, ..]).assign(
                &(&v_current
                    + 0.5 * self.dt * (&self.a.slice(s![i, ..]) + &self.a.slice(s![i + 1, ..]))),
            );
        }
    }

    fn update_grav_forces_all_bodies(&self, forces: &mut Array3<f64>, x: &Array1<f64>, masses: &Array1<f64>) {
        let n_bodies = masses.len();
        assert_eq!(x.len(), 3 * n_bodies);
        assert_eq!(forces.shape()[0], n_bodies);
        assert_eq!(forces.shape()[1], n_bodies);
        assert_eq!(forces.shape()[2], 3);

        let mut r_i: [f64; 3] = Default::default();
        let mut r_j: [f64; 3] = Default::default();
    
        for i in 0..n_bodies {
            for k in 0..3 {
                r_i[k] = x[3 * i + k];
            }
            for j in 0..i {
                for k in 0..3 {
                    r_j[k] = x[3 * j + k];
                }
                forces
                    .slice_mut(s![i, j, ..])
                    .assign(calc_grav_force_two_bodies(self.masses[i], self.masses[j], r_i, r_j))
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
        gmm_over_dist_cubed * r_1to2[0],
        gmm_over_dist_cubed * r_1to2[1],
        gmm_over_dist_cubed * r_1to2[2],
    ]
}
