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
}
