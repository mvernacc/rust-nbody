use ndarray::prelude::*;
use ndarray::{Array, Array1, Array2};

pub struct LeapfrogIntegrator {
    dt: f64,
    n_steps: usize,
    n_states: usize,
    masses: Array1<f64>,
    x: Array2<f64>,
    v: Array2<f64>,
    a: Array2<f64>,
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

        return Self {
            dt,
            n_steps,
            n_states,
            masses: masses.clone(),
            x,
            v,
            a,
        };
    }

    pub fn integrate(&mut self) {
        let dt2 = self.dt * self.dt;
        for i in 0..self.n_steps - 1 {
            // TODO
            self.a
                .slice_mut(s![i, ..])
                .assign(&Array::ones(self.n_states));

            self.x.slice_mut(s![i + 1, ..]).assign(
                &(&self.x.slice(s![i, ..])
                    + self.dt * &self.v.slice(s![i, ..])
                    + 0.5 * dt2 * &self.a.slice(s![i, ..])),
            );

            self.v.slice_mut(s![i + 1, ..]).assign(
                &(&self.v.slice(s![i, ..])
                    + 0.5 * self.dt * (&self.a.slice(s![i, ..]) + &self.a.slice(s![i + 1, ..]))),
            );
        }
    }
}
