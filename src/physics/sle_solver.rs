use nalgebra::{Dyn, OVector};

use crate::physics::sparse_matrix::SparseMatrix;

const CONJUGATE_MAX_ITERATIONS: usize = 1000;
const MAX_ERROR: f32 = 1e-2;
const MIN_ERROR: f32 = 1e-3;

pub enum SLESolver {
    ConjugateGradient
}

impl SLESolver {
    pub fn solve(&self, j: &SparseMatrix, inv_masses: &OVector<f32, Dyn>, rhs: &OVector<f32, Dyn>, previous: &Option<OVector<f32, Dyn>>) -> Option<OVector<f32, Dyn>> {
        match self {
            SLESolver::ConjugateGradient => solve_conjugate_gradient(j, inv_masses, rhs, previous),
        }
    }
}

fn solve_conjugate_gradient(j: &SparseMatrix, inv_masses: &OVector<f32, Dyn>, rhs: &OVector<f32, Dyn>, previous: &Option<OVector<f32, Dyn>>) -> Option<OVector<f32, Dyn>> {
    let mut x = if let Some(previous_solution) = previous{
        previous_solution.to_owned()
    } else {
        OVector::<f32, Dyn>::zeros(rhs.nrows())
    };

    let mut r = rhs.to_owned() - calculate_lhs_multiplied_to_vec(j, inv_masses, &x);
    let mut p = r.to_owned();

    for _ in 0..CONJUGATE_MAX_ITERATIONS {
        let j_p = calculate_lhs_multiplied_to_vec(j, inv_masses, &p);
        let rk_magnitude = r.magnitude_squared();
        let alpha = rk_magnitude / &p.dot(&j_p);
        x += alpha * &p;

        r -= alpha * j_p;
        if r.amax() < (rhs.amax() * MAX_ERROR).max(MIN_ERROR) {
            return Some(x);
        }

        let beta = r.magnitude_squared() / rk_magnitude;
        p = &r + beta * p;
    }
    None
}

fn calculate_lhs_multiplied_to_vec(j: &SparseMatrix, inv_masses: &OVector<f32, Dyn>, factor: &OVector<f32, Dyn>) -> OVector<f32, Dyn> {
    let j_factor = j.tr_multiply_vector(factor);
    j.multiply_vector(&j_factor.component_mul(inv_masses))
}