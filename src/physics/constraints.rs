use std::ops::Neg;

use nalgebra::{DVector, Dyn, OVector, SMatrix, SVector};

use crate::physics::constraints::fixed_orientation_constraint::FixedOrientationConstraint;
use crate::physics::constraints::fixed_position_constraint::FixToPointConstraint;
use crate::physics::PhysicsState;
use crate::physics::sle_solver::SLESolver;
use crate::physics::sparse_matrix::SparseMatrix;

pub mod fixed_position_constraint;
pub mod fixed_orientation_constraint;

const MAX_CONSTRAINT_BODIES: usize = 2;
const MAX_CONSTRAINT_COUNT: usize = 3;

const SLE_SOLVER: SLESolver = SLESolver::ConjugateGradient;

pub struct ConstraintOutput {
    c: SVector<f32, MAX_CONSTRAINT_COUNT>,
    j: SMatrix<f32, MAX_CONSTRAINT_COUNT, { 6 * MAX_CONSTRAINT_BODIES }>,
    j_dot: SMatrix<f32, MAX_CONSTRAINT_COUNT, { 6 * MAX_CONSTRAINT_BODIES }>,
    ks: SVector<f32, MAX_CONSTRAINT_COUNT>,
    kd: SVector<f32, MAX_CONSTRAINT_COUNT>,
}

pub trait Constraint {
    fn calculate(&self, physics_state: &PhysicsState) -> ConstraintOutput;
    fn get_constraint_count(&self) -> usize;
    fn get_rigid_bodies(&self) -> Vec<usize>;
}

pub enum Constraints {
    FixedPosition(FixToPointConstraint),
    FixedOrientation(FixedOrientationConstraint),

}

impl Constraints {
    pub(crate) fn get_output(&self, physics_state: &PhysicsState) -> ConstraintOutput {
        match self {
            Constraints::FixedPosition(c) => c.calculate(physics_state),
            Constraints::FixedOrientation(c) => c.calculate(physics_state)
        }
    }

    fn get_constraint_count(&self) -> usize {
        match self {
            Constraints::FixedPosition(c) => c.get_constraint_count(),
            Constraints::FixedOrientation(c) => c.get_constraint_count(),
        }
    }
    fn get_rigid_bodies(&self) -> Vec<usize> {
        match self {
            Constraints::FixedPosition(c) => c.get_rigid_bodies(),
            Constraints::FixedOrientation(c) => c.get_rigid_bodies(),
        }
    }
}

pub struct ConstraintSolver {
    pub constraints: Vec<Constraints>,
}


impl ConstraintSolver {
    pub fn solve_constraints(
        &self,
        physics_state: &PhysicsState,
        previous_solution: &Option<OVector<f32, Dyn>>,
    ) -> Option<(OVector<f32, Dyn>, OVector<f32, Dyn>)> {
        let inv_masses = OVector::<f32, Dyn>::from_iterator(
            physics_state.entities.len() * 6,
            physics_state.entities.iter().flat_map(|body| {
                let inv_mass = 1.0 / body.body.mass;
                [inv_mass; 6]
            }),
        );
        let q_dot = DVector::<f32>::from_iterator(
            physics_state.entities.len() * 6,
            physics_state.entities.iter().flat_map(|body| {
                [
                    body.body.lin_velocity.x,
                    body.body.lin_velocity.y,
                    body.body.lin_velocity.z,
                    body.body.angular_velocity.x,
                    body.body.angular_velocity.y,
                    body.body.angular_velocity.z,
                ]
            }),
        );
        let existing_forces = DVector::<f32>::from_iterator(
            physics_state.entities.len() * 6,
            physics_state.entities.iter().flat_map(|body| {
                [
                    body.body.force.x,
                    body.body.force.y,
                    body.body.force.z,
                    body.body.torque.x,
                    body.body.torque.y,
                    body.body.torque.z,
                ]
            }),
        );

        let constraint_count = self.get_full_constraint_count();
        let mut j = SparseMatrix::new(constraint_count, physics_state.entities.len() * 6);
        let mut j_dot = SparseMatrix::new(constraint_count, physics_state.entities.len() * 6);

        let mut k_d = OVector::<f32, Dyn>::zeros(constraint_count);
        let mut k_s = OVector::<f32, Dyn>::zeros(constraint_count);

        let mut c = OVector::<f32, Dyn>::zeros(constraint_count);

        let mut constraint_index = 0;
        for constraint in &self.constraints {
            let output = constraint.get_output(&physics_state);
            let single_constraint_count = constraint.get_constraint_count();
            for i in 0..single_constraint_count {
                k_d[(constraint_index + i, 0)] = output.kd[(i, 0)];
                k_s[(constraint_index + i, 0)] = output.ks[(i, 0)];
                c[(constraint_index + i, 0)] = output.c[(i, 0)];
            }

            for (i, body) in constraint.get_rigid_bodies().iter().enumerate() {
                let j_slice = output
                    .j
                    .view((0, i * 6), (single_constraint_count, 6))
                    .into_owned();
                let j_dot_slice = output
                    .j_dot
                    .view((0, i * 6), (single_constraint_count, 6))
                    .into_owned();

                j.add_block(
                    constraint_index,
                    body * 6,
                    single_constraint_count,
                    6,
                    j_slice,
                );
                j_dot.add_block(
                    constraint_index,
                    body * 6,
                    single_constraint_count,
                    6,
                    j_dot_slice,
                );
            }
            constraint_index += single_constraint_count;
        }

        let j_dot_times_q_dot = j_dot.multiply_vector(&q_dot).neg();

        let c_dot = j.multiply_vector(&q_dot);
        k_d = k_d.component_mul(&c_dot);
        k_s = k_s.component_mul(&c);

        let rhs =
            j_dot_times_q_dot - (j.multiply_vector(&existing_forces.component_mul(&inv_masses))) - k_s - k_d;

        let lambda = SLE_SOLVER.solve(&j, &inv_masses, &rhs, previous_solution);

        if let Some(sol) = lambda {
            let matrix = j.tr_multiply_vector(&sol);
            return Some((sol, matrix));
        }
        None
    }


    fn get_full_constraint_count(&self) -> usize {
        self.constraints
            .iter()
            .map(|c| c.get_constraint_count()).sum()
    }
}
