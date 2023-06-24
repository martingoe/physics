use std::ops::Neg;

use nalgebra::{DVector, Dyn, OVector, SMatrix, SVector};
use specs::prelude::*;
use specs::Component;
use specs::System;

use crate::physics::constraints::fixed_orientation_constraint::FixedOrientationConstraint;
use crate::physics::constraints::fixed_position_constraint::FixToPointConstraint;
use crate::physics::sle_solver::SLESolver;
use crate::physics::sparse_matrix::SparseMatrix;
use crate::Position;
use crate::Rotation;

use super::rigid_body::RigidBody;

pub mod fixed_orientation_constraint;
pub mod fixed_position_constraint;

const MAX_CONSTRAINT_COUNT: usize = 3;

const SLE_SOLVER: SLESolver = SLESolver::ConjugateGradient;

pub struct ConstraintOutput {
    c: SVector<f32, MAX_CONSTRAINT_COUNT>,
    j: SMatrix<f32, MAX_CONSTRAINT_COUNT, 6>,
    j_dot: SMatrix<f32, MAX_CONSTRAINT_COUNT, 6>,
    ks: SVector<f32, MAX_CONSTRAINT_COUNT>,
    kd: SVector<f32, MAX_CONSTRAINT_COUNT>,
}

pub trait ConstraintTrait {
    fn calculate(&self, body: &RigidBody, pos: &Position, rot: &Rotation) -> ConstraintOutput;
    fn get_constraint_count(&self) -> usize;
}

#[derive(Component)]
#[storage(VecStorage)]
pub struct ConstraintComponent(pub Vec<Constraint>);

pub enum Constraint {
    FixedPosition(FixToPointConstraint),
    FixedOrientation(FixedOrientationConstraint),
}

impl Constraint {
    pub(crate) fn get_output(
        &self,
        body: &RigidBody,
        pos: &Position,
        rot: &Rotation,
    ) -> ConstraintOutput {
        match self {
            Constraint::FixedPosition(c) => c.calculate(body, pos, rot),
            Constraint::FixedOrientation(c) => c.calculate(body, pos, rot),
        }
    }

    fn get_constraint_count(&self) -> usize {
        match self {
            Constraint::FixedPosition(c) => c.get_constraint_count(),
            Constraint::FixedOrientation(c) => c.get_constraint_count(),
        }
    }
}

pub struct ConstraintSolver;

impl<'a> System<'a> for ConstraintSolver {
    type SystemData = (
        Write<'a, PreviousConstraintSolution>,
        WriteStorage<'a, RigidBody>,
        WriteStorage<'a, ConstraintComponent>,
        ReadStorage<'a, Position>,
        ReadStorage<'a, Rotation>,
    );

    fn run(&mut self, (mut prev_sol, mut bodies, mut constraints, pos, rot): Self::SystemData) {
        let mut data: Vec<(
            &mut RigidBody,
            &mut ConstraintComponent,
            &Position,
            &Rotation,
        )> = (&mut bodies, &mut constraints, &pos, &rot).join().collect();

        let n = data.len();
        if n == 0 {
            return;
        }

        let inv_masses = OVector::<f32, Dyn>::from_iterator(
            n * 6,
            data.iter().flat_map(|(body, _, _, _)| {
                let inv_mass = 1.0 / body.mass;
                [inv_mass; 6]
            }),
        );
        let q_dot = DVector::<f32>::from_iterator(
            n * 6,
            data.iter().flat_map(|(body, _, _, _)| {
                [
                    body.lin_velocity.x,
                    body.lin_velocity.y,
                    body.lin_velocity.z,
                    body.angular_velocity.x,
                    body.angular_velocity.y,
                    body.angular_velocity.z,
                ]
            }),
        );
        let existing_forces = DVector::<f32>::from_iterator(
            n * 6,
            data.iter().flat_map(|(body, _, _, _)| {
                [
                    body.force.x,
                    body.force.y,
                    body.force.z,
                    body.torque.x,
                    body.torque.y,
                    body.torque.z,
                ]
            }),
        );

        let constraint_count = data.iter().fold(0_usize, |b, (_, c, _, _)| {
            b + c.0.iter().map(|c| c.get_constraint_count()).sum::<usize>()
        });

        let mut j = SparseMatrix::new(constraint_count, n * 6);
        let mut j_dot = SparseMatrix::new(constraint_count, n * 6);

        let mut k_d = OVector::<f32, Dyn>::zeros(constraint_count);
        let mut k_s = OVector::<f32, Dyn>::zeros(constraint_count);

        let mut c = OVector::<f32, Dyn>::zeros(constraint_count);

        let mut constraint_index = 0;
        let mut i = 0;
        for (body, constraints, pos, rot) in data.iter() {
            for constraint in &constraints.0 {
                let output = constraint.get_output(&body, &pos, &rot);
                let single_constraint_count = constraint.get_constraint_count();
                for i in 0..single_constraint_count {
                    k_d[(constraint_index + i, 0)] = output.kd[(i, 0)];
                    k_s[(constraint_index + i, 0)] = output.ks[(i, 0)];
                    c[(constraint_index + i, 0)] = output.c[(i, 0)];
                }

                let j_slice = output
                    .j
                    .view((0, 0), (single_constraint_count, 6))
                    .into_owned();
                let j_dot_slice = output
                    .j_dot
                    .view((0, 0), (single_constraint_count, 6))
                    .into_owned();

                j.add_block(constraint_index, i * 6, single_constraint_count, 6, j_slice);
                j_dot.add_block(
                    constraint_index,
                    i * 6,
                    single_constraint_count,
                    6,
                    j_dot_slice,
                );

                constraint_index += single_constraint_count;
            }
            i += 1;
        }

        let j_dot_times_q_dot = j_dot.multiply_vector(&q_dot).neg();

        let c_dot = j.multiply_vector(&q_dot);
        k_d = k_d.component_mul(&c_dot);
        k_s = k_s.component_mul(&c);

        let rhs = j_dot_times_q_dot
            - (j.multiply_vector(&existing_forces.component_mul(&inv_masses)))
            - k_s
            - k_d;

        let lambda = SLE_SOLVER.solve(&j, &inv_masses, &rhs, &prev_sol.0);

        if let Some(sol) = lambda {
            let matrix = j.tr_multiply_vector(&sol);
            for (i, (body, _, _, _)) in data.iter_mut().enumerate() {
                let column = matrix.column(i);
                body.force += column.view((0, 0), (3, 1));
                body.torque += column.view((3, 0), (3, 1));
            }
            prev_sol.0 = Some(sol);
        } else {
            // TODO: Log failure
        }
    }
}

#[derive(Default)]
pub struct PreviousConstraintSolution(pub Option<OVector<f32, Dyn>>);
