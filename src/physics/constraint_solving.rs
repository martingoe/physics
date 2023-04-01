use std::ops::{Mul, MulAssign};

use nalgebra::{Const, Dim, DMatrix, DVector, Dyn, Matrix, MatrixSum, OMatrix, OVector, SMatrix, SquareMatrix, SVector, U1, Vector, Vector3};

use crate::physics::PhysicsState;
use crate::physics::rigid_body::RigidBody;

const MAX_CONSTRAINT_BODIES: usize = 2;
const MAX_CONSTRAINT_COUNT: usize = 3;

trait Constraint {
    fn calculate(&self) -> ConstraintOutput;
    fn get_constraint_count(&self) -> usize;
    fn get_rigid_bodies(&self) -> Vec<usize>;
}

pub enum Constraints {
    ConstantRotation(ConstantRotationConstraint)
}

impl Constraints {
    pub(crate) fn get_output(&self) -> ConstraintOutput {
        match self {
            Constraints::ConstantRotation(c) => c.calculate()
        }
    }

    fn get_constraint_count(&self) -> usize {
        match self {
            Constraints::ConstantRotation(c) => c.get_constraint_count()
        }
    }
    fn get_rigid_bodies(&self) -> Vec<usize> {
        match self {
            Constraints::ConstantRotation(c) => c.get_rigid_bodies()
        }
    }
}


pub struct ConstraintSolver {
    pub constraints: Vec<Constraints>,
}

pub struct ConstraintOutput {
    c: SVector<f32, MAX_CONSTRAINT_COUNT>,
    j: SMatrix<f32, MAX_CONSTRAINT_COUNT, { 6 * MAX_CONSTRAINT_BODIES }>,
    j_dot: SMatrix<f32, MAX_CONSTRAINT_COUNT, { 6 * MAX_CONSTRAINT_BODIES }>,
}

pub struct ConstantRotationConstraint {
    pub rigid_body: usize,
    pub rotation: Vector3<f32>,
}

impl Constraint for ConstantRotationConstraint {
    fn calculate(&self) -> ConstraintOutput {
        let c = SVector::zeros();
        let j_dot = SMatrix::zeros();
        let mut j: SMatrix<f32, MAX_CONSTRAINT_COUNT, { 6 * MAX_CONSTRAINT_BODIES }> = SMatrix::zeros();
        j[(0, 3)] = self.rotation[0];
        j[(0, 4)] = self.rotation[1];
        j[(0, 5)] = self.rotation[2];
        ConstraintOutput {
            c,
            j,
            j_dot,
        }
    }

    fn get_constraint_count(&self) -> usize {
        1
    }

    fn get_rigid_bodies(&self) -> Vec<usize> {
        return vec![self.rigid_body];
    }
}

impl ConstraintSolver {
    pub fn solve_constraints(&self, physics_state: &mut PhysicsState) {
        let inv_masses = SquareMatrix::from_diagonal(&DVector::<f32>::from_iterator(physics_state.entities.len() * 6, physics_state.entities.iter().flat_map(|body| {
            let inv_mass = 1.0 / body.body.mass;
            [inv_mass; 6]
        })));
        let q_dot = DVector::<f32>::from_iterator(physics_state.entities.len() * 6, physics_state.entities.iter().flat_map(|body| {
            [
                body.body.lin_velocity.x,
                body.body.lin_velocity.y,
                body.body.lin_velocity.z,
                body.body.angular_velocity.x,
                body.body.angular_velocity.y,
                body.body.angular_velocity.z
            ]
        }));
        let existing_forces = DVector::<f32>::from_iterator(physics_state.entities.len() * 6, physics_state.entities.iter().flat_map(|body| {
            [
                body.body.force.x,
                body.body.force.y,
                body.body.force.z,
                body.body.torque.x,
                body.body.torque.y,
                body.body.torque.z
            ]
        }));

        let constaint_count = self.get_full_constraint_count();
        let mut j = OMatrix::<f32, Dyn, Dyn>::repeat(constaint_count, physics_state.entities.len() * 6, 0.0);
        let mut j_dot = OMatrix::<f32, Dyn, Dyn>::repeat(constaint_count, physics_state.entities.len() * 6, 0.0);

        let mut constraint_index = 0;
        println!("{:?}", j);
        for constraint in &self.constraints {
            let output = constraint.get_output();
            for (i, body) in constraint.get_rigid_bodies().iter().enumerate() {
                for loc_c_index in 0..constraint.get_constraint_count() {
                    j.view_mut((constraint_index, *body), (1, 6))
                        .set_row(0, &output.j.view((loc_c_index, i * 6), (1, 6)).row(0));
                    j_dot.view_mut((constraint_index, *body), (1, 6))
                        .set_row(0, &output.j_dot.view((loc_c_index, i * 6), (1, 6)).row(0));
                }
            }
        }
        let j_dot_times_q_dot = -j_dot * q_dot;
        let j_t = j.transpose();
        j.mul_assign(inv_masses);
        let mut lhs = DMatrix::<f32>::repeat(self.constraints.len(), self.constraints.len(), 0.0);
        j.mul_to(&j_t, &mut lhs);

        let rhs = j_dot_times_q_dot - j * existing_forces;
        let ncols = lhs.ncols();
        lhs = lhs.insert_column(ncols, 0.0);
        for i in 0..lhs.nrows(){
            lhs[(i, ncols)] = rhs[(i, 0)];
        }

        let lambda = solve_system_of_equations(lhs);
        let res = j_t * lambda;
        for (i, row) in res.row_iter().enumerate(){
            physics_state.entities[i].body.force += row.view((0, 0), (1, 3)).transpose();
            physics_state.entities[i].body.torque += row.view((0, 3), (1, 3)).transpose();
        }
    }

    fn get_full_constraint_count(&self) -> usize {
        self.constraints.iter().fold(0, |sum, c| sum + c.get_constraint_count())
    }
}

fn solve_system_of_equations(mut lhs: OMatrix<f32, Dyn, Dyn>) -> OVector<f32, Dyn> {
    // Gaussian elimination
    let mut h = 0;
    let mut k = 0;
    let m = lhs.nrows();
    let n = lhs.ncols();


    while h < m && k < n {
        let (i_max, index) = get_max_abs_value_in_column_between(h, m, k, &mut lhs);

        if i_max == 0.0 {
            k = k + 1;
        } else {
            lhs.swap_rows(h, index);
            for i in (h + 1)..m {
                let f = lhs[(i, k)] / lhs[(h, k)];
                lhs[(i, k)] = 0.0;
                for j in (k + 1)..n {
                    lhs[(i, j)] = lhs[(i, j)] - lhs[(h, j)] * f;
                }
            }
        }
        h += 1;
        k += 1;
    }

    for i in (1..m).rev() {
        for j in (1..i + 1).rev() {
            let factor = lhs[(j - 1, i)] / lhs[(i, i)];
            for k in (0..n).rev(){
                lhs[(j - 1, k)] -= factor * lhs[(i, k)];
            }
        }
    }
    let mut result: Vec<f32> = vec![0f32; m];
    for i in 0..m {
        result[i] = lhs[(i, m)] / lhs[(i, i)];
    }
    OVector::<f32, Dyn>::from(result)
}
fn get_max_abs_value_in_column_between(from: usize, to: usize, column: usize, matrix: &OMatrix<f32, Dyn, Dyn>) -> (f32, usize) {
    let mut max = 0.0;
    let mut index = 0;
    for i in from..to {
        if matrix[(i, column)].abs() > max {
            max = matrix[(i, column)].abs();
            index = i;
        }
    }
    return (max, index);
}