use nalgebra::{SMatrix, SVector, Vector3};
use crate::physics::constraints::{Constraint, ConstraintOutput, MAX_CONSTRAINT_COUNT};
use crate::physics::PhysicsState;

const KD: f32 = 1.0;
const KS: f32 = 10.0;
pub struct FixToPointConstraint {
    pub rigid_body: usize,
    pub position: Vector3<f32>,
}

impl Constraint for FixToPointConstraint {
    fn calculate(&self, physics_state: &PhysicsState) -> ConstraintOutput {
        let mut c = SVector::<f32, 3>::zeros();
        let delta = physics_state.entities[self.rigid_body].body.position - self.position;
        c.x = delta.x;
        c.y = delta.y;
        c.z = delta.z;
        let j_dot = SMatrix::zeros();
        let mut j: SMatrix<f32, { MAX_CONSTRAINT_COUNT }, { 6 * super::MAX_CONSTRAINT_BODIES }> =
            SMatrix::zeros();
        j[(0, 0)] = 1.0;
        j[(1, 1)] = 1.0;
        j[(2, 2)] = 1.0;

        ConstraintOutput { c, j, j_dot, kd: Vector3::from_element(KD), ks: Vector3::from_element(KS) }
    }

    fn get_constraint_count(&self) -> usize {
        3
    }

    fn get_rigid_bodies(&self) -> Vec<usize> {
        return vec![self.rigid_body];
    }
}