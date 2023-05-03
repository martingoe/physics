use nalgebra::{SMatrix, SVector, Vector3};
use crate::physics::constraints::{Constraint, ConstraintOutput, MAX_CONSTRAINT_COUNT};
use crate::physics::PhysicsState;
use super::MAX_CONSTRAINT_BODIES;

const KD: f32 = 1.0;
const KS: f32 = 10.0;

pub struct FixedOrientationConstraint {
    pub rigid_body: usize,
    pub position: Vector3<f32>,
}

impl Constraint for FixedOrientationConstraint {
    fn calculate(&self, physics_state: &PhysicsState) -> ConstraintOutput {
        let mut c = SVector::<f32, 3>::zeros();
        let delta = physics_state.entities[self.rigid_body].body.rotation.euler_angles();
        c.x = delta.0 - self.position.x;
        c.y = delta.1 - self.position.y;
        c.z = delta.2 - self.position.z;
        let j_dot = SMatrix::zeros();
        let mut j: SMatrix<f32, { MAX_CONSTRAINT_COUNT }, { 6 * MAX_CONSTRAINT_BODIES }> =
            SMatrix::zeros();

        j[(0, 3)] = 1.0;
        j[(1, 4)] = 1.0;
        j[(2, 5)] = 1.0;

        ConstraintOutput { c, j, j_dot, kd: Vector3::from_element(KD), ks: Vector3::from_element(KS) }
    }

    fn get_constraint_count(&self) -> usize {
        3
    }

    fn get_rigid_bodies(&self) -> Vec<usize> {
        return vec![self.rigid_body];
    }
}