use crate::physics::constraints::{ConstraintOutput, ConstraintTrait, MAX_CONSTRAINT_COUNT};
use crate::physics::rigid_body::RigidBody;
use crate::{Position, Rotation};
use nalgebra::{SMatrix, SVector, Vector3};

const KD: f32 = 1.0;
const KS: f32 = 10.0;
pub struct FixToPointConstraint {
    pub position: Vector3<f32>,
}

impl ConstraintTrait for FixToPointConstraint {
    fn calculate(&self, _body: &RigidBody, pos: &Position, _rot: &Rotation) -> ConstraintOutput {
        let mut c = SVector::<f32, 3>::zeros();
        let delta = pos.0 - self.position;
        c.x = delta.x;
        c.y = delta.y;
        c.z = delta.z;
        let j_dot = SMatrix::zeros();
        let mut j: SMatrix<f32, { MAX_CONSTRAINT_COUNT }, 6> = SMatrix::zeros();
        j[(0, 0)] = 1.0;
        j[(1, 1)] = 1.0;
        j[(2, 2)] = 1.0;

        ConstraintOutput {
            c,
            j,
            j_dot,
            kd: Vector3::from_element(KD),
            ks: Vector3::from_element(KS),
        }
    }

    fn get_constraint_count(&self) -> usize {
        3
    }
}
