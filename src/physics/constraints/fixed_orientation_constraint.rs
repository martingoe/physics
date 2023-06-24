use crate::physics::constraints::{ConstraintOutput, ConstraintTrait, MAX_CONSTRAINT_COUNT};
use crate::physics::rigid_body::RigidBody;
use crate::{Position, Rotation};
use nalgebra::{SMatrix, SVector, Vector3};

const KD: f32 = 1.0;
const KS: f32 = 10.0;

pub struct FixedOrientationConstraint {
    pub rotation: Vector3<f32>,
}

impl ConstraintTrait for FixedOrientationConstraint {
    fn calculate(&self, _body: &RigidBody, _pos: &Position, rot: &Rotation) -> ConstraintOutput {
        let mut c = SVector::<f32, 3>::zeros();
        let delta = rot.0.euler_angles();
        c.x = delta.0 - self.rotation.x;
        c.y = delta.1 - self.rotation.y;
        c.z = delta.2 - self.rotation.z;
        let j_dot = SMatrix::zeros();
        let mut j: SMatrix<f32, { MAX_CONSTRAINT_COUNT }, 6> = SMatrix::zeros();

        j[(0, 3)] = 1.0;
        j[(1, 4)] = 1.0;
        j[(2, 5)] = 1.0;

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
