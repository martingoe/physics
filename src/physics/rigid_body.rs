use nalgebra::{Matrix3, UnitQuaternion, Vector3};
use specs::prelude::*;
use specs::Component;

use crate::DeltaTime;
use crate::Position;
use crate::Rotation;

#[derive(Debug, Component)]
#[storage(VecStorage)]
pub struct RigidBody {
    pub mass: f32,

    pub lin_velocity: Vector3<f32>,
    pub angular_velocity: Vector3<f32>,

    pub force: Vector3<f32>,
    pub torque: Vector3<f32>,

    pub inertia_tensor: Matrix3<f32>,
}

pub struct RigidBodyStepSys;
impl<'a> System<'a> for RigidBodyStepSys {
    type SystemData = (
        Read<'a, DeltaTime>,
        WriteStorage<'a, Position>,
        WriteStorage<'a, Rotation>,
        WriteStorage<'a, RigidBody>,
    );

    fn run(&mut self, (dt, mut pos, mut rot, mut rigid_body): Self::SystemData) {
        let dt = dt.0;
        for (pos, rot, body) in (&mut pos, &mut rot, &mut rigid_body).join() {
            let dt = dt.as_secs_f32();
            // Euler Translation
            body.lin_velocity += body.force / body.mass * dt;
            pos.0 += body.lin_velocity * dt;

            let angular_momentum = body.torque * dt;
            body.angular_velocity += body.inertia_tensor.try_inverse().unwrap() * angular_momentum;
            if body.angular_velocity != Vector3::<f32>::zeros() {
                let a = body.angular_velocity.normalize();
                let theta = body.angular_velocity.magnitude() * dt;
                let dq = UnitQuaternion::new(a * (theta * 0.5).sin());
                rot.0 = dq * rot.0;
            }
            body.force = Vector3::zeros();
            body.torque = Vector3::zeros();
        }
    }
}
impl RigidBody {
    pub fn apply_force_centre_of_gravity(&mut self, force: Vector3<f32>) {
        self.force += force;
    }

    pub fn apply_force_at_position(
        &mut self,
        force: Vector3<f32>,
        point: Vector3<f32>,
        position: &Vector3<f32>,
    ) {
        self.torque += (point - position).cross(&force);
        self.force += force;
    }
    pub fn apply_force_at_offset(&mut self, force: Vector3<f32>, offset: Vector3<f32>) {
        self.torque += (offset).cross(&force);
        self.force += force;
    }
}
impl Default for RigidBody {
    fn default() -> Self {
        Self {
            mass: 1.0,
            lin_velocity: Vector3::zeros(),
            angular_velocity: Vector3::zeros(),
            force: Vector3::zeros(),
            torque: Vector3::zeros(),
            inertia_tensor: Matrix3::identity(),
        }
    }
}
