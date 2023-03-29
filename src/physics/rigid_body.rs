use std::time::Duration;
use nalgebra::{Matrix3, Quaternion, UnitQuaternion, Vector3};

use super::Entity;

pub struct RigidBody {
    mass: f32,

    lin_velocity: Vector3<f32>,
    angular_velocity: Vector3<f32>,

    force: Vector3<f32>,
    torque: Vector3<f32>,

    inertia_tensor: Matrix3<f32>,

    pub(crate) position: Vector3<f32>,
    pub(crate) rotation: UnitQuaternion<f32>,
}

impl RigidBody {
    pub fn update_entity(&self, entity: &mut Entity) {
        entity.position = self.position;
        entity.rotation = self.rotation;
    }
    pub fn step(&mut self, dt: &Duration) {
        let dt = dt.as_secs_f32();
        // Euler Translation
        self.lin_velocity += self.force / self.mass * dt;
        //self.position += self.lin_velocity * dt;

        let angular_momentum = self.torque * dt;
        self.angular_velocity += self.inertia_tensor.try_inverse().unwrap() * angular_momentum;
        // self.angular_velocity += self.torque.cross(self.lin_velocity) / self.mass * dt;
        let a = self.angular_velocity.normalize();
        let theta = self.angular_velocity.magnitude() * dt;
        //let dq = UnitQuaternion::new((theta * 0.5).cos(), a * (theta * 0.5).sin());
        let dq = UnitQuaternion::new(a * (theta * 0.5).sin());
        self.rotation = dq * self.rotation;

        self.force = Vector3::zeros();
        self.torque = Vector3::zeros();
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
    ) {
        self.torque += (point - self.position).cross(&force);
        self.force += force;
    }
    pub fn apply_force_at_offset(
        &mut self,
        force: Vector3<f32>,
        offset: Vector3<f32>,
    ) {
        self.torque += (offset).cross(&force);
        self.force += force;
    }

    pub fn new() -> Self{
        Self{
            mass: 0.001,
            lin_velocity: Vector3::zeros(),
            angular_velocity: Vector3::zeros(),
            force: Vector3::zeros(),
            torque: Vector3::zeros(),
            inertia_tensor: Matrix3::identity(),
            position: Vector3::zeros(),
            rotation: UnitQuaternion::from_axis_angle(&Vector3::x_axis(), 0.0),
        }
    }
}
