use std::ops::AddAssign;
use std::time::Duration;
use cgmath::{Deg, InnerSpace, Matrix3, Quaternion, Rotation, Rotation3, SquareMatrix, Vector3, Zero};

use super::Entity;

pub struct RigidBody {
    mass: f32,

    lin_velocity: Vector3<f32>,
    angular_velocity: Vector3<f32>,

    force: Vector3<f32>,
    torque: Vector3<f32>,

    inertia_tensor: Matrix3<f32>,

    pub(crate) position: Vector3<f32>,
    pub(crate) rotation: Quaternion<f32>,
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
        self.angular_velocity += self.inertia_tensor.invert().unwrap() * angular_momentum;
        // self.angular_velocity += self.torque.cross(self.lin_velocity) / self.mass * dt;
        let a = self.angular_velocity.normalize();
        let theta = self.angular_velocity.magnitude() * dt;
        let dq = Quaternion::from_sv((theta * 0.5).cos(), a * (theta * 0.5).sin());
        self.rotation = dq * self.rotation;

        self.force = Vector3::zero();
        self.torque = Vector3::zero();
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
        self.torque += (point - self.position).cross(force);
        self.force += force;
    }
    pub fn apply_force_at_offset(
        &mut self,
        force: Vector3<f32>,
        offset: Vector3<f32>,
    ) {
        self.torque += (offset).cross(force);
        self.force += force;
    }

    pub fn new() -> Self{
        Self{
            mass: 0.001,
            lin_velocity: Vector3::zero(),
            angular_velocity: Vector3::zero(),
            force: Vector3::zero(),
            torque: Vector3::zero(),
            inertia_tensor: Matrix3::identity(),
            position: Vector3::zero(),
            rotation: Quaternion::from_angle_x(Deg(0.0)),
        }
    }
}
