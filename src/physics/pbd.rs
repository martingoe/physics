use crate::{DeltaTime, Position, Rotation};
use enum_dispatch::enum_dispatch;
use nalgebra::{Matrix3, Vector3};
use specs::{prelude::*, Component};

use super::rigid_body::RigidBody;

const K_D: f32 = 0.2;

#[derive(Debug, PartialEq, Eq)]
enum ConstraintType {
    Equality,
    Inequality,
}
#[enum_dispatch(PBDConstraintComponent)]
trait Constraint {
    fn num_bodies(&self) -> usize;

    fn bodies(&self) -> &Vec<usize>;
    fn constraint_type(&self) -> ConstraintType;

    fn get_output(&self, data: &[Vector3<f32>]) -> (f32, Vec<Vector3<f32>>);
}

pub struct DistanceConstraint {
    bodies: Vec<usize>,
    distance: f32,
}
impl DistanceConstraint {
    pub(crate) fn new(body_1: usize, body_2: usize, distance: f32) -> DistanceConstraint {
        DistanceConstraint {
            bodies: vec![body_1, body_2],
            distance,
        }
    }
}

impl Constraint for DistanceConstraint {
    fn num_bodies(&self) -> usize {
        2
    }

    fn bodies(&self) -> &Vec<usize> {
        &self.bodies
    }

    fn constraint_type(&self) -> ConstraintType {
        ConstraintType::Equality
    }

    fn get_output(&self, data: &[Vector3<f32>]) -> (f32, Vec<Vector3<f32>>) {
        let diff = data[self.bodies[0]] - data[self.bodies[1]];
        let normalized = diff.normalize();
        return (diff.norm() - self.distance, vec![normalized, -normalized]);
    }
}

pub struct PinToPointConstraint {
    bodies: Vec<usize>,
    point: Vector3<f32>,
    distance: f32,
}

impl PinToPointConstraint {
    pub(crate) fn new(body_1: usize, point: Vector3<f32>, distance: f32) -> PinToPointConstraint {
        PinToPointConstraint {
            bodies: vec![body_1],
            point,
            distance,
        }
    }
}

impl Constraint for PinToPointConstraint {
    fn num_bodies(&self) -> usize {
        1
    }

    fn bodies(&self) -> &Vec<usize> {
        &self.bodies
    }

    fn constraint_type(&self) -> ConstraintType {
        ConstraintType::Equality
    }

    fn get_output(&self, data: &[Vector3<f32>]) -> (f32, Vec<Vector3<f32>>) {
        let diff = data[self.bodies[0]] - self.point;
        return (
            (diff.norm() - self.distance).max(0.0),
            vec![diff.normalize()],
        );
    }
}

#[derive(Component)]
#[storage(VecStorage)]
#[enum_dispatch]
pub enum PBDConstraintComponent {
    DistanceConstraint,
    PinToPointConstraint,
}

pub struct PBDSolver;

impl<'a> System<'a> for PBDSolver {
    type SystemData = (
        Read<'a, DeltaTime>,
        ReadStorage<'a, PBDConstraintComponent>,
        WriteStorage<'a, RigidBody>,
        WriteStorage<'a, Position>,
        WriteStorage<'a, Rotation>,
    );

    fn run(&mut self, (dt, constraints, mut body, mut pos, mut rot): Self::SystemData) {
        let mut data = (&mut body, &mut pos, &mut rot).join().collect::<Vec<(
            &mut RigidBody,
            &mut Position,
            &mut Rotation,
        )>>();

        for (body, _, _) in &mut data {
            body.lin_velocity += dt.0.as_secs_f32() * body.inv_mass * body.force;
            body.force = Vector3::zeros();
        }
        dampen_velocity(&mut data);

        let mut p = data
            .iter()
            .map(|(body, p, _)| p.0.clone() + dt.0.as_secs_f32() * body.lin_velocity)
            .collect::<Vec<Vector3<f32>>>();

        for constraint in (&constraints).join() {
            let (c, dc_dp) = constraint.get_output(&p);
            let mut sum = 0.0;
            for (i, global_index) in constraint.bodies().iter().enumerate() {
                sum += data[*global_index].0.inv_mass * dc_dp[i].norm_squared();
            }
            let s = c / sum;
            let constraint_type = constraint.constraint_type();
            if constraint_type == ConstraintType::Equality && c != 0.0
                || constraint_type == ConstraintType::Inequality && c < 0.0
            {
                for (i, global_index) in constraint.bodies().iter().enumerate() {
                    p[*global_index] -= s * data[*global_index].0.inv_mass * dc_dp[i];
                }
            }
        }

        for (i, (body, pos, _)) in &mut data.iter_mut().enumerate() {
            body.lin_velocity = (p[i] - pos.0) / dt.0.as_secs_f32();

            pos.0 = p[i];
            if pos.0.y <= 0.0 {
                pos.0.y = 0.0;
            }
        }
    }
}

fn dampen_velocity(data: &mut [(&mut RigidBody, &mut Position, &mut Rotation)]) {
    if data.len() <= 1 {
        return;
    }
    let mut x_cm = Vector3::zeros();
    let mut v_cm = Vector3::zeros();
    let mut full_mass = 0.0;

    for (body, pos, _) in data.iter() {
        x_cm += pos.0 * body.mass;
        v_cm += body.lin_velocity * body.mass;
        full_mass += body.mass;
    }
    x_cm /= full_mass;

    v_cm /= full_mass;

    let mut l = Vector3::zeros();

    let mut i = Matrix3::zeros();
    for (body, pos, _) in data.iter() {
        let r = (pos.0 - x_cm).cross_matrix();
        l += r * (body.mass * body.lin_velocity);
        i += r * r.transpose() * body.mass;
    }
    let omega = i.try_inverse().unwrap_or_default() * l;

    for (body, pos, _) in data {
        let dv = v_cm + omega.cross(&(pos.0 - x_cm)) - body.lin_velocity;
        body.lin_velocity += K_D * dv;
    }
}
