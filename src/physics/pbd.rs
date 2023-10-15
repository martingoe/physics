use std::{rc::Rc, time::Duration};

use crate::{rendering::model::DeformableModel, DeformableMeshVertex};
use enum_dispatch::enum_dispatch;
use nalgebra::{Matrix3, Vector3};

const K_D: f32 = 0.2;

pub struct PBDActor {
    pub(crate) pos: Vector3<f32>,
    pub(crate) mass: f32,
    pub(crate) inv_mass: f32,
    pub(crate) mesh_vertex: Option<Vec<DeformableMeshVertex>>,
    pub(crate) velocity: Vector3<f32>,
    pub force: Vector3<f32>,
}

pub struct PBDState {
    pub actors: Vec<PBDActor>,
    pub deformable_models: Vec<DeformableModel>,
    pub constraints: Vec<PBDConstraintComponent>,
}

impl PBDState {
    pub fn new() -> Self {
        Self {
            actors: vec![],
            deformable_models: vec![],
            constraints: vec![],
        }
    }

    fn apply_gravity(&mut self) {
        for body in &mut self.actors {
            body.force += Vector3::new(0.0, -9.81e-2, 0.0);
        }
    }

    fn update_deformable_models(&mut self) {
        for actor in self.actors.iter().filter(|a| a.mesh_vertex.is_some()) {
            let def_vertex = actor.mesh_vertex.as_ref().unwrap();
            for i in def_vertex {
                self.deformable_models[i.model].meshes[i.mesh].vertices[i.vertex_index].position =
                    actor.pos.into();
            }
        }
    }

    fn apply_constraints_pbd(&mut self, dt: &Duration) {
        for actor in &mut self.actors {
            actor.velocity += dt.as_secs_f32() * actor.inv_mass * actor.force;
            actor.force = Vector3::zeros();
        }
        self.dampen_velocity();

        let mut p = self
            .actors
            .iter()
            .map(|a| a.pos.clone() + dt.as_secs_f32() * a.velocity)
            .collect::<Vec<Vector3<f32>>>();

        for constraint in &self.constraints {
            let (c, dc_dp) = constraint.get_output(&p);
            let mut sum = 0.0;
            for (i, global_index) in constraint.bodies().iter().enumerate() {
                sum += self.actors[*global_index].inv_mass * dc_dp[i].norm_squared();
            }
            let s = c / sum;
            let constraint_type = constraint.constraint_type();
            if constraint_type == ConstraintType::Equality && c != 0.0
                || constraint_type == ConstraintType::Inequality && c < 0.0
            {
                for (i, global_index) in constraint.bodies().iter().enumerate() {
                    p[*global_index] -= constraint.get_stiffness()
                        * s
                        * self.actors[*global_index].inv_mass
                        * dc_dp[i];
                }
            }
        }

        for (i, actor) in &mut self.actors.iter_mut().enumerate() {
            actor.velocity = (p[i] - actor.pos) / dt.as_secs_f32();

            actor.pos = p[i];
            if actor.pos.y <= -1.0 {
                actor.pos.y = -1.0;
            }
        }
    }

    fn apply_constraints_xpbd(&mut self, dt: f32) {
        const SOLVE_ITERATIONS_XPBD: usize = 5;
        let dt = dt / SOLVE_ITERATIONS_XPBD as f32;
        for c in &mut self.constraints {
            c.clear_lambda();
        }
        for _ in 0..SOLVE_ITERATIONS_XPBD {
            let mut p = self
                .actors
                .iter()
                .map(|a| {
                    let mut x = a.pos + dt * a.velocity + dt * dt * a.inv_mass * a.force;
                    if x.y <= -3.0 {
                        x.y = -3.0
                    }
                    x
                })
                .collect::<Vec<Vector3<f32>>>();
            for constraint in &mut self.constraints {
                constraint.solve(&mut self.actors, &mut p, dt);
            }

            for (i, actor) in &mut self.actors.iter_mut().enumerate() {
                actor.velocity = (p[i] - actor.pos) / dt;

                actor.pos = p[i];
            }
        }
    }

    fn dampen_velocity(&mut self) {
        if self.actors.len() <= 1 {
            return;
        }
        let mut x_cm = Vector3::zeros();
        let mut v_cm = Vector3::zeros();
        let mut full_mass = 0.0;

        for actor in &self.actors {
            x_cm += actor.pos * actor.mass;
            v_cm += actor.velocity * actor.mass;
            full_mass += actor.mass;
        }
        x_cm /= full_mass;

        v_cm /= full_mass;

        let mut l = Vector3::zeros();

        let mut i = Matrix3::zeros();
        for actor in &self.actors {
            let r = (actor.pos - x_cm).cross_matrix();
            l += r * (actor.mass * actor.velocity);
            i += r * r.transpose() * actor.mass;
        }
        let omega = i.try_inverse().unwrap_or_default() * l;

        for actor in &mut self.actors {
            let dv = v_cm + omega.cross(&(actor.pos - x_cm)) - actor.velocity;
            actor.velocity += K_D * dv;
        }
    }

    pub(crate) fn step(&mut self, dt: f32) {
        self.apply_gravity();
        self.apply_constraints_xpbd(dt);
        self.update_deformable_models();
    }
}

#[derive(Debug, PartialEq, Eq)]
enum ConstraintType {
    Equality,
    Inequality,
}
#[enum_dispatch(PBDConstraintComponent)]
trait Constraint {
    fn num_bodies(&self) -> usize;

    fn bodies(&self) -> &Vec<usize>;
    fn solve(&mut self, actors: &mut Vec<PBDActor>, predicted_pos: &mut [Vector3<f32>], dt: f32);
    fn clear_lambda(&mut self);
    fn constraint_type(&self) -> ConstraintType;
    fn get_stiffness(&self) -> f32;
    fn get_dampening(&self) -> f32;

    fn get_output(&self, data: &[Vector3<f32>]) -> (f32, Vec<Vector3<f32>>);
}

pub struct DistanceConstraint {
    pub(crate) bodies: Vec<usize>,
    pub(crate) distance: f32,
    pub distance_squared: f32,
    pub(crate) lambda: f32,
}
impl DistanceConstraint {
    pub(crate) fn new(body_1: usize, body_2: usize, distance: f32) -> DistanceConstraint {
        DistanceConstraint {
            bodies: vec![body_1, body_2],
            distance,
            distance_squared: distance * distance,
            lambda: 0.0,
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
        return (
            diff.norm_squared() - self.distance_squared,
            vec![normalized, -normalized],
        );
    }

    fn get_stiffness(&self) -> f32 {
        0.0000001
    }

    fn get_dampening(&self) -> f32 {
        0.01
    }

    fn solve(&mut self, actors: &mut Vec<PBDActor>, predicted_pos: &mut [Vector3<f32>], dt: f32) {
        let (c, dc_dp) = self.get_output(predicted_pos);

        let dp_dc_times_delta = (predicted_pos[self.bodies[0]] - actors[self.bodies[0]].pos)
            .dot(&dc_dp[0])
            + (predicted_pos[self.bodies[1]] - actors[self.bodies[1]].pos).dot(&dc_dp[1]);

        let stiffness = self.get_stiffness() / (dt * dt);

        let x: f32 = actors[self.bodies[0]].inv_mass + actors[self.bodies[1]].inv_mass;

        let gamma_i = self.get_stiffness() * self.get_dampening() / dt;
        let d_lambda = (-c - stiffness * self.lambda - gamma_i * dp_dc_times_delta)
            / ((1.0 + gamma_i) * x + stiffness);

        let d_x_0 = actors[self.bodies[0]].inv_mass * &dc_dp[0] * d_lambda;

        let d_x_1 = actors[self.bodies[1]].inv_mass * &dc_dp[1] * d_lambda;

        predicted_pos[self.bodies[0]] += d_x_0;

        predicted_pos[self.bodies[1]] += d_x_1;
        self.lambda += d_lambda;
    }

    fn clear_lambda(&mut self) {
        self.lambda = 0.0;
    }
}

pub struct PinToPointConstraint {
    bodies: Vec<usize>,
    point: Vector3<f32>,
    distance: f32,
    lambda: f32,
}

impl PinToPointConstraint {
    pub(crate) fn new(body_1: usize, point: Vector3<f32>, distance: f32) -> PinToPointConstraint {
        PinToPointConstraint {
            bodies: vec![body_1],
            point,
            distance,
            lambda: 0.0,
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

    fn get_stiffness(&self) -> f32 {
        0.0001
    }

    fn get_dampening(&self) -> f32 {
        0.001
    }

    fn solve(&mut self, actors: &mut Vec<PBDActor>, predicted_pos: &mut [Vector3<f32>], dt: f32) {
        let (c, dc_dp) = self.get_output(predicted_pos);

        let dp_dc_times_delta =
            (predicted_pos[self.bodies[0]] - actors[self.bodies[0]].pos).dot(&dc_dp[0]);

        let stiffness = self.get_stiffness() / (dt * dt);

        let x: f32 = actors[self.bodies[0]].inv_mass;

        let gamma_i = self.get_stiffness() * self.get_dampening() / dt;
        let d_lambda = (-c - stiffness * self.lambda - gamma_i * dp_dc_times_delta)
            / ((1.0 + gamma_i) * x + stiffness);

        let d_x_0 = actors[self.bodies[0]].inv_mass * &dc_dp[0] * d_lambda;

        predicted_pos[self.bodies[0]] += d_x_0;
        self.lambda += d_lambda;
    }

    fn clear_lambda(&mut self) {
        self.lambda = 0.0;
    }
}

pub struct VolumeConstraint {
    bodies: Vec<usize>,
    initial_volume: f32,
    indices: Rc<[usize]>,
    lambda: f32,
    pressure: f32,
}
impl VolumeConstraint {
    pub(crate) fn new(
        bodies: Vec<usize>,
        indices: Vec<u32>,
        pressure: f32,
        initial_volume: f32,
    ) -> VolumeConstraint {
        VolumeConstraint {
            bodies,
            initial_volume,
            indices: indices
                .iter()
                .map(|i| *i as usize)
                .collect::<Vec<usize>>()
                .into(),
            lambda: 0.0,
            pressure,
        }
    }
}

impl Constraint for VolumeConstraint {
    fn num_bodies(&self) -> usize {
        self.bodies.len()
    }

    fn bodies(&self) -> &Vec<usize> {
        &self.bodies
    }

    fn constraint_type(&self) -> ConstraintType {
        ConstraintType::Equality
    }

    fn get_output(&self, data: &[Vector3<f32>]) -> (f32, Vec<Vector3<f32>>) {
        let mut sum = 0.0;

        let mut grad = std::iter::repeat(Vector3::zeros())
            .take(self.num_bodies())
            .collect::<Vec<Vector3<f32>>>();
        for i in 0..self.indices.len() / 3 {
            sum += data[self.bodies[self.indices[3 * i] as usize]]
                .cross(&data[self.bodies[self.indices[3 * i + 1] as usize]])
                .dot(&data[self.bodies[self.indices[3 * i + 2] as usize]]);

            grad[self.bodies[self.indices[3 * i]]] += data[self.bodies[self.indices[3 * i + 1]]]
                .cross(&data[self.bodies[self.indices[3 * i + 2]]]);
            grad[self.bodies[self.indices[3 * i + 1]]] += data
                [self.bodies[self.indices[3 * i + 2]]]
                .cross(&data[self.bodies[self.indices[3 * i]]]);
            grad[self.bodies[self.indices[3 * i + 2]]] += data[self.bodies[self.indices[3 * i]]]
                .cross(&data[self.bodies[self.indices[3 * i + 1]]]);
        }

        (sum - self.pressure * self.initial_volume, grad)
    }

    fn get_stiffness(&self) -> f32 {
        0.0000001
    }

    fn get_dampening(&self) -> f32 {
        0.001
    }

    fn solve(&mut self, actors: &mut Vec<PBDActor>, predicted_pos: &mut [Vector3<f32>], dt: f32) {
        let (c, dc_dp) = self.get_output(predicted_pos);

        let mut dp_dc_times_delta = 0.0;
        let mut x = 0.0;
        for (i, a) in self.bodies.iter().enumerate() {
            dp_dc_times_delta += dc_dp[i].dot(&(predicted_pos[*a] - actors[*a].pos));
            x += (dc_dp[i] * actors[*a].inv_mass).dot(&dc_dp[i]);
        }

        let stiffness = self.get_stiffness() / (dt * dt);

        let gamma_i = self.get_stiffness() * self.get_dampening() / dt;
        let d_lambda = (-c - stiffness * self.lambda - gamma_i * dp_dc_times_delta)
            / ((1.0 + gamma_i) * x + stiffness);

        for (i, a) in self.bodies.iter().enumerate() {
            let d_x_a = actors[*a].inv_mass * &dc_dp[i] * d_lambda;

            predicted_pos[*a] += d_x_a;
        }
        self.lambda += d_lambda;
    }

    fn clear_lambda(&mut self) {
        self.lambda = 0.0;
    }
}

#[enum_dispatch]
pub enum PBDConstraintComponent {
    DistanceConstraint,
    PinToPointConstraint,
    VolumeConstraint,
}
