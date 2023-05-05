use nalgebra::{Dyn, OVector, Vector3};
use std::iter::zip;
use std::time::Duration;
use wgpu::{util::DeviceExt, Device};
use crate::physics::constraints::ConstraintSolver;
use crate::rendering::graphics::Instance;
use crate::rendering::model::Model;

use self::rigid_body::RigidBody;

pub mod constraints;
pub mod rigid_body;
mod sle_solver;
mod sparse_matrix;

pub struct Entity {
    pub body: RigidBody,
    pub instance: u32,
}

pub struct InstanceData {
    pub model: Model,
}

pub struct PhysicsState {
    pub entities: Vec<Entity>,
    pub instances: Vec<InstanceData>,
    pub(crate) constraint_solver: ConstraintSolver,
    pub(crate) previous_solution: Option<OVector<f32, Dyn>>,

}



pub struct InstanceRenderData<'a> {
    pub model: &'a Model,
    pub instance_buffer: wgpu::Buffer,
    pub instance_count: u32,
}
impl PhysicsState {
    pub fn update(&mut self, dt: &Duration){
        self.apply_gravity();
        let lambda = self.constraint_solver.solve_constraints(&self, &self.previous_solution);

        if let Some((solution, matrix)) = lambda {
            self.previous_solution = Some(solution);
            for (i, column) in matrix.column_iter().enumerate() {
                self.entities[i].body.force += column.view((0, 0), (3, 1));
                self.entities[i].body.torque += column.view((3, 0), (3, 1));
            }
        }


        self.step(dt);
    }


    pub fn get_render_data(&self, device: &Device) -> Vec<InstanceRenderData> {
        let mut instance_data = Vec::with_capacity(self.instances.len());
        instance_data.resize_with(self.instances.len(), || Vec::new());
        for entity in &self.entities {
            instance_data[entity.instance as usize].push(
                Instance {
                    position: entity.body.position,
                    rotation: entity.body.rotation,
                }
                .to_raw(),
            );
        }
        zip(instance_data, &self.instances)
            .map(|(instance_data_vec, instance)| {
                let instance_buffer =
                    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("Instance Buffer"),
                        contents: bytemuck::cast_slice(&instance_data_vec),
                        usage: wgpu::BufferUsages::VERTEX,
                    });
                InstanceRenderData {
                    model: &instance.model,
                    instance_buffer,
                    instance_count: instance_data_vec.len() as u32,
                }
            })
            .collect()
    }

    pub fn apply_gravity(&mut self) {
        for entity in self.entities.iter_mut() {
            entity.body.apply_force_at_offset(
                Vector3::new(0.0, -9.81, 0.0),
                Vector3::new(0.0, 0.0, 1.5),
            );
        }
    }
    pub fn step(&mut self, dt: &Duration) {
        for entity in self.entities.iter_mut() {
            entity.body.step(dt);
        }
    }
}
