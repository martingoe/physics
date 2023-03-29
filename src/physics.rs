use std::iter::zip;
use std::time::Duration;
use nalgebra::{Quaternion, UnitQuaternion, Vector3};
use crate::model::Model;
use wgpu::{util::DeviceExt,
           Device};
use crate::graphics::Instance;

use self::rigid_body::RigidBody;

pub mod rigid_body;
mod constraint_solving;

pub enum EntityComponent {
    RigidBodyEntity(RigidBody),
}

trait UpdatingEntityComponent {
    fn update_entity(&self, entity: &mut Entity);
}

pub struct Entity {
    pub position: Vector3<f32>,
    pub rotation: UnitQuaternion<f32>,
    pub components: Vec<EntityComponent>,
    pub instance: u32,
}

pub struct InstanceData {
    pub model: Model,
}

pub struct PhysicsState {
    pub(crate) entities: Vec<Entity>,
    pub(crate) instances: Vec<InstanceData>,
}

pub struct InstanceRenderData<'a> {
    pub model: &'a Model,
    pub instance_buffer: wgpu::Buffer,
    pub instance_count: u32,
}
impl PhysicsState {
    pub fn get_render_data(&self, device: &Device) -> Vec<InstanceRenderData> {
        let mut instance_data = Vec::with_capacity(self.instances.len());
        instance_data.resize_with(self.instances.len(), || Vec::new());
        for entity in &self.entities{
            instance_data[entity.instance as usize].push(Instance {
                position: entity.position,
                rotation: entity.rotation,
            }
                .to_raw());
        }
        zip(instance_data, &self.instances).map(|(instance_data_vec, instance)| {
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
        }).collect()
    }

    pub fn apply_gravity(&mut self) {
        for entity in self.entities.iter_mut() {
            for component  in entity.components.iter_mut(){
                match component {
                    EntityComponent::RigidBodyEntity(ref mut rigid_body) => {
                        rigid_body.apply_force_at_offset(Vector3::new(0.0, -0.00000981, 0.0), Vector3::new(0.0, 0.0, 1000.5));
                    }
                }
            }
        }
    }
    pub fn step(&mut self, dt: &Duration) {
        for entity in self.entities.iter_mut(){
            for component in entity.components.iter_mut(){
                match component {
                    EntityComponent::RigidBodyEntity(rigid_body) => {
                        rigid_body.step(dt);
                        entity.position = rigid_body.position;
                        entity.rotation = rigid_body.rotation;
                    }
                }
            }
        }
    }
}
