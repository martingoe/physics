use crate::rendering::model::Model;

pub mod pbd;

pub struct InstanceData {
    pub model: Model,
}

pub struct InstanceRenderData {
    pub instance_index: usize,
    pub instance_buffer: wgpu::Buffer,
    pub instance_count: u32,
}
