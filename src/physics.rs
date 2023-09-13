use crate::rendering::model::Model;

pub mod pbd;
pub mod rigid_body;
mod sle_solver;
mod sparse_matrix;

pub struct InstanceData {
    pub model: Model,
}

pub struct InstanceRenderData {
    pub instance_index: usize,
    pub instance_buffer: wgpu::Buffer,
    pub instance_count: u32,
}
