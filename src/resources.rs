use wgpu::util::DeviceExt;

use crate::rendering::model::{
    DeformableMesh, DeformableModel, Material, Mesh, Model, ModelVertex,
};
use crate::rendering::texture::Texture;
use std::collections::HashMap;
use std::io::{BufReader, Cursor};

pub async fn load_string(file_name: &str) -> anyhow::Result<String> {
    let path = std::path::Path::new(env!("OUT_DIR"))
        .join("res")
        .join(file_name);
    let data = std::fs::read_to_string(path)?;
    Ok(data)
}

pub async fn load_binary(file_name: &str) -> anyhow::Result<Vec<u8>> {
    let path = std::path::Path::new(env!("OUT_DIR"))
        .join("res")
        .join(file_name);
    let data = std::fs::read(path)?;
    Ok(data)
}

pub async fn load_texture(
    file_name: &str,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
) -> anyhow::Result<Texture> {
    let data = load_binary(file_name).await?;
    Texture::from_bytes(device, queue, &data, file_name)
}

pub async fn load_model(
    file_name: &str,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    layout: &wgpu::BindGroupLayout,
) -> anyhow::Result<Model> {
    let obj_text = load_string(file_name).await?;
    let obj_cursor = Cursor::new(obj_text);
    let mut obj_reader = BufReader::new(obj_cursor);

    let (models, obj_materials) = tobj::load_obj_buf_async(
        &mut obj_reader,
        &tobj::LoadOptions {
            triangulate: true,
            single_index: true,
            ..Default::default()
        },
        |p| async move {
            let mat_text = load_string(&p).await.unwrap();
            tobj::load_mtl_buf(&mut BufReader::new(Cursor::new(mat_text)))
        },
    )
    .await?;

    let mut materials = Vec::new();
    for m in obj_materials? {
        let diffuse_texture = load_texture(&m.diffuse_texture, device, queue).await?;

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&diffuse_texture.view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&diffuse_texture.sampler),
                },
            ],
        });
        materials.push(Material {
            name: m.name,
            diffuse_texture,
            bind_group,
        })
    }

    let meshes = models
        .into_iter()
        .map(|m| {
            let vertices = (0..m.mesh.positions.len() / 3)
                .map(|i| ModelVertex {
                    position: [
                        m.mesh.positions[i * 3],
                        m.mesh.positions[i * 3 + 1],
                        m.mesh.positions[i * 3 + 2],
                    ],
                    tex_coords: [m.mesh.texcoords[i * 2], m.mesh.texcoords[i * 2 + 1]],
                    normal: [
                        m.mesh.normals[i * 3],
                        m.mesh.normals[i * 3 + 1],
                        m.mesh.normals[i * 3 + 2],
                    ],
                })
                .collect::<Vec<_>>();
            let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(&format!("{:?} Vertex Buffer", file_name)),
                contents: bytemuck::cast_slice(&vertices),
                usage: wgpu::BufferUsages::VERTEX,
            });
            let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(&format!("{:?} Index Buffer", file_name)),
                contents: bytemuck::cast_slice(&m.mesh.indices),
                usage: wgpu::BufferUsages::INDEX,
            });

            Mesh {
                name: file_name.to_string(),
                vertex_buffer,
                index_buffer,
                num_elements: m.mesh.indices.len() as u32,
                material: m.mesh.material_id.unwrap_or(0),
            }
        })
        .collect::<Vec<_>>();
    Ok(Model { meshes, materials })
}

pub async fn load_deformable_model(
    file_name: &str,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    layout: &wgpu::BindGroupLayout,
) -> anyhow::Result<DeformableModel> {
    // TODO: Repeated, clean up
    let obj_text = load_string(file_name).await?;
    let obj_cursor = Cursor::new(obj_text);
    let mut obj_reader = BufReader::new(obj_cursor);

    let (models, obj_materials) = tobj::load_obj_buf_async(
        &mut obj_reader,
        &tobj::LoadOptions {
            triangulate: true,
            single_index: false,
            ..Default::default()
        },
        // &tobj::OFFLINE_RENDERING_LOAD_OPTIONS,
        |p| async move {
            let mat_text = load_string(&p).await.unwrap();
            tobj::load_mtl_buf(&mut BufReader::new(Cursor::new(mat_text)))
        },
    )
    .await?;

    let mut materials = Vec::new();
    for m in obj_materials? {
        let diffuse_texture = load_texture(&m.diffuse_texture, device, queue).await?;

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&diffuse_texture.view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&diffuse_texture.sampler),
                },
            ],
        });
        materials.push(Material {
            name: m.name,
            diffuse_texture,
            bind_group,
        })
    }

    let meshes = models
        .iter()
        .map(|m| {
            let mut set = HashMap::<(u32, u32), u32>::new();
            let mut indices = Vec::new();
            let mut vertices = Vec::new();

            let mut pos_arr: Vec<Vec<u32>> = std::iter::repeat(vec![])
                .take(m.mesh.positions.len() / 3)
                .collect();

            let mut current_index = 0;
            for ((vertex, normal), texcoord) in m
                .mesh
                .indices
                .iter()
                .zip(m.mesh.normal_indices.iter())
                .zip(m.mesh.texcoord_indices.iter())
            {
                if let Some(index) = set.get(&(*vertex, *normal)) {
                    indices.push(*index as u32);
                } else {
                    set.insert((*vertex, *normal), current_index);

                    pos_arr
                        .get_mut(*vertex as usize)
                        .expect("Should be initialized")
                        .push(current_index);
                    indices.push(current_index as u32);
                    vertices.push(ModelVertex {
                        position: [
                            m.mesh.positions[*vertex as usize * 3],
                            m.mesh.positions[*vertex as usize * 3 + 1],
                            m.mesh.positions[*vertex as usize * 3 + 2],
                        ],
                        tex_coords: [
                            m.mesh.texcoords[*texcoord as usize * 2],
                            m.mesh.texcoords[*texcoord as usize * 2 + 1],
                        ],
                        normal: [
                            m.mesh.normals[*normal as usize * 3 as usize],
                            m.mesh.normals[*normal as usize * 3 + 1 as usize],
                            m.mesh.normals[*normal as usize * 3 + 2 as usize],
                        ],
                    });

                    current_index += 1;
                };
            }

            let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(&format!("{:?} Index Buffer", file_name)),
                contents: bytemuck::cast_slice(&indices),
                usage: wgpu::BufferUsages::INDEX,
            });

            let num_elements = indices.len() as u32;
            DeformableMesh {
                name: file_name.to_string(),
                vertices,
                vertex_buffer: None,
                indices,
                old_indices: m.mesh.indices.clone(),
                index_buffer,
                num_elements,
                material: m.mesh.material_id.unwrap_or(0),
                pos_arr,
            }
        })
        .collect::<Vec<_>>();
    Ok(DeformableModel { meshes, materials })
}
