use specs::prelude::*;
use std::f32::consts::{FRAC_PI_2, FRAC_PI_8, PI};
use std::time::Duration;

use anyhow::Result;
use nalgebra::{Matrix4, Point3, UnitQuaternion, Vector3};
use specs::System;
use wgpu::util::DeviceExt;
use wgpu::BindGroupLayout;
use winit::{event::*, window::Window};

use camera::Camera;
use model::DrawModel;

use crate::physics::{InstanceData, InstanceRenderData};
use crate::rendering::graphics::InstanceRaw;
use crate::rendering::model::{Model, Vertex};
use crate::rendering::texture::Texture;
use crate::{resources, Position, RenderingInstance, Rotation};

use self::graphics::Instance;
use self::model::DeformableModel;

pub mod graphics;
pub mod texture;

mod camera;
pub mod model;

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct CameraUniform {
    view_proj: [[f32; 4]; 4],
}

impl CameraUniform {
    fn new() -> Self {
        Self {
            view_proj: Matrix4::identity().into(),
        }
    }

    fn update_view_proj(&mut self, camera: &Camera, projection: &camera::Projection) {
        self.view_proj = (projection.calc_matrix() * camera.calc_matrix()).into();
    }
}

pub struct RenderingState {
    surface: wgpu::Surface,
    pub(crate) device: wgpu::Device,
    pub(crate) queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    pub(crate) size: winit::dpi::PhysicalSize<u32>,
    render_pipeline: wgpu::RenderPipeline,
    camera: Camera,
    projection: camera::Projection,
    pub(crate) camera_controller: camera::CameraController,
    camera_uniform: CameraUniform,
    camera_buffer: wgpu::Buffer,
    camera_bind_group: wgpu::BindGroup,
    depth_texture: Texture,
    pub(crate) mouse_pressed: bool,
    // pub(crate) imgui: imgui::Context,
    // imgui_renderer: Renderer,
    // pub(crate) imgui_winit_platform: imgui_winit_support::WinitPlatform,
    pub texture_bind_group_layout: BindGroupLayout,
    pub instances: Vec<InstanceData>,
    pub instance_render_data: Vec<InstanceRenderData>,
    deformable_instance_buffer: wgpu::Buffer,
}

pub struct RenderSystem;

impl<'a> System<'a> for RenderSystem {
    type SystemData = (
        ReadStorage<'a, Position>,
        ReadStorage<'a, Rotation>,
        ReadStorage<'a, RenderingInstance>,
        Read<'a, crate::Renderer>,
    );

    fn run(&mut self, (pos, rot, ins, renderer): Self::SystemData) {
        let renderer_mutex = renderer.0.as_ref().expect("Did not find renderer");
        let mut renderer = renderer_mutex.lock().expect("Could not lock renderer");

        let mut instances = Vec::with_capacity(renderer.instances.len());
        // TODO: Is this bad?
        instances.resize_with(renderer.instances.len(), || Vec::new());
        for (pos, rot, ins) in (&pos, &rot, &ins).join() {
            instances[ins.0].push(
                Instance {
                    position: pos.0,
                    rotation: rot.0,
                }
                .to_raw(),
            );
        }

        renderer.instance_render_data = instances
            .iter()
            .enumerate()
            .map(|(i, instance_data_vec)| {
                let instance_buffer =
                    renderer
                        .device
                        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                            label: Some("Instance Buffer"),
                            contents: bytemuck::cast_slice(&instance_data_vec),
                            usage: wgpu::BufferUsages::VERTEX,
                        });
                InstanceRenderData {
                    instance_index: i,
                    instance_count: instance_data_vec.len() as u32,
                    instance_buffer,
                }
            })
            .collect();
    }
}

impl RenderingState {
    pub(crate) fn input(&mut self, event: &WindowEvent) -> bool {
        match event {
            WindowEvent::KeyboardInput {
                input:
                    KeyboardInput {
                        state,
                        virtual_keycode: Some(key),
                        ..
                    },
                ..
            } => self.camera_controller.process_keyboard(*key, *state),
            WindowEvent::MouseWheel { delta, .. } => {
                self.camera_controller.process_scroll(delta);
                true
            }
            WindowEvent::MouseInput {
                state,
                button: MouseButton::Right,
                ..
            } => {
                self.mouse_pressed = *state == ElementState::Pressed;
                true
            }
            _ => false,
        }
    }

    pub(crate) async fn new(window: &Window) -> Self {
        let size = window.inner_size();

        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        let surface = unsafe { instance.create_surface(window) }.unwrap();
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptionsBase {
                power_preference: wgpu::PowerPreference::default(),
                force_fallback_adapter: false,
                compatible_surface: Some(&surface),
            })
            .await
            .unwrap();

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: None,
                    features: wgpu::Features::empty(),
                    limits: if cfg!(target_arch = "wasm32") {
                        wgpu::Limits::downlevel_webgl2_defaults()
                    } else {
                        wgpu::Limits::default()
                    },
                },
                None,
            )
            .await
            .unwrap();

        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: wgpu::TextureFormat::Bgra8UnormSrgb,
            width: size.width,
            height: size.height,
            present_mode: wgpu::PresentMode::Fifo,
            alpha_mode: wgpu::CompositeAlphaMode::Auto,
            view_formats: vec![wgpu::TextureFormat::Bgra8UnormSrgb],
        };
        surface.configure(&device, &config);

        let camera = Camera::new(Point3::new(0.0, 1.0, 2.0), -FRAC_PI_2, -PI / 20.0);
        let projection =
            camera::Projection::new(config.width, config.height, FRAC_PI_8, 0.1, 100.0);
        let camera_controller = camera::CameraController::new(4.0, 0.4);
        let mut camera_uniform = CameraUniform::new();
        camera_uniform.update_view_proj(&camera, &projection);

        let camera_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Camera Buffer"),
            contents: bytemuck::cast_slice(&[camera_uniform]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let camera_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("camerad_bind_group_layout"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });

        let camera_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("camera_bind_group"),
            layout: &camera_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: camera_buffer.as_entire_binding(),
            }],
        });

        let texture_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("texture_bind_group_layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                ],
            });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shader.wgsl").into()),
        });

        let depth_texture = Texture::create_depth_texture(&device, &config, "depth_texture");

        let render_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Render Pipeline Layout"),
                bind_group_layouts: &[&texture_bind_group_layout, &camera_bind_group_layout],
                push_constant_ranges: &[],
            });

        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Render Pipeline"),
            layout: Some(&render_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_main",
                buffers: &[model::ModelVertex::desc(), InstanceRaw::desc()],
            },
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: Some(wgpu::Face::Back),
                unclipped_depth: false,
                polygon_mode: wgpu::PolygonMode::Fill,
                conservative: false,
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: Texture::DEPTH_FORMAT,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fs_main",
                targets: &[Some(wgpu::ColorTargetState {
                    format: config.format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            multiview: None,
        });

        // imGUI
        // let mut imgui = imgui::Context::create();
        // let mut platform = imgui_winit_support::WinitPlatform::init(&mut imgui);
        // platform.attach_window(
        //     imgui.io_mut(),
        //     &window,
        //     imgui_winit_support::HiDpiMode::Default,
        // );

        // let font_size = (13.0 * window.scale_factor()) as f32;
        // imgui.set_ini_filename(None);
        // imgui.fonts().add_font(&[FontSource::DefaultFontData {
        //     config: Some(imgui::FontConfig {
        //         size_pixels: font_size,
        //         oversample_h: 1,
        //         pixel_snap_h: true,
        //         ..Default::default()
        //     }),
        // }]);

        // let renderer_config = RendererConfig {
        //     texture_format: config.format,
        //     ..Default::default()
        // };

        // let imgui_renderer = Renderer::new(&mut imgui, &device, &queue, renderer_config);

        let instances = vec![Instance {
            position: Vector3::zeros(),
            rotation: UnitQuaternion::default(),
        }
        .to_raw()];

        let deformable_instance_buffer =
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Instance Buffer"),
                contents: bytemuck::cast_slice(&instances),
                usage: wgpu::BufferUsages::VERTEX,
            });

        Self {
            surface,
            device,
            queue,
            config,
            size,
            render_pipeline,
            camera,
            projection,
            camera_controller,
            camera_uniform,
            camera_buffer,
            camera_bind_group,
            depth_texture,
            mouse_pressed: false,
            texture_bind_group_layout,
            instances: Vec::new(),
            instance_render_data: Vec::new(),
            deformable_instance_buffer,
        }
    }
    pub async fn create_model(&self, name: &str) -> Model {
        resources::load_model(
            name,
            &self.device,
            &self.queue,
            &self.texture_bind_group_layout,
        )
        .await
        .unwrap()
    }

    pub async fn create_deformable_model(&self, name: &str) -> DeformableModel {
        resources::load_deformable_model(
            name,
            &self.device,
            &self.queue,
            &self.texture_bind_group_layout,
        )
        .await
        .unwrap()
    }

    pub(crate) fn render(
        &mut self,
        deformable_models: &mut Vec<DeformableModel>,
    ) -> Result<(), wgpu::SurfaceError> {
        let output = self.surface.get_current_texture()?;
        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Render Encoder"),
            });

        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.0,
                            g: 0.0,
                            b: 0.0,
                            a: 1.0,
                        }),
                        store: true,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.depth_texture.view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: true,
                    }),
                    stencil_ops: None,
                }),
            });

            render_pass.set_pipeline(&self.render_pipeline);
            for instance_render_data in &self.instance_render_data {
                render_pass.set_vertex_buffer(1, instance_render_data.instance_buffer.slice(..));

                render_pass.draw_model_instanced(
                    &self.instances[instance_render_data.instance_index].model,
                    0..instance_render_data.instance_count,
                    &self.camera_bind_group,
                );
            }

            for model in deformable_models {
                for mesh in &mut model.meshes {
                    render_pass.set_vertex_buffer(1, self.deformable_instance_buffer.slice(..));
                    mesh.vertex_buffer = Some(self.device.create_buffer_init(
                        &wgpu::util::BufferInitDescriptor {
                            label: Some(&format!("{:?} Vertex Buffer", mesh.name)),
                            contents: bytemuck::cast_slice(&mesh.vertices),
                            usage: wgpu::BufferUsages::VERTEX,
                        },
                    ));
                    render_pass.draw_deformable_mesh_instanced(
                        mesh,
                        &model.materials[mesh.material],
                        0..1,
                        &self.camera_bind_group,
                    )
                }
            }

            drop(render_pass);
        }

        self.queue.submit(std::iter::once(encoder.finish()));
        output.present();

        Ok(())
    }

    pub(crate) fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        if new_size.width > 0 && new_size.height > 0 {
            self.size = new_size;
            self.config.width = new_size.width;
            self.config.height = new_size.height;
            self.surface.configure(&self.device, &self.config);
            self.depth_texture =
                Texture::create_depth_texture(&self.device, &self.config, "depth_texture");
            self.projection.resize(new_size.width, new_size.height);
        }
    }

    pub(crate) fn update(&mut self, dt: Duration) {
        self.camera_controller.update_camera(&mut self.camera, dt);
        self.camera_uniform
            .update_view_proj(&self.camera, &self.projection);
        self.queue.write_buffer(
            &self.camera_buffer,
            0,
            bytemuck::cast_slice(&[self.camera_uniform]),
        );
    }
}
