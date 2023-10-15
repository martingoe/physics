use std::collections::HashSet;
use std::f32::consts::{FRAC_PI_2, FRAC_PI_8, PI};
use std::time::Duration;

use anyhow::Result;
use imgui::{Condition, FontSource};
use imgui_wgpu::{Renderer, RendererConfig};
use nalgebra::{Matrix4, Point3, UnitQuaternion, Vector3};
use wgpu::util::DeviceExt;
use wgpu::BindGroupLayout;
use winit::{event::*, window::Window};

use camera::Camera;
use model::DrawModel;

use crate::physics::pbd::{DistanceConstraint, PBDState, PinToPointConstraint, VolumeConstraint};
use crate::physics::{InstanceData, InstanceRenderData};
use crate::rendering::graphics::InstanceRaw;
use crate::rendering::model::{Model, Vertex};
use crate::rendering::texture::Texture;
use crate::resources;

use self::graphics::Instance;

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
    pub(crate) imgui: imgui::Context,
    imgui_renderer: Renderer,
    pub(crate) imgui_winit_platform: imgui_winit_support::WinitPlatform,
    pub texture_bind_group_layout: BindGroupLayout,
    pub instances: Vec<InstanceData>,
    pub instance_render_data: Vec<InstanceRenderData>,
    deformable_instance_buffer: wgpu::Buffer,
    pub pbd_state: PBDState,
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

    pub(crate) async fn new(window: &Window, pbd_state: PBDState) -> Self {
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

        let camera = Camera::new(Point3::new(0.0, 1.0, 8.0), -FRAC_PI_2, -PI / 20.0);
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
        let mut imgui = imgui::Context::create();
        let mut platform = imgui_winit_support::WinitPlatform::init(&mut imgui);
        platform.attach_window(
            imgui.io_mut(),
            &window,
            imgui_winit_support::HiDpiMode::Default,
        );

        let font_size = (13.0 * window.scale_factor()) as f32;
        imgui.set_ini_filename(None);
        imgui.fonts().add_font(&[FontSource::DefaultFontData {
            config: Some(imgui::FontConfig {
                size_pixels: font_size,
                oversample_h: 1,
                pixel_snap_h: true,
                ..Default::default()
            }),
        }]);

        let renderer_config = RendererConfig {
            texture_format: config.format,
            ..Default::default()
        };

        let imgui_renderer = Renderer::new(&mut imgui, &device, &queue, renderer_config);

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
            pbd_state,
            imgui,
            imgui_renderer,
            imgui_winit_platform: platform,
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

    pub async fn create_deformable_model(&mut self, name: &str) {
        let res = resources::load_deformable_model(
            name,
            &self.device,
            &self.queue,
            &self.texture_bind_group_layout,
        )
        .await
        .unwrap();
        // Add actors for vertices
        let first_actor_index = self.pbd_state.actors.len();
        for (mesh_index, m) in res.meshes.iter().enumerate() {
            self.pbd_state.actors.reserve(m.pos_arr.len());
            for vertex_leader in &m.pos_arr {
                self.pbd_state.actors.push(crate::physics::pbd::PBDActor {
                    pos: m.vertices[vertex_leader[0] as usize].position.into(),
                    mass: 0.1,
                    inv_mass: 10.0,
                    mesh_vertex: Some(
                        vertex_leader
                            .iter()
                            .map(|i| crate::DeformableMeshVertex {
                                model: self.pbd_state.deformable_models.len(),
                                mesh: mesh_index,
                                vertex_index: *i as usize,
                            })
                            .collect(),
                    ),
                    velocity: Vector3::zeros(),
                    force: Vector3::zeros(),
                });
            }
            let mut neighbors = HashSet::<(u32, u32)>::new();

            for i in (0..m.old_indices.len()).step_by(3) {
                let i = i as usize;

                insert_pair(&mut neighbors, (m.old_indices[i], m.old_indices[i + 1]));
                insert_pair(&mut neighbors, (m.old_indices[i + 1], m.old_indices[i + 2]));
                insert_pair(&mut neighbors, (m.old_indices[i], m.old_indices[i + 2]));
            }

            let mut initial_volume = 0.0;
            for i in 0..m.old_indices.len() / 3 {
                initial_volume += Into::<Vector3<f32>>::into(
                    m.vertices[m.pos_arr[m.old_indices[3 * i] as usize][0] as usize].position,
                )
                .cross(&Into::<Vector3<f32>>::into(
                    m.vertices[m.pos_arr[m.old_indices[3 * i + 1] as usize][0] as usize].position,
                ))
                .dot(&Into::<Vector3<f32>>::into(
                    m.vertices[m.pos_arr[m.old_indices[3 * i + 2] as usize][0] as usize].position,
                ));
            }
            self.pbd_state.constraints.push(
                crate::physics::pbd::PBDConstraintComponent::VolumeConstraint(
                    VolumeConstraint::new(
                        (first_actor_index..first_actor_index + m.pos_arr.len())
                            .collect::<Vec<usize>>(),
                        m.old_indices.clone(),
                        1.0,
                        initial_volume,
                    ),
                ),
            );

            self.pbd_state.constraints.reserve(neighbors.len());
            for (i, j) in neighbors.iter() {
                self.pbd_state.constraints.push(
                    crate::physics::pbd::PBDConstraintComponent::DistanceConstraint(
                        DistanceConstraint::new(
                            *i as usize + first_actor_index,
                            *j as usize + first_actor_index,
                            (Into::<Vector3<f32>>::into(
                                m.vertices[m.pos_arr[*i as usize][0] as usize].position,
                            ) - Into::<Vector3<f32>>::into(
                                m.vertices[m.pos_arr[*j as usize][0] as usize].position,
                            ))
                            .norm(),
                        ),
                    ),
                )
            }
        }

        self.pbd_state.deformable_models.push(res);
    }

    pub(crate) fn render(
        &mut self,
        dt: Duration,
        window: &Window,
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

            for model in &mut self.pbd_state.deformable_models {
                for mesh in &mut model.meshes {
                    mesh.vertex_buffer = Some(self.device.create_buffer_init(
                        &wgpu::util::BufferInitDescriptor {
                            label: Some(&format!("{:?} Vertex Buffer", mesh.name)),
                            contents: bytemuck::cast_slice(&mesh.vertices),
                            usage: wgpu::BufferUsages::VERTEX,
                        },
                    ));
                    render_pass.set_vertex_buffer(1, self.deformable_instance_buffer.slice(..));
                    // render_pass
                    // .set_vertex_buffer(1, mesh.vertex_buffer.as_ref().unwrap().slice(..));

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
        {
            self.imgui.io_mut().update_delta_time(dt);

            self.imgui_winit_platform
                .prepare_frame(self.imgui.io_mut(), window)
                .expect("Failed to prepare frame");

            self.prepare_imgui(dt);
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("imGUI Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,

                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: true,
                    },
                })],
                depth_stencil_attachment: None,
            });
            self.imgui_renderer
                .render(
                    self.imgui.render(),
                    &self.queue,
                    &self.device,
                    &mut render_pass,
                )
                .expect("Could not render imgui");
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

    pub(crate) fn update(&mut self, dt: Duration, physics_dt: f32) {
        self.pbd_state.step(physics_dt);
        self.camera_controller.update_camera(&mut self.camera, dt);
        self.camera_uniform
            .update_view_proj(&self.camera, &self.projection);
        self.queue.write_buffer(
            &self.camera_buffer,
            0,
            bytemuck::cast_slice(&[self.camera_uniform]),
        );
    }
    fn prepare_imgui(&mut self, dt: Duration) {
        let ui = self.imgui.frame();

        {
            let window = ui.window("Hello world");
            window
                .size([300.0, 100.0], Condition::FirstUseEver)
                .build(|| {
                    ui.text("Hello world!");
                    ui.text("This...is...imgui-rs on WGPU!");
                    ui.input_float("dt", &mut 0.0001);
                    ui.separator();
                    let is_clicked = ui.button("test");
                    if is_clicked {
                        ui.text("clicked the button");
                    }
                    let mouse_pos = ui.io().mouse_pos;
                    ui.text(format!(
                        "Mouse Position: ({:.1},{:.1})",
                        mouse_pos[0], mouse_pos[1]
                    ));
                });

            let window = ui.window("Hello too");

            window
                .size([400.0, 200.0], Condition::FirstUseEver)
                .position([400.0, 200.0], Condition::FirstUseEver)
                .build(|| {
                    ui.text(format!("FPS: {:?}", 1.0 / dt.as_secs_f32()));
                });
        }
    }
}

fn insert_pair(neighbors: &mut HashSet<(u32, u32)>, i: (u32, u32)) {
    if neighbors.contains(&i) || neighbors.contains(&(i.1, i.0)) {
        return;
    }
    neighbors.insert(i);
}
