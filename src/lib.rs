use crate::rendering::RenderingState;
use nalgebra::{UnitQuaternion, Vector3};

use physics::pbd::PBDConstraintComponent;
use physics::pbd::PBDSolver;

use physics::rigid_body::RigidBody;

use rendering::model::DeformableModel;
use rendering::RenderSystem;
use specs::prelude::*;
use specs::Component;
use std::sync::Arc;
use std::sync::Mutex;
use std::time::{Duration, Instant};
use winit::event::{DeviceEvent, ElementState, Event, KeyboardInput, VirtualKeyCode, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoop};
use winit::window::WindowBuilder;

pub mod physics;
pub mod rendering;
pub mod resources;

pub struct Gravity;

pub struct PBDActor {
    pos: Vector3<f32>,
    mass: f32,
    mesh_vertex: Option<DeformableMeshVertex>,
    velocity: Vector3<f32>,
    pub force: Vector3<f32>,
}

pub struct PBDState {
    actors: Vec<PBDActor>,
    deformable_models: Vec<DeformableModel>,
    constraints: Vec<PBDConstraintComponent>,
}

impl<'a> System<'a> for Gravity {
    type SystemData = WriteStorage<'a, RigidBody>;

    fn run(&mut self, mut body: Self::SystemData) {
        for body in (&mut body).join() {
            body.force += Vector3::new(0.0, -9.81, 0.0);
        }
    }
}
pub struct DeformableUpdator;

impl<'a> System<'a> for DeformableUpdator {
    type SystemData = (
        Write<'a, DeformableModels>,
        ReadStorage<'a, Position>,
        ReadStorage<'a, DeformableMeshVertex>,
    );

    fn run(&mut self, (def_models, pos, def_vertices): Self::SystemData) {
        // let mut deformable_models = &mut def_models.0;
        // for (pos, def_vertex) in (&pos, &def_vertices).join() {
        //     deformable_models[def_vertex.model].meshes[def_vertex.mesh].vertices
        //         [def_vertex.vertex_index]
        //         .position = pos.into();
        // }
    }
}

#[derive(Component)]
#[storage(VecStorage)]
pub struct Position(Vector3<f32>);
impl Into<[f32; 3]> for &Position {
    fn into(self) -> [f32; 3] {
        self.0.into()
    }
}
#[derive(Default, Component)]
#[storage(VecStorage)]
pub struct DeformableMeshVertex {
    model: usize,
    mesh: usize,
    vertex_index: usize,
}

#[derive(Component)]
#[storage(VecStorage)]
pub struct Rotation(UnitQuaternion<f32>);

#[derive(Component)]
#[storage(VecStorage)]
pub struct RenderingInstance(usize);

#[derive(Component)]
#[storage(VecStorage)]
pub struct TriangleVertexIndex(usize);

#[derive(Default)]
pub struct DeltaTime(Duration);

#[derive(Default)]
pub struct DeformableModels(Vec<DeformableModel>);

#[derive(Default)]
pub struct Renderer(Option<Arc<Mutex<RenderingState>>>);

pub async fn run() {
    env_logger::init();

    let mut world = World::new();
    world.register::<Position>();
    world.register::<Rotation>();
    world.register::<RigidBody>();
    world.register::<RenderingInstance>();

    world.register::<DeformableMeshVertex>();
    world.register::<PBDConstraintComponent>();

    // for i in 0..100 {
    //     world
    //         .create_entity()
    //         .with(Position(Vector3::new(i as f32 * 0.5, 100.5, 0.0)))
    //         .with(Rotation(UnitQuaternion::identity()))
    //         .with(RigidBody::default())
    //         .with(RenderingInstance(0))
    //         .build();
    //     if i < 99 {
    //         world
    //             .create_entity()
    //             .with(Into::<PBDConstraintComponent>::into(
    //                 DistanceConstraint::new(i, i + 1, 0.5),
    //             ))
    //             .build();
    //     }
    // }

    // world
    //     .create_entity()
    //     .with(Into::<PBDConstraintComponent>::into(
    //         PinToPointConstraint::new(0, Vector3::new(1.0, 100.5, 0.0), 0.0),
    //     ))
    //     .build();

    world
        .create_entity()
        .with(Position(Vector3::new(0.0, 0.5, 0.0)))
        .with(Rotation(UnitQuaternion::identity()))
        .with(RigidBody::default())
        .with(DeformableMeshVertex {
            model: 0,
            mesh: 0,
            vertex_index: 0,
        })
        .build();

    world.insert(DeltaTime(Duration::new(0, 1 << 20)));

    let event_loop = EventLoop::new();

    let window = WindowBuilder::new().build(&event_loop).unwrap();

    let state = RenderingState::new(&window).await;

    world.insert(DeformableModels(vec![
        state.create_deformable_model("cube.obj").await,
    ]));

    let state_mutex = Arc::new(Mutex::new(state));
    world.insert(Renderer(Some(state_mutex.clone())));

    let mut dispatcher = DispatcherBuilder::new()
        .with(Gravity, "gravity", &[])
        .with(PBDSolver, "pbd_solver", &["gravity"])
        // .with_thread_local(DeformableUpdator)
        .with_thread_local(RenderSystem)
        .build();

    let mut last_render_time = Instant::now();
    event_loop.run(move |event, _, control_flow| {
        *control_flow = ControlFlow::Poll;
        match event {
            Event::DeviceEvent {
                event: DeviceEvent::MouseMotion { delta },
                ..
            } => {
                let mut state = state_mutex.lock().expect("Could not lock state_mutex");

                if state.mouse_pressed {
                    state.camera_controller.process_mouse(delta.0, delta.1);
                }
            }
            Event::RedrawRequested(window_id) if window_id == window.id() => {
                let now = Instant::now();
                let dt = now - last_render_time;
                let mut delta_time = world.write_resource::<DeltaTime>();
                *delta_time = DeltaTime(dt);
                drop(delta_time);
                let deformable_models = &mut world.write_resource::<DeformableModels>().0;

                last_render_time = now;
                dispatcher.dispatch(&world);
                let mut state = state_mutex.lock().expect("Could not lock state");
                state.update(dt);
                match state.render(deformable_models) {
                    Ok(_) => {}
                    Err(wgpu::SurfaceError::Lost) => {
                        let size = state.size;
                        state.resize(size)
                    }

                    // Out of memory:
                    Err(wgpu::SurfaceError::OutOfMemory) => *control_flow = ControlFlow::Exit,

                    Err(e) => eprintln!("{:?}", e),
                }
            }
            Event::MainEventsCleared => {
                window.request_redraw();
            }

            Event::WindowEvent {
                ref event,
                window_id,
            } if window_id == window.id() => match event {
                WindowEvent::Resized(physical_size) => {
                    state_mutex
                        .lock()
                        .expect("Could not lock mutex")
                        .resize(*physical_size);
                }
                WindowEvent::ScaleFactorChanged {
                    scale_factor: _,
                    new_inner_size,
                } => {
                    state_mutex
                        .lock()
                        .expect("Could not lock mutex")
                        .resize(**new_inner_size);
                }

                WindowEvent::CloseRequested
                | WindowEvent::KeyboardInput {
                    input:
                        KeyboardInput {
                            state: ElementState::Pressed,
                            virtual_keycode: Some(VirtualKeyCode::Escape),
                            ..
                        },
                    ..
                } => *control_flow = ControlFlow::Exit,
                _ => {
                    state_mutex
                        .lock()
                        .expect("Could not lock mutex")
                        .input(event);
                }
            },
            _ => {}
        }
    });
}
