use crate::rendering::RenderingState;
use nalgebra::{UnitQuaternion, Vector3};
use physics::constraints::Constraint;
use physics::constraints::ConstraintComponent;
use physics::constraints::ConstraintSolver;
use physics::constraints::PreviousConstraintSolution;
use physics::rigid_body::RigidBody;
use physics::rigid_body::RigidBodyStepSys;
use physics::InstanceData;
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

#[derive(Component)]
#[storage(VecStorage)]
pub struct Position(Vector3<f32>);

#[derive(Component)]
#[storage(VecStorage)]
pub struct Rotation(UnitQuaternion<f32>);

#[derive(Component)]
#[storage(VecStorage)]
pub struct RenderingInstance(usize);

#[derive(Default)]
pub struct DeltaTime(Duration);

#[derive(Default)]
pub struct Renderer(Option<Arc<Mutex<RenderingState>>>);

pub async fn run() {
    env_logger::init();

    let mut world = World::new();
    world.register::<Position>();
    world.register::<Rotation>();
    world.register::<RigidBody>();
    world.register::<ConstraintComponent>();
    world.register::<RenderingInstance>();

    world
        .create_entity()
        .with(Position(Vector3::zeros()))
        .with(Rotation(UnitQuaternion::identity()))
        .with(RigidBody::default())
        .with(ConstraintComponent(vec![Constraint::FixedPosition(
            physics::constraints::fixed_position_constraint::FixToPointConstraint {
                position: Vector3::new(1.0, 0.0, 1.0),
            },
        )]))
        .with(RenderingInstance(0))
        .build();

    world.insert(DeltaTime(Duration::new(0, 50)));
    world.insert(PreviousConstraintSolution(None));

    let event_loop = EventLoop::new();

    let window = WindowBuilder::new().build(&event_loop).unwrap();

    let mut state = RenderingState::new(&window).await;
    state.instances.push(InstanceData {
        model: state.create_model("cube.obj").await,
    });

    let state_mutex = Arc::new(Mutex::new(state));
    world.insert(Renderer(Some(state_mutex.clone())));

    let mut dispatcher = DispatcherBuilder::new()
        .with(ConstraintSolver, "constraint_solver", &[])
        .with(RigidBodyStepSys, "rigid_body_step", &["constraint_solver"])
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

                last_render_time = now;
                dispatcher.dispatch(&world);
                let mut state = state_mutex.lock().expect("Could not lock state");
                state.update(dt);
                match state.render() {
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
