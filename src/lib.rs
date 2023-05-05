use std::time::Instant;
use nalgebra::{UnitQuaternion, Vector3};
use winit::event::{DeviceEvent, ElementState, Event, KeyboardInput, VirtualKeyCode, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoop};
use winit::window::WindowBuilder;
use crate::physics::constraints::{Constraints, ConstraintSolver};
use crate::physics::constraints::fixed_orientation_constraint::FixedOrientationConstraint;
use crate::physics::constraints::fixed_position_constraint::FixToPointConstraint;
use crate::physics::{Entity, InstanceData, PhysicsState};
use crate::physics::rigid_body::RigidBody;
use crate::rendering::RenderingState;

pub mod physics;
pub mod resources;
pub mod rendering;

pub async fn run(){
    env_logger::init();

    let mut rigid_body = RigidBody::new(0);
    rigid_body.position = Vector3::<f32>::new(1.0, 0.0, 0.0);
    rigid_body.rotation = UnitQuaternion::from_euler_angles(1.0, 0.0, 0.0);
    // let bodies = vec![EntityComponent::RigidBodyEntity(rigid_body)];
    let constraint = Constraints::FixedPosition(FixToPointConstraint { rigid_body: rigid_body.index, position: Vector3::new(0.0, 0.0, 0.0) });
    let constraint2 = Constraints::FixedOrientation(FixedOrientationConstraint { rigid_body: rigid_body.index, position: Vector3::new(0.0, 0.0, 0.0) });




    let event_loop = EventLoop::new();
    let window = WindowBuilder::new().build(&event_loop).unwrap();

    let mut state = RenderingState::new(&window).await;
    let mut physics_state = PhysicsState {
        entities: vec![Entity {
            body: rigid_body,
            instance: 0,
        }],
        instances: vec![InstanceData { model: state.create_model("cube.obj").await }],
        constraint_solver: ConstraintSolver { constraints: vec![constraint, constraint2] },
        previous_solution: None,
    };
    let mut last_render_time = Instant::now();
    event_loop.run(move |event, _, control_flow| {
        *control_flow = ControlFlow::Poll;
        match event {
            Event::DeviceEvent {
                event: DeviceEvent::MouseMotion { delta },
                ..
            } => {
                if state.mouse_pressed {
                    state.camera_controller.process_mouse(delta.0, delta.1);
                }
            }
            Event::RedrawRequested(window_id) if window_id == window.id() => {
                let now = Instant::now();
                let dt = now - last_render_time;
                last_render_time = now;
                physics_state.update(&dt);
                state.update(dt);
                match state.render(dt, &window, &physics_state) {
                    Ok(_) => {}
                    Err(wgpu::SurfaceError::Lost) => state.resize(state.size),
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
                    state.resize(*physical_size);
                }
                WindowEvent::ScaleFactorChanged {
                    scale_factor: _,
                    new_inner_size,
                } => {
                    state.resize(**new_inner_size);
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
                    state.input(event);
                }
            },
            _ => {}
        }
        state
            .imgui_winit_platform
            .handle_event(state.imgui.io_mut(), &window, &event);
    });
}