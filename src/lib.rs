use crate::rendering::RenderingState;

use physics::pbd::PBDState;

use std::sync::Arc;
use std::sync::Mutex;
use std::time::Instant;
use winit::event::{DeviceEvent, ElementState, Event, KeyboardInput, VirtualKeyCode, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoop};
use winit::window::WindowBuilder;

pub mod physics;
pub mod rendering;
pub mod resources;

#[derive(Debug)]
pub struct DeformableMeshVertex {
    model: usize,
    mesh: usize,
    vertex_index: usize,
}

pub async fn run() {
    env_logger::init();

    let event_loop = EventLoop::new();

    let window = WindowBuilder::new().build(&event_loop).unwrap();

    let pbd_state = PBDState::new();

    let mut state = RenderingState::new(&window, pbd_state).await;

    state.create_deformable_model("torus.obj").await;
    println!("Loaded data");

    let state_mutex = Arc::new(Mutex::new(state));

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

                last_render_time = now;
                let mut state = state_mutex.lock().expect("Could not lock state");
                state.update(dt, 1e-2);
                match state.render(dt, &window) {
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
