pub mod camera;
pub mod model;
pub mod physics;
pub mod resources;
pub mod texture;
pub mod graphics;


fn main() {
    pollster::block_on(physics_engine::run());
}
