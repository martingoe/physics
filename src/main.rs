pub mod camera;
pub mod model;
pub mod resources;
pub mod texture;

use graphics::run;

fn main() {
    pollster::block_on(run());
}
