use std::mem::swap;

use nalgebra::clamp;

pub struct FluidSimulation {
    vx: Vec<f32>,
    pub vx_prev: Vec<f32>,
    pub vy: Vec<f32>,
    pub vy_prev: Vec<f32>,

    vz: Vec<f32>,
    pub vz_prev: Vec<f32>,

    pub d: Vec<f32>,
    pub d_prev: Vec<f32>,

    n: usize,
}

fn index(n: usize, i: usize, j: usize, k: usize) -> usize {
    i + j * (n + 2) + k * (n + 2) * (n + 2)
}

impl FluidSimulation {
    fn add_source(n: usize, x: &mut Vec<f32>, s: &Vec<f32>, dt: f32) {
        for i in 0..(n + 2).pow(3) {
            x[i] += s[i] * dt;
        }
    }
    fn diffuse(n: usize, b: i32, x: &mut Vec<f32>, x_0: &Vec<f32>, diff: f32, dt: f32) {
        let a = dt * diff * (n as f32) * (n as f32) * (n as f32);

        gauss_seidel(n, x, x_0, a, b, 1.0 + 6.0 * a);
    }

    fn advect(
        n: usize,
        d: &mut Vec<f32>,
        d_prev: &Vec<f32>,
        vx: &Vec<f32>,
        vy: &Vec<f32>,
        vz: &Vec<f32>,
        b: i32,
        dt: f32,
    ) {
        let dt0 = (n as f32) * dt;
        for i in 1..=n {
            for j in 1..=n {
                for k in 1..=n {
                    let mut x = (i as f32) - dt0 * vx[index(n, i, j, k)];
                    let mut y = (j as f32) - dt0 * vy[index(n, i, j, k)];
                    let mut z = (k as f32) - dt0 * vz[index(n, i, j, k)];

                    x = clamp(x, 0.5, n as f32 + 0.5);
                    y = clamp(y, 0.5, n as f32 + 0.5);
                    z = clamp(z, 0.5, n as f32 + 0.5);
                    let i0 = x as usize;
                    let j0 = y as usize;
                    let k0 = z as usize;
                    let i1 = i0 + 1;
                    let j1 = j0 + 1;
                    let k1 = k0 + 1;

                    let d_x1 = x - i0 as f32;
                    let d_x0 = 1.0 - d_x1;
                    let d_y1 = y - j0 as f32;
                    let d_y0 = 1.0 - d_y1;
                    let d_z1 = z - k0 as f32;
                    let d_z0 = 1.0 - d_z1;

                    d[index(n, i, j, k)] = d_x1
                        * (d_y0
                            * (d_z0 * d_prev[index(n, i1, j0, k0)]
                                + d_z1 * d_prev[index(n, i1, j0, k1)])
                            + d_y1
                                * (d_z0 * d_prev[index(n, i1, j1, k0)]
                                    + d_z1 * d_prev[index(n, i1, j1, k1)]))
                        + d_x0
                            * (d_y0
                                * (d_z0 * d_prev[index(n, i0, j0, k0)]
                                    + d_z1 * d_prev[index(n, i0, j0, k1)])
                                + d_y1
                                    * (d_z0 * d_prev[index(n, i0, j1, k0)]
                                        + d_z1 * d_prev[index(n, i0, j1, k1)]))
                }
            }
        }
        Self::set_bnd(n, b, d);
    }

    pub fn dens_step(&mut self, diff: f32, dt: f32) {
        Self::add_source(self.n, &mut self.d, &self.d_prev, dt);
        swap(&mut self.d_prev, &mut self.d);
        Self::diffuse(self.n, 0, &mut self.d, &self.d_prev, diff, dt);
        swap(&mut self.d_prev, &mut self.d);
        Self::advect(
            self.n,
            &mut self.d,
            &self.d_prev,
            &self.vx,
            &self.vy,
            &self.vz,
            0,
            dt,
        );
    }

    pub fn vel_step(&mut self, visc: f32, dt: f32) {
        Self::add_source(self.n, &mut self.vx, &self.vx_prev, dt);
        Self::add_source(self.n, &mut self.vy, &self.vy_prev, dt);
        Self::add_source(self.n, &mut self.vz, &self.vz_prev, dt);
        swap(&mut self.vx, &mut self.vx_prev);

        Self::diffuse(self.n, 1, &mut self.vx, &self.vx_prev, visc, dt);
        swap(&mut self.vy, &mut self.vy_prev);
        Self::diffuse(self.n, 2, &mut self.vy, &self.vy_prev, visc, dt);
        swap(&mut self.vz, &mut self.vz_prev);
        Self::diffuse(self.n, 2, &mut self.vy, &self.vy_prev, visc, dt);

        self.project();
        swap(&mut self.vx, &mut self.vx_prev);
        swap(&mut self.vz, &mut self.vy_prev);
        swap(&mut self.vy, &mut self.vz_prev);

        Self::advect(
            self.n,
            &mut self.vx,
            &self.vx_prev,
            &self.vx_prev,
            &self.vy_prev,
            &self.vz_prev,
            1,
            dt,
        );

        Self::advect(
            self.n,
            &mut self.vy,
            &self.vy_prev,
            &self.vx_prev,
            &self.vy_prev,
            &self.vz_prev,
            2,
            dt,
        );
        Self::advect(
            self.n,
            &mut self.vz,
            &self.vz_prev,
            &self.vx_prev,
            &self.vy_prev,
            &self.vz_prev,
            3,
            dt,
        );
        self.project();
    }

    fn project(&mut self) {
        let h = 1.0 / (self.n as f32);
        for i in 1..=self.n {
            for j in 1..=self.n {
                for k in 1..=self.n {
                    // Use self.vx_prev as gradient array and misuse self.vy_prev as well
                    self.vx_prev[index(self.n, i, j, k)] = -h
                        * (self.vx[index(self.n, i + 1, j, k)]
                            - self.vx[index(self.n, i - 1, j, k)]
                            + self.vy[index(self.n, i, j + 1, k)]
                            - self.vy[index(self.n, i, j - 1, k)]
                            + self.vz[index(self.n, i, j, k + 1)]
                            - self.vz[index(self.n, i, j, k - 1)]);

                    self.vy_prev[index(self.n, i, j, k)] = 0.0;
                }
            }
        }

        Self::set_bnd(self.n, 0, &mut self.vx);
        Self::set_bnd(self.n, 0, &mut self.vy);

        gauss_seidel(self.n, &mut self.vy_prev, &self.vx_prev, 1.0, 0, 6.0);

        for i in 1..=self.n {
            for j in 1..=self.n {
                for k in 1..=self.n {
                    self.vx[index(self.n, i, j, k)] -= 0.5
                        * (self.n as f32)
                        * (self.vy_prev[index(self.n, i + 1, j, k)]
                            - self.vy_prev[index(self.n, i - 1, j, k)]);

                    self.vy[index(self.n, i, j, k)] -= 0.5
                        * (self.n as f32)
                        * (self.vy_prev[index(self.n, i, j + 1, k)]
                            - self.vy_prev[index(self.n, i, j - 1, k)]);
                    self.vz[index(self.n, i, j, k)] -= 0.5
                        * (self.n as f32)
                        * (self.vy_prev[index(self.n, i, j, k + 1)]
                            - self.vy_prev[index(self.n, i, j, k - 1)]);
                }
            }
        }
        Self::set_bnd(self.n, 1, &mut self.vx);
        Self::set_bnd(self.n, 2, &mut self.vy);
        Self::set_bnd(self.n, 3, &mut self.vz);
    }

    fn set_bnd(n: usize, b: i32, x: &mut Vec<f32>) {
        for i in 1..=n {
            for j in 1..=n {
                if b == 1 {
                    x[index(n, 0, i, j)] = -x[index(n, 1, i, j)];
                    x[index(n, n + 1, i, j)] = -x[index(n, n, i, j)];
                } else {
                    x[index(n, 0, i, j)] = x[index(n, 1, i, j)];
                    x[index(n, n + 1, i, j)] = x[index(n, n, i, j)];
                }

                if b == 2 {
                    x[index(n, i, 0, j)] = -x[index(n, i, 1, j)];
                    x[index(n, i, n + 1, j)] = -x[index(n, i, n, j)];
                } else {
                    x[index(n, i, 0, j)] = x[index(n, i, 1, j)];
                    x[index(n, i, n + 1, j)] = x[index(n, i, n, j)];
                }
                if b == 3 {
                    x[index(n, i, j, 0)] = -x[index(n, i, j, 1)];
                    x[index(n, i, j, n + 1)] = -x[index(n, i, j, n)];
                } else {
                    x[index(n, i, j, 0)] = x[index(n, i, j, 1)];
                    x[index(n, i, j, n + 1)] = x[index(n, i, j, n)];
                }
            }
        }
        for i in 1..=n {
            x[index(n, i, 0, 0)] = (x[index(n, i, 1, 0)] + x[index(n, i, 0, 1)]) / 2.0;
            x[index(n, i, 0, n + 1)] = (x[index(n, i, 1, n + 1)] + x[index(n, i, 0, n)]) / 2.0;
            x[index(n, i, n + 1, 0)] = (x[index(n, i, n, 0)] + x[index(n, i, n + 1, 1)]) / 2.0;
            x[index(n, i, n + 1, n + 1)] =
                (x[index(n, i, n, n + 1)] + x[index(n, i, n + 1, n)]) / 2.0;

            x[index(n, 0, i, 0)] = (x[index(n, 1, i, 0)] + x[index(n, 0, i, 1)]) / 2.0;
            x[index(n, n + 1, i, 0)] = (x[index(n, n, i, 0)] + x[index(n, n + 1, i, 0)]) / 2.0;
            x[index(n, 0, i, n + 1)] = (x[index(n, 0, i, n)] + x[index(n, 1, i, n + 1)]) / 2.0;

            x[index(n, n + 1, i, n + 1)] =
                (x[index(n, n + 1, i, n)] + x[index(n, n, i, n + 1)]) / 2.0;

            x[index(n, 0, 0, i)] = (x[index(n, 0, 1, i)] + x[index(n, 1, 0, i)]) / 2.0;
            x[index(n, n + 1, 0, i)] = (x[index(n, n + 1, 1, i)] + x[index(n, n, 0, i)]) / 2.0;
            x[index(n, 0, n + 1, i)] = (x[index(n, 0, n, i)] + x[index(n, 1, n + 1, i)]) / 2.0;

            x[index(n, n + 1, n + 1, i)] =
                (x[index(n, n + 1, n, i)] + x[index(n, n, n + 1, i)]) / 2.0;
        }
        x[index(n, 0, 0, 0)] =
            (x[index(n, 1, 0, 0)] + x[index(n, 0, 1, 0)] + x[index(n, 0, 0, 1)]) / 3.0;
        x[index(n, 0, 0, n + 1)] =
            (x[index(n, 1, 0, n + 1)] + x[index(n, 0, 1, n + 1)] + x[index(n, 0, 0, n)]) / 3.0;
        x[index(n, 0, n + 1, 0)] =
            (x[index(n, 1, n + 1, 0)] + x[index(n, 0, n, 0)] + x[index(n, 0, n + 1, 1)]) / 3.0;
        x[index(n, 0, n + 1, n + 1)] =
            (x[index(n, 1, n + 1, n + 1)] + x[index(n, 0, n, n + 1)] + x[index(n, 0, n + 1, n)])
                / 3.0;

        x[index(n, n + 1, 0, 0)] =
            (x[index(n, n, 0, 0)] + x[index(n, 0, 1, 0)] + x[index(n, 0, 0, 1)]) / 3.0;
        x[index(n, n + 1, 0, n + 1)] =
            (x[index(n, n, 0, n + 1)] + x[index(n, n + 1, 1, n + 1)] + x[index(n, n + 1, 0, n)])
                / 3.0;
        x[index(n, n + 1, n + 1, 0)] =
            (x[index(n, n, n + 1, 0)] + x[index(n, n + 1, n, 0)] + x[index(n, n + 1, n + 1, 1)])
                / 3.0;
        x[index(n, n + 1, n + 1, n + 1)] = (x[index(n, n, n + 1, n + 1)]
            + x[index(n, n + 1, n, n + 1)]
            + x[index(n, n + 1, n + 1, n)])
            / 3.0;
    }

    pub fn new(n: usize) -> FluidSimulation {
        Self {
            vx: vec![0.0; (n + 2).pow(3)],
            vx_prev: vec![0.0; (n + 2).pow(3)],
            vy: vec![0.0; (n + 2).pow(3)],
            vy_prev: vec![0.0; (n + 2).pow(3)],
            vz: vec![0.0; (n + 2).pow(3)],
            vz_prev: vec![0.0; (n + 2).pow(3)],
            d: vec![0.0; (n + 2).pow(3)],
            d_prev: vec![0.0; (n + 2).pow(3)],
            n,
        }
    }

    pub fn clear_prev_values(&mut self) {
        self.vx_prev.iter_mut().for_each(|x| *x = 0.0);
        self.vy_prev.iter_mut().for_each(|x| *x = 0.0);
        self.vz_prev.iter_mut().for_each(|x| *x = 0.0);
        self.d_prev.iter_mut().for_each(|x| *x = 0.0);
    }
}

fn gauss_seidel(n: usize, x: &mut Vec<f32>, x_0: &Vec<f32>, a: f32, b: i32, divisor: f32) {
    for _ in 0..20 {
        for i in 1..=n {
            for j in 1..=n {
                for k in 1..=n {
                    x[index(n, i, j, k)] = (x_0[index(n, i, j, k)]
                        + a * (x[index(n, i - 1, j, k)]
                            + x[index(n, i + 1, j, k)]
                            + x[index(n, i, j + 1, k)]
                            + x[index(n, i, j - 1, k)]
                            + x[index(n, i, j, k + 1)]
                            + x[index(n, i, j, k - 1)]))
                        / (divisor);
                }
            }
        }
        FluidSimulation::set_bnd(n, b, x);
    }
}
