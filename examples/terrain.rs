use glow::HasContext;
use minimg::ImageBuffer;
use minvect::*;
extern crate glow_mesh;
use glow_mesh::xyzrgbauv::*;
use glutin::event::{Event, WindowEvent, VirtualKeyCode, ElementState};
use std::collections::HashSet;
use std::f32::consts::PI;

pub struct Demo {
    xres: i32,
    yres: i32,
    window: glutin::ContextWrapper<glutin::PossiblyCurrent, glutin::window::Window>,
    gl: glow::Context,

    prog: ProgramXYZRGBAUV,
    h: HandleXYZRGBAUV,

    held_keys: HashSet<VirtualKeyCode>,

    pub cam_pos: Vec3,
    pub cam_polar_angle: f32,
    pub cam_azimuthal_angle: f32,
    pub lock_cursor: bool,
}

impl Demo {
    pub fn new(event_loop: &glutin::event_loop::EventLoop<()>) -> Self {
        let xres = 512;
        let yres = 512;
    
        unsafe {
            let window_builder = glutin::window::WindowBuilder::new()
                .with_title("uv test")
                .with_inner_size(glutin::dpi::PhysicalSize::new(xres, yres));
            let window = glutin::ContextBuilder::new()
                .with_vsync(true)
                .build_windowed(window_builder, &event_loop)
                .unwrap()
                .make_current()
                .unwrap();
    
            let gl = glow::Context::from_loader_function(|s| window.get_proc_address(s) as *const _);
    
            let prog = ProgramXYZRGBAUV::new(&gl, &DEFAULT_VS, &FS_UV_DEMO, &ImageBuffer::new(512, 512));

            let tmat = [
                0.1, 0.0, 0.0, 0.0,
                0.0, 0.1, 0.0, 0.0,
                0.0, 0.0, 10.0, 0.0,
                0.0, 0.0, 0.0, 1.0,
            ];

            // let tmat = [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0];
    
            let buf = &mut vec![];
            push_terrain_mesh(buf, 1000, 69);
            transform_mesh(buf, &tmat);
            let h = upload_xyzrgbauv_mesh(buf, &gl);
            prog.bind(&gl);

            Demo {
                xres,
                yres,
                window,
                gl,
                prog,
                h,
                cam_pos: vec3(0.0, -0.2, 0.0),
                cam_polar_angle: PI/2.0,
                cam_azimuthal_angle: 0.0,
                lock_cursor: false,
                held_keys: HashSet::new(),
            }
        }
    }

    pub fn handle_event(&mut self, event: glutin::event::Event<()>) {
        unsafe {
            match event {
                Event::LoopDestroyed |
                Event::WindowEvent {event: WindowEvent::CloseRequested, ..} => {
                    std::process::exit(0);
                },

                Event::WindowEvent {event, .. } => {
                    match event {
                        WindowEvent::Resized(size) => {
                            self.xres = size.width as i32;
                            self.yres = size.height as i32;
                            self.window.resize(size);
                            self.gl.viewport(0, 0, size.width as i32, size.height as i32);
                        },
                        WindowEvent::KeyboardInput {input, ..} => {
                            match input {
                                glutin::event::KeyboardInput {virtual_keycode: Some(code), state: ElementState::Pressed, ..} => {
                                    self.held_keys.insert(code);
                                    if code == VirtualKeyCode::Escape {
                                        self.lock_cursor = !self.lock_cursor;
                                    }
                                },
                                glutin::event::KeyboardInput {virtual_keycode: Some(code), state: ElementState::Released, ..} => {
                                    self.held_keys.remove(&code);
                                },
                                _ => {},
                            }
                        },
                        _ => {},
                    }
                },
                Event::MainEventsCleared => {
                    self.gl.clear_color(0.5, 0.5, 0.5, 1.0);
                    self.gl.clear(glow::COLOR_BUFFER_BIT | glow::DEPTH_BUFFER_BIT); 
                    self.simulate(0.016);
                    let cam_mat = cam_vp(self.cam_pos, self.cam_dir(), 2.0, self.xres as f32 / self.yres as f32, 0.01, 1000.0);
                    let mat4_ident = [1.0f32, 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1. ];
                    self.prog.set_proj(&cam_mat, &self.gl);
                    // self.prog.set_proj(&mat4_ident, &self.gl);
                    
                    println!("{} {} {} {}", self.cam_azimuthal_angle, self.cam_polar_angle, self.cam_dir(), self.cam_pos);
                    
                    
                    self.h.render(&self.gl);
                    self.window.swap_buffers().unwrap();

                },
                Event::DeviceEvent {device_id: _, event}  => {
                    match event {
                        glutin::event::DeviceEvent::MouseMotion { delta } => {
                            if self.lock_cursor {
                                self.turn_camera(vec2(delta.0 as f32, delta.1 as f32));
                            }
                        },
                        _ => {},
                    }
                },
                _ => {},
            }
        }
    }
}

pub fn main() {
        let event_loop = glutin::event_loop::EventLoop::new();
        let mut triangle_demo = Demo::new(&event_loop);
        event_loop.run(move |event, _, _| triangle_demo.handle_event(event));
}

pub const FS_UV_DEMO: &str = r#"#version 330 core
in vec4 col;
in vec2 uv;
out vec4 frag_colour;
uniform sampler2D tex;

void main() {
    frag_colour = vec4(uv.xy, 0.0, 1.0);
}
"#;

use glow_mesh::xyzrgbauv::*;
use minvect::*;
use std::f32::NEG_INFINITY;
use std::f32::INFINITY;

pub fn push_terrain_mesh(buf: &mut Vec<XYZRGBAUV>, s: usize, seed: u32) {
    for i in 0..s {
        for j in 0..s {
            // doing a quad
            let x = i as f32 / s as f32;
            let y = j as f32 / s as f32;
            let d = 1.0 / s as f32;

            let p00 = vec2(x, y);
            let p01 = vec2(x+d, y);
            let p10 = vec2(x, y+d);
            let p11 = vec2(x+d, y+d);

            let h00 = h(p00, seed);
            let h01 = h(p01, seed);
            let h10 = h(p10, seed);
            let h11 = h(p11, seed);

            let verts = [
                vert(p00, h00),
                vert(p01, h01),
                vert(p10, h10),
                vert(p11, h11),
            ];

            push_quad(buf, verts);
        }
    }
}
pub fn vert(p: Vec2, h: f32) -> XYZRGBAUV {
    XYZRGBAUV {
        xyz: vec3(p.x, h, p.y),
        rgba: colourmap(h),
        uv: p,
    }
}

fn id(v: &XYZRGBAUV) -> XYZRGBAUV {
    XYZRGBAUV { xyz: v.xyz, rgba: v.rgba, uv: v.uv }
}

pub fn push_quad(buf: &mut Vec<XYZRGBAUV>, verts: [XYZRGBAUV; 4]) {
    buf.push(id(&verts[0])); buf.push(id(&verts[1])); buf.push(id(&verts[3]));
    buf.push(id(&verts[1])); buf.push(id(&verts[2])); buf.push(id(&verts[3]));
}

pub fn colourmap(h: f32) -> Vec4 {
    let col_deep_water = vec4(0.0, 0.3, 0.6, 1.0);
    let col_shallow_water = vec4(0.0, 0.6, 0.8, 1.0);
    let col_plains = vec4(0.4, 0.8, 0.4, 1.0);
    let col_beach = vec4(0.8, 0.8, 0.3, 1.0);
    let col_mountain = vec4(0.5, 0.5, 0.5, 1.0);
    let col_snow = vec4(1.0, 1.0, 1.0, 1.0);
    let col_forest = vec4(0.1, 0.5, 0.1, 1.0);

    let h_gradient = vec![
        (NEG_INFINITY, col_deep_water),
        (0.0, col_deep_water),
        (0.6, col_shallow_water),
        (0.601, col_beach),
        (0.68, col_plains),
        (1.4, col_forest),
        (1.5, col_mountain),
        (2.0, col_snow),
        (INFINITY, col_snow),
    ];

    let col_h = gradient(&h_gradient, h);
    col_h
}

pub fn h(p: Vec2, seed: u32) -> f32 {
    use wavelet_noise::wavelet_noise::*;
    gen(64.0*p, seed) + 1.0
}

fn gradient(cols: &[(f32, Vec4)], t: f32) -> Vec4 {
    for i in 1..cols.len() {
        let lo = cols[i-1].0;
        let hi = cols[i].0;

        if t > lo && t <= hi {
            let range = hi-lo;
            let t = (t - lo) / range;
            return cols[i-1].1.lerp(cols[i].1, t);
        }
    }
    dbg!(cols);
    dbg!(t);
    panic!("bad gradient");
}

// camera
impl Demo {
    
    pub fn cam_right(&self) -> Vec3 {
        let up = vec3(0.0, 1.0, 0.0);
        up.cross(self.cam_dir()).normalize()
    }

    pub fn cam_up(&self) -> Vec3 {
        self.cam_right().cross(self.cam_dir()).normalize() // see if it works without normalize
    }

    // pub fn cam_dir(&self) -> Vec3 {
    //     vec3(
    //         self.cam_polar_angle.sin() * self.cam_azimuthal_angle.cos(),
    //         self.cam_polar_angle.sin() * self.cam_azimuthal_angle.sin(),
    //         self.cam_polar_angle.cos(),
    //     )
    // }

        // note: self.azimuthal_angle
        // note: self.polar_angle
        // note: north pole 0,1,0
    pub fn cam_dir(&self) -> Vec3 {
        vec3(
            self.cam_polar_angle.sin() * self.cam_azimuthal_angle.cos(),
            self.cam_polar_angle.cos(),
            self.cam_polar_angle.sin() * self.cam_azimuthal_angle.sin(),
        )
    }
 
    pub fn turn_camera(&mut self, r: Vec2) {
        // let mut spherical = self.cam_dir().cartesian_to_spherical();
        // let r2 = r * 0.001;
        // spherical.y += r2.y;
        // spherical.z += r2.x;
        // spherical.x = 1.0;
        // self.cam_dir = spherical.spherical_to_cartesian().normalize();

        let mut r = r * 0.001;
        // r.y *= -1.0;
        self.cam_polar_angle -= r.y;
        self.cam_polar_angle = self.cam_polar_angle.max(0.0).min(PI);
        self.cam_azimuthal_angle += r.x;

        // let inclination = self.cam_dir.y.acos();    // theta
        // let azimuth = -self.cam_dir.z.atan2(self.cam_dir.x); // not sure if need the -  // phi
        // let sin_theta = inclination.sin();
        // let cos_theta = inclination.cos();
        // let sin_phi = azimuth.sin();
        // let cos_phi = azimuth.cos();    // cam_dir.y
        // let rot_spherical = [
        //     cos_phi, 0.0, -sin_phi,
        //     0.0, 1.0, 0.0,
        //     sin_phi, 0.0, cos_phi,
        // ];

    }
    
    pub fn movement(&mut self, dir: Vec3, dt: f32) {
        let speed = 1.0;

        // let cam_right = (up.cross(self.cam_dir)).normalize();
        // let cam_up = cam_right.cross(self.cam_dir).normalize();

        let cam_dir = self.cam_dir();
        let cam_dir = vec3(cam_dir.x, 0.0, cam_dir.z).normalize();
        let cam_right = self.cam_right();
        let cam_up = self.cam_up();
        let v = dir.z * cam_dir + dir.y * cam_up + dir.x * cam_right;

        // but cam_dir projected into xz plane

        // let v = self.cam_dir() * dir.dot(self.cam_dir()) + self.cam_right() * dir.dot(self.cam_right()) + self.cam_up() * dir.dot(self.cam_up());

        self.cam_pos += dt * speed * v;
    }

    pub fn simulate(&mut self, dt: f32) {
        let x = if self.held_keys.contains(&VirtualKeyCode::A) {
            1.0f32
        } else if self.held_keys.contains(&VirtualKeyCode::D) {
            -1.0
        } else {
            0.0
        };
        let z = if self.held_keys.contains(&VirtualKeyCode::W) {
            1.0f32
        } else if self.held_keys.contains(&VirtualKeyCode::S) {
            -1.0
        } else {
            0.0
        };
        let y = if self.held_keys.contains(&VirtualKeyCode::LControl) {
            -1.0f32
        } else if self.held_keys.contains(&VirtualKeyCode::LShift) {
            1.0
        } else {
            0.0
        };
        use glutin::window::CursorIcon;
        self.movement(vec3(x, y, z).normalize(), dt);
        if self.lock_cursor {
            // self.window.window().set_cursor_position(winit::dpi::LogicalPosition::new(self.xres/2, self.yres/2))
            //     .expect("set_cursor_position");
            self.window.window().set_cursor_visible(false);
            self.window.window().set_cursor_icon(CursorIcon::Crosshair);
            self.window.window().set_cursor_grab(true)
                .expect("set_cursor_grab true");
        } else {
            self.window.window().set_cursor_icon(CursorIcon::Default);
            self.window.window().set_cursor_grab(false)
                .expect("set_cursor_grab false");
            self.window.window().set_cursor_visible(true);
        }
    }
}

// terrains been generated in uv space
// also camera looking doesnt seem to be working

pub fn transform_mesh(v: &mut Vec<XYZRGBAUV>, mat: &[f32; 16]) {
    for i in 0..v.len() {
        // transform the xyz
        v[i].xyz = homog_transform(v[i].xyz, mat);
    }
}

pub fn homog_transform(v: Vec3, mat: &[f32; 16]) -> Vec3 {
    let x = v.x * mat[0] + v.y * mat[4] + v.z * mat[8] + mat[12];
    let y = v.x * mat[1] + v.y * mat[5] + v.z * mat[9] + mat[13];
    let z = v.x * mat[2] + v.y * mat[6] + v.z * mat[10] + mat[14];

    vec3(x, y, z)
}
