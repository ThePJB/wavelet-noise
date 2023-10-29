use minvect::*;
use minirng::hash::*;
use std::f32::consts::PI;

pub fn smoothstep(x: f32) -> f32 {
    x*x*(3.0-2.0*x)
}
fn smooth_max(a: f32, b: f32, tau: f32) -> f32 {
    let e_a = (-a / tau).exp();
    let e_b = (-b / tau).exp();
    return (a * e_a + b * e_b) / (e_a + e_b);
}

fn smooth_max_array(arr: &[f32], tau: f32) -> f32 {
    let mut result = arr[0];
    for &value in &arr[1..] {
        // result = smooth_max(result, value, tau);
        result = result.max(value);
    }
    return result;
}


// gaussian rolloff, apply domain distortion to p
pub fn wavelet_amplitude(p: Vec2, freq: f32, initial_phase: f32) -> f32 {
    // get p in local coords
    // just rotate p basically
    // does that mean dir out. ye
    // so its going x axis
    let phase = initial_phase + p.x * freq;
    let amp_sin = phase.sin();
    let amp_env = (1.0 - p.dot(p).sqrt()).max(0.0);
    let amp_env = smoothstep(amp_env);
    amp_sin*amp_env
}

// tuples -> tuples and seeds -> positions and seeds

// look in 3x3 neighbourhood a la voronoise
// 1 control point per cell
// control points get to do shit: like eg width, height, direciton, amplitude, frequency, scale
// keep it local (window / gaussian)
// compose uh go with smooth min i guess

// also do u correlate directions of different octaves? could be interesting
// could have derivative for it as well, y not

/// Output domain is what exactly
pub fn gen(p: Vec2, seed: u32) -> f32 {
    let pfract = vec2(p.x.fract(), p.y.fract());
    let pfloor = p - pfract;

    let x = pfloor.x as i32;
    let y = pfloor.y as i32;

    let mut seeds = [0u32; 9];

    // something is wrong with how im doing the centers
    // i think its all ending up in 0 quadrant or something

    let mut height_contributions = [0.0f32; 9];
    let mut idx = 0;
    for i in (x-1)..=(x+1) {
        for j in (y-1)..=(y+1) {
            let mut seed = khash(seed.wrapping_add((i as u32).wrapping_mul(2138242137)).wrapping_add((j as u32).wrapping_mul(712214797)));
            seeds[idx] = seed;
            let c = vec2(next_f32(&mut seed), next_f32(&mut seed)) + vec2(i as f32, j as f32);
            let theta = next_f32(&mut seed) * 2.0 * PI;
            let x = next_f32(&mut seed);
            let freq = x * (11.5 - 2.5) + 2.5;
            // remember exponential distribution good
            // all about shifting the distributions
            let initial_phase = next_f32(&mut seed) * 2.0 * PI;
            let rt = next_f32(&mut seed) * 0.5 + 0.5;
            let rn = next_f32(&mut seed) * 0.4 + 0.8;
            // let rt = rng.uniform_float(1.0, 4.0);
            // let rn = rng.uniform_float(1.0, 6.0);
            // let rt = 4.0 / freq;
            // let rn = 0.5;


            
            // determine pt - p relative to wavelet and scaled
            // determine basis vectors of wavelet relative space. its like frame of reference in physics really.
            let v = c - p;
            // basis unit vectors:
            let st = theta.sin();
            let ct = theta.cos();
            let tu = vec2(ct, st);
            let nu = vec2(-st, ct);
            // basis vectors
            let tp = tu * rt;
            let np = nu * rn;

            // let tp = vec2(1.0, 0.0);
            // let np = vec2(0.0, 1.0);
            // let px = v proj tp
            // let py = v proj np
            // yea or would we sorta say its v * [tp np] which is tp dot v, np dot v. projeciton is more than just dot product

            // maybe needs a rotation for 'approach angle?'

            let pt = vec2(v.dot(tp), v.dot(np));
            // listen, if scaling is 1 then tp np dont matter. tu nu are rotation. if no rotation, tu is 10 and nu is 01. so v dot i is v.x and v dot j is v.y
            // maybe this part gets something wrong

            // let pt = pt * 0.5;
            let h = wavelet_amplitude(pt, freq, initial_phase);

            height_contributions[idx] = h;
            idx += 1;
        }
    }
    let mut acc = 0.0;
    for h in height_contributions {
        acc += h;
    }
    acc
}
pub fn gen_frac(mut p: Vec2, octaves: usize, a_scale: f32, f_scale: f32, seed: u32) -> f32 {
    let mut a = 1.0;
    let mut acc = 0.0;
    for i in 0..octaves {
        let seed = khash(seed + i as u32 * 218732137);
        acc += a * gen(p, seed);
        a *= a_scale;
        p = p*f_scale;
    }
    acc
}

#[cfg(test)]
mod test {
    use minimg::*;
    use minvect::*;
    use crate::wavelet_noise::*;

    #[test]
    pub fn test_wavelet_noise() {
        let w = 1024;
        let h = 1024;
        let mut buf = ImageBuffer::new(w, h);
        for i in 0..w {
            for j in 0..h {
                let p = vec2(i as f32 / w as f32, j as f32 / h as f32);
                let p = 20.0 * p;
                let height = gen(p, 69);
                let k = height / 2.0 + 0.5;
                buf.set(i, j, vec4(k, k, k, 1.0));
            }
        }
        buf.dump_to_file("test4.png");
    }
    
    #[test]
    pub fn test_fractal_wavelet_noise() {
        let w = 1024;
        let h = 1024;
        let mut buf = ImageBuffer::new(w, h);
        for i in 0..w {
            for j in 0..h {
                let p = vec2(i as f32 / w as f32, j as f32 / h as f32);
                let p = 20.0 * p;
                let height = gen_frac(p, 4, 0.5, 2.0, 69);
                let k = height / 2.0 + 0.5;
                buf.set(i, j, vec4(k, k, k, 1.0));
            }
        }
        buf.dump_to_file("fract1.png");
        let mut buf = ImageBuffer::new(w, h);
        for i in 0..w {
            for j in 0..h {
                let p = vec2(i as f32 / w as f32, j as f32 / h as f32);
                let p = 20.0 * p;
                let height = gen_frac(p, 6, 0.6, 1.5, 41);
                let k = height / 2.0 + 0.5;
                buf.set(i, j, vec4(k, k, k, 1.0));
            }
        }
        buf.dump_to_file("fract2.png");
    }
    
    #[test]
    pub fn test_wavelets() {
        let w = 1024;
        let h = 1024;
    
        let mut buf = ImageBuffer::new(w, h);
        for i in 0..w {
            for j in 0..h {
                let p = vec2(i as f32 / w as f32, j as f32 / h as f32);
                let p = p * 2.0 - vec2(1.0, 1.0);
                let height = wavelet_amplitude(p, 6.5, 0.31);
                let k = height / 2.0 + 0.5;
                buf.set(i, j, vec4(k, k, k, 1.0));
            }
        }
        buf.dump_to_file("wavelet3.png");
    }
}