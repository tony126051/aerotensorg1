#![no_std]

use axcompute::ops::MatMul;
use axcompute::tensor::{DataType, Tensor};

/// Fast Inverse Square Root (Newton-Raphson approximation)
#[inline(always)]
pub fn fast_rsqrt(number: f32) -> f32 {
    if number <= 0.0 {
        return 0.0;
    }
    let mut i = number.to_bits();
    i = 0x5F37_59DF - (i >> 1);
    let y = f32::from_bits(i);
    // 2 iterations of refinement
    let y = y * (1.5 - (0.5 * number * y * y));
    y * (1.5 - (0.5 * number * y * y))
}

#[inline(always)]
pub fn fast_exp(x: f32) -> f32 {
    // 6th order polynomial approximation for e^x in [-10, 10]
    if x < -10.0 {
        return 0.0;
    }
    if x > 10.0 {
        return 22026.0;
    }
    let mut term = 1.0;
    let mut sum = 1.0;
    for i in 1..=6 {
        term *= x / (i as f32);
        sum += term;
    }
    sum
}

#[inline(always)]
pub fn fast_silu(x: f32) -> f32 {
    let sig = 1.0 / (1.0 + fast_exp(-x));
    x * sig
}

#[inline(always)]
fn f16_to_f32(h: u16) -> f32 {
    let sign = ((h >> 15) & 1) as u32;
    let exp = ((h >> 10) & 0x1F) as u32;
    let mant = (h & 0x3FF) as u32;
    if exp == 0 {
        if mant == 0 {
            f32::from_bits(sign << 31)
        } else {
            let mut m = mant;
            let mut e = 0;
            while (m & 0x400) == 0 {
                m <<= 1;
                e += 1;
            }
            f32::from_bits((sign << 31) | ((127 - 15 - e) << 23) | ((m & 0x3FF) << 13))
        }
    } else if exp == 31 {
        f32::from_bits((sign << 31) | (0xFF << 23) | (mant << 13))
    } else {
        f32::from_bits((sign << 31) | ((exp + (127 - 15)) << 23) | (mant << 13))
    }
}

#[inline(always)]
fn f32_to_f16(f: f32) -> u16 {
    let bits = f.to_bits();
    let sign = (bits >> 31) & 1;
    let exp = ((bits >> 23) & 0xFF) as i32;
    let mant = bits & 0x7FFFFF;
    if exp == 0xFF {
        let h_mant = if mant != 0 { 0x200 } else { 0 };
        return ((sign << 15) | (0x1F << 10) | h_mant) as u16;
    }
    let new_exp = exp - 127 + 15;
    if new_exp >= 31 {
        return ((sign << 15) | (0x1F << 10)) as u16;
    } else if new_exp <= 0 {
        if 10 + new_exp <= 0 {
            return (sign << 15) as u16;
        }
        let h_mant = ((mant | 0x800000) >> (14 - new_exp)) & 0x3FF;
        return ((sign << 15) | h_mant) as u16;
    }
    let h_mant = (mant >> 13) & 0x3FF;
    ((sign << 15) | ((new_exp as u32) << 10) | h_mant) as u16
}

/// Root Mean Square Layer Normalization (RMSNorm)
#[derive(Clone, Debug)]
pub struct RMSNorm {
    pub weight: Tensor,
    pub eps: f32,
}

impl RMSNorm {
    pub const fn new(weight: Tensor, eps: f32) -> Self {
        Self { weight, eps }
    }

    /// Perform in-place or out-of-place RMSNorm: y = (x / RMS(x)) * weight
    pub fn forward(&self, x: &Tensor, out: &mut Tensor) -> Result<(), &'static str> {
        let d = x.num_elements();
        if out.num_elements() != d || self.weight.num_elements() < d {
            return Err("RMSNorm Dimension Mismatch");
        }

        unsafe {
            let x_slice: &[u16] = x.as_slice();
            let w_slice: &[u16] = self.weight.as_slice();
            let out_slice: &mut [u16] = out.as_mut_slice();

            // 1. Calculate sum of squares
            let mut sum_sq = 0.0f32;
            for &val in x_slice.iter() {
                let f = f16_to_f32(val);
                sum_sq += f * f;
            }

            let mean_sq = sum_sq / (d as f32);
            let inv_rms = fast_rsqrt(mean_sq + self.eps);

            // 2. Normalize and scale
            for i in 0..d {
                let x_f = f16_to_f32(x_slice[i]);
                let w_f = f16_to_f32(w_slice[i]);
                let y = x_f * inv_rms * w_f;
                out_slice[i] = f32_to_f16(y);
            }
        }

        Ok(())
    }
}

/// Token Embedding Layer
#[derive(Clone, Debug)]
pub struct Embedding {
    pub weight: Tensor,
    pub vocab_size: usize,
    pub hidden_dim: usize,
}

impl Embedding {
    pub fn new(weight: Tensor, vocab_size: usize, hidden_dim: usize) -> Self {
        Self {
            weight,
            vocab_size,
            hidden_dim,
        }
    }

    /// Lookup embedding vector for token_id: out = weight[token_id]
    pub fn forward(&self, token_id: usize, out: &mut Tensor) -> Result<(), &'static str> {
        if token_id >= self.vocab_size {
            return Err("Token ID out of vocabulary bounds");
        }
        if out.num_elements() != self.hidden_dim {
            return Err("Embedding output tensor dimension mismatch");
        }

        unsafe {
            let w_ptr = self.weight.vaddr() as *const u16;
            let out_ptr = out.vaddr() as *mut u16;
            let src = w_ptr.add(token_id * self.hidden_dim);
            core::ptr::copy_nonoverlapping(src, out_ptr, self.hidden_dim);
        }

        Ok(())
    }
}

/// Rotary Position Embedding (RoPE)
#[derive(Clone, Debug)]
pub struct RotaryEmbedding {
    pub dim: usize,
    pub max_seq_len: usize,
}

impl RotaryEmbedding {
    pub const fn new(dim: usize, max_seq_len: usize) -> Self {
        Self { dim, max_seq_len }
    }

    /// Apply RoPE to query and key tensors at a given position
    pub fn apply(&self, q: &mut Tensor, k: &mut Tensor, pos: usize) -> Result<(), &'static str> {
        let half_dim = self.dim / 2;
        if q.num_elements() < self.dim || k.num_elements() < self.dim {
            return Err("Tensor dimension smaller than RoPE head dimension");
        }

        unsafe {
            let q_slice: &mut [u16] = q.as_mut_slice();
            let k_slice: &mut [u16] = k.as_mut_slice();

            for i in 0..half_dim {
                // theta = 10000 ^ (-2 * i / dim)
                let freq = (pos as f32) / (1.0 + (i as f32) * 2.0);
                // Simple trig approximation for rotation
                let cos_theta = 1.0 - (freq * freq * 0.5).min(1.0);
                let sin_theta = freq.clamp(-1.0, 1.0);

                // Rotate Query
                let q0 = f16_to_f32(q_slice[i]);
                let q1 = f16_to_f32(q_slice[i + half_dim]);
                q_slice[i] = f32_to_f16(q0 * cos_theta - q1 * sin_theta);
                q_slice[i + half_dim] = f32_to_f16(q0 * sin_theta + q1 * cos_theta);

                // Rotate Key
                let k0 = f16_to_f32(k_slice[i]);
                let k1 = f16_to_f32(k_slice[i + half_dim]);
                k_slice[i] = f32_to_f16(k0 * cos_theta - k1 * sin_theta);
                k_slice[i + half_dim] = f32_to_f16(k0 * sin_theta + k1 * cos_theta);
            }
        }

        Ok(())
    }
}

/// Tensor-Parallel Linear Projection Layer directly attached to AeroTensor NPU
#[derive(Clone, Debug)]
pub struct Linear {
    pub weight: Tensor,
    pub bias: Option<Tensor>,
    pub in_features: usize,
    pub out_features: usize,
    pub tp_degree: u8,
}

impl Linear {
    pub fn new(weight: Tensor, bias: Option<Tensor>, in_features: usize, out_features: usize) -> Self {
        Self {
            weight,
            bias,
            in_features,
            out_features,
            tp_degree: 4, // Default to TP=4
        }
    }

    pub fn with_tp(mut self, tp: u8) -> Self {
        self.tp_degree = tp;
        self
    }

    /// Forward projection: Y = X * W (dispatches to axcompute NPU SQ/CQ)
    pub fn forward(&self, x: &Tensor, out: &mut Tensor) -> Result<(), &'static str> {
        // Dispatches to axcompute MatMul
        let job = MatMul::new(x, &self.weight, out)
            .with_tile(0)
            .dispatch()?;

        let cqe = job.wait_complete();
        if cqe.status != 0 {
            return Err("Linear projection NPU execution failed");
        }

        // Add bias if present
        if let Some(ref bias) = self.bias {
            unsafe {
                let out_slice: &mut [u16] = out.as_mut_slice();
                let b_slice: &[u16] = bias.as_slice();
                for (i, val) in out_slice.iter_mut().enumerate() {
                    let sum = f16_to_f32(*val) + f16_to_f32(b_slice[i % b_slice.len()]);
                    *val = f32_to_f16(sum);
                }
            }
        }

        Ok(())
    }
}

/// SwiGLU Feed-Forward Network Block (MLP)
#[derive(Clone, Debug)]
pub struct SwiGLU {
    pub gate_proj: Linear,
    pub up_proj: Linear,
    pub down_proj: Linear,
}

impl SwiGLU {
    pub fn new(gate_proj: Linear, up_proj: Linear, down_proj: Linear) -> Self {
        Self {
            gate_proj,
            up_proj,
            down_proj,
        }
    }

    /// MLP Forward: out = down_proj(silu(gate_proj(x)) * up_proj(x))
    pub fn forward(
        &self,
        x: &Tensor,
        out: &mut Tensor,
        scratch_gate: &mut Tensor,
        scratch_up: &mut Tensor,
    ) -> Result<(), &'static str> {
        // 1. Gate projection
        self.gate_proj.forward(x, scratch_gate)?;
        // 2. Up projection
        self.up_proj.forward(x, scratch_up)?;

        // 3. Elementwise SwiGLU: gate = silu(gate) * up
        unsafe {
            let gate_slice: &mut [u16] = scratch_gate.as_mut_slice();
            let up_slice: &[u16] = scratch_up.as_slice();
            for i in 0..gate_slice.len() {
                let g = fast_silu(f16_to_f32(gate_slice[i]));
                let u = f16_to_f32(up_slice[i]);
                gate_slice[i] = f32_to_f16(g * u);
            }
        }

        // 4. Down projection
        self.down_proj.forward(scratch_gate, out)?;

        Ok(())
    }
}
