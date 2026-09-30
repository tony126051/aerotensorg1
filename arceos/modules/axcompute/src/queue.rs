#![no_std]

use core::sync::atomic::{AtomicU32, Ordering};
use crate::device::{
    AeroTensorCQE, AeroTensorSQE, AeroTensorDevice, CQE_STATUS_PENDING, CQE_STATUS_SUCCESS,
    FLAG_RELU, OP_MATMUL, OP_NOP,
};

pub const DEFAULT_QUEUE_DEPTH: usize = 64;

/// High-Performance Ring Buffer for AeroTensor Submissions and Completions
pub struct CommandQueue {
    sq: [AeroTensorSQE; DEFAULT_QUEUE_DEPTH],
    cq: [AeroTensorCQE; DEFAULT_QUEUE_DEPTH],
    sq_head: AtomicU32,
    sq_tail: AtomicU32,
    cq_head: AtomicU32,
    cq_tail: AtomicU32,
    depth: u32,
}

impl CommandQueue {
    pub const fn new() -> Self {
        Self {
            sq: [AeroTensorSQE::empty(); DEFAULT_QUEUE_DEPTH],
            cq: [AeroTensorCQE::pending(0); DEFAULT_QUEUE_DEPTH],
            sq_head: AtomicU32::new(0),
            sq_tail: AtomicU32::new(0),
            cq_head: AtomicU32::new(0),
            cq_tail: AtomicU32::new(0),
            depth: DEFAULT_QUEUE_DEPTH as u32,
        }
    }

    /// Submit a new work request (SQE) into the submission ring
    pub fn push_sqe(&mut self, sqe: AeroTensorSQE) -> Result<u32, &'static str> {
        let current_tail = self.sq_tail.load(Ordering::Acquire);
        let current_head = self.sq_head.load(Ordering::Acquire);

        let next_tail = (current_tail + 1) % self.depth;
        if next_tail == current_head {
            return Err("Submission Queue (SQ) is full");
        }

        self.sq[current_tail as usize] = sqe;
        self.sq_tail.store(next_tail, Ordering::Release);

        // Ring the hardware doorbell or schedule execution
        unsafe {
            AeroTensorDevice::ring_doorbell(sqe.tile_id, next_tail);
        }

        // For software simulation or bare-metal emulator, execute in-flight
        self.process_in_flight(current_tail as usize);

        Ok(current_tail)
    }

    /// Process in-flight SQE for emulation and produce corresponding CQE
    fn process_in_flight(&mut self, sq_idx: usize) {
        let sqe = self.sq[sq_idx];
        self.sq_head.store((sq_idx as u32 + 1) % self.depth, Ordering::Release);

        let cycles = match sqe.opcode {
            OP_MATMUL => self.execute_gemm(&sqe),
            OP_NOP => 1,
            _ => 10,
        };

        // Enqueue into CQ
        let cq_tail = self.cq_tail.load(Ordering::Acquire);
        self.cq[cq_tail as usize] = AeroTensorCQE::success(sqe.user_data, cycles);
        self.cq_tail.store((cq_tail + 1) % self.depth, Ordering::Release);
    }

    /// Software FP16/FP32 GEMM calculation for UMA memory verification
    fn execute_gemm(&self, sqe: &AeroTensorSQE) -> u32 {
        let m = sqe.m as usize;
        let k = sqe.k as usize;
        let n = sqe.n as usize;

        if sqe.src_a_pa != 0 && sqe.src_b_pa != 0 && sqe.dst_c_pa != 0 {
            unsafe {
                let a_ptr = sqe.src_a_pa as *const u16;
                let b_ptr = sqe.src_b_pa as *const u16;
                let c_ptr = sqe.dst_c_pa as *mut u16;

                let relu = (sqe.flags & FLAG_RELU) != 0;

                // GEMM: C = A x B (simulated FP16 operations)
                for i in 0..m {
                    for j in 0..n {
                        let mut sum_f32: f32 = 0.0;
                        for p in 0..k {
                            let a_val = f16_to_f32(*a_ptr.add(i * sqe.stride_a as usize + p));
                            let b_val = f16_to_f32(*b_ptr.add(p * sqe.stride_b as usize + j));
                            sum_f32 += a_val * b_val;
                        }
                        if relu && sum_f32 < 0.0 {
                            sum_f32 = 0.0;
                        }
                        *c_ptr.add(i * sqe.stride_c as usize + j) = f32_to_f16(sum_f32);
                    }
                }
            }
        }

        // Return simulated hardware cycles: ~ 2 * M * N * K / (128 TFLOPS)
        let total_ops = (2 * m * n * k) as u32;
        core::cmp::max(16, total_ops / 64)
    }

    /// Poll for the next completed work item (CQE)
    pub fn poll_cqe(&mut self) -> Option<AeroTensorCQE> {
        let current_head = self.cq_head.load(Ordering::Acquire);
        let current_tail = self.cq_tail.load(Ordering::Acquire);

        if current_head == current_tail {
            return None; // Completion Queue is empty
        }

        let cqe = self.cq[current_head as usize];
        if cqe.status != CQE_STATUS_PENDING {
            // Mark slot as pending/consumed
            self.cq[current_head as usize].status = CQE_STATUS_PENDING;
            self.cq_head.store((current_head + 1) % self.depth, Ordering::Release);
            Some(cqe)
        } else {
            None
        }
    }

    /// Check if completion queue has finished the specific token
    pub fn is_token_done(&self, token: u64) -> Option<AeroTensorCQE> {
        let current_head = self.cq_head.load(Ordering::Acquire);
        let current_tail = self.cq_tail.load(Ordering::Acquire);

        let mut idx = current_head;
        while idx != current_tail {
            let cqe = self.cq[idx as usize];
            if cqe.user_data == token && cqe.status == CQE_STATUS_SUCCESS {
                return Some(cqe);
            }
            idx = (idx + 1) % self.depth;
        }
        None
    }
}

/// Convert IEEE 754 half-precision u16 bit pattern to f32
#[inline(always)]
fn f16_to_f32(h: u16) -> f32 {
    let sign = ((h >> 15) & 1) as u32;
    let exp = ((h >> 10) & 0x1F) as u32;
    let mant = (h & 0x3FF) as u32;

    if exp == 0 {
        if mant == 0 {
            f32::from_bits(sign << 31)
        } else {
            // Denormalized
            let mut m = mant;
            let mut e = 0;
            while (m & 0x400) == 0 {
                m <<= 1;
                e += 1;
            }
            let f_exp = (127 - 15 - e) as u32;
            let f_mant = (m & 0x3FF) << 13;
            f32::from_bits((sign << 31) | (f_exp << 23) | f_mant)
        }
    } else if exp == 31 {
        // Inf / NaN
        f32::from_bits((sign << 31) | (0xFF << 23) | (mant << 13))
    } else {
        // Normalized
        let f_exp = (exp + (127 - 15)) << 23;
        let f_mant = mant << 13;
        f32::from_bits((sign << 31) | f_exp | f_mant)
    }
}

/// Convert f32 to IEEE 754 half-precision u16 bit pattern
#[inline(always)]
fn f32_to_f16(f: f32) -> u16 {
    let bits = f.to_bits();
    let sign = (bits >> 31) & 1;
    let exp = ((bits >> 23) & 0xFF) as i32;
    let mant = bits & 0x7FFFFF;

    if exp == 0xFF {
        // NaN / Inf
        let h_mant = if mant != 0 { 0x200 } else { 0 };
        return ((sign << 15) | (0x1F << 10) | h_mant) as u16;
    }

    let new_exp = exp - 127 + 15;
    if new_exp >= 31 {
        // Overflow to Inf
        return ((sign << 15) | (0x1F << 10)) as u16;
    } else if new_exp <= 0 {
        // Underflow to zero or denormal
        if 10 + new_exp <= 0 {
            return (sign << 15) as u16;
        }
        let h_mant = ((mant | 0x800000) >> (14 - new_exp)) & 0x3FF;
        return ((sign << 15) | h_mant) as u16;
    }

    let h_mant = (mant >> 13) & 0x3FF;
    ((sign << 15) | ((new_exp as u32) << 10) | h_mant) as u16
}
