#![no_std]

use crate::device::AeroTensorCQE;
use crate::ops::{JobHandle, MatMul};
use crate::tensor::Tensor;

/// Ping-Pong Double Buffering structure for Zero-Jitter CPU/NPU Overlap
pub struct PingPongBuffer {
    pub buf_0: Tensor,
    pub buf_1: Tensor,
    active_idx: usize,
}

impl PingPongBuffer {
    pub const fn new(buf_0: Tensor, buf_1: Tensor) -> Self {
        Self {
            buf_0,
            buf_1,
            active_idx: 0,
        }
    }

    /// Front buffer designated for CPU Preprocessing / Sensor Ingestion
    #[inline(always)]
    pub fn cpu_front(&self) -> &Tensor {
        if self.active_idx == 0 {
            &self.buf_0
        } else {
            &self.buf_1
        }
    }

    #[inline(always)]
    pub fn cpu_front_mut(&mut self) -> &mut Tensor {
        if self.active_idx == 0 {
            &mut self.buf_0
        } else {
            &mut self.buf_1
        }
    }

    /// Back buffer designated for NPU Matrix Compute / DMA
    #[inline(always)]
    pub fn npu_back(&self) -> &Tensor {
        if self.active_idx == 0 {
            &self.buf_1
        } else {
            &self.buf_0
        }
    }

    #[inline(always)]
    pub fn npu_back_mut(&mut self) -> &mut Tensor {
        if self.active_idx == 0 {
            &mut self.buf_1
        } else {
            &mut self.buf_0
        }
    }

    /// Swap Front and Back buffers
    #[inline(always)]
    pub fn swap(&mut self) {
        self.active_idx ^= 1;
    }
}

/// Pipeline Runner that executes Overlapped CPU Preprocessing and NPU Computing
pub struct OverlappedPipeline;

impl OverlappedPipeline {
    /// Execute one pipeline step with CPU and NPU completely overlapped in time
    /// 
    /// 1. Asynchronously dispatch NPU GEMM on back buffer
    /// 2. In parallel, CPU preprocesses the front buffer for the next step
    /// 3. Synchronize completion and swap buffers (Zero Memory Copy)
    pub fn step<F>(
        ping_pong: &mut PingPongBuffer,
        weights: &Tensor,
        output: &mut Tensor,
        cpu_preprocess: F,
    ) -> Result<AeroTensorCQE, &'static str>
    where
        F: FnOnce(&mut Tensor),
    {
        // 1. Dispatch NPU computation on the back buffer (in-flight hardware task)
        let job = MatMul::new(ping_pong.npu_back(), weights, output)
            .with_relu(true)
            .dispatch()?;

        // 2. Concurrently execute CPU preprocessing on the front buffer
        cpu_preprocess(ping_pong.cpu_front_mut());

        // 3. Wait for NPU computation to finish
        let cqe = job.wait_complete();

        // 4. Swap Ping-Pong buffers for next step
        ping_pong.swap();

        Ok(cqe)
    }
}
