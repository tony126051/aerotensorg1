#![no_std]

use core::sync::atomic::{AtomicU64, Ordering};
use crate::device::{AeroTensorCQE, AeroTensorSQE};
use crate::tensor::Tensor;
use crate::GLOBAL_ENGINE;

static TOKEN_COUNTER: AtomicU64 = AtomicU64::new(1);

/// JobHandle representing an in-flight asynchronous computation task
#[derive(Clone, Copy, Debug)]
pub struct JobHandle {
    pub token: u64,
}

impl JobHandle {
    /// Poll for job completion
    pub fn poll(&self) -> Option<AeroTensorCQE> {
        let mut engine = GLOBAL_ENGINE.lock();
        engine.is_token_done(self.token)
    }

    /// Block and wait until the computation completes (hardware interrupt / completion queue)
    pub fn wait_complete(&self) -> AeroTensorCQE {
        loop {
            if let Some(cqe) = self.poll() {
                return cqe;
            }
            #[cfg(target_arch = "aarch64")]
            unsafe {
                core::arch::asm!("wfe", options(nomem, nostack));
            }
            #[cfg(not(target_arch = "aarch64"))]
            core::hint::spin_loop();
        }
    }
}

/// Matrix Multiplication (GEMM) Operator Builder
pub struct MatMul<'a> {
    a: &'a Tensor,
    b: &'a Tensor,
    c: &'a mut Tensor,
    tile_id: u8,
    relu: bool,
}

impl<'a> MatMul<'a> {
    pub fn new(a: &'a Tensor, b: &'a Tensor, c: &'a mut Tensor) -> Self {
        Self {
            a,
            b,
            c,
            tile_id: 0,
            relu: false,
        }
    }

    pub fn with_relu(mut self, relu: bool) -> Self {
        self.relu = relu;
        self
    }

    pub fn with_tile(mut self, tile_id: u8) -> Self {
        self.tile_id = tile_id & 0xF;
        self
    }

    /// Dispatch GEMM operation directly into the hardware CommandQueue
    pub fn dispatch(self) -> Result<JobHandle, &'static str> {
        let m = self.a.dim(0);
        let k_a = self.a.dim(1);
        let k_b = self.b.dim(0);
        let n = self.b.dim(1);

        if k_a != k_b {
            return Err("MatMul Dimension Mismatch: K dimensions of A and B must match");
        }
        if self.c.dim(0) != m || self.c.dim(1) != n {
            return Err("MatMul Dimension Mismatch: Target C dimension must match (M, N)");
        }

        let token = TOKEN_COUNTER.fetch_add(1, Ordering::SeqCst);

        let sqe = AeroTensorSQE::new_matmul(
            self.tile_id,
            self.a.paddr(),
            self.b.paddr(),
            self.c.paddr(),
            m as u32,
            k_a as u32,
            n as u32,
            self.relu,
            token,
        );

        let mut engine = GLOBAL_ENGINE.lock();
        engine.push_sqe(sqe)?;

        Ok(JobHandle { token })
    }

    /// Dispatch GEMM operation and return an asynchronous Rust Future
    pub fn dispatch_async(self) -> Result<crate::async_op::ComputeFuture, &'static str> {
        let handle = self.dispatch()?;
        Ok(crate::async_op::ComputeFuture::new(handle.token))
    }
}
