#![no_std]

use core::future::Future;
use core::pin::Pin;
use core::task::{Context, Poll};
use crate::device::AeroTensorCQE;
use crate::GLOBAL_ENGINE;

/// Asynchronous Future representing an in-flight computation kernel
pub struct ComputeFuture {
    pub token: u64,
}

impl ComputeFuture {
    pub const fn new(token: u64) -> Self {
        Self { token }
    }
}

impl Future for ComputeFuture {
    type Output = AeroTensorCQE;

    fn poll(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<Self::Output> {
        let engine = GLOBAL_ENGINE.lock();
        if let Some(cqe) = engine.is_token_done(self.token) {
            Poll::Ready(cqe)
        } else {
            Poll::Pending
        }
    }
}
