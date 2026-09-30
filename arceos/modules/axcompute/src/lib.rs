#![no_std]

pub mod device;
pub mod queue;
pub mod tensor;
pub mod ops;
pub mod async_op;
pub mod pipeline;
pub mod driver;
pub mod irq;
pub mod coherency;

pub use device::{AeroTensorCQE, AeroTensorDevice, AeroTensorSQE};
pub use queue::CommandQueue;
pub use tensor::{DataType, Tensor};
pub use ops::{JobHandle, MatMul};
pub use async_op::ComputeFuture;
pub use pipeline::{OverlappedPipeline, PingPongBuffer};
pub use driver::{AeroTensorBoardDriver, TensorComputeDriver};
pub use irq::{GicSpiDispatcher, GIC_DISPATCHER, GIC_SPI_BASE, GIC_SPI_MAX};
pub use coherency::{CoherencyManager, CoherencyMode, COHERENCY_MGR};

use kspin::SpinNoIrq;

/// Global AeroTensor Command Queue and Execution Engine
pub static GLOBAL_ENGINE: SpinNoIrq<CommandQueue> = SpinNoIrq::new(CommandQueue::new());

/// Initialize AeroTensor AI Compute Subsystem
pub fn init() {
    // Reset or probe device
    log::info!("AeroTensor AI Compute Subsystem initialized.");
}
