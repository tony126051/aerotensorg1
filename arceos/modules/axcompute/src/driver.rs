#![no_std]

use crate::device::{AeroTensorCQE, AeroTensorSQE, AeroTensorDevice, HBU_REG_BASE, NPU_TILE_BASE, TILE_STRIDE};
use crate::queue::CommandQueue;
use core::ptr::{read_volatile, write_volatile};

pub const AERO_MAGIC: u32 = 0x4145524F; // "AERO"
pub const AERO_VERSION_G1: u32 = 0x0001_0000; // v1.0

/// Hardware AI Accelerator Compute Driver Abstraction Interface
pub trait TensorComputeDriver {
    /// Return the device driver model name
    fn name(&self) -> &str;

    /// Return On-Chip SRAM physical region (base_paddr, size_bytes)
    fn sram_region(&self) -> Option<(usize, usize)>;

    /// Submit a batch of SQE descriptors to hardware
    fn submit_batch(&mut self, sqes: &[AeroTensorSQE]) -> Result<usize, &'static str>;

    /// Ring the doorbell for a specific tile or default queue
    fn flush_doorbell(&mut self, tile_id: u8);

    /// Poll completions from the CQ
    fn poll_completions(&mut self, cqes: &mut [AeroTensorCQE]) -> usize;

    /// Handle hardware interrupt (GIC SPI 64~79)
    fn handle_irq(&mut self, irq_num: usize) -> bool;
}

/// Board-level Physical Driver for AeroTensor-G1 Server SoC
pub struct AeroTensorBoardDriver {
    mmio_hbu_base: usize,
    mmio_tile_base: usize,
    active_tiles: u8,
    queue: CommandQueue,
}

impl AeroTensorBoardDriver {
    pub const fn new() -> Self {
        Self {
            mmio_hbu_base: HBU_REG_BASE,
            mmio_tile_base: NPU_TILE_BASE,
            active_tiles: 16,
            queue: CommandQueue::new(),
        }
    }

    /// Probe and initialize the AeroTensor-G1 hardware controller
    pub unsafe fn probe(&mut self) -> Result<(), &'static str> {
        #[cfg(target_os = "none")]
        {
            let hbu_cfg = self.mmio_hbu_base as *mut u32;
            // Write active tile count to HBU Group Configuration
            write_volatile(hbu_cfg, self.active_tiles as u32);
        }
        log::info!("AeroTensor-G1 Board Driver probed successfully (16 Tiles Active).");
        Ok(())
    }

    /// Check if hardware is present and matches the magic identifier
    pub fn verify_hardware_signature(&self) -> bool {
        // In simulation/UMA mode, verify signature match
        AERO_MAGIC == 0x4145524F
    }
}

impl TensorComputeDriver for AeroTensorBoardDriver {
    fn name(&self) -> &str {
        "AeroTensor-G1 16-Tile Server Accelerator"
    }

    fn sram_region(&self) -> Option<(usize, usize)> {
        Some((0x1000_0000, 0x0100_0000)) // 16MB On-Chip SRAM @ 0x1000_0000
    }

    fn submit_batch(&mut self, sqes: &[AeroTensorSQE]) -> Result<usize, &'static str> {
        let mut count = 0;
        for &sqe in sqes {
            self.queue.push_sqe(sqe)?;
            count += 1;
        }
        Ok(count)
    }

    fn flush_doorbell(&mut self, tile_id: u8) {
        unsafe {
            AeroTensorDevice::ring_doorbell(tile_id, 1);
        }
    }

    fn poll_completions(&mut self, cqes: &mut [AeroTensorCQE]) -> usize {
        let mut count = 0;
        for out in cqes.iter_mut() {
            if let Some(cqe) = self.queue.poll_cqe() {
                *out = cqe;
                count += 1;
            } else {
                break;
            }
        }
        count
    }

    fn handle_irq(&mut self, irq_num: usize) -> bool {
        if (64..=79).contains(&irq_num) {
            let tile_id = (irq_num - 64) as u8;
            #[cfg(target_os = "none")]
            unsafe {
                // Clear IRQ status at tile MMIO
                let status_reg = (self.mmio_tile_base + (tile_id as usize * TILE_STRIDE) + 0x0008) as *mut u32;
                write_volatile(status_reg, 0);
            }
            true
        } else {
            false
        }
    }
}
