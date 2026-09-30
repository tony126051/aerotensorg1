#![no_std]

use core::sync::atomic::{AtomicU32, Ordering};

pub const GIC_SPI_BASE: usize = 64;
pub const GIC_SPI_MAX: usize = 79;
pub const TOTAL_TILES: usize = 16;

/// GICv3 SPI Interrupt Controller & Multi-Tile Dispatcher
pub struct GicSpiDispatcher {
    irq_received_mask: AtomicU32,
    target_tiles: u8,
}

impl GicSpiDispatcher {
    pub const fn new(target_tiles: u8) -> Self {
        Self {
            irq_received_mask: AtomicU32::new(0),
            target_tiles,
        }
    }

    /// Dispatch incoming GICv3 SPI interrupt (SPI 64 ~ 79)
    pub fn dispatch_irq(&self, irq_num: usize) -> bool {
        if irq_num < GIC_SPI_BASE || irq_num > GIC_SPI_MAX {
            return false;
        }

        let tile_id = (irq_num - GIC_SPI_BASE) as u8;
        let bit = 1u32 << tile_id;
        let prev = self.irq_received_mask.fetch_or(bit, Ordering::AcqRel);
        let current = prev | bit;

        let expected_mask = if self.target_tiles >= 32 {
            0xFFFF_FFFF
        } else {
            (1u32 << self.target_tiles) - 1
        };

        // If all tiles have arrived at the barrier
        if (current & expected_mask) == expected_mask {
            // Trigger multi-core wake-up
            #[cfg(target_arch = "aarch64")]
            unsafe {
                core::arch::asm!("sev", options(nomem, nostack));
            }
            self.irq_received_mask.store(0, Ordering::Release);
        }

        true
    }

    /// Reset IRQ reception mask
    pub fn reset(&self) {
        self.irq_received_mask.store(0, Ordering::Release);
    }

    /// Get current bitmask of tiles that have signaled completion
    pub fn current_mask(&self) -> u32 {
        self.irq_received_mask.load(Ordering::Acquire)
    }
}

pub static GIC_DISPATCHER: GicSpiDispatcher = GicSpiDispatcher::new(16);
