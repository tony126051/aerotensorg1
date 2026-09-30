#![no_std]

/// Cache Coherency Protocol Configuration
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CoherencyMode {
    /// Full AMBA 5 CHI Hardware Cache Coherency (Zero-Overhead Direct Access)
    Amba5ChiHardwareCoherent,
    /// Software-Managed Coherency (Requires CMO Clean & Invalidate)
    SoftwareManagedNonCoherent,
}

pub struct CoherencyManager {
    mode: CoherencyMode,
}

impl CoherencyManager {
    pub const fn new(mode: CoherencyMode) -> Self {
        Self { mode }
    }

    #[inline(always)]
    pub fn is_hardware_coherent(&self) -> bool {
        matches!(self.mode, CoherencyMode::Amba5ChiHardwareCoherent)
    }

    /// Flush data from CPU cache to physical memory (only if non-coherent)
    pub unsafe fn clean_range(&self, vaddr: usize, size: usize) {
        if self.is_hardware_coherent() {
            // Hardware automatically intercepts cache lines via Snoop Filter - zero software penalty
            return;
        }

        #[cfg(target_arch = "aarch64")]
        {
            let line_size = 64usize;
            let mut ptr = vaddr & !(line_size - 1);
            let end = vaddr + size;
            while ptr < end {
                core::arch::asm!("dc cvac, {0}", in(reg) ptr, options(nomem, nostack));
                ptr += line_size;
            }
            core::arch::asm!("dsb ish", options(nomem, nostack));
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
            let _ = (vaddr, size);
        }
    }

    /// Invalidate CPU cache lines so new DMA/NPU written data is read from memory
    pub unsafe fn invalidate_range(&self, vaddr: usize, size: usize) {
        if self.is_hardware_coherent() {
            // Hardware Snoop Filter ensures CPU always reads up-to-date NPU writes
            return;
        }

        #[cfg(target_arch = "aarch64")]
        {
            let line_size = 64usize;
            let mut ptr = vaddr & !(line_size - 1);
            let end = vaddr + size;
            while ptr < end {
                core::arch::asm!("dc ivac, {0}", in(reg) ptr, options(nomem, nostack));
                ptr += line_size;
            }
            core::arch::asm!("dsb ish", options(nomem, nostack));
            core::arch::asm!("isb", options(nomem, nostack));
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
            let _ = (vaddr, size);
        }
    }
}

/// Global AMBA 5 CHI Coherency Manager (Defaults to Hardware Coherent Domain)
pub static COHERENCY_MGR: CoherencyManager =
    CoherencyManager::new(CoherencyMode::Amba5ChiHardwareCoherent);
