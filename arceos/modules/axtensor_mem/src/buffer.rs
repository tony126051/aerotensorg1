#![no_std]

use core::slice;

/// Hardware Memory Hierarchy Tiering
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MemoryTier {
    /// L1: On-Chip Fast SRAM (Scratchpad for current kernel activations)
    L1Scratchpad,
    /// L2: Dedicated High-Bandwidth Local HBM Bank
    L2LocalHbm,
    /// L3: System-Wide Coherent Unified Memory Pool (96GB HBM3e @ 0x4000_0000)
    L3HostUma,
}

/// Continuous Physical Memory Descriptor for Tensors
#[derive(Debug)]
pub struct TensorBuffer {
    pub paddr: u64,
    pub vaddr: usize,
    pub size_bytes: usize,
    pub align_bytes: usize,
    pub tier: MemoryTier,
}

impl TensorBuffer {
    pub const fn new(
        paddr: u64,
        vaddr: usize,
        size_bytes: usize,
        align_bytes: usize,
        tier: MemoryTier,
    ) -> Self {
        Self {
            paddr,
            vaddr,
            size_bytes,
            align_bytes,
            tier,
        }
    }

    #[inline(always)]
    pub fn paddr(&self) -> u64 {
        self.paddr
    }

    #[inline(always)]
    pub fn vaddr(&self) -> usize {
        self.vaddr
    }

    #[inline(always)]
    pub fn size(&self) -> usize {
        self.size_bytes
    }

    #[inline(always)]
    pub fn tier(&self) -> MemoryTier {
        self.tier
    }

    /// Read-only slice view
    pub unsafe fn as_slice<T>(&self) -> &[T] {
        let count = self.size_bytes / core::mem::size_of::<T>();
        slice::from_raw_parts(self.vaddr as *const T, count)
    }

    /// Mutable slice view
    pub unsafe fn as_mut_slice<T>(&mut self) -> &mut [T] {
        let count = self.size_bytes / core::mem::size_of::<T>();
        slice::from_raw_parts_mut(self.vaddr as *mut T, count)
    }

    /// Zero out buffer memory
    pub unsafe fn fill_zero(&mut self) {
        let ptr = self.vaddr as *mut u8;
        core::ptr::write_bytes(ptr, 0, self.size_bytes);
    }
}
