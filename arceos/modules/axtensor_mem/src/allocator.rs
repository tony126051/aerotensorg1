#![no_std]

use crate::buffer::{MemoryTier, TensorBuffer};
use kspin::SpinNoIrq;

pub const PAGE_SIZE_4K: usize = 0x1000;
pub const HUGEPAGE_2MB: usize = 0x0020_0000;
pub const HUGEPAGE_1GB: usize = 0x4000_0000;

pub const SRAM_BASE_PADDR: u64 = 0x1000_0000;
pub const SRAM_SIZE: usize = 0x0100_0000; // 16MB On-Chip SRAM

pub const UMA_BASE_PADDR: u64 = 0x4000_0000;
pub const UMA_TOTAL_SIZE: u64 = 0x18_0000_0000; // 96GB HBM3e

/// Bump & Block Continuous Allocator for Physical HugePages
pub struct ContinuousHugePageAllocator {
    sram_offset: usize,
    uma_2mb_offset: usize,
    uma_1gb_offset: usize,
}

impl ContinuousHugePageAllocator {
    pub const fn new() -> Self {
        Self {
            sram_offset: 0,
            // Reserve first 16GB for 1GB blocks (Weights), remainder for 2MB blocks (Activations & KV-Cache)
            uma_1gb_offset: 0,
            uma_2mb_offset: 0x04_0000_0000, // Starts at +16GB offset
        }
    }

    /// Allocate a 1GB-aligned huge page block (ideal for static Model Weights)
    pub fn alloc_1gb_block(&mut self) -> Result<TensorBuffer, &'static str> {
        let align = HUGEPAGE_1GB;
        let aligned_offset = (self.uma_1gb_offset + align - 1) & !(align - 1);
        if aligned_offset + HUGEPAGE_1GB > 0x04_0000_0000 {
            return Err("Out of 1GB Static Weight Arena Memory");
        }

        let paddr = UMA_BASE_PADDR + (aligned_offset as u64);
        let vaddr = paddr as usize; // ArceOS identity/direct mapping
        self.uma_1gb_offset = aligned_offset + HUGEPAGE_1GB;

        Ok(TensorBuffer::new(
            paddr,
            vaddr,
            HUGEPAGE_1GB,
            align,
            MemoryTier::L3HostUma,
        ))
    }

    /// Allocate a 2MB-aligned huge page block (ideal for KV-Cache and Dynamic Activation Tensors)
    pub fn alloc_2mb_block(&mut self) -> Result<TensorBuffer, &'static str> {
        let align = HUGEPAGE_2MB;
        let aligned_offset = (self.uma_2mb_offset + align - 1) & !(align - 1);
        if aligned_offset + HUGEPAGE_2MB > (UMA_TOTAL_SIZE as usize) {
            return Err("Out of 2MB Dynamic Activation Arena Memory");
        }

        let paddr = UMA_BASE_PADDR + (aligned_offset as u64);
        let vaddr = paddr as usize;
        self.uma_2mb_offset = aligned_offset + HUGEPAGE_2MB;

        Ok(TensorBuffer::new(
            paddr,
            vaddr,
            HUGEPAGE_2MB,
            align,
            MemoryTier::L3HostUma,
        ))
    }

    /// Allocate On-Chip SRAM (L1 Scratchpad) buffer
    pub fn alloc_sram(&mut self, size: usize, align: usize) -> Result<TensorBuffer, &'static str> {
        let align_req = if align == 0 { 64 } else { align };
        let aligned_offset = (self.sram_offset + align_req - 1) & !(align_req - 1);
        if aligned_offset + size > SRAM_SIZE {
            return Err("Out of L1 On-Chip SRAM Scratchpad Memory");
        }

        let paddr = SRAM_BASE_PADDR + (aligned_offset as u64);
        let vaddr = paddr as usize;
        self.sram_offset = aligned_offset + size;

        Ok(TensorBuffer::new(
            paddr,
            vaddr,
            size,
            align_req,
            MemoryTier::L1Scratchpad,
        ))
    }

    /// Generic Tensor Buffer Allocator with Tier selection
    pub fn alloc_tensor_buffer(
        &mut self,
        size: usize,
        align: usize,
        tier: MemoryTier,
    ) -> Result<TensorBuffer, &'static str> {
        match tier {
            MemoryTier::L1Scratchpad => self.alloc_sram(size, align),
            MemoryTier::L2LocalHbm | MemoryTier::L3HostUma => {
                if size >= HUGEPAGE_1GB {
                    self.alloc_1gb_block()
                } else if size >= HUGEPAGE_2MB || align >= HUGEPAGE_2MB {
                    self.alloc_2mb_block()
                } else {
                    // Cacheline aligned allocation inside 2MB arena
                    let align_req = if align == 0 { 64 } else { align };
                    let aligned_offset = (self.uma_2mb_offset + align_req - 1) & !(align_req - 1);
                    let paddr = UMA_BASE_PADDR + (aligned_offset as u64);
                    let vaddr = paddr as usize;
                    self.uma_2mb_offset = aligned_offset + size;
                    Ok(TensorBuffer::new(paddr, vaddr, size, align_req, tier))
                }
            }
        }
    }
}

/// Global Thread-Safe HugePage Allocator Instance
pub static HUGEPAGE_ALLOCATOR: SpinNoIrq<ContinuousHugePageAllocator> =
    SpinNoIrq::new(ContinuousHugePageAllocator::new());
