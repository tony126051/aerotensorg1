#![no_std]

pub mod buffer;
pub mod allocator;

pub use buffer::{MemoryTier, TensorBuffer};
pub use allocator::{
    ContinuousHugePageAllocator, HUGEPAGE_ALLOCATOR, HUGEPAGE_1GB, HUGEPAGE_2MB, PAGE_SIZE_4K,
    SRAM_BASE_PADDR, UMA_BASE_PADDR,
};

/// Allocate a 1GB-aligned contiguous huge page
pub fn alloc_1gb_block() -> Result<TensorBuffer, &'static str> {
    HUGEPAGE_ALLOCATOR.lock().alloc_1gb_block()
}

/// Allocate a 2MB-aligned contiguous huge page
pub fn alloc_2mb_block() -> Result<TensorBuffer, &'static str> {
    HUGEPAGE_ALLOCATOR.lock().alloc_2mb_block()
}

/// Allocate fast On-Chip SRAM (L1 Scratchpad)
pub fn alloc_sram(size: usize, align: usize) -> Result<TensorBuffer, &'static str> {
    HUGEPAGE_ALLOCATOR.lock().alloc_sram(size, align)
}

/// Generic tensor buffer allocation
pub fn alloc_tensor_buffer(
    size: usize,
    align: usize,
    tier: MemoryTier,
) -> Result<TensorBuffer, &'static str> {
    HUGEPAGE_ALLOCATOR.lock().alloc_tensor_buffer(size, align, tier)
}
