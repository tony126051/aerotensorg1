#![no_std]

use core::sync::atomic::{compiler_fence, Ordering};

#[repr(C, align(64))]
pub struct StreamTableEntry {
    pub dword: [u32; 8],
}

impl StreamTableEntry {
    /// 啟用 Stage-1 SVA 模式，令 NPU SMMU 直接讀取 CPU 的 TTBR0_EL1
    pub fn enable_stage1_sva(&mut self, context_descriptor_pa: u64) {
        // Word 0: V=1 (Valid), Config=0b101 (Stage 1 only)
        self.dword[0] = 1 | (0b101 << 1);

        // Word 1: Context Descriptor 低 32 位元 (對齊 64 位元組)
        self.dword[1] = (context_descriptor_pa & 0xFFFF_FFC0) as u32;
        self.dword[2] = (context_descriptor_pa >> 32) as u32;

        compiler_fence(Ordering::SeqCst);
    }
}

#[repr(C, align(64))]
pub struct ContextDescriptor {
    pub dword: [u32; 16],
}

impl ContextDescriptor {
    /// 綁定 CPU TTBR0 與 TCR 屬性
    pub fn sync_with_cpu(&mut self, ttbr0_pa: u64, tcr: u64, mair: u64) {
        self.dword[0] = (tcr & 0xFFFF_FFFF) as u32 | (1 << 31); // V=1
        self.dword[1] = (ttbr0_pa & 0xFFFF_FFF0) as u32;
        self.dword[2] = (ttbr0_pa >> 32) as u32;
        self.dword[3] = (mair & 0xFFFF_FFFF) as u32;
        self.dword[4] = (mair >> 32) as u32;
        compiler_fence(Ordering::SeqCst);
    }
}
