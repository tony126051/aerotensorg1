#![no_std]

use core::sync::atomic::{compiler_fence, Ordering};

pub const L1_BLOCK_SIZE: usize = 0x4000_0000; // 1GB
pub const L2_BLOCK_SIZE: usize = 0x0020_0000; // 2MB
pub const L1_ALIGN_MASK: usize = L1_BLOCK_SIZE - 1;
pub const L2_ALIGN_MASK: usize = L2_BLOCK_SIZE - 1;

pub const PTE_VALID: u64         = 1 << 0;
pub const PTE_IS_TABLE: u64      = 1 << 1;
pub const PTE_ATTR_NORMAL_WB: u64= 2 << 2; // Attr 2: Normal Inner/Outer Write-Back
pub const PTE_INNER_SH: u64      = 3 << 8; // Inner Shareable (硬體一致性網域)
pub const PTE_AF: u64            = 1 << 10; // Access Flag
pub const PTE_UXN: u64           = 1 << 54; // User Execute Never

#[derive(Debug)]
pub enum PagingError {
    MisalignedPhysicalAddress,
    MisalignedVirtualAddress,
    PageAllocationFailed,
}

pub struct UmaPageTable {
    ttbr0_pa: usize,
}

impl UmaPageTable {
    pub const unsafe fn new(ttbr0_pa: usize) -> Self {
        Self { ttbr0_pa }
    }

    /// 映射 1GB 靜態大模型權重區間
    pub unsafe fn map_1g_block(&mut self, va: usize, pa: usize) -> Result<(), PagingError> {
        if (va & L1_ALIGN_MASK) != 0 {
            return Err(PagingError::MisalignedVirtualAddress);
        }
        if (pa & L1_ALIGN_MASK) != 0 {
            return Err(PagingError::MisalignedPhysicalAddress);
        }

        let l0_idx = (va >> 39) & 0x1FF;
        let l1_idx = (va >> 30) & 0x1FF;

        let root_l0 = self.ttbr0_pa as *mut u64;
        let l0_entry = root_l0.add(l0_idx);

        let l1_table_pa = if (*l0_entry & PTE_VALID) == 0 {
            let page = self.alloc_zeroed_page()?;
            *l0_entry = (page as u64) | PTE_VALID | PTE_IS_TABLE;
            page
        } else {
            (*l0_entry & 0x0000_FFFF_FFFF_F000) as usize
        };

        let l1_table = l1_table_pa as *mut u64;
        let l1_entry = l1_table.add(l1_idx);

        // 寫入 1GB Block 項目 (IS_TABLE 位元設為 0)
        let mut entry = (pa as u64) & 0x0000_FFFF_C000_0000;
        entry |= PTE_VALID | PTE_AF | PTE_INNER_SH | PTE_ATTR_NORMAL_WB | PTE_UXN;

        *l1_entry = entry;
        compiler_fence(Ordering::SeqCst);

        self.tlb_invalidate_is(va);
        Ok(())
    }

    /// 映射 2MB 動態 KV-Cache 區間
    pub unsafe fn map_2m_block(&mut self, va: usize, pa: usize) -> Result<(), PagingError> {
        if (va & L2_ALIGN_MASK) != 0 {
            return Err(PagingError::MisalignedVirtualAddress);
        }
        if (pa & L2_ALIGN_MASK) != 0 {
            return Err(PagingError::MisalignedPhysicalAddress);
        }

        let l0_idx = (va >> 39) & 0x1FF;
        let l1_idx = (va >> 30) & 0x1FF;
        let l2_idx = (va >> 21) & 0x1FF;

        let root_l0 = self.ttbr0_pa as *mut u64;
        let l1_table_pa = self.get_or_create_table(root_l0.add(l0_idx))?;
        let l2_table_pa = self.get_or_create_table((l1_table_pa as *mut u64).add(l1_idx))?;

        let l2_entry = (l2_table_pa as *mut u64).add(l2_idx);
        let mut entry = (pa as u64) & 0x0000_FFFF_FFE0_0000;
        entry |= PTE_VALID | PTE_AF | PTE_INNER_SH | PTE_ATTR_NORMAL_WB | PTE_UXN;

        *l2_entry = entry;
        compiler_fence(Ordering::SeqCst);

        self.tlb_invalidate_is(va);
        Ok(())
    }

    unsafe fn get_or_create_table(&self, entry: *mut u64) -> Result<usize, PagingError> {
        if (*entry & PTE_VALID) == 0 {
            let page = self.alloc_zeroed_page()?;
            *entry = (page as u64) | PTE_VALID | PTE_IS_TABLE;
            Ok(page)
        } else {
            Ok((*entry & 0x0000_FFFF_FFFF_F000) as usize)
        }
    }

    fn alloc_zeroed_page(&self) -> Result<usize, PagingError> {
        extern "C" {
            fn early_page_alloc() -> usize;
        }
        // If not linked with external early_page_alloc, fallback to 0
        #[allow(unreachable_code)]
        let page = unsafe {
            #[cfg(target_os = "none")]
            { early_page_alloc() }
            #[cfg(not(target_os = "none"))]
            { 0x5000_0000 }
        };
        if page == 0 {
            Err(PagingError::PageAllocationFailed)
        } else {
            Ok(page)
        }
    }

    #[inline(always)]
    unsafe fn tlb_invalidate_is(&self, va: usize) {
        #[cfg(target_arch = "aarch64")]
        core::arch::asm!(
            "dsb ishst",
            "tlbi vaae1is, {0}",
            "dsb ish",
            "isb",
            in(reg) va >> 12,
            options(nostack)
        );
        #[cfg(not(target_arch = "aarch64"))]
        {
            let _ = va;
        }
    }
}
