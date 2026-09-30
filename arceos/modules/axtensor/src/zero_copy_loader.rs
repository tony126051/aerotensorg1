#![no_std]

use axmm::vmsa_uma::{UmaPageTable, L1_BLOCK_SIZE};

#[repr(C, packed)]
pub struct ModelBlobHeader {
    pub magic: [u8; 4],          // "AERO"
    pub layers: u32,
    pub hidden_dimension: u32,
    pub weight_bytes: u64,
}

pub struct ModelContext {
    pub base_vaddr: usize,
    pub base_paddr: usize,
    pub total_layers: usize,
    pub hidden_dim: usize,
}

impl ModelContext {
    /// 透過 1GB Block 直接映射已預先放入 UMA 區間的模型權重
    pub unsafe fn map_uma_model(
        pt: &mut UmaPageTable,
        uma_pool_pa: usize,
        total_size: usize,
    ) -> Result<Self, &'static str> {
        let vaddr_start = 0xFFFF_0001_0000_0000usize;
        let blocks = (total_size + L1_BLOCK_SIZE - 1) / L1_BLOCK_SIZE;

        for i in 0..blocks {
            let va = vaddr_start + (i * L1_BLOCK_SIZE);
            let pa = uma_pool_pa + (i * L1_BLOCK_SIZE);
            pt.map_1g_block(va, pa).map_err(|_| "Failed 1GB Block Mapping")?;
        }

        let header = &*(vaddr_start as *const ModelBlobHeader);
        if &header.magic != b"AERO" {
            return Err("Model Magic Invalid");
        }

        Ok(Self {
            base_vaddr: vaddr_start + core::mem::size_of::<ModelBlobHeader>(),
            base_paddr: uma_pool_pa + core::mem::size_of::<ModelBlobHeader>(),
            total_layers: header.layers as usize,
            hidden_dim: header.hidden_dimension as usize,
        })
    }
}
