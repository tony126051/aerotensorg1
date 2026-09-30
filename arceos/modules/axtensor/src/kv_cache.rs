#![no_std]

use axtensor_mem::HUGEPAGE_2MB;

pub const MAX_LAYERS: usize = 32;

/// Dynamic KV-Cache Slot Manager directly backed by 2MB Physical HugePages
#[derive(Clone, Debug)]
pub struct KVCache {
    num_layers: usize,
    max_seq_len: usize,
    num_heads: usize,
    head_dim: usize,
    layer_stride_bytes: usize,
    token_stride_bytes: usize,
    base_paddr: u64,
    base_vaddr: usize,
    current_pos: usize,
}

impl KVCache {
    pub fn new(
        num_layers: usize,
        max_seq_len: usize,
        num_heads: usize,
        head_dim: usize,
        base_paddr: u64,
        base_vaddr: usize,
    ) -> Result<Self, &'static str> {
        if num_layers > MAX_LAYERS {
            return Err("Number of layers exceeds MAX_LAYERS limit");
        }

        // Each token has (num_heads * head_dim) FP16 elements (2 bytes)
        let token_stride_bytes = num_heads * head_dim * 2;
        // Each layer stores both K and V: 2 * (max_seq_len * token_stride_bytes)
        let layer_stride_bytes = 2 * max_seq_len * token_stride_bytes;

        // Ensure alignment within 2MB block boundary
        let total_bytes = layer_stride_bytes * num_layers;
        if total_bytes == 0 {
            return Err("KV Cache allocation size must be non-zero");
        }

        Ok(Self {
            num_layers,
            max_seq_len,
            num_heads,
            head_dim,
            layer_stride_bytes,
            token_stride_bytes,
            base_paddr,
            base_vaddr,
            current_pos: 0,
        })
    }

    /// Calculate physical address for Key cache of (layer, pos)
    #[inline(always)]
    pub fn k_paddr(&self, layer: usize, pos: usize) -> u64 {
        let offset = layer * self.layer_stride_bytes + pos * self.token_stride_bytes;
        self.base_paddr + (offset as u64)
    }

    /// Calculate physical address for Value cache of (layer, pos)
    #[inline(always)]
    pub fn v_paddr(&self, layer: usize, pos: usize) -> u64 {
        let half_layer = self.layer_stride_bytes / 2;
        let offset = layer * self.layer_stride_bytes + half_layer + pos * self.token_stride_bytes;
        self.base_paddr + (offset as u64)
    }

    /// Calculate virtual address for Key cache of (layer, pos)
    #[inline(always)]
    pub fn k_vaddr(&self, layer: usize, pos: usize) -> usize {
        let offset = layer * self.layer_stride_bytes + pos * self.token_stride_bytes;
        self.base_vaddr + offset
    }

    /// Calculate virtual address for Value cache of (layer, pos)
    #[inline(always)]
    pub fn v_vaddr(&self, layer: usize, pos: usize) -> usize {
        let half_layer = self.layer_stride_bytes / 2;
        let offset = layer * self.layer_stride_bytes + half_layer + pos * self.token_stride_bytes;
        self.base_vaddr + offset
    }

    /// Store newly projected Key and Value tokens directly into the UMA HugePage slots
    pub fn write_kv_slot(
        &mut self,
        layer: usize,
        pos: usize,
        k_src: &[u16],
        v_src: &[u16],
    ) -> Result<(), &'static str> {
        if layer >= self.num_layers || pos >= self.max_seq_len {
            return Err("KV-Cache Layer or Position index out of range");
        }

        let len = self.num_heads * self.head_dim;
        if k_src.len() < len || v_src.len() < len {
            return Err("Source vector length smaller than head dimension");
        }

        unsafe {
            let k_dst = self.k_vaddr(layer, pos) as *mut u16;
            let v_dst = self.v_vaddr(layer, pos) as *mut u16;

            core::ptr::copy_nonoverlapping(k_src.as_ptr(), k_dst, len);
            core::ptr::copy_nonoverlapping(v_src.as_ptr(), v_dst, len);
        }

        if pos >= self.current_pos {
            self.current_pos = pos + 1;
        }

        Ok(())
    }

    #[inline(always)]
    pub fn current_pos(&self) -> usize {
        self.current_pos
    }

    #[inline(always)]
    pub fn reset(&mut self) {
        self.current_pos = 0;
    }
}
