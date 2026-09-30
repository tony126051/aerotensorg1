#![no_std]

use crate::kv_cache::KVCache;
use crate::models::llama::{LlamaLayerScratch, LlamaModel};
use axcompute::tensor::Tensor;

pub const MAX_MOUNTED_MODELS: usize = 8;

/// Descriptor defining resource allocations for an isolated mounted LLM
#[derive(Clone, Copy, Debug)]
pub struct ModelDescriptor {
    pub model_id: u32,
    pub name: &'static str,
    pub base_paddr: u64,
    pub weight_size_bytes: usize,
    pub tile_mask: u16,  // Bitmask of assigned NPU Tiles (e.g. 0x000F for Tiles 0..3)
    pub priority: u8,   // 0 = Normal/Batch, 1 = Realtime/Latency-Critical
}

/// A mounted, isolated LLM instance with dedicated KV-Cache and Tile affinity
pub struct MultiModelInstance {
    pub descriptor: ModelDescriptor,
    pub model: LlamaModel,
    pub kv_cache: KVCache,
}

/// Central OS-level Multi-Model Registry and Co-Scheduler
pub struct MultiModelRegistry {
    instances: [Option<MultiModelInstance>; MAX_MOUNTED_MODELS],
    count: usize,
}

impl MultiModelRegistry {
    pub const fn new() -> Self {
        const NONE_INST: Option<MultiModelInstance> = None;
        Self {
            instances: [NONE_INST; MAX_MOUNTED_MODELS],
            count: 0,
        }
    }

    /// Mount and register a new LLM model with spatial/temporal resource verification
    pub fn mount_model(&mut self, instance: MultiModelInstance) -> Result<u32, &'static str> {
        if self.count >= MAX_MOUNTED_MODELS {
            return Err("Max mounted models capacity reached");
        }

        // 1. Verify physical memory range isolation against previously mounted models
        let new_start = instance.descriptor.base_paddr;
        let new_end = new_start + instance.descriptor.weight_size_bytes as u64;

        for i in 0..self.count {
            if let Some(ref existing) = self.instances[i] {
                let ex_start = existing.descriptor.base_paddr;
                let ex_end = ex_start + existing.descriptor.weight_size_bytes as u64;

                // Check overlap
                if !(new_end <= ex_start || new_start >= ex_end) {
                    return Err("Physical memory collision with existing mounted model");
                }
            }
        }

        let id = instance.descriptor.model_id;
        self.instances[self.count] = Some(instance);
        self.count += 1;

        log::info!("Mounted LLM Model ID: {}, Total Mounted: {}", id, self.count);
        Ok(id)
    }

    /// Query mounted model by ID
    pub fn get(&self, model_id: u32) -> Option<&MultiModelInstance> {
        for i in 0..self.count {
            if let Some(ref inst) = self.instances[i] {
                if inst.descriptor.model_id == model_id {
                    return Some(inst);
                }
            }
        }
        None
    }

    /// Query mutable mounted model by ID
    pub fn get_mut(&mut self, model_id: u32) -> Option<&mut MultiModelInstance> {
        for i in 0..self.count {
            if let Some(ref mut inst) = self.instances[i] {
                if inst.descriptor.model_id == model_id {
                    return Some(inst);
                }
            }
        }
        None
    }

    /// Execute an autoregressive forward step for a specific target model
    pub fn forward_step(
        &mut self,
        model_id: u32,
        token_id: usize,
        pos: usize,
        hidden_state: &mut Tensor,
        out_logits: &mut Tensor,
        scratch: &mut LlamaLayerScratch,
    ) -> Result<(), &'static str> {
        let inst = self
            .get_mut(model_id)
            .ok_or("Target model ID not found in registry")?;

        inst.model.forward_step(
            token_id,
            pos,
            &mut inst.kv_cache,
            hidden_state,
            out_logits,
            scratch,
        )
    }

    #[inline(always)]
    pub fn mounted_count(&self) -> usize {
        self.count
    }
}
