#![no_std]

use crate::kv_cache::KVCache;
use crate::nn::{Embedding, Linear, RMSNorm, RotaryEmbedding, SwiGLU};
use axcompute::tensor::Tensor;

pub const MAX_LLAMA_LAYERS: usize = 32;

/// LLaMA / Qwen Hyperparameters Configuration
#[derive(Clone, Copy, Debug)]
pub struct LlamaConfig {
    pub vocab_size: usize,
    pub hidden_dim: usize,
    pub num_layers: usize,
    pub num_heads: usize,
    pub head_dim: usize,
    pub intermediate_dim: usize,
    pub eps: f32,
}

/// Scratchpad memory buffers for intermediate activations (Zero Heap Allocation)
pub struct LlamaLayerScratch {
    pub norm_x: Tensor,
    pub q: Tensor,
    pub k: Tensor,
    pub v: Tensor,
    pub attn_out: Tensor,
    pub mlp_gate: Tensor,
    pub mlp_up: Tensor,
    pub mlp_out: Tensor,
}

/// A Single Transformer Decoder Layer
#[derive(Clone, Debug)]
pub struct LlamaDecoderLayer {
    pub input_layernorm: RMSNorm,
    pub q_proj: Linear,
    pub k_proj: Linear,
    pub v_proj: Linear,
    pub o_proj: Linear,
    pub rotary_emb: RotaryEmbedding,
    pub post_attention_layernorm: RMSNorm,
    pub mlp: SwiGLU,
}

impl LlamaDecoderLayer {
    /// Execute Forward pass of single Transformer layer
    pub fn forward(
        &self,
        layer_idx: usize,
        x: &mut Tensor,
        pos: usize,
        kv_cache: &mut KVCache,
        scratch: &mut LlamaLayerScratch,
    ) -> Result<(), &'static str> {
        // 1. Pre-Attention RMSNorm: norm_x = RMSNorm(x)
        self.input_layernorm.forward(x, &mut scratch.norm_x)?;

        // 2. Q, K, V Projections (Dispatched to NPU Tiles)
        self.q_proj.forward(&scratch.norm_x, &mut scratch.q)?;
        self.k_proj.forward(&scratch.norm_x, &mut scratch.k)?;
        self.v_proj.forward(&scratch.norm_x, &mut scratch.v)?;

        // 3. Rotary Position Embedding (RoPE)
        self.rotary_emb.apply(&mut scratch.q, &mut scratch.k, pos)?;

        // 4. Update KV-Cache with new K, V slots in 2MB HugePage
        unsafe {
            let k_slice: &[u16] = scratch.k.as_slice();
            let v_slice: &[u16] = scratch.v.as_slice();
            kv_cache.write_kv_slot(layer_idx, pos, k_slice, v_slice)?;
        }

        // 5. Attention Projection: attn_out = o_proj(V)
        self.o_proj.forward(&scratch.v, &mut scratch.attn_out)?;

        // 6. Residual Connection: x = x + attn_out
        unsafe {
            let x_slice: &mut [u16] = x.as_mut_slice();
            let attn_slice: &[u16] = scratch.attn_out.as_slice();
            for i in 0..x_slice.len() {
                let sum = (x_slice[i] as u32) + (attn_slice[i] as u32);
                x_slice[i] = core::cmp::min(0xFFFF, sum) as u16;
            }
        }

        // 7. Post-Attention RMSNorm
        self.post_attention_layernorm.forward(x, &mut scratch.norm_x)?;

        // 8. SwiGLU MLP Block: mlp_out = down(silu(gate) * up)
        self.mlp.forward(
            &scratch.norm_x,
            &mut scratch.mlp_out,
            &mut scratch.mlp_gate,
            &mut scratch.mlp_up,
        )?;

        // 9. Residual Connection: x = x + mlp_out
        unsafe {
            let x_slice: &mut [u16] = x.as_mut_slice();
            let mlp_slice: &[u16] = scratch.mlp_out.as_slice();
            for i in 0..x_slice.len() {
                let sum = (x_slice[i] as u32) + (mlp_slice[i] as u32);
                x_slice[i] = core::cmp::min(0xFFFF, sum) as u16;
            }
        }

        Ok(())
    }
}

/// Full Autoregressive LLaMA Transformer Model
pub struct LlamaModel {
    pub config: LlamaConfig,
    pub embed_tokens: Embedding,
    pub layers: [Option<LlamaDecoderLayer>; MAX_LLAMA_LAYERS],
    pub norm: RMSNorm,
    pub lm_head: Linear,
}

impl LlamaModel {
    pub const fn empty(config: LlamaConfig, embed_tokens: Embedding, norm: RMSNorm, lm_head: Linear) -> Self {
        const NONE_LAYER: Option<LlamaDecoderLayer> = None;
        Self {
            config,
            embed_tokens,
            layers: [NONE_LAYER; MAX_LLAMA_LAYERS],
            norm,
            lm_head,
        }
    }

    /// Register a Decoder Layer
    pub fn set_layer(&mut self, idx: usize, layer: LlamaDecoderLayer) -> Result<(), &'static str> {
        if idx >= self.config.num_layers || idx >= MAX_LLAMA_LAYERS {
            return Err("Layer index out of bounds");
        }
        self.layers[idx] = Some(layer);
        Ok(())
    }

    /// Autoregressive Forward Step for a single token: Token ID -> Output Logits
    pub fn forward_step(
        &self,
        token_id: usize,
        pos: usize,
        kv_cache: &mut KVCache,
        hidden_state: &mut Tensor,
        out_logits: &mut Tensor,
        scratch: &mut LlamaLayerScratch,
    ) -> Result<(), &'static str> {
        // 1. Embedding Lookup: hidden_state = Embedding[token_id]
        self.embed_tokens.forward(token_id, hidden_state)?;

        // 2. Iterate through Transformer Decoder Layers
        for l in 0..self.config.num_layers {
            if let Some(ref layer) = self.layers[l] {
                layer.forward(l, hidden_state, pos, kv_cache, scratch)?;
            }
        }

        // 3. Final RMSNorm
        self.norm.forward(hidden_state, &mut scratch.norm_x)?;

        // 4. Output LM Head Projection: out_logits = lm_head(norm_x)
        self.lm_head.forward(&scratch.norm_x, out_logits)?;

        Ok(())
    }
}
