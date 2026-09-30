#![no_std]

use axtp::topology_scheduler::TP_ENGINE;

pub struct LayerParallelExecutor;

impl LayerParallelExecutor {
    /// 執行單一 Transformer 層的並行矩陣計算
    pub unsafe fn forward_layer(
        _layer_index: usize,
        weight_base_pa: u64,
        kv_cache_pa: u64,
        hidden_dim: usize,
    ) {
        let chunk_size = (hidden_dim / 4) as u64; // TP=4, 切分為 4 個 Tile

        // 1. 將運算任務分派至 4 個內建 NPU Tile (同時間於 UMA 原地運算)
        for tile in 0..4 {
            let tile_weight_pa = weight_base_pa + (tile as u64 * chunk_size * 2); // FP16 (2 Bytes)

            TP_ENGINE.dispatch_matrix_multiply(
                tile as u8,
                tile as u8, // Core 0~3 一對一綁定
                tile_weight_pa,
                kv_cache_pa,
            );
        }

        // 2. 等候硬體加法樹自動完成 All-Reduce
        TP_ENGINE.wait_sync_barrier();
    }
}
