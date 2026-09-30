#![no_std]

use core::sync::atomic::{AtomicU32, Ordering};
use core::ptr::{read_volatile, write_volatile};

pub const HBU_REG_BASE: usize = 0xFFFF_0000_2000_0000;
pub const NPU_REG_BASE: usize = 0xFFFF_0000_2010_0000;

pub struct CoreTpEngine {
    barrier_mask: AtomicU32,
    sync_epoch: AtomicU32,
}

impl CoreTpEngine {
    pub const fn new() -> Self {
        Self {
            barrier_mask: AtomicU32::new(0),
            sync_epoch: AtomicU32::new(0),
        }
    }

    /// 設定親和性並指派 NPU Tile 開始矩陣運算
    pub unsafe fn dispatch_matrix_multiply(
        &self,
        tile_id: u8,
        core_id: u8,
        weight_pa: u64,
        kv_pa: u64,
    ) {
        // 1. 設定 CPU 親和性
        #[cfg(target_arch = "aarch64")]
        core::arch::asm!("msr tpidr_el1, {0}", in(reg) core_id as u64, options(nomem, nostack));
        #[cfg(not(target_arch = "aarch64"))]
        let _ = core_id;

        // 2. 寫入 Tile MMIO 指令 (如果是模擬測試環境，防止野指標崩潰，做邊界校驗)
        if NPU_REG_BASE != 0 {
            // 在特定硬體或模擬環境寫入
            #[cfg(target_os = "none")]
            {
                let tile_ptr = (NPU_REG_BASE + (tile_id as usize * 0x10000)) as *mut u64;
                write_volatile(tile_ptr.add(0), weight_pa);
                write_volatile(tile_ptr.add(1), kv_pa);
                write_volatile(tile_ptr.add(2), 1); // START 計算
            }
            #[cfg(not(target_os = "none"))]
            {
                let _ = (tile_id, weight_pa, kv_pa);
            }
        }
    }

    /// 核心態自旋等待硬體屏障 (WFE)
    pub fn wait_sync_barrier(&self) {
        let current = self.sync_epoch.load(Ordering::Acquire);
        loop {
            if self.sync_epoch.load(Ordering::Acquire) != current {
                break;
            }
            #[cfg(target_arch = "aarch64")]
            unsafe { core::arch::asm!("wfe", options(nomem, nostack)) };
            #[cfg(not(target_arch = "aarch64"))]
            core::hint::spin_loop();
        }
    }

    /// GICv3 SPI 中斷常式（由 Tile 計算結束引發）
    pub fn on_tile_interrupt_received(&self, tile_id: u8, target_nodes: u8) {
        let bit = 1u32 << tile_id;
        let prev = self.barrier_mask.fetch_or(bit, Ordering::AcqRel);
        let current = prev | bit;
        let expected = (1u32 << target_nodes) - 1;

        if current == expected {
            unsafe {
                #[cfg(target_os = "none")]
                {
                    // 觸發片上硬體 Reduction Tree
                    let hbu_cmd = (HBU_REG_BASE + 0x0010) as *mut u32;
                    write_volatile(hbu_cmd, 0x8000_0001); // FP16 加總

                    // 輪詢等待硬體 Reduction 完成標誌
                    while (read_volatile(hbu_cmd) & (1 << 30)) == 0 {}
                }
            }

            // 重設屏障並喚醒所有核心
            self.barrier_mask.store(0, Ordering::Release);
            self.sync_epoch.fetch_add(1, Ordering::Release);

            #[cfg(target_arch = "aarch64")]
            unsafe { core::arch::asm!("sev", options(nomem, nostack)) };
        }
    }
}

pub static TP_ENGINE: CoreTpEngine = CoreTpEngine::new();
