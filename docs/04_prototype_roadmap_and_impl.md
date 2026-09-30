# AI Native OS 原型開發路線圖與 PoC 實作方案

## 1. 原型開發分期規劃 (Roadmap & Milestones)

為了降低軟硬體耦合風險，建議採用 **「模擬先行 -> 軟體棧垂直打通 -> FPGA 原型板驗證 -> 晶片帶片驗收（Silicon Bring-up）」** 的四階段迭代策略：

| 階段 (Phase) | 里程碑目標 (Milestone Target) | 核心產出 (Deliverables) | 預估驗收標準 |
| :--- | :--- | :--- | :--- |
| **Phase 1: 架構驗證與軟體模擬器 (Emulation & Mock Driver)** | 在 QEMU (RISC-V / ARM64) 環境下擴充 ArceOS，實作軟體模擬的 AeroTensor G1 虛擬設備 | - 建立 `modules/axcompute`<br>- 實作虛擬 SQ/CQ 與軟體矩陣乘法運算子<br>- 完成單一應用程序端到端跑通 | 成功在 QEMU 中執行 1024x1024 FP16 GEMM 計算，驗證環形緩衝區無死鎖 |
| **Phase 2: 零拷貝張量記憶體與非同步執行棧** | 整合 `axdma`，實作連續物理大頁張量分配器與 Rust `Future` 算子非同步管線 | - `axtensor_mem` 大頁分配器<br>- 非同步 `async/await` 算子排程<br>- 雙緩衝（Ping-Pong Buffer）流水線 | CPU 資料前處理與 AI 晶片 DMA/Compute 時間完全重疊（Overlapped） |
| **Phase 3: FPGA 原型板實機調試 (Hardware In-the-Loop)** | 將 ArceOS 移植至搭載 PCIe / AXI 的 FPGA 驗證板（如 Xilinx VU19P / ZCU102 或自研 SoC 原型） | - 實際 MMIO / PCIe BAR 驅動<br>- 實體 MSI-X / Pin 中斷處置<br>- 硬體快取無效化/刷寫驗證 | 測得真實硬體延遲與頻寬，驗證長時間高負載穩定性 |
| **Phase 4: AI Native 推理引擎與模型載入** | 在 ArceOS `axstd` 上移植純 Rust 輕量推理引擎（如微型 ONNX Runtime 或 Burn/Candle 子集） | - 支援標準模型格式權重讀取<br>- 算子自動下發至 AeroTensor G1<br>- 建立微秒級啟動的 AI 邊緣專用系統二進位 | 實現免 Linux OS 依賴、開機 20ms 內完成首個 Token/特徵推理 |

---

## 2. ArceOS 擴展代碼結構建議

在現有的 ArceOS 目錄樹中，新增專屬模組：

```
arceos/
├── modules/
│   ├── axcompute/             # [新增] AI 運算加速核心子系統
│   │   ├── Cargo.toml
│   │   └── src/
│   │       ├── lib.rs         # 模組對外統一匯出
│   │       ├── device.rs      # AeroTensor 設備抽象與暫存器定義
│   │       ├── queue.rs       # SQ/CQ 環形緩衝區控制邏輯
│   │       ├── tensor.rs      # 張量結構描述符與大頁記憶體封裝
│   │       └── ops.rs         # 高階算子封裝 (MatMul, Conv2d, Softmax)
│   ├── axdriver/
│   │   └── src/
│   │       └── compute/       # [擴展] 註冊至 axdriver 設備容器
├── api/
│   └── arceos_api/
│       └── src/
│           └── compute.rs     # [擴展] 匯出 ax_compute_* 系統介面
└── examples/
    └── ai-gemm-demo/          # [新增] 端到端 AI 原型驗證 App
        ├── Cargo.toml
        └── src/
            └── main.rs
```

---

## 3. 關鍵原型代碼設計 (PoC Implementation Design)

### 3.1 `modules/axcompute/src/queue.rs`（環形佇列核心邏輯）

```rust
use core::sync::atomic::{AtomicU32, Ordering};
use crate::device::{AeroTensorSQE, AeroTensorCQE};

pub struct CommandQueue {
    sq_entries: &'static mut [AeroTensorSQE],
    cq_entries: &'static mut [AeroTensorCQE],
    sq_head: u32,
    sq_tail: u32,
    cq_head: u32,
    depth: u32,
}

impl CommandQueue {
    pub fn new(
        sq_entries: &'static mut [AeroTensorSQE],
        cq_entries: &'static mut [AeroTensorCQE],
        depth: u32,
    ) -> Self {
        Self {
            sq_entries,
            cq_entries,
            sq_head: 0,
            sq_tail: 0,
            cq_head: 0,
            depth,
        }
    }

    /// 提交一個算子指令
    pub fn push_sqe(&mut self, sqe: AeroTensorSQE) -> Result<(), &'static str> {
        let next_tail = (self.sq_tail + 1) % self.depth;
        if next_tail == self.sq_head {
            return Err("SQ is full");
        }
        self.sq_entries[self.sq_tail as usize] = sqe;
        self.sq_tail = next_tail;
        Ok(())
    }

    /// 取得當前 SQ Tail，用於敲擊硬體 Doorbell
    pub fn current_sq_tail(&self) -> u32 {
        self.sq_tail
    }

    /// 輪詢完成狀態
    pub fn poll_cqe(&mut self) -> Option<AeroTensorCQE> {
        let cqe = self.cq_entries[self.cq_head as usize];
        if cqe.status != 0xFFFF_FFFF { // 假設 0xFFFF_FFFF 代表硬體未完成標記
            self.cq_entries[self.cq_head as usize].status = 0xFFFF_FFFF; // 清除
            self.cq_head = (self.cq_head + 1) % self.depth;
            Some(cqe)
        } else {
            None
        }
    }
}
```

### 3.2 `examples/ai-gemm-demo/src/main.rs`（應用端使用範例）

```rust
#![no_std]
#![no_main]

#[macro_use]
extern crate axstd;

use axstd::println;
use axcompute::tensor::{Tensor, DataType};
use axcompute::ops::MatMul;

#[unsafe(no_mangle)]
fn main() {
    println!("==================================================");
    println!("  AeroTensor AI Native OS - Sub-millisecond GEMM  ");
    println!("==================================================");

    // 1. 分配物理連續且快取對齊的輸入張量 (A: M x K, B: K x N)
    let m = 512;
    let k = 512;
    let n = 512;

    println!("[AI-OS] Allocating contiguous zero-copy tensors...");
    let tensor_a = Tensor::zeros(&[m, k], DataType::Float16).expect("Alloc A failed");
    let tensor_b = Tensor::zeros(&[k, n], DataType::Float16).expect("Alloc B failed");
    let mut tensor_c = Tensor::zeros(&[m, n], DataType::Float16).expect("Alloc C failed");

    // 2. 透過原生 axcompute 下發 GEMM 計算圖
    println!("[AI-OS] Submitting MatMul to AeroTensor G1 Accelerator...");
    let start_time = axstd::time::Instant::now();

    let job_handle = MatMul::new(&tensor_a, &tensor_b, &mut tensor_c)
        .with_relu(true)
        .dispatch()
        .expect("Dispatch failed");

    // 3. 非同步等待或硬體中斷通知
    job_handle.wait_complete();
    let elapsed = start_time.elapsed();

    println!("[AI-OS] GEMM 512x512 completed in {} microseconds!", elapsed.as_micros());
    println!("==================================================");
}
```

---

## 4. 總結與建議

透過將 ArceOS 作為 AI Native OS 的底座，我們獲得了：
1. **極速啟動與極小佔用**：捨棄 Linux 龐大冗餘的驅動與多進程機制，模型專注於專用硬體。
2. **極致硬體直達**：消除了驅動在用戶態與內核態之間的來回切換，張量吞吐達到匯流排物理極限。
3. **軟硬體協同優勢**：在晶片流片前期即可藉由 ArceOS 模擬環境同步開發編譯器後端、算子庫與執行時期，實現 **「晶片回片當日即完成作業系統點亮與模型推論」**。
