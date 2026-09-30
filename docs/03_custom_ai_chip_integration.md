# 自研 AI 晶片 (AeroTensor G1) 與 ArceOS 軟硬體協同整合指南

## 1. 晶片介面架構模型 (Hardware-Software Interface)

為了發揮 AI 晶片的極致效能，AeroTensor G1 晶片與 Host CPU（ARM64 / RISC-V 64 / x86_64）採用典型的高性能加速器架構。

可支援兩種實體介面：
1. **PCIe Gen4/Gen5 x8/x16**（外掛加速卡模式，透過 BAR 空間映射暫存器）。
2. **SoC 晶上 AXI-4 / NoC 總線**（邊緣晶片或嵌入式 SoC 模式，透過 MMIO 直接映射）。

```
+-------------------------------------------------------------------------+
|                              Host CPU                                   |
|   +-----------------------------------------------------------------+   |
|   |                       ArceOS Kernel                             |   |
|   |   +---------------------------------------------------------+   |   |
|   |   |             axcompute / AeroTensor Driver               |   |   |
|   |   +---------------------------------------------------------+   |   |
|   |     | (Write Cmd)                       | (Read Completion)         |   |
|   |     v                                   v                           |   |
|   |   [ SQ Ring Buffer ]                  [ CQ Ring Buffer ]            |   |
|   |   (Physical Host DMA Memory)          (Physical Host DMA Memory)    |   |
|   +-----------------------------------------------------------------+   |
+-------------------------------------------------------------------------+
       | (Ring Doorbell)                              ^ (MSI-X / Pin IRQ)
       v                                              |
+-------------------------------------------------------------------------+
|                   AeroTensor G1 AI Accelerator                          |
|   +-----------------------------------------------------------------+   |
|   | MMIO / PCIe BAR Control Registers                               |   |
|   |   - REG_DEV_STATUS / REG_DEV_CTRL                               |   |
|   |   - REG_SQ_DOORBELL / REG_CQ_HEAD                               |   |
|   |   - REG_IRQ_MASK / REG_IRQ_STATUS                               |   |
|   +-----------------------------------------------------------------+   |
|   +--------------------------+   +----------------------------------+   |
|   | Command Dispatcher & DMA |   | Tensor Engine / Matrix Multiply  |   |
|   | Controller               |   | Processing Units (TPU / Systolic)|   |
|   +--------------------------+   +----------------------------------+   |
|   +-----------------------------------------------------------------+   |
|   | On-Chip SRAM (Scratchpad Memory: 16MB ~ 64MB)                   |   |
|   +-----------------------------------------------------------------+   |
+-------------------------------------------------------------------------+
```

---

## 2. 暫存器規格定義 (Register Map / MMIO)

以 64-bit 對齊的暫存器佈局為例（BAR 0 或 MMIO Base 偏移量）：

| 偏移位址 (Offset) | 暫存器名稱 (Register Name) | 讀/寫 (R/W) | 說明 (Description) |
| :--- | :--- | :--- | :--- |
| `0x0000` | `MAGIC_NUMBER` | R | 讀取固定值 `0x4145524F` ("AERO")，確認晶片型號 |
| `0x0008` | `VERSION` | R | 晶片硬體架構版本（如 `0x0001_0000` 代表 G1 v1.0） |
| `0x0010` | `DEVICE_STATUS` | R | 晶片狀態（Bit 0: Ready, Bit 1: Busy, Bit 2: Error） |
| `0x0018` | `DEVICE_CONTROL` | R/W | 晶片控制（Bit 0: Reset, Bit 1: Enable, Bit 2: IRQ Enable） |
| `0x0020` | `SQ_BASE_PADDR` | R/W | 提交佇列（SQ）在主記憶體的 64-bit 實體位址 |
| `0x0028` | `SQ_DEPTH` | R/W | 提交佇列深度（例如 256, 1024 項） |
| `0x0030` | `SQ_TAIL_DOORBELL` | W | Host 寫入最新的 SQ Tail 索引，觸發晶片硬體抓取命令 |
| `0x0038` | `CQ_BASE_PADDR` | R/W | 完成佇列（CQ）在主記憶體的 64-bit 實體位址 |
| `0x0040` | `CQ_DEPTH` | R/W | 完成佇列深度 |
| `0x0048` | `CQ_HEAD` | R/W | Host 讀取並更新已處理的 CQ Head 索引 |
| `0x0050` | `IRQ_STATUS` | R/W1C | 中斷狀態暫存器（寫 1 清除中斷） |
| `0x0058` | `IRQ_MASK` | R/W | 中斷遮罩暫存器 |
| `0x0080` | `SRAM_BASE_PADDR` | R | 晶上 SRAM 實體基底位址 |
| `0x0088` | `SRAM_SIZE_BYTES` | R | 晶上 SRAM 總容量位元組數 |

---

## 3. 環形緩衝區（Ring Buffer）與命令描述符

AeroTensor 採用類 NVMe / RDMA 的非同步環形佇列模型：

### 3.1 提交佇列描述符 (Submission Queue Entry - SQE, 64 Bytes)

每個命令大小固定為 64 位元組，確保剛好佔據一個 CPU 快取行（Cache Line），避免偽共享（False Sharing）：

```rust
#[repr(C, align(64))]
#[derive(Debug, Clone, Copy)]
pub struct AeroTensorSQE {
    pub opcode: u16,          // 運算碼: 0x01=DMA_IN, 0x02=DMA_OUT, 0x10=MATMUL, 0x11=CONV2D, 0x20=ACTIVATION
    pub flags: u16,           // 標誌位: Bit 0: Completion IRQ Enable, Bit 1: Fence/Barrier
    pub command_id: u32,      // Host 分配的唯一請求標識符，用於在 CQ 中匹配
    pub src_addr: u64,        // 來源實體位址 (Host 實體記憶體或晶上 SRAM 位址)
    pub dst_addr: u64,        // 目標實體位址 (Host 實體記憶體或晶上 SRAM 位址)
    pub weight_addr: u64,     // 權重張量實體位址 (可選)
    pub dim_m: u32,           // 矩陣維度 M (或 Batch / Channel)
    pub dim_k: u32,           // 矩陣維度 K (或 Input Feature)
    pub dim_n: u32,           // 矩陣維度 N (或 Output Feature)
    pub data_type: u32,       // 數據型別: 0=FP32, 1=FP16, 2=BF16, 3=INT8, 4=INT4
    pub extra_params: [u32; 4], // 算子專屬參數（如 ReLU/GELU 激活型別、Padding、Stride 等）
}
```

### 3.2 完成佇列描述符 (Completion Queue Entry - CQE, 16 Bytes)

```rust
#[repr(C, align(16))]
#[derive(Debug, Clone, Copy)]
pub struct AeroTensorCQE {
    pub command_id: u32,      // 對應 SQE 的 command_id
    pub status: u32,          // 執行結果: 0=Success, 1=InvalidParam, 2=HardwareError, 3=Timeout
    pub cycles_taken: u64,    // 晶片硬體執行的時脈週期計數（用於極致性能 Profiling）
}
```

---

## 4. ArceOS 驅動層抽象設計 (`TensorComputeDriver`)

在 ArceOS `modules/axdriver` 中擴充 `compute` 設備類別：

```rust
use axerrno::AxResult;

/// AI 計算加速器抽象介面
pub trait TensorComputeDriver: Send + Sync {
    /// 取得硬體名稱與型號
    fn name(&self) -> &str;

    /// 取得晶上 SRAM 實體範圍
    fn sram_region(&self) -> Option<(usize, usize)>;

    /// 提交一批運算描述符至硬體 SQ，返回提交數量
    fn submit_batch(&mut self, sqes: &[AeroTensorSQE]) -> AxResult<usize>;

    /// 敲擊 Doorbell 通知晶片開始執行
    fn flush_doorbell(&mut self);

    /// 輪詢完成佇列 CQ，取得已完成的 CQE
    fn poll_completions(&mut self, cqes: &mut [AeroTensorCQE]) -> usize;

    /// 是否產生中斷
    fn handle_irq(&mut self) -> bool;
}
```

### 4.1 快取一致性處置 (Cache Invalidation & Flush)
在非硬體快取一致（Non-coherent）的嵌入式或邊緣架構下：
- **Host 寫入輸入張量後**：必須在敲擊 Doorbell 前執行 `axhal::arch::cache_flush(tensor_vaddr, size)`，確保資料寫回實體 DDR。
- **晶片計算完成並 DMA 回寫主記憶體後**：Host CPU 讀取前必須執行 `axhal::arch::cache_invalidate(tensor_vaddr, size)`，強迫 CPU 重新從實體記憶體讀取最新數據。
- 在支援硬體 CCI（Cache Coherent Interconnect，如 CXL 或 ARM AMBA 5 CHI）的架構中，則可跳過快取操作，實現純零延遲直讀。

---

## 5. 實體頻寬與功耗熱設計 (Physical Bandwidth & Power Specifications)

AeroTensor-G1 針對航太載具、車載邊緣與高密度伺服器提供明確的頻寬階層與散熱規範：

### 5.1 頻寬規格表 (Bandwidth Breakdown)
| 階層 (Level) | 介面 / 協定 | 匯流排位寬 / 頻率 | 峰值頻寬 (Peak Bandwidth) | 特性說明 |
| :--- | :--- | :--- | :--- | :--- |
| **L1 SRAM** | 16-Tile 內部 Scratchpad | 512-bit x2 / Tile @ 1.4 GHz | **2,867 GB/s (2.87 TB/s)** | 零等待循環，直接提供 Systolic 乘加器矩陣操作數 |
| **L3 SLC** | 16-Slice 分散式系統快取 | 512-bit / Slice @ 2.0 GHz | **2,048 GB/s (2.05 TB/s)** | 共享快取，攔截跨 Tile 與 CPU 之間的熱門資料 |
| **片上網絡 (NoC)** | AMBA 5 CHI 4x4 2D Torus | 512-bit Flit @ 2.0 GHz | **1,024 GB/s (1.02 TB/s)** | 二分頻寬；全晶片 Crossbar 聚合吞吐量達 4.1 TB/s |
| **統一記憶體 (UMA)** | 4x 24GB 12-Hi HBM3e 堆疊 | 4096-bit 總寬 @ 6.25 Gbps | **3,200 GB/s (3.20 TB/s)** | CPU 與 NPU 共享 96GB 物理位址，零拷貝 DMA 直通 |

### 5.2 熱設計功耗 (TDP) 與散熱設定
* **資料中心標準模式 (Server TDP: 450 W)**:
  - 適用場景：機架式 2U 伺服器、航電集中計算艙。
  - 散熱配置：標準 2U 風冷鰭片或液冷冷板。
  - 輸出算力：512 TFLOPS FP16 dense, 1024 TOPS INT8。
* **航太/無人機低功耗模式 (Flight Mission TDP: 230 W)**:
  - 適用場景：抗輻照密閉艙、戰術無人載具機載電腦。
  - 動態能耗調節 (DVFS)：CPU 降至 2.2 GHz (55W)、NPU 降至 1.0 GHz (95W)、HBM3e 降至 4.8 Gbps (48W, 2.45 TB/s)。
  - 總功耗收斂至 230 W，滿足嚴苛之無風扇或熱管傳導散熱邊界。
