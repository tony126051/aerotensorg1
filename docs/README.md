# AeroTensor OS (AI Native OS) 專案文檔中心

本目錄存放基於 [ArceOS](https://github.com/arceos-org/arceos) 模組化 Unikernel 架構、搭配自研 AI 晶片（AeroTensor G1）所規劃的 **AI Native OS 原型系統** 全套架構設計與研讀報告。

## 文檔清單與導航

0. [AeroTensor-G1 晶片與 ArceOS 軟硬體整合架構規格書 (SDD)](./00_aerotensor_g1_sdd.md) **[官方基準規範]**
   - 64-Core ARMv9.2-A + 16-Tile NPU + AMBA 5 CHI Coherent NoC + 96GB HBM3e UMA
   - QEMU 虛擬週邊模型 (`aerotensor_soc.c`) 與啟動指令碼
   - ArceOS 混合大頁（1GB/2MB VMSA）、SMMUv3 SVA（共享 TTBR0）、拓撲感知的 4x4 Mesh 排程器
   - 零拷貝 LLM 模型載入器與張量並行（TP=4）Transformer 推論引擎
   - 整合自測套件與驗證日誌

1. [ArceOS 核心架構深度解析與研讀報告](./01_arceos_deep_dive.md)
   - Unikernel vs 傳統宏內核 (Monolithic) vs 微內核 (Microkernel) 對比
   - ArceOS 分層架構：Crates、Modules、API、Ulib
   - 核心流程剖析：引導啟動 (Bootstrapping)、記憶體管理 (`axalloc`, `axmm`, `axdma`)、多工排程 (`axtask`)、驅動框架 (`axdriver`)
   - ArceOS 在 AI 負載下的優勢與現存局限分析

2. [AI Native OS 原型系統設計架構 (AeroTensor OS)](./02_ai_native_os_architecture.md)
   - AI Native OS 的核心典範轉移：以張量（Tensor）為作業系統第一類公民
   - 總體架構設計：應用層、AI 執行時期、原生內核模組、自研晶片層
   - 原生張量記憶體子系統 (`axtensor_mem`) 與晶上 SRAM 分層管理
   - 異構協同排程架構（Heterogeneous Co-Scheduling）與 DAG 算子流水線
   - 確定性執行與超低抖動（Zero OS Jitter）保障機制

3. [自研 AI 晶片 (AeroTensor G1) 與 ArceOS 軟硬體協同整合指南](./03_custom_ai_chip_integration.md)
   - 晶片介面模型：PCIe Gen4/5 與 SoC MMIO
   - 暫存器規格對齊（Register Map: Magic, Control/Status, Doorbell, SRAM）
   - 環形緩衝區架構：提交佇列 (SQE, 64-Byte) 與完成佇列 (CQE, 16-Byte)
   - ArceOS 驅動層抽象介面 (`TensorComputeDriver`)
   - 非一致性架構下的快取無效化與寫回（Cache Flush / Invalidation）機制

4. [AI Native OS 原型開發路線圖與 PoC 實作方案](./04_prototype_roadmap_and_impl.md)
   - 四階段演進藍圖：QEMU 軟體模擬 -> 零拷貝張量管線 -> FPGA 實體驗證 -> 帶片驗收
   - ArceOS 代碼庫目錄擴展規劃（`modules/axcompute`, `examples/ai-gemm-demo`）
   - 環形佇列控制核心代碼設計
   - 端到端 AI 應用調用範例代碼
