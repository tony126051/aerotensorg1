# AeroTensor OS (AI Native OS) 專案文檔中心

本目錄存放基於 [ArceOS](https://github.com/arceos-org/arceos) 模組化 Unikernel 架構、搭配自研 AI 晶片（AeroTensor G1）所規劃的 **AI Native OS 原型系統** 全套架構設計、實作代碼與驗收報告。

## 文檔清單與導航 (Documentation Index)

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
   - 四階段演進藍圖：Phase 1 軟體模擬 -> Phase 2 零拷貝張量管線 -> Phase 3 板級硬化與 16-Tile 排程 -> Phase 4 推理引擎與冷啟動驗收
   - 各階段實作代碼與交付驗收指標 (100% 驗收通過)
   - TTFT 4.32ms、1,470 Tokens/s 基準測量實證

5. [Multi-LLM 並行共存與 Candle 架構無痛轉化白皮書](./05_multi_llm_and_candle_integration.md) **[新核心技術]**
   - Candle 深度架構研讀與 Unikernel 移植邊界（為什麼不能直接搬整套 Candle？）
   - 「外借其形，內造其心」：純 `no_std` 零堆疊配置 Safetensors 解析器與硬體直通 Transformer 算子
   - 多模型共存掛載機制 (`MultiModelRegistry`)：記憶體衝突碰撞檢測、16-Tile 空間分區 (`tile_mask`)、搶佔優先權排程
   - 2MB 巨大頁槽位式 KV-Cache 與 12 項全鏈路測試套件

6. [AI Agent 協同研發工作流與工程技能規範](./06_agent_workflow_and_skills.md) **[工程協同標準]**
   - 軟硬體協同 AI Agent 四大角色分工（硬體、內核、張量、測試）
   - 5 步驟閉環研發流程（規格對齊 -> no_std 實作 -> 範例橋接 -> C 測試驗收 -> 指標量化）
   - 專案自定義技能 (`.agents/skills/aerotensor-dev`) 與避坑指南
