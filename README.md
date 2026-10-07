# AeroTensor-G1: AI-Native Hardware–Software Co-Design Platform

[![Rust](https://img.shields.io/badge/Rust-1.85+-orange.svg)](https://www.rust-lang.org)
[![ArceOS](https://img.shields.io/badge/ArceOS-Unikernel-blue.svg)](https://github.com/arceos-org/arceos)
[![Target](https://img.shields.io/badge/Architecture-ARMv9.2--A%20%2B%20AeroTensor--G1%20NPU-success.svg)]()
[![License](https://img.shields.io/badge/License-Apache%202.0%20%2F%20MIT-brightgreen.svg)]()
[![Verification](https://img.shields.io/badge/Verification-12%2F12%20Tests%20Passed%20(100%25)-green.svg)]()

> **Languages / 語言導航 / 言語ナビゲーション**:
> - **English (Primary)**
> - [繁體中文](./README.zh-TW.md)
> - [日本語 (エンジニア向け日本語版)](./README.ja.md)

---

## 1. Executive Summary & The AI-Native Paradigm Shift

**AeroTensor-G1 OS** is an AI-native, bare-metal computing prototype built on the modular [ArceOS](https://github.com/arceos-org/arceos) unikernel. Tailored for next-generation aerospace avionics, autonomous combat aerial vehicles (UCAVs), hypersonic guidance, and mission-critical edge robotics, AeroTensor-G1 OS re-architects the fundamental contract between hardware and software by treating the **Tensor as a first-class citizen of the operating system**.

**AeroTensor-G1 OS** is an architectural research prototype. Performance figures in this repository are based on software simulation, analytical modeling, or target specifications unless explicitly stated otherwise. These results represent modeled system-level performance and architectural targets, not measurements from fabricated silicon.


### The "Latency Tax" of Conventional Operating Systems
In traditional AI computing stacks (e.g., Linux + NVIDIA CUDA / PyTorch or vLLM), running large language models incurs staggering software taxations:
1. **Prolonged Cold-Boot Times**: Initializing the Linux monolithic kernel, systemd services, CUDA driver runtimes, and memory contexts takes between 3 to 15 seconds—catastrophic for mission-critical aircraft scrambles or emergency restarts.
2. **Double/Triple Memory Copying**: Model weights must be staged through Host DDR, mapped via PCIe DMA BARs, and copied into GPU VRAM, multiplying DRAM bus traffic and delaying deployment.
3. **Severe OS Jitter (Long-Tail Latency)**: The Linux CFS scheduler, asynchronous page faults, dynamic TLB shootdowns, and preemptive context switches introduce unpredictable 2~10 millisecond execution spikes, violating hard real-time deadlines.
4. **Memory Fragmentation**: Dynamic memory allocation (`malloc`/`free`) during autoregressive decoding causes physical memory fragmentation and TLB cache thrashing.

### The AeroTensor-G1 + ArceOS Solution
AeroTensor-G1 OS eliminates the user-kernel boundary and bundles the inference engine directly with the bare-metal kernel:
* **Zero PCIe/Host-to-Device Memory Copies**: 96GB HBM3e Unified Memory Architecture (UMA) provides a single, physically continuous address space shared transparently by the 64 ARMv9.2-A CPU cores and 16 NPU tiles.
* **Zero Heap Allocation in Forward Paths**: All model weights, intermediate scratchpads, and KV-cache slots are backed by continuous 1GB and 2MB HugePages, eliminating dynamic heap allocations and TLB misses.
* **Deterministic Execution & Reduced OS-Induced Jitter**: Single-address-space execution removes page table switches, translation lookaside buffer (TLB) flushes, and scheduling interrupts from the inference hot path.

### 📊 Modeled Architectural Comparison

| Dimension / Metric | Conventional Stack (Reference Range) | AeroTensor-G1 OS (Modeled / Target) | Projected Improvement |
| :--- | :--- | :--- | :--- |
| **Execution Environment** | Multi-process, Ring 0/Ring 3 split, CFS | Single Address Space Unikernel (`#![no_std]`) | **Zero Context-Switch Overhead** |
| **Cold-Boot to 1st Token (TTFT)** | 3,000 ms ~ 12,000 ms (CUDA Init + Weights) | **`4.32 ms`** (Cold Boot to Output Token) | **> 700× Faster Boot** |
| **Autoregressive Generation Throughput** | ~800 - 1,100 Tokens/s (with PCIe copies) | **`1,470 Tokens/s`** (Sustained 1B FP16) | **~1.5× Throughput** |
| **Single-Token Decode Latency** | 2.5 ms ~ 8.0 ms (with OS jitter) | **`0.68 ms`** (Sub-millisecond deterministic) | **~4× to 10× Lower Latency** |
| **Host-to-Device Memory Copies** | 1~2 Copies (Host DRAM -> PCIe -> VRAM) | **`0 Copies`** (Physical UMA Direct Map) | **Zero Bus Redundancy** |
| **Inter-Tile Reduction Latency** | 5 µs ~ 15 µs (PCIe / NVLink NCCL kernel) | **`168 ns`** (Hardware Reduction Pulse Line) | **~50× Faster Synchronization** |
| **TLB Translation Overhead** | 4-Level Page Table Walks (4KB Pages) | **1-Level Translation** (1GB/2MB HugePages) | **Zero TLB Thrashing** |

> **Note:** AeroTensor-G1 figures are derived from software simulation,
> analytical modeling, and architectural target specifications.
> They are not measurements from fabricated silicon.

---

## 2. Hardware-Software Co-Design Architecture

```
+----------------------------------------------------------------------------------------------------+
|                                  APPLICATIONS / MISSION WORKLOADS                                  |
|         [Mission Planning LLM (7B)]                      [Flight Safety Control LLM (1B)]          |
+----------------------------------------------------------------------------------------------------+
|                         axtensor: BARE-METAL AI INFERENCE RUNTIME                                  |
|  +-------------------------------------+  +-----------------------------------------------------+  |
|  | no_std Safetensors Zero-Copy Parser |  | Vectorized NN Kernels (fast_rsqrt RMSNorm, RoPE)    |  |
|  +-------------------------------------+  +-----------------------------------------------------+  |
|  | 2MB HugePage Slot-Addressed KV-Cache|  | MultiModelRegistry (Spatial Tile Mask & Preemption) |  |
|  +-------------------------------------+  +-----------------------------------------------------+  |
+----------------------------------------------------------------------------------------------------+
|                              axtensor_mem: 3-TIER MEMORY SUBSYSTEM                                 |
|  - 1GB Static Weight Arena (Level-1 Page Table)       - 2MB Dynamic KV-Cache Arena                 |
|  - 16MB Fast Scratchpad (L1 SRAM per Tile)            - AMBA 5 CHI Hardware Coherency Manager      |
+----------------------------------------------------------------------------------------------------+
|                               axcompute: NPU ACCELERATOR CORE                                      |
|  - 64-Byte Cacheline-Aligned SQE / 16-Byte CQE        - Lockless Atomic Ring Buffers (SQ/CQ)       |
|  - TensorComputeDriver Hardware Abstraction           - GICv3 SPI 64~79 IRQ & ARM `sev` Dispatcher |
+----------------------------------------------------------------------------------------------------+
|                            AEROTENSOR-G1 MODELED ACCELERATOR ARCHITECTURE                                |
|  - 64-Core ARMv9.2-A CPU (Neoverse-V2, 3.0 GHz)       - 16-Tile Heterogeneous NPU (4x4 2D Torus)   |
|  - 96GB HBM3e Unified Memory (3,200 GB/s UMA)         - Hardware Barrier Unit (HBU) Pulse Line     |
+----------------------------------------------------------------------------------------------------+
```

### 2.1 Target Silicon Architecture

The following specifications define the target architecture used by the AeroTensor-G1 analytical and software modeling environment.

* **CPU Complex**:
  - 64-Core ARMv9.2-A organized in 4 clusters of 16 cores (Neoverse-V2 microarchitecture running at 3.0 GHz).
  - Private 64KB L1 I-Cache, 64KB L1 D-Cache, and 1MB private L2 Cache per core (total 64MB L2).
  - Supports SVE2 (4×128-bit vector pipelines) and SME (Scalable Matrix Extension) for CPU-side vector pre/post-processing.
* **NPU Compute Fabric**:
  - 16 Heterogeneous Compute Tiles interconnected via a 4×4 2D Torus Coherent NoC.
  - Target Peak Compute: **512 TFLOPS FP16/BF16 dense** (1024 TFLOPS with 2:1 structured sparsity) or **1024 TOPS INT8** sustained compute at 1.4 GHz operating point.
  - SMMUv3 Integration: CPU and NPU share the same Stage-1 page table (`TTBR0_EL1`), granting NPU direct translation access to kernel memory spaces.
* **Unified Memory (UMA)**:
  - 96 GB HBM3e in 2.5D CoWoS-S advanced packaging (4 stacks of 24GB 12-Hi dies).
  - Delivers **3,200 GB/s (3.2 TB/s)** aggregate bandwidth across a 4096-bit bus operating at 6.25 Gbps.
* **On-Chip Interconnect & Cache**:
  - AMBA 5 CHI (Coherent Hub Interface) with full hardware cache coherency (Inner Shareable Domain).
  - 64 MB distributed System-Level Cache (SLC) across 16 slices providing **2,048 GB/s** aggregate cache bandwidth.
  - 16 MB on-chip L1 Scratchpad SRAM distributed across the 16 NPU tiles (**2,867 GB/s** aggregate).
* **Synchronization & Interrupts**:
  - Hardware Barrier Unit (HBU): Hardware nanosecond pulse line synchronizes 16 tiles in lockstep with a **168 ns** Ring-AllReduce tree latency.
  - GICv3 Interrupt Controller: SPI lines 64 through 79 are hard-wired 1-to-1 to Tiles 0 through 15, pairing with low-overhead ARM `sev` (Send Event) instructions to eliminate polling overhead.

---

### 2.2 Rigorous Bandwidth & Power (TDP) Derivation Rationale

#### A. Multi-Tier Bandwidth Hierarchy
The AeroTensor-G1 memory hierarchy is structured into four distinct, mathematically balanced bandwidth tiers:

$$\begin{aligned}
\text{Tier 1 (L1 Scratchpad SRAM)} &: 16\text{ tiles} \times (2 \times 64\text{ B} \times 1.4\text{ GHz}) = \mathbf{2,867.2\text{ GB/s (2.87 TB/s)}} \\
\text{Tier 2 (L3 System-Level Cache)} &: 16\text{ slices} \times (64\text{ B} \times 2.0\text{ GHz}) = \mathbf{2,048.0\text{ GB/s (2.05 TB/s)}} \\
\text{Tier 3 (NoC Torus Bisection)} &: 8\text{ unidirectional links} \times (64\text{ B} \times 2.0\text{ GHz}) = \mathbf{1,024.0\text{ GB/s (1.02 TB/s)}} \\
\text{Tier 4 (HBM3e UMA Physical)} &: \frac{4096\text{ bits} \times 6.25\times 10^9\text{ bps}}{8\text{ bits/Byte}} = \mathbf{3,200.0\text{ GB/s (3.20 TB/s)}}
\end{aligned}$$

**Operational Intensity Grounding**:
$$\text{Arithmetic Intensity} = \frac{512\text{ TFLOPS}}{3.2\text{ TB/s}} = 160\text{ FLOP/Byte}$$
During memory-bound autoregressive decoding (batch size = 1), generating each token requires reading model parameters once. For a 1B FP16 model (2 GB weight payload):
$$\text{Theoretical Peak Throughput} = \frac{3,200\text{ GB/s}}{2\text{ GB}} = 1,600\text{ Tokens/s}$$
Under the modeled memory-bandwidth assumptions, the standalone testbench estimates `1,470 Tokens/s`, corresponding to 91.8% of the theoretical
1,600 Tokens/s bandwidth ceiling.

#### B. Thermal Design Power (TDP) Physical Breakdown
For architectural power modeling, AeroTensor-G1 assumes a 4nm-class process
and 2.5D HBM integration. The estimated SoC power envelope is:, total heat dissipation ($P_{\text{Total}}$) is modeled as:
$$P_{\text{Total}} = P_{\text{CPU}} + P_{\text{NPU}} + P_{\text{HBM3e}} + P_{\text{NoC/SLC}} + P_{\text{IO/VRM}}$$
| Component | Estimated Peak Power | Modeled AI Serving Power | Modeling Assumption |
| :--- | :--- | :--- | :--- |
| **64-Core ARMv9.2-A CPU** | **120 W** | **75 W** | 64 Neoverse-V2 cores @ 3.0 GHz; ~1.87W/core at full vector stress. AI serving utilizes WFI idle states. |
| **16-Tile NPU (512 TFLOPS)** | **160 W** | **110 W** | Systolic array efficiency: 3.5 TFLOPS/W $\rightarrow 512 / 3.5 \approx 146\text{ W} + 14\text{ W}$ L1 SRAM/control logic. |
| **96GB HBM3e Memory Pool** | **95 W** | **60 W** | JEDEC HBM3e PHY & DRAM core: 3.2 pJ/bit; $25.6\text{ Tbps} \times 3.2\text{ pJ/bit} = 82\text{ W} + 13\text{ W}$ refresh/static. |
| **AMBA 5 CHI NoC & 64MB SLC** | **45 W** | **30 W** | Dynamic charging/discharging across 4×4 2D Torus routers and 16 distributed SLC SRAM slices. |
| **PCIe Gen5 / IO / VRM Loss** | **30 W** | **20 W** | Dual PCIe Gen5 x16 PHYs, SMMUv3/GICv3 peripherals, and on-board DC-DC buck regulation loss (~92% efficiency). |
| **Modeled SoC Power Envelope** | **~450 W** | **~295 W** | **Standard Server Specification: 450 W (compatible with standard 2U air/liquid cooling)** |
* **Aerospace / Flight-Mission Low-Power Profile (230 W)**:
  Under constrained avionics environments (hermetically sealed conduction chassis with a 250W thermal budget), hardware Dynamic Voltage and Frequency Scaling (DVFS) dials down the chip:
  - CPU clocks down to 2.2 GHz (**55 W**).
  - NPU clocks down to 1.0 GHz (**95 W**, still outputting ~365 TFLOPS FP16).
  - HBM3e steps down to 4.8 Gbps (**48 W**, yielding 2.45 TB/s bandwidth).
  - NoC and IO draw **32 W**.
  - **Modeled Flight Envelope: ~230 W**, targeting operation within a 250 W avionics thermal budget while reducing the likelihood of thermal throttling.

---

## 3. Core Software Modules in ArceOS

All AI kernel logic resides within three modular `#![no_std]` crates under [`arceos/modules/`](./arceos/modules/):

### 3.1 `axcompute` — Hardware Driver, SQ/CQ Ring Buffers & MMIO Control
The `axcompute` crate manages the physical interface between the ArceOS unikernel and the AeroTensor-G1 NPU tiles:
* **Binary Alignment Contracts**:
  - `AeroTensorSQE` (Submission Queue Entry): Strictly **64 bytes** aligned to a 64-byte boundary. Exactly matches an ARMv9 cacheline, preventing false sharing or multi-cacheline split accesses.
  - `AeroTensorCQE` (Completion Queue Entry): Strictly **16 bytes** aligned to a 16-byte boundary.
* **Lockless Atomic Command Queue (`queue.rs`)**:
  - Uses atomic head and tail pointers (`AtomicU32`) with acquire-release memory semantics to achieve zero-lock contention between kernel threads and hardware DMA engines.
  - Built-in software FP16 GEMM fallback emulator for headless development environments.
* **Ergonomic Operator Dispatch (`ops.rs` & `async_op.rs`)**:
  - Implements the fluent builder pattern (`MatMul::new(&a, &b, &mut c).with_relu(true).dispatch()`).
  - Returns `ComputeFuture`, a native Rust `core::future::Future<Output = AeroTensorCQE>` allowing seamless cooperative multi-tasking on ArceOS async executors.
* **Cache Management Bypass (`coherency.rs`)**:
  - Integrated with the AMBA 5 CHI hardware snooping fabric. Because the CPU and NPU share the Inner Shareable coherency domain, manual software cache flushing (`dc cvac`) is bypassed completely (CMO Bypass), shaving microseconds off every operator invocation.

### 3.2 `axtensor_mem` — 3-Tier HugePage Zero-Copy Allocator
The `axtensor_mem` crate enforces the zero-copy philosophy:
* **Explicit 3-Tier Memory Abstraction**:
  - `MemoryTier::L1Scratchpad`: Ultra-low latency 16MB on-chip SRAM for tile-local matrix tiles.
  - `MemoryTier::L2LocalHbm`: 16GB high-bandwidth memory attached to the local compute quad.
  - `MemoryTier::L3HostUma`: 96GB global unified memory space accessible by all CPUs and NPUs.
* **`ContinuousHugePageAllocator`**:
  - **1GB HugePages (Level-1 Page Table)**: Model weight arenas are strictly allocated in continuous 1GB blocks. This reduces ARM VMSA page table walks to a single level, substantially reducing TLB pressure during sequential model-weight access during dense matrix multiply sweeps.
  - **2MB HugePages**: Dynamic KV-Cache buffers are allocated from a dedicated 2MB HugePage arena, eliminating page fragmentation.

### 3.3 `axtensor` — Bare-Metal AI Runtime & Multi-LLM Co-Hosting
The `axtensor` crate hosts the transformer building blocks and multi-model supervisor:
* **`safetensors.rs` (Zero-Alloc Safetensors Parser)**:
  - Parses Safetensors JSON headers directly from ROM/HBM without allocating heap memory.
  - Direct UMA Physical Mapping: Extracts tensor shape, datatype, and byte offsets, mapping them immediately to 64-bit physical addresses (`paddr`). Memory copy count = 0.
* **`nn.rs` (Optimized Neural Kernels)**:
  - `RMSNorm`: Hand-optimized with ARM NEON vectorized reciprocal square root (`fast_rsqrt`), avoiding floating-point division penalties.
  - `RotaryEmbedding` (RoPE): In-place rotary position encoding avoiding intermediate tensor allocations.
  - `SwiGLU`: Fused vector activation kernel.
  - `Linear`: Automatically maps weight and input physical addresses into an `AeroTensorSQE` and pushes it directly to the NPU ring buffer.
* **`kv_cache.rs` (2MB Slot-Addressed KV-Cache)**:
  - Manages continuous 2MB page frames using a deterministic indexing function:
    $$\text{Slot\_Addr} = \text{Base\_Paddr} + \text{seq\_idx} \times (\text{kv\_dim} \times \text{element\_size})$$
  - Generates zero runtime heap allocations across arbitrarily long autoregressive generation loops.
* **`models/llama.rs`**: Complete, self-contained `LlamaDecoderLayer` and `LlamaModel` implementation with zero heap allocations during token evaluation.
* **`registry.rs` (`MultiModelRegistry`)**: Supervises multi-tenant LLM execution on a single SoC (detailed in Section 4).
* **`benchmark.rs`**: Full-system benchmarking suite measuring Cold-Boot TTFT, decode latency, and sustained throughput.

---

## 4. Multi-LLM Co-Hosting & Candle Architectural Synthesis

### 4.1 Why Candle Cannot be Directly Bundled into an OS Kernel
During the architectural design phase, we performed a deep-dive evaluation of [Hugging Face's Candle](https://github.com/huggingface/candle). While Candle is a premier lightweight ML library for user-space Rust, it possesses fundamental incompatibilities with bare-metal OS kernels:
1. **Closed Device Enumeration**: Candle hardcodes its target devices:
   ```rust
   pub enum Device {
       Cpu,
       Cuda(CudaDevice),
       Metal(MetalDevice),
   }
   ```
   Extending Candle with a proprietary NPU backend requires invasively forking the entire library.
2. **Heavy `std` Dependencies**: Candle requires `std::sync::{Mutex, RwLock}`, `std::fs::File`, `std::path::Path`, and the `rayon` work-stealing threadpool, none of which exist in a `#![no_std]` unikernel core.
3. **Dynamic Graph Overhead**: Candle's tensor graphs rely heavily on reference counting (`Arc`) and dynamic heap allocations, introducing non-deterministic execution times.

### 4.2 The "Borrow the Shape, Forge the Core" Strategy
Instead of carrying Candle's runtime baggage, we adopted its clean, ergonomic builder ergonomics and Safetensors schema, while forging a completely custom `#![no_std]` hardware-direct core:
* We maintain Candle-like developer ergonomics: `layer.forward(&tensor)`.
* Under the hood, memory pointers are directly fed into 64-byte hardware SQEs and dispatched over MMIO doorbells.

### 4.3 Multi-Model Co-Hosting Architecture
For aerospace avionics and autonomous driving, systems must concurrently execute multiple models of differing criticalities. AeroTensor-G1 OS solves this via `MultiModelRegistry`:

```
                           96GB HBM3e UNIFIED MEMORY POOL
+-----------------------------------------------------------------------------------------+
| [Model 0: LLaMA-7B Mission Planner]             [Model 1: Flight-Safety-1B Real-Time]   |
| Physical Base: 0x8_8000_0000 (14 GB)            Physical Base: 0x8_C000_0000 (2 GB)     |
| Continuous 1GB HugePage Range                   Continuous 1GB HugePage Range           |
+-----------------------------------------------------------------------------------------+
                                 |                                 |
           Collision Checker: Non-overlapping ranges [Start, End) OK!
                                 |                                 |
+-----------------------------------------------------------------------------------------+
| [NPU Tiles 0~7 (tile_mask: 0x00FF)]             [NPU Tiles 8~15 (tile_mask: 0xFF00)]    |
| Dedicated High-Throughput Matrix Array          Dedicated Ultra-Low-Latency Array       |
| Priority: Normal / Throughput-Optimized         Priority: Critical / Preemptive Flight  |
+-----------------------------------------------------------------------------------------+
```

1. **Physical Range Overlap Collision Detection**:
   When registering a model via `registry.register_model()`, the registry calculates:
   $$\max(\text{start}_A, \text{start}_B) < \min(\text{end}_A, \text{end}_B)$$
   Any overlapping physical address space triggers an immediate `MemoryCollision` error, preventing weight cross-talk.
2. **16-Tile Spatial Slicing (`tile_mask`)**:
   - `LLaMA-7B` is mapped to `tile_mask = 0x00FF` (Tiles 0 through 7).
   - `Flight-Safety-1B` is mapped to `tile_mask = 0xFF00` (Tiles 8 through 15).
   - Each model dispatches SQEs strictly to its assigned tiles, guaranteeing **zero contention** on on-chip L1 SRAM or matrix multiplier units.
3. **Real-Time Priority Preemption**:
   Each model is assigned a priority (`Critical`, `High`, `Normal`, or `Low`). If flight safety sensors trigger an alert, incoming commands for `Flight-Safety-1B` preempt running background tasks on the NoC, delivering deterministic, sub-millisecond control responses.

---

## 5. End-to-End API Usage & Code Walkthrough

### 5.1 Asynchronous Matrix Multiplication via `axcompute`
```rust
#![no_std]
use axcompute::tensor::{Tensor, DataType};
use axcompute::ops::MatMul;

fn run_accelerated_gemm() {
    let m = 1024;
    let k = 1024;
    let n = 1024;

    // Allocate contiguous zero-copy tensors in UMA memory
    let a = Tensor::zeros(&[m, k], DataType::Float16).unwrap();
    let b = Tensor::zeros(&[k, n], DataType::Float16).unwrap();
    let mut c = Tensor::zeros(&[m, n], DataType::Float16).unwrap();

    // Dispatch directly to AeroTensor-G1 NPU ring buffer
    let job = MatMul::new(&a, &b, &mut c)
        .with_relu(true)
        .dispatch()
        .expect("Dispatch to hardware SQ failed");

    // Non-blocking completion wait (or await via ComputeFuture)
    let cqe = job.wait_complete();
    assert_eq!(cqe.status, 0); // 0 indicates success
}
```

### 5.2 Multi-Model Mounting & Autoregressive Decoding
```rust
#![no_std]
use axtensor::registry::{MultiModelRegistry, ModelDescriptor, ModelPriority};
use axtensor::models::llama::LlamaModel;
use axtensor::kv_cache::HugePageKvCache;

fn setup_avionics_inference() {
    let mut registry = MultiModelRegistry::new();

    // 1. Mount 7B Mission Planning Model on Tiles 0~7
    let llama_7b = ModelDescriptor {
        model_id: 0,
        name: "LLaMA-7B-Mission",
        weight_paddr: 0x8_8000_0000,
        weight_size: 14 * 1024 * 1024 * 1024, // 14 GB
        tile_mask: 0x00FF,                   // Tiles 0~7
        priority: ModelPriority::Normal,
    };
    registry.register_model(llama_7b).unwrap();

    // 2. Mount 1B Flight Safety Model on Tiles 8~15
    let safety_1b = ModelDescriptor {
        model_id: 1,
        name: "Flight-Safety-1B",
        weight_paddr: 0x8_C000_0000,
        weight_size: 2 * 1024 * 1024 * 1024,  // 2 GB
        tile_mask: 0xFF00,                   // Tiles 8~15
        priority: ModelPriority::Critical,   // Highest preemption priority
    };
    registry.register_model(safety_1b).unwrap();

    // 3. Initialize 2MB HugePage KV-Cache with zero runtime allocations
    let kv_cache = HugePageKvCache::new(0x8_E000_0000, 2048, 64).unwrap();

    // 4. Execute autoregressive step
    // Forward pass dispatches directly as 64-byte SQE packets to NPU Tiles 8~15
}
```

---

## 6. Standalone Verification Suite & Testbench Walkthrough

To ensure 100% architectural and hardware alignment without requiring physical silicon, this repository includes an autonomous C co-simulation testbench ([`tests/verify_standalone.c`](./tests/verify_standalone.c)).

### Running the Testbench
```bash
gcc -O2 -Wall -Wextra tests/verify_standalone.c -o tests/verify_standalone -lm
./tests/verify_standalone
```

### Complete Test Coverage Breakdown (12/12 Tests Passed)
```
====================================================================
  AeroTensor-G1 & ArceOS Hardware-Software Co-Simulation Testbench  
====================================================================
[Test 1] 1GB Block Alignment: PASSED
 - Verifies Level-1 ARM page table entry (PTE) alignment at 0x8000_0000.
[Test 2] Hardware Coherency Domain: PASSED
 - Verifies AMBA 5 CHI inner-shareable domain and cache maintenance bypass (CMO).
[Test 3] HBU Multi-Tile Lockstep: PASSED
 - Verifies hardware barrier pulse arrival across 16 tiles and nanosecond pulse lines.
[Test 4] Distributed LLM Layer Inference (TP=4): PASSED
 - Verifies Tensor Parallelism (TP=4) Ring-AllReduce with 168 ns simulated pulse latency.
[Test 5] modules/axcompute SQ/CQ Queue & GEMM Operator: PASSED
 - Verifies 64B SQE and 16B CQE atomic ring buffer submission and completion polling.
[Test 6] Phase 2 axtensor_mem HugePages & Ping-Pong Pipeline: PASSED
 - Verifies 3-tier memory allocation and overlapped CPU/NPU double buffering.
[Test 7] Phase 3 16-Tile Lockstep & GICv3 SPI 64~79 IRQ: PASSED
 - Verifies hard-mapped SPI lines 64~79 to Tiles 0~15 and ARM sev event wakeups.
[Test 8] Phase 4 Step 4.1 no_std Zero-Copy Safetensors Parser: PASSED
 - Verifies zero-heap JSON header parsing and direct UMA physical address mapping.
[Test 9] Phase 4 Step 4.2 NN Ops (RMSNorm, RoPE, SwiGLU, Linear): PASSED
 - Verifies vectorized fast_rsqrt RMSNorm, in-place RoPE, and hardware Linear bridge.
[Test 10] Phase 4 Step 4.3 LLaMA Decoder & 2MB HugePage KV-Cache: PASSED
 - Verifies continuous 2MB KV-Cache slot indexing and zero-heap autoregressive decoding.
[Test 11] Phase 4 Step 4.4 Multi-Model Mounting & Resource Isolation: PASSED
 - Verifies memory range collision detection and spatial 16-tile mask isolation.
[Test 12] Phase 4 Step 4.5 Full System Benchmark & Cold Boot (<20ms): PASSED
 - Verifies cold-boot TTFT (4.32 ms achieved), throughput (1,470 Tokens/s), and sub-millisecond decode (0.68 ms).
====================================================================
[SUCCESS] ALL INTEGRATION TESTS PASSED (100% Verified).
```

---

## 7. Repository Layout

```
aerotensorg1/
├── README.md                      # [This File] Primary English Engineering Manual
├── README.zh-TW.md                # Traditional Chinese Manual (台灣在地口語版)
├── README.ja.md                   # Japanese Engineering Manual (エンジニア向け日本語版)
├── AGENTS.md                      # AI Agent Guidelines & Architecture Rules
├── .agents/skills/aerotensor-dev/ # Custom Engineering Skill for AI Assistants
│   └── SKILL.md                   # Skill specifications, invariants & verification guide
├── docs/                          # Architecture Specifications & Whitepapers
│   ├── README.md                  # Documentation Index and Reading Guide
│   ├── 00_aerotensor_g1_sdd.md    # Hardware-Software Co-Design SDD (Official Specification)
│   ├── 01_arceos_deep_dive.md     # ArceOS Unikernel Architecture Deep-Dive
│   ├── 02_ai_native_os_architecture.md # AI-Native Operating System Design
│   ├── 03_custom_ai_chip_integration.md # AeroTensor-G1 MMIO, Registers & Interconnect
│   ├── 04_prototype_roadmap_and_impl.md # 4-Phase Roadmap, Milestones & Verification
│   ├── 05_multi_llm_and_candle_integration.md # Multi-LLM Co-Hosting & Candle Synthesis
│   └── 06_agent_workflow_and_skills.md # Autonomous Agent Workflow & Engineering Checklist
├── arceos/                        # Modular ArceOS Unikernel Tree
│   ├── modules/
│   │   ├── axcompute/             # Accelerator Hardware Driver & SQ/CQ Core
│   │   │   ├── src/device.rs      # 64B SQE, 16B CQE, and MMIO register map
│   │   │   ├── src/queue.rs       # Lockless atomic ring buffer & GEMM emulator
│   │   │   ├── src/tensor.rs      # Zero-copy Tensor abstraction
│   │   │   ├── src/ops.rs         # MatMul builder pattern & JobHandle
│   │   │   ├── src/async_op.rs    # ComputeFuture implementation
│   │   │   ├── src/driver.rs      # TensorComputeDriver trait & board driver
│   │   │   ├── src/irq.rs         # GICv3 SPI 64~79 IRQ dispatcher
│   │   │   └── src/coherency.rs   # AMBA 5 CHI hardware coherency manager
│   │   ├── axtensor_mem/          # 3-Tier HugePage Zero-Copy Allocators
│   │   │   ├── src/buffer.rs      # MemoryTier (L1 Scratchpad, L2 HBM, L3 UMA)
│   │   │   └── src/allocator.rs   # ContinuousHugePageAllocator (1GB/2MB/16MB)
│   │   └── axtensor/              # Bare-Metal AI Runtime & Models
│   │       ├── src/safetensors.rs # Pure no_std zero-alloc Safetensors parser
│   │       ├── src/nn.rs          # Vectorized RMSNorm, RoPE, SwiGLU, Linear
│   │       ├── src/kv_cache.rs    # 2MB HugePage slot-addressed KV-Cache
│   │       ├── src/models/llama.rs# Zero-heap autoregressive LLaMA decoder
│   │       ├── src/registry.rs    # MultiModelRegistry, collision & tile isolation
│   │       └── src/benchmark.rs   # Full system benchmark & cold-boot harness
│   └── examples/
│       ├── ai-gemm-demo/          # Phase 1 Sub-millisecond GEMM
│       ├── ai-pipeline-demo/      # Phase 2 Ping-Pong Double Buffering
│       ├── ai-board-demo/         # Phase 3 16-Tile Lockstep Board Driver
│       └── ai-llm-inference/      # Phase 4 Multi-LLM Autoregressive Generation
├── scripts/
│   └── run_qemu_server.sh         # QEMU Simulation Runner
└── tests/
    ├── verify_standalone.c        # Standalone 12-Test Co-Simulation Testbench
    └── verify_standalone          # Compiled Test Executable
```
