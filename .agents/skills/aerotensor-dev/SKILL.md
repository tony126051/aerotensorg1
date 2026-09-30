---
name: aerotensor-dev
description: >-
  Expert engineering skill for AeroTensor-G1 AI-Native OS development.
  Covers hardware-software co-design, ArceOS modular unikernel, no_std Safetensors,
  ring buffers (SQE/CQE), 16-Tile NPU partitioning, hugepage memory, and verification testbench.
---

# AeroTensor-G1 AI-Native OS Development Skill

This skill guides agents and developers when implementing, optimizing, or debugging modules within the AeroTensor-G1 and ArceOS AI-Native Operating System repository.

---

## 1. Architectural Guardrails

Whenever writing or refactoring code in `arceos/modules/axcompute`, `axtensor_mem`, or `axtensor`:
1. **Strict `no_std` Compliance**:
   - Never import `std` in kernel modules.
   - Use `core` and `alloc` only when essential.
   - Hot paths (e.g. forward pass, SQE generation, KV-cache lookup) must have **zero heap allocation**.
2. **Hardware Sizing Invariants**:
   - `AeroTensorSQE`: Exactly **64 bytes** (matches a 64-byte ARM cacheline). Must align to 64 bytes.
   - `AeroTensorCQE`: Exactly **16 bytes**.
   - Tile Base Address: `0x8000_0000 + tile_id * 0x0010_0000`.
   - HBU Registers: `0x8100_0000`.
   - Interrupts: GICv3 SPI 64~79 correspond to Tiles 0~15.
3. **Memory Tiering & Alignment**:
   - L1 Scratchpad: 16 MB.
   - L2 Local HBM: 16 GB per NPU cluster.
   - L3 Host UMA: 96 GB (physically contiguous).
   - Weights Arena: Must be aligned to 1 GB HugePages (Level 1 page translation).
   - KV-Cache: Must use 2 MB HugePage slots.

---

## 2. Multi-Model Co-Hosting Protocol

When registering or dispatching multiple LLMs via `MultiModelRegistry`:
- **Collision Detection**: Run `[start, end)` overlap checks on UMA physical addresses.
- **Hardware Isolation**: Assign distinct `tile_mask` bitmasks (e.g., `0x00FF` for throughput-oriented 7B models, `0xFF00` for latency-critical 1B flight models).
- **Priority Handling**: Support `Critical`, `High`, `Normal`, and `Low` priority tags for hard real-time preemption.

---

## 3. Verification Protocol

Always verify modifications using the standalone C testbench:
```bash
gcc -O2 -Wall -Wextra tests/verify_standalone.c -o tests/verify_standalone -lm
./tests/verify_standalone
```
All 12 test assertions must pass with exit code 0.
Ensure Cold-Boot TTFT is under 20 ms (benchmark target: ~4.32 ms).
