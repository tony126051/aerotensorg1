# Agent Guidelines for AeroTensor-G1 Project

Welcome to the **AeroTensor-G1 & ArceOS AI-Native Operating System** project.
When working as an autonomous coding agent in this repository, you must observe the following technical standards and practices:

## 1. Core Principles
* **Bare-Metal & Unikernel First**: The codebase targets bare-metal execution on 64-Core ARMv9.2-A with 16-Tile AeroTensor-G1 NPU. Maintain strict `#![no_std]` in all kernel crates under `arceos/modules/`.
* **Zero-Copy Philosophy**: Never copy model weights or tensor payloads between Host and Device. Leverage the 96GB HBM3e UMA physical memory map and 1GB/2MB continuous hugepages.
* **Exact Binary Sizing**: Hardware structures (like `AeroTensorSQE` at 64 bytes and `AeroTensorCQE` at 16 bytes) must strictly match hardware cacheline and bus specifications.

## 2. Testing & Verification
* Run the standalone testbench before declaring any task complete:
  ```bash
  gcc -O2 -Wall -Wextra tests/verify_standalone.c -o tests/verify_standalone -lm
  ./tests/verify_standalone
  ```
* Ensure all 12 tests pass without warnings or errors.

## 3. Reference Documentation
* [Architecture SDD](file:///home/tony/repo/projects/aerotensorg1/docs/00_aerotensor_g1_sdd.md)
* [ArceOS Deep Dive](file:///home/tony/repo/projects/aerotensorg1/docs/01_arceos_deep_dive.md)
* [AI Native OS Architecture](file:///home/tony/repo/projects/aerotensorg1/docs/02_ai_native_os_architecture.md)
* [Custom Chip Integration](file:///home/tony/repo/projects/aerotensorg1/docs/03_custom_ai_chip_integration.md)
* [Roadmap & Verification](file:///home/tony/repo/projects/aerotensorg1/docs/04_prototype_roadmap_and_impl.md)
* [Multi-LLM & Candle Integration](file:///home/tony/repo/projects/aerotensorg1/docs/05_multi_llm_and_candle_integration.md)
* [Agent Workflow & Skills](file:///home/tony/repo/projects/aerotensorg1/docs/06_agent_workflow_and_skills.md)
