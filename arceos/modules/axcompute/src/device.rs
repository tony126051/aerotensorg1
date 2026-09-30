#![no_std]

use core::ptr::{read_volatile, write_volatile};

/// MMIO Base Address for AeroTensor-G1 Hardware Barrier Unit (HBU)
pub const HBU_REG_BASE: usize = 0xFFFF_0000_2000_0000;
/// MMIO Base Address for 16-Tile NPU Matrix Array
pub const NPU_TILE_BASE: usize = 0xFFFF_0000_2010_0000;
/// Stride between each NPU Tile (64KB)
pub const TILE_STRIDE: usize = 0x0001_0000;

// Opcodes supported by AeroTensor Accelerator
pub const OP_NOP: u16 = 0;
pub const OP_MATMUL: u16 = 1;
pub const OP_REDUCE: u16 = 2;
pub const OP_MEMCPY: u16 = 3;

// Execution Flags
pub const FLAG_NONE: u16 = 0;
pub const FLAG_RELU: u16 = 1 << 0;
pub const FLAG_IRQ: u16 = 1 << 1;
pub const FLAG_SYNC: u16 = 1 << 2;
pub const FLAG_SW_EMULATE: u16 = 1 << 3;

// CQE Status Values
pub const CQE_STATUS_SUCCESS: u32 = 0;
pub const CQE_STATUS_PENDING: u32 = 0xFFFF_FFFF;
pub const CQE_STATUS_ERR_SHAPE: u32 = 1;
pub const CQE_STATUS_ERR_ALIGN: u32 = 2;

/// Submission Queue Entry (SQE) - 64 bytes aligned (1 cacheline)
#[repr(C, align(64))]
#[derive(Clone, Copy, Debug)]
pub struct AeroTensorSQE {
    pub opcode: u16,        // OP_MATMUL, OP_REDUCE, etc.
    pub flags: u16,         // FLAG_RELU, FLAG_IRQ, etc.
    pub tile_id: u8,        // Target NPU Tile (0..15)
    pub reserved0: [u8; 3],
    pub user_data: u64,     // Request tag/token for completion mapping
    pub src_a_pa: u64,      // Physical Address Matrix A
    pub src_b_pa: u64,      // Physical Address Matrix B
    pub dst_c_pa: u64,      // Physical Address Result C
    pub m: u32,             // Rows of A and C
    pub k: u32,             // Cols of A / Rows of B
    pub n: u32,             // Cols of B and C
    pub stride_a: u32,      // Row stride for A (in elements)
    pub stride_b: u32,      // Row stride for B (in elements)
    pub stride_c: u32,      // Row stride for C (in elements)
}

impl AeroTensorSQE {
    pub const fn empty() -> Self {
        Self {
            opcode: OP_NOP,
            flags: 0,
            tile_id: 0,
            reserved0: [0; 3],
            user_data: 0,
            src_a_pa: 0,
            src_b_pa: 0,
            dst_c_pa: 0,
            m: 0,
            k: 0,
            n: 0,
            stride_a: 0,
            stride_b: 0,
            stride_c: 0,
        }
    }

    pub fn new_matmul(
        tile_id: u8,
        src_a_pa: u64,
        src_b_pa: u64,
        dst_c_pa: u64,
        m: u32,
        k: u32,
        n: u32,
        relu: bool,
        user_data: u64,
    ) -> Self {
        let mut flags = FLAG_NONE;
        if relu {
            flags |= FLAG_RELU;
        }
        Self {
            opcode: OP_MATMUL,
            flags,
            tile_id,
            reserved0: [0; 3],
            user_data,
            src_a_pa,
            src_b_pa,
            dst_c_pa,
            m,
            k,
            n,
            stride_a: k,
            stride_b: n,
            stride_c: n,
        }
    }
}

/// Completion Queue Entry (CQE) - 16 bytes aligned
#[repr(C, align(16))]
#[derive(Clone, Copy, Debug)]
pub struct AeroTensorCQE {
    pub user_data: u64,     // Echoes the SQE user_data tag
    pub status: u32,        // 0 = Success, 0xFFFF_FFFF = Pending
    pub cycle_count: u32,   // Hardware execution cycle count
}

impl AeroTensorCQE {
    pub const fn pending(user_data: u64) -> Self {
        Self {
            user_data,
            status: CQE_STATUS_PENDING,
            cycle_count: 0,
        }
    }

    pub const fn success(user_data: u64, cycles: u32) -> Self {
        Self {
            user_data,
            status: CQE_STATUS_SUCCESS,
            cycle_count: cycles,
        }
    }

    pub fn is_completed(&self) -> bool {
        self.status != CQE_STATUS_PENDING
    }
}

/// AeroTensor Device Controller Interface
pub struct AeroTensorDevice;

impl AeroTensorDevice {
    /// Ring the Doorbell for a specific NPU Tile to notify hardware of new work
    pub unsafe fn ring_doorbell(tile_id: u8, sq_tail: u32) {
        #[cfg(target_os = "none")]
        {
            let tile_doorbell = (NPU_TILE_BASE + (tile_id as usize * TILE_STRIDE) + 0x0004) as *mut u32;
            write_volatile(tile_doorbell, sq_tail);
        }
        #[cfg(not(target_os = "none"))]
        {
            let _ = (tile_id, sq_tail);
        }
    }

    /// Read Tile Status
    pub unsafe fn read_status(tile_id: u8) -> u32 {
        #[cfg(target_os = "none")]
        {
            let tile_status = (NPU_TILE_BASE + (tile_id as usize * TILE_STRIDE) + 0x0008) as *const u32;
            read_volatile(tile_status)
        }
        #[cfg(not(target_os = "none"))]
        {
            let _ = tile_id;
            0
        }
    }
}
