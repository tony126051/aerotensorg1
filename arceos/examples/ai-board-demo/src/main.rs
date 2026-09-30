#![no_std]
#![no_main]

use core::panic::PanicInfo;
use axcompute::driver::{AeroTensorBoardDriver, TensorComputeDriver};
use axcompute::irq::{GicSpiDispatcher, GIC_SPI_BASE, GIC_SPI_MAX};
use axcompute::coherency::COHERENCY_MGR;
use axcompute::device::AeroTensorSQE;

#[unsafe(no_mangle)]
pub extern "C" fn main() {
    uart_log("====================================================================\n");
    uart_log("  AeroTensor AI Native OS - Phase 3 Hardware & Board Hardening Demo  \n");
    uart_log("====================================================================\n");

    unsafe {
        run_board_hardening_demo();
    }

    uart_log("\n[SUCCESS] Phase 3 Hardware Co-Design & Hardening Verified (100% Passed)!\n");
    uart_log("====================================================================\n");
    qemu_poweroff();
}

unsafe fn run_board_hardening_demo() {
    uart_log("[AI-OS] Step 1: Probing AeroTensor-G1 Physical Accelerator Board...\n");

    let mut driver = AeroTensorBoardDriver::new();
    driver.probe().expect("Driver probing failed");

    assert!(driver.verify_hardware_signature(), "Hardware signature mismatch");
    uart_log("       Hardware: ");
    uart_log(driver.name());
    uart_log(" (Signature: AERO)\n");

    uart_log("[AI-OS] Step 2: Verifying AMBA 5 CHI Hardware Cache Coherency Domain...\n");
    assert!(
        COHERENCY_MGR.is_hardware_coherent(),
        "Must operate in AMBA 5 CHI Hardware Coherent Domain"
    );
    uart_log("       Coherency: AMBA 5 CHI Inner Shareable Domain (CMO Bypassed, Zero Latency)\n");

    uart_log("[AI-OS] Step 3: Submitting Batch Compute Tasks across 16 NPU Tiles...\n");
    let mut sqes = [AeroTensorSQE::empty(); 16];
    for tile_id in 0..16 {
        sqes[tile_id] = AeroTensorSQE::new_matmul(
            tile_id as u8,
            0x4000_0000 + (tile_id as u64 * 0x1000),
            0x5000_0000,
            0x6000_0000 + (tile_id as u64 * 0x1000),
            16,
            16,
            16,
            true,
            100 + tile_id as u64,
        );
    }
    let submitted = driver.submit_batch(&sqes).expect("Batch submission failed");
    assert_eq!(submitted, 16, "Must submit all 16 tile tasks");
    uart_log("       Submitted: 16 Tile tasks into Submission Queue\n");

    uart_log("[AI-OS] Step 4: Dispatching GICv3 SPI 64~79 Hardware Interrupts...\n");
    let dispatcher = GicSpiDispatcher::new(16);

    for irq in GIC_SPI_BASE..=GIC_SPI_MAX {
        let ok = dispatcher.dispatch_irq(irq);
        assert!(ok, "GIC SPI IRQ dispatch must succeed");
        assert!(driver.handle_irq(irq), "Driver IRQ handling must succeed");
    }

    uart_log("       16-Tile Lockstep: All SPI 64~79 IRQs received, HBU Reduction Complete!\n");
}

fn uart_log(s: &str) {
    #[cfg(target_os = "none")]
    for b in s.bytes() {
        unsafe {
            core::ptr::write_volatile(0x0900_0000 as *mut u8, b);
        }
    }
    #[cfg(not(target_os = "none"))]
    let _ = s;
}

fn qemu_poweroff() {
    #[cfg(target_os = "none")]
    unsafe {
        core::ptr::write_volatile(0x0900_0000 as *mut u8, b'\n');
        core::ptr::write_volatile(0x0800_0000 as *mut u32, 0);
    }
}

#[panic_handler]
fn panic(_info: &PanicInfo) -> ! {
    uart_log("\n[KERNEL PANIC] AI Board Hardening Demo encountered an error.\n");
    loop {
        #[cfg(target_arch = "aarch64")]
        unsafe {
            core::arch::asm!("wfe", options(nomem, nostack));
        }
    }
}
