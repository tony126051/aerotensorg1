#!/bin/bash
set -e

QEMU_EXEC="qemu-system-aarch64"
KERNEL_IMAGE="target/aarch64-unknown-none-softfloat/release/arceos_kernel"
WEIGHT_IMAGE="target/weights_llama70b_fp16.bin"

$QEMU_EXEC \
    -M virt,gic-version=3,iommu=smmuv3 \
    -cpu max \
    -smp 16 \
    -m 32G \
    -device aerotensor-soc,addr=0x20000000 \
    -drive file=$WEIGHT_IMAGE,if=none,format=raw,id=nvm0 \
    -device virtio-blk-device,drive=nvm0 \
    -kernel $KERNEL_IMAGE \
    -nographic \
    -serial mon:stdio \
    -append "console=ttyAMA0 earlycon=pl011,0x09000000 UMA=1 FEATURES=server_uma"
