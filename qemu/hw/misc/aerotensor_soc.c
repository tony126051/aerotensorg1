#include "qemu/osdep.h"
#include "hw/sysbus.h"
#include "hw/irq.h"
#include "qemu/module.h"
#include "sysemu/dma.h"
#include "exec/address-spaces.h"

#define TYPE_AEROTENSOR_SOC "aerotensor-soc"
#define AEROTENSOR_SOC(obj) OBJECT_CHECK(AeroTensorSoCState, (obj), TYPE_AEROTENSOR_SOC)

#define REG_HBU_GROUP_CFG       0x0000
#define REG_HBU_ARRIVE_PULSE    0x0004
#define REG_HBU_STATUS_MASK     0x0008
#define REG_HBU_REDUCE_CMD      0x0010
#define REG_HBU_REDUCE_SRC_PA   0x0018
#define REG_HBU_REDUCE_LEN      0x0020

typedef struct {
    SysBusDevice parent_obj;
    MemoryRegion iomem;
    qemu_irq irq[16]; // 映射至 GIC SPI 64 ~ 79

    uint32_t group_cfg;
    uint32_t status_mask;
    uint32_t reduce_cmd;
    uint64_t reduce_src_pa;
    uint64_t reduce_len;
} AeroTensorSoCState;

static uint64_t aerotensor_read(void *opaque, hwaddr offset, unsigned size) {
    AeroTensorSoCState *s = (AeroTensorSoCState *)opaque;
    switch (offset) {
        case REG_HBU_GROUP_CFG:     return s->group_cfg;
        case REG_HBU_STATUS_MASK:   return s->status_mask;
        case REG_HBU_REDUCE_CMD:    return s->reduce_cmd;
        case REG_HBU_REDUCE_SRC_PA: return s->reduce_src_pa;
        case REG_HBU_REDUCE_LEN:    return s->reduce_len;
        default:                    return 0;
    }
}

static void aerotensor_execute_hardware_reduction(AeroTensorSoCState *s) {
    // 模擬 UMA 硬體加法樹：直接存取系統共享實體記憶體空間
    if (s->reduce_src_pa >= 0x40000000 && s->reduce_len > 0) {
        // 確保實體位址落在 UMA 記憶體區間內
        uint16_t *buf = g_malloc(s->reduce_len);
        dma_memory_read(&address_space_memory, s->reduce_src_pa, buf, s->reduce_len, MEMTXATTRS_UNSPECIFIED);

        // 模擬 FP16 加總歸納運算
        for (size_t i = 0; i < s->reduce_len / 2; i++) {
            buf[i] = buf[i] * (s->group_cfg & 0xFF); // 模擬累加權重
        }

        dma_memory_write(&address_space_memory, s->reduce_src_pa, buf, s->reduce_len, MEMTXATTRS_UNSPECIFIED);
        g_free(buf);
    }

    s->reduce_cmd |= (1 << 30); // 標記硬體 Reduction 完成 (Bit 30)
    s->status_mask = 0;         // 清空屏障遮罩
}

static void aerotensor_write(void *opaque, hwaddr offset, uint64_t value, unsigned size) {
    AeroTensorSoCState *s = (AeroTensorSoCState *)opaque;
    switch (offset) {
        case REG_HBU_GROUP_CFG:
            s->group_cfg = (uint32_t)value;
            break;
        case REG_HBU_ARRIVE_PULSE: {
            uint8_t tile_id = (uint8_t)value;
            s->status_mask |= (1 << tile_id);
            uint8_t total_nodes = s->group_cfg & 0xFF;
            uint32_t complete_mask = (1 << total_nodes) - 1;

            // 觸發硬體 SPI 中斷通知 CPU 核心
            qemu_set_irq(s->irq[tile_id], 1);
            qemu_set_irq(s->irq[tile_id], 0);

            if (s->status_mask == complete_mask) {
                aerotensor_execute_hardware_reduction(s);
            }
            break;
        }
        case REG_HBU_REDUCE_CMD:
            s->reduce_cmd = (uint32_t)value;
            if (value & (1 << 31)) { // 手動觸發 Reduction
                aerotensor_execute_hardware_reduction(s);
            }
            break;
        case REG_HBU_REDUCE_SRC_PA:
            s->reduce_src_pa = value;
            break;
        case REG_HBU_REDUCE_LEN:
            s->reduce_len = value;
            break;
    }
}

static const MemoryRegionOps aerotensor_ops = {
    .read = aerotensor_read,
    .write = aerotensor_write,
    .endianness = DEVICE_LITTLE_ENDIAN,
    .valid = { .min_access_size = 4, .max_access_size = 8 },
};

static void aerotensor_soc_init(Object *obj) {
    AeroTensorSoCState *s = AEROTENSOR_SOC(obj);
    SysBusDevice *dev = SYS_BUS_DEVICE(obj);

    memory_region_init_io(&s->iomem, obj, &aerotensor_ops, s, "aerotensor-soc", 0x200000);
    sysbus_init_mmio(dev, &s->iomem);

    for (int i = 0; i < 16; i++) {
        sysbus_init_irq(dev, &s->irq[i]);
    }
}

static const TypeInfo aerotensor_soc_info = {
    .name          = TYPE_AEROTENSOR_SOC,
    .parent        = TYPE_SYS_BUS_DEVICE,
    .instance_size = sizeof(AeroTensorSoCState),
    .instance_init = aerotensor_soc_init,
};

static void aerotensor_soc_register_types(void) {
    type_register_static(&aerotensor_soc_info);
}

type_init(aerotensor_soc_register_types)
