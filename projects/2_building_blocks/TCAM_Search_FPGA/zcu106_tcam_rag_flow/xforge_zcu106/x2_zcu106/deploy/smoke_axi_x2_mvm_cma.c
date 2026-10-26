#include <errno.h>
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>

#include "libxlnk_cma.h"

#define ACCEL_BASE 0xA0000000UL
#define ACCEL_SIZE 0x1000UL
#define DMA_BASE   0xA0010000UL
#define DMA_SIZE   0x10000UL

#define REG_CTRL     0x00
#define REG_STATUS   0x04
#define REG_CONFIG   0x08
#define REG_EXPECTED 0x0C
#define REG_RX_WORDS 0x10
#define REG_TX_WORDS 0x14
#define REG_CYCLES   0x18
#define REG_ERROR    0x1C
#define REG_MAGIC    0x20
#define REG_VERSION  0x24

#define CTRL_START 0x00000001U
#define CTRL_ABORT 0x00000002U
#define CTRL_CLEAR 0x00000100U

#define DMA_MM2S_DMACR 0x00
#define DMA_MM2S_DMASR 0x04
#define DMA_MM2S_SA    0x18
#define DMA_MM2S_SA_MSB 0x1C
#define DMA_MM2S_LENGTH 0x28
#define DMA_S2MM_DMACR 0x30
#define DMA_S2MM_DMASR 0x34
#define DMA_S2MM_DA    0x48
#define DMA_S2MM_DA_MSB 0x4C
#define DMA_S2MM_LENGTH 0x58

#define DMA_CR_RUNSTOP 0x00000001U
#define DMA_CR_RESET   0x00000004U
#define DMA_SR_IDLE    0x00000002U
#define DMA_SR_ERR     0x00007000U
#define MAGIC_X2MV     0x58324D56U

static volatile uint32_t *map_regs(int fd, unsigned long base, unsigned long size) {
    void *ptr = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, (off_t)base);
    if (ptr == MAP_FAILED) {
        fprintf(stderr, "mmap 0x%lx failed: %s\n", base, strerror(errno));
        exit(1);
    }
    return (volatile uint32_t *)ptr;
}

static inline uint32_t rd(volatile uint32_t *regs, uint32_t off) {
    return regs[off / 4U];
}

static inline void wr(volatile uint32_t *regs, uint32_t off, uint32_t value) {
    regs[off / 4U] = value;
}

static int wait_mask(volatile uint32_t *regs, uint32_t off, uint32_t mask, uint32_t value, int timeout_ms) {
    struct timespec ts = {.tv_sec = 0, .tv_nsec = 1000000L};
    for (int i = 0; i < timeout_ms; ++i) {
        uint32_t v = rd(regs, off);
        if ((v & mask) == value) {
            return 0;
        }
        nanosleep(&ts, NULL);
    }
    return -1;
}

static void reset_dma(volatile uint32_t *dma) {
    wr(dma, DMA_MM2S_DMACR, DMA_CR_RESET);
    wr(dma, DMA_S2MM_DMACR, DMA_CR_RESET);
    if (wait_mask(dma, DMA_MM2S_DMACR, DMA_CR_RESET, 0, 1000) != 0 ||
        wait_mask(dma, DMA_S2MM_DMACR, DMA_CR_RESET, 0, 1000) != 0) {
        fprintf(stderr, "DMA reset did not complete\n");
        exit(1);
    }
    wr(dma, DMA_MM2S_DMASR, 0xFFFFFFFFU);
    wr(dma, DMA_S2MM_DMASR, 0xFFFFFFFFU);
}

static void wait_dma_idle(volatile uint32_t *dma, const char *name, uint32_t status_off) {
    for (int i = 0; i < 5000; ++i) {
        uint32_t status = rd(dma, status_off);
        if (status & DMA_SR_ERR) {
            fprintf(stderr, "%s DMA error status=0x%08x\n", name, status);
            exit(1);
        }
        if (status & DMA_SR_IDLE) {
            return;
        }
        struct timespec ts = {.tv_sec = 0, .tv_nsec = 1000000L};
        nanosleep(&ts, NULL);
    }
    fprintf(stderr, "%s DMA timeout status=0x%08x\n", name, rd(dma, status_off));
    exit(1);
}

int main(void) {
    int memfd = open("/dev/mem", O_RDWR | O_SYNC);
    if (memfd < 0) {
        fprintf(stderr, "open /dev/mem failed: %s\n", strerror(errno));
        return 1;
    }

    volatile uint32_t *accel = map_regs(memfd, ACCEL_BASE, ACCEL_SIZE);
    volatile uint32_t *dma = map_regs(memfd, DMA_BASE, DMA_SIZE);

    uint32_t magic = rd(accel, REG_MAGIC);
    uint32_t version = rd(accel, REG_VERSION);
    printf("magic=0x%08x version=0x%08x\n", magic, version);
    if (magic != MAGIC_X2MV) {
        fprintf(stderr, "Unexpected accelerator magic. Is zcu106_axi_x2_mvm.bit loaded?\n");
        return 1;
    }

    uint32_t *tx = (uint32_t *)cma_alloc(4096, 0);
    uint32_t *rx = (uint32_t *)cma_alloc(4096, 0);
    if ((long)tx == -1 || (long)rx == -1 || tx == NULL || rx == NULL) {
        fprintf(stderr, "cma_alloc failed\n");
        return 1;
    }
    memset(tx, 0, 4096);
    memset(rx, 0, 4096);

    tx[0] = 0x00030201U;
    tx[1] = 0x07060504U;
    tx[2] = 0x00000908U;

    unsigned long tx_phys = cma_get_phy_addr(tx);
    unsigned long rx_phys = cma_get_phy_addr(rx);
    printf("tx_phys=0x%08lx rx_phys=0x%08lx\n", tx_phys, rx_phys);
    if ((tx_phys >> 32) || (rx_phys >> 32)) {
        fprintf(stderr, "AXI DMA is configured for 32-bit addresses; CMA returned >4GB\n");
        return 1;
    }

    cma_flush_cache(tx, (unsigned int)tx_phys, 4096);
    cma_flush_cache(rx, (unsigned int)rx_phys, 4096);

    reset_dma(dma);
    wr(accel, REG_CTRL, CTRL_ABORT | CTRL_CLEAR);
    wr(accel, REG_CONFIG, (2U << 8) | 3U);

    wr(dma, DMA_S2MM_DMACR, DMA_CR_RUNSTOP);
    wr(dma, DMA_MM2S_DMACR, DMA_CR_RUNSTOP);
    wr(dma, DMA_S2MM_DA, (uint32_t)rx_phys);
    wr(dma, DMA_S2MM_DA_MSB, 0);
    wr(dma, DMA_S2MM_LENGTH, 8);

    wr(accel, REG_CTRL, CTRL_START);

    wr(dma, DMA_MM2S_SA, (uint32_t)tx_phys);
    wr(dma, DMA_MM2S_SA_MSB, 0);
    wr(dma, DMA_MM2S_LENGTH, 12);

    wait_dma_idle(dma, "MM2S", DMA_MM2S_DMASR);
    wait_dma_idle(dma, "S2MM", DMA_S2MM_DMASR);
    cma_invalidate_cache(rx, (unsigned int)rx_phys, 4096);

    int32_t y0 = (int32_t)rx[0];
    int32_t y1 = (int32_t)rx[1];
    printf("result=[%d,%d]\n", y0, y1);
    printf("status=0x%08x expected=%u rx_words=%u tx_words=%u cycles=%u error=%u\n",
           rd(accel, REG_STATUS), rd(accel, REG_EXPECTED), rd(accel, REG_RX_WORDS),
           rd(accel, REG_TX_WORDS), rd(accel, REG_CYCLES), rd(accel, REG_ERROR));

    cma_free(tx);
    cma_free(rx);
    munmap((void *)accel, ACCEL_SIZE);
    munmap((void *)dma, DMA_SIZE);
    close(memfd);

    if (y0 != 40 || y1 != 46) {
        fprintf(stderr, "FAIL: expected [40,46]\n");
        return 1;
    }
    puts("PASS: zcu106_axi_x2_mvm CMA/MMIO/DMA smoke");
    return 0;
}
