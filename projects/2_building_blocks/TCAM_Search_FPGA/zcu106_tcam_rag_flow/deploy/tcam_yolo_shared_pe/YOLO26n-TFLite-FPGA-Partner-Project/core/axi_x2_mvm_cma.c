#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <stdarg.h>
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
#define REG_DBG_STREAM0 0x2C
#define REG_DBG_STREAM1 0x30
#define REG_DBG_STREAM2 0x34
#define REG_DBG_STREAM3 0x38
#define REG_DBG_WEIGHT0 0x3C
#define REG_DBG_WEIGHT1 0x40
#define REG_DBG_VECTOR0 0x44
#define REG_DBG_VECTOR1 0x48
#define REG_DBG_RESULT0 0x4C
#define REG_DBG_RESULT1 0x50
#define REG_DBG_RESULT2 0x54
#define REG_DBG_RESULT3 0x58
#define REG_DBG_AXIS0   0x5C
#define REG_DBG_AXIS1   0x60
#define REG_DBG_AXIS2   0x64
#define REG_DBG_AXIS3   0x68
#define REG_DBG_AXIS_META 0x6C
#define REG_WR_ADDR_LO  0x70
#define REG_WR_ADDR_HI  0x74
#define REG_WR_COUNT    0x78
#define REG_WR_STATUS   0x7C
#define REG_RD_ADDR_LO  0x80
#define REG_RD_ADDR_HI  0x84
#define REG_RD_COUNT    0x88
#define REG_RD_STATUS   0x8C

#define CTRL_START 0x00000001U
#define CTRL_ABORT 0x00000002U
#define CTRL_CLEAR 0x00000100U

#define DMA_MM2S_DMACR  0x00
#define DMA_MM2S_DMASR  0x04
#define DMA_MM2S_SA     0x18
#define DMA_MM2S_SA_MSB 0x1C
#define DMA_MM2S_LENGTH 0x28
#define DMA_S2MM_DMACR  0x30
#define DMA_S2MM_DMASR  0x34
#define DMA_S2MM_DA     0x48
#define DMA_S2MM_DA_MSB 0x4C
#define DMA_S2MM_LENGTH 0x58

#define DMA_CR_RUNSTOP 0x00000001U
#define DMA_CR_RESET   0x00000004U
#define DMA_SR_HALTED  0x00000001U
#define DMA_SR_IDLE    0x00000002U
#define DMA_SR_ERR     0x00004070U
#define MAGIC_X2MV     0x58324D56U
#define AXI_READER_VERSION 0x00020011U
#define STRIDED_AXI_READER_VERSION 0x00020013U

#define WR_STATUS_IDLE  0x00000100U
#define WR_STATUS_BUSY  0x00000200U
#define WR_STATUS_ERR   0x00000400U
#define WR_STATUS_BRESP_SHIFT 11U
#define WR_STATUS_BRESP_MASK  0x00001800U

#define MAX_ROWS 32U
#define MAX_COLS 32U
#define MAX_BATCH 32U
#define DMA_BUF_BYTES 16384U
#define AXI_READER_WORD_STRIDE 4U
#define DMA_ADDR_LIMIT 0x80000000UL
#define LOW32_ALLOC_ATTEMPTS 32768U
#define RESERVED_DMA_BASE_DEFAULT 0x74000000UL
/*
 * Legacy fallback for bring-up bitstreams whose MM2S path corrupted longer
 * transfers. Version 0x00020013 bypasses MM2S with the PL AXI input reader;
 * older versions can still force two-word staging with the environment value.
 */
#define DMA_TX_CHUNK_WORDS_LEGACY 2U
#define DMA_RX_CHUNK_WORDS (DMA_BUF_BYTES / sizeof(uint32_t))
#define DMA_FAST_POLL_SPINS 10000U
#define DMA_POLL_SLEEP_NS 10000L

static int g_memfd = -1;
static int g_uiofd = -1;
static volatile uint32_t *g_accel = NULL;
static volatile uint32_t *g_dma = NULL;
static uint32_t *g_tx = NULL;
static uint32_t *g_rx = NULL;
static void *g_tx_pin = NULL;
static void *g_rx_pin = NULL;
static unsigned long g_tx_phys = 0;
static unsigned long g_rx_phys = 0;
static int g_use_cma = 0;
static int g_use_reserved = 0;
static int g_use_axi_reader = 0;
static int g_use_strided_axi_reader = 0;
static char g_last_error[512] = {0};

void axi_x2_close(void);

static int set_error(const char *fmt, ...) {
    va_list args;
    va_start(args, fmt);
    vsnprintf(g_last_error, sizeof(g_last_error), fmt, args);
    va_end(args);
    return -1;
}

static void clear_error(void) {
    g_last_error[0] = '\0';
}

const char *axi_x2_last_error(void) {
    return g_last_error[0] ? g_last_error : "no error";
}

static volatile uint32_t *map_regs(int fd, unsigned long base, unsigned long size) {
    void *ptr = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, (off_t)base);
    if (ptr == MAP_FAILED) {
        set_error("mmap 0x%lx failed: %s", base, strerror(errno));
        return NULL;
    }
    return (volatile uint32_t *)ptr;
}

static inline uint32_t rd(volatile uint32_t *regs, uint32_t off) {
    return regs[off / 4U];
}

static inline void wr(volatile uint32_t *regs, uint32_t off, uint32_t value) {
    regs[off / 4U] = value;
}

static int virt_to_phys(void *ptr, unsigned long *phys_out) {
    long page_size = sysconf(_SC_PAGESIZE);
    if (page_size <= 0) {
        return set_error("sysconf(_SC_PAGESIZE) failed");
    }

    uint64_t virt = (uint64_t)(uintptr_t)ptr;
    uint64_t page_index = virt / (uint64_t)page_size;
    uint64_t page_offset = virt % (uint64_t)page_size;
    uint64_t entry = 0;

    int fd = open("/proc/self/pagemap", O_RDONLY);
    if (fd < 0) {
        return set_error("open /proc/self/pagemap failed: %s", strerror(errno));
    }
    off_t off = (off_t)(page_index * sizeof(entry));
    ssize_t n = pread(fd, &entry, sizeof(entry), off);
    close(fd);
    if (n != (ssize_t)sizeof(entry)) {
        return set_error("pread /proc/self/pagemap failed: %s", strerror(errno));
    }
    if ((entry & (1ULL << 63)) == 0) {
        return set_error("pagemap entry is not present for VA 0x%" PRIx64, virt);
    }

    uint64_t pfn = entry & ((1ULL << 55) - 1ULL);
    if (pfn == 0) {
        return set_error("pagemap PFN is zero; run as root/CAP_SYS_ADMIN");
    }
    *phys_out = (unsigned long)((pfn * (uint64_t)page_size) + page_offset);
    return 0;
}

static void release_pinned_page(void *ptr) {
    if (!ptr) {
        return;
    }
    munlock(ptr, DMA_BUF_BYTES);
    free(ptr);
}

static void release_pinned_pages(void **pages, uint32_t count) {
    if (!pages) {
        return;
    }
    for (uint32_t i = 0; i < count; ++i) {
        release_pinned_page(pages[i]);
    }
    free(pages);
}

static uint32_t *alloc_dma_page(unsigned long *phys_out, void **pinned_out) {
    long page_size = sysconf(_SC_PAGESIZE);
    if (page_size < (long)DMA_BUF_BYTES) {
        set_error("system page size %ld is smaller than DMA buffer %u", page_size, DMA_BUF_BYTES);
        return NULL;
    }

    void **held_pages = (void **)calloc(LOW32_ALLOC_ATTEMPTS, sizeof(void *));
    if (!held_pages) {
        set_error("calloc held DMA pages failed: %s", strerror(errno));
        return NULL;
    }

    unsigned long last_phys = 0;
    uint32_t held_count = 0;
    for (uint32_t attempt = 0; attempt < LOW32_ALLOC_ATTEMPTS; ++attempt) {
        void *ptr = NULL;
        int rc = posix_memalign(&ptr, (size_t)page_size, DMA_BUF_BYTES);
        if (rc != 0 || ptr == NULL) {
            set_error("posix_memalign failed: %s", strerror(rc));
            release_pinned_pages(held_pages, held_count);
            return NULL;
        }
        memset(ptr, 0, DMA_BUF_BYTES);
        if (mlock(ptr, DMA_BUF_BYTES) != 0) {
            set_error("mlock failed: %s", strerror(errno));
            free(ptr);
            release_pinned_pages(held_pages, held_count);
            return NULL;
        }
        if (virt_to_phys(ptr, phys_out) != 0) {
            release_pinned_page(ptr);
            release_pinned_pages(held_pages, held_count);
            return NULL;
        }

        last_phys = *phys_out;
        if (*phys_out >= DMA_ADDR_LIMIT) {
            held_pages[held_count++] = ptr;
            continue;
        }

        cma_flush_cache(ptr, (unsigned int)*phys_out, DMA_BUF_BYTES);

        void *io = mmap(NULL, DMA_BUF_BYTES, PROT_READ | PROT_WRITE, MAP_SHARED, g_memfd, (off_t)*phys_out);
        if (io == MAP_FAILED) {
            set_error("mmap /dev/mem DMA page phys=0x%lx failed: %s", *phys_out, strerror(errno));
            release_pinned_page(ptr);
            release_pinned_pages(held_pages, held_count);
            return NULL;
        }

        *pinned_out = ptr;
        release_pinned_pages(held_pages, held_count);
        return (uint32_t *)io;
    }

    release_pinned_pages(held_pages, held_count);
    set_error(
        "could not allocate a DMA page below 0x%lx after %u attempts; last phys=0x%lx",
        DMA_ADDR_LIMIT,
        LOW32_ALLOC_ATTEMPTS,
        last_phys
    );
    return NULL;
}

static uint32_t *map_reserved_dma_page(unsigned long phys) {
    if ((phys & (DMA_BUF_BYTES - 1U)) != 0) {
        set_error("reserved DMA phys address 0x%lx is not %u-byte aligned", phys, DMA_BUF_BYTES);
        return NULL;
    }
    if (phys >= DMA_ADDR_LIMIT || (phys + DMA_BUF_BYTES) > DMA_ADDR_LIMIT) {
        set_error("reserved DMA phys address 0x%lx is outside the low DDR DMA aperture", phys);
        return NULL;
    }

    void *io = mmap(NULL, DMA_BUF_BYTES, PROT_READ | PROT_WRITE, MAP_SHARED, g_memfd, (off_t)phys);
    if (io == MAP_FAILED) {
        set_error("mmap /dev/mem reserved DMA page phys=0x%lx failed: %s", phys, strerror(errno));
        return NULL;
    }
    return (uint32_t *)io;
}

static void flush_tx_buffer(void) {
    __sync_synchronize();
    if (g_use_cma) {
        cma_flush_cache(g_tx, (unsigned int)g_tx_phys, DMA_BUF_BYTES);
        cma_flush_cache(g_rx, (unsigned int)g_rx_phys, DMA_BUF_BYTES);
    } else if (g_use_reserved) {
        /*
         * msync() on /dev/mem mappings is not enough cache maintenance for
         * PS-side DMA. Clean the reserved physical range through xlnk before
         * MM2S reads it, and clean RX before S2MM writes into it.
         */
        cma_flush_cache(g_tx, (unsigned int)g_tx_phys, DMA_BUF_BYTES);
        cma_flush_cache(g_rx, (unsigned int)g_rx_phys, DMA_BUF_BYTES);
        (void)msync(g_tx, DMA_BUF_BYTES, MS_SYNC);
        (void)msync(g_rx, DMA_BUF_BYTES, MS_SYNC);
    } else {
        (void)msync(g_tx, DMA_BUF_BYTES, MS_SYNC);
        (void)msync(g_rx, DMA_BUF_BYTES, MS_SYNC);
    }
    __sync_synchronize();
}

static void invalidate_rx_buffer(void) {
    __sync_synchronize();
    if (g_use_cma) {
        cma_invalidate_cache(g_rx, (unsigned int)g_rx_phys, DMA_BUF_BYTES);
    } else if (g_use_reserved) {
        cma_invalidate_cache(g_rx, (unsigned int)g_rx_phys, DMA_BUF_BYTES);
        (void)msync(g_rx, DMA_BUF_BYTES, MS_SYNC | MS_INVALIDATE);
    } else {
        (void)msync(g_rx, DMA_BUF_BYTES, MS_SYNC | MS_INVALIDATE);
    }
    __sync_synchronize();
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

static int wait_dma_idle(const char *name, uint32_t status_off);

static int wait_dma_halted(const char *name, uint32_t status_off) {
    struct timespec ts = {.tv_sec = 0, .tv_nsec = 1000000L};
    for (int i = 0; i < 1000; ++i) {
        uint32_t status = rd(g_dma, status_off);
        if (status & DMA_SR_ERR) {
            return set_error("%s DMA error while halting status=0x%08x", name, status);
        }
        if (status & DMA_SR_HALTED) {
            return 0;
        }
        nanosleep(&ts, NULL);
    }
    return set_error("%s DMA halt timeout status=0x%08x", name, rd(g_dma, status_off));
}

static uint64_t monotonic_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ((uint64_t)ts.tv_sec * 1000000000ULL) + (uint64_t)ts.tv_nsec;
}

static int reset_dma(void) {
    wr(g_dma, DMA_MM2S_DMACR, DMA_CR_RESET);
    if (wait_mask(g_dma, DMA_MM2S_DMACR, DMA_CR_RESET, 0, 1000) != 0) {
        return set_error("MM2S DMA reset did not complete");
    }
    wr(g_dma, DMA_MM2S_DMASR, 0xFFFFFFFFU);
    return 0;
}

static int reset_mm2s(void) {
    wr(g_dma, DMA_MM2S_DMACR, DMA_CR_RESET);
    if (wait_mask(g_dma, DMA_MM2S_DMACR, DMA_CR_RESET, 0, 1000) != 0) {
        return set_error("MM2S DMA reset did not complete");
    }
    wr(g_dma, DMA_MM2S_DMASR, 0xFFFFFFFFU);
    return 0;
}

static inline uint32_t min_u32(uint32_t a, uint32_t b) {
    return a < b ? a : b;
}

static uint32_t configured_tx_chunk_words(uint32_t total_words) {
    const char *value = getenv("YOLO_PL_TX_CHUNK_WORDS");
    if (!value || !value[0] || strcmp(value, "0") == 0 || strcmp(value, "full") == 0 || strcmp(value, "burst") == 0) {
        return total_words;
    }
    if (strcmp(value, "legacy") == 0 || strcmp(value, "safe") == 0) {
        return min_u32(DMA_TX_CHUNK_WORDS_LEGACY, total_words);
    }

    char *end = NULL;
    errno = 0;
    unsigned long parsed = strtoul(value, &end, 0);
    if (errno != 0 || end == value || (end && *end != '\0') || parsed == 0UL) {
        return total_words;
    }
    if (parsed > (unsigned long)total_words) {
        return total_words;
    }
    return (uint32_t)parsed;
}

static int trace_chunks_enabled(void) {
    const char *value = getenv("YOLO_PL_TRACE_CHUNKS");
    return value && value[0] && strcmp(value, "0") != 0;
}

static int start_mm2s_chunk_once(uint32_t word_offset, uint32_t words, int use_source_offset) {
    unsigned long phys = g_tx_phys;
    if (use_source_offset) {
        phys += ((unsigned long)word_offset * sizeof(uint32_t));
    }
    uint32_t status = rd(g_dma, DMA_MM2S_DMASR);
    if (status & DMA_SR_ERR) {
        return set_error("MM2S DMA error before chunk status=0x%08x", status);
    }
    if (status & DMA_SR_HALTED) {
        wr(g_dma, DMA_MM2S_DMASR, 0xFFFFFFFFU);
        wr(g_dma, DMA_MM2S_DMACR, DMA_CR_RUNSTOP);
    }
    if ((rd(g_dma, DMA_MM2S_DMACR) & DMA_CR_RUNSTOP) == 0U) {
        wr(g_dma, DMA_MM2S_DMACR, DMA_CR_RUNSTOP);
    }
    wr(g_dma, DMA_MM2S_SA, (uint32_t)phys);
    wr(g_dma, DMA_MM2S_SA_MSB, 0);
    if (trace_chunks_enabled()) {
        fprintf(stderr, "axi_x2 mm2s chunk staged_word_offset=%u words=%u phys=0x%lx\n", word_offset, words, phys);
    }
    wr(g_dma, DMA_MM2S_LENGTH, words * 4U);
    return wait_dma_idle("MM2S", DMA_MM2S_DMASR);
}

static int start_mm2s_chunk(uint32_t word_offset, uint32_t words, int use_source_offset) {
    for (uint32_t attempt = 0; attempt < 3U; ++attempt) {
        clear_error();
        if (start_mm2s_chunk_once(word_offset, words, use_source_offset) == 0) {
            return 0;
        }
        uint32_t status = rd(g_dma, DMA_MM2S_DMASR);
        uint32_t rx_seen = rd(g_accel, REG_RX_WORDS);
        if (trace_chunks_enabled()) {
            fprintf(
                stderr,
                "axi_x2 mm2s retry attempt=%u staged_word_offset=%u words=%u status=0x%08x accel_rx_words=%u\n",
                attempt + 1U,
                word_offset,
                words,
                status,
                rx_seen
            );
        }
        if (reset_mm2s() != 0) {
            return -1;
        }
    }
    return set_error("MM2S DMA retry exhausted status=0x%08x", rd(g_dma, DMA_MM2S_DMASR));
}

static int start_s2mm_chunk(uint32_t word_offset, uint32_t words) {
    (void)word_offset;
    unsigned long phys = g_rx_phys;
    wr(g_accel, REG_WR_ADDR_LO, (uint32_t)phys);
    wr(g_accel, REG_WR_ADDR_HI, 0);
    if (trace_chunks_enabled()) {
        fprintf(stderr, "axi_x2 axi-writer addr=0x%lx words=%u\n", phys, words);
    }
    return 0;
}

static int wait_writer_idle(uint32_t expected_words) {
    const uint64_t timeout_ns = 5000000000ULL;
    const uint64_t start_ns = monotonic_ns();
    struct timespec ts = {.tv_sec = 0, .tv_nsec = DMA_POLL_SLEEP_NS};
    uint32_t spins = 0;
    while ((monotonic_ns() - start_ns) < timeout_ns) {
        uint32_t status = rd(g_accel, REG_WR_STATUS);
        uint32_t count = rd(g_accel, REG_WR_COUNT);
        if (status & WR_STATUS_ERR) {
            uint32_t bresp = (status & WR_STATUS_BRESP_MASK) >> WR_STATUS_BRESP_SHIFT;
            return set_error("AXI result writer error status=0x%08x count=%u bresp=%u", status, count, bresp);
        }
        if ((status & WR_STATUS_IDLE) && count >= expected_words) {
            return 0;
        }
        if (spins++ >= DMA_FAST_POLL_SPINS) {
            nanosleep(&ts, NULL);
        }
    }
    return set_error(
        "AXI result writer timeout status=0x%08x count=%u expected=%u",
        rd(g_accel, REG_WR_STATUS),
        rd(g_accel, REG_WR_COUNT),
        expected_words
    );
}

static int wait_dma_idle(const char *name, uint32_t status_off) {
    const uint64_t timeout_ns = 5000000000ULL;
    const uint64_t start_ns = monotonic_ns();
    struct timespec ts = {.tv_sec = 0, .tv_nsec = DMA_POLL_SLEEP_NS};
    uint32_t spins = 0;
    while ((monotonic_ns() - start_ns) < timeout_ns) {
        uint32_t status = rd(g_dma, status_off);
        if (status & DMA_SR_ERR) {
            return set_error("%s DMA error status=0x%08x", name, status);
        }
        if (status & DMA_SR_IDLE) {
            return 0;
        }
        if (spins++ >= DMA_FAST_POLL_SPINS) {
            nanosleep(&ts, NULL);
        }
    }
    return set_error("%s DMA timeout status=0x%08x", name, rd(g_dma, status_off));
}

static inline uint32_t ceil_div4(uint32_t value) {
    return (value + 3U) >> 2;
}

static void put_i8_byte(uint32_t *words, uint32_t byte_index, int8_t value) {
    uint32_t word_index = byte_index >> 2;
    uint32_t shift = (byte_index & 3U) * 8U;
    uint32_t mask = 0xFFU << shift;
    words[word_index] = (words[word_index] & ~mask) | (((uint32_t)(uint8_t)value) << shift);
}

int axi_x2_open(void) {
    if (g_accel && g_dma && g_tx && g_rx) {
        return 0;
    }

    const char *accel_base_env = getenv("YOLO_PL_ACCEL_BASE");
    unsigned long accel_base = ACCEL_BASE;
    int use_accel_devmem = 0;
    if (accel_base_env && accel_base_env[0]) {
        char *end = NULL;
        errno = 0;
        accel_base = strtoul(accel_base_env, &end, 0);
        if (errno != 0 || end == accel_base_env || (end && *end != '\0')) {
            return set_error("bad YOLO_PL_ACCEL_BASE value: %s", accel_base_env);
        }
        use_accel_devmem = 1;
    }

    if (use_accel_devmem) {
        g_memfd = open("/dev/mem", O_RDWR | O_SYNC);
        if (g_memfd < 0) {
            return set_error("open /dev/mem failed: %s", strerror(errno));
        }
        g_accel = map_regs(g_memfd, accel_base, ACCEL_SIZE);
        if (!g_accel) {
            axi_x2_close();
            return -1;
        }
    } else {
    const char *uio_path = getenv("YOLO_PL_ACCEL_UIO");
    if (!uio_path || !uio_path[0]) {
        uio_path = "/dev/uio4";
    }

    g_uiofd = open(uio_path, O_RDWR | O_SYNC);
    if (g_uiofd < 0) {
        return set_error("open %s failed: %s", uio_path, strerror(errno));
    }
    g_accel = map_regs(g_uiofd, 0, ACCEL_SIZE);
    if (!g_accel) {
        axi_x2_close();
        return -1;
    }
    }

    if (g_memfd < 0) {
        g_memfd = open("/dev/mem", O_RDWR | O_SYNC);
    }
    if (g_memfd < 0) {
        axi_x2_close();
        return set_error("open /dev/mem failed: %s", strerror(errno));
    }

    g_dma = map_regs(g_memfd, DMA_BASE, DMA_SIZE);
    if (!g_dma) {
        axi_x2_close();
        return -1;
    }

    uint32_t magic = rd(g_accel, REG_MAGIC);
    if (magic != MAGIC_X2MV) {
        axi_x2_close();
        return set_error(
            "unexpected accelerator magic 0x%08x, expected 0x%08x; load zcu106_axi_x2_mvm.bit",
            magic,
            MAGIC_X2MV
        );
    }
    uint32_t version = rd(g_accel, REG_VERSION);
    g_use_axi_reader = version >= AXI_READER_VERSION;
    g_use_strided_axi_reader = version >= STRIDED_AXI_READER_VERSION;

    const char *alloc_mode = getenv("YOLO_PL_ALLOC");
    g_use_cma = alloc_mode && strcmp(alloc_mode, "cma") == 0;
    g_use_reserved = (!alloc_mode || !alloc_mode[0] || strcmp(alloc_mode, "reserved") == 0);
    if (g_use_cma) {
        uint32_t cma_cacheable = 1U;
        const char *cacheable_env = getenv("YOLO_PL_CMA_CACHEABLE");
        if (cacheable_env && cacheable_env[0]) {
            cma_cacheable = (uint32_t)(strtoul(cacheable_env, NULL, 0) != 0);
        }
        g_tx = (uint32_t *)cma_alloc(DMA_BUF_BYTES, cma_cacheable);
        g_rx = (uint32_t *)cma_alloc(DMA_BUF_BYTES, cma_cacheable);
        if ((long)g_tx == -1 || (long)g_rx == -1 || g_tx == NULL || g_rx == NULL) {
            axi_x2_close();
            return set_error("cma_alloc failed; /dev/xlnk likely needs root/device permissions");
        }
        g_tx_phys = cma_get_phy_addr(g_tx);
        g_rx_phys = cma_get_phy_addr(g_rx);
    } else if (g_use_reserved) {
        const char *base_env = getenv("YOLO_PL_RESERVED_BASE");
        unsigned long base = RESERVED_DMA_BASE_DEFAULT;
        if (base_env && base_env[0]) {
            char *end = NULL;
            errno = 0;
            base = strtoul(base_env, &end, 0);
            if (errno != 0 || end == base_env || (end && *end != '\0')) {
                axi_x2_close();
                return set_error("bad YOLO_PL_RESERVED_BASE value: %s", base_env);
            }
        }

        g_tx_phys = base;
        g_rx_phys = base + DMA_BUF_BYTES;
        g_tx = map_reserved_dma_page(g_tx_phys);
        g_rx = map_reserved_dma_page(g_rx_phys);
        if (!g_tx || !g_rx) {
            axi_x2_close();
            return -1;
        }
    } else {
        g_tx = alloc_dma_page(&g_tx_phys, &g_tx_pin);
        g_rx = alloc_dma_page(&g_rx_phys, &g_rx_pin);
        if (!g_tx || !g_rx) {
            axi_x2_close();
            return -1;
        }
    }
    if (g_tx_phys >= DMA_ADDR_LIMIT || g_rx_phys >= DMA_ADDR_LIMIT) {
        unsigned long tx_phys = g_tx_phys;
        unsigned long rx_phys = g_rx_phys;
        axi_x2_close();
        return set_error("AXI DMA DDR aperture requires buffers below 0x%lx, got tx=0x%lx rx=0x%lx", DMA_ADDR_LIMIT, tx_phys, rx_phys);
    }

    const char *trace = getenv("YOLO_PL_TRACE");
    if (trace && trace[0]) {
        const char *mode_name = g_use_cma ? "cma" : (g_use_reserved ? "reserved" : "pagemap");
        fprintf(stderr, "axi_x2 buffers tx=0x%lx rx=0x%lx mode=%s version=0x%08x input=%s\n",
                g_tx_phys, g_rx_phys, mode_name, version,
                g_use_strided_axi_reader ? "axi-reader-stride16" :
                (g_use_axi_reader ? "axi-reader" : "axi-dma-mm2s"));
    }

    if (!g_use_axi_reader && reset_dma() != 0) {
        axi_x2_close();
        return -1;
    }

    return 0;
}

void axi_x2_close(void) {
    if (g_tx && (long)g_tx != -1 && g_use_cma) {
        cma_free(g_tx);
    } else if (g_tx) {
        munmap(g_tx, DMA_BUF_BYTES);
        if (g_tx_pin) {
            munlock(g_tx_pin, DMA_BUF_BYTES);
            free(g_tx_pin);
        }
    }
    if (g_rx && (long)g_rx != -1 && g_use_cma) {
        cma_free(g_rx);
    } else if (g_rx) {
        munmap(g_rx, DMA_BUF_BYTES);
        if (g_rx_pin) {
            munlock(g_rx_pin, DMA_BUF_BYTES);
            free(g_rx_pin);
        }
    }
    g_tx = NULL;
    g_rx = NULL;
    g_tx_pin = NULL;
    g_rx_pin = NULL;
    g_tx_phys = 0;
    g_rx_phys = 0;
    g_use_cma = 0;
    g_use_reserved = 0;
    g_use_axi_reader = 0;
    g_use_strided_axi_reader = 0;

    if (g_accel) {
        munmap((void *)g_accel, ACCEL_SIZE);
    }
    if (g_dma) {
        munmap((void *)g_dma, DMA_SIZE);
    }
    g_accel = NULL;
    g_dma = NULL;

    if (g_memfd >= 0) {
        close(g_memfd);
    }
    g_memfd = -1;
    if (g_uiofd >= 0) {
        close(g_uiofd);
    }
    g_uiofd = -1;
}

int axi_x2_get_stats(uint32_t *status, uint32_t *expected, uint32_t *rx_words, uint32_t *tx_words, uint32_t *cycles, uint32_t *error_code) {
    if (!g_accel) {
        return set_error("accelerator is not open");
    }
    if (status) {
        *status = rd(g_accel, REG_STATUS);
    }
    if (expected) {
        *expected = rd(g_accel, REG_EXPECTED);
    }
    if (rx_words) {
        *rx_words = rd(g_accel, REG_RX_WORDS);
    }
    if (tx_words) {
        *tx_words = rd(g_accel, REG_TX_WORDS);
    }
    if (cycles) {
        *cycles = rd(g_accel, REG_CYCLES);
    }
    if (error_code) {
        *error_code = rd(g_accel, REG_ERROR);
    }
    return 0;
}

int axi_x2_run_i8_batch(const int8_t *vectors, const int8_t *weights, uint32_t batch, uint32_t rows, uint32_t cols, int32_t *out) {
    if (!vectors || !weights || !out) {
        return set_error("null vectors, weights, or output pointer");
    }
    if (axi_x2_open() != 0) {
        return -1;
    }
    if (batch < 1U || batch > MAX_BATCH || rows < 1U || rows > MAX_ROWS || cols < 1U || cols > MAX_COLS) {
        return set_error(
            "bad batched MVM shape batch=%u rows=%u cols=%u; supported batch/rows/cols range is 1..32",
            batch,
            rows,
            cols
        );
    }

    uint32_t vector_words = ceil_div4(rows);
    uint32_t weight_bytes = rows * cols;
    uint32_t weight_words = ceil_div4(weight_bytes);
    uint32_t total_words = weight_words + (batch * vector_words);
    uint32_t output_words = batch * cols;
    uint32_t input_bytes = total_words * sizeof(uint32_t) *
                           (g_use_strided_axi_reader ? AXI_READER_WORD_STRIDE : 1U);
    if (input_bytes > DMA_BUF_BYTES || (output_words * 4U) > DMA_BUF_BYTES) {
        return set_error("DMA buffer too small for batch=%u rows=%u cols=%u", batch, rows, cols);
    }

    uint32_t packet_words[DMA_BUF_BYTES / sizeof(uint32_t)];
    memset(packet_words, 0, sizeof(packet_words));
    memset(g_tx, 0, DMA_BUF_BYTES);
    memset(g_rx, 0, DMA_BUF_BYTES);

    uint32_t weight_byte_base = vector_words * 4U;
    (void)weight_byte_base;
    for (uint32_t i = 0; i < weight_bytes; ++i) {
        put_i8_byte(packet_words, i, weights[i]);
    }
    uint32_t vector_byte_base = weight_words * 4U;
    uint32_t vector_stride_bytes = vector_words * 4U;
    for (uint32_t b = 0; b < batch; ++b) {
        for (uint32_t r = 0; r < rows; ++r) {
            put_i8_byte(packet_words, vector_byte_base + (b * vector_stride_bytes) + r, vectors[(b * rows) + r]);
        }
    }

    const char *trace = getenv("YOLO_PL_TRACE");
    if (trace && trace[0]) {
        uint32_t trace_words = total_words < 8U ? total_words : 8U;
        fprintf(stderr, "axi_x2 run batch=%u rows=%u cols=%u total_words=%u output_words=%u\n", batch, rows, cols, total_words, output_words);
        fprintf(stderr, "axi_x2 tx:");
        for (uint32_t i = 0; i < trace_words; ++i) {
            fprintf(stderr, " %08x", packet_words[i]);
        }
        fprintf(stderr, "\n");
    }

    if (!g_use_axi_reader && reset_dma() != 0) {
        return -1;
    }
    wr(g_accel, REG_CTRL, CTRL_ABORT | CTRL_CLEAR);
    wr(g_accel, REG_CONFIG, ((batch & 0xFFU) << 16) | ((cols & 0xFFU) << 8) | (rows & 0xFFU));

    uint32_t tx_chunk_words = g_use_axi_reader ? total_words : configured_tx_chunk_words(total_words);
    int use_source_offsets = g_use_axi_reader || (tx_chunk_words == total_words);
    if (trace && trace[0]) {
        fprintf(stderr, "axi_x2 input policy=%s tx_chunk_words=%u\n",
                g_use_axi_reader ? "axi-reader" : "axi-dma-mm2s", tx_chunk_words);
    }

    memset(g_tx, 0, DMA_BUF_BYTES);
    if (g_use_strided_axi_reader) {
        for (uint32_t i = 0; i < total_words; ++i) {
            g_tx[i * AXI_READER_WORD_STRIDE] = packet_words[i];
        }
    } else if (use_source_offsets) {
        memcpy(g_tx, packet_words, total_words * sizeof(uint32_t));
    }
    memset(g_rx, 0, DMA_BUF_BYTES);
    flush_tx_buffer();
    if (start_s2mm_chunk(0, output_words) != 0) {
        return -1;
    }
    if (g_use_axi_reader) {
        wr(g_accel, REG_RD_ADDR_LO, (uint32_t)g_tx_phys);
        wr(g_accel, REG_RD_ADDR_HI, 0);
    }

    wr(g_accel, REG_CTRL, CTRL_START);

    if (!g_use_axi_reader) {
        for (uint32_t tx_offset = 0; tx_offset < total_words; tx_offset += tx_chunk_words) {
            uint32_t tx_chunk = min_u32(tx_chunk_words, total_words - tx_offset);
            if (!use_source_offsets) {
                memset(g_tx, 0, DMA_BUF_BYTES);
                memcpy(g_tx, &packet_words[tx_offset], tx_chunk * sizeof(uint32_t));
                flush_tx_buffer();
            }
            if (start_mm2s_chunk(tx_offset, tx_chunk, use_source_offsets) != 0) {
                return -1;
            }
        }
    }

    if (wait_writer_idle(output_words) != 0) {
        return -1;
    }
    if (trace && trace[0]) {
        fprintf(stderr, "axi_x2 transport mm2s=0x%08x reader_status=0x%08x reader_count=%u writer_status=0x%08x writer_count=%u\n",
                rd(g_dma, DMA_MM2S_DMASR), rd(g_accel, REG_RD_STATUS), rd(g_accel, REG_RD_COUNT),
                rd(g_accel, REG_WR_STATUS), rd(g_accel, REG_WR_COUNT));
    }
    invalidate_rx_buffer();
    for (uint32_t i = 0; i < output_words; ++i) {
        out[i] = (int32_t)g_rx[i];
    }

    if (trace && trace[0]) {
        uint32_t trace_words = output_words < 8U ? output_words : 8U;
        fprintf(stderr, "axi_x2 rx:");
        for (uint32_t i = 0; i < trace_words; ++i) {
            fprintf(stderr, " %08x", (uint32_t)out[i]);
        }
        fprintf(stderr, "\n");
        fprintf(
            stderr,
            "axi_x2 debug input:%08x %08x %08x %08x weight:%08x %08x vector:%08x %08x\n",
            rd(g_accel, REG_DBG_STREAM0),
            rd(g_accel, REG_DBG_STREAM1),
            rd(g_accel, REG_DBG_STREAM2),
            rd(g_accel, REG_DBG_STREAM3),
            rd(g_accel, REG_DBG_WEIGHT0),
            rd(g_accel, REG_DBG_WEIGHT1),
            rd(g_accel, REG_DBG_VECTOR0),
            rd(g_accel, REG_DBG_VECTOR1)
        );
        fprintf(
            stderr,
            "axi_x2 debug result:%08x %08x %08x %08x output:%08x %08x %08x %08x output_meta:%08x\n",
            rd(g_accel, REG_DBG_RESULT0),
            rd(g_accel, REG_DBG_RESULT1),
            rd(g_accel, REG_DBG_RESULT2),
            rd(g_accel, REG_DBG_RESULT3),
            rd(g_accel, REG_DBG_AXIS0),
            rd(g_accel, REG_DBG_AXIS1),
            rd(g_accel, REG_DBG_AXIS2),
            rd(g_accel, REG_DBG_AXIS3),
            rd(g_accel, REG_DBG_AXIS_META)
        );
    }

    uint32_t status = rd(g_accel, REG_STATUS);
    uint32_t error_code = rd(g_accel, REG_ERROR);
    uint32_t rx_seen = rd(g_accel, REG_RX_WORDS);
    uint32_t tx_seen = rd(g_accel, REG_TX_WORDS);
    uint32_t reader_seen = g_use_axi_reader ? rd(g_accel, REG_RD_COUNT) : total_words;
    uint32_t reader_status = g_use_axi_reader ? rd(g_accel, REG_RD_STATUS) : 0U;
    if (g_use_axi_reader && ((reader_status & 0x8U) || reader_seen != total_words)) {
        return set_error(
            "AXI input reader failed status=0x%08x read_words=%u expected=%u",
            reader_status,
            reader_seen,
            total_words
        );
    }
    if (status & 0x4U) {
        return set_error("accelerator status error=0x%08x code=%u", status, error_code);
    }
    if (rx_seen != total_words || tx_seen != output_words) {
        return set_error(
            "unexpected transfer count rx_words=%u tx_words=%u expected_rx=%u expected_tx=%u status=0x%08x",
            rx_seen,
            tx_seen,
            total_words,
            output_words,
            status
        );
    }

    return 0;
}

int axi_x2_run_i8(const int8_t *vector, const int8_t *weights, uint32_t rows, uint32_t cols, int32_t *out) {
    return axi_x2_run_i8_batch(vector, weights, 1U, rows, cols, out);
}

#ifdef AXI_X2_MVM_STANDALONE
static int run_standalone_case(const char *name, uint32_t batch, uint32_t rows, uint32_t cols) {
    int8_t vectors[MAX_BATCH * MAX_ROWS];
    int8_t weights[MAX_ROWS * MAX_COLS];
    int32_t out[MAX_BATCH * MAX_COLS];

    memset(vectors, 0, sizeof(vectors));
    memset(weights, 0, sizeof(weights));
    memset(out, 0, sizeof(out));

    for (uint32_t b = 0; b < batch; ++b) {
        for (uint32_t r = 0; r < rows; ++r) {
            vectors[(b * rows) + r] = (int8_t)(((int32_t)(b * 3U + r) % 17) - 8);
        }
    }
    for (uint32_t r = 0; r < rows; ++r) {
        for (uint32_t c = 0; c < cols; ++c) {
            weights[(r * cols) + c] = (int8_t)(((int32_t)(r * 5U + c * 3U) % 23) - 11);
        }
    }

    int rc = axi_x2_run_i8_batch(vectors, weights, batch, rows, cols, out);
    if (rc != 0) {
        fprintf(stderr, "%s: %s\n", name, axi_x2_last_error());
        return 1;
    }

    for (uint32_t b = 0; b < batch; ++b) {
        for (uint32_t c = 0; c < cols; ++c) {
            int32_t expected = 0;
            for (uint32_t r = 0; r < rows; ++r) {
                expected += (int32_t)vectors[(b * rows) + r] * (int32_t)weights[(r * cols) + c];
            }
            uint32_t out_index = (b * cols) + c;
            if (out[out_index] != expected) {
                fprintf(
                    stderr,
                    "%s mismatch batch=%u col=%u expected=%d got=%d\n",
                    name,
                    b,
                    c,
                    expected,
                    out[out_index]
                );
                return 1;
            }
        }
    }

    printf("PASS: %s batch=%u rows=%u cols=%u\n", name, batch, rows, cols);
    return 0;
}

int main(void) {
    setenv("YOLO_PL_TX_CHUNK_WORDS", "full", 0);

    int failed = 0;
    failed |= run_standalone_case("small_full_burst", 1, 3, 2);
    failed |= run_standalone_case("unaligned_length_full_burst", 3, 5, 3);
    failed |= run_standalone_case("large_full_burst", 32, 32, 32);

    axi_x2_close();
    if (failed) {
        return 1;
    }
    puts("PASS: zcu106_axi_x2_mvm full-burst CMA/MMIO/DMA smoke");
    return 0;
}
#endif
