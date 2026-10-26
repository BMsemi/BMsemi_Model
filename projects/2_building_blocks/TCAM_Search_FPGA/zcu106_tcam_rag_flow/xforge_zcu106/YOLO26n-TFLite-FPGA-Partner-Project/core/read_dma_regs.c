#include <errno.h>
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>

int main(void) {
    int fd = open("/dev/mem", O_RDWR | O_SYNC);
    if (fd < 0) {
        fprintf(stderr, "open /dev/mem failed: %s\n", strerror(errno));
        return 1;
    }
    void *ptr = mmap(NULL, 0x10000, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0xA0010000UL);
    if (ptr == MAP_FAILED) {
        fprintf(stderr, "mmap DMA failed: %s\n", strerror(errno));
        close(fd);
        return 1;
    }
    volatile uint32_t *regs = (volatile uint32_t *)ptr;
    printf("mm2s_dmacr=0x%08x mm2s_dmasr=0x%08x s2mm_dmacr=0x%08x s2mm_dmasr=0x%08x\n",
           regs[0x00 / 4], regs[0x04 / 4], regs[0x30 / 4], regs[0x34 / 4]);
    munmap(ptr, 0x10000);
    close(fd);
    return 0;
}
