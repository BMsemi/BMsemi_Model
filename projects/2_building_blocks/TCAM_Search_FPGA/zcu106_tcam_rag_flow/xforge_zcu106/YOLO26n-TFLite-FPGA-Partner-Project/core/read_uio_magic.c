#include <errno.h>
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>

int main(void) {
    int fd = open("/dev/uio4", O_RDWR | O_SYNC);
    if (fd < 0) {
        fprintf(stderr, "open /dev/uio4 failed: %s\n", strerror(errno));
        return 1;
    }
    void *ptr = mmap(NULL, 0x1000, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (ptr == MAP_FAILED) {
        fprintf(stderr, "mmap /dev/uio4 failed: %s\n", strerror(errno));
        close(fd);
        return 1;
    }
    volatile uint32_t *regs = (volatile uint32_t *)ptr;
    printf("magic=0x%08x version=0x%08x\n", regs[0x20 / 4], regs[0x24 / 4]);
    munmap(ptr, 0x1000);
    close(fd);
    return 0;
}
