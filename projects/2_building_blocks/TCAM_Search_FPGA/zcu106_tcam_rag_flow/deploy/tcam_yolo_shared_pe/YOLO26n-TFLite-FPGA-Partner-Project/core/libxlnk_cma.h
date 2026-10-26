#ifndef LIBXLNK_CMA_H
#define LIBXLNK_CMA_H

#include <stddef.h>

void *cma_alloc(size_t length, unsigned int cacheable);
void cma_free(void *buffer);
unsigned long cma_get_phy_addr(void *buffer);
void cma_flush_cache(void *buffer, unsigned int physical_address, int size);
void cma_invalidate_cache(void *buffer, unsigned int physical_address, int size);

#endif
