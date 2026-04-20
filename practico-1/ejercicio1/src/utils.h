#ifndef UTILS_H
#define UTILS_H

#include <time.h>
#include <sys/time.h>

static inline double get_time() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

#endif
