#include "include/openviking_cache_provider_v1.h"
typedef struct NativeCache NativeCache;
typedef struct {
  int status;
  OvCacheOwnedBufferV1 value;
  uint64_t count;
  int found;
} NativeResult;
NativeCache *native_open(const char *library, const void *params, size_t len,
                         int *status);
NativeResult native_get(NativeCache *, const void *, size_t);
NativeResult native_set(NativeCache *, const void *, size_t, const void *,
                        size_t, int64_t);
NativeResult native_delete(NativeCache *, const void *, size_t);
int native_ping(NativeCache *);
int native_close(NativeCache *);
void native_release(OvCacheOwnedBufferV1);
