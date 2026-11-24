//go:build cgo && (linux || darwin)

#include "dynamic_native.h"
#include <dlfcn.h>
#include <pthread.h>
#include <stdlib.h>
#include <string.h>

#define MAX_ALLOC (64u * 1024u * 1024u)
struct allocation {
  void *ptr;
  size_t len, align;
  struct allocation *next;
};
static struct allocation *allocations;
static pthread_mutex_t alloc_mu = PTHREAD_MUTEX_INITIALIZER;
static uint8_t *host_alloc(size_t n, size_t align) {
  if (!n || n > MAX_ALLOC || !align || (align & (align - 1)) || align > 4096)
    return NULL;
  void *p = NULL;
  if (posix_memalign(&p, align < sizeof(void *) ? sizeof(void *) : align, n))
    return NULL;
  struct allocation *a = malloc(sizeof(*a));
  if (!a) {
    free(p);
    return NULL;
  }
  *a = (struct allocation){p, n, align, NULL};
  pthread_mutex_lock(&alloc_mu);
  a->next = allocations;
  allocations = a;
  pthread_mutex_unlock(&alloc_mu);
  return p;
}
static void host_free(uint8_t *p, size_t n, size_t align) {
  (void)n;
  (void)align;
  if (!p)
    return;
  pthread_mutex_lock(&alloc_mu);
  struct allocation **q = &allocations;
  while (*q && (*q)->ptr != p)
    q = &(*q)->next;
  if (*q) {
    struct allocation *a = *q;
    *q = a->next;
    free(a->ptr);
    free(a);
  }
  pthread_mutex_unlock(&alloc_mu);
}
static int valid(OvCacheOwnedBufferV1 b) {
  if (!b.data)
    return b.len == 0;
  if (!b.len || b.len > MAX_ALLOC)
    return 0;
  int ok = 0;
  pthread_mutex_lock(&alloc_mu);
  for (struct allocation *a = allocations; a; a = a->next) {
    if (a->ptr == b.data) {
      ok = a->len == b.len && a->align == 1;
      break;
    }
  }
  pthread_mutex_unlock(&alloc_mu);
  return ok;
}
void native_release(OvCacheOwnedBufferV1 b) { host_free(b.data, b.len, 1); }
static const OvCacheHostApiV1 host = {
    OV_CACHE_PROVIDER_ABI_V1, sizeof(OvCacheHostApiV1), host_alloc, host_free};
struct NativeCache {
  void *lib;
  const OvCacheProviderApiV1 *api;
  void *provider;
};
static int finish(int status, OvCacheOwnedBufferV1 err) {
  int good = valid(err) && (status != 0 || (!err.data && !err.len));
  native_release(err);
  return good ? status : OV_CACHE_STATUS_INVALID_DATA;
}
NativeCache *native_open(const char *library, const void *params, size_t len,
                         int *status) {
  *status = OV_CACHE_STATUS_UNSUPPORTED_OPERATION;
  void *lib = dlopen(library, RTLD_NOW | RTLD_LOCAL);
  if (!lib)
    return NULL;
  const OvCacheProviderApiV1 *(*entry)(void) =
      dlsym(lib, "openviking_cache_provider_v1");
  if (!entry) {
    dlclose(lib);
    return NULL;
  }
  const OvCacheProviderApiV1 *api = entry();
  if (!api || api->abi_version != 1 || api->struct_size < sizeof(*api) ||
      !api->create || !api->ping || !api->close || !api->get || !api->set ||
      !api->del) {
    dlclose(lib);
    return NULL;
  }
  NativeCache *n = calloc(1, sizeof(*n));
  if (!n) {
    dlclose(lib);
    *status = OV_CACHE_STATUS_UNAVAILABLE;
    return NULL;
  }
  n->lib = lib;
  n->api = api;
  OvCacheOwnedBufferV1 err = {0};
  int created =
      api->create(&host, (OvCacheByteSliceV1){params, len}, &n->provider, &err);
  *status = finish(created, err);
  if (*status || !n->provider) {
    if (!*status)
      *status = OV_CACHE_STATUS_INVALID_DATA;
    if (n->provider) {
      OvCacheOwnedBufferV1 e = {0};
      api->close(n->provider, &e);
      native_release(e);
    }
    dlclose(lib);
    free(n);
    return NULL;
  }
  return n;
}
NativeResult native_get(NativeCache *n, const void *k, size_t len) {
  OvCacheOptionalOwnedBufferV1 out = {0};
  OvCacheOwnedBufferV1 err = {0};
  int s = n->api->get(n->provider, (OvCacheByteSliceV1){k, len}, &out, &err);
  if (out.value.data && out.value.data == err.data) {
    native_release(err);
    return (NativeResult){.status = OV_CACHE_STATUS_INVALID_DATA};
  }
  int ok = valid(out.value) && out.present <= 1 &&
           (out.present || (!out.value.data && !out.value.len)) &&
           (!s || (!out.present && !out.value.data && !out.value.len));
  s = finish(s, err);
  if (!ok)
    s = OV_CACHE_STATUS_INVALID_DATA;
  if (s) {
    native_release(out.value);
    return (NativeResult){.status = s};
  }
  return (NativeResult){.value = out.value, .found = out.present};
}
NativeResult native_set(NativeCache *n, const void *k, size_t kl, const void *v,
                        size_t vl, int64_t ttl) {
  OvCacheOwnedBufferV1 err = {0};
  uint8_t result = 0;
  OvCacheSetOptionsV1 options = {ttl, OV_CACHE_SET_CONDITION_NONE, 0};
  int s = n->api->set(n->provider, (OvCacheByteSliceV1){k, kl},
                      (OvCacheByteSliceV1){v, vl}, options, &result, &err);
  s = finish(s, err);
  if ((!s && result != OV_CACHE_SET_APPLIED) || (s && result != 0))
    s = OV_CACHE_STATUS_INVALID_DATA;
  return (NativeResult){.status = s};
}
NativeResult native_delete(NativeCache *n, const OvCacheByteSliceV1 *keys,
                           size_t len) {
  OvCacheOwnedBufferV1 err = {0};
  uint64_t count = 0;
  int s = n->api->del(n->provider, keys, len, &count, &err);
  s = finish(s, err);
  if ((!s && count > len) || (s && count))
    s = OV_CACHE_STATUS_INVALID_DATA;
  return (NativeResult){.status = s, .count = count};
}
int native_ping(NativeCache *n) {
  OvCacheOwnedBufferV1 err = {0};
  int s = n->api->ping(n->provider, &err);
  return finish(s, err);
}
int native_close(NativeCache *n) {
  OvCacheOwnedBufferV1 err = {0};
  int s = n->api->close(n->provider, &err);
  s = finish(s, err);
  dlclose(n->lib);
  free(n);
  return s;
}
