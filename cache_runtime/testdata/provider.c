#include "openviking_cache_provider_v1.h"
#include <pthread.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#ifndef BAD_ABI
#define BAD_ABI 0
#endif
// Small linked-map fixture; no external service or language runtime required.
typedef struct Entry {
  unsigned char *key, *value;
  size_t keylen, len;
  int64_t expires;
  struct Entry *next;
} Entry;
typedef struct {
  pthread_mutex_t mu;
  const OvCacheHostApiV1 *host;
  Entry *entries;
  int bad, alias_output, require_batch;
} Store;
static int64_t now_ms(void) {
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return t.tv_sec * 1000 + t.tv_nsec / 1000000;
}
static void release_entry(Entry *e) {
  free(e->key);
  free(e->value);
  free(e);
}
static Entry **find(Store *s, OvCacheByteSliceV1 k) {
  Entry **e = &s->entries;
  while (*e) {
    if ((*e)->expires >= 0 && (*e)->expires <= now_ms()) {
      Entry *old = *e;
      *e = old->next;
      release_entry(old);
      continue;
    }
    if ((*e)->keylen == k.len && (!k.len || !memcmp((*e)->key, k.data, k.len)))
      break;
    e = &(*e)->next;
  }
  return e;
}
static int create(const OvCacheHostApiV1 *h, OvCacheByteSliceV1 params,
                  void **out, OvCacheOwnedBufferV1 *err) {
  (void)err;
  Store *s = calloc(1, sizeof(*s));
  if (!s)
    return OV_CACHE_STATUS_UNAVAILABLE;
  pthread_mutex_init(&s->mu, NULL);
  s->host = h;
  char *p = calloc(params.len + 1, 1);
  if (!p) {
    pthread_mutex_destroy(&s->mu);
    free(s);
    return OV_CACHE_STATUS_UNAVAILABLE;
  }
  memcpy(p, params.data, params.len);
  s->bad = strstr(p, "badoutput") != NULL;
  s->alias_output = strstr(p, "aliasoutput") != NULL;
  s->require_batch = strstr(p, "batchdelete") != NULL;
  free(p);
  *out = s;
  return 0;
}
static int ping(void *p, OvCacheOwnedBufferV1 *e) {
  (void)p;
  (void)e;
  return 0;
}
static int close_store(void *p, OvCacheOwnedBufferV1 *e) {
  (void)e;
  Store *s = p;
  while (s->entries) {
    Entry *old = s->entries;
    s->entries = old->next;
    release_entry(old);
  }
  pthread_mutex_destroy(&s->mu);
  free(s);
  return 0;
}
static int get(void *p, OvCacheByteSliceV1 k, OvCacheOptionalOwnedBufferV1 *out,
               OvCacheOwnedBufferV1 *e) {
  (void)e;
  Store *s = p;
  pthread_mutex_lock(&s->mu);
  Entry *entry = *find(s, k);
  if (entry && entry->len) {
    out->value.data = s->host->alloc(entry->len, 1);
    if (!out->value.data) {
      pthread_mutex_unlock(&s->mu);
      return OV_CACHE_STATUS_UNAVAILABLE;
    }
    out->value.len = entry->len;
    memcpy(out->value.data, entry->value, entry->len);
    if (s->bad)
      out->value.len += 123;
    if (s->alias_output)
      *e = out->value;
  }
  out->present = entry != NULL;
  pthread_mutex_unlock(&s->mu);
  return 0;
}
static int set(void *p, OvCacheByteSliceV1 k, OvCacheByteSliceV1 v,
               OvCacheSetOptionsV1 opts, uint8_t *result,
               OvCacheOwnedBufferV1 *e) {
  (void)e;
  Store *s = p;
  Entry *n = calloc(1, sizeof(*n));
  if (!n)
    return OV_CACHE_STATUS_UNAVAILABLE;
  n->key = malloc(k.len ? k.len : 1);
  n->value = malloc(v.len ? v.len : 1);
  if (!n->key || !n->value) {
    release_entry(n);
    return OV_CACHE_STATUS_UNAVAILABLE;
  }
  if (k.len)
    memcpy(n->key, k.data, k.len);
  if (v.len)
    memcpy(n->value, v.data, v.len);
  n->keylen = k.len;
  n->len = v.len;
  n->expires = opts.expiration_ms < 0 ? -1 : now_ms() + opts.expiration_ms;
  pthread_mutex_lock(&s->mu);
  Entry **slot = find(s, k);
  Entry *old = *slot;
  n->next = old ? old->next : NULL;
  *slot = n;
  if (old)
    release_entry(old);
  pthread_mutex_unlock(&s->mu);
  *result = 0;
  return 0;
}
static int del(void *p, const OvCacheByteSliceV1 *k, size_t count,
               uint64_t *removed, OvCacheOwnedBufferV1 *e) {
  (void)e;
  Store *s = p;
  if (s->require_batch && count < 2)
    return OV_CACHE_STATUS_INVALID_ARGUMENT;
  pthread_mutex_lock(&s->mu);
  for (size_t i = 0; i < count; i++) {
    Entry **slot = find(s, k[i]);
    if (*slot) {
      Entry *old = *slot;
      *slot = old->next;
      release_entry(old);
      (*removed)++;
    }
  }
  pthread_mutex_unlock(&s->mu);
  return 0;
}
static const OvCacheProviderApiV1 api = {.abi_version = BAD_ABI ? 99 : 1,
                                         .struct_size =
                                             sizeof(OvCacheProviderApiV1),
                                         .create = create,
                                         .ping = ping,
                                         .close = close_store,
                                         .get = get,
                                         .set = set,
                                         .del = del};
OV_CACHE_PROVIDER_EXPORT const OvCacheProviderApiV1 *
openviking_cache_provider_v1(void) {
  return &api;
}
