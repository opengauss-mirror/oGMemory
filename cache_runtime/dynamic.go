//go:build cgo && (linux || darwin)

package cacheruntime

/*
#cgo linux LDFLAGS: -ldl -lpthread
#include <stdlib.h>
#include "dynamic_native.h"
*/
import "C"
import (
	"context"
	"encoding/json"
	"fmt"
	"time"
	"unsafe"
)

type dynamicProvider struct{ native *C.NativeCache }

func nativeError(s C.int) error {
	if s == 0 {
		return nil
	}
	switch s {
	case C.OV_CACHE_STATUS_TIMEOUT:
		return context.DeadlineExceeded
	case C.OV_CACHE_STATUS_INVALID_ARGUMENT:
		return ErrInvalidArgument
	case C.OV_CACHE_STATUS_INVALID_DATA:
		return ErrInvalidData
	case C.OV_CACHE_STATUS_UNSUPPORTED_OPERATION:
		return ErrUnsupported
	}
	return fmt.Errorf("%w: native status %d", ErrUnavailable, int(s))
}
func newDynamic(params map[string]interface{}) (Provider, error) {
	p := map[string]interface{}{}
	for k, v := range params {
		if k != "library" {
			p[k] = v
		}
	}
	b, e := json.Marshal(p)
	if e != nil {
		return nil, ErrInvalidArgument
	}
	data := C.CBytes(b)
	defer C.free(data)
	lib := C.CString(params["library"].(string))
	defer C.free(unsafe.Pointer(lib))
	var status C.int
	n := C.native_open(lib, data, C.size_t(len(b)), &status)
	if n == nil {
		if e := nativeError(status); e != nil {
			return nil, e
		}
		return nil, ErrInvalidData
	}
	return &dynamicProvider{n}, nil
}
func (p *dynamicProvider) Get(c context.Context, k string) (Value, error) {
	return p.get(c, k, 0)
}
func (p *dynamicProvider) getNonEmpty(c context.Context, k string, maxBytes int) (Value, error) {
	return p.get(c, k, maxBytes)
}
func (p *dynamicProvider) get(c context.Context, k string, maxBytes int) (Value, error) {
	if e := c.Err(); e != nil {
		return Value{}, e
	}
	key := C.CBytes([]byte(k))
	defer C.free(key)
	r := C.native_get(p.native, key, C.size_t(len(k)))
	defer C.native_release(r.value)
	if e := nativeError(r.status); e != nil {
		return Value{}, e
	}
	if maxBytes > 0 && (r.value.len == 0 || r.value.len > C.size_t(maxBytes)) {
		return Value{}, nil
	}
	return Value{C.GoBytes(unsafe.Pointer(r.value.data), C.int(r.value.len)), r.found != 0}, nil
}
func (p *dynamicProvider) Set(c context.Context, k string, v []byte, t time.Duration) error {
	if e := c.Err(); e != nil {
		return e
	}
	key := C.CBytes([]byte(k))
	defer C.free(key)
	data := C.CBytes(v)
	defer C.free(data)
	ttl := int64(-1)
	if t > 0 {
		ttl = int64(t / time.Millisecond)
		if t%time.Millisecond != 0 {
			ttl++
		}
	}
	return nativeError(C.native_set(p.native, key, C.size_t(len(k)), data, C.size_t(len(v)), C.int64_t(ttl)).status)
}
func (p *dynamicProvider) Delete(c context.Context, keys []string) (uint64, error) {
	if len(keys) == 0 {
		return 0, nil
	}
	if err := c.Err(); err != nil {
		return 0, err
	}
	// Elements point only to C-owned bytes and are borrowed for this call.
	nativeKeys := make([]C.OvCacheByteSliceV1, len(keys))
	defer func() {
		for _, key := range nativeKeys {
			C.free(unsafe.Pointer(key.data))
		}
	}()
	for i, key := range keys {
		if key != "" {
			nativeKeys[i].data = (*C.uint8_t)(C.CBytes([]byte(key)))
			nativeKeys[i].len = C.size_t(len(key))
		}
	}
	if err := c.Err(); err != nil {
		return 0, err
	}
	result := C.native_delete(p.native, &nativeKeys[0], C.size_t(len(nativeKeys)))
	if err := nativeError(result.status); err != nil {
		return 0, err
	}
	return uint64(result.count), nil
}
func (p *dynamicProvider) Ping(c context.Context) error {
	if e := c.Err(); e != nil {
		return e
	}
	return nativeError(C.native_ping(p.native))
}
func (p *dynamicProvider) Close() error { return nativeError(C.native_close(p.native)) }
