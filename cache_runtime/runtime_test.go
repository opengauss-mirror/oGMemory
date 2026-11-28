package cacheruntime

import (
	"bytes"
	"context"
	"errors"
	"github.com/alicebob/miniredis/v2"
	"sync/atomic"
	"testing"
	"time"
)

func TestRedisContract(t *testing.T) {
	server := miniredis.RunT(t)
	r, e := New(Config{Provider: "redis", CommandTimeoutMS: 500, Params: map[string]interface{}{"endpoints": []string{"redis://" + server.Addr()}}})
	if e != nil {
		t.Fatal(e)
	}
	testKVContract(t, r, server.FastForward)
}

// The same assertions run against Redis and a native Provider. Each backend
// supplies only how its clock advances for expiration.
func testKVContract(t *testing.T, r Runtime, advance func(time.Duration)) {
	t.Helper()
	ctx := context.Background()
	t.Cleanup(func() { _ = r.Close(ctx) })
	if err := r.Ping(ctx); err != nil {
		t.Fatal(err)
	}
	if v, err := r.Get(ctx, "missing"); err != nil || v.Found {
		t.Fatalf("miss: %+v, %v", v, err)
	}
	for k, v := range map[string][]byte{"empty": {}, "binary": {0, 1, 255}} {
		if err := r.Set(ctx, k, v, 0); err != nil {
			t.Fatal(err)
		}
		if got, err := r.Get(ctx, k); err != nil || !got.Found || !bytes.Equal(got.Data, v) {
			t.Fatalf("%s: %+v, %v", k, got, err)
		}
	}
	vals, err := r.MGet(ctx, []string{"binary", "empty", "missing", "binary"})
	if err != nil || len(vals) != 4 || !vals[0].Found || !bytes.Equal(vals[0].Data, []byte{0, 1, 255}) ||
		!vals[1].Found || len(vals[1].Data) != 0 || vals[2].Found || !vals[3].Found || !bytes.Equal(vals[3].Data, vals[0].Data) {
		t.Fatalf("mget: %+v, %v", vals, err)
	}
	if err := r.Set(ctx, "bad", nil, -time.Second); !errors.Is(err, ErrInvalidArgument) {
		t.Fatal(err)
	}
	if err := r.Set(ctx, "expires", []byte("value"), time.Second); err != nil {
		t.Fatal(err)
	}
	if v, err := r.Get(ctx, "expires"); err != nil || !v.Found || !bytes.Equal(v.Data, []byte("value")) {
		t.Fatalf("before TTL: %+v, %v", v, err)
	}
	advance(1100 * time.Millisecond)
	if v, err := r.Get(ctx, "expires"); err != nil || v.Found {
		t.Fatalf("TTL: %+v, %v", v, err)
	}
	if v, err := r.Get(ctx, "empty"); err != nil || !v.Found || len(v.Data) != 0 {
		t.Fatalf("zero TTL: %+v, %v", v, err)
	}
	if n, err := r.Delete(ctx, []string{"empty", "empty", "missing"}); err != nil || n != 1 {
		t.Fatalf("delete: %d, %v", n, err)
	}
	if v, err := r.Get(ctx, "empty"); err != nil || v.Found {
		t.Fatalf("deleted key: %+v, %v", v, err)
	}
	if err := r.Close(ctx); err != nil {
		t.Fatal(err)
	}
	if _, err := r.Get(ctx, "x"); !errors.Is(err, ErrClosed) {
		t.Fatal(err)
	}
	if err := r.Close(ctx); err != nil {
		t.Fatal(err)
	}
}

type blockedProvider struct {
	started chan struct{}
	release chan struct{}
	closed  atomic.Int64
}

func (p *blockedProvider) Get(context.Context, string) (Value, error) {
	close(p.started)
	<-p.release
	return Value{}, nil
}
func (p *blockedProvider) Set(context.Context, string, []byte, time.Duration) error { return nil }
func (p *blockedProvider) Delete(context.Context, []string) (uint64, error)         { return 0, nil }
func (p *blockedProvider) Ping(context.Context) error                               { return nil }
func (p *blockedProvider) Close() error                                             { p.closed.Add(1); return nil }
func TestTimeoutRetainsSlotAndCloseWaits(t *testing.T) {
	p := &blockedProvider{started: make(chan struct{}), release: make(chan struct{})}
	r, _ := WithProvider(p, Config{CommandTimeoutMS: 20, MaxInflight: 1})
	if _, e := r.Get(context.Background(), "x"); !errors.Is(e, context.DeadlineExceeded) {
		t.Fatal(e)
	}
	if e := r.Ping(context.Background()); !errors.Is(e, context.DeadlineExceeded) {
		t.Fatal("timed-out native call released slot")
	}
	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Millisecond)
	defer cancel()
	if e := r.Close(ctx); !errors.Is(e, context.DeadlineExceeded) {
		t.Fatal(e)
	}
	if p.closed.Load() != 0 {
		t.Fatal("closed active provider")
	}
	close(p.release)
	if e := r.Close(context.Background()); e != nil {
		t.Fatal(e)
	}
	if p.closed.Load() != 1 {
		t.Fatal("close not once")
	}
}

func TestRedisBoundedNonEmptyRead(t *testing.T) {
	server := miniredis.RunT(t)
	r, err := New(Config{Provider: "redis", CommandTimeoutMS: 500,
		Params: map[string]interface{}{"endpoints": []string{"redis://" + server.Addr()}}})
	if err != nil {
		t.Fatal(err)
	}
	ctx := context.Background()
	defer r.Close(ctx)
	for _, tc := range []struct {
		key, data string
		found     bool
	}{
		{"empty", "", false},
		{"exact", "abc", true},
		{"oversized", "abcd", false},
		{"missing", "", false},
	} {
		if tc.key != "missing" {
			if err := r.Set(ctx, tc.key, []byte(tc.data), 0); err != nil {
				t.Fatal(err)
			}
		}
		v, err := GetNonEmpty(ctx, r, tc.key, 3)
		if err != nil || v.Found != tc.found || (v.Found && string(v.Data) != tc.data) {
			t.Fatalf("%s: %+v, %v", tc.key, v, err)
		}
	}
}
