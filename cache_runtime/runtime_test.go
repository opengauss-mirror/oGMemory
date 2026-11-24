package cacheruntime

import (
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
	defer r.Close(context.Background())
	ctx := context.Background()
	if e = r.Ping(ctx); e != nil {
		t.Fatal(e)
	}
	if v, e := r.Get(ctx, "missing"); e != nil || v.Found {
		t.Fatalf("miss %v %v", v, e)
	}
	for k, v := range map[string][]byte{"empty": {}, "binary": {0, 1, 255}} {
		if e = r.Set(ctx, k, v, time.Second); e != nil {
			t.Fatal(e)
		}
	}
	vals, e := r.MGet(ctx, []string{"binary", "empty", "missing", "binary"})
	if e != nil || len(vals) != 4 || !vals[1].Found || vals[2].Found || len(vals[0].Data) != 3 || vals[3].Data[2] != 255 {
		t.Fatalf("mget %v %v", vals, e)
	}
	if e = r.Set(ctx, "bad", nil, -time.Second); !errors.Is(e, ErrInvalidArgument) {
		t.Fatal(e)
	}
	server.FastForward(time.Second)
	if v, e := r.Get(ctx, "binary"); e != nil || v.Found {
		t.Fatal("TTL")
	}
	if e = r.Set(ctx, "persist", nil, 0); e != nil {
		t.Fatal(e)
	}
	server.FastForward(time.Hour)
	if v, e := r.Get(ctx, "persist"); e != nil || !v.Found {
		t.Fatal("no TTL")
	}
	if n, e := r.Delete(ctx, []string{"persist", "persist"}); e != nil || n != 1 {
		t.Fatalf("delete %d %v", n, e)
	}
	if e = r.Close(ctx); e != nil {
		t.Fatal(e)
	}
	if _, e = r.Get(ctx, "x"); !errors.Is(e, ErrClosed) {
		t.Fatal(e)
	}
	if e = r.Close(ctx); e != nil {
		t.Fatal(e)
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
