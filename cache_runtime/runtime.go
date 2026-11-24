// Package cacheruntime supplies process-owned byte KV storage to independent consumers.
package cacheruntime

import (
	"context"
	"errors"
	"fmt"
	"sync"
	"time"
)

type Operation string

const (
	OpGet    Operation = "get"
	OpSet    Operation = "set"
	OpDelete Operation = "delete"
	OpMGet   Operation = "mget"
)

var (
	ErrClosed          = errors.New("cache runtime closed")
	ErrUnsupported     = errors.New("unsupported cache operation")
	ErrInvalidArgument = errors.New("invalid cache argument")
	ErrInvalidData     = errors.New("invalid cache data")
	ErrUnavailable     = errors.New("cache unavailable")
)

type Value struct {
	Data  []byte
	Found bool
}
type Runtime interface {
	Get(context.Context, string) (Value, error)
	Set(context.Context, string, []byte, time.Duration) error
	Delete(context.Context, []string) (uint64, error)
	MGet(context.Context, []string) ([]Value, error)
	Require(...Operation) error
	Ping(context.Context) error
	Close(context.Context) error
}

// Provider owns backend resources. The runtime bounds calls and handles shutdown.
// A timeout does not imply a mutation did not execute. Consumers must isolate late writes.
type Provider interface {
	Get(context.Context, string) (Value, error)
	Set(context.Context, string, []byte, time.Duration) error
	Delete(context.Context, []string) (uint64, error)
	Ping(context.Context) error
	Close() error
}
type Config struct {
	Provider         string                 `yaml:"provider" json:"provider"`
	CommandTimeoutMS int                    `yaml:"command_timeout_ms" json:"command_timeout_ms"`
	MaxInflight      int                    `yaml:"max_inflight" json:"max_inflight"`
	Params           map[string]interface{} `yaml:"params" json:"params"`
}

func (c Config) Validate() error {
	if c.CommandTimeoutMS < 0 || c.CommandTimeoutMS > 60000 || c.MaxInflight < 0 || c.MaxInflight > 4096 {
		return ErrInvalidArgument
	}
	switch c.Provider {
	case "redis":
		return validateRedis(c.Params)
	case "dynamic":
		return validateDynamic(c.Params)
	default:
		return fmt.Errorf("%w: provider %q", ErrUnsupported, c.Provider)
	}
}
func New(c Config) (Runtime, error) {
	if err := c.Validate(); err != nil {
		return nil, err
	}
	var p Provider
	var err error
	switch c.Provider {
	case "redis":
		p, err = newRedis(c.Params)
	case "dynamic":
		p, err = newDynamic(c.Params)
	}
	if err != nil {
		return nil, err
	}
	return WithProvider(p, c)
}

// WithProvider is the injection seam for non-AGFS consumers and contract tests.
func WithProvider(p Provider, c Config) (Runtime, error) {
	if p == nil || c.CommandTimeoutMS < 0 || c.CommandTimeoutMS > 60000 || c.MaxInflight < 0 || c.MaxInflight > 4096 {
		return nil, ErrInvalidArgument
	}
	if c.CommandTimeoutMS == 0 {
		c.CommandTimeoutMS = 50
	}
	if c.MaxInflight == 0 {
		c.MaxInflight = 32
	}
	return &managed{p: p, timeout: time.Duration(c.CommandTimeoutMS) * time.Millisecond, slots: make(chan struct{}, c.MaxInflight), done: make(chan struct{})}, nil
}

type managed struct {
	p        Provider
	timeout  time.Duration
	slots    chan struct{}
	mu       sync.Mutex
	closed   bool
	calls    sync.WaitGroup
	done     chan struct{}
	closeErr error
}
type result[T any] struct {
	value T
	err   error
}

func call[T any](r *managed, ctx context.Context, f func(context.Context) (T, error)) (T, error) {
	var zero T
	ctx, cancel := context.WithTimeout(ctx, r.timeout)
	defer cancel()
	r.mu.Lock()
	if r.closed {
		r.mu.Unlock()
		return zero, ErrClosed
	}
	r.calls.Add(1)
	r.mu.Unlock()
	select {
	case r.slots <- struct{}{}:
	case <-ctx.Done():
		r.calls.Done()
		return zero, ctx.Err()
	}
	ch := make(chan result[T], 1)
	go func() { defer r.calls.Done(); defer func() { <-r.slots }(); v, e := f(ctx); ch <- result[T]{v, e} }()
	select {
	case v := <-ch:
		return v.value, v.err
	case <-ctx.Done():
		return zero, ctx.Err()
	}
}
func (r *managed) Require(ops ...Operation) error {
	for _, op := range ops {
		switch op {
		case OpGet, OpSet, OpDelete, OpMGet:
		default:
			return fmt.Errorf("%w: %s", ErrUnsupported, op)
		}
	}
	return nil
}
func (r *managed) Get(ctx context.Context, k string) (Value, error) {
	return call(r, ctx, func(c context.Context) (Value, error) { return r.p.Get(c, k) })
}
func (r *managed) Set(ctx context.Context, k string, v []byte, ttl time.Duration) error {
	if ttl < 0 {
		return ErrInvalidArgument
	}
	data := append([]byte(nil), v...)
	_, e := call(r, ctx, func(c context.Context) (struct{}, error) { return struct{}{}, r.p.Set(c, k, data, ttl) })
	return e
}
func (r *managed) Delete(ctx context.Context, keys []string) (uint64, error) {
	ks := append([]string(nil), keys...)
	return call(r, ctx, func(c context.Context) (uint64, error) { return r.p.Delete(c, ks) })
}
func (r *managed) MGet(ctx context.Context, keys []string) ([]Value, error) {
	ks := append([]string(nil), keys...)
	return call(r, ctx, func(c context.Context) ([]Value, error) {
		values := make([]Value, len(ks))
		for i, k := range ks {
			if e := c.Err(); e != nil {
				return nil, e
			}
			v, e := r.p.Get(c, k)
			if e != nil {
				return nil, e
			}
			values[i] = v
		}
		return values, nil
	})
}
func (r *managed) Ping(ctx context.Context) error {
	_, e := call(r, ctx, func(c context.Context) (struct{}, error) { return struct{}{}, r.p.Ping(c) })
	return e
}
func (r *managed) Close(ctx context.Context) error {
	r.mu.Lock()
	if !r.closed {
		r.closed = true
		go func() { r.calls.Wait(); r.closeErr = r.p.Close(); close(r.done) }()
	}
	r.mu.Unlock()
	select {
	case <-r.done:
		return r.closeErr
	case <-ctx.Done():
		return ctx.Err()
	}
}
