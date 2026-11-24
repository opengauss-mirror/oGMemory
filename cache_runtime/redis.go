package cacheruntime

import (
	"context"
	"encoding/json"
	"fmt"
	"github.com/redis/go-redis/v9"
	"os"
	"time"
)

type redisParams struct {
	Mode             string   `json:"mode"`
	Endpoints        []string `json:"endpoints"`
	PoolSize         int      `json:"pool_size"`
	ConnectTimeoutMS int      `json:"connect_timeout_ms"`
	PasswordEnv      string   `json:"password_env"`
}

func parseRedis(params map[string]interface{}) (redisParams, error) {
	var p redisParams
	b, e := json.Marshal(params)
	if e != nil {
		return p, ErrInvalidArgument
	}
	if e = json.Unmarshal(b, &p); e != nil {
		return p, ErrInvalidArgument
	}
	if p.Mode != "" && p.Mode != "standalone" {
		return p, fmt.Errorf("%w: only Redis standalone is supported", ErrUnsupported)
	}
	if len(p.Endpoints) != 1 || p.PoolSize < 0 || p.PoolSize > 4096 || p.ConnectTimeoutMS < 0 || p.ConnectTimeoutMS > 60000 {
		return p, ErrInvalidArgument
	}
	return p, nil
}
func validateRedis(params map[string]interface{}) error {
	p, e := parseRedis(params)
	if e != nil {
		return e
	}
	_, e = redis.ParseURL(p.Endpoints[0])
	if e != nil {
		return fmt.Errorf("%w: Redis endpoint", ErrInvalidArgument)
	}
	return nil
}

type redisProvider struct{ client *redis.Client }

func newRedis(params map[string]interface{}) (Provider, error) {
	p, e := parseRedis(params)
	if e != nil {
		return nil, e
	}
	o, e := redis.ParseURL(p.Endpoints[0])
	if e != nil {
		return nil, ErrInvalidArgument
	}
	if p.PoolSize == 0 {
		p.PoolSize = 32
	}
	if p.ConnectTimeoutMS == 0 {
		p.ConnectTimeoutMS = 1000
	}
	o.PoolSize = p.PoolSize
	o.DialTimeout = time.Duration(p.ConnectTimeoutMS) * time.Millisecond
	o.ContextTimeoutEnabled = true
	o.MaxRetries = -1
	if p.PasswordEnv != "" {
		v, ok := os.LookupEnv(p.PasswordEnv)
		if !ok {
			return nil, fmt.Errorf("%w: Redis password environment variable not set", ErrInvalidArgument)
		}
		o.Password = v
	}
	return &redisProvider{redis.NewClient(o)}, nil
}
func (r *redisProvider) Get(c context.Context, k string) (Value, error) {
	v, e := r.client.Get(c, k).Bytes()
	if e == redis.Nil {
		return Value{}, nil
	}
	if e != nil {
		return Value{}, e
	}
	return Value{v, true}, nil
}
func (r *redisProvider) getNonEmpty(c context.Context, k string, maxBytes int) (Value, error) {
	// Redis range endpoints are inclusive. One extra byte detects an
	// oversized value without fetching or returning the complete value.
	v, err := r.client.GetRange(c, k, 0, int64(maxBytes)).Bytes()
	if err == redis.Nil {
		return Value{}, nil
	}
	return nonEmpty(Value{Data: v, Found: true}, err, maxBytes)
}
func (r *redisProvider) Set(c context.Context, k string, v []byte, t time.Duration) error {
	return r.client.Set(c, k, v, t).Err()
}
func (r *redisProvider) Delete(c context.Context, keys []string) (uint64, error) {
	if len(keys) == 0 {
		return 0, nil
	}
	n, e := r.client.Del(c, keys...).Result()
	return uint64(n), e
}
func (r *redisProvider) Ping(c context.Context) error { return r.client.Ping(c).Err() }
func (r *redisProvider) Close() error                 { return r.client.Close() }
