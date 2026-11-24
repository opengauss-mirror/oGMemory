package mountablefs

import (
	"context"
	"fmt"
	"github.com/c4pt0r/agfs/agfs-server/pkg/cachefs"
	"github.com/c4pt0r/agfs/agfs-server/pkg/filesystem"
	"github.com/c4pt0r/agfs/agfs-server/pkg/plugin"
	cache "github.com/ogmemory/cache-runtime"
	"path/filepath"
)

// ConfigureCache runs before mounts; all selected mounts share one runtime.
func (mfs *MountableFS) ConfigureCache(c cachefs.Config, rc cache.Config) error {
	if err := c.Validate(); err != nil {
		return err
	}
	if !c.Enabled {
		return nil
	}
	r, err := cache.New(rc)
	if err != nil {
		return err
	}
	if err = mfs.UseCache(c, r); err != nil {
		_ = r.Close(context.Background())
		return err
	}
	mfs.ownsCache = true
	return nil
}

// UseCache injects an application-owned runtime without transferring ownership.
func (mfs *MountableFS) UseCache(c cachefs.Config, r cache.Runtime) error {
	if err := c.Validate(); err != nil {
		return err
	}
	mfs.mu.Lock()
	defer mfs.mu.Unlock()
	if mfs.closing.Load() || mfs.cacheRuntime != nil || len(mfs.GetMounts()) > 0 {
		return fmt.Errorf("configure cache before mounting")
	}
	if c.Enabled && r == nil {
		return fmt.Errorf("cachefs requires runtime")
	}
	mfs.cacheConfig = c.Defaults()
	mfs.cacheRuntime = r
	return nil
}
func allowedCachePlugin(name string) bool {
	return name == "localfs" || name == "s3fs" || name == "sqlfs"
}
func (mfs *MountableFS) PrepareMountConfig(name, mount string, c map[string]interface{}) (map[string]interface{}, error) {
	result := make(map[string]interface{}, len(c)+1)
	for k, v := range c {
		result[k] = v
	}
	result["mount_path"] = mount
	if mfs.cacheConfig.Selected(mount) {
		if !allowedCachePlugin(name) {
			return nil, fmt.Errorf("cachefs cannot wrap plugin %s", name)
		}
		if name == "s3fs" || name == "sqlfs" {
			result["cache_enabled"] = false
		}
	}
	return result, nil
}
func (mfs *MountableFS) executionFS(mount string, p plugin.ServicePlugin) (filesystem.FileSystem, *cachefs.FS, error) {
	backend := p.GetFileSystem()
	selected := mfs.cacheConfig.Selected(mount)
	if selected && !allowedCachePlugin(p.Name()) {
		return nil, nil, fmt.Errorf("cachefs cannot wrap plugin %s", p.Name())
	}
	for _, old := range mfs.GetMounts() {
		if !selected && old.cache == nil {
			continue
		}
		a, ok := backend.(interface{ CacheStorageIdentity() (string, string) })
		if !ok {
			continue
		}
		b, ok := old.Plugin.GetFileSystem().(interface{ CacheStorageIdentity() (string, string) })
		if !ok {
			continue
		}
		aid, ap := a.CacheStorageIdentity()
		bid, bp := b.CacheStorageIdentity()
		if aid == bid && aid != "" && (pathWithin(ap, bp) || pathWithin(bp, ap)) {
			return nil, nil, fmt.Errorf("cachefs storage aliases mounts %s and %s", old.Path, mount)
		}
	}
	if !selected {
		return backend, nil, nil
	}
	if disable, ok := backend.(interface{ DisableInternalCache() }); ok {
		disable.DisableInternalCache()
	}
	f, err := cachefs.New(backend, mfs.cacheRuntime, mfs.cacheConfig, mount)
	if err != nil {
		return nil, nil, err
	}
	return cachefs.Wrap(f), f, nil
}
func pathWithin(p, root string) bool {
	p = filepath.ToSlash(p)
	root = filepath.ToSlash(root)
	return root == "/" || p == root || len(p) > len(root) && p[:len(root)+1] == root+"/"
}

// ExecutionFS is the routed filesystem; Plugin is for metadata and shutdown.
func (m *MountPoint) ExecutionFS() filesystem.FileSystem {
	if m.FileSystem != nil {
		return m.FileSystem
	}
	return m.Plugin.GetFileSystem()
}
func (mfs *MountableFS) CacheStats() map[string]cachefs.Counters {
	out := map[string]cachefs.Counters{}
	for _, m := range mfs.GetMounts() {
		if m.cache != nil {
			out[m.Path] = m.cache.Stats()
		}
	}
	return out
}

type closeAttempt struct {
	done chan struct{}
	err  error
}

// Close bounds the caller's wait, not the lifetime of live native resources.
// A failed drain keeps the runtime alive; after closing streams callers may retry.
func (mfs *MountableFS) Close(ctx context.Context) error {
	mfs.closing.Store(true)
	mfs.closeMu.Lock()
	a := mfs.closeAttempt
	if a != nil {
		select {
		case <-a.done:
			if a.err != nil {
				a = nil
			}
		default:
		}
	}
	if a == nil {
		a = &closeAttempt{done: make(chan struct{})}
		mfs.closeAttempt = a
		go func() { a.err = mfs.drainCache(); close(a.done) }()
	}
	mfs.closeMu.Unlock()
	select {
	case <-a.done:
		return a.err
	case <-ctx.Done():
		return ctx.Err()
	}
}
func (mfs *MountableFS) drainCache() error {
	mfs.handleInfosMu.RLock()
	ids := make([]int64, 0, len(mfs.handleInfos))
	for id := range mfs.handleInfos {
		ids = append(ids, id)
	}
	mfs.handleInfosMu.RUnlock()
	for _, id := range ids {
		if err := mfs.CloseHandle(id); err != nil {
			return err
		}
	}
	for _, m := range mfs.GetMounts() {
		if err := mfs.Unmount(m.Path); err != nil {
			return err
		}
	}
	if mfs.ownsCache {
		return mfs.cacheRuntime.Close(context.Background())
	}
	return nil
}
