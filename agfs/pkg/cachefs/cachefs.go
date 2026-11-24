// Package cachefs caches full small-file reads and raw directory listings for a
// single-writer mount. All backend mutations must pass through this wrapper.
package cachefs

import (
	"bytes"
	"context"
	"crypto/rand"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"path"
	"path/filepath"
	"reflect"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/c4pt0r/agfs/agfs-server/pkg/filesystem"
	cache "github.com/ogmemory/cache-runtime"
)

const maxStateEntries = 65536
const maxFlights = 1024
const magic = "OGCF\x01"

type envelope struct {
	Path    string
	Version string
	Expires int64
	EOF     bool
	Data    []byte
	Entries []filesystem.FileInfo
}
type flight struct {
	done  chan struct{}
	value envelope
	err   error
}
type Counters struct{ Hits, Misses, Bypasses, Failures, Fills, Coalesced, Epochs uint64 }
type metrics struct{ hits, misses, bypasses, failures, fills, coalesced, epochs atomic.Uint64 }

// FS owns only mount-local state. Closing it never closes the shared Runtime.
type FS struct {
	backend                         filesystem.FileSystem
	runtime                         cache.Runtime
	cfg                             Config
	mount, prefix                   string
	ops                             sync.RWMutex // reads/load/fill vs every mutation and writer activation
	mu                              sync.Mutex
	epoch                           string
	seq                             uint64
	versions                        map[string]uint64
	flights                         map[string]*flight
	bypass, closed, uncertainWriter bool
	writers, readers                int
	stop                            chan struct{}
	done                            chan struct{}
	stats                           metrics
}

func token() string {
	var b [16]byte
	if _, e := rand.Read(b[:]); e != nil {
		panic(e)
	}
	return hex.EncodeToString(b[:])
}
func digest(s string) string { h := sha256.Sum256([]byte(s)); return hex.EncodeToString(h[:]) }
func New(backend filesystem.FileSystem, r cache.Runtime, c Config, mount string) (*FS, error) {
	if backend == nil || r == nil {
		return nil, fmt.Errorf("cachefs requires backend and runtime")
	}
	if err := c.Validate(); err != nil {
		return nil, err
	}
	if err := r.Require(cache.OpGet, cache.OpSet, cache.OpDelete, cache.OpMGet); err != nil {
		return nil, err
	}
	c = c.Defaults()
	f := &FS{backend: backend, runtime: r, cfg: c, mount: path.Clean(mount), epoch: token(), versions: map[string]uint64{}, flights: map[string]*flight{}, stop: make(chan struct{}), done: make(chan struct{})}
	f.prefix = "ogmem:cachefs:v1:" + digest(c.Namespace) + ":" + digest(f.mount) + ":"
	go f.recoverLoop()
	return f, nil
}
func (f *FS) Stats() Counters {
	return Counters{f.stats.hits.Load(), f.stats.misses.Load(), f.stats.bypasses.Load(), f.stats.failures.Load(), f.stats.fills.Load(), f.stats.coalesced.Load(), f.stats.epochs.Load()}
}
func (f *FS) rotateLocked() {
	f.epoch = token()
	f.versions = map[string]uint64{}
	f.seq++
	f.stats.epochs.Add(1)
}
func (f *FS) fail() {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.stats.failures.Add(1)
	f.bypass = true
	f.rotateLocked()
}
func (f *FS) Close() error {
	f.ops.Lock()
	f.mu.Lock()
	if f.writers > 0 || f.readers > 0 || f.uncertainWriter {
		f.mu.Unlock()
		f.ops.Unlock()
		return fmt.Errorf("cachefs has active or unconfirmed streams/handles")
	}
	if !f.closed {
		f.closed = true
		close(f.stop)
		f.rotateLocked()
	}
	f.mu.Unlock()
	f.ops.Unlock()
	<-f.done
	return nil
}
func (f *FS) recoverLoop() {
	defer close(f.done)
	ticker := time.NewTicker(time.Second)
	defer ticker.Stop()
	for {
		select {
		case <-f.stop:
			return
		case <-ticker.C:
			f.mu.Lock()
			probe := f.bypass && !f.closed && f.writers == 0 && !f.uncertainWriter
			f.mu.Unlock()
			if !probe {
				continue
			}
			ctx, cancel := context.WithTimeout(context.Background(), time.Second)
			key := f.prefix + "probe:" + token()
			v := []byte("probe")
			err := f.runtime.Ping(ctx)
			if err == nil {
				err = f.runtime.Set(ctx, key, v, time.Second)
			}
			if err == nil {
				var got cache.Value
				got, err = f.runtime.Get(ctx, key)
				if err == nil && (!got.Found || !bytes.Equal(got.Data, v)) {
					err = cache.ErrInvalidData
				}
			}
			if err == nil {
				_, err = f.runtime.Delete(ctx, []string{key})
			}
			cancel()
			if err == nil {
				f.ops.Lock()
				f.mu.Lock()
				if !f.closed && f.writers == 0 && !f.uncertainWriter {
					f.rotateLocked()
					f.bypass = false
				}
				f.mu.Unlock()
				f.ops.Unlock()
			}
		}
	}
}

// Each mutation advances exact-object versions, plus subtree versions where
// needed. This authoritative in-process table is valid only for the single
// writer and current random epoch; capacity rollover discards the entire epoch.
func (f *FS) versionLocked(p string) string {
	var b strings.Builder
	fmt.Fprintf(&b, "%d", f.versions["exact:"+p])
	for q := p; ; q = path.Dir(q) {
		fmt.Fprintf(&b, "/%d", f.versions["tree:"+q])
		if q == "/" {
			break
		}
	}
	return digest(b.String())
}
func (f *FS) invalidate(paths []string, subtree bool) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.seq++
	keys := []string{}
	for _, p := range paths {
		p = filesystem.NormalizePath(p)
		f.versions["exact:"+p] = f.seq
		// Ancestor directory entries can change (mtime or S3 implicit prefixes).
		for q := path.Dir(p); ; q = path.Dir(q) {
			f.versions["exact:"+q] = f.seq
			keys = append(keys, f.keyLocked("dir", q))
			if q == "/" {
				break
			}
		}
		if subtree {
			f.versions["tree:"+p] = f.seq
		}
		keys = append(keys, f.keyLocked("file", p), f.keyLocked("dir", p))
	}
	if len(f.versions) > maxStateEntries {
		f.rotateLocked()
		return
	}
	if f.closed || f.bypass || f.writers > 0 {
		return
	}
	if _, err := f.runtime.Delete(context.Background(), keys); err != nil {
		f.bypass = true
		f.rotateLocked()
		f.stats.failures.Add(1)
	}
}
func (f *FS) keyLocked(kind, p string) string {
	return f.prefix + f.epoch + ":" + kind + ":" + digest(p)
}
func (f *FS) allowed(p string) bool {
	if Policy(p, f.mount, f.cfg.BypassPrefixes) != Cacheable {
		return false
	}
	if c, ok := f.backend.(filesystem.CapabilityProvider); ok {
		caps := c.GetPathCapabilities(p)
		if caps.IsReadDestructive || caps.IsBroadcast {
			return false
		}
	}
	return f.aliasSafe(p)
}

// Local aliases cannot have independently cached names: bypass symlinks and
// multiply-linked regular files. External writes remain outside the topology.
func (f *FS) aliasSafe(p string) bool {
	resolver, ok := f.backend.(interface{ ResolvePath(string) string })
	if !ok {
		return true
	}
	root := resolver.ResolvePath("/")
	target := resolver.ResolvePath(p)
	for q := target; ; q = filepath.Dir(q) {
		info, err := os.Lstat(q)
		if err != nil {
			if !os.IsNotExist(err) {
				return false
			}
		} else {
			if info.Mode()&os.ModeSymlink != 0 {
				return false
			}
			if info.Mode().IsRegular() {
				v := reflect.ValueOf(info.Sys())
				if v.Kind() == reflect.Pointer && !v.IsNil() {
					n := v.Elem().FieldByName("Nlink")
					if n.IsValid() && n.CanUint() && n.Uint() > 1 {
						return false
					}
				}
			}
		}
		if q == root {
			break
		}
		if q == filepath.Dir(q) {
			return false
		}
	}
	return true
}
func (f *FS) ResolvePath(p string) string {
	if f.cfg.TraversalMode != "backend" {
		return ""
	}
	if r, ok := f.backend.(interface{ ResolvePath(string) string }); ok {
		return r.ResolvePath(p)
	}
	return ""
}
func (f *FS) load(kind, p string, read func() (envelope, error)) (envelope, error) {
	f.ops.RLock()
	defer f.ops.RUnlock()
	if f.isClosed() {
		return envelope{}, os.ErrClosed
	}
	if !f.allowed(p) {
		f.stats.bypasses.Add(1)
		return read()
	}
	f.mu.Lock()
	if f.closed || f.bypass || f.writers > 0 || f.uncertainWriter {
		f.mu.Unlock()
		f.stats.bypasses.Add(1)
		return read()
	}
	epoch := f.epoch
	version := f.versionLocked(p)
	key := f.keyLocked(kind, p)
	flightKey := fmt.Sprintf("%s:%d", key, f.seq)
	if pending := f.flights[flightKey]; pending != nil {
		f.mu.Unlock()
		f.stats.coalesced.Add(1)
		<-pending.done
		return clone(pending.value), pending.err
	}
	if len(f.flights) >= maxFlights {
		f.mu.Unlock()
		f.stats.bypasses.Add(1)
		return read()
	}
	pending := &flight{done: make(chan struct{})}
	f.flights[flightKey] = pending
	f.mu.Unlock()
	v, err := f.cachedLoad(kind, p, key, epoch, version, read)
	f.mu.Lock()
	pending.value = clone(v)
	pending.err = err
	delete(f.flights, flightKey)
	close(pending.done)
	f.mu.Unlock()
	return v, err
}
func clone(v envelope) envelope {
	v.Data = bytes.Clone(v.Data)
	if v.Entries != nil {
		v.Entries = append([]filesystem.FileInfo{}, v.Entries...)
		for i := range v.Entries {
			m := v.Entries[i].Meta.Content
			if m != nil {
				n := make(map[string]string, len(m))
				for k, v := range m {
					n[k] = v
				}
				v.Entries[i].Meta.Content = n
			}
		}
	}
	return v
}
func (f *FS) limit(kind string) int {
	if kind == "file" {
		return f.cfg.MaxFileSizeBytes*2 + 8192
	}
	return f.cfg.MaxDirEncodedBytes + 8192
}
func (f *FS) cachedLoad(kind, p, key, epoch, version string, read func() (envelope, error)) (envelope, error) {
	got, err := f.runtime.Get(context.Background(), key)
	if err != nil {
		f.fail()
	} else if got.Found {
		var v envelope
		if len(got.Data) <= f.limit(kind) && bytes.HasPrefix(got.Data, []byte(magic)) && json.Unmarshal(got.Data[len(magic):], &v) == nil && v.Path == p && v.Version == version && v.Expires > time.Now().UnixNano() && ((kind == "file" && len(v.Data) <= f.cfg.MaxFileSizeBytes) || (kind == "dir" && len(v.Entries) <= f.cfg.MaxDirEntries)) {
			f.mu.Lock()
			valid := !f.bypass && !f.closed && f.epoch == epoch
			f.mu.Unlock()
			if valid {
				f.stats.hits.Add(1)
				if v.EOF {
					return v, io.EOF
				}
				return v, nil
			}
		}
	}
	f.stats.misses.Add(1)
	v, e := read()
	if e != nil && !errors.Is(e, io.EOF) {
		return v, e
	}
	if kind == "file" && len(v.Data) > f.cfg.MaxFileSizeBytes {
		return v, e
	}
	if kind == "dir" {
		if len(v.Entries) > f.cfg.MaxDirEntries {
			return v, e
		}
		for _, entry := range v.Entries {
			if len(entry.Meta.Content) > 0 && Policy(path.Join(p, entry.Name), f.mount, nil) == MandatoryBypass {
				return v, e
			}
		}
	}
	ttl := time.Duration(f.cfg.FileTTLSeconds) * time.Second
	if kind == "dir" {
		ttl = time.Duration(f.cfg.DirTTLSeconds) * time.Second
	}
	v.Path = p
	v.Version = version
	v.Expires = time.Now().Add(ttl).UnixNano()
	v.EOF = errors.Is(e, io.EOF)
	b, encErr := json.Marshal(v)
	if encErr != nil {
		return v, e
	}
	if kind == "dir" && len(b) > f.cfg.MaxDirEncodedBytes {
		return v, e
	}
	f.mu.Lock()
	active := !f.closed && !f.bypass && f.epoch == epoch && f.writers == 0
	f.mu.Unlock()
	if active {
		if err = f.runtime.Set(context.Background(), key, append([]byte(magic), b...), ttl); err != nil {
			f.fail()
		} else {
			f.stats.fills.Add(1)
		}
	}
	return v, e
}
func (f *FS) Read(p string, offset, size int64) ([]byte, error) {
	p = filesystem.NormalizePath(p)
	if offset != 0 || size != -1 {
		f.ops.RLock()
		defer f.ops.RUnlock()
		if f.isClosed() {
			return nil, os.ErrClosed
		}
		return f.backend.Read(p, offset, size)
	}
	v, e := f.load("file", p, func() (envelope, error) { b, e := f.backend.Read(p, offset, size); return envelope{Data: b}, e })
	return v.Data, e
}
func (f *FS) ReadDir(p string) ([]filesystem.FileInfo, error) {
	p = filesystem.NormalizePath(p)
	v, e := f.load("dir", p, func() (envelope, error) { entries, e := f.backend.ReadDir(p); return envelope{Entries: entries}, e })
	return v.Entries, e
}

// ReadDirFresh preserves the caller's request to bypass directory caches.
func (f *FS) ReadDirFresh(p string) ([]filesystem.FileInfo, error) {
	f.ops.RLock()
	defer f.ops.RUnlock()
	if f.isClosed() {
		return nil, os.ErrClosed
	}
	f.stats.bypasses.Add(1)
	if b, ok := f.backend.(interface {
		ReadDirFresh(string) ([]filesystem.FileInfo, error)
	}); ok {
		return b.ReadDirFresh(p)
	}
	return f.backend.ReadDir(p)
}
func (f *FS) mutation(paths []string, subtree bool, run func() error) error {
	f.ops.Lock()
	defer f.ops.Unlock()
	if f.isClosed() {
		return os.ErrClosed
	}
	unsafe := false
	for _, p := range paths {
		if !f.aliasSafe(filesystem.NormalizePath(p)) {
			unsafe = true
		}
	}
	e := run()
	f.invalidate(paths, subtree)
	if unsafe {
		f.mu.Lock()
		f.rotateLocked()
		f.mu.Unlock()
	}
	return e
}
func (f *FS) Create(p string) error {
	return f.mutation([]string{p}, false, func() error { return f.backend.Create(p) })
}
func (f *FS) Mkdir(p string, m uint32) error {
	return f.mutation([]string{p}, true, func() error { return f.backend.Mkdir(p, m) })
}
func (f *FS) Remove(p string) error {
	return f.mutation([]string{p}, true, func() error { return f.backend.Remove(p) })
}
func (f *FS) RemoveAll(p string) error {
	return f.mutation([]string{p}, true, func() error { return f.backend.RemoveAll(p) })
}
func (f *FS) Rename(a, b string) error {
	return f.mutation([]string{a, b}, true, func() error { return f.backend.Rename(a, b) })
}
func (f *FS) Chmod(p string, m uint32) error {
	return f.mutation([]string{p}, true, func() error { return f.backend.Chmod(p, m) })
}
func (f *FS) Write(p string, b []byte, o int64, flags filesystem.WriteFlag) (int64, error) {
	var n int64
	e := f.mutation([]string{p}, false, func() error { var e error; n, e = f.backend.Write(p, b, o, flags); return e })
	return n, e
}
func (f *FS) Stat(p string) (*filesystem.FileInfo, error) {
	f.ops.RLock()
	defer f.ops.RUnlock()
	if f.isClosed() {
		return nil, os.ErrClosed
	}
	return f.backend.Stat(p)
}
func (f *FS) Open(p string) (io.ReadCloser, error) {
	f.ops.RLock()
	defer f.ops.RUnlock()
	if f.isClosed() {
		return nil, os.ErrClosed
	}
	r, e := f.backend.Open(p)
	if e != nil {
		return nil, e
	}
	f.addReader()
	return &reader{ReadCloser: r, fs: f}, nil
}
func (f *FS) startWriter() { f.mu.Lock(); f.writers++; f.rotateLocked(); f.mu.Unlock() }

// BeginScopedWrite bypasses the cache while a pinned backend session can write.
func (f *FS) BeginScopedWrite() (func(), error) {
	f.ops.Lock()
	defer f.ops.Unlock()
	if f.isClosed() {
		return nil, os.ErrClosed
	}
	f.startWriter()
	return sync.OnceFunc(func() { f.endWriter(nil) }), nil
}

func (f *FS) endWriter(err error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.writers--
	if err != nil {
		f.uncertainWriter = true
	}
	f.rotateLocked()
}
func (f *FS) OpenWrite(p string) (io.WriteCloser, error) {
	f.ops.Lock()
	defer f.ops.Unlock()
	if f.isClosed() {
		return nil, os.ErrClosed
	}
	f.startWriter()
	w, e := f.backend.OpenWrite(p)
	if e != nil {
		f.endWriter(nil)
		return nil, e
	}
	return &writer{WriteCloser: w, fs: f}, nil
}

type writer struct {
	io.WriteCloser
	fs   *FS
	once sync.Once
	err  error
}

func (w *writer) Close() error {
	w.once.Do(func() { w.err = w.WriteCloser.Close(); w.fs.endWriter(w.err) })
	return w.err
}
func (f *FS) GetCapabilities() filesystem.Capabilities {
	if c, ok := f.backend.(filesystem.CapabilityProvider); ok {
		return c.GetCapabilities()
	}
	return filesystem.DefaultCapabilities()
}
func (f *FS) GetPathCapabilities(p string) filesystem.Capabilities {
	if c, ok := f.backend.(filesystem.CapabilityProvider); ok {
		return c.GetPathCapabilities(p)
	}
	return f.GetCapabilities()
}

func (f *FS) isClosed() bool { f.mu.Lock(); defer f.mu.Unlock(); return f.closed }
func (f *FS) addReader()     { f.mu.Lock(); f.readers++; f.mu.Unlock() }
func (f *FS) endReader()     { f.mu.Lock(); f.readers--; f.mu.Unlock() }

type reader struct {
	io.ReadCloser
	fs   *FS
	once sync.Once
	err  error
}

func (r *reader) Close() error {
	r.once.Do(func() {
		r.err = r.ReadCloser.Close()
		if r.err == nil {
			r.fs.endReader()
		}
	})
	return r.err
}
