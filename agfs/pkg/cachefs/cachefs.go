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
	"slices"
	"strings"
	"sync"
	"sync/atomic"
	"time"
	"unicode/utf8"

	"github.com/c4pt0r/agfs/agfs-server/pkg/filesystem"
	cache "github.com/ogmemory/cache-runtime"
)

const maxStateEntries = 65536
const maxFlights = 1024
const maxPathBytes = 4096
const maxPathDepth = 64
const maxVersionBytes = 16 << 20
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
	done    chan struct{}
	value   envelope
	err     error
	waiters bool
	share   bool
}
type Counters struct{ Hits, Misses, Bypasses, Failures, Fills, Coalesced, Epochs uint64 }
type metrics struct{ hits, misses, bypasses, failures, fills, coalesced, epochs atomic.Uint64 }

type invalidationMode uint8

const (
	invalidateExact invalidationMode = iota
	invalidateSubtree
	invalidateSQLSubtree
)

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
	versionBytes                    int
	disabled                        bool
	sqlite                          bool
	local                           bool
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
	validator, sqlite := backend.(interface{ ValidateSQLiteCache() error })
	if sqlite {
		if err := validator.ValidateSQLiteCache(); err != nil {
			return nil, err
		}
	}
	_, local := backend.(interface{ ResolvePath(string) string })
	c = c.Defaults()
	f := &FS{backend: backend, runtime: r, cfg: c, mount: path.Clean(mount), sqlite: sqlite, local: local, epoch: token(), versions: map[string]uint64{}, flights: map[string]*flight{}, stop: make(chan struct{}), done: make(chan struct{})}
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
	f.versionBytes = 0
	f.seq++
	f.stats.epochs.Add(1)
}
func (f *FS) fail() {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.failLocked()
}
func (f *FS) failLocked() {
	f.stats.failures.Add(1)
	f.bypass = true
	f.rotateLocked()
}
func (f *FS) Close() error {
	f.ops.Lock()
	f.mu.Lock()
	if f.writers > 0 || f.readers > 0 {
		f.mu.Unlock()
		f.ops.Unlock()
		return fmt.Errorf("cachefs has active streams/handles")
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
			probe := f.bypass && !f.disabled && !f.closed && f.writers == 0 && !f.uncertainWriter
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
				if !f.disabled && !f.closed && f.writers == 0 && !f.uncertainWriter {
					f.rotateLocked()
					f.bypass = false
				}
				f.mu.Unlock()
				f.ops.Unlock()
			}
		}
	}
}

// Disable retires the cache for the lifetime of this mount. The backend remains usable.
func (f *FS) Disable() {
	f.ops.Lock()
	defer f.ops.Unlock()
	f.mu.Lock()
	defer f.mu.Unlock()
	if !f.disabled {
		f.disabled = true
		f.rotateLocked()
	}
}

func trackablePath(p string) bool {
	return len(p) <= maxPathBytes && strings.Count(p, "/") <= maxPathDepth
}

func asciiPath(p string) bool {
	for i := 0; i < len(p); i++ {
		if p[i] >= utf8.RuneSelf || p[i] == '\\' {
			return false
		}
	}
	return true
}

func (f *FS) versionPath(p string) string {
	if f.local || f.sqlite {
		// Aliases share invalidation state while the actual cache keys retain their spelling.
		return strings.ToLower(p)
	}
	return p
}

// The table only belongs to this mount and epoch. Never evict individual versions.
func (f *FS) versionLocked(p string) string {
	p = f.versionPath(p)
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

func (f *FS) setVersionLocked(key string) bool {
	if _, exists := f.versions[key]; !exists {
		if len(f.versions) >= maxStateEntries || f.versionBytes+len(key) > maxVersionBytes {
			return false
		}
		f.versionBytes += len(key)
	}
	f.versions[key] = f.seq
	return true
}

func (f *FS) invalidate(paths []string, subtree, rotate bool) {
	f.mu.Lock()
	defer f.mu.Unlock()
	if f.closed || f.disabled || f.bypass || f.uncertainWriter || f.writers > 0 {
		return
	}
	if rotate {
		f.rotateLocked()
		return
	}
	f.seq++
	keys := make([]string, 0, len(paths)*3)
	for _, p := range paths {
		versionPath := f.versionPath(p)
		if !f.setVersionLocked("exact:"+versionPath) ||
			(subtree && !f.setVersionLocked("tree:"+versionPath)) {
			f.rotateLocked()
			return
		}
		qp := path.Dir(p)
		for q := path.Dir(versionPath); ; q = path.Dir(q) {
			if !f.setVersionLocked("exact:" + q) {
				f.rotateLocked()
				return
			}
			keys = append(keys, f.keyLocked("dir", qp))
			if q == "/" {
				break
			}
			qp = path.Dir(qp)
		}
		keys = append(keys, f.keyLocked("file", p), f.keyLocked("dir", p))
	}
	slices.Sort(keys)
	keys = slices.Compact(keys)
	if _, err := f.runtime.Delete(context.Background(), keys); err != nil {
		f.failLocked()
	}
}
func (f *FS) keyLocked(kind, p string) string {
	return f.prefix + f.epoch + ":" + kind + ":" + digest(p)
}
func (f *FS) allowed(kind, p string) bool {
	if !trackablePath(p) || (f.local && !asciiPath(p)) ||
		(f.sqlite && kind == "dir" && strings.ContainsAny(p, "%_")) {
		return false
	}
	if policy(p, f.mount, f.cfg.BypassPrefixes, f.local) != Cacheable {
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
func (f *FS) load(kind, p string, read func() (envelope, error)) (value envelope, err error) {
	f.ops.RLock()
	defer f.ops.RUnlock()
	f.mu.Lock()
	closed := f.closed
	unavailable := f.disabled || f.bypass || f.writers > 0 || f.uncertainWriter
	f.mu.Unlock()
	if closed {
		return envelope{}, os.ErrClosed
	}
	if unavailable || !f.allowed(kind, p) {
		f.stats.bypasses.Add(1)
		return read()
	}
	f.mu.Lock()
	// Another read can trip bypass while this path is being checked.
	if f.bypass || f.uncertainWriter {
		f.mu.Unlock()
		f.stats.bypasses.Add(1)
		return read()
	}
	epoch := f.epoch
	version := f.versionLocked(p)
	key := f.keyLocked(kind, p)
	flightKey := fmt.Sprintf("%s:%d", key, f.seq)
	if pending := f.flights[flightKey]; pending != nil {
		pending.waiters = true
		f.mu.Unlock()
		f.stats.coalesced.Add(1)
		<-pending.done
		if !pending.share {
			return read()
		}
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

	completed := false
	share := false
	defer func() {
		f.mu.Lock()
		delete(f.flights, flightKey)
		if !completed {
			pending.share = true
			pending.err = errors.New("cachefs read did not complete")
		} else if share && pending.waiters {
			pending.share = true
			pending.value, pending.err = value, err
		}
		close(pending.done)
		f.mu.Unlock()
		if completed && pending.share {
			// Waiters and the leader must each own their result. The shared copy
			// is immutable and copies do not hold the mount state lock.
			value = clone(value)
		}
	}()
	value, err, share = f.cachedLoad(kind, p, key, epoch, version, read)
	completed = true
	return value, err
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
func (f *FS) cachedLoad(kind, p, key, epoch, version string, read func() (envelope, error)) (envelope, error, bool) {
	got, err := cache.GetNonEmpty(context.Background(), f.runtime, key, f.limit(kind))
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
					return v, io.EOF, true
				}
				return v, nil, true
			}
		}
	}
	f.stats.misses.Add(1)
	v, e := read()
	if e != nil && !errors.Is(e, io.EOF) {
		return v, e, len(v.Data) == 0 && len(v.Entries) == 0
	}
	if kind == "file" && len(v.Data) > f.cfg.MaxFileSizeBytes {
		return v, e, false
	}
	if kind == "dir" {
		if len(v.Entries) > f.cfg.MaxDirEntries {
			return v, e, false
		}
		for _, entry := range v.Entries {
			if !validEntryEncoding(entry) ||
				(len(entry.Meta.Content) > 0 && policy(path.Join(p, entry.Name), f.mount, nil, f.local) == MandatoryBypass) {
				return v, e, false
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
		return v, e, false
	}
	if kind == "dir" && len(b) > f.cfg.MaxDirEncodedBytes {
		return v, e, false
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
	return v, e, true
}
func validEntryEncoding(entry filesystem.FileInfo) bool {
	if !utf8.ValidString(entry.Name) || !utf8.ValidString(entry.Meta.Name) || !utf8.ValidString(entry.Meta.Type) {
		return false
	}
	for key, value := range entry.Meta.Content {
		if !utf8.ValidString(key) || !utf8.ValidString(value) {
			return false
		}
	}
	return true
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
func (f *FS) mutation(paths []string, mode invalidationMode, run func() error) error {
	f.ops.Lock()
	defer f.ops.Unlock()
	if f.isClosed() {
		return os.ErrClosed
	}
	rotate := false
	for i, raw := range paths {
		p := filesystem.NormalizePath(raw)
		paths[i] = p
		if !trackablePath(p) || (f.local && (!asciiPath(p) || !f.aliasSafe(p))) ||
			(f.sqlite && mode == invalidateSQLSubtree && (strings.ContainsAny(p, "%_") || !asciiPath(p))) {
			rotate = true
		}
	}
	completed := false
	defer func() { f.invalidate(paths, mode != invalidateExact, rotate || !completed) }()
	err := run()
	completed = true
	return err
}
func (f *FS) Create(p string) error {
	return f.mutation([]string{p}, invalidateExact, func() error { return f.backend.Create(p) })
}
func (f *FS) Mkdir(p string, m uint32) error {
	return f.mutation([]string{p}, invalidateSubtree, func() error { return f.backend.Mkdir(p, m) })
}
func (f *FS) Remove(p string) error {
	return f.mutation([]string{p}, invalidateSubtree, func() error { return f.backend.Remove(p) })
}
func (f *FS) RemoveAll(p string) error {
	return f.mutation([]string{p}, invalidateSQLSubtree, func() error { return f.backend.RemoveAll(p) })
}
func (f *FS) Rename(a, b string) error {
	return f.mutation([]string{a, b}, invalidateSQLSubtree, func() error { return f.backend.Rename(a, b) })
}
func (f *FS) Chmod(p string, m uint32) error {
	return f.mutation([]string{p}, invalidateSubtree, func() error { return f.backend.Chmod(p, m) })
}
func (f *FS) Write(p string, b []byte, o int64, flags filesystem.WriteFlag) (int64, error) {
	var n int64
	e := f.mutation([]string{p}, invalidateExact, func() error { var e error; n, e = f.backend.Write(p, b, o, flags); return e })
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
		r.fs.endReader()
	})
	return r.err
}
