package cachefs

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"github.com/c4pt0r/agfs/agfs-server/pkg/filesystem"
	"github.com/c4pt0r/agfs/agfs-server/pkg/plugins/localfs"
	"github.com/c4pt0r/agfs/agfs-server/pkg/plugins/memfs"
	cache "github.com/ogmemory/cache-runtime"
	"io"
	"os"
	"path/filepath"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

type store struct {
	mu                           sync.Mutex
	data                         map[string][]byte
	gets, sets                   int
	failGet, failSet, failDelete bool
}

func (s *store) Get(_ context.Context, k string) (cache.Value, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.gets++
	if s.failGet {
		return cache.Value{}, errors.New("down")
	}
	v, ok := s.data[k]
	return cache.Value{Data: append([]byte(nil), v...), Found: ok}, nil
}
func (s *store) Set(_ context.Context, k string, v []byte, _ time.Duration) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.sets++
	if s.failSet {
		return errors.New("down")
	}
	s.data[k] = append([]byte(nil), v...)
	return nil
}
func (s *store) Delete(_ context.Context, ks []string) (uint64, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.failDelete {
		return 0, errors.New("down")
	}
	var n uint64
	for _, k := range ks {
		if _, ok := s.data[k]; ok {
			n++
			delete(s.data, k)
		}
	}
	return n, nil
}
func (s *store) Ping(context.Context) error { return nil }
func (s *store) Close() error               { return nil }

type counted struct {
	filesystem.FileSystem
	reads, dirs    atomic.Int64
	block, started chan struct{}
	startOnce      sync.Once
	writeFail      bool
}

func (c *counted) Read(p string, o, n int64) ([]byte, error) {
	c.reads.Add(1)
	if c.block != nil {
		c.startOnce.Do(func() { close(c.started) })
		<-c.block
	}
	return c.FileSystem.Read(p, o, n)
}
func (c *counted) ReadDir(p string) ([]filesystem.FileInfo, error) {
	c.dirs.Add(1)
	return c.FileSystem.ReadDir(p)
}
func (c *counted) Write(p string, b []byte, o int64, flags filesystem.WriteFlag) (int64, error) {
	n, e := c.FileSystem.Write(p, b, o, flags)
	if c.writeFail {
		return n, errors.New("partial result")
	}
	return n, e
}
func fixture(t *testing.T) (*FS, *counted, *store, string) {
	t.Helper()
	root := t.TempDir()
	local, e := localfs.NewLocalFS(root)
	if e != nil {
		t.Fatal(e)
	}
	b := &counted{FileSystem: local}
	s := &store{data: map[string][]byte{}}
	r, e := cache.WithProvider(s, cache.Config{CommandTimeoutMS: 500})
	if e != nil {
		t.Fatal(e)
	}
	f, e := New(b, r, Config{Enabled: true, Namespace: "test", Mounts: []string{"/local"}}, "/local")
	if e != nil {
		t.Fatal(e)
	}
	t.Cleanup(func() {
		if e := f.Close(); e != nil {
			t.Error(e)
		}
		_ = r.Close(context.Background())
	})
	return f, b, s, root
}
func write(t *testing.T, f filesystem.FileSystem, p, v string) {
	t.Helper()
	if _, e := f.Write(p, []byte(v), -1, filesystem.WriteFlagCreate|filesystem.WriteFlagTruncate); e != nil {
		t.Fatal(e)
	}
}
func read(t *testing.T, f filesystem.FileSystem, p, want string) {
	t.Helper()
	b, e := f.Read(p, 0, -1)
	if e != nil && !errors.Is(e, io.EOF) {
		t.Error(e)
		return
	}
	if string(b) != want {
		t.Errorf("%s: got %q want %q", p, b, want)
	}
}
func TestReadMutations(t *testing.T) {
	f, b, _, _ := fixture(t)
	write(t, f, "/.meta.json", "one")
	read(t, f, "/.meta.json", "one")
	read(t, f, "/.meta.json", "one")
	if b.reads.Load() != 1 {
		t.Fatal("no hit")
	}
	data, e := f.Read("/.meta.json", 0, -1)
	if e != io.EOF {
		t.Fatal("lost EOF")
	}
	data[0] = 'X'
	read(t, f, "/.meta.json", "one")
	if _, e := f.Write("/.meta.json", []byte("!"), -1, filesystem.WriteFlagAppend); e != nil {
		t.Fatal(e)
	}
	read(t, f, "/.meta.json", "one!")
	if _, e := f.Write("/.meta.json", []byte("T"), 0, filesystem.WriteFlagNone); e != nil {
		t.Fatal(e)
	}
	read(t, f, "/.meta.json", "Tne!")
	b.writeFail = true
	if _, e := f.Write("/.meta.json", []byte("new"), -1, filesystem.WriteFlagTruncate); e == nil {
		t.Fatal("expected error")
	}
	b.writeFail = false
	read(t, f, "/.meta.json", "new")
	if e = f.Remove("/.meta.json"); e != nil {
		t.Fatal(e)
	}
	if _, e = f.Read("/.meta.json", 0, -1); e == nil || e == io.EOF {
		t.Fatal("stale after remove")
	}
	write(t, f, "/.meta.json", "")
	read(t, f, "/.meta.json", "")
	read(t, f, "/.meta.json", "")
}
func TestDirectoryAndSubtree(t *testing.T) {
	f, b, _, _ := fixture(t)
	if e := f.Mkdir("/src", 0755); e != nil {
		t.Fatal(e)
	}
	write(t, f, "/src/a", "old")
	read(t, f, "/src/a", "old")
	a, e := f.ReadDir("/src")
	if e != nil {
		t.Fatal(e)
	}
	a[0].Name = "bad"
	again, _ := f.ReadDir("/src")
	if again[0].Name != "a" || b.dirs.Load() != 1 {
		t.Fatal("directory alias or miss")
	}
	write(t, f, "/src/file.lock", "lock")
	a, _ = f.ReadDir("/src")
	if len(a) != 2 {
		t.Fatal("bypass write left stale parent")
	}
	if e = f.Rename("/src", "/dst"); e != nil {
		t.Fatal(e)
	}
	if _, e = f.Read("/src/a", 0, -1); e == nil || e == io.EOF {
		t.Fatal("stale source")
	}
	read(t, f, "/dst/a", "old")
	if e = f.RemoveAll("/dst"); e != nil {
		t.Fatal(e)
	}
	if e = f.Mkdir("/dst", 0755); e != nil {
		t.Fatal(e)
	}
	write(t, f, "/dst/a", "new")
	read(t, f, "/dst/a", "new")
	if e = f.Mkdir("/.n.create-123", 0755); e != nil {
		t.Fatal(e)
	}
	write(t, f, "/.n.create-123/content.md", "published")
	if e = f.Rename("/.n.create-123", "/final"); e != nil {
		t.Fatal(e)
	}
	read(t, f, "/final/content.md", "published")
	read(t, f, "/final/content.md", "published")
}
func TestPolicy(t *testing.T) {
	cases := map[string]Decision{
		"/a/.node.resource-mutation-lock/lease-x.json": MandatoryBypass, "/a/.node.update-lock-abc/content.md": MandatoryBypass, "/a/file.lock/child": MandatoryBypass, "/.path.ovlock": MandatoryBypass,
		"/a/.n.resource-create-ab/content.md": DefaultBypass, "/a/.meta.update-abc.json": DefaultBypass, "/a/.delete_tombstone.mutation-abc.json": DefaultBypass, "/a/.outbox/x.processing": DefaultBypass,
		"/plugin/accounts/a/_system/code_workspace_sync/b.json.mutation-token":        DefaultBypass,
		"/plugin/accounts/a/user/u/agents/g/sessions/s/tool-results/id/_writing.json": DefaultBypass,
		"/plugin/.ogmemory/trace-store/accounts/hash/candidates/x.json":               MandatoryBypass,
		"/plugin/.ogmemory/trace-store/accounts/hash/candidate_work/x.json":           DefaultBypass,
		"/plugin/.ogmemory/trace-store/accounts/hash/state.json":                      MandatoryBypass,
		"/a/.meta.json": Cacheable, "/a/.delete_tombstone.json": Cacheable, "/a/.relations.json": Cacheable,
		"/plugin/accounts/a/sessions/s/state.json/content.md": Cacheable, "/plugin/accounts/a/sessions/s/stage_results/id/content.md": Cacheable,
		"/plugin/accounts/a/sessions": Cacheable, "/plugin/accounts": Cacheable,
		"/plugin/.ogmemory/trace-store/accounts/hash/events/s/id.json": Cacheable,
		"/a/.outbox/1.json": Cacheable, "/a/.outbox-done/1.json": Cacheable, "/a/.write_operations/id/content.md": Cacheable, "/business/clock.json": Cacheable,
		"/plugin/accounts/a/user/u/agents/g/sessions/s/tool-results/id/metadata.json": Cacheable,
	}
	for p, want := range cases {
		for _, prefix := range []string{"", "/another-root"} {
			if got := Policy(prefix+p, "/local", nil); got != want {
				t.Errorf("%s: %s want %s", prefix+p, got, want)
			}
		}
	}
	if Policy("/foo", "/local", []string{"/local/foo"}) != UserBypass || Policy("/foobar", "/local", []string{"/local/foo"}) != Cacheable {
		t.Fatal("prefix boundary")
	}
}
func TestBypass(t *testing.T) {
	f, b, s, _ := fixture(t)
	for _, p := range []string{"/file.lock", "/.meta.update-t.json"} {
		write(t, f, p, "x")
		s.mu.Lock()
		gets, sets := s.gets, s.sets
		s.mu.Unlock()
		read(t, f, p, "x")
		read(t, f, p, "x")
		s.mu.Lock()
		if s.gets != gets || s.sets != sets {
			t.Error("bypass contacted provider")
		}
		s.mu.Unlock()
	}
	if b.reads.Load() != 4 {
		t.Fatal("bypass cached")
	}
}
func TestFailureRecoveryRestart(t *testing.T) {
	f, b, s, _ := fixture(t)
	write(t, f, "/x", "old")
	read(t, f, "/x", "old")
	s.mu.Lock()
	s.failDelete = true
	s.mu.Unlock()
	write(t, f, "/x", "new")
	read(t, f, "/x", "new")
	read(t, f, "/x", "new")
	s.mu.Lock()
	s.failDelete = false
	s.mu.Unlock()
	deadline := time.Now().Add(3 * time.Second)
	for {
		f.mu.Lock()
		active := !f.bypass
		f.mu.Unlock()
		if active {
			break
		}
		if time.Now().After(deadline) {
			t.Fatal("did not recover")
		}
		time.Sleep(10 * time.Millisecond)
	}
	read(t, f, "/x", "new")
	before := b.reads.Load()
	read(t, f, "/x", "new")
	if b.reads.Load() != before {
		t.Fatal("not caching after recovery")
	}
	next, e := New(b, f.runtime, f.cfg, f.mount)
	if e != nil {
		t.Fatal(e)
	}
	defer next.Close()
	read(t, next, "/x", "new")
	if b.reads.Load() != before+1 {
		t.Fatal("reused epoch")
	}
}
func TestSingleflightWriterOrdering(t *testing.T) {
	f, b, _, _ := fixture(t)
	write(t, f, "/x", "old")
	b.block = make(chan struct{})
	b.started = make(chan struct{})
	var wg sync.WaitGroup
	for i := 0; i < 12; i++ {
		wg.Add(1)
		go func() { defer wg.Done(); read(t, f, "/x", "old") }()
	}
	<-b.started
	deadline := time.Now().Add(time.Second)
	for f.Stats().Coalesced < 11 {
		if time.Now().After(deadline) {
			close(b.block)
			wg.Wait()
			t.Fatal("not coalesced")
		}
		time.Sleep(time.Millisecond)
	}
	done := make(chan struct{})
	go func() { write(t, f, "/x", "new"); close(done) }()
	close(b.block)
	wg.Wait()
	<-done
	read(t, f, "/x", "new")
	if b.reads.Load() != 2 {
		t.Fatal("extra reads")
	}
}
func TestStreams(t *testing.T) {
	f, b, _, _ := fixture(t)
	write(t, f, "/x", "old")
	read(t, f, "/x", "old")
	w, e := f.OpenWrite("/x")
	if e != nil {
		t.Fatal(e)
	}
	if _, e = w.Write([]byte("new")); e != nil {
		t.Fatal(e)
	}
	read(t, f, "/x", "new")
	read(t, f, "/x", "new")
	if b.reads.Load() != 3 {
		t.Fatal("writer did not bypass")
	}
	if e = f.Close(); e == nil {
		t.Fatal("closed live mount")
	}
	if e = w.Close(); e != nil {
		t.Fatal(e)
	}
	if e = w.Close(); e != nil {
		t.Fatal(e)
	}
	read(t, f, "/x", "new")
	read(t, f, "/x", "new")
	if b.reads.Load() != 4 {
		t.Fatal("did not resume")
	}
}
func TestLimitsCorruption(t *testing.T) {
	f, b, s, _ := fixture(t)
	f.cfg.MaxFileSizeBytes = 3
	write(t, f, "/big", "1234")
	read(t, f, "/big", "1234")
	read(t, f, "/big", "1234")
	if b.reads.Load() != 2 {
		t.Fatal("large cached")
	}
	for i := 0; i < 2; i++ {
		_, _ = f.Read("/absent", 0, -1)
	}
	if b.reads.Load() != 4 {
		t.Fatal("negative cached")
	}
	write(t, f, "/x", "123")
	read(t, f, "/x", "123")
	s.mu.Lock()
	for k := range s.data {
		s.data[k] = []byte("broken")
	}
	s.mu.Unlock()
	read(t, f, "/x", "123")
	if b.reads.Load() != 6 {
		t.Fatal("corruption not reloaded")
	}
	_, _ = f.Read("/x", 1, 1)
	_, _ = f.Read("/x", 1, 1)
	if b.reads.Load() != 8 {
		t.Fatal("range cached")
	}
}
func TestAliasesExtensions(t *testing.T) {
	f, _, _, root := fixture(t)
	local, e := localfs.NewLocalFS(root)
	if e != nil {
		t.Fatal(e)
	}
	f.backend = local
	write(t, f, "/secret.lock", "a")
	if e = os.Symlink("secret.lock", filepath.Join(root, "alias")); e != nil {
		t.Fatal(e)
	}
	read(t, f, "/alias", "a")
	write(t, f, "/secret.lock", "b")
	read(t, f, "/alias", "b")
	if f.Stats().Fills != 0 {
		t.Fatal("cached symlink")
	}
	w := Wrap(f)
	if _, ok := w.(filesystem.HandleFS); ok {
		t.Fatal("invented handles")
	}
	if _, ok := w.(filesystem.Streamer); !ok {
		t.Fatal("lost streamer")
	}
	if _, ok := w.(filesystem.Toucher); ok {
		t.Fatal("invented touch")
	}
}

func TestWritableHandles(t *testing.T) {
	f, _, _, _ := fixture(t)
	f.backend = memfs.NewMemoryFS()
	write(t, f, "/x", "old")
	read(t, f, "/x", "old")
	hfs := Wrap(f).(filesystem.HandleFS)
	h, e := hfs.OpenHandle("/x", filesystem.O_RDWR|filesystem.O_TRUNC, 0644)
	if e != nil {
		t.Fatal(e)
	}
	same, e := hfs.GetHandle(h.ID())
	if e != nil || same != h {
		t.Fatal("unwrapped handle")
	}
	if _, e = same.Write([]byte("new")); e != nil {
		t.Fatal(e)
	}
	before := f.Stats().Bypasses
	read(t, f, "/x", "new")
	read(t, f, "/x", "new")
	if f.Stats().Bypasses != before+2 {
		t.Fatal("writer did not bypass")
	}
	if e = f.Close(); e == nil {
		t.Fatal("closed live handle")
	}
	if e = hfs.CloseHandle(h.ID()); e != nil {
		t.Fatal(e)
	}
	if e = h.Close(); e != nil {
		t.Fatal(e)
	}
	if _, e = hfs.GetHandle(h.ID()); e == nil {
		t.Fatal("retained closed handle")
	}
	read(t, f, "/x", "new")
	before = f.Stats().Hits
	read(t, f, "/x", "new")
	if f.Stats().Hits != before+1 {
		t.Fatal("did not resume cache")
	}
}

func TestLogicalExpiryAndGenerationCapacity(t *testing.T) {
	f, b, s, _ := fixture(t)
	write(t, f, "/x", "data")
	read(t, f, "/x", "data")
	s.mu.Lock()
	for k, data := range s.data {
		var v envelope
		if e := json.Unmarshal(data[len(magic):], &v); e != nil {
			t.Fatal(e)
		}
		v.Expires = 1
		encoded, _ := json.Marshal(v)
		s.data[k] = append([]byte(magic), encoded...)
	}
	s.mu.Unlock()
	read(t, f, "/x", "data")
	if b.reads.Load() != 2 {
		t.Fatal("logical expiry ignored")
	}
	f.mu.Lock()
	epoch := f.epoch
	for i := 0; i < maxStateEntries; i++ {
		f.versions[fmt.Sprint(i)] = 1
	}
	f.mu.Unlock()
	write(t, f, "/y", "new")
	f.mu.Lock()
	changed := f.epoch != epoch && len(f.versions) == 0
	f.mu.Unlock()
	if !changed {
		t.Fatal("unsafe capacity eviction")
	}
	read(t, f, "/x", "data")
	if b.reads.Load() != 3 {
		t.Fatal("old epoch hit")
	}
}

func TestReadAndFillFailure(t *testing.T) {
	for _, op := range []string{"get", "set"} {
		t.Run(op, func(t *testing.T) {
			f, b, s, _ := fixture(t)
			write(t, f, "/x", "old")
			s.mu.Lock()
			s.failGet = op == "get"
			s.failSet = op == "set"
			s.mu.Unlock()
			read(t, f, "/x", "old")
			write(t, f, "/x", "new")
			read(t, f, "/x", "new")
			read(t, f, "/x", "new")
			if b.reads.Load() != 3 || f.Stats().Failures == 0 {
				t.Fatal("failure did not bypass")
			}
		})
	}
}

type lateStore struct {
	*store
	started, release, finished chan struct{}
	once                       sync.Once
}

func (s *lateStore) Set(c context.Context, k string, b []byte, ttl time.Duration) error {
	delayed := false
	s.once.Do(func() { delayed = true; close(s.started) })
	if delayed {
		<-s.release
		defer close(s.finished)
	}
	return s.store.Set(c, k, b, ttl)
}
func TestLateFillCannotReviveOldEpoch(t *testing.T) {
	f, _, s, _ := fixture(t)
	delayed := &lateStore{store: s, started: make(chan struct{}), release: make(chan struct{}), finished: make(chan struct{})}
	r, e := cache.WithProvider(delayed, cache.Config{CommandTimeoutMS: 20})
	if e != nil {
		t.Fatal(e)
	}
	f.runtime = r
	defer r.Close(context.Background())
	write(t, f, "/x", "old")
	read(t, f, "/x", "old") // fill times out but backend read succeeds
	f.mu.Lock()
	oldEpoch := f.epoch
	bypass := f.bypass
	f.mu.Unlock()
	if !bypass {
		t.Error("fill timeout did not bypass")
	}
	write(t, f, "/x", "new")
	close(delayed.release)
	<-delayed.finished
	deadline := time.Now().Add(3 * time.Second)
	for {
		f.mu.Lock()
		active := !f.bypass
		f.mu.Unlock()
		if active {
			break
		}
		if time.Now().After(deadline) {
			t.Fatal("recovery timed out")
		}
		time.Sleep(10 * time.Millisecond)
	}
	f.mu.Lock()
	changed := f.epoch != oldEpoch
	f.mu.Unlock()
	if !changed {
		t.Fatal("recovered old epoch")
	}
	read(t, f, "/x", "new")
	read(t, f, "/x", "new")
}

type metaFS struct {
	filesystem.FileSystem
	entries     []filesystem.FileInfo
	destructive bool
	reads       int
}

func (m *metaFS) ReadDir(string) ([]filesystem.FileInfo, error) { m.reads++; return m.entries, nil }
func (m *metaFS) GetCapabilities() filesystem.Capabilities      { return filesystem.DefaultCapabilities() }
func (m *metaFS) GetPathCapabilities(string) filesystem.Capabilities {
	c := m.GetCapabilities()
	c.IsReadDestructive = m.destructive
	return c
}
func TestDirectoryMetadataAndDestructiveBypass(t *testing.T) {
	f, _, _, _ := fixture(t)
	m := &metaFS{FileSystem: f.backend, entries: []filesystem.FileInfo{{Name: "file", Size: 4, Mode: 0644, ModTime: time.Now().UTC(), Meta: filesystem.MetaData{Name: "custom", Type: "file", Content: map[string]string{"version": "v1"}}}}}
	f.backend = m
	first, e := f.ReadDir("/")
	if e != nil {
		t.Fatal(e)
	}
	first[0].Meta.Content["version"] = "changed"
	second, e := f.ReadDir("/")
	if e != nil {
		t.Fatal(e)
	}
	if second[0].Meta.Content["version"] != "v1" || second[0].Mode != 0644 || !second[0].ModTime.Equal(m.entries[0].ModTime) || m.reads != 1 {
		t.Fatal("metadata not preserved", second)
	}
	m.destructive = true
	_, _ = f.ReadDir("/")
	_, _ = f.ReadDir("/")
	if m.reads != 3 {
		t.Fatal("destructive reads cached")
	}
}

func TestMutationInvalidatesAncestorDirectoryMetadata(t *testing.T) {
	f, b, _, _ := fixture(t)
	if e := f.Mkdir("/parent", 0755); e != nil {
		t.Fatal(e)
	}
	_, _ = f.ReadDir("/")
	_, _ = f.ReadDir("/")
	if b.dirs.Load() != 1 {
		t.Fatal("no root hit")
	}
	write(t, f, "/parent/child", "data")
	_, _ = f.ReadDir("/")
	if b.dirs.Load() != 2 {
		t.Fatal("stale ancestor directory metadata")
	}
}
