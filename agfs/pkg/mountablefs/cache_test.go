package mountablefs

import (
	"context"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/c4pt0r/agfs/agfs-server/pkg/cachefs"
	"github.com/c4pt0r/agfs/agfs-server/pkg/filesystem"
	"github.com/c4pt0r/agfs/agfs-server/pkg/plugin"
	"github.com/c4pt0r/agfs/agfs-server/pkg/plugin/api"
	"github.com/c4pt0r/agfs/agfs-server/pkg/plugins/localfs"
	"github.com/c4pt0r/agfs/agfs-server/pkg/plugins/sqlfs"
	cache "github.com/ogmemory/cache-runtime"
)

type testStore struct{ data map[string][]byte }

func (s *testStore) Get(_ context.Context, k string) (cache.Value, error) {
	b, ok := s.data[k]
	return cache.Value{Data: b, Found: ok}, nil
}
func (s *testStore) Set(_ context.Context, k string, b []byte, _ time.Duration) error {
	s.data[k] = append([]byte(nil), b...)
	return nil
}
func (s *testStore) Delete(_ context.Context, ks []string) (uint64, error) {
	for _, k := range ks {
		delete(s.data, k)
	}
	return 0, nil
}
func (s *testStore) Ping(context.Context) error { return nil }
func (s *testStore) Close() error               { return nil }
func cachedMounts(t *testing.T) *MountableFS {
	t.Helper()
	r, e := cache.WithProvider(&testStore{data: map[string][]byte{}}, cache.Config{CommandTimeoutMS: 500, MaxInflight: 1})
	if e != nil {
		t.Fatal(e)
	}
	m := NewMountableFS(api.PoolConfig{})
	if e = m.UseCache(cachefs.Config{Enabled: true, Namespace: "test", Mounts: []string{"/data"}}, r); e != nil {
		t.Fatal(e)
	}
	t.Cleanup(func() {
		if e := m.Close(context.Background()); e != nil {
			t.Error(e)
		}
		_ = r.Close(context.Background())
	})
	return m
}
func TestCacheStaticDynamicMounts(t *testing.T) {
	for _, kind := range []string{"localfs", "sqlfs"} {
		for _, dynamic := range []bool{false, true} {
			t.Run(kind+map[bool]string{false: "/static", true: "/dynamic"}[dynamic], func(t *testing.T) {
				m := cachedMounts(t)
				var factory PluginFactory
				var cfg map[string]interface{}
				if kind == "localfs" {
					factory = func() plugin.ServicePlugin { return localfs.NewLocalFSPlugin() }
					cfg = map[string]interface{}{"local_dir": t.TempDir()}
				} else {
					factory = func() plugin.ServicePlugin { return sqlfs.NewSQLFSPlugin() }
					cfg = map[string]interface{}{"backend": "sqlite", "db_path": filepath.Join(t.TempDir(), "test.db"), "cache_enabled": true}
				}
				m.RegisterPluginFactory(kind, factory)
				if dynamic {
					if e := m.MountPlugin(kind, "/data", cfg); e != nil {
						t.Fatal(e)
					}
				} else {
					p := factory()
					prepared, e := m.PrepareMountConfig(kind, "/data", cfg)
					if e != nil {
						t.Fatal(e)
					}
					if kind == "sqlfs" && prepared["cache_enabled"] != false {
						t.Fatal("inner cache enabled")
					}
					if e = p.Initialize(prepared); e != nil {
						t.Fatal(e)
					}
					if e = m.Mount("/data", p); e != nil {
						t.Fatal(e)
					}
				}
				put := func(value string) {
					t.Helper()
					if _, e := m.Write("/data/.meta.json", []byte(value), -1, filesystem.WriteFlagCreate|filesystem.WriteFlagTruncate); e != nil {
						t.Fatal(e)
					}
				}
				get := func(want string) {
					t.Helper()
					b, e := m.Read("/data/.meta.json", 0, -1)
					if (e != nil && !errors.Is(e, io.EOF)) || string(b) != want {
						t.Fatalf("read %q %v", b, e)
					}
				}
				put("old")
				get("old")
				get("old")
				if m.CacheStats()["/data"].Hits != 1 {
					t.Fatal("routing missed cache")
				}
				put("new")
				get("new")
				if kind == "localfs" {
					if e := m.MountPlugin(kind, "/alias", cfg); e == nil {
						t.Fatal("accepted storage alias")
					}
				}
				if e := m.Unmount("/data"); e != nil {
					t.Fatal(e)
				}
				if e := m.MountPlugin(kind, "/data", cfg); e != nil {
					t.Fatal(e)
				}
				get("new")
				if m.CacheStats()["/data"].Hits != 0 {
					t.Fatal("reused epoch")
				}
			})
		}
	}
}

type blockingShutdown struct {
	MockPlugin
	started, release chan struct{}
}

func (p *blockingShutdown) Shutdown() error { close(p.started); <-p.release; return nil }
func TestCloseDeadlineAndMountRejection(t *testing.T) {
	m := NewMountableFS(api.PoolConfig{})
	p := &blockingShutdown{MockPlugin: MockPlugin{name: "test"}, started: make(chan struct{}), release: make(chan struct{})}
	if e := m.Mount("/x", p); e != nil {
		t.Fatal(e)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Millisecond)
	defer cancel()
	if e := m.Close(ctx); !errors.Is(e, context.DeadlineExceeded) {
		t.Fatal(e)
	}
	close(p.release)
	if e := m.Close(context.Background()); e != nil {
		t.Fatal(e)
	}
	if e := m.Mount("/y", &MockPlugin{}); e == nil {
		t.Fatal("mounted after shutdown")
	}
}

type registrationBackend struct {
	filesystem.FileSystem
	scoped filesystem.FileSystem
}

func (b registrationBackend) RegistrationView() (filesystem.FileSystem, func(), error) {
	return b.scoped, func() {}, nil
}

func TestCacheBypassesPinnedRegistrationSession(t *testing.T) {
	m := cachedMounts(t)
	local, err := localfs.NewLocalFS(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	p := registrationPlugin{&MockPlugin{name: "localfs"}, registrationBackend{local, local}}
	if err := m.Mount("/data", p); err != nil {
		t.Fatal(err)
	}
	put := func(fs filesystem.FileSystem, text string) {
		t.Helper()
		if _, err := fs.Write("/data/item", []byte(text), -1, filesystem.WriteFlagCreate|filesystem.WriteFlagTruncate); err != nil {
			t.Fatal(err)
		}
	}
	get := func(want string) {
		t.Helper()
		data, err := m.Read("/data/item", 0, -1)
		if (err != nil && !errors.Is(err, io.EOF)) || string(data) != want {
			t.Fatalf("read %q, %v; want %q", data, err, want)
		}
	}
	put(m, "old")
	get("old")
	session, err := m.LockRegistration("/data/_system/code_workspace_registration.lock")
	if err != nil {
		t.Fatal(err)
	}
	defer session.Close()
	put(session.FS, "new")
	if mount := session.FS.GetMounts()[0]; mount.ExecutionFS() != local {
		t.Fatal("registration operation was not routed to the pinned view")
	}
	get("new")
	if stats := m.CacheStats()["/data"]; stats.Bypasses != 1 || stats.Hits != 0 {
		t.Fatalf("expected one bypass during registration: %+v", stats)
	}
	session.Close()
	get("new")
	get("new")
	if stats := m.CacheStats()["/data"]; stats.Hits != 1 {
		t.Fatalf("expected cache to resume after registration: %+v", stats)
	}
}

func mountLocalForCacheTest(t *testing.T, m *MountableFS, path, dir string) {
	t.Helper()
	m.RegisterPluginFactory("localfs", func() plugin.ServicePlugin { return localfs.NewLocalFSPlugin() })
	if err := m.MountPlugin("localfs", path, map[string]interface{}{"local_dir": dir}); err != nil {
		t.Fatal(err)
	}
}

func TestCacheSQLiteRecursiveOperationsMatchBackend(t *testing.T) {
	for _, tc := range []struct {
		name, source, sibling string
		rename                bool
	}{
		{"underscore", "old_name", "oldXname", false},
		{"percent", "old%name", "oldLongname", true},
		{"case", "branch", "Branch", false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			m := cachedMounts(t)
			m.RegisterPluginFactory("sqlfs", func() plugin.ServicePlugin { return sqlfs.NewSQLFSPlugin() })
			if err := m.MountPlugin("sqlfs", "/data", map[string]interface{}{
				"backend": "sqlite", "db_path": filepath.Join(t.TempDir(), "store.db"),
			}); err != nil {
				t.Fatal(err)
			}
			for _, dir := range []string{tc.source, tc.sibling} {
				if err := m.Mkdir("/data/"+dir, 0755); err != nil {
					t.Fatal(err)
				}
			}
			item := "/" + tc.sibling + "/item"
			if _, err := m.Write("/data"+item, []byte("stored"), -1, filesystem.WriteFlagCreate); err != nil {
				t.Fatal(err)
			}
			for i := 0; i < 2; i++ {
				_, _ = m.Read("/data"+item, 0, -1)
			}
			if m.CacheStats()["/data"].Hits != 1 {
				t.Fatal("expected warm file")
			}
			var err error
			if tc.rename {
				err = m.Rename("/data/"+tc.source, "/data/destination")
			} else {
				err = m.RemoveAll("/data/" + tc.source)
			}
			if err != nil {
				t.Fatal(err)
			}
			want, backendErr := m.GetMounts()[0].Plugin.GetFileSystem().Read(item, 0, -1)
			got, cachedErr := m.Read("/data"+item, 0, -1)
			if string(want) != string(got) || fmt.Sprint(backendErr) != fmt.Sprint(cachedErr) {
				t.Fatalf("backend=%q %v, cache=%q %v", want, backendErr, got, cachedErr)
			}
		})
	}
}

func TestCacheLocalRegistrationSidecars(t *testing.T) {
	m := cachedMounts(t)
	mountLocalForCacheTest(t, m, "/data", t.TempDir())
	for i := 0; i < 2; i++ {
		if _, err := m.ReadDir("/data"); err != nil {
			t.Fatal(err)
		}
	}
	lockPath := "/data/_system/code_workspace_registration.lock"
	session, err := m.LockRegistration(lockPath)
	if err != nil {
		t.Fatal(err)
	}
	session.Close()
	entries, err := m.ReadDir("/data")
	if err != nil || len(entries) != 1 || entries[0].Name != "_system" {
		t.Fatalf("registration created a missing directory entry: %v, %v", entries, err)
	}
	before := m.CacheStats()["/data"].Bypasses
	_, _ = m.Read(lockPath+".sqlite", 0, -1)
	if m.CacheStats()["/data"].Bypasses != before+1 {
		t.Fatal("registration lock database was admitted to cache")
	}
}

func TestCacheLocalCrossMountAliasRetiresCache(t *testing.T) {
	m := cachedMounts(t)
	root, other := t.TempDir(), t.TempDir()
	if err := os.Symlink(root, filepath.Join(other, "link")); err != nil {
		t.Skipf("symlink unavailable: %v", err)
	}
	mountLocalForCacheTest(t, m, "/data", root)
	if _, err := m.Write("/data/item", []byte("old"), -1, filesystem.WriteFlagCreate); err != nil {
		t.Fatal(err)
	}
	for i := 0; i < 2; i++ {
		_, _ = m.Read("/data/item", 0, -1)
	}
	if m.CacheStats()["/data"].Hits != 1 {
		t.Fatal("expected warm file before alias mount")
	}
	mountLocalForCacheTest(t, m, "/other", other)
	if _, err := m.Write("/other/link/item", []byte("new"), -1, filesystem.WriteFlagTruncate); err != nil {
		t.Fatal(err)
	}
	data, err := m.Read("/data/item", 0, -1)
	if string(data) != "new" || (err != nil && !errors.Is(err, io.EOF)) {
		t.Fatalf("cached old value after cross-mount write: %q, %v", data, err)
	}
}

func TestCacheLocalPreservesInvalidUTF8DirectoryName(t *testing.T) {
	m := cachedMounts(t)
	root := t.TempDir()
	name := "local-" + string([]byte{0xff})
	if err := os.WriteFile(filepath.Join(root, name), []byte("x"), 0644); err != nil {
		t.Skipf("filesystem does not support non-UTF-8 names: %v", err)
	}
	mountLocalForCacheTest(t, m, "/data", root)
	for i := 0; i < 2; i++ {
		entries, err := m.ReadDir("/data")
		if err != nil || len(entries) != 1 || entries[0].Name != name {
			t.Fatalf("read %d: %+v, %v", i, entries, err)
		}
	}
}

func TestCacheLocalCaseAliasInvalidates(t *testing.T) {
	m := cachedMounts(t)
	root := t.TempDir()
	mountLocalForCacheTest(t, m, "/data", root)
	if _, err := m.Write("/data/Item", []byte("old"), -1, filesystem.WriteFlagCreate); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(filepath.Join(root, "item")); os.IsNotExist(err) {
		t.Skip("case-sensitive local volume")
	} else if err != nil {
		t.Fatal(err)
	}
	for i := 0; i < 2; i++ {
		_, _ = m.Read("/data/Item", 0, -1)
	}
	if m.CacheStats()["/data"].Hits != 1 {
		t.Fatal("expected warm file")
	}
	if _, err := m.Write("/data/item", []byte("new"), -1, filesystem.WriteFlagTruncate); err != nil {
		t.Fatal(err)
	}
	data, err := m.Read("/data/Item", 0, -1)
	if string(data) != "new" || (err != nil && !errors.Is(err, io.EOF)) {
		t.Fatalf("cached old value after case-alias write: %q, %v", data, err)
	}
}
