package mountablefs

import (
	"context"
	"errors"
	"io"
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
