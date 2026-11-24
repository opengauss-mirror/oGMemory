package handlers

import (
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/alicebob/miniredis/v2"
	"github.com/c4pt0r/agfs/agfs-server/pkg/cachefs"
	"github.com/c4pt0r/agfs/agfs-server/pkg/mountablefs"
	"github.com/c4pt0r/agfs/agfs-server/pkg/plugin"
	"github.com/c4pt0r/agfs/agfs-server/pkg/plugin/api"
	"github.com/c4pt0r/agfs/agfs-server/pkg/plugins/localfs"
	cache "github.com/ogmemory/cache-runtime"
)

func TestCacheFSHTTP(t *testing.T) {
	redis := miniredis.RunT(t)
	m := mountablefs.NewMountableFS(api.PoolConfig{})
	if e := m.ConfigureCache(cachefs.Config{Enabled: true, Namespace: "http", Mounts: []string{"/local"}}, cache.Config{Provider: "redis", CommandTimeoutMS: 500, Params: map[string]interface{}{"endpoints": []string{"redis://" + redis.Addr()}}}); e != nil {
		t.Fatal(e)
	}
	defer m.Close(context.Background())
	m.RegisterPluginFactory("localfs", func() plugin.ServicePlugin { return localfs.NewLocalFSPlugin() })
	if e := m.MountPlugin("localfs", "/local", map[string]interface{}{"local_dir": t.TempDir()}); e != nil {
		t.Fatal(e)
	}
	h := NewHandler(m, nil)
	mux := http.NewServeMux()
	h.SetupRoutes(mux)
	server := httptest.NewServer(mux)
	defer server.Close()
	request := func(method, p, body string) string {
		t.Helper()
		req, _ := http.NewRequest(method, server.URL+p, strings.NewReader(body))
		resp, e := server.Client().Do(req)
		if e != nil {
			t.Fatal(e)
		}
		defer resp.Body.Close()
		b, e := io.ReadAll(resp.Body)
		if e != nil || resp.StatusCode != 200 {
			t.Fatalf("%d %s %v", resp.StatusCode, b, e)
		}
		return string(b)
	}
	request("PUT", "/api/v1/files?path=/local/.meta.json", "one")
	for i := 0; i < 2; i++ {
		if b := request("GET", "/api/v1/files?path=/local/.meta.json", ""); b != "one" {
			t.Fatal(b)
		}
	}
	if m.CacheStats()["/local"].Hits != 1 {
		t.Fatal("HTTP did not use cache")
	}
	request("GET", "/api/v1/directories?path=/local", "")
	request("GET", "/api/v1/directories?path=/local", "")
	beforeFresh := m.CacheStats()["/local"]
	request("GET", "/api/v1/directories?path=/local&fresh=true", "")
	afterFresh := m.CacheStats()["/local"]
	if afterFresh.Hits != beforeFresh.Hits || afterFresh.Bypasses != beforeFresh.Bypasses+1 {
		t.Fatal("fresh directory request did not bypass cache")
	}
	request("PUT", "/api/v1/files?path=/local/file.lock", "lock")
	if b := request("GET", "/api/v1/directories?path=/local", ""); !strings.Contains(b, "file.lock") {
		t.Fatal("stale parent")
	}
	before := m.CacheStats()["/local"].Hits
	request("GET", "/api/v1/files?path=/local/file.lock", "")
	request("GET", "/api/v1/files?path=/local/file.lock", "")
	if m.CacheStats()["/local"].Hits != before {
		t.Fatal("cached lock")
	}
	redis.Close()
	request("PUT", "/api/v1/files?path=/local/.meta.json", "two")
	if b := request("GET", "/api/v1/files?path=/local/.meta.json", ""); b != "two" {
		t.Fatal("outage stale data", b)
	}
	if m.CacheStats()["/local"].Failures == 0 {
		t.Fatal("outage not detected")
	}
}
