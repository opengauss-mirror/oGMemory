package main

import (
	"fmt"
	"os"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/alicebob/miniredis/v2"
	"github.com/c4pt0r/agfs/agfs-server/pkg/filesystem"
	"github.com/c4pt0r/agfs/agfs-server/pkg/plugin"
	"github.com/c4pt0r/agfs/agfs-server/pkg/plugins/localfs"
)

type orderedLocalPlugin struct {
	plugin.ServicePlugin
	path, first               string
	firstMountEnteredAssembly chan struct{}
	once                      *sync.Once
}

func (p *orderedLocalPlugin) Initialize(cfg map[string]interface{}) error {
	p.path, _ = cfg["mount_path"].(string)
	if p.path != p.first {
		select {
		case <-p.firstMountEnteredAssembly:
		case <-time.After(5 * time.Second):
			return fmt.Errorf("first mount %s did not start", p.first)
		}
	}
	return p.ServicePlugin.Initialize(cfg)
}

func (p *orderedLocalPlugin) GetFileSystem() filesystem.FileSystem {
	// Mount holds the routing mutex here. The second mount can initialize,
	// but cannot enter its conflict check until the first mount is published.
	if p.path == p.first {
		p.once.Do(func() { close(p.firstMountEnteredAssembly) })
	}
	return p.ServicePlugin.GetFileSystem()
}

func TestCacheStartupRejectsStorageAlias(t *testing.T) {
	if first := os.Getenv("OGMEM_STARTUP_FIRST"); first != "" {
		firstMountEnteredAssembly := make(chan struct{})
		once := new(sync.Once)
		availablePlugins["localfs"] = func() plugin.ServicePlugin {
			return &orderedLocalPlugin{ServicePlugin: localfs.NewLocalFSPlugin(), first: first, firstMountEnteredAssembly: firstMountEnteredAssembly, once: once}
		}
		runStartupMain()
		return
	}
	for _, first := range []string{"/cached", "/uncached"} {
		t.Run(first, func(t *testing.T) {
			redis := miniredis.RunT(t)
			root := t.TempDir()
			second := "/cached"
			if first == second {
				second = "/uncached"
			}
			cfg := fmt.Sprintf(`server:
  address: "127.0.0.1:0"
cache:
  provider: redis
  params:
    endpoints: ["redis://%s"]
cachefs:
  enabled: true
  namespace: startup-test
  mounts: ["/cached"]
plugins:
  localfs:
    - name: first
      enabled: true
      path: %q
      config:
        local_dir: %q
    - name: second
      enabled: true
      path: %q
      config:
        local_dir: %q
`, redis.Addr(), first, root, second, root)
			cmd, ctx := startupTestCommand(t, "TestCacheStartupRejectsStorageAlias", cfg, "OGMEM_STARTUP_FIRST="+first)
			out, err := cmd.CombinedOutput()
			if ctx.Err() != nil || err == nil {
				t.Fatalf("expected startup failure, got %v (%v)\n%s", err, ctx.Err(), out)
			}
			log := string(out)
			if !strings.Contains(log, "CacheFS mount setup failed") || !strings.Contains(log, "cachefs storage aliases mounts") ||
				!strings.Contains(log, "/cached") || !strings.Contains(log, "/uncached") || strings.Contains(log, "Starting AGFS server") {
				t.Fatalf("expected alias rejection before HTTP listen\n%s", out)
			}
		})
	}
}
