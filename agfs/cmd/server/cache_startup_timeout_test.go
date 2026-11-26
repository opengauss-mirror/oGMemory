package main

import (
	"context"
	"fmt"
	"net"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/alicebob/miniredis/v2"
	"github.com/c4pt0r/agfs/agfs-server/pkg/filesystem"
	"github.com/c4pt0r/agfs/agfs-server/pkg/plugin"
	"github.com/c4pt0r/agfs/agfs-server/pkg/plugins/localfs"
)

type controlledLocalPlugin struct {
	plugin.ServicePlugin
	late bool
}

func (p *controlledLocalPlugin) Validate(cfg map[string]interface{}) error {
	p.late = cfg["mount_path"] == "/late"
	if p.late && os.Getenv("OGMEM_STARTUP_STAGE") == "validate" {
		startupMarker(os.Getenv("OGMEM_STARTUP_ENTERED"))
		awaitStartupMarker(os.Getenv("OGMEM_STARTUP_RELEASE"))
	}
	return p.ServicePlugin.Validate(cfg)
}
func (p *controlledLocalPlugin) Initialize(cfg map[string]interface{}) error {
	if p.late {
		if os.Getenv("OGMEM_STARTUP_STAGE") == "validate" {
			panic("initialized after validation timeout")
		}
		startupMarker(os.Getenv("OGMEM_STARTUP_ENTERED"))
		awaitStartupMarker(os.Getenv("OGMEM_STARTUP_RELEASE"))
	}
	return p.ServicePlugin.Initialize(cfg)
}
func (p *controlledLocalPlugin) GetFileSystem() filesystem.FileSystem {
	// Mount holds the routing lock and preparation has already been accepted.
	// Canceling startup after this point cannot discard the healthy preparation.
	if !p.late {
		startupMarker(os.Getenv("OGMEM_STARTUP_HEALTHY"))
	}
	return p.ServicePlugin.GetFileSystem()
}
func (p *controlledLocalPlugin) Shutdown() error {
	err := p.ServicePlugin.Shutdown()
	if p.late {
		startupMarker(os.Getenv("OGMEM_STARTUP_CLEANED"))
	}
	return err
}

func TestCacheStartupTimeout(t *testing.T) {
	if os.Getenv("OGMEM_STARTUP_TIMEOUT_HELPER") == "1" {
		newStartupContext = func() (context.Context, context.CancelFunc) {
			ctx, cancel := context.WithCancel(context.Background())
			go func() { awaitStartupMarker(os.Getenv("OGMEM_STARTUP_CANCEL")); cancel() }()
			return ctx, cancel
		}
		availablePlugins["localfs"] = func() plugin.ServicePlugin { return &controlledLocalPlugin{ServicePlugin: localfs.NewLocalFSPlugin()} }
		serveHTTP = func(server *http.Server) error {
			listener, err := net.Listen("tcp", server.Addr)
			if err != nil {
				return err
			}
			defer listener.Close()
			if err := publishStartupAddress(os.Getenv("OGMEM_STARTUP_READY"), listener.Addr().String()); err != nil {
				return err
			}
			return server.Serve(listener)
		}
		runStartupMain()
		return
	}
	for _, stage := range []string{"validate", "initialize"} {
		for _, selected := range []bool{false, true} {
			t.Run(fmt.Sprintf("%s/%v", stage, selected), func(t *testing.T) {
				redis := miniredis.RunT(t)
				cached, late, work := t.TempDir(), t.TempDir(), t.TempDir()
				if err := os.WriteFile(filepath.Join(late, "present"), []byte("late"), 0600); err != nil {
					t.Fatal(err)
				}
				cachePath := "/cached"
				if selected {
					cachePath = "/late"
				}
				cfg := fmt.Sprintf(`server:
  address: "127.0.0.1:0"
cache:
  provider: redis
  params:
    endpoints: ["redis://%s"]
cachefs:
  enabled: true
  namespace: timeout-test
  mounts: [%q]
plugins:
  localfs:
    - name: healthy
      enabled: true
      path: /cached
      config:
        local_dir: %q
    - name: slow
      enabled: true
      path: /late
      config:
        local_dir: %q
`, redis.Addr(), cachePath, cached, late)
				markers := map[string]string{}
				env := []string{"OGMEM_STARTUP_TIMEOUT_HELPER=1", "OGMEM_STARTUP_STAGE=" + stage}
				for _, name := range []string{"ENTERED", "RELEASE", "HEALTHY", "CANCEL", "CLEANED", "READY"} {
					markers[name] = filepath.Join(work, name)
					env = append(env, "OGMEM_STARTUP_"+name+"="+markers[name])
				}
				cmd, ctx := startupTestCommand(t, "TestCacheStartupTimeout", cfg, env...)
				logPath := filepath.Join(work, "server.log")
				log, err := os.Create(logPath)
				if err != nil {
					t.Fatal(err)
				}
				defer log.Close()
				cmd.Stdout, cmd.Stderr = log, log
				if err := cmd.Start(); err != nil {
					t.Fatal(err)
				}
				waited := false
				defer func() {
					if !waited {
						_ = cmd.Process.Kill()
						_ = cmd.Wait()
					}
				}()
				waitStartupMarker(t, ctx, markers["HEALTHY"])
				waitStartupMarker(t, ctx, markers["ENTERED"])
				startupMarker(markers["CANCEL"])
				if selected {
					err := cmd.Wait()
					waited = true
					out, _ := os.ReadFile(logPath)
					if ctx.Err() != nil || err == nil || !strings.Contains(string(out), "context canceled") || strings.Contains(string(out), "Starting AGFS server") {
						t.Fatalf("selected mount must fail before listening: %v\n%s", err, out)
					}
					return
				}
				waitStartupMarker(t, ctx, markers["READY"])
				addr, err := os.ReadFile(markers["READY"])
				if err != nil {
					t.Fatal(err)
				}
				client := &http.Client{}
				check := func(path string, want int) {
					t.Helper()
					request, err := http.NewRequestWithContext(ctx, http.MethodGet, "http://"+string(addr)+"/api/v1/stat?path="+path, nil)
					if err != nil {
						t.Fatal(err)
					}
					resp, err := client.Do(request)
					if err != nil {
						t.Fatal(err)
					}
					defer resp.Body.Close()
					if resp.StatusCode != want {
						t.Fatalf("%s status=%d, want %d", path, resp.StatusCode, want)
					}
				}
				check("/cached", http.StatusOK)
				if stage == "initialize" {
					startupMarker(markers["RELEASE"])
					waitStartupMarker(t, ctx, markers["CLEANED"])
				}
				// The synchronous preparation unit test checks cancellation
				// after Validate returns. This test checks startup visibility.
				check("/late/present", http.StatusNotFound)
				_ = cmd.Process.Signal(os.Interrupt)
				if err := cmd.Wait(); err != nil {
					t.Fatal(err)
				}
				waited = true
			})
		}
	}
}
