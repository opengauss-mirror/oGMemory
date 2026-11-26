package main

import (
	"context"
	"flag"
	"fmt"
	"net"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/alicebob/miniredis/v2"
	"github.com/c4pt0r/agfs/agfs-server/pkg/plugin"
	"github.com/c4pt0r/agfs/agfs-server/pkg/plugins/localfs"
)

type delayedLocalPlugin struct {
	plugin.ServicePlugin
	late bool
}

func (p *delayedLocalPlugin) Initialize(cfg map[string]interface{}) error {
	p.late = cfg["mount_path"] == "/late"
	if p.late {
		time.Sleep(1500 * time.Millisecond)
	}
	return p.ServicePlugin.Initialize(cfg)
}

func (p *delayedLocalPlugin) Shutdown() error {
	err := p.ServicePlugin.Shutdown()
	if p.late {
		_ = os.WriteFile(os.Getenv("OGMEM_STARTUP_CLEANED"), nil, 0600)
	}
	return err
}

func TestCacheStartupTimeout(t *testing.T) {
	if os.Getenv("OGMEM_STARTUP_TIMEOUT_HELPER") == "1" {
		startupMountTimeout = 500 * time.Millisecond
		availablePlugins["localfs"] = func() plugin.ServicePlugin {
			return &delayedLocalPlugin{ServicePlugin: localfs.NewLocalFSPlugin()}
		}
		flag.CommandLine = flag.NewFlagSet("agfs", flag.ExitOnError)
		os.Args = []string{"agfs", "-c", os.Getenv("OGMEM_STARTUP_CONFIG")}
		main()
		return
	}
	for _, selected := range []bool{false, true} {
		t.Run(fmt.Sprint(selected), func(t *testing.T) {
			redis := miniredis.RunT(t)
			listener, err := net.Listen("tcp", "127.0.0.1:0")
			if err != nil {
				t.Fatal(err)
			}
			addr := listener.Addr().String()
			listener.Close()
			cached, late, work := t.TempDir(), t.TempDir(), t.TempDir()
			if err := os.WriteFile(filepath.Join(late, "present"), []byte("late"), 0600); err != nil {
				t.Fatal(err)
			}
			cachePath := "/cached"
			if selected {
				cachePath = "/late"
			}
			cfg := fmt.Sprintf(`server:
  address: %q
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
`, addr, redis.Addr(), cachePath, cached, late)
			path, cleaned := filepath.Join(work, "config.yaml"), filepath.Join(work, "cleaned")
			if err := os.WriteFile(path, []byte(cfg), 0600); err != nil {
				t.Fatal(err)
			}
			ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
			defer cancel()
			cmd := exec.CommandContext(ctx, os.Args[0], "-test.run=^TestCacheStartupTimeout$")
			cmd.Env = append(os.Environ(), "OGMEM_STARTUP_TIMEOUT_HELPER=1", "OGMEM_STARTUP_CONFIG="+path, "OGMEM_STARTUP_CLEANED="+cleaned)
			if selected {
				out, err := cmd.CombinedOutput()
				if ctx.Err() != nil || err == nil || !strings.Contains(string(out), "context deadline exceeded") || strings.Contains(string(out), "Starting AGFS server") {
					t.Fatalf("selected cache mount must fail startup on timeout: %v\n%s", err, out)
				}
				return
			}
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
			defer func() {
				_ = cmd.Process.Signal(os.Interrupt)
				_ = cmd.Wait()
			}()
			client := &http.Client{Timeout: 200 * time.Millisecond}
			deadline := time.Now().Add(8 * time.Second)
			for {
				resp, err := client.Get("http://" + addr + "/api/v1/stat?path=/cached")
				if err == nil {
					resp.Body.Close()
					if resp.StatusCode == http.StatusOK {
						break
					}
				}
				if time.Now().After(deadline) {
					out, _ := os.ReadFile(logPath)
					t.Fatalf("healthy cached mount did not start\n%s", out)
				}
				time.Sleep(10 * time.Millisecond)
			}
			for {
				if _, err := os.Stat(cleaned); err == nil {
					break
				}
				if time.Now().After(deadline) {
					t.Fatal("late plugin was not shut down")
				}
				time.Sleep(10 * time.Millisecond)
			}
			resp, err := client.Get("http://" + addr + "/api/v1/stat?path=/late/present")
			if err != nil {
				t.Fatal(err)
			}
			defer resp.Body.Close()
			if resp.StatusCode != http.StatusNotFound {
				t.Fatalf("late initialized plugin was published: status %d", resp.StatusCode)
			}
		})
	}
}
