package main

import (
	"context"
	"flag"
	"os"
	"os/exec"
	"path/filepath"
	"testing"
	"time"
)

func runStartupMain() {
	flag.CommandLine = flag.NewFlagSet("agfs", flag.ExitOnError)
	os.Args = []string{"agfs", "-c", os.Getenv("OGMEM_STARTUP_CONFIG")}
	main()
}

func startupTestCommand(t *testing.T, testName, cfg string, env ...string) (*exec.Cmd, context.Context) {
	t.Helper()
	path := filepath.Join(t.TempDir(), "config.yaml")
	if err := os.WriteFile(path, []byte(cfg), 0600); err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	t.Cleanup(cancel)
	cmd := exec.CommandContext(ctx, os.Args[0], "-test.run=^"+testName+"$")
	cmd.Env = append(os.Environ(), append([]string{"OGMEM_STARTUP_CONFIG=" + path}, env...)...)
	return cmd, ctx
}

func startupMarker(path string) {
	if err := os.WriteFile(path, nil, 0600); err != nil {
		panic(err)
	}
}
func awaitStartupMarker(path string) {
	deadline := time.Now().Add(20 * time.Second)
	for {
		if _, err := os.Stat(path); err == nil {
			return
		}
		if time.Now().After(deadline) {
			panic("startup marker missing: " + path)
		}
		time.Sleep(5 * time.Millisecond)
	}
}
func waitStartupMarker(t *testing.T, ctx context.Context, path string) {
	t.Helper()
	for {
		if _, err := os.Stat(path); err == nil {
			return
		}
		select {
		case <-ctx.Done():
			t.Fatalf("startup marker missing %s: %v", path, ctx.Err())
		case <-time.After(5 * time.Millisecond):
		}
	}
}
