//go:build cgo && (linux || darwin)

package cacheruntime

import (
	"context"
	"errors"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"testing"
	"time"
)

func compileNativeTest(t *testing.T, args ...string) {
	t.Helper()
	cc, e := exec.LookPath("cc")
	if e != nil {
		t.Skip("C compiler unavailable")
	}
	if runtime.GOOS == "darwin" {
		if root := os.Getenv("SDKROOT"); root != "" {
			args = append(args, "-isysroot", root)
		}
	}
	if b, e := exec.Command(cc, args...).CombinedOutput(); e != nil {
		t.Fatalf("compile: %v\n%s", e, b)
	}
}

func compileFixture(t *testing.T, bad bool) string {
	t.Helper()
	out := filepath.Join(t.TempDir(), "provider.so")
	args := []string{"-shared", "-fPIC", "-pthread", "-Iinclude", "testdata/provider.c", "-o", out}
	if runtime.GOOS == "darwin" {
		args[0] = "-dynamiclib"
	}
	if bad {
		args = append(args, "-DBAD_ABI=1")
	}
	compileNativeTest(t, args...)
	return out
}

func TestDynamicContract(t *testing.T) {
	r, err := New(Config{Provider: "dynamic", CommandTimeoutMS: 500, Params: map[string]interface{}{
		"library": compileFixture(t, false),
	}})
	if err != nil {
		t.Fatal(err)
	}
	testKVContract(t, r, time.Sleep)
}

func TestDynamicInvalidOutputs(t *testing.T) {
	r, err := New(Config{Provider: "dynamic", CommandTimeoutMS: 500, Params: map[string]interface{}{
		"library": compileFixture(t, false), "mode": "badoutput",
	}})
	if err != nil {
		t.Fatal(err)
	}
	ctx := context.Background()
	defer r.Close(ctx)
	// Empty values are valid, while this fixture corrupts nonempty outputs.
	if err := r.Set(ctx, "empty", nil, 0); err != nil {
		t.Fatal(err)
	}
	if v, err := r.Get(ctx, "empty"); err != nil || !v.Found || len(v.Data) != 0 {
		t.Fatal(v, err)
	}
	if err := r.Set(ctx, "k", []byte{0, 255, 1}, 0); err != nil {
		t.Fatal(err)
	}
	if _, err := r.Get(ctx, "k"); !errors.Is(err, ErrInvalidData) {
		t.Fatal("malformed buffer accepted", err)
	}
	if _, err := New(Config{Provider: "dynamic", Params: map[string]interface{}{"library": compileFixture(t, true)}}); err == nil {
		t.Fatal("accepted wrong ABI")
	}
}

func TestDynamicCompatibilityEntry(t *testing.T) {
	lib := compileFixture(t, false)
	out := filepath.Join(t.TempDir(), "entry-check")
	args := []string{"testdata/entry_check.c", "-o", out}
	if runtime.GOOS == "linux" {
		args = append(args, "-ldl")
	}
	compileNativeTest(t, args...)
	if b, err := exec.Command(out, lib).CombinedOutput(); err != nil {
		t.Fatalf("v1 entry compatibility: %v\n%s", err, b)
	}
}

func TestDynamicAliasedOutputAndBatchDelete(t *testing.T) {
	ctx := context.Background()
	r, err := New(Config{Provider: "dynamic", CommandTimeoutMS: 500, Params: map[string]interface{}{
		"library": compileFixture(t, false), "mode": "aliasoutput,batchdelete",
	}})
	if err != nil {
		t.Fatal(err)
	}
	defer r.Close(ctx)
	for _, key := range []string{"first", "second"} {
		if err := r.Set(ctx, key, []byte("value"), 0); err != nil {
			t.Fatal(err)
		}
		if _, err := r.Get(ctx, key); !errors.Is(err, ErrInvalidData) {
			t.Fatalf("aliased output: %v", err)
		}
	}
	if count, err := r.Delete(ctx, []string{"first", "", "second", "first"}); err != nil || count != 2 {
		t.Fatalf("batch delete: count=%d err=%v", count, err)
	}
}
