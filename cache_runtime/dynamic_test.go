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

func compileFixture(t *testing.T, bad bool) string {
	t.Helper()
	cc, e := exec.LookPath("cc")
	if e != nil {
		t.Skip("C compiler unavailable")
	}
	out := filepath.Join(t.TempDir(), "provider.so")
	args := []string{"-shared", "-fPIC", "-pthread", "-Iinclude", "testdata/provider.c", "-o", out}
	if runtime.GOOS == "darwin" {
		args[0] = "-dynamiclib"
		if root := os.Getenv("SDKROOT"); root != "" {
			args = append(args, "-isysroot", root)
		}
	}
	if bad {
		args = append(args, "-DBAD_ABI=1")
	}
	if b, e := exec.Command(cc, args...).CombinedOutput(); e != nil {
		t.Fatalf("compile: %v\n%s", e, b)
	}
	return out
}
func TestDynamicKVAndInvalidOutputs(t *testing.T) {
	lib := compileFixture(t, false)
	for _, bad := range []bool{false, true} {
		params := map[string]interface{}{"library": lib}
		if bad {
			params["mode"] = "badoutput"
		}
		r, e := New(Config{Provider: "dynamic", CommandTimeoutMS: 500, Params: params})
		if e != nil {
			t.Fatal(e)
		}
		ctx := context.Background()
		if e = r.Ping(ctx); e != nil {
			t.Fatal(e)
		}
		if v, e := r.Get(ctx, "k"); e != nil || v.Found {
			t.Fatal("miss", v, e)
		}
		if e = r.Set(ctx, "k", nil, 0); e != nil {
			t.Fatal(e)
		}
		if v, e := r.Get(ctx, "k"); e != nil || !v.Found || len(v.Data) != 0 {
			t.Fatal("empty", v, e)
		}
		if e = r.Set(ctx, "k", []byte{0, 255, 1}, time.Second); e != nil {
			t.Fatal(e)
		}
		v, e := r.Get(ctx, "k")
		if bad {
			if !errors.Is(e, ErrInvalidData) {
				t.Fatal("malformed buffer accepted", e)
			}
		} else if e != nil || !v.Found || len(v.Data) != 3 || v.Data[1] != 255 {
			t.Fatal(v, e)
		}
		if !bad {
			if e := r.Set(ctx, "other", nil, 0); e != nil {
				t.Fatal(e)
			}
			values, e := r.MGet(ctx, []string{"k", "absent", "other", "k"})
			if e != nil || len(values) != 4 || values[1].Found || !values[2].Found || len(values[2].Data) != 0 || values[3].Data[1] != 255 {
				t.Fatal(values, e)
			}
			if e := r.Set(ctx, "expires", []byte("value"), 5*time.Millisecond); e != nil {
				t.Fatal(e)
			}
			time.Sleep(20 * time.Millisecond)
			if v, e := r.Get(ctx, "expires"); e != nil || v.Found {
				t.Fatal("TTL", v, e)
			}
			if v, e := r.Get(ctx, "other"); e != nil || !v.Found {
				t.Fatal("zero TTL", v, e)
			}
		}

		if n, e := r.Delete(ctx, []string{"k", "k"}); e != nil || n != 1 {
			t.Fatal(n, e)
		}
		if e = r.Close(ctx); e != nil {
			t.Fatal(e)
		}
		if e = r.Close(ctx); e != nil {
			t.Fatal(e)
		}
	}
	if _, e := New(Config{Provider: "dynamic", Params: map[string]interface{}{"library": compileFixture(t, true)}}); e == nil {
		t.Fatal("accepted wrong ABI")
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
