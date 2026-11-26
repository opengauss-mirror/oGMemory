package sqlfs

import (
	"os"
	"os/exec"
	"path/filepath"
	"testing"
)

func TestSQLiteStorageIdentity(t *testing.T) {
	// A separate working directory lets the driver open a filename beginning
	// with '?' without changing the working directory of other tests.
	if os.Getenv("OGMEM_SQLITE_IDENTITY_HELPER") != "1" {
		cmd := exec.Command(os.Args[0], "-test.run=^TestSQLiteStorageIdentity$", "-test.v")
		cmd.Dir = t.TempDir()
		cmd.Env = append(os.Environ(), "OGMEM_SQLITE_IDENTITY_HELPER=1")
		if out, err := cmd.CombinedOutput(); err != nil {
			t.Fatalf("SQLite identity subprocess: %v\n%s", err, out)
		}
		return
	}
	for _, tc := range []struct {
		name, plain, alias string
	}{
		{"escaped-dot", "store.db", "file:store%2Edb?mode=rwc"},
		{"driver-options", "store.db", "store.db?_busy_timeout=5000"},
		{"leading-question", "?store.db", "file:%3Fstore.db"},
		{"question-in-name", "?store.db", "file:%3Fstore.db?_busy_timeout=5000"},
		{"memory-prefix-file", ":memory:backup.db", "file::memory:backup.db"},
		{"fragment", "store.db", "file:store.db#ignored"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var want string
			for _, dsn := range []string{tc.plain, tc.alias} {
				fs, err := NewSQLFS(NewSQLiteBackend(), map[string]interface{}{"db_path": dsn})
				if err != nil {
					t.Fatalf("open %q: %v", dsn, err)
				}
				t.Cleanup(func() { _ = fs.Close() })
				got := fs.CacheStorageIdentity().BackendID
				if want == "" {
					abs, err := filepath.Abs(tc.plain)
					if err != nil {
						t.Fatal(err)
					}
					resolved, err := filepath.EvalSymlinks(abs)
					if err != nil {
						t.Fatal(err)
					}
					want = "sqlite:" + resolved
				}
				if got != want {
					t.Fatalf("%q identity = %q, want %q", dsn, got, want)
				}
			}
		})
	}
	for _, dsn := range []string{":memory:", ":memory:?cache=shared", "file:%3Amemory%3A", "file:named?mode=memory&cache=private"} {
		t.Run(dsn, func(t *testing.T) {
			fs, err := NewSQLFS(NewSQLiteBackend(), map[string]interface{}{"db_path": dsn})
			if err != nil {
				t.Fatal(err)
			}
			defer fs.Close()
			if id := fs.CacheStorageIdentity().BackendID; id != "" {
				t.Fatalf("memory database %q identity = %q", dsn, id)
			}
		})
	}
	for _, dsn := range []string{"file:named?mode=memory&cache=shared", "file:na%6Ded?cache=shared&mode=memory"} {
		fs, err := NewSQLFS(NewSQLiteBackend(), map[string]interface{}{"db_path": dsn})
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(func() { _ = fs.Close() })
		if id := fs.CacheStorageIdentity().BackendID; id != "sqlite-memory:named" {
			t.Fatalf("shared memory database %q identity = %q", dsn, id)
		}
	}

}
