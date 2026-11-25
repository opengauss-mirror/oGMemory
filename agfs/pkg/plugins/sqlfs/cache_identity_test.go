package sqlfs

import (
	"path/filepath"
	"testing"
)

func TestNormalizeSQLitePathAliases(t *testing.T) {
	root := t.TempDir()
	plain := filepath.Join(root, "store.db")
	uri := "file:" + filepath.ToSlash(root) + "/store%2Edb?mode=rwc"
	if got := normalizeSQLitePath(":memory:"); got != "" {
		t.Fatalf("memory path identity = %q", got)
	}
	if got := normalizeSQLitePath("file::memory:?cache=shared"); got != "" {
		t.Fatalf("shared memory path identity = %q", got)
	}
	if got := normalizeSQLitePath(uri); got != plain {
		t.Fatalf("URI path = %q, want %q", got, plain)
	}
	if got := normalizeSQLitePath("file:" + filepath.ToSlash(root) + "/store%ZZdb"); got != "" {
		t.Fatalf("malformed URI path = %q", got)
	}
	if got := cacheStorageID(NewSQLiteBackend(), map[string]interface{}{"db_path": uri}); got != "sqlite:"+plain {
		t.Fatalf("URI identity = %q, want %q", got, "sqlite:"+plain)
	}
}
