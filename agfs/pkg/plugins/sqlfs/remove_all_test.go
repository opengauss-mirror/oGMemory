//go:build cgo

package sqlfs

import (
	"path/filepath"
	"testing"
)

func TestSQLiteRemoveAllPreservesTheParent(t *testing.T) {
	fs, err := NewSQLFS(NewSQLiteBackend(), map[string]interface{}{
		"db_path": filepath.Join(t.TempDir(), "files.db"),
	})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = fs.db.Close() })

	_, err = fs.db.Exec(`INSERT INTO files (path, is_dir, mode, size, mod_time)
		VALUES ('/locks', 1, 493, 0, 0), ('/locks/first', 1, 493, 0, 0),
		       ('/locks/first/staging', 1, 493, 0, 0), ('/locks/second', 1, 493, 0, 0)`)
	if err != nil {
		t.Fatal(err)
	}
	_, err = fs.db.Exec(`WITH RECURSIVE batch(i) AS (VALUES (1) UNION ALL SELECT i + 1 FROM batch WHERE i < 1001)
		INSERT INTO files (path, is_dir, mode, size, mod_time)
		SELECT '/locks/first/staging/event-' || i || '.json', 0, 420, 0, 0 FROM batch`)
	if err != nil {
		t.Fatal(err)
	}
	if err := fs.RemoveAll("/locks/first"); err != nil {
		t.Fatal(err)
	}
	var remaining int
	err = fs.db.QueryRow("SELECT COUNT(*) FROM files WHERE path = '/locks/first' OR path LIKE '/locks/first/%'").Scan(&remaining)
	if err != nil || remaining != 0 {
		t.Fatalf("expected the whole subtree to be removed, got %d rows: %v", remaining, err)
	}
	children, err := fs.ReadDir("/locks")
	if err != nil || len(children) != 1 || children[0].Name != "second" {
		t.Fatalf("expected the sibling lock alone, got %v: %v", children, err)
	}

	if err := fs.RemoveAll("/"); err != nil {
		t.Fatal(err)
	}
	children, err = fs.ReadDir("/")
	if err != nil || len(children) != 0 {
		t.Fatalf("expected an empty root, got %v: %v", children, err)
	}
	if _, err := fs.Stat("/"); err != nil {
		t.Fatal(err)
	}
}
