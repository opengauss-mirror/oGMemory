package localfs

import (
	"errors"
	"os"
	"path/filepath"
	"testing"

	"github.com/c4pt0r/agfs/agfs-server/pkg/filesystem"
)

func TestRenameExistingDirectoryIsConfirmedConflict(t *testing.T) {
	root := t.TempDir()
	for _, name := range []string{"source", "target"} {
		if err := os.Mkdir(filepath.Join(root, name), 0700); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(filepath.Join(root, name, "content"), []byte(name), 0600); err != nil {
			t.Fatal(err)
		}
	}
	fs, err := NewLocalFS(root)
	if err != nil {
		t.Fatal(err)
	}
	if err := fs.Rename("/source", "/target"); !errors.Is(err, filesystem.ErrAlreadyExists) {
		t.Fatalf("expected conflict, got %v", err)
	}
	for _, name := range []string{"source", "target"} {
		data, err := os.ReadFile(filepath.Join(root, name, "content"))
		if err != nil || string(data) != name {
			t.Fatalf("%s changed: %q, %v", name, data, err)
		}
	}
}
