package mountablefs

import (
	"database/sql"
	"errors"
	"net/url"
	"path/filepath"
	"strings"
	"sync"

	"github.com/c4pt0r/agfs/agfs-server/pkg/filesystem"
	"github.com/c4pt0r/agfs/agfs-server/pkg/plugin"
	iradix "github.com/hashicorp/go-immutable-radix"
	_ "github.com/mattn/go-sqlite3"
)

var ErrRegistrationUnsupported = errors.New("mount does not support workspace registration sessions")

type Registration struct {
	FS      *MountableFS
	release func()
	mu      sync.RWMutex
	closed  bool
}

// Enter pins the storage lock until this operation returns, even if its owner disconnects.
func (r *Registration) Enter() (func(), bool) {
	r.mu.RLock()
	if r.closed {
		r.mu.RUnlock()
		return nil, false
	}
	return r.mu.RUnlock, true
}

func (r *Registration) Close() {
	r.mu.Lock()
	defer r.mu.Unlock()
	if !r.closed {
		r.closed = true
		r.release()
	}
}

type registrationPlugin struct {
	plugin.ServicePlugin
	fs filesystem.FileSystem
}

func (p registrationPlugin) GetFileSystem() filesystem.FileSystem { return p.fs }

// Lock the same backing storage across HTTP servers and native workers. A separate
// SQLite file keeps the registration session independent of ordinary SQLFS writes.
func (mfs *MountableFS) LockRegistration(path string) (*Registration, error) {
	path = filesystem.NormalizePath(path)
	if !strings.HasSuffix(path, "/_system/code_workspace_registration.lock") {
		return nil, ErrRegistrationUnsupported
	}
	mount, relative, ok := mfs.findMount(path)
	if !ok {
		return nil, ErrRegistrationUnsupported
	}
	fs := mount.Plugin.GetFileSystem()
	if provider, ok := fs.(interface {
		RegistrationView() (filesystem.FileSystem, func(), error)
	}); ok {
		scoped, release, err := provider.RegistrationView()
		if err != nil {
			return nil, err
		}
		if scoped != nil {
			if scoped == fs {
				return &Registration{FS: mfs, release: release}, nil
			}
			// All registry I/O on a database session must use its pinned connection.
			// A nested accounts mount would otherwise route a stale writer elsewhere.
			root := strings.TrimSuffix(path, "/_system/code_workspace_registration.lock")
			for _, branch := range []string{root + "/accounts", root + "/_system"} {
				owner, _, found := mfs.findMount(branch)
				if !found || owner != mount {
					release()
					return nil, ErrRegistrationUnsupported
				}
				for _, other := range mfs.GetMounts() {
					if strings.HasPrefix(other.Path, branch+"/") {
						release()
						return nil, ErrRegistrationUnsupported
					}
				}
			}
			point := *mount
			point.Plugin = registrationPlugin{point.Plugin, scoped}
			point.FileSystem = scoped
			point.cache = nil
			if mount.cache != nil {
				endWrite, err := mount.cache.BeginScopedWrite()
				if err != nil {
					release()
					return nil, err
				}
				closeSession := release
				release = func() {
					defer endWrite()
					closeSession()
				}
			}
			tree, _, _ := iradix.New().Insert([]byte(point.Path), &point)
			view := new(MountableFS)
			view.mountTree.Store(tree)
			return &Registration{FS: view, release: release}, nil
		}
	}
	provider, ok := fs.(interface{ RegistrationLockFile(string) (string, error) })
	if !ok {
		return nil, ErrRegistrationUnsupported
	}
	filename, err := provider.RegistrationLockFile(relative)
	if err != nil {
		return nil, err
	}
	if filename == "" {
		return nil, ErrRegistrationUnsupported
	}
	pathURI := filepath.ToSlash(filename)
	if !strings.HasPrefix(pathURI, "/") {
		pathURI = "/" + pathURI
	}
	uri := url.URL{Scheme: "file", Path: pathURI, RawQuery: "_busy_timeout=5000&_txlock=immediate"}
	db, err := sql.Open("sqlite3", uri.String())
	if err != nil {
		return nil, err
	}
	tx, err := db.Begin()
	if err != nil {
		db.Close()
		if err.Error() == "database is locked" {
			return nil, filesystem.ErrRegistrationBusy
		}
		return nil, err
	}
	return &Registration{FS: mfs, release: func() { tx.Rollback(); db.Close() }}, nil
}
