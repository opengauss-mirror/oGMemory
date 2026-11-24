package cachefs

import (
	"github.com/c4pt0r/agfs/agfs-server/pkg/filesystem"
	"os"
	"sync"
)

// Facets are attached only when the original backend implements the interface.
// Unsupported optional methods must remain absent, preserving type assertions.
type touch struct{ f *FS }

func (t *touch) Touch(p string) error {
	return t.f.mutation([]string{p}, false, func() error { return t.f.backend.(filesystem.Toucher).Touch(p) })
}

type truncate struct{ f *FS }

func (t *truncate) Truncate(p string, n int64) error {
	return t.f.mutation([]string{p}, false, func() error { return t.f.backend.(filesystem.Truncater).Truncate(p, n) })
}

type randomWrite struct{ f *FS }

func (t *randomWrite) WriteAt(p string, b []byte, o int64) (int64, error) {
	var n int64
	e := t.f.mutation([]string{p}, false, func() error { var e error; n, e = t.f.backend.(filesystem.RandomWriter).WriteAt(p, b, o); return e })
	return n, e
}

type syncFile struct{ f *FS }

func (t *syncFile) Sync(p string) error {
	return t.f.mutation([]string{p}, false, func() error { return t.f.backend.(filesystem.Syncer).Sync(p) })
}

type stream struct{ f *FS }

func (t *stream) OpenStream(p string) (filesystem.StreamReader, error) {
	t.f.ops.RLock()
	defer t.f.ops.RUnlock()
	if t.f.isClosed() {
		return nil, os.ErrClosed
	}
	r, e := t.f.backend.(filesystem.Streamer).OpenStream(p)
	if e != nil {
		return nil, e
	}
	t.f.addReader()
	return &cachedStream{StreamReader: r, f: t.f}, nil
}

type handles struct {
	f    *FS
	mu   sync.Mutex
	open map[int64]*cachedHandle
}

func (t *handles) OpenHandle(p string, flags filesystem.OpenFlag, mode uint32) (filesystem.FileHandle, error) {
	t.f.ops.Lock()
	defer t.f.ops.Unlock()
	if t.f.isClosed() {
		return nil, os.ErrClosed
	}
	writable := flags&(filesystem.O_WRONLY|filesystem.O_RDWR|filesystem.O_APPEND|filesystem.O_CREATE|filesystem.O_TRUNC) != 0
	if writable {
		t.f.startWriter()
	}
	h, e := t.f.backend.(filesystem.HandleFS).OpenHandle(p, flags, mode)
	if e != nil {
		if writable {
			t.f.endWriter(nil)
		}
		return nil, e
	}
	if !writable {
		t.f.addReader()
	}
	w := &cachedHandle{FileHandle: h, owner: t, writable: writable}
	t.mu.Lock()
	t.open[h.ID()] = w
	t.mu.Unlock()
	return w, nil
}
func (t *handles) GetHandle(id int64) (filesystem.FileHandle, error) {
	t.mu.Lock()
	defer t.mu.Unlock()
	h := t.open[id]
	if h == nil {
		return nil, filesystem.ErrNotFound
	}
	return h, nil
}
func (t *handles) CloseHandle(id int64) error {
	h, e := t.GetHandle(id)
	if e != nil {
		return e
	}
	return h.Close()
}

type cachedHandle struct {
	filesystem.FileHandle
	owner    *handles
	writable bool
	once     sync.Once
	err      error
}

func (h *cachedHandle) Close() error {
	h.once.Do(func() {
		h.err = h.FileHandle.Close()
		if h.writable {
			h.owner.f.endWriter(h.err)
		} else if h.err == nil {
			h.owner.f.endReader()
		}
		h.owner.mu.Lock()
		delete(h.owner.open, h.ID())
		h.owner.mu.Unlock()
	})
	return h.err
}

type cachedStream struct {
	filesystem.StreamReader
	f    *FS
	once sync.Once
	err  error
}

func (s *cachedStream) Close() error {
	s.once.Do(func() {
		s.err = s.StreamReader.Close()
		if s.err == nil {
			s.f.endReader()
		}
	})
	return s.err
}
