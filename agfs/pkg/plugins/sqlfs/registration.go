package sqlfs

import (
	"context"
	"database/sql"
	"database/sql/driver"
	"errors"

	"github.com/c4pt0r/agfs/agfs-server/pkg/filesystem"
)

type sqlExecutor interface {
	Exec(string, ...any) (sql.Result, error)
	Query(string, ...any) (*sql.Rows, error)
	QueryRow(string, ...any) *sql.Row
	Close() error
}

// A Conn cannot silently reconnect. If the database releases its session lock,
// every subsequent operation on this view fails on the same dead connection.
type registrationConn struct{ *sql.Conn }

func (c registrationConn) Exec(q string, args ...any) (sql.Result, error) {
	return c.ExecContext(context.Background(), q, args...)
}
func (c registrationConn) Query(q string, args ...any) (*sql.Rows, error) {
	return c.QueryContext(context.Background(), q, args...)
}
func (c registrationConn) QueryRow(q string, args ...any) *sql.Row {
	return c.QueryRowContext(context.Background(), q, args...)
}

func (fs *SQLFS) RegistrationView() (filesystem.FileSystem, func(), error) {
	if fs.backend.GetDriverName() != "mysql" {
		return nil, nil, nil
	}
	pool, ok := fs.db.(*sql.DB)
	if !ok {
		return nil, nil, errors.New("nested registration session")
	}
	conn, err := pool.Conn(context.Background())
	if err != nil {
		return nil, nil, err
	}
	// Discard the physical session, including on an ambiguous acquisition error.
	closeSession := func() {
		conn.Raw(func(any) error { return driver.ErrBadConn })
		conn.Close()
	}
	pinned := registrationConn{conn}
	var acquired int
	const lockName = "SHA2(CONCAT('ogmem.registration.', DATABASE()), 256)"
	err = pinned.QueryRow("SELECT GET_LOCK(" + lockName + ", 5)").Scan(&acquired)
	if err == nil && acquired == 1 {
		err = pinned.QueryRow("SELECT IS_USED_LOCK(" + lockName + ") = CONNECTION_ID()").Scan(&acquired)
	}
	if err != nil || acquired != 1 {
		closeSession()
		if err == nil {
			err = filesystem.ErrRegistrationBusy
		}
		return nil, nil, err
	}
	return &SQLFS{db: pinned, backend: fs.backend, mu: fs.mu, pluginName: fs.pluginName, listCache: fs.listCache}, closeSession, nil
}
