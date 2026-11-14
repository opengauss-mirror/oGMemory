package sqlfs

import (
	"database/sql"
	"fmt"
	"os"
	"sync"
	"testing"
)

// Uses two real database sessions. Only the connection opened by this test is killed.
func TestLostMySQLRegistrationSessionCannotWrite(t *testing.T) {
	dsn := os.Getenv("AGFS_TEST_MYSQL_DSN")
	if dsn == "" {
		t.Skip("set AGFS_TEST_MYSQL_DSN to an isolated test database")
	}
	db, err := sql.Open("mysql", dsn)
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()
	fs := &SQLFS{db: db, mu: new(sync.RWMutex), backend: NewTiDBBackend()}
	old, release, err := fs.RegistrationView()
	if err != nil {
		t.Fatal(err)
	}
	defer release()
	var id uint64
	oldDB := old.(*SQLFS).db
	if err := oldDB.QueryRow("SELECT CONNECTION_ID()").Scan(&id); err != nil {
		t.Fatal(err)
	}
	if _, err := db.Exec(fmt.Sprintf("KILL CONNECTION %d", id)); err != nil {
		t.Fatal(err)
	}
	next, unlock, err := fs.RegistrationView()
	if err != nil {
		t.Fatal(err)
	}
	defer unlock()
	const write = "CREATE TEMPORARY TABLE registration_session_test (id INT)"
	if _, err := next.(*SQLFS).db.Exec(write); err != nil {
		t.Fatal(err)
	}
	if _, err := oldDB.Exec(write); err == nil {
		t.Fatal("stale writer reconnected after losing its database lock")
	}
}
