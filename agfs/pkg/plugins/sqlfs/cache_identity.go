package sqlfs

import (
	"fmt"
	"github.com/go-sql-driver/mysql"
	"net/url"
	"path/filepath"
	"strings"
)

func cacheStorageID(backend DBBackend, c map[string]interface{}) string {
	if backend.GetDriverName() == "sqlite3" {
		p := getStringConfig(c, "db_path", "sqlfs.db")
		p = normalizeSQLitePath(p)
		if p == "" {
			return ""
		}
		abs, e := filepath.Abs(p)
		if e != nil {
			return ""
		}
		if resolved, e := filepath.EvalSymlinks(abs); e == nil {
			abs = resolved
		}
		return "sqlite:" + abs
	}
	if dsn := getStringConfig(c, "dsn", ""); dsn != "" {
		parsed, e := mysql.ParseDSN(dsn)
		if e != nil {
			return ""
		}
		return "mysql:" + parsed.Net + ":" + parsed.Addr + "/" + parsed.DBName
	}
	return fmt.Sprintf("mysql:tcp:%s:%s/%s", getStringConfig(c, "host", "127.0.0.1"), getStringConfig(c, "port", "4000"), getStringConfig(c, "database", "sqlfs"))
}

// normalizeSQLitePath follows the path forms accepted by mattn/go-sqlite3.
// URI escaping must be decoded before the identity is compared, otherwise a
// file URI and its ordinary path spelling can create two cache domains.
func normalizeSQLitePath(raw string) string {
	raw = strings.SplitN(raw, "?", 2)[0]
	if raw == ":memory:" {
		return ""
	}
	if strings.HasPrefix(raw, "file:") {
		raw = strings.TrimPrefix(raw, "file:")
		if raw == ":memory:" {
			return ""
		}
		decoded, err := url.PathUnescape(raw)
		if err != nil {
			return ""
		}
		raw = decoded
	}
	if raw == "" {
		return ""
	}
	return filepath.Clean(raw)
}

// ValidateSQLiteCache restricts caching to the path semantics verified for SQLite.
// MySQL/TiDB can use a deployment-specific collation for equality and LIKE.
func (fs *SQLFS) ValidateSQLiteCache() error {
	if fs.backend.GetDriverName() != "sqlite3" {
		return fmt.Errorf("cachefs SQLFS mounts currently require SQLite path semantics")
	}
	return nil
}
