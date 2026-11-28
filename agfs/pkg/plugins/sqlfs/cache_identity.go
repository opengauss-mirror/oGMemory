package sqlfs

import (
	"fmt"
	"github.com/go-sql-driver/mysql"
	"net/url"
	"path/filepath"
	"strings"
)

func cacheStorageID(backend DBBackend, c map[string]interface{}, db sqlExecutor) (string, error) {
	if backend.GetDriverName() == "sqlite3" {
		// Use the file SQLite actually opened. Re-parsing the DSN here can
		// disagree with driver options, URI authorities or escaped filenames.
		var p string
		if err := db.QueryRow("SELECT file FROM pragma_database_list WHERE name = 'main'").Scan(&p); err != nil {
			return "", fmt.Errorf("read SQLite storage identity: %w", err)
		}
		if p == "" {
			// Shared in-memory databases have no filename in database_list.
			// Keep their URI names comparable across mounts as well.
			dsn := getStringConfig(c, "db_path", "sqlfs.db")
			if !strings.HasPrefix(dsn, "file:") {
				return "", nil
			}
			uri, err := url.Parse(dsn)
			if err != nil {
				return "", fmt.Errorf("invalid SQLite memory storage identity")
			}
			// SQLite applies URI options in order, so the last cache value wins.
			cacheModes := uri.Query()["cache"]
			if uri.Scheme == "file" && len(cacheModes) > 0 && cacheModes[len(cacheModes)-1] == "shared" {
				name := uri.Path
				if uri.Opaque != "" {
					name, err = url.PathUnescape(uri.Opaque)
					if err != nil {
						return "", fmt.Errorf("invalid SQLite memory storage identity")
					}
				}
				return "sqlite-memory:" + name, nil
			}
			return "", nil
		}
		abs, e := filepath.Abs(p)
		if e != nil {
			return "", e
		}
		if resolved, e := filepath.EvalSymlinks(abs); e == nil {
			abs = resolved
		}
		return "sqlite:" + abs, nil
	}
	if dsn := getStringConfig(c, "dsn", ""); dsn != "" {
		parsed, e := mysql.ParseDSN(dsn)
		if e != nil {
			return "", nil
		}
		return "mysql:" + parsed.Net + ":" + parsed.Addr + "/" + parsed.DBName, nil
	}
	return fmt.Sprintf("mysql:tcp:%s:%s/%s", getStringConfig(c, "host", "127.0.0.1"), getStringConfig(c, "port", "4000"), getStringConfig(c, "database", "sqlfs")), nil
}

// ValidateSQLiteCache restricts caching to the path semantics verified for SQLite.
// MySQL/TiDB can use a deployment-specific collation for equality and LIKE.
func (fs *SQLFS) ValidateSQLiteCache() error {
	if fs.backend.GetDriverName() != "sqlite3" {
		return fmt.Errorf("cachefs SQLFS mounts currently require SQLite path semantics")
	}
	return nil
}
