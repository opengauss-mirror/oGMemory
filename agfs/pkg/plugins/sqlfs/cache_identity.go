package sqlfs

import (
	"fmt"
	"github.com/go-sql-driver/mysql"
	"path/filepath"
	"strings"
)

func cacheStorageID(backend DBBackend, c map[string]interface{}) string {
	if backend.GetDriverName() == "sqlite3" {
		p := getStringConfig(c, "db_path", "sqlfs.db")
		if p == ":memory:" {
			return ""
		}
		p = strings.TrimPrefix(strings.SplitN(p, "?", 2)[0], "file:")
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
