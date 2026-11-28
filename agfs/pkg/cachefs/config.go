package cachefs

import (
	"fmt"
	"path"
	"strings"
)

type Config struct {
	Enabled            bool     `yaml:"enabled" json:"enabled"`
	Namespace          string   `yaml:"namespace" json:"namespace"`
	Mounts             []string `yaml:"mounts" json:"mounts"`
	MaxFileSizeBytes   int      `yaml:"max_file_size_bytes" json:"max_file_size_bytes"`
	MaxDirEntries      int      `yaml:"max_dir_entries" json:"max_dir_entries"`
	MaxDirEncodedBytes int      `yaml:"max_dir_encoded_bytes" json:"max_dir_encoded_bytes"`
	FileTTLSeconds     int      `yaml:"file_ttl_seconds" json:"file_ttl_seconds"`
	DirTTLSeconds      int      `yaml:"dir_ttl_seconds" json:"dir_ttl_seconds"`
	TraversalMode      string   `yaml:"traversal_mode" json:"traversal_mode"`
	BypassPrefixes     []string `yaml:"bypass_prefixes" json:"bypass_prefixes"`
}

func (c Config) Defaults() Config {
	if c.MaxFileSizeBytes == 0 {
		c.MaxFileSizeBytes = 1 << 20
	}
	if c.MaxDirEntries == 0 {
		c.MaxDirEntries = 4096
	}
	if c.MaxDirEncodedBytes == 0 {
		c.MaxDirEncodedBytes = 4 << 20
	}
	if c.FileTTLSeconds == 0 {
		c.FileTTLSeconds = 300
	}
	if c.DirTTLSeconds == 0 {
		c.DirTTLSeconds = 30
	}
	if c.TraversalMode == "" {
		c.TraversalMode = "backend"
	}
	return c
}
func ValidPath(p string) bool {
	if !strings.HasPrefix(p, "/") || strings.ContainsAny(p, "\x00\\") {
		return false
	}
	for _, s := range strings.Split(p, "/") {
		if s == ".." {
			return false
		}
	}
	return true
}
func (c Config) Validate() error {
	if !c.Enabled {
		return nil
	}
	c = c.Defaults()
	if strings.TrimSpace(c.Namespace) == "" || len(c.Mounts) == 0 {
		return fmt.Errorf("cachefs requires namespace and mounts")
	}
	if c.MaxFileSizeBytes < 1 || c.MaxDirEntries < 1 || c.MaxDirEncodedBytes < 1 || c.FileTTLSeconds < 1 || c.DirTTLSeconds < 1 {
		return fmt.Errorf("cachefs limits and TTLs must be positive")
	}
	if c.MaxFileSizeBytes > 16<<20 || c.MaxDirEncodedBytes > 32<<20 || c.MaxDirEntries > 65536 || c.FileTTLSeconds > 86400 || c.DirTTLSeconds > 86400 {
		return fmt.Errorf("cachefs limits exceed file 16 MiB, directory 32 MiB/65536 entries, or TTL 86400 seconds")
	}
	if c.TraversalMode != "backend" && c.TraversalMode != "cached_traversal" {
		return fmt.Errorf("invalid cachefs traversal_mode")
	}
	seen := map[string]bool{}
	for _, m := range c.Mounts {
		if !ValidPath(m) || seen[path.Clean(m)] {
			return fmt.Errorf("invalid or duplicate cachefs mount %q", m)
		}
		seen[path.Clean(m)] = true
	}
	for _, p := range c.BypassPrefixes {
		if !ValidPath(p) {
			return fmt.Errorf("invalid cachefs bypass prefix %q", p)
		}
	}
	return nil
}
func (c Config) Selected(mount string) bool {
	if !c.Enabled {
		return false
	}
	for _, m := range c.Mounts {
		if path.Clean(m) == path.Clean(mount) {
			return true
		}
	}
	return false
}
func within(p, root string) bool { return root == "/" || p == root || strings.HasPrefix(p, root+"/") }
