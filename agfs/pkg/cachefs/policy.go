package cachefs

import (
	"path"
	"regexp"
	"strings"
)

// Decision distinguishes correctness exclusions from the initial low-value policy.
type Decision string

const (
	Cacheable       Decision = "cacheable"
	MandatoryBypass Decision = "mandatory"
	DefaultBypass   Decision = "default"
	UserBypass      Decision = "user"
)

var lockDir = regexp.MustCompile(`^\..+\.(resource-mutation-lock|update-lock)(-[a-zA-Z0-9_-]+)?$`)
var staging = regexp.MustCompile(`^\..+\.(create|resource-create)-[a-zA-Z0-9_-]+$`)
var tempMeta = regexp.MustCompile(`^\.(meta\.(update|mutation)|delete_tombstone\.mutation)-.+\.json$`)

// Policy takes the canonical mount-relative path. Rules match components, never
// arbitrary substrings or JSON contents. Formal metadata and recovery files cache.
func Policy(p, mount string, prefixes []string) Decision {
	p = path.Clean("/" + strings.TrimPrefix(p, "/"))
	parts := strings.Split(strings.Trim(p, "/"), "/")
	d := Cacheable
	for i, s := range parts {
		if s == ".path.ovlock" || strings.HasSuffix(s, ".lock") || strings.HasSuffix(s, ".lck") || lockDir.MatchString(s) {
			return MandatoryBypass
		}
		if staging.MatchString(s) || tempMeta.MatchString(s) {
			d = DefaultBypass
		}
		if s == ".outbox" || s == ".outbox-done" {
			for _, v := range parts[i+1:] {
				if strings.HasPrefix(v, ".tmp_") || strings.HasPrefix(v, ".mutation-") || strings.HasSuffix(v, ".processing") || (s == ".outbox-done" && strings.HasSuffix(v, ".tmp")) {
					d = DefaultBypass
				}
			}
		}
		if s == "code_workspace_sync" && i >= 3 && parts[i-1] == "_system" && parts[i-3] == "accounts" {
			for _, v := range parts[i+1:] {
				if strings.Contains(v, ".mutation-") {
					d = DefaultBypass
				}
			}
		}
		if s == "tool-results" && i >= 8 && parts[i-8] == "accounts" && parts[i-6] == "user" && parts[i-4] == "agents" && parts[i-2] == "sessions" && len(parts) == i+3 && parts[i+2] == "_writing.json" {
			d = DefaultBypass
		}
		if s == "trace-store" && i > 0 && parts[i-1] == ".ogmemory" {
			tail := parts[i+1:]
			if len(tail) >= 3 && tail[0] == "accounts" {
				if tail[2] == "candidates" || tail[2] == "candidate_claims" || tail[2] == "state.json" {
					return MandatoryBypass
				}
				if tail[2] == "candidate_work" {
					d = DefaultBypass
				}
			}
			if len(tail) > 0 && (tail[0] == "work-accounts" || tail[0] == "work-accounts-index.json") {
				d = DefaultBypass
			}
			for _, v := range tail {
				if strings.Contains(v, ".tmp-") || strings.HasSuffix(v, ".bak") {
					d = DefaultBypass
				}
			}
		}
	}
	if d != Cacheable {
		return d
	}
	full := path.Join(mount, p)
	for _, prefix := range prefixes {
		if within(full, path.Clean(prefix)) {
			return UserBypass
		}
	}
	return Cacheable
}
