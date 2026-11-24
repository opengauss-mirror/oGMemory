package cachefs

import "testing"

func TestConfigRejectsUnsafePathsAndLimits(t *testing.T) {
	base := Config{Enabled: true, Namespace: "test", Mounts: []string{"/local"}}
	cases := []Config{
		{Enabled: true, Mounts: []string{"/local"}},
		{Enabled: true, Namespace: "test"},
		{Enabled: true, Namespace: "test", Mounts: []string{"/local", "/local/"}},
	}
	for _, p := range []string{"relative", "/local/../other", "/local\\other", "/local\x00"} {
		c := base
		c.BypassPrefixes = []string{p}
		cases = append(cases, c)
	}
	c := base
	c.MaxFileSizeBytes = 17 << 20
	cases = append(cases, c)
	c = base
	c.FileTTLSeconds = 1 << 60
	cases = append(cases, c)
	c = base
	c.TraversalMode = "unknown"
	cases = append(cases, c)
	for _, c := range cases {
		if e := c.Validate(); e == nil {
			t.Errorf("accepted %#v", c)
		}
	}
	if e := base.Validate(); e != nil {
		t.Fatal(e)
	}
	if e := (Config{}).Validate(); e != nil {
		t.Fatal("disabled config must remain compatible", e)
	}
}
