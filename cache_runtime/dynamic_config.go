package cacheruntime

import (
	"fmt"
	"path/filepath"
)

func validateDynamic(p map[string]interface{}) error {
	v, ok := p["library"].(string)
	if !ok || !filepath.IsAbs(v) {
		return fmt.Errorf("%w: dynamic library must be an absolute path", ErrInvalidArgument)
	}
	return nil
}
