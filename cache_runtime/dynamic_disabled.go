//go:build !cgo || (!linux && !darwin)

package cacheruntime

import "fmt"

func newDynamic(map[string]interface{}) (Provider, error) {
	return nil, fmt.Errorf("%w: dynamic requires cgo on Linux or macOS", ErrUnsupported)
}
