package main

import (
	"context"
	"time"

	"github.com/c4pt0r/agfs/agfs-server/pkg/plugin"
)

var startupMountTimeout = 30 * time.Second

// initializeStartupPlugin bounds startup's wait for the legacy Initialize API.
// Only the caller can publish a mount. A result that arrives after cancellation
// is discarded, and its initialized plugin is shut down by the worker.
func initializeStartupPlugin(ctx context.Context, p plugin.ServicePlugin, cfg map[string]interface{}) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	result := make(chan error)
	go func() {
		err := p.Initialize(cfg)
		select {
		case result <- err:
		case <-ctx.Done():
			if err == nil {
				_ = p.Shutdown()
			}
		}
	}()
	select {
	case err := <-result:
		if ctxErr := ctx.Err(); ctxErr != nil {
			if err == nil {
				go p.Shutdown()
			}
			return ctxErr
		}
		return err
	case <-ctx.Done():
		return ctx.Err()
	}
}
