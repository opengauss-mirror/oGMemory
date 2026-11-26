package main

import (
	"context"
	"fmt"
	"time"

	"github.com/c4pt0r/agfs/agfs-server/pkg/plugin"
)

const startupMountTimeout = 30 * time.Second

var newStartupContext = func() (context.Context, context.CancelFunc) {
	return context.WithTimeout(context.Background(), startupMountTimeout)
}

// prepareStartupPlugin bounds startup validation and initialization.
// Only the caller can publish a mount. A result that arrives after cancellation
// is discarded, and its initialized plugin is shut down by the worker.
func prepareStartupPlugin(ctx context.Context, p plugin.ServicePlugin, cfg map[string]interface{}) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	result := make(chan error)
	go func() {
		err := preparePlugin(ctx, p, cfg)
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

// preparePlugin owns the preparation sequence for both startup paths.
func preparePlugin(ctx context.Context, p plugin.ServicePlugin, cfg map[string]interface{}) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	if err := p.Validate(cfg); err != nil {
		return fmt.Errorf("validate plugin: %w", err)
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	if err := p.Initialize(cfg); err != nil {
		return fmt.Errorf("initialize plugin: %w", err)
	}
	return nil
}
