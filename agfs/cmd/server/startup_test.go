package main

import (
	"context"
	"errors"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/c4pt0r/agfs/agfs-server/pkg/plugin"
)

type blockedStartupPlugin struct {
	plugin.ServicePlugin
	started, release, stopped chan struct{}
}

func (p *blockedStartupPlugin) Validate(map[string]interface{}) error { return nil }
func (p *blockedStartupPlugin) Initialize(map[string]interface{}) error {
	close(p.started)
	<-p.release
	return nil
}
func (p *blockedStartupPlugin) Shutdown() error { close(p.stopped); return nil }

func TestStartupDiscardsAndClosesLateInitialization(t *testing.T) {
	p := &blockedStartupPlugin{started: make(chan struct{}), release: make(chan struct{}), stopped: make(chan struct{})}
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	var release sync.Once
	t.Cleanup(func() { release.Do(func() { close(p.release) }) })
	result := make(chan error, 1)
	go func() { result <- prepareStartupPlugin(ctx, p, nil) }()
	waitPreparationStarted(t, p.started, result)
	cancel()
	select {
	case err := <-result:
		if !errors.Is(err, context.Canceled) {
			t.Fatal(err)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("startup did not stop waiting after cancellation")
	}
	release.Do(func() { close(p.release) })
	select {
	case <-p.stopped:
	case <-time.After(5 * time.Second):
		t.Fatal("late initialized plugin was not closed")
	}
}

type blockedValidationPlugin struct {
	plugin.ServicePlugin
	started, release chan struct{}
	initialized      atomic.Bool
}

func (p *blockedValidationPlugin) Validate(map[string]interface{}) error {
	close(p.started)
	<-p.release
	return nil
}
func (p *blockedValidationPlugin) Initialize(map[string]interface{}) error {
	p.initialized.Store(true)
	return nil
}

// Both paths must stop waiting or stop before Initialize. Only the synchronous
// result proves preparation has actually finished after Validate returns.
func TestStartupDiscardsBlockedValidation(t *testing.T) {
	for _, timed := range []bool{false, true} {
		name := "synchronous"
		if timed {
			name = "timed"
		}
		t.Run(name, func(t *testing.T) {
			p := &blockedValidationPlugin{started: make(chan struct{}), release: make(chan struct{})}
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			var release sync.Once
			t.Cleanup(func() { release.Do(func() { close(p.release) }) })
			result := make(chan error, 1)
			go func() {
				if timed {
					result <- prepareStartupPlugin(ctx, p, nil)
				} else {
					result <- preparePlugin(ctx, p, nil)
				}
			}()
			waitPreparationStarted(t, p.started, result)
			cancel()
			if !timed {
				release.Do(func() { close(p.release) })
			}
			select {
			case err := <-result:
				if !errors.Is(err, context.Canceled) {
					t.Fatal(err)
				}
			case <-time.After(5 * time.Second):
				t.Fatal("preparation did not finish after cancellation")
			}
			if !timed && p.initialized.Load() {
				t.Fatal("initialized after validation cancellation")
			}
		})
	}
}

func waitPreparationStarted(t *testing.T, started <-chan struct{}, result <-chan error) {
	t.Helper()
	select {
	case <-started:
	case err := <-result:
		t.Fatalf("preparation returned before expected phase: %v", err)
	case <-time.After(5 * time.Second):
		t.Fatal("preparation never entered expected phase")
	}
}

type failedPreparationPlugin struct {
	plugin.ServicePlugin
	validationErr, initializationErr error
}

func (p failedPreparationPlugin) Validate(map[string]interface{}) error   { return p.validationErr }
func (p failedPreparationPlugin) Initialize(map[string]interface{}) error { return p.initializationErr }
func TestStartupPreservesPreparationErrorStage(t *testing.T) {
	sentinel := errors.New("backend failure")
	for _, stage := range []string{"validate", "initialize"} {
		t.Run(stage, func(t *testing.T) {
			p := failedPreparationPlugin{}
			if stage == "validate" {
				p.validationErr = sentinel
			} else {
				p.initializationErr = sentinel
			}
			err := prepareStartupPlugin(context.Background(), p, nil)
			if !errors.Is(err, sentinel) || !strings.Contains(err.Error(), stage+" plugin") {
				t.Fatalf("missing %s stage: %v", stage, err)
			}
		})
	}
}
