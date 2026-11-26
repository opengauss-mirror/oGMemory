package main

import (
	"context"
	"errors"
	"strings"
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
	result := make(chan error, 1)
	go func() { result <- prepareStartupPlugin(ctx, p, nil) }()
	<-p.started
	cancel()
	select {
	case err := <-result:
		if !errors.Is(err, context.Canceled) {
			t.Fatal(err)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("startup did not stop waiting after cancellation")
	}
	close(p.release)
	select {
	case <-p.stopped:
	case <-time.After(5 * time.Second):
		t.Fatal("late initialized plugin was not closed")
	}
}

type blockedValidationPlugin struct {
	plugin.ServicePlugin
	started, release, validated chan struct{}
	initialized                 atomic.Bool
}

func (p *blockedValidationPlugin) Validate(map[string]interface{}) error {
	close(p.started)
	<-p.release
	defer close(p.validated)
	return nil
}
func (p *blockedValidationPlugin) Initialize(map[string]interface{}) error {
	p.initialized.Store(true)
	return nil
}

func TestStartupDiscardsBlockedValidation(t *testing.T) {
	p := &blockedValidationPlugin{started: make(chan struct{}), release: make(chan struct{}), validated: make(chan struct{})}
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	result := make(chan error, 1)
	go func() { result <- prepareStartupPlugin(ctx, p, nil) }()
	<-p.started
	cancel()
	select {
	case err := <-result:
		if !errors.Is(err, context.Canceled) {
			t.Fatal(err)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("startup did not stop waiting for validation")
	}
	close(p.release)
	<-p.validated
	if p.initialized.Load() {
		t.Fatal("initialized after validation cancellation")
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
