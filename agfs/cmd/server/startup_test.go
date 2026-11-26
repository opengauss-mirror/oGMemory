package main

import (
	"context"
	"errors"
	"testing"
	"time"

	"github.com/c4pt0r/agfs/agfs-server/pkg/plugin"
)

type blockedStartupPlugin struct {
	plugin.ServicePlugin
	started, release, stopped chan struct{}
}

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
	go func() { result <- initializeStartupPlugin(ctx, p, nil) }()
	<-p.started
	cancel()
	select {
	case err := <-result:
		if !errors.Is(err, context.Canceled) {
			t.Fatal(err)
		}
	case <-time.After(time.Second):
		t.Fatal("startup did not stop waiting after cancellation")
	}
	close(p.release)
	select {
	case <-p.stopped:
	case <-time.After(time.Second):
		t.Fatal("late initialized plugin was not closed")
	}
}
