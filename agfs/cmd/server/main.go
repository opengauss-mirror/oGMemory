package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"net/http"
	"os"
	"os/signal"
	"path/filepath"
	"runtime"
	"sync"
	"syscall"
	"time"

	"github.com/c4pt0r/agfs/agfs-server/pkg/config"
	"github.com/c4pt0r/agfs/agfs-server/pkg/filesystem"
	"github.com/c4pt0r/agfs/agfs-server/pkg/handlers"
	"github.com/c4pt0r/agfs/agfs-server/pkg/mountablefs"
	"github.com/c4pt0r/agfs/agfs-server/pkg/plugin"
	"github.com/c4pt0r/agfs/agfs-server/pkg/plugin/api"
	"github.com/c4pt0r/agfs/agfs-server/pkg/plugins/gptfs"
	"github.com/c4pt0r/agfs/agfs-server/pkg/plugins/heartbeatfs"
	"github.com/c4pt0r/agfs/agfs-server/pkg/plugins/hellofs"
	"github.com/c4pt0r/agfs/agfs-server/pkg/plugins/httpfs"
	"github.com/c4pt0r/agfs/agfs-server/pkg/plugins/kvfs"
	"github.com/c4pt0r/agfs/agfs-server/pkg/plugins/localfs"
	"github.com/c4pt0r/agfs/agfs-server/pkg/plugins/memfs"
	"github.com/c4pt0r/agfs/agfs-server/pkg/plugins/proxyfs"
	"github.com/c4pt0r/agfs/agfs-server/pkg/plugins/queuefs"
	"github.com/c4pt0r/agfs/agfs-server/pkg/plugins/s3fs"
	"github.com/c4pt0r/agfs/agfs-server/pkg/plugins/serverinfofs"
	"github.com/c4pt0r/agfs/agfs-server/pkg/plugins/sqlfs"
	"github.com/c4pt0r/agfs/agfs-server/pkg/plugins/sqlfs2"
	"github.com/c4pt0r/agfs/agfs-server/pkg/plugins/streamfs"
	"github.com/c4pt0r/agfs/agfs-server/pkg/plugins/streamrotatefs"
	log "github.com/sirupsen/logrus"
)

var (
	// Version information, injected during build
	Version   = "1.4.0"
	BuildTime = "unknown"
	GitCommit = "unknown"
)

// PluginFactory is a function that creates a new plugin instance
type PluginFactory func() plugin.ServicePlugin

// availablePlugins maps plugin names to their factory functions
var availablePlugins = map[string]PluginFactory{
	"serverinfofs":   func() plugin.ServicePlugin { return serverinfofs.NewServerInfoFSPlugin() },
	"memfs":          func() plugin.ServicePlugin { return memfs.NewMemFSPlugin() },
	"queuefs":        func() plugin.ServicePlugin { return queuefs.NewQueueFSPlugin() },
	"kvfs":           func() plugin.ServicePlugin { return kvfs.NewKVFSPlugin() },
	"hellofs":        func() plugin.ServicePlugin { return hellofs.NewHelloFSPlugin() },
	"heartbeatfs":    func() plugin.ServicePlugin { return heartbeatfs.NewHeartbeatFSPlugin() },
	"httpfs":         func() plugin.ServicePlugin { return httpfs.NewHTTPFSPlugin() },
	"proxyfs":        func() plugin.ServicePlugin { return proxyfs.NewProxyFSPlugin("") },
	"s3fs":           func() plugin.ServicePlugin { return s3fs.NewS3FSPlugin() },
	"streamfs":       func() plugin.ServicePlugin { return streamfs.NewStreamFSPlugin() },
	"streamrotatefs": func() plugin.ServicePlugin { return streamrotatefs.NewStreamRotateFSPlugin() },
	"sqlfs":          func() plugin.ServicePlugin { return sqlfs.NewSQLFSPlugin() },
	"sqlfs2":         func() plugin.ServicePlugin { return sqlfs2.NewSQLFS2Plugin() },
	"localfs":        func() plugin.ServicePlugin { return localfs.NewLocalFSPlugin() },
	"gptfs":          func() plugin.ServicePlugin { return gptfs.NewGptfs() },
}

const sampleConfig = `# AGFS Server Configuration File
# This is a sample configuration showing all available options

server:
  address: ":8080"          # Server listen address
  log_level: "info"         # Log level: debug, info, warn, error

# Plugin configurations
plugins:
  # Server Info Plugin - provides server information and stats
  serverinfofs:
    enabled: true
    path: "/serverinfofs"

  # Memory File System - in-memory file storage
  memfs:
    enabled: true
    path: "/memfs"

  # Queue File System - message queue operations
  queuefs:
    enabled: true
    path: "/queuefs"

  # Key-Value File System - key-value store
  kvfs:
    enabled: true
    path: "/kvfs"

  # Hello File System - example plugin
  hellofs:
    enabled: true
    path: "/hellofs"

  # Stream File System - streaming file operations
  streamfs:
    enabled: true
    path: "/streamfs"

  # Local File System - mount local directories
  localfs:
    enabled: false
    path: "/localfs"
    config:
      root_path: "/path/to/local/directory"  # Local directory to mount

  # S3 File System - mount S3 buckets
  s3fs:
    enabled: false
    path: "/s3fs"
    config:
      bucket: "your-bucket-name"
      region: "us-west-2"
      access_key: "YOUR_ACCESS_KEY"
      secret_key: "YOUR_SECRET_KEY"
      endpoint: ""  # Optional: custom S3 endpoint

  # SQL File System - file system backed by SQL database
  sqlfs:
    enabled: false
    # Multi-instance example: mount multiple SQL databases
    instances:
      - name: "sqlfs-sqlite"
        enabled: true
        path: "/sqlfs/sqlite"
        config:
          backend: "sqlite"
          db_path: "/tmp/agfs-sqlite.db"

      - name: "sqlfs-postgres"
        enabled: false
        path: "/sqlfs/postgres"
        config:
          backend: "postgres"
          connection_string: "postgres://user:pass@localhost/dbname?sslmode=disable"

  # Proxy File System - proxy to another AGFS server
  proxyfs:
    enabled: false
    # Multi-instance example: proxy multiple remote servers
    instances:
      - name: "proxy-remote1"
        enabled: true
        path: "/proxy/remote1"
        config:
          base_url: "http://remote-server-1:8080/api/v1"
          remote_path: "/"

      - name: "proxy-remote2"
        enabled: false
        path: "/proxy/remote2"
        config:
          base_url: "http://remote-server-2:8080/api/v1"
          remote_path: "/memfs"
`

func main() {
	configFile := flag.String("c", "config.yaml", "Path to configuration file")
	addr := flag.String("addr", "", "Server listen address (will override addr in config file)")
	printSampleConfig := flag.Bool("print-sample-config", false, "Print a sample configuration file and exit")
	version := flag.Bool("version", false, "Print version information and exit")
	flag.Parse()

	// Handle --version
	if *version {
		fmt.Printf("agfs-server version: %s\n", Version)
		fmt.Printf("Git commit: %s\n", GitCommit)
		fmt.Printf("Build time: %s\n", BuildTime)
		return
	}

	// Handle --print-sample-config
	if *printSampleConfig {
		fmt.Print(sampleConfig)
		return
	}

	// Load configuration
	cfg, err := config.LoadConfig(*configFile)
	if err != nil {
		log.Fatalf("Failed to load config file: %v", err)
	}

	// Configure logrus
	logLevel := log.InfoLevel
	if cfg.Server.LogLevel != "" {
		if level, err := log.ParseLevel(cfg.Server.LogLevel); err == nil {
			logLevel = level
		}
	}
	log.SetFormatter(&log.TextFormatter{
		FullTimestamp: true,
		CallerPrettyfier: func(f *runtime.Frame) (string, string) {
			filename := filepath.Base(f.File)
			return "", fmt.Sprintf(" | %s:%d | ", filename, f.Line)
		},
	})
	log.SetReportCaller(true)
	log.SetLevel(logLevel)

	// Determine server address
	serverAddr := cfg.Server.Address
	if *addr != "" {
		serverAddr = *addr // Command line override
	}
	if serverAddr == "" {
		serverAddr = ":8080" // Default
	}

	// Create WASM instance pool configuration from config
	wasmConfig := cfg.GetWASMConfig()
	poolConfig := api.PoolConfig{
		MaxInstances:        wasmConfig.InstancePoolSize,
		InstanceMaxLifetime: time.Duration(wasmConfig.InstanceMaxLifetime) * time.Second,
		InstanceMaxRequests: int64(wasmConfig.InstanceMaxRequests),
		HealthCheckInterval: time.Duration(wasmConfig.HealthCheckInterval) * time.Second,
		EnableStatistics:    wasmConfig.EnablePoolStatistics,
	}

	// Create mountable file system
	mfs := mountablefs.NewMountableFS(poolConfig)
	if err := mfs.ConfigureCache(cfg.CacheFS, cfg.Cache); err != nil {
		log.Fatalf("Cache configuration failed: %v", err)
	}

	// Create traffic monitor early so it can be injected into plugins during mounting
	trafficMonitor := handlers.NewTrafficMonitor()

	// Register plugin factories for dynamic mounting
	for pluginName, factory := range availablePlugins {
		// Capture factory in local variable to avoid closure issues
		f := factory
		mfs.RegisterPluginFactory(pluginName, func() plugin.ServicePlugin {
			return f()
		})
	}

	// With CacheFS enabled, every configured mount must finish before serving:
	// a non-cached mount can conflict with an already mounted cached backend.
	var startupMounts sync.WaitGroup
	var mountErrorsMu sync.Mutex
	var mountErrors []error
	startupCtx := context.Background()
	mountPlugin := func(pluginName, instanceName, mountPath string, pluginConfig map[string]interface{}) {
		selected := cfg.CacheFS.Selected(mountPath)
		reportMountError := func(err error) {
			if selected || errors.Is(err, mountablefs.ErrCacheStorageAlias) {
				mountErrorsMu.Lock()
				mountErrors = append(mountErrors, err)
				mountErrorsMu.Unlock()
			}
			log.Error(err)
		}
		// Get plugin factory (try built-in first, then external)
		factory, ok := availablePlugins[pluginName]
		var p plugin.ServicePlugin

		if !ok {
			// Try to get external plugin from mfs
			p = mfs.CreatePlugin(pluginName)
			if p == nil {
				if selected {
					reportMountError(fmt.Errorf("CacheFS mount %s uses unknown plugin %s", mountPath, pluginName))
				}
				log.Warnf("Unknown plugin: %s, skipping instance '%s'", pluginName, instanceName)
				return
			}
		} else {
			// Create plugin instance from built-in factory
			p = factory()
		}

		// Special handling for httpfs: inject rootFS reference
		if pluginName == "httpfs" {
			if httpfsPlugin, ok := p.(*httpfs.HTTPFSPlugin); ok {
				httpfsPlugin.SetRootFS(mfs)
			}
		}

		// Special handling for serverinfofs: inject traffic monitor
		if pluginName == "serverinfofs" {
			if serverInfoPlugin, ok := p.(*serverinfofs.ServerInfoFSPlugin); ok {
				serverInfoPlugin.SetTrafficMonitor(trafficMonitor)
			}
		}

		startupMounts.Add(1)
		mountInstance := func() {
			defer startupMounts.Done()
			// Inject mount_path into config
			configWithPath, err := mfs.PrepareMountConfig(pluginName, mountPath, pluginConfig)
			if err != nil {
				reportMountError(fmt.Errorf("Mount cache configuration failed: %w", err))
				return
			}

			var prepareErr error
			if cfg.CacheFS.Enabled {
				prepareErr = prepareStartupPlugin(startupCtx, p, configWithPath)
			} else {
				prepareErr = preparePlugin(context.Background(), p, configWithPath)
			}
			if prepareErr != nil {
				reportMountError(fmt.Errorf("Failed to prepare %s instance '%s' at %s: %w", pluginName, instanceName, mountPath, prepareErr))
				return
			}

			// Mount plugin
			if err := mfs.Mount(mountPath, p); err != nil {
				_ = p.Shutdown()
				reportMountError(fmt.Errorf("Failed to mount %s instance '%s' at %s: %w", pluginName, instanceName, mountPath, err))
				return
			}

			// Log success
			log.Infof("%s instance '%s' mounted at %s", pluginName, instanceName, mountPath)
		}
		if selected {
			mountInstance()
		} else {
			go mountInstance()
		}
	}

	// Load external plugins if enabled
	if cfg.ExternalPlugins.Enabled {
		log.Info("Loading external plugins...")

		// Auto-load from plugin directory
		if cfg.ExternalPlugins.AutoLoad && cfg.ExternalPlugins.PluginDir != "" {
			log.Infof("Auto-loading plugins from: %s", cfg.ExternalPlugins.PluginDir)
			loaded, errors := mfs.LoadExternalPluginsFromDirectory(cfg.ExternalPlugins.PluginDir)
			if len(errors) > 0 {
				log.Warnf("Encountered %d error(s) while loading plugins:", len(errors))
				for _, err := range errors {
					log.Warnf("- %v", err)
				}
			}
			if len(loaded) > 0 {
				log.Infof("Auto-loaded %d plugin(s)", len(loaded))
			}
		}

		// Load specific plugin paths
		for _, pluginPath := range cfg.ExternalPlugins.PluginPaths {
			log.Infof("Loading plugin: %s", pluginPath)
			p, err := mfs.LoadExternalPlugin(pluginPath)
			if err != nil {
				log.Errorf("Failed to load plugin %s: %v", pluginPath, err)
			} else {
				log.Infof("Loaded plugin: %s", p.Name())
			}
		}
	}

	// Mount all enabled plugins
	if cfg.CacheFS.Enabled {
		var cancelStartup context.CancelFunc
		startupCtx, cancelStartup = newStartupContext()
		defer cancelStartup()
	}
	log.Info("Mounting plugin filesytems...")
	for pluginName, pluginCfg := range cfg.Plugins {
		// Normalize to instance array (convert single instance to array of one)
		instances := pluginCfg.Instances
		if len(instances) == 0 {
			// Single instance mode: treat as array with one instance
			instances = []config.PluginInstance{
				{
					Name:    pluginName, // Use plugin name as instance name
					Enabled: pluginCfg.Enabled,
					Path:    pluginCfg.Path,
					Config:  pluginCfg.Config,
				},
			}
		}

		// Mount all instances
		for _, instance := range instances {
			if !instance.Enabled {
				log.Infof("%s instance '%s' is disabled, skipping", pluginName, instance.Name)
				continue
			}

			mountPlugin(pluginName, instance.Name, instance.Path, instance.Config)
		}
	}

	if cfg.CacheFS.Enabled {
		startupMounts.Wait()
		if err := errors.Join(mountErrors...); err != nil {
			log.Fatalf("CacheFS mount setup failed: %v", err)
		}
		mounted := mfs.CacheStats()
		for _, mount := range cfg.CacheFS.Mounts {
			if _, ok := mounted[filesystem.NormalizePath(mount)]; !ok {
				log.Fatalf("CacheFS mount %s is missing or disabled", mount)
			}
		}
	}

	// Create handlers
	handler := handlers.NewHandler(mfs, trafficMonitor)
	handler.SetVersionInfo(Version, GitCommit, BuildTime)
	pluginHandler := handlers.NewPluginHandler(mfs)

	// Setup routes
	mux := http.NewServeMux()
	handler.SetupRoutes(mux)
	pluginHandler.SetupRoutes(mux)

	// Wrap with logging middleware
	loggedMux := handlers.LoggingMiddleware(mux)
	// Start server
	log.Infof("Starting AGFS server on %s", serverAddr)

	server := &http.Server{Addr: serverAddr, Handler: loggedMux}
	stop, cancel := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer cancel()
	stopped := make(chan struct{})
	go func() {
		defer close(stopped)
		<-stop.Done()
		ctx, done := context.WithTimeout(context.Background(), 10*time.Second)
		defer done()
		if err := server.Shutdown(ctx); err != nil {
			log.Errorf("HTTP shutdown: %v", err)
		}
		if err := mfs.Close(ctx); err != nil {
			log.Errorf("Filesystem shutdown: %v", err)
		}
	}()
	serveErr := serveHTTP(server)
	cancel()
	<-stopped
	if serveErr != nil && !errors.Is(serveErr, http.ErrServerClosed) {
		log.Error(serveErr)
		os.Exit(1)
	}
}
