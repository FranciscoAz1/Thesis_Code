package main

import (
	"context"
	"log"
	"os"
	"os/signal"
	"syscall"
)

func main() {
	// Load configuration
	config, err := LoadConfig()
	if err != nil {
		log.Fatalf("Failed to load configuration: %v", err)
	}

	// Initialize logger
	logger := NewLogger(config)
	logger.Info("Starting Weaviate MCP Server v0.1.0")
	logger.Info("Configuration: host=%s, scheme=%s, transport=%s, read-only=%v",
		config.WeaviateHost, config.WeaviateScheme, config.Transport, config.ReadOnly)

	// Create MCP server
	server, err := NewMCPServer(config, logger)
	if err != nil {
		logger.Error("Failed to create MCP server: %v", err)
		log.Fatalf("Failed to create MCP server: %v", err)
	}

	// Handle graceful shutdown
	ctx, cancel := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer cancel()

	// Start server based on transport
	switch config.Transport {
	case "stdio":
		logger.Info("Starting server with stdio transport")
		go func() {
			if err := server.ServeStdio(); err != nil {
				logger.Error("Server error: %v", err)
				cancel()
			}
		}()
	case "http":
		logger.Info("Starting server with HTTP transport on %s:%d", config.HTTPHost, config.HTTPPort)
		go func() {
			if err := server.ServeHTTP(config.HTTPHost, config.HTTPPort); err != nil {
				logger.Error("Server error: %v", err)
				cancel()
			}
		}()
	default:
		logger.Error("Unsupported transport: %s", config.Transport)
		log.Fatalf("Unsupported transport: %s", config.Transport)
	}

	// Wait for shutdown signal
	<-ctx.Done()
	logger.Info("Shutting down server...")
}
