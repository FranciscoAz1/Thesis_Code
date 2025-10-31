package main

import (
	"flag"
	"fmt"
	"os"
	"strconv"
	"strings"
)

// Config holds all configuration for the MCP server
type Config struct {
	// Weaviate connection
	WeaviateHost   string
	WeaviateScheme string

	// Server configuration
	Transport string // "stdio" or "http"
	HTTPPort  int
	HTTPHost  string

	// Logging
	LogLevel  string // "debug", "info", "warn", "error"
	LogOutput string // "stderr", "file", or "both"

	// Security
	ReadOnly      bool
	DisabledTools []string

	// Other
	DefaultCollection string
}

// LoadConfig loads configuration from environment variables and command-line flags
func LoadConfig() (*Config, error) {
	config := &Config{
		// Defaults
		WeaviateHost:      getEnvOrDefault("WEAVIATE_HOST", "host.docker.internal:8080"),
		WeaviateScheme:    getEnvOrDefault("WEAVIATE_SCHEME", "http"),
		Transport:         getEnvOrDefault("MCP_TRANSPORT", "stdio"),
		HTTPPort:          3000,
		HTTPHost:          "127.0.0.1",
		LogLevel:          getEnvOrDefault("MCP_LOG_LEVEL", "info"),
		LogOutput:         getEnvOrDefault("MCP_LOG_OUTPUT", "stderr"),
		ReadOnly:          getEnvBool("MCP_READ_ONLY"),
		DefaultCollection: getEnvOrDefault("MCP_DEFAULT_COLLECTION", "Dataset"),
	}

	// Parse HTTP port
	if portStr := os.Getenv("MCP_HTTP_PORT"); portStr != "" {
		if port, err := strconv.Atoi(portStr); err == nil {
			config.HTTPPort = port
		}
	}

	// Parse disabled tools
	if disabled := os.Getenv("MCP_DISABLED_TOOLS"); disabled != "" {
		config.DisabledTools = strings.Split(disabled, ",")
		for i, tool := range config.DisabledTools {
			config.DisabledTools[i] = strings.TrimSpace(tool)
		}
	}

	// Command-line flags (override environment variables)
	flag.StringVar(&config.WeaviateHost, "weaviate-host", config.WeaviateHost, "Weaviate host")
	flag.StringVar(&config.WeaviateScheme, "weaviate-scheme", config.WeaviateScheme, "Weaviate scheme (http/https)")
	flag.StringVar(&config.Transport, "transport", config.Transport, "Transport protocol (stdio/http)")
	flag.IntVar(&config.HTTPPort, "http-port", config.HTTPPort, "HTTP port when using http transport")
	flag.StringVar(&config.HTTPHost, "http-host", config.HTTPHost, "HTTP host when using http transport")
	flag.StringVar(&config.LogLevel, "log-level", config.LogLevel, "Log level (debug/info/warn/error)")
	flag.StringVar(&config.LogOutput, "log-output", config.LogOutput, "Log output (stderr/file/both)")
	flag.BoolVar(&config.ReadOnly, "read-only", config.ReadOnly, "Enable read-only mode")
	flag.StringVar(&config.DefaultCollection, "default-collection", config.DefaultCollection, "Default collection name")

	flag.Parse()

	// Validate configuration
	if err := config.Validate(); err != nil {
		return nil, fmt.Errorf("invalid configuration: %w", err)
	}

	return config, nil
}

// Validate checks if the configuration is valid
func (c *Config) Validate() error {
	if c.Transport != "stdio" && c.Transport != "http" {
		return fmt.Errorf("invalid transport: %s, must be 'stdio' or 'http'", c.Transport)
	}

	validLogLevels := map[string]bool{"debug": true, "info": true, "warn": true, "error": true}
	if !validLogLevels[c.LogLevel] {
		return fmt.Errorf("invalid log level: %s", c.LogLevel)
	}

	validLogOutputs := map[string]bool{"stderr": true, "file": true, "both": true}
	if !validLogOutputs[c.LogOutput] {
		return fmt.Errorf("invalid log output: %s", c.LogOutput)
	}

	return nil
}

// IsToolDisabled checks if a tool is disabled
func (c *Config) IsToolDisabled(toolName string) bool {
	for _, disabled := range c.DisabledTools {
		if disabled == toolName {
			return true
		}
	}
	return false
}

// Helper functions
func getEnvOrDefault(key, defaultValue string) string {
	if value := os.Getenv(key); value != "" {
		return value
	}
	return defaultValue
}

func getEnvBool(key string) bool {
	value := os.Getenv(key)
	return value == "true" || value == "1" || value == "yes"
}
