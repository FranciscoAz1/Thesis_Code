package main

import (
	"fmt"
	"io"
	"log"
	"os"
	"strings"
)

// Logger provides structured logging
type Logger struct {
	level  LogLevel
	output io.Writer
}

// LogLevel represents different logging levels
type LogLevel int

const (
	LogLevelDebug LogLevel = iota
	LogLevelInfo
	LogLevelWarn
	LogLevelError
)

// String returns the string representation of the log level
func (l LogLevel) String() string {
	switch l {
	case LogLevelDebug:
		return "DEBUG"
	case LogLevelInfo:
		return "INFO"
	case LogLevelWarn:
		return "WARN"
	case LogLevelError:
		return "ERROR"
	default:
		return "UNKNOWN"
	}
}

// ParseLogLevel parses a string into a LogLevel
func ParseLogLevel(s string) LogLevel {
	switch strings.ToLower(s) {
	case "debug":
		return LogLevelDebug
	case "info":
		return LogLevelInfo
	case "warn", "warning":
		return LogLevelWarn
	case "error":
		return LogLevelError
	default:
		return LogLevelInfo
	}
}

// NewLogger creates a new logger with the specified configuration
func NewLogger(config *Config) *Logger {
	var output io.Writer

	switch config.LogOutput {
	case "stderr":
		output = os.Stderr
	case "file":
		// Create logs directory if it doesn't exist
		logDir := "logs"
		if err := os.MkdirAll(logDir, 0755); err != nil {
			log.Printf("Failed to create log directory: %v", err)
			output = os.Stderr
		} else {
			logFile, err := os.OpenFile(fmt.Sprintf("%s/mcp-server.log", logDir), os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0666)
			if err != nil {
				log.Printf("Failed to open log file: %v", err)
				output = os.Stderr
			} else {
				output = logFile
			}
		}
	case "both":
		logDir := "logs"
		if err := os.MkdirAll(logDir, 0755); err != nil {
			log.Printf("Failed to create log directory: %v", err)
			output = os.Stderr
		} else {
			logFile, err := os.OpenFile(fmt.Sprintf("%s/mcp-server.log", logDir), os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0666)
			if err != nil {
				log.Printf("Failed to open log file: %v", err)
				output = os.Stderr
			} else {
				output = io.MultiWriter(os.Stderr, logFile)
			}
		}
	default:
		output = os.Stderr
	}

	return &Logger{
		level:  ParseLogLevel(config.LogLevel),
		output: output,
	}
}

// Debug logs a debug message
func (l *Logger) Debug(format string, args ...interface{}) {
	l.log(LogLevelDebug, format, args...)
}

// Info logs an info message
func (l *Logger) Info(format string, args ...interface{}) {
	l.log(LogLevelInfo, format, args...)
}

// Warn logs a warning message
func (l *Logger) Warn(format string, args ...interface{}) {
	l.log(LogLevelWarn, format, args...)
}

// Error logs an error message
func (l *Logger) Error(format string, args ...interface{}) {
	l.log(LogLevelError, format, args...)
}

// log writes a log message if the level is enabled
func (l *Logger) log(level LogLevel, format string, args ...interface{}) {
	if level < l.level {
		return
	}

	message := fmt.Sprintf(format, args...)
	timestamp := fmt.Sprintf("[%s] %s: %s\n", level.String(), "MCP-Server", message)

	if _, err := l.output.Write([]byte(timestamp)); err != nil {
		// Fallback to standard log if our logger fails
		log.Printf("Logger error: %v", err)
		log.Print(timestamp)
	}
}
