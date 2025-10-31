.PHONY: build dev start clean test lint format install help

# Install dependencies
install:
	npm install

# Build TypeScript to JavaScript
build: clean
	npm run build

# Development mode with auto-restart
dev:
	npm run dev

# Production start
start: build
	npm run start

# Development with debugging
dev-debug:
	npm run dev -- --log-level debug --log-output both

# HTTP transport development
dev-http:
	npm run dev -- --transport http --log-level debug --log-output both

# Clean build artifacts
clean:
	npm run clean

# Run linter
lint:
	npm run lint

# Format code
format:
	npm run format

# Run tests (placeholder)
test:
	node test/test-server.js

# Docker build
docker-build:
	docker build -t mcp-server-weaviate-ts .

# Docker run with stdio transport
docker-run:
	docker run --rm -it --name mcp-server-weaviate-ts mcp-server-weaviate-ts

# Docker run with HTTP transport
docker-run-http:
	docker run --rm -it -p 3000:3000 --name mcp-server-weaviate-ts mcp-server-weaviate-ts node dist/main.js --transport http

# Clean Docker artifacts
docker-clean:
	docker rmi mcp-server-weaviate-ts 2>/dev/null || true

# Clean logs
clean-logs:
	rm -rf logs/

help:
	@echo "Available targets:"
	@echo "  install        - Install Node.js dependencies"
	@echo "  build          - Build TypeScript to JavaScript"
	@echo "  dev            - Run in development mode with auto-restart"
	@echo "  start          - Run the built application"
	@echo "  dev-debug      - Run in development mode with debug logging"
	@echo "  dev-http       - Run in development mode with HTTP transport"
	@echo "  clean          - Clean build artifacts"
	@echo "  lint           - Run ESLint"
	@echo "  format         - Format code with Prettier"
	@echo "  test           - Run tests (placeholder)"
	@echo "  docker-build   - Build Docker image"
	@echo "  docker-run     - Run Docker container with stdio transport"
	@echo "  docker-run-http - Run Docker container with HTTP transport"
	@echo "  docker-clean   - Remove Docker image"
	@echo "  clean-logs     - Remove log files"
	@echo "  help           - Show this help"