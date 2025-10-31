ifeq ($(OS),Windows_NT)
    EXT = .exe
else
    EXT =
endif

.PHONY: build run-client clean test help

build:
	go build -o client/mcp-server$(EXT) .

run-client: build
	cd client && go run client.go

run-server: build
	client/mcp-server$(EXT)

run-server-debug: build
	client/mcp-server$(EXT) --log-level debug --log-output both

run-server-http-debug: build
	client/mcp-server$(EXT) --log-level debug --log-output both --transport http

test: build
	cd client && go run client.go

clean:
	rm -f mcp-server$(EXT) client/mcp-server$(EXT)
	rm -rf logs/

help:
	@echo "Available targets:"
	@echo "  build          - Build the MCP server"
	@echo "  run-client     - Build and run the test client"
	@echo "  run-server     - Build and run the MCP server"
	@echo "  run-server-debug - Build and run server with debug logging"
	@echo "  test           - Build and run tests"
	@echo "  clean          - Clean build artifacts"
	@echo "  help           - Show this help"
