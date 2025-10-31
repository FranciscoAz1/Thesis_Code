# Weaviate MCP Server (TypeScript)

A Model Context Protocol (MCP) server for Weaviate vector database, implemented in TypeScript. This server provides MCP-compatible tools for querying and interacting with Weaviate collections.

## Features

- **Query Tool**: Query Weaviate collections using hybrid search
- **Generate Text Tool**: Generate text using Weaviate's generative search capabilities
- **Resource Access**: Access collection schemas and metadata
- **Multiple Transport Modes**: Support for stdio and HTTP transports
- **Configurable Logging**: Flexible logging with multiple output options
- **Docker Support**: Containerized deployment ready

## Installation

### Prerequisites

- Node.js 18.0.0 or higher
- npm or yarn
- Access to a Weaviate instance

### Install Dependencies

```bash
npm install
```

## Configuration

Configuration can be provided through environment variables or command-line arguments.

### Environment Variables

Create a `.env` file based on `.env.example`:

```bash
cp .env.example .env
```

Available environment variables:

- `WEAVIATE_HOST` - Weaviate host (default: `host.docker.internal:8080`)
- `WEAVIATE_SCHEME` - Connection scheme (default: `http`)
- `MCP_TRANSPORT` - Transport protocol: `stdio` or `http` (default: `stdio`)
- `MCP_HTTP_PORT` - HTTP port when using http transport (default: `3000`)
- `MCP_HTTP_HOST` - HTTP host when using http transport (default: `127.0.0.1`)
- `MCP_LOG_LEVEL` - Log level: `debug`, `info`, `warn`, `error` (default: `info`)
- `MCP_LOG_OUTPUT` - Log output: `stderr`, `file`, `both` (default: `stderr`)
- `MCP_READ_ONLY` - Enable read-only mode (default: `false`)
- `MCP_DISABLED_TOOLS` - Comma-separated list of disabled tools
- `MCP_DEFAULT_COLLECTION` - Default collection name (default: `DefaultCollection`)

### Command Line Arguments

```bash
npm run dev -- --weaviate-host localhost:8080 --transport stdio --log-level debug
```

Available arguments:
- `--weaviate-host` - Weaviate host
- `--weaviate-scheme` - Weaviate scheme (http/https)
- `--transport` - Transport protocol (stdio/http)
- `--http-port` - HTTP port
- `--http-host` - HTTP host
- `--log-level` - Log level
- `--log-output` - Log output
- `--read-only` - Enable read-only mode
- `--default-collection` - Default collection name

## Usage

### Development Mode

```bash
# Start in development mode with auto-restart
npm run dev

# Start with debug logging
npm run dev -- --log-level debug --log-output both

# Start with HTTP transport
npm run dev -- --transport http
```

### Production Mode

```bash
# Build the project
npm run build

# Start the built application
npm start
```

### Using Make

```bash
# Install dependencies
make install

# Development mode
make dev

# Production build and start
make build
make start

# Debug mode
make dev-debug

# HTTP transport mode
make dev-http
```

## Tools

### weaviate-query

Query objects from a Weaviate collection using hybrid search.

**Parameters:**
- `query` (string, required) - Search query
- `collection` (string, required) - Target collection name
- `targetProperties` (array, required) - Properties to return
- `limit` (number, optional) - Maximum results (default: 3)

**Example:**
```json
{
  "query": "artificial intelligence",
  "collection": "Articles",
  "targetProperties": ["title", "content", "author"],
  "limit": 5
}
```

### weaviate-generate-text

Generate text using Weaviate's generative search capabilities.

**Parameters:**
- `prompt` (string, required) - Text prompt for generation
- `collection` (string, required) - Target collection name
- `maxTokens` (number, optional) - Maximum tokens to generate (default: 100)

**Example:**
```json
{
  "prompt": "Summarize the main points about AI",
  "collection": "Articles",
  "maxTokens": 200
}
```

## Resources

### Schema Resources

Access collection schema information via URIs like:
- `weaviate://schema/{collection}` - Schema for a specific collection

## Docker

### Build Docker Image

```bash
make docker-build
```

### Run with Docker

```bash
# Run with stdio transport
make docker-run

# Run with HTTP transport
make docker-run-http
```

### Manual Docker Commands

```bash
# Build
docker build -t mcp-server-weaviate-ts .

# Run with stdio transport
docker run --rm -it mcp-server-weaviate-ts

# Run with HTTP transport
docker run --rm -it -p 3000:3000 mcp-server-weaviate-ts node dist/main.js --transport http
```

## Development

### Code Quality

```bash
# Lint code
npm run lint

# Format code
npm run format
```

### Project Structure

```
src/
├── main.ts           # Main entry point
├── config.ts         # Configuration management
├── logger.ts         # Logging utilities
├── weaviate.ts       # Weaviate client wrapper
└── mcp.ts           # MCP server implementation
```

## Transport Modes

### stdio (Recommended)

The stdio transport is the primary mode for MCP communication:

```bash
npm run dev -- --transport stdio
```

### HTTP (Experimental)

HTTP transport is available but may require additional setup:

```bash
npm run dev -- --transport http --http-port 3000
```

## Logging

Logs can be output to:
- `stderr` - Standard error stream
- `file` - Log file in `logs/mcp-server.log`
- `both` - Both stderr and file

Log levels: `debug`, `info`, `warn`, `error`

## Troubleshooting

### Common Issues

1. **Module not found errors**: Run `npm install` to install dependencies
2. **Weaviate connection errors**: Check `WEAVIATE_HOST` and `WEAVIATE_SCHEME` settings
3. **Permission errors**: Ensure the `logs/` directory is writable

### Debug Mode

Enable debug logging to see detailed information:

```bash
npm run dev -- --log-level debug --log-output both
```

## Migration from Go Version

This TypeScript version provides the same functionality as the original Go implementation:

- All tools and resources are preserved
- Configuration options remain the same
- Transport modes are compatible
- Docker deployment works similarly

## License

MIT License - see LICENSE file for details.