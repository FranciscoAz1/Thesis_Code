# Weaviate MCP Server

[![Go Version](https://img.shields.io/badge/Go-1.23.1+-blue.svg)](https://golang.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A Model Context Protocol (MCP) server for interacting with Weaviate vector databases. Enables seamless integration between MCP-compatible clients (like Claude Desktop, VS Code with Copilot Chat) and Weaviate for semantic search and data retrieval.

## 🚀 Features
- **🔍 Hybrid Search**: Query Weaviate using natural language with hybrid search capabilities
- **⚙️ Configurable**: Flexible configuration via environment variables and command-line options
- **📝 Logging**: Structured logging with multiple output options and debug modes
- **🔒 Security**: Read-only mode and selective tool disabling for secure deployments
- **🌍 Cross-platform**: Works on Windows, macOS, and Linux
- **🔧 Easy Integration**: Simple setup with VS Code, Claude Desktop, and other MCP clients
- **📊 Resource Discovery**: Automatic schema detection and property listing

## 📋 Prerequisites
- Go 1.23.1 or later
- Weaviate instance (local or cloud)
- MCP-compatible client (VS Code with Copilot Chat, Claude Desktop, etc.)

## 🛠️ Quick Start

### Option 1: VS Code Integration (Recommended)

1. **Clone and build**:
```bash
git clone https://github.com/FranciscoAz1/mcp-server-weaviate.git
cd mcp-server-weaviate
make build
```
1. **Start Weaviate locally**:
```bash
docker-compose up -d
```

2. **Configure VS Code**:
Create `.vscode/mcp.json`:
```json
{
	"servers": {
		"weaviate-server": {
			"type": "stdio",
			"command": "client/mcp-server.exe",
			"args": ["--log-level=debug"]
		}
	},
	"inputs": []
}
```

3. **Start using**:
Open VS Code and use the `weaviate-query` tool in Copilot Chat!

### Option 2: Claude Desktop Integration

Add to your Claude Desktop MCP configuration:
```json
{
  "mcpServers": {
    "weaviate": {
      "command": "client/mcp-server.exe",
      "args": ["--log-level=info"]
    }
  }
}
```

## 📦 Installation

### From Source

```bash
git clone https://github.com/FranciscoAz1/mcp-server-weaviate.git
cd mcp-server-weaviate
go build -o mcp-server .
```

### Using Make (Recommended)
```bash
make build    # Build the server
make test     # Run tests (if available)
make clean    # Clean build artifacts
```

### Docker
```bash
docker build -t weaviate-mcp-server .
```

### Pre-built Binaries
Check the [Releases](https://github.com/FranciscoAz1/mcp-server-weaviate/releases) page for pre-built binaries.

## ⚙️ Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `WEAVIATE_HOST` | `host.docker.internal:8080` | Weaviate server host |
| `WEAVIATE_SCHEME` | `http` | Weaviate connection scheme |
| `MCP_TRANSPORT` | `stdio` | Transport protocol (`stdio` or `http`) |
| `MCP_HTTP_PORT` | `3000` | HTTP port when using HTTP transport |
| `MCP_HTTP_HOST` | `127.0.0.1` | HTTP host when using HTTP transport |
| `MCP_LOG_LEVEL` | `info` | Log level (`debug`, `info`, `warn`, `error`) |
| `MCP_LOG_OUTPUT` | `stderr` | Log output (`stderr`, `file`, `both`) |
| `MCP_READ_ONLY` | `false` | Enable read-only mode |
| `MCP_DISABLED_TOOLS` | (none) | Comma-separated list of disabled tools |
| `MCP_DEFAULT_COLLECTION` | `DefaultCollection` | Default collection name |

### Command-Line Flags

```bash
./mcp-server --help
```

Available flags:
- `--weaviate-host`: Weaviate host
- `--weaviate-scheme`: Weaviate scheme
- `--transport`: Transport protocol
- `--http-port`: HTTP port
- `--http-host`: HTTP host
- `--log-level`: Log level
- `--log-output`: Log output
- `--read-only`: Enable read-only mode
- `--default-collection`: Default collection name

## 🚀 Setup

### Local Development

1. **Start Weaviate**:
```bash
docker-compose up -d
```

2. **Build the server**:
```bash
make build
```

3. **Run the test client**:
```bash
make run-client
```

### Docker Setup

1. **Start Weaviate stack**:
```bash
docker-compose up -d
```

2. **Run MCP server in Docker**:
```bash
docker run --rm -i \
  -e WEAVIATE_HOST=host.docker.internal:8080 \
  -e WEAVIATE_SCHEME=http \
  --network host \
  weaviate-mcp-server
```
> **Note**: This is not yet tested for docker mcp toolkit

## 🔧 Tools

### weaviate-query

Query objects from a Weaviate collection using hybrid search.

**Parameters:**
- `query` (string, required): Natural language query
- `collection` (string, required): Target collection name  
- `targetProperties` (array of strings, required): Properties to return
- `limit` (number, optional): Maximum results to return (default: 3)

**Example:**
```json
{
  "query": "What country is Valencia in?",
  "collection": "WorldMap",
  "targetProperties": ["continent", "country", "city"],
  "limit": 5
}
```

**Response:**
```json
{
  "data": {
    "Get": {
      "WorldMap": [
        {
          "continent": "Europe",
          "country": "Spain", 
          "city": "Valencia"
        }
      ]
    }
  }
}
```

**Note**: `targetProperties` must contain at least one property name. Empty arrays will result in an error.

### weaviate-insert-one

Insert an object into a Weaviate collection.

**Parameters:**
- `collection` (string, optional): Target collection name
- `properties` (object, required): Object properties to insert

**Example:**
```json
{
  "collection": "WorldMap",
  "properties": {
    "continent": "Europe",
    "country": "Spain",
    "city": "Valencia"
  }
}
```

## 📋 Resources

### Schema Discovery

The server automatically discovers available collections and their properties:

- **Resource URI**: `weaviate://schema/{collection}`
- **Description**: Lists all available properties for a collection
- **Usage**: Helps you understand what `targetProperties` are available for queries

**Example**: Access `weaviate://schema/Dataset` to see all properties in the Dataset collection.

## 📝 Prompts

The server includes example prompts for testing the tools. See [`prompts.md`](prompts.md) for ready-to-use prompt templates that demonstrate how to use the insert and query tools.

The prompts include:
- **Dataset-specific prompts** for LiHua-World data (dinner events, travel stories, personal events, relationships)
- **General testing prompts** for basic tool functionality
- **Custom search prompts** for flexible querying

These prompts can be copied into MCP clients that support custom prompts (like Claude Desktop) to provide guided interactions with the Weaviate database.

## 🧪 Testing

### MCP Inspector (Recommended)

Use the MCP Inspector for debugging and testing:

```bash
npm install -g @modelcontextprotocol/inspector
mcp-inspector --command "./mcp-server"
```

This provides a web interface to interact with your MCP server.

### Test Client

The included test client demonstrates basic usage:

```bash
make run-client
```

### Manual Testing with curl (HTTP Transport)

If using HTTP transport:

```bash
# Initialize
curl -X POST http://localhost:3000/jsonrpc \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}}'

# List tools
curl -X POST http://localhost:3000/jsonrpc \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}}'

# Call tool
curl -X POST http://localhost:3000/jsonrpc \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc": "2.0", "id": 3, "method": "tools/call", "params": {"name": "weaviate-query", "arguments": {"query": "test", "targetProperties": ["field"]}}}'
```

## 🐛 Troubleshooting

### Common Issues

1. **"tool parameters array type must have items" Error**: 
   - **Solution**: This was fixed in recent versions. Ensure you're using the latest build.
   - **Cause**: Missing `items` specification in array schema.

2. **Connection Errors**: 
   - Check Weaviate is running and accessible at the configured host/port
   - Verify firewall settings allow connections
   - Test connection manually: `curl http://your-weaviate-host:8080/v1/.well-known/ready`

3. **Tool Not Found**: 
   - Verify tool isn't disabled in configuration
   - Check read-only mode isn't preventing tool usage
   - Restart your MCP client after configuration changes

4. **VS Code Integration Issues**:
   - Ensure `.vscode/mcp.json` is in the correct location
   - Restart VS Code after creating/modifying the configuration
   - Check VS Code Developer Console for MCP-related errors

5. **Logging Issues**: 
   - Check log output configuration matches your environment
   - Verify write permissions for log files when using file output

### Debug Mode

Enable debug logging for detailed troubleshooting:

```bash
# Environment variable
export MCP_LOG_LEVEL=debug
./mcp-server

# Command line flag  
./mcp-server --log-level=debug

# VS Code configuration
{
  "servers": {
    "weaviate-server": {
      "command": "client/mcp-server.exe",
      "args": ["--log-level=debug", "--log-output=both"]
    }
  }
}
```

### Log Files

When using file logging, logs are written to `logs/mcp-server.log` by default.

## 📚 Examples

### Basic Query in VS Code Copilot Chat

After setting up the MCP server, you can use it directly in VS Code:

```
Query the LiHua dataset for dinner conversations using weaviate-query with collection "Dataset", query "LiHua dinner restaurant", and targetProperties ["text", "file_path"]
```

### Programmatic Usage

```bash
# List available tools
echo '{"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}}' | ./mcp-server

# Execute a query
echo '{"jsonrpc": "2.0", "id": 2, "method": "tools/call", "params": {"name": "weaviate-query", "arguments": {"query": "travel stories", "collection": "Dataset", "targetProperties": ["text", "file_path"]}}}' | ./mcp-server
```

## 🚀 Registry Submission

To submit to the MCP registry, use the provided `server.yaml` file.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🔗 Links

- [Model Context Protocol Documentation](https://modelcontextprotocol.io/)
- [Weaviate Documentation](https://weaviate.io/developers/weaviate)
- [VS Code MCP Integration Guide](https://code.visualstudio.com/docs/copilot/copilot-chat)

## ⭐ Star History

If you find this project useful, please consider giving it a star on GitHub!
