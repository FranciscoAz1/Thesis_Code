package main

import (
	"context"
	"fmt"
	"log"
	"runtime"

	"github.com/mark3labs/mcp-go/client"
	"github.com/mark3labs/mcp-go/mcp"
)

func main() {
	ctx := context.Background()
	cmd := "./mcp-server"
	if runtime.GOOS == "windows" {
		cmd = "./mcp-server.exe"
	}

	log.Println("Starting MCP test client...")
	c, err := newMCPClient(ctx, cmd)
	if err != nil {
		log.Fatal(err)
	}
	defer c.Close()

	log.Println("Testing tool listing...")
	if err := testListTools(ctx, c); err != nil {
		log.Printf("Tool listing test failed: %v", err)
	} else {
		log.Println("✓ Tool listing test passed")
	}

	log.Println("Testing generate text operation...")
	if err := testGenerateText(ctx, c); err != nil {
		log.Printf("Generate text test failed: %v", err)
	} else {
		log.Println("✓ Generate text test passed")
	}

	// log.Println("Testing prompt listing...")
	// if err := testListPrompts(ctx, c); err != nil {
	// 	log.Printf("Prompt listing test failed: %v", err)
	// } else {
	// 	log.Println("✓ Prompt listing test passed")
	// }

	// log.Println("Testing insert operation...")
	// if err := testInsert(ctx, c); err != nil {
	// 	log.Printf("Insert test failed: %v", err)
	// } else {
	// 	log.Println("✓ Insert test passed")
	// }

	// log.Println("Testing query operation...")
	// if err := testQuery(ctx, c); err != nil {
	// 	log.Printf("Query test failed: %v", err)
	// } else {
	// 	log.Println("✓ Query test passed")
	// }

	log.Println("All tests completed!")
}

func newMCPClient(ctx context.Context, cmd string) (*client.Client, error) {
	c, err := client.NewStdioMCPClient(cmd, nil)
	if err != nil {
		return nil, fmt.Errorf("failed to create client: %w", err)
	}
	if err := c.Start(ctx); err != nil {
		return nil, fmt.Errorf("failed to start client: %w", err)
	}
	initRes, err := c.Initialize(ctx, mcp.InitializeRequest{})
	if err != nil {
		return nil, fmt.Errorf("failed to init client: %w", err)
	}
	log.Printf("init result: %+v", initRes)
	if err := c.Ping(ctx); err != nil {
		return nil, fmt.Errorf("failed to ping server: %w", err)
	}
	return c, nil
}

func testListTools(ctx context.Context, c *client.Client) error {
	toolsRes, err := c.ListTools(ctx, mcp.ListToolsRequest{})
	if err != nil {
		return fmt.Errorf("failed to list tools: %w", err)
	}

	log.Printf("Available tools: %d", len(toolsRes.Tools))
	for _, tool := range toolsRes.Tools {
		log.Printf("  - %s: %s", tool.Name, tool.Description)
	}

	return nil
}

func testListPrompts(ctx context.Context, c *client.Client) error {
	promptsRes, err := c.ListPrompts(ctx, mcp.ListPromptsRequest{})
	if err != nil {
		return fmt.Errorf("failed to list prompts: %w", err)
	}

	log.Printf("Available prompts: %d", len(promptsRes.Prompts))
	for _, prompt := range promptsRes.Prompts {
		log.Printf("  - %s: %s", prompt.Name, prompt.Description)
		if len(prompt.Arguments) > 0 {
			log.Printf("    Arguments:")
			for _, arg := range prompt.Arguments {
				log.Printf("      - %s (%s): %s", arg.Name, arg.Name, arg.Description)
			}
		}
	}

	return nil
}

func testGenerateText(ctx context.Context, c *client.Client) error {
	request := mcp.CallToolRequest{}
	request.Params.Name = "weaviate-generate-text"
	request.Params.Arguments = map[string]interface{}{
		"prompt":     "What are the main topics in these government documents?",
		"collection": "Dataset",
		"maxTokens":  150,
	}
	log.Printf("generate text request: %+v", request)
	res, err := c.CallTool(ctx, request)
	if err != nil {
		return fmt.Errorf("failed to call generate text tool: %v", err)
	}
	log.Printf("generate text result: %+v", res)
	return nil
}

func testInsert(ctx context.Context, c *client.Client) error {
	request := mcp.CallToolRequest{}
	request.Params.Name = "weaviate-insert-one"
	request.Params.Arguments = map[string]interface{}{
		"collection": "TestCollection",
		"properties": map[string]interface{}{
			"name":     "Test Item",
			"category": "test",
		},
	}
	log.Printf("insert request: %+v", request)
	_, err := c.CallTool(ctx, request)
	if err != nil {
		return fmt.Errorf("failed to call insert-one tool: %v", err)
	}
	return nil
}

func testQuery(ctx context.Context, c *client.Client) error {
	request := mcp.CallToolRequest{}
	request.Params.Name = "weaviate-query"
	request.Params.Arguments = map[string]interface{}{
		"query":            "lihua",
		"targetProperties": []string{"text", "file_path"},
		"collection":       "Dataset",
	}
	log.Printf("query request: %+v", request)
	res, err := c.CallTool(ctx, request)
	if err != nil {
		return fmt.Errorf("failed to call query tool: %v", err)
	}
	log.Printf("query result: %+v", res)
	return nil
}

func insertRequest(ctx context.Context, c *client.Client) (*mcp.CallToolResult, error) {
	request := mcp.CallToolRequest{}
	request.Params.Name = "weaviate-insert-one"
	request.Params.Arguments = map[string]interface{}{
		"collection": "WorldMap",
		"properties": map[string]interface{}{
			"continent": "Europe",
			"country":   "Spain",
			"city":      "Valencia",
		},
	}
	log.Printf("insert request: %+v", request)
	res, err := c.CallTool(ctx, request)
	if err != nil {
		return nil, fmt.Errorf("failed to call insert-one tool: %v", err)
	}
	return res, nil
}

func queryRequest(ctx context.Context, c *client.Client) (*mcp.CallToolResult, error) {
	request := mcp.CallToolRequest{}
	request.Params.Name = "weaviate-query"
	request.Params.Arguments = map[string]interface{}{
		"collection":       "WorldMap",
		"query":            "What country is Valencia in?",
		"targetProperties": []string{"continent", "country", "city"},
	}
	log.Printf("query request: %+v", request)
	res, err := c.CallTool(ctx, request)
	if err != nil {
		return nil, fmt.Errorf("failed to call query tool: %v", err)
	}
	return res, nil
}
