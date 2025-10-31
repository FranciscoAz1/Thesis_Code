package main

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"

	"github.com/mark3labs/mcp-go/mcp"
	"github.com/mark3labs/mcp-go/server"
)

type MCPServer struct {
	server            *server.MCPServer
	weaviateConn      *WeaviateConnection
	defaultCollection string
	config            *Config
	logger            *Logger
}

func NewMCPServer(config *Config, logger *Logger) (*MCPServer, error) {
	logger.Info("Initializing Weaviate connection...")
	conn, err := NewWeaviateConnection(config, logger)
	if err != nil {
		return nil, fmt.Errorf("failed to create Weaviate connection: %w", err)
	}

	s := &MCPServer{
		server: server.NewMCPServer(
			"Weaviate MCP Server",
			"0.1.0",
			server.WithToolCapabilities(true),
			server.WithPromptCapabilities(false),
			server.WithResourceCapabilities(true, true),
			server.WithRecovery(),
		),
		weaviateConn:      conn,
		defaultCollection: config.DefaultCollection,
		config:            config,
		logger:            logger,
	}

	logger.Info("Registering tools...")
	s.registerTools()

	logger.Info("Registering resources...")
	s.registerResources()

	// logger.Info("Registering prompts...")
	// s.registerPrompts()

	logger.Info("MCP Server initialized successfully")
	return s, nil
}

func (s *MCPServer) Serve() {
	s.logger.Info("MCPServer starting: waiting for requests...")
	server.ServeStdio(s.server)
}

func (s *MCPServer) ServeStdio() error {
	s.logger.Info("Starting stdio server...")
	return server.ServeStdio(s.server)
}

func (s *MCPServer) ServeHTTP(host string, port int) error {
	s.logger.Info("Starting HTTP server on %s:%d", host, port)

	// Log a warning about HTTP support
	s.logger.Info("Note: The mcp-go StreamableHTTP server appears to have compatibility issues")
	s.logger.Info("The HTTP transport may not be fully functional in this version")
	s.logger.Info("For reliable operation, use stdio transport with: --transport stdio")

	// Try to create StreamableHTTP server for MCP
	httpServer := server.NewStreamableHTTPServer(s.server)

	// Start the server
	addr := fmt.Sprintf("%s:%d", host, port)
	s.logger.Info("HTTP server listening on %s", addr)
	s.logger.Info("If HTTP endpoints don't respond, please use stdio transport instead")

	return httpServer.Start(addr)
}

func (s *MCPServer) registerTools() {
	var tools []server.ServerTool

	// For now let's just implement the weaviate-query tool
	// and leave weaviate-insert-one commented out
	// until we finalize the design for inserts.
	// if !s.config.IsToolDisabled("weaviate-insert-one") && !s.config.ReadOnly {
	// 	insertOne := mcp.NewTool(
	// 		"weaviate-insert-one",
	// 		mcp.WithString(
	// 			"collection",
	// 			mcp.Description("Name of the target collection"),
	// 		),
	// 		mcp.WithObject(
	// 			"properties",
	// 			mcp.Description("Object properties to insert"),
	// 			mcp.Required(),
	// 		),
	// 	)
	// 	tools = append(tools, server.ServerTool{Tool: insertOne, Handler: s.weaviateInsertOne})
	// 	s.logger.Info("Registered tool: weaviate-insert-one")
	// } else if s.config.ReadOnly {
	// 	s.logger.Info("Skipped tool weaviate-insert-one: read-only mode enabled")
	// } else {
	// 	s.logger.Info("Skipped tool weaviate-insert-one: disabled")
	// }

	// weaviate-query tool
	if !s.config.IsToolDisabled("weaviate-query") {
		query := mcp.NewTool(
			"weaviate-query",
			mcp.WithDescription("Query objects from a Weaviate collection using hybrid search"),
			mcp.WithString(
				"query",
				mcp.Description("Query data within Weaviate"),
				mcp.Required(),
			),
			mcp.WithString(
				"collection",
				mcp.Description("Name of the target collection"),
				mcp.Required(),
			),
			mcp.WithArray(
				"targetProperties",
				mcp.Description("Properties to return with the query. Check available properties via weaviate://schema/{collection} resources"),
				mcp.Required(),
			),
			mcp.WithNumber(
				"limit",
				mcp.DefaultNumber(3),
				mcp.Description("Maximum number of results to return (default: 3)"),
			),
		)

		// Optional: log the schema to catch issues early
		if b, err := json.MarshalIndent(query.InputSchema, "", "  "); err == nil {
			s.logger.Debug("weaviate-query schema before modification:\n" + string(b))
		}

		// Fix the array schema by adding missing "items" property
		if query.InputSchema.Properties != nil {
			if targetProps, ok := query.InputSchema.Properties["targetProperties"].(map[string]interface{}); ok {
				targetProps["items"] = map[string]interface{}{"type": "string"}
				targetProps["minItems"] = 1
			}
		}

		// Log schema after modification
		if b, err := json.MarshalIndent(query.InputSchema, "", "  "); err == nil {
			s.logger.Debug("weaviate-query schema after modification:\n" + string(b))
		}

		tools = append(tools, server.ServerTool{Tool: query, Handler: s.weaviateQuery})
		s.logger.Info("Registered tool: weaviate-query")
	} else {
		s.logger.Info("Skipped tool weaviate-query: disabled")
	}

	// weaviate-generate-text tool
	if !s.config.IsToolDisabled("weaviate-generate-text") {
		generateText := mcp.NewTool(
			"weaviate-generate-text",
			mcp.WithDescription("Generate text using Weaviate's generative search capabilities"),
			mcp.WithString(
				"prompt",
				mcp.Description("Text prompt for generation"),
				mcp.Required(),
			),
			mcp.WithString(
				"collection",
				mcp.Description("Name of the target collection"),
				mcp.Required(),
			),
			mcp.WithNumber(
				"maxTokens",
				mcp.DefaultNumber(100),
				mcp.Description("Maximum number of tokens to generate (default: 100)"),
			),
		)

		tools = append(tools, server.ServerTool{Tool: generateText, Handler: s.weaviateGenerateText})
		s.logger.Info("Registered tool: weaviate-generate-text")
	} else {
		s.logger.Info("Skipped tool weaviate-generate-text: disabled")
	}

	s.server.AddTools(tools...)
}

func (s *MCPServer) registerPrompts() {
	var prompts []server.ServerPrompt

	// LiHua Dinner Events prompt
	dinnerPrompt := mcp.NewPrompt(
		"lihua-dinner-events",
		mcp.WithPromptDescription("Search for LiHua dinner and restaurant events"),
	)
	prompts = append(prompts, server.ServerPrompt{
		Prompt:  dinnerPrompt,
		Handler: s.handleLihuaDinnerPrompt,
	})

	// LiHua Travel Stories prompt
	travelPrompt := mcp.NewPrompt(
		"lihua-travel-stories",
		mcp.WithPromptDescription("Find LiHua's travel experiences and journeys"),
	)
	prompts = append(prompts, server.ServerPrompt{
		Prompt:  travelPrompt,
		Handler: s.handleLihuaTravelPrompt,
	})

	// LiHua Personal Events prompt
	personalPrompt := mcp.NewPrompt(
		"lihua-personal-events",
		mcp.WithPromptDescription("Search for LiHua's personal events and activities"),
	)
	prompts = append(prompts, server.ServerPrompt{
		Prompt:  personalPrompt,
		Handler: s.handleLihuaPersonalPrompt,
	})

	// LiHua Relationships prompt
	relationshipsPrompt := mcp.NewPrompt(
		"lihua-relationships",
		mcp.WithPromptDescription("Find information about LiHua's relationships and social connections"),
	)
	prompts = append(prompts, server.ServerPrompt{
		Prompt:  relationshipsPrompt,
		Handler: s.handleLihuaRelationshipsPrompt,
	})

	// Insert Sample Data prompt
	insertSamplePrompt := mcp.NewPrompt(
		"insert-sample-data",
		mcp.WithPromptDescription("Insert sample test data into Weaviate"),
	)
	prompts = append(prompts, server.ServerPrompt{
		Prompt:  insertSamplePrompt,
		Handler: s.handleInsertSamplePrompt,
	})

	// Query Sample Data prompt
	querySamplePrompt := mcp.NewPrompt(
		"query-sample-data",
		mcp.WithPromptDescription("Search for test data in Weaviate"),
	)
	prompts = append(prompts, server.ServerPrompt{
		Prompt:  querySamplePrompt,
		Handler: s.handleQuerySamplePrompt,
	})

	// Test Both Operations prompt
	testBothPrompt := mcp.NewPrompt(
		"test-both-operations",
		mcp.WithPromptDescription("Test both insert and query operations in sequence"),
	)
	prompts = append(prompts, server.ServerPrompt{
		Prompt:  testBothPrompt,
		Handler: s.handleTestBothPrompt,
	})

	// Search by Topic prompt (template with argument)
	searchTopicPrompt := mcp.NewPrompt(
		"search-by-topic",
		mcp.WithPromptDescription("Search for any topic in your dataset"),
		mcp.WithArgument("topic", mcp.ArgumentDescription("The topic to search for"), mcp.RequiredArgument()),
	)
	prompts = append(prompts, server.ServerPrompt{
		Prompt:  searchTopicPrompt,
		Handler: s.handleSearchByTopicPrompt,
	})

	// Add New Content prompt (template with arguments)
	addContentPrompt := mcp.NewPrompt(
		"add-new-content",
		mcp.WithPromptDescription("Add new content to your dataset"),
		mcp.WithArgument("text", mcp.ArgumentDescription("The text content to add"), mcp.RequiredArgument()),
		mcp.WithArgument("file_path", mcp.ArgumentDescription("The source file path"), mcp.RequiredArgument()),
	)
	prompts = append(prompts, server.ServerPrompt{
		Prompt:  addContentPrompt,
		Handler: s.handleAddNewContentPrompt,
	})

	s.server.AddPrompts(prompts...)
	s.logger.Info("Registered %d prompts", len(prompts))
}

func (s *MCPServer) registerResources() {
	var resources []server.ServerResource

	// Get the full schema to list all collections
	schema, err := s.weaviateConn.client.Schema().Getter().Do(context.Background())
	if err != nil {
		s.logger.Error("Failed to get schema for resources: %v", err)
		return
	}

	for _, class := range schema.Classes {
		resource := mcp.NewResource(
			fmt.Sprintf("weaviate://schema/%s", class.Class),
			fmt.Sprintf("Schema for collection %s", class.Class),
		)
		resources = append(resources, server.ServerResource{
			Resource: resource,
			Handler:  s.handleSchemaResource,
		})
	}

	s.server.AddResources(resources...)
	s.logger.Info("Registered %d resources", len(resources))
}

func (s *MCPServer) handleSchemaResource(ctx context.Context, req mcp.ReadResourceRequest) ([]mcp.ResourceContents, error) {
	// Extract collection from URI, e.g., weaviate://schema/Dataset -> Dataset
	uri := req.Params.URI
	if !strings.HasPrefix(uri, "weaviate://schema/") {
		return nil, fmt.Errorf("invalid resource URI: %s", uri)
	}
	collection := strings.TrimPrefix(uri, "weaviate://schema/")

	classSchema, err := s.weaviateConn.GetClassSchema(ctx, collection)
	if err != nil {
		return nil, fmt.Errorf("failed to get schema for collection %s: %w", collection, err)
	}

	var properties []string
	for _, prop := range classSchema.Properties {
		properties = append(properties, prop.Name)
	}

	content := fmt.Sprintf("Properties for collection '%s': %s", collection, strings.Join(properties, ", "))
	return []mcp.ResourceContents{
		mcp.TextResourceContents{
			URI:      uri,
			MIMEType: "text/plain",
			Text:     content,
		},
	}, nil
}

func (s *MCPServer) weaviateInsertOne(ctx context.Context, req mcp.CallToolRequest) (*mcp.CallToolResult, error) {
	args := req.GetArguments()
	s.logger.Debug("InsertOne called: collection=%v, args=%v", args["collection"], args)
	targetCol := s.parseTargetCollection(req)
	propsRaw, ok := args["properties"]
	if !ok {
		s.logger.Error("Missing 'properties' argument")
		return mcp.NewToolResultError("Missing 'properties' argument"), nil
	}
	props, ok := propsRaw.(map[string]interface{})
	if !ok {
		s.logger.Error("'properties' argument is not a map: %T", propsRaw)
		return mcp.NewToolResultError("'properties' argument must be an object"), nil
	}
	res, err := s.weaviateConn.InsertOne(context.Background(), targetCol, props)
	if err != nil {
		s.logger.Error("InsertOne error: %v", err)
		return mcp.NewToolResultErrorFromErr("failed to insert object", err), nil
	}
	s.logger.Info("InsertOne success: id=%v", res.ID.String())
	return mcp.NewToolResultText(res.ID.String()), nil
}

func (s *MCPServer) weaviateQuery(ctx context.Context, req mcp.CallToolRequest) (*mcp.CallToolResult, error) {
	args := req.GetArguments()
	s.logger.Debug("Query called: collection=%v, args=%v", args["collection"], args)
	targetCol := s.parseTargetCollection(req)
	queryRaw, ok := args["query"]
	if !ok {
		s.logger.Error("Missing 'query' argument")
		return mcp.NewToolResultError("Missing 'query' argument"), nil
	}
	query, ok := queryRaw.(string)
	if !ok {
		s.logger.Error("'query' argument is not a string: %T", queryRaw)
		return mcp.NewToolResultError("'query' argument must be a string"), nil
	}
	propsRaw, ok := args["targetProperties"]
	if !ok {
		s.logger.Error("Missing 'targetProperties' argument")
		return mcp.NewToolResultError("Missing 'targetProperties' argument"), nil
	}
	props, ok := propsRaw.([]interface{})
	if !ok {
		s.logger.Error("'targetProperties' argument is not an array: %T", propsRaw)
		return mcp.NewToolResultError("'targetProperties' argument must be an array"), nil
	}
	var targetProps []string
	for _, prop := range props {
		typed, ok := prop.(string)
		if !ok {
			s.logger.Error("targetProperties contains non-string: %v (%T)", prop, prop)
			return mcp.NewToolResultError("targetProperties must contain only strings"), nil
		}
		targetProps = append(targetProps, typed)
	}
	if len(targetProps) == 0 {
		s.logger.Error("targetProperties array is empty")
		return mcp.NewToolResultError("targetProperties must contain at least one property name"), nil
	}
	// Handle limit parameter (default to 3)
	limit := 3
	if limitRaw, ok := args["limit"]; ok {
		if limitFloat, ok := limitRaw.(float64); ok {
			limit = int(limitFloat)
		} else {
			s.logger.Error("'limit' argument is not a number: %T", limitRaw)
			return mcp.NewToolResultError("'limit' argument must be a number"), nil
		}
	}
	// Validate targetProps against schema
	classSchema, err := s.weaviateConn.GetClassSchema(context.Background(), targetCol)
	if err != nil {
		s.logger.Error("Failed to get schema for collection %s: %v", targetCol, err)
		return mcp.NewToolResultErrorFromErr("failed to get collection schema", err), nil
	}
	propertyMap := make(map[string]bool)
	for _, prop := range classSchema.Properties {
		propertyMap[prop.Name] = true
	}
	for _, prop := range targetProps {
		if !propertyMap[prop] {
			s.logger.Error("Invalid property '%s' for collection '%s'", prop, targetCol)
			return mcp.NewToolResultError(fmt.Sprintf("property '%s' does not exist in collection '%s'", prop, targetCol)), nil
		}
	}
	res, err := s.weaviateConn.Query(context.Background(), targetCol, query, targetProps, limit)
	if err != nil {
		s.logger.Error("Query error: %v", err)
		return mcp.NewToolResultErrorFromErr("failed to process query", err), nil
	}
	s.logger.Info("Query success: result length=%d", len(res))
	return mcp.NewToolResultText(res), nil
}

func (s *MCPServer) weaviateGenerateText(ctx context.Context, req mcp.CallToolRequest) (*mcp.CallToolResult, error) {
	args := req.GetArguments()
	s.logger.Debug("GenerateText called: collection=%v, args=%v", args["collection"], args)

	targetCol := s.parseTargetCollection(req)

	promptRaw, ok := args["prompt"]
	if !ok {
		s.logger.Error("Missing 'prompt' argument")
		return mcp.NewToolResultError("Missing 'prompt' argument"), nil
	}
	prompt, ok := promptRaw.(string)
	if !ok {
		s.logger.Error("'prompt' argument is not a string: %T", promptRaw)
		return mcp.NewToolResultError("'prompt' argument must be a string"), nil
	}

	// Handle maxTokens parameter (default to 100)
	maxTokens := 100
	if maxTokensRaw, ok := args["maxTokens"]; ok {
		if maxTokensFloat, ok := maxTokensRaw.(float64); ok {
			maxTokens = int(maxTokensFloat)
		} else {
			s.logger.Error("'maxTokens' argument is not a number: %T", maxTokensRaw)
			return mcp.NewToolResultError("'maxTokens' argument must be a number"), nil
		}
	}

	res, err := s.weaviateConn.GenerateText(context.Background(), targetCol, prompt, maxTokens)
	if err != nil {
		s.logger.Error("GenerateText error: %v", err)
		return mcp.NewToolResultErrorFromErr("failed to generate text", err), nil
	}

	s.logger.Info("GenerateText success: result length=%d", len(res))
	return mcp.NewToolResultText(res), nil
}

func (s *MCPServer) parseTargetCollection(req mcp.CallToolRequest) string {
	var (
		targetCol = s.defaultCollection
	)
	args := req.GetArguments()
	col, ok := args["collection"].(string)
	if ok {
		targetCol = col
	}
	return targetCol
}

// Prompt handlers

func (s *MCPServer) handleLihuaDinnerPrompt(ctx context.Context, req mcp.GetPromptRequest) (*mcp.GetPromptResult, error) {
	return mcp.NewGetPromptResult(
		"Search for LiHua dinner and restaurant events",
		[]mcp.PromptMessage{
			mcp.NewPromptMessage(
				mcp.RoleUser,
				mcp.TextContent{
					Type: "text",
					Text: `Search the LiHua-World dataset for information about when LiHua went to dinner or restaurant events.

Use the weaviate-query tool with these parameters:
{
  "query": "LiHua dinner restaurant meal food",
  "targetProperties": ["text", "file_path"],
  "collection": "Dataset"
}

Look for mentions of dinner, restaurants, meals, or food-related activities involving LiHua.`,
				},
			),
		},
	), nil
}

func (s *MCPServer) handleLihuaTravelPrompt(ctx context.Context, req mcp.GetPromptRequest) (*mcp.GetPromptResult, error) {
	return mcp.NewGetPromptResult(
		"Find LiHua's travel experiences and journeys",
		[]mcp.PromptMessage{
			mcp.NewPromptMessage(
				mcp.RoleUser,
				mcp.TextContent{
					Type: "text",
					Text: `Find stories or information about LiHua's travels and journeys in the dataset.

Use the weaviate-query tool with these parameters:
{
  "query": "LiHua travel journey trip vacation",
  "targetProperties": ["text", "file_path"],
  "collection": "Dataset"
}

Include any mentions of trips, journeys, vacations, or travel experiences.`,
				},
			),
		},
	), nil
}

func (s *MCPServer) handleLihuaPersonalPrompt(ctx context.Context, req mcp.GetPromptRequest) (*mcp.GetPromptResult, error) {
	return mcp.NewGetPromptResult(
		"Search for LiHua's personal events and activities",
		[]mcp.PromptMessage{
			mcp.NewPromptMessage(
				mcp.RoleUser,
				mcp.TextContent{
					Type: "text",
					Text: `Search for personal events, activities, or experiences involving LiHua.

Use the weaviate-query tool with these parameters:
{
  "query": "LiHua event activity personal life",
  "targetProperties": ["text", "file_path"],
  "collection": "Dataset"
}

Look for birthdays, celebrations, meetings, or other personal events.`,
				},
			),
		},
	), nil
}

func (s *MCPServer) handleLihuaRelationshipsPrompt(ctx context.Context, req mcp.GetPromptRequest) (*mcp.GetPromptResult, error) {
	return mcp.NewGetPromptResult(
		"Find information about LiHua's relationships and social connections",
		[]mcp.PromptMessage{
			mcp.NewPromptMessage(
				mcp.RoleUser,
				mcp.TextContent{
					Type: "text",
					Text: `Search for information about LiHua's relationships, friends, family, or social interactions.

Use the weaviate-query tool with these parameters:
{
  "query": "LiHua friend family relationship social",
  "targetProperties": ["text", "file_path"],
  "collection": "Dataset"
}

Include mentions of friends, family members, social connections, or relationships.`,
				},
			),
		},
	), nil
}

func (s *MCPServer) handleInsertSamplePrompt(ctx context.Context, req mcp.GetPromptRequest) (*mcp.GetPromptResult, error) {
	return mcp.NewGetPromptResult(
		"Insert sample test data into Weaviate",
		[]mcp.PromptMessage{
			mcp.NewPromptMessage(
				mcp.RoleUser,
				mcp.TextContent{
					Type: "text",
					Text: `Please insert this sample data into the Weaviate collection "TestCollection":

Use the weaviate-insert-one tool with these properties:
{
  "collection": "TestCollection",
  "properties": {
    "title": "Sample Document",
    "content": "This is a test document for Weaviate MCP server",
    "category": "test",
    "timestamp": "2025-01-15T10:00:00Z"
  }
}`,
				},
			),
		},
	), nil
}

func (s *MCPServer) handleQuerySamplePrompt(ctx context.Context, req mcp.GetPromptRequest) (*mcp.GetPromptResult, error) {
	return mcp.NewGetPromptResult(
		"Search for test data in Weaviate",
		[]mcp.PromptMessage{
			mcp.NewPromptMessage(
				mcp.RoleUser,
				mcp.TextContent{
					Type: "text",
					Text: `Please search for "test" in the Weaviate collection "TestCollection":

Use the weaviate-query tool with these parameters:
{
  "query": "test",
  "targetProperties": ["title", "content", "category", "timestamp"],
  "collection": "TestCollection"
}`,
				},
			),
		},
	), nil
}

func (s *MCPServer) handleTestBothPrompt(ctx context.Context, req mcp.GetPromptRequest) (*mcp.GetPromptResult, error) {
	return mcp.NewGetPromptResult(
		"Test both insert and query operations in sequence",
		[]mcp.PromptMessage{
			mcp.NewPromptMessage(
				mcp.RoleUser,
				mcp.TextContent{
					Type: "text",
					Text: `Let's test the Weaviate MCP server tools!

First, insert some test data into collection "TestCollection":
Use weaviate-insert-one with:
{
  "collection": "TestCollection",
  "properties": {
    "title": "Weaviate MCP Test",
    "content": "Testing the Model Context Protocol server for Weaviate",
    "category": "testing",
    "tags": ["mcp", "weaviate", "test"]
  }
}

Then query for it:
Use weaviate-query with:
{
  "query": "MCP test",
  "targetProperties": ["title", "content", "category", "tags"],
  "collection": "TestCollection"
}`,
				},
			),
		},
	), nil
}

func (s *MCPServer) handleSearchByTopicPrompt(ctx context.Context, req mcp.GetPromptRequest) (*mcp.GetPromptResult, error) {
	topic, ok := req.Params.Arguments["topic"]
	if !ok {
		return nil, fmt.Errorf("missing 'topic' argument")
	}

	return mcp.NewGetPromptResult(
		fmt.Sprintf("Search for topic: %s", topic),
		[]mcp.PromptMessage{
			mcp.NewPromptMessage(
				mcp.RoleUser,
				mcp.TextContent{
					Type: "text",
					Text: fmt.Sprintf(`I want to search for "%s" in my LiHua-World dataset.

Use the weaviate-query tool with:
{
  "query": "%s",
  "targetProperties": ["text", "file_path"],
  "collection": "Dataset"
}`, topic, topic),
				},
			),
		},
	), nil
}

func (s *MCPServer) handleAddNewContentPrompt(ctx context.Context, req mcp.GetPromptRequest) (*mcp.GetPromptResult, error) {
	text, ok := req.Params.Arguments["text"]
	if !ok {
		return nil, fmt.Errorf("missing 'text' argument")
	}
	filePath, ok := req.Params.Arguments["file_path"]
	if !ok {
		return nil, fmt.Errorf("missing 'file_path' argument")
	}

	return mcp.NewGetPromptResult(
		"Add new content to dataset",
		[]mcp.PromptMessage{
			mcp.NewPromptMessage(
				mcp.RoleUser,
				mcp.TextContent{
					Type: "text",
					Text: fmt.Sprintf(`I want to add new content to my LiHua-World dataset.

Use the weaviate-insert-one tool with:
{
  "collection": "Dataset",
  "properties": {
    "text": "%s",
    "file_path": "%s"
  }
}`, text, filePath),
				},
			),
		},
	), nil
}
