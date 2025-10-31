package main

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"time"

	"github.com/weaviate/weaviate-go-client/v4/weaviate"
	"github.com/weaviate/weaviate-go-client/v4/weaviate/graphql"
	"github.com/weaviate/weaviate/entities/models"
)

type WeaviateConnection struct {
	client *weaviate.Client
}

func NewWeaviateConnection(config *Config, logger *Logger) (*WeaviateConnection, error) {
	logger.Info("Connecting to Weaviate at %s://%s", config.WeaviateScheme, config.WeaviateHost)
	client, err := weaviate.NewClient(weaviate.Config{
		Host:           config.WeaviateHost,
		Scheme:         config.WeaviateScheme,
		StartupTimeout: time.Second,
	})
	if err != nil {
		logger.Error("Failed to connect to Weaviate: %v", err)
		return nil, fmt.Errorf("connect to weaviate: %w", err)
	}
	logger.Info("Successfully connected to Weaviate")
	return &WeaviateConnection{client}, nil
}

func (conn *WeaviateConnection) InsertOne(ctx context.Context,
	collection string, props interface{},
) (*models.Object, error) {
	obj := models.Object{
		Class:      collection,
		Properties: props,
	}
	// Use batch to leverage autoschema and gRPC
	resp, err := conn.batchInsert(ctx, &obj)
	if err != nil {
		return nil, fmt.Errorf("insert one object: %w", err)
	}

	return &resp[0].Object, err
}

func (conn *WeaviateConnection) Query(ctx context.Context, collection,
	query string, targetProps []string, limit int,
) (string, error) {
	hybrid := graphql.HybridArgumentBuilder{}
	hybrid.WithQuery(query)
	builder := conn.client.GraphQL().Get().
		WithClassName(collection).WithHybrid(&hybrid).
		WithFields(func() []graphql.Field {
			fields := make([]graphql.Field, len(targetProps))
			for i, prop := range targetProps {
				fields[i] = graphql.Field{Name: prop}
			}
			return fields
		}()...)
	if limit > 0 {
		builder = builder.WithLimit(limit)
	}
	res, err := builder.Do(context.Background())
	if err != nil {
		return "", err
	}
	b, err := json.Marshal(res)
	if err != nil {
		return "", fmt.Errorf("unmarshal query response: %w", err)
	}
	return string(b), nil
}

// Create here a function, the same as the query, but instead of using just query hybrid, use Weaviate's generate text
func (conn *WeaviateConnection) GenerateText(ctx context.Context, collection, query string, maxTokens int) (string, error) {
	// Use Weaviate's generate text feature with hybrid search + generation
	// Note: This requires Ollama to be running and accessible from Weaviate

	// Setup hybrid search
	hybrid := graphql.HybridArgumentBuilder{}
	hybrid.WithQuery(query)
	hybrid.WithTargetVectors("text")

	// Set a reasonable limit for hybrid search results
	limit := 3 // Default limit
	if maxTokens > 0 {
		// Use maxTokens as a proxy for result limit (capped at 10)
		calculatedLimit := maxTokens / 10
		if calculatedLimit < 1 {
			limit = 1
		} else if calculatedLimit > 10 {
			limit = 10
		} else {
			limit = calculatedLimit
		}
	}
	prompt := "Answer briefly: " + query
	// Try with generative search first
	generativeSearch := graphql.NewGenerativeSearch()
	generativeSearch.GroupedResult(prompt, "text")

	// Build the query with generative search
	builder := conn.client.GraphQL().Get().
		WithClassName(collection).
		WithHybrid(&hybrid).
		WithLimit(limit).
		WithGenerativeSearch(generativeSearch)

	resp, err := builder.Do(ctx)

	if err != nil {
		return "", fmt.Errorf("weaviate generate text request failed: %w", err)
	}
	b, err := json.Marshal(resp)
	if err != nil {
		return "", fmt.Errorf("unmarshal generate text response: %w", err)
	}
	return string(b), nil
}

func (conn *WeaviateConnection) GetClassSchema(ctx context.Context, className string) (*models.Class, error) {
	class, err := conn.client.Schema().ClassGetter().WithClassName(className).Do(ctx)
	if err != nil {
		return nil, fmt.Errorf("get class schema: %w", err)
	}
	return class, nil
}

func (conn *WeaviateConnection) batchInsert(ctx context.Context, objs ...*models.Object) ([]models.ObjectsGetResponse, error) {
	resp, err := conn.client.Batch().ObjectsBatcher().WithObjects(objs...).Do(ctx)
	if err != nil {
		return nil, fmt.Errorf("make insertion request: %w", err)
	}
	for _, res := range resp {
		if res.Result != nil && res.Result.Errors != nil && res.Result.Errors.Error != nil {
			for _, nestedErr := range res.Result.Errors.Error {
				err = errors.Join(err, errors.New(nestedErr.Message))
			}
		}
	}

	return resp, err
}
