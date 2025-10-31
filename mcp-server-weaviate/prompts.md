# Weaviate MCP Server - Example Prompts

This file contains example prompts you can use to test the Weaviate MCP server tools. Since the MCP Go library doesn't support dynamic prompts yet, you can copy these prompts and use them with MCP clients that support prompt templates.

## Dataset-Specific Prompts (LiHua-World)

Based on your LiHua-World dataset, here are prompts tailored to searching your content:

### Prompt: Search LiHua Dinner Events

**Description:** Search for when LiHua went to dinner or restaurant events

**Prompt Text:**
```
Search the LiHua-World dataset for information about when LiHua went to dinner or restaurant events.

Use the weaviate-query tool with these parameters:
{
  "query": "LiHua dinner restaurant meal food",
  "targetProperties": ["text", "file_path"],
  "collection": "Dataset"
}

Look for mentions of dinner, restaurants, meals, or food-related activities involving LiHua.
```

### Prompt: Find LiHua Travel Stories

**Description:** Search for LiHua's travel experiences and stories

**Prompt Text:**
```
Find stories or information about LiHua's travels and journeys in the dataset.

Use the weaviate-query tool with these parameters:
{
  "query": "LiHua travel journey trip vacation",
  "targetProperties": ["text", "file_path"],
  "collection": "Dataset"
}

Include any mentions of trips, journeys, vacations, or travel experiences.
```

### Prompt: LiHua Personal Events

**Description:** Search for personal events and activities involving LiHua

**Prompt Text:**
```
Search for personal events, activities, or experiences involving LiHua.

Use the weaviate-query tool with these parameters:
{
  "query": "LiHua event activity personal life",
  "targetProperties": ["text", "file_path"],
  "collection": "Dataset"
}

Look for birthdays, celebrations, meetings, or other personal events.
```

### Prompt: LiHua Relationships

**Description:** Find information about LiHua's relationships and interactions

**Prompt Text:**
```
Search for information about LiHua's relationships, friends, family, or social interactions.

Use the weaviate-query tool with these parameters:
{
  "query": "LiHua friend family relationship social",
  "targetProperties": ["text", "file_path"],
  "collection": "Dataset"
}

Include mentions of friends, family members, social connections, or relationships.
```

## General Testing Prompts

### Prompt: Insert Sample Data

**Description:** Insert sample test data into Weaviate

**Prompt Text:**
```
Please insert this sample data into the Weaviate collection "TestCollection":

Use the weaviate-insert-one tool with these properties:
{
  "collection": "TestCollection",
  "properties": {
    "title": "Sample Document",
    "content": "This is a test document for Weaviate MCP server",
    "category": "test",
    "timestamp": "2025-01-15T10:00:00Z"
  }
}
```

### Prompt: Query Sample Data

**Description:** Search for test data in Weaviate

**Prompt Text:**
```
Please search for "test" in the Weaviate collection "TestCollection":

Use the weaviate-query tool with these parameters:
{
  "query": "test",
  "targetProperties": ["title", "content", "category", "timestamp"],
  "collection": "TestCollection"
}
```

### Prompt: Test Both Operations

**Description:** Test both insert and query operations in sequence

**Prompt Text:**
```
Let's test the Weaviate MCP server tools!

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
}
```

## Custom Search Prompts

### Prompt: Search by Topic

**Description:** Search for any topic in your dataset

**Prompt Text:**
```
I want to search for [TOPIC] in my LiHua-World dataset.

Use the weaviate-query tool with:
{
  "query": "[TOPIC]",
  "targetProperties": ["text", "file_path"],
  "collection": "Dataset"
}

Replace [TOPIC] with what you want to search for.
```

### Prompt: Add New Content

**Description:** Add new content to your dataset

**Prompt Text:**
```
I want to add new content to my LiHua-World dataset.

Use the weaviate-insert-one tool with:
{
  "collection": "Dataset",
  "properties": {
    "text": "[YOUR_CONTENT_HERE]",
    "file_path": "[SOURCE_FILE_PATH]"
  }
}

Replace the placeholders with your actual content and source file path.
```

## Using Prompts with MCP Clients

### Claude Desktop
Add these prompts to your `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "weaviate": {
      "command": "/path/to/mcp-server",
      "env": {
        "WEAVIATE_HOST": "localhost:8080"
      }
    }
  },
  "prompts": {
    "lihua-dinner": {
      "description": "Search for LiHua dinner events",
      "prompt": "Search the LiHua-World dataset for information about when LiHua went to dinner or restaurant events.\n\nUse the weaviate-query tool with these parameters:\n{\n  \"query\": \"LiHua dinner restaurant meal food\",\n  \"targetProperties\": [\"text\", \"file_path\"],\n  \"collection\": \"Dataset\"\n}\n\nLook for mentions of dinner, restaurants, meals, or food-related activities involving LiHua."
    },
    "lihua-travel": {
      "description": "Find LiHua travel stories",
      "prompt": "Find stories or information about LiHua's travels and journeys in the dataset.\n\nUse the weaviate-query tool with these parameters:\n{\n  \"query\": \"LiHua travel journey trip vacation\",\n  \"targetProperties\": [\"text\", \"file_path\"],\n  \"collection\": \"Dataset\"\n}\n\nInclude any mentions of trips, journeys, vacations, or travel experiences."
    }
  }
}
```

### Other MCP Clients
Check your client's documentation for how to add custom prompts. Most support prompt templates that you can copy from this file.