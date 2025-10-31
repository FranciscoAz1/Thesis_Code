import requests

response = requests.post(
    "http://localhost:11434/api/embeddings",
    json={"model": "mxbai-embed-large", "prompt": "test"}
)

print(response.json())  # ✅ If this fails, Ollama isn't responding correctly

