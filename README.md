# Ember - Ollama-like interface for embedding models

Ember offers ANE accelerated embedding models with a convenient server!

## Getting Started

Firstly, clone the repository. Then run:
```bash
uv sync
```
Once that's done, you should be ready to get started!

Ember ships with 2 commands out of the box: `ember generate` and `ember serve`

```bash
ember generate
```
Generate creates a dropdown menu, allowing you to select 

You could select `intfloat/multilingual-e5-small` as the model to generate for
example.

You should now have a generated CoreML model. You can serve this on a local
server using:

```bash
ember serve
```

When you hit the server with a POST request like the following:

```bash
curl http://localhost:11434/api/embed \ 
  -H "Content-Type: application/json" \
  -d '{
    "model": "intfloat/multilingual-e5-small",
    "messages": [
      { "role": "user", "content": "Hello, world!" },
      { "role": "user", "content": "Open source for the win 🤗!" }
    ], 
    "options": { 
        "keep_alive": 1 
    }
  }'
```

You should get some embeddings returned! The model process will stay running
until the `keep_alive` duration is reached (1 minute in this case).
