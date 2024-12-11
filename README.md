# Ember - Ollama-like interface for embedding models

Ember offers GPU and ANE accelerated embedding models with a convenient server!

Ember works by converting [sentence-transformers models](https://www.sbert.net) to Core ML, then launching a local server you can query to retrieve document embeddings. You can select from a few recommended models, or [choose from any of the ones available in Hugging Face](https://huggingface.co/models?library=sentence-transformers&sort=trending).

Sentence transformers models generate representations (called embeddings) from documents, which you can use for tasks such as semantic search, similarity, retrieval or clustering. For more information, please refer to [the documentation](https://www.sbert.net).

## Getting Started

Firstly, clone the repository. Then run:
```bash
uv sync
```

If you don't use `uv` to configure your virtual environments, just run `pip install ./ember` after cloning.

Once that's done, you should be ready to get started!

Ember ships with 2 commands out of the box: `ember generate` and `ember serve`

```bash
ember generate
```
Generate displays a dropdown menu, allowing you to select one of a few popular sentence-transformers models. There's also a `Custom Model` option where you can choose your desired [sentence transformers model from Hugging Face](https://huggingface.co/models?library=sentence-transformers&sort=trending). The selected model will be automatically downloaded and converted to Core ML for you.

For example, you could select [`intfloat/multilingual-e5-small`](https://huggingface.co/intfloat/multilingual-e5-small), a small but multilingual model that generates embeddings with 384 dimensions.

Ember will convert the model to Core ML. After that's done, you can spawn a local server using:

```bash
ember serve
```

The server provides an endpoint for any of the models you converted with `ember generate`. To query the model we just converted in our previous example, you can use a POST request like the following:

```bash
curl http://localhost:11434/api/embed \
  -H "Content-Type: application/json" \
  -d '{
    "model": "intfloat/multilingual-e5-small",
    "documents": [
      { "role": "user", "content": "Hello, world!" },
      { "role": "user", "content": "Open source for the win 🤗!" }
    ], 
    "options": { 
        "keep_alive": 1 
    }
  }'
```

This example will return embeddings for the documents you supplied, which in this case are the sentences `Hello, world!` and `Open source for the win 🤗!`. The model process will stay running
until the `keep_alive` duration is reached (1 minute in this case), so subsequent requests will be processed fast.
