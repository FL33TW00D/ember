# Ember - Ollama-like interface for embedding models

Ember offers GPU and ANE accelerated embedding models with a convenient server!

Ember works by converting [sentence-transformers models](https://www.sbert.net) to Core ML, then launching a local server you can query to retrieve document embeddings. You can select from a few recommended models, or [choose from any of the ones available in Hugging Face](https://huggingface.co/models?library=sentence-transformers&sort=trending).

Sentence transformers models generate representations (called embeddings) from documents, which you can use for tasks such as semantic search, similarity, retrieval or clustering. For more information, please refer to [the documentation](https://www.sbert.net).

## Getting Started

Follow the drop down menu instructions with `ember create` to get started:
```bash
ember create
```
Generate displays a dropdown menu, allowing you to select one of a few popular sentence-transformers models. There's also a `Custom Model` option where you can choose your desired [sentence transformers model from Hugging Face](https://huggingface.co/models?library=sentence-transformers&sort=trending). The selected model will be automatically downloaded and converted to Core ML for you.

You could select `intfloat/multilingual-e5-small` as the model to create for
example.

You should now have a CoreML model. You can serve this on a local server using:

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
      { "content": "Hello, world!" },
      { "content": "Open source for the win 🤗!" }
    ],
    "options": {
      "keep_alive": 1
    }
  }'
```

This example will return embeddings for the documents you supplied, which in this case are the sentences `Hello, world!` and `Open source for the win 🤗!`. The model process will stay running
until the `keep_alive` duration is reached (1 minute in this case), so subsequent requests will be processed fast.

## Examples

- [similarity.py](examples/similarity.py) Simple similarity computation between a query and a set of documents.