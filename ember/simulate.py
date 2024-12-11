import requests
import time

TEST_MESSAGES = [
    "Open source for the win! 🤗 Contributing to community projects is the way forward",
    "Neural networks are awesome and revolutionizing every industry sector imaginable",
    "FastAPI makes servers easy and scalable with great documentation",
    "Programming is fun especially when solving complex problems",
    "AI is the future and will transform how we live and work",
    "Validation complete with all test cases passing successfully",
    "Vectors are neat and essential for modern machine learning",
    "Transformer models have completely changed the NLP landscape forever",
    "Computer vision is enabling unprecedented applications",
    "Foundation models are reshaping AI development worldwide"
]

TOO_MANY_MESSAGES = ["Too many messages"] * 1000

def send_request(messages):
    url = "http://localhost:11434/api/embed"
    payload = {
        "model": "intfloat/multilingual-e5-small",
        "documents": [{"content": message} for message in messages]
    }

    start_time = time.time()
    response = requests.post(url, json=payload)
    duration = time.time() - start_time
    print(f"Request completed in {duration:.2f}s")
    print(f"Response: {response.text}")
    return response.json()

def main():
    _response = send_request(TOO_MANY_MESSAGES)

if __name__ == "__main__":
    main()
