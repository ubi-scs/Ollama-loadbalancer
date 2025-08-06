import ollama

if __name__ == "__main__":

    PROXY_HOST_URL = "http://localhost:8000"
    USER = "testuser"
    KEY = "secretkey"

    client = ollama.Client(host=PROXY_HOST_URL, headers={"Authorization": f"Bearer {USER}:{KEY}"})
    response = client.chat(model='phi4:latest', messages=[{'role': 'user', 'content': 'Schreib ein Gedicht über dich selbst!'}])
    print("Response:")
    print(f"Content: {response['message']['content']}")
