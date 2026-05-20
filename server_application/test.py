import ollama

if __name__ == "__main__":

    PROXY_HOST_URL = "http://127.0.0.1:11434"
    USER = "testii"
    KEY = "asdf1234"

    client = ollama.Client(host=PROXY_HOST_URL, headers={"Authorization": f"{USER}:{KEY}"})
    response = client.chat(model='qwen2.5:0.5b', messages=[{'role': 'user', 'content': 'How are you?'}])
    print("Response:")
    print(f"Content: {response['message']['content']}")
