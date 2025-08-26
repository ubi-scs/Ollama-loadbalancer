import ollama

if __name__ == "__main__":

    PROXY_HOST_URL = "http://scs-ai-proxy:11434"
    USER = ""
    KEY = ""

    client = ollama.Client(host=PROXY_HOST_URL, headers={"Authorization": f"{USER}:{KEY}"})
    response = client.chat(model='qwen:0.5b', messages=[{'role': 'user', 'content': 'How are you?'}])
    print("Response:")
    print(f"Content: {response['message']['content']}")
