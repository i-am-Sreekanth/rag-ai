import os, requests

api_key = os.getenv("DEEPSEEK_API_KEY")
url = "https://openrouter.ai/api/v1/chat/completions"
headers = {
    "Authorization": f"Bearer {api_key}",
    "HTTP-Referer": "http://localhost:8501",
    "X-Title": "Connection Test",
    "Content-Type": "application/json",
}
payload = {
    "model": "deepseek/deepseek-chat",
    "messages": [{"role": "user", "content": "Say hello"}],
}

resp = requests.post(url, headers=headers, json=payload)
print(resp.status_code)
print(resp.text)
