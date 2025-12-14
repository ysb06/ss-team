import urllib.request
import sys

# Configuration
# slrun default listens on 0.0.0.0:8080
# Format: http://localhost:PORT/FUNCTION_NAME
SLRUN_URL = "http://localhost:8080"
FUNCTION_NAME = "func1"

def send_ollama_request(prompt):
    url = f"{SLRUN_URL}/{FUNCTION_NAME}"
    print(f"Sending prompt to {url}")
    print(f"Prompt: {prompt}\n")
    
    try:
        # The function expects raw text in the POST body
        data = prompt.encode('utf-8')
        
        req = urllib.request.Request(
            url, 
            data=data, 
            method='POST'
        )
        req.add_header('Content-Type', 'text/plain; charset=utf-8')

        with urllib.request.urlopen(req) as response:
            result = response.read().decode('utf-8')
            print("\nResponse from Ollama:")
            print("-" * 60)
            print(result)
            print("-" * 60)

    except urllib.error.URLError as e:
        print(f"Error connecting to server: {e}")
        print(f"Make sure slrun is running on {SLRUN_URL}")
        print("Example: go run main.go slrun.json")

if __name__ == "__main__":
    prompt = "Why is the sky blue?"
    if len(sys.argv) > 1:
        prompt = " ".join(sys.argv[1:])
    
    send_ollama_request(prompt)
