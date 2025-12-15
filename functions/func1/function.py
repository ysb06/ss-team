from http.server import BaseHTTPRequestHandler, HTTPServer, ThreadingHTTPServer
import json
import urllib.request
import os

# Ollama URL - using host.docker.internal to access host machine from container
# Note: This requires Docker to be configured to resolve host.docker.internal
# On Linux, you might need to run docker with --add-host=host.docker.internal:host-gateway
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://host.docker.internal:11434/api/generate")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llava:7b")

class SimpleHTTPRequestHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-type", "text/plain")
        self.end_headers()
        self.wfile.write(b"Send a POST request with text body to interact with Ollama.\n")

    def do_POST(self):
        content_length = int(self.headers.get('Content-Length', 0))
        post_data = self.rfile.read(content_length).decode('utf-8')

        if not post_data:
            self.send_response(400)
            self.end_headers()
            self.wfile.write(b"No input text provided")
            return

        payload = {
            "model": OLLAMA_MODEL,
            "prompt": post_data,
            "stream": False
        }

        try:
            req = urllib.request.Request(
                OLLAMA_URL,
                data=json.dumps(payload).encode('utf-8'),
                headers={'Content-Type': 'application/json'}
            )
            
            with urllib.request.urlopen(req) as response:
                if response.status == 200:
                    resp_body = response.read().decode('utf-8')
                    resp_json = json.loads(resp_body)
                    answer = resp_json.get("response", "")
                    
                    self.send_response(200)
                    self.send_header("Content-type", "text/plain; charset=utf-8")
                    self.end_headers()
                    self.wfile.write(answer.encode('utf-8'))
                else:
                    self.send_response(response.status)
                    self.end_headers()
                    self.wfile.write(b"Error from Ollama")
                    
        except Exception as e:
            self.send_response(500)
            self.end_headers()
            self.wfile.write(f"Internal Error: {str(e)}".encode('utf-8'))


def run(server_class=ThreadingHTTPServer, handler_class=SimpleHTTPRequestHandler, port=80):
    server_address = ("", port)
    httpd = server_class(server_address, handler_class)
    print(f"Starting threaded httpd server on port {port}...")
    httpd.serve_forever()


if __name__ == "__main__":
    run()
