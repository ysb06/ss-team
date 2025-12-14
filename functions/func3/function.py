from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import urllib.request
import urllib.error
import os
import sys

# LPM Server URL - using host.docker.internal to access host machine from container
LPM_SERVER_URL = os.environ.get("LPM_SERVER_URL", "http://host.docker.internal:9000/generate")

class LPMRequestHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-type", "text/plain")
        self.end_headers()
        msg = f"Send a POST request with text body to interact with LPM Server.\nLPM Server URL: {LPM_SERVER_URL}\n"
        self.wfile.write(msg.encode('utf-8'))

    def do_POST(self):
        content_length = int(self.headers.get('Content-Length', 0))
        post_data = self.rfile.read(content_length).decode('utf-8')

        if not post_data:
            self.send_response(400)
            self.end_headers()
            self.wfile.write(b"No input text provided")
            return

        print(f"Received request, forwarding to LPM Server: {LPM_SERVER_URL}", file=sys.stderr)
        
        try:
            # Send to LPM Server (which will handle scheduling and forward to Ollama)
            req = urllib.request.Request(
                LPM_SERVER_URL,
                data=post_data.encode('utf-8'),
                headers={'Content-Type': 'text/plain'}
            )
            
            print(f"Sending to LPM server...", file=sys.stderr)
            with urllib.request.urlopen(req, timeout=120) as response:
                print(f"Got response from LPM server: {response.status}", file=sys.stderr)
                if response.status == 200:
                    resp_body = response.read().decode('utf-8')
                    
                    self.send_response(200)
                    self.send_header("Content-type", "text/plain; charset=utf-8")
                    self.end_headers()
                    self.wfile.write(resp_body.encode('utf-8'))
                else:
                    self.send_response(response.status)
                    self.end_headers()
                    self.wfile.write(b"Error from LPM Server")
                    
        except urllib.error.URLError as e:
            error_msg = f"Connection Error: Cannot reach LPM Server at {LPM_SERVER_URL}\nError: {str(e)}\nMake sure LPM server is running and Docker can access host.docker.internal"
            print(error_msg, file=sys.stderr)
            self.send_response(500)
            self.end_headers()
            self.wfile.write(error_msg.encode('utf-8'))
        except Exception as e:
            error_msg = f"Internal Error: {str(e)}"
            print(error_msg, file=sys.stderr)
            self.send_response(500)
            self.end_headers()
            self.wfile.write(error_msg.encode('utf-8'))


def run(server_class=HTTPServer, handler_class=LPMRequestHandler, port=80):
    server_address = ("", port)
    httpd = server_class(server_address, handler_class)
    print(f"Starting httpd server on port {port}...")
    print(f"LPM Server URL: {LPM_SERVER_URL}")
    httpd.serve_forever()


if __name__ == "__main__":
    run()
