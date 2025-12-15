from http.server import BaseHTTPRequestHandler, HTTPServer, ThreadingHTTPServer


class SimpleHTTPRequestHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-type", "text/plain")
        self.end_headers()
        self.wfile.write(b"Hello from function 2!\n")
        self.wfile.write(bytes("path=" + self.path, "utf-8"))


def run(server_class=ThreadingHTTPServer, handler_class=SimpleHTTPRequestHandler, port=80):
    server_address = ("", port)
    httpd = server_class(server_address, handler_class)
    print(f"Starting threaded httpd server on port {port}...")
    httpd.serve_forever()


if __name__ == "__main__":
    run()
